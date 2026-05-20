# 100B+ MoE 设计 Cookbook：12 步决策树

> **目的**：把 [[28_open_source_moe_catalog]] + [[04_deepseek_v3]] + [[35_glm45]] + [[36_longcat]] + [[37_ling1t]] + [[06_kimi_k2]] + [[24_dots1]] 等 100B-1T 段位的设计经验**结晶为一个可执行的决策流程**。
> **使用方法**：从上至下按顺序回答每个 step；前面的答案会影响后面的可选项。
> **目标读者**：要做 100B-1T 段位 MoE 的架构师 / project lead，不是入门读者。

---

## TL;DR 决策流（12 步速览）

| Step | 决策 | 默认值（中位选择） |
|---|---|---|
| 1 | 目标 active params（决定 FLOPs） | 12-32B (100B-1T 段) |
| 2 | 活率 = active / total | **1/12 ~ 1/20** (2025 主流) |
| 3 | Expert 数 N_routed (离散跳档) | **128, 160, 256, 384** 之一 |
| 4 | Top-K, shared expert | **K=8, 1 shared** (V3 派共识) |
| 5 | 路由函数 | **sigmoid + ALF** (V3 派) 或 **softmax + aux** (Mixtral 派) |
| 6 | Attention 形态 | **MLA** (V3/K2) 或 **GQA + 重压缩** (GLM/Ling/Qwen) |
| 7 | 层数 (depth vs width) | **GLM 深 (92L)** vs **LongCat 浅 (28L)** vs **中庸 (60L)** |
| 8 | MTP D 与 λ schedule | **D=1, λ=0.1-0.3** |
| 9 | Optimizer | **AdamW** (默认) 或 **Muon** (Moonshot/Z.ai 派) |
| 10 | LR Scheduler | **WSM** (Ling) > **WSD** (V3) > cosine (GLM) |
| 11 | 精度 | **BF16** (默认) 或 **FP8** (1T 段) |
| 12 | 并行拓扑 (EP/PP/TP/NLR) | **EP+PP 主导**，必开 NLR |

---

## Step 1：目标 active params（你的 FLOPs 预算）

**X 因为 Y**：FLOPs ∝ active params × 6 (Chinchilla 公式)。所以 active 决定训练成本上限。

| 段位 | active 区间 | 训练成本 (H100, 25T tokens) | 典型 |
|---|---|---|---|
| 16-30B base | 1.4-3.5B | $0.5-1.5M | Ling-mini, V2-Lite, Qwen3-30B |
| 80-120B | 6-17B | $4-9M | Ling-flash (6.1B), Llama4-Scout (17B), GLM-4.5-Air (12B), dots1 (14B) |
| 200-400B | 15-50B | $12-25M | V2-236B (21B), GLM-4.5 (32B), Step-3 (38B), V3 (37B) |
| 500B-1T | 27-55B | $25-60M | LongCat (27B), Ling-1T (51B), K2 (32B) |

**决策依据**：
- 客户场景需要的 max inference latency → active 上限 (典型 30B active 是 H100 推理舒适区上限)
- 训练预算 → 设定 active 后倒推 token 数 (Chinchilla optimal: token ≈ 20 × active for dense, 25-35× active for MoE)
- **不要直接选 total params**！total 是 step 2-3 的派生量

---

## Step 2：活率 = active / total

[[18_params_vs_flops]] (Apple) + [[37_ling1t]] (Ling Scaling Law) 双重证据：**sparser + larger total 在固定 FLOPs 下永远更优**。

| 活率区间 | 段位 | 代表 | 备注 |
|---|---|---|---|
| 1/4 ~ 1/6 (半稠密) | 老一代 | Mixtral, DBRX | 已过时 |
| 1/8 ~ 1/12 (V3 主流) | 100-400B | V3 (1/18), GLM-4.5 (1/11), Qwen3-235B (1/11) | 安全默认 |
| 1/15 ~ 1/25 (新主流) | 100B-1T | Ling-flash (1/17), GLM-4.5-Air (1/9 偏稠), K2 (1/32), Ling-1T (1/20) | **强 reasoning + 大 batch 友好** |
| 1/30+ (极稀疏) | 1T 段 | K2 (1/31), Intern-S1-Pro (1/45), Qwen3-Next (1/27) | 实验性 |

**X 因为 Y**：active 越少，每 token FLOPs 越少，**可以训更多 tokens** = D-scaling 收益放大。但稀疏度越大，infra 复杂度越高（all-to-all 通信占比上升）。

**决策**：
- 简单 chat / general LLM → 1/8-1/12
- Reasoning / coding / agent → 1/15-1/25（你可以接受 active 少，需要 total capacity 大）
- 总参 > 500B → 1/20+ 几乎必选（否则 active 拉到 40B+ inference 受不了）

---

## Step 3：Expert 数 N_routed（离散跳档）

[[17_finegrained_scaling]] (Krajewski) 给 G_opt 公式 → N_routed = E × G ≈ 64 × {16, 32, 64} = {128, 256, 512}

**N_routed 离散跳点**：64, 128, 160, 256, 384, 512 （都是 EP topology 整除友好的数）

| N_routed | 段位 | 代表 |
|---|---|---|
| 64 | 16B (老一代) | DeepSeekMoE-16B, V2-Lite, Moonlight |
| 128 | 100B | dots1 (128), GLM-4.5-Air (128), gpt-oss-120b (128) |
| 160 | 100-300B | Qwen3-Coder (160), GLM-4.5 (160), GLM-4.6 (160), V2-236B (160) |
| **256** | **100B-1T 主流** | **V3, V3.1, V3.2, Ling-mini/flash/1T, MiniMax-M2, dots1 派少数** |
| 384 | 1T (K2 风格) | Kimi K2 (384) |
| 512 | 80B-560B 极稀疏 | Qwen3-Next (512), LongCat (512+256 zero) |

**决策**：
- **若你跟 V3 派 / Ling 派 → 默认 256**
- **若你跟 GLM 派 → 160 (100B) 或 160 (300B+)**
- **若你跟 Qwen / LongCat 派 → 512** (但 EP topology 必须能整除 512)
- **EP topology 约束**：N_routed 必须能被你的 EP size 整除。EP=8 → N ∈ {64, 128, 256, 384, 512}；EP=16 → {128, 256, 384, 512}；EP=32 → {128, 256, 512}

---

## Step 4：Top-K 和 Shared Expert

[[28_open_source_moe_catalog]] §3.2 实证：**2025+ 主流 K=8**，全 100B+ 段位（除 Llama 4 K=1, MiniMax K=2 等少数反例）。

| 配置 | 个数 (catalog 66 个独立 release) | 段位特点 |
|---|---|---|
| K=8 | 24 | 100B+ 段位标配 |
| K=2 | 14 | Mixtral 派老模型 |
| K=4 | 3 | gpt-oss, DBRX |
| K=1 | 4 | Llama 4, Switch (历史) |
| K=12 | 1 | LongCat (含 zero-experts) |

**Shared expert**：
- 0 shared：Mixtral 派 + Qwen3 (除 Coder) + Llama 4 (用"shared MLP"不算 shared expert)
- 1 shared：V3 / K2 / Ling 全系 / GLM / dots1 / Hunyuan-A13B
- 2 shared：DeepSeekMoE / V2-Lite / Moonlight (V2 派旧设计)

**决策**：**默认 K=8 + 1 shared expert**。除非：
- 极简 (Mixtral 复刻) → K=2 + 0 shared
- 极致推理速度 (Llama 4) → K=1（**但 benchmark 弱，不推荐**）
- 动态 active (LongCat) → K=12 + 256 zero-experts

---

## Step 5：路由函数

| 范式 | 路由 + balance | 代表 (2025+ 100B+) |
|---|---|---|
| **V3 派** | **sigmoid gate + Aux-Loss-Free bias** | V3, V3.1, V3.2, K2, Moonlight, **Ling 全系, GLM 全系, dots1 (变体)** |
| Mixtral 派 | softmax + aux loss | 老一代 Mixtral, DBRX, OpenAI gpt-oss, Llama 4, ERNIE 4.5 |
| 特殊 | grouped (Pangu MoGE), attention router (Yuan-M32), GRIN (Phi-3.5) | 单点尝试 |

**ALF 变种**：
- **V3 原版**：`b_i ← b_i + u·sign(e_i)`
- **Ling zero-mean**：`b_i ← b_i + u·(sign(e_i) − mean(sign(e)))` —— 防止 bias 整体漂移
- **LongCat PID controller** ([[36_longcat]] Eq. 2)：把 bias 升级为 negative-feedback controller

**routed_scaling_factor**:
- **V3 派固定 2.5** —— 这是 V3/dots1/Ling/GLM 共有指纹
- LongCat 用 6 (因为 zero-experts 数多)

**决策**：
- **默认走 sigmoid + ALF** (V3 派)，scaling factor = 2.5
- 若做 16B-100B：用 Ling zero-mean 修正
- 若做 200B+：考虑 LongCat PID（实现复杂）
- 训练后期 bias rate 可降 (V3=直接关, Ling=降 10× 不关, GLM=15T 后关) —— 这是开放问题

---

## Step 6：Attention 形态

| 类型 | 代表 | 段位 | KV cache 单位 token |
|---|---|---|---|
| **MHA** | 老一代 | 16B 段位 | n_head × d_head × n_layer (典型 19GB/64K tokens) |
| **GQA (重压缩 96Q/8KV)** | GLM-4.5, Ling-1T, Qwen3-235B, dots1 | 80B-1T | 1/8 ~ 1/12 of MHA |
| **MLA** | V3, K2, V2-236B, LongCat | 200B-1T | 1/20 (latent 压缩到 ~512) |
| **MFA** | Step-3 | 320B | 试验性 |
| **Hybrid** (Mamba/Lightning/DeltaNet + softmax) | MiniMax-01/M1, Qwen3-Next, Jamba, Granite 4 | 80B-560B | 不可比较 |
| **Sparse Attention (DSA)** | V3.2-Exp | 671B | 实验性 |

**X 因为 Y**：100B+ 模型 prefill 时 KV cache 经常 20GB+，是 H100 内存最大占用。

**决策**：
- 100B 段：**GQA 32Q/8KV + QK-norm**（dots1/Ling-flash/Air 共识）
- 200B 段：**GQA 重压缩 (96Q/8KV) 或 MLA**（如果 infra 有 MLA kernel）
- 400B+ 段：**MLA 几乎必选**（KV cache 已不可控）
- **如果你不是 DeepSeek 阵营**：MLA training kernel 风险（FlashMLA 是 inference only，训练侧 vLLM/SGLang 还在追）→ 保守用重 GQA
- **Hybrid attention 是 long-context 才考虑**（128K+ 才有明显收益），且 MiniMax-M2 退回 full softmax 是反向信号

---

## Step 7：层数（depth vs width）

[[32_depth_width_tradeoff]] §2 全市场分布表。100B+ 段位三个流派：

| 流派 | L/√H | 代表 (100B+) | rationale |
|---|---|---|---|
| **GLM 深** | 1.06-1.29 | GLM-4.5 (92L/5120), Qwen3-Coder (62L/6144), dots1 (62L/4096) | reasoning 受益于深度 |
| **中庸** | 0.5-0.72 | V3 (61L/7168), K2 (61L/7168), GLM-4.5-Air (46L/4096), Llama 4 Scout (48L/5120) | inference latency 和 quality 平衡 |
| **LongCat 极宽** | 0.36-0.44 | LongCat (28L/6144), Ling-flash (32L/4096), Qwen3-30B (48L/2048) | computation-communication overlap 窗口大 |

**X 因为 Y**：
- 深 → 每 token decode 顺序步多 → inference latency 高，但 CoT 用户对 latency 不敏感 → GLM 选深
- 浅 → PP bubble 小，但 reasoning capacity 上限低 → LongCat 选浅
- 中庸 → V3/K2 选

**决策**：
- 产品场景 = reasoning / coding agent → **L/√H ≈ 1.0-1.3** (GLM 路线)
- 产品场景 = chat / general → **L/√H ≈ 0.5-0.7** (V3 路线)
- 产品场景 = throughput-critical / 大 batch → **L/√H ≈ 0.4** (LongCat 路线)

---

## Step 8：MTP 配置

[[20_mtp_gloeckle]] + [[23_mtp_investigation]] 双重论证。**[[28_open_source_moe_catalog]] §3.6 实证**：MTP 在 sigmoid+ALF 阵营 ~70% 采用，softmax+aux 阵营 ~0%。

**两个变种**：
- **D=1**（V3 风格 single chain）：V3, V3.1, V3.2, K2 (D=0), Moonlight, Ling 全系, GLM 全系, Qwen3-Next
- **D=3**（多 chain）：MiniMax-M2 唯一

**λ schedule（weight）**：
- **Ling**：λ=0.1 恒定
- **V3**：λ=0.3 → 0.1 at end
- **GLM**：λ=0.3 first 15T → 0.1 remaining (类 V3)
- **K2**：D=0（不开 MTP）

**决策**：
- **默认 D=1, λ=0.1**（Ling 风格，最保守）
- 若你想拿 MTP 在 inference 加速（speculative decoding） → 必开 D=1
- 若你怀疑 MTP 损 main model → wind tunnel 验证开关差 ≤ 0.001 loss 即固化
- 若你 active < 3B → MTP 是 boundary case ([[23_mtp_investigation]] §8)，建议先不开
- **不要 D=3**（MiniMax-M2 孤例，没人复现）

---

## Step 9：Optimizer

详见 [[39_muon]] 完整论证。

| Optimizer | 100B+ 段位代表 |
|---|---|
| AdamW | V3 全系, Qwen 全系, dots1, **Ling 全系（含 1T！）**, LongCat, Hunyuan, Hy3 |
| **Muon (modified)** | **Moonlight, Kimi K2 (+ MuonClip), GLM-4.5 / 4.5-Air / 4.6** |

**X 因为 Y**：
- Muon 对 MoE router 权重正交化收益大 ([[39_muon]] Figure 4)
- Muon dense scaling law 给 ~2× compute efficiency
- **但 Ling-1T 1T 段位坚持 AdamW + 取得 SOTA** → 说明 Muon 不是必要
- Muon kernel 实现复杂度 / Megatron-LM 支持度低于 AdamW

**决策**：
- **默认 AdamW** (跟主流 + 跟 Ling 派 + 训练成熟度高)
- 若有 wind tunnel B 预算 → 必加 Muon 对照（800M 模型 100B tokens, ~$15K, 看 ≥ 0.005 loss 改善则切）
- 若你 ε 选 1e-16（LongCat 风格）或 1e-20（DeepSeek 风格）—— **不要默认 1e-8**

---

## Step 10：LR Scheduler

| Scheduler | 100B+ 段位代表 | 关键证据 |
|---|---|---|
| **WSM (Warmup-Stable-Merge)** | **Ling 全系 (mini/flash/1T)** | [[37_ling1t]] §3.2.3 Figure 7 = WSM > WSD +1~2pt 全 benchmark |
| **WSD (Warmup-Stable-Decay)** | V3, K2, Moonlight, dots1, LongCat | [[06_kimi_k2]] / [[36_longcat]] / [[24_dots1]] 全用 |
| **Cosine** | **GLM-4.5 全系** | [[35_glm45]] §2.4 明确 reject WSD ("underfits in stable stage") |

**X 因为 Y**：
- WSD 主流但需要预定 decay 时长 → 不灵活
- WSM 替换 LR decay 为 checkpoint merging → 灵活，且 Ling 1T 上 +1-2pt
- Cosine 是经典稳定，但 GLM 用 Muon 时 cosine 比 WSD 强 → **GLM 的 reject WSD 与 Muon 耦合，不是 cosine 独立证据**

**决策**：
- **默认 WSM** (Ling 给最大规模实证)
- 若 AdamW + 不愿存 32 个 checkpoint → WSD
- 若 Muon → 跟 GLM 走 cosine

---

## Step 11：精度（BF16 vs FP8）

| 段位 | 推荐 | 代表 |
|---|---|---|
| 16-100B | **BF16** | 几乎全部 |
| 100-400B | BF16 or FP8 partial | V3.1 (FP8 partial), DeepSeek-V3.2-Exp |
| 400B-1T | **FP8 (fine-grained)** | **Ling-1T** (largest FP8 base, ≤0.25% gap), V3.1, V3.2 |

**Ling-1T FP8 配方**：
- Activations / gradients: **per-row [1, 128]** quantization
- Weights: **per-block [128, 128]** quantization
- ≤ 0.25% loss gap to BF16 after 900B tokens
- 15%+ end-to-end speedup, 15%+ memory reduction

**决策**：
- **100B-300B 段位用 BF16**（简单，FP8 ROI 低）
- 400B+ 段位考虑 FP8（参考 Ling-1T 配方）
- LongCat 也是 BF16 + hidden z-loss(λ=1e-7) 防爆 → **如果 BF16 + 大模型 → 必加 hidden z-loss**

---

## Step 12：并行拓扑

| 段位 | 推荐拓扑 | 拓扑约束 |
|---|---|---|
| 16-30B | EP=8 single-node, PP=1, DP=fill | **不开 NLR** |
| 80-150B | EP=16 (2 node), PP=2-4, DP=fill | **开始考虑 NLR** |
| 200-400B | EP=32 (4 node), PP=4-8, TP=2, DP=fill | **NLR M=4 必须** |
| 500B-1T | EP=64+ (8+ node), PP=8-16, TP=1-2, DP=fill | **DualPipe 或 1F1B-overlap 必须** |

详细见 [[25_node_limited_routing]] + [[04_deepseek_v3]] §5。

**Special**：
- **MTP layer 需要单独 PP partitioning** ([[37_ling1t]] §infrastructure)
- **First-K-Dense layer 是异构** → 1F1B 需要 partial recomputation
- **Hidden z-loss (λ=1e-7) 配合 BF16 必须** ([[36_longcat]] Eq. 10)

---

## 附录 A：5 个典型配置 cookbook

### Profile P1：100B Reasoning-Focused (GLM-4.5-Air 路线)
```yaml
total: 106B   active: 12B   activation_ratio: 11.3%
layers: 46    hidden: 4096    L/sqrt(H): 0.72
experts: 128 routed + 1 shared    top_k: 8
attention: GQA 96Q/8KV + partial RoPE   QK-norm: No (Air不用)
routing: sigmoid + ALF + 2.5 scaling
MTP: D=1, λ=0.3→0.1
optimizer: Muon (GLM 派)
scheduler: cosine (GLM 派, 反对 WSD)
precision: BF16
parallelism: EP=16, PP=4, NLR M=4
tokens: 22-25T
```

### Profile P2：100B Efficiency-First (Ling-flash 路线)
```yaml
total: 103B   active: 6.1B   activation_ratio: 5.9%
layers: 32    hidden: 4096    L/sqrt(H): 0.50
experts: 256 routed + 1 shared    top_k: 8
attention: GQA 32Q/4KV + QK-norm + partial RoPE 前 64 dim
routing: sigmoid + ALF (zero-mean) + 2.5 scaling
MTP: D=1, λ=0.1 恒定
optimizer: AdamW (β1=0.9, β2=0.95)
scheduler: WSM (N=16 checkpoint avg)
precision: BF16 + hidden z-loss λ=1e-7
parallelism: EP=16, PP=2, NLR M=4
tokens: 20T+
```

### Profile P3：220B Lean (插值 真空带, [[38_100b_to_200b_gap]] §4)
```yaml
total: 220B   active: 13B   activation_ratio: 5.9%
layers: 48    hidden: 4608
experts: 256 routed + 1 shared    top_k: 8    first_k_dense: 2
attention: GQA 36Q/4KV + QK-norm + partial RoPE
routing: sigmoid + ALF (zero-mean)
MTP: D=1, λ=0.1
optimizer: AdamW
scheduler: WSM
precision: FP8 fine-grained quant
parallelism: EP=32, PP=6, TP=2, NLR M=4
tokens: 22-25T
```

### Profile P4：355B GLM-4.5 复刻
```yaml
total: 355B   active: 32B   activation_ratio: 9%
layers: 92    hidden: 5120    L/sqrt(H): 1.29 (极深)
experts: 160 routed + 1 shared    top_k: 8    first_k_dense: 3
attention: GQA 96Q/8KV (head_dim=128) + QK-norm + partial RoPE
routing: sigmoid + ALF (V3 原版 sign) + 2.5 scaling
MTP: D=1, λ=0.3 first 15T → 0.1
optimizer: Muon (Newton-Schulz N=5, μ=0.95, RMS=0.2)
scheduler: cosine (LR 2.5e-4 → 2.5e-5, batch 16M→64M)
precision: BF16
parallelism: EP=32, PP=16, NLR M=4, DualPipe
tokens: 23T (15T general + 7T code/reasoning + 1.1T mid-training)
```

### Profile P5：1T Reasoning Foundation (Ling-1T 路线)
```yaml
total: 1000B   active: 51B   activation_ratio: 5.1%
layers: 80    hidden: 8192    L/sqrt(H): 0.88
experts: 256 routed + 1 shared    top_k: 8    first_k_dense: 4
attention: GQA 64Q/32KV (head_dim=128) + QK-norm + partial RoPE 前 64 dim
routing: sigmoid + ALF (Ling zero-mean) + 2.5 scaling
MTP: D=1, λ=0.1 恒定
optimizer: AdamW (β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0)
scheduler: WSM (N=32 checkpoint avg, linear warmup 2000 steps to peak LR=1.86e-4)
precision: FP8 fine-grained (act/grad [1,128], weight [128,128])
parallelism: EP=64+, PP=16+, interleaved 1F1B + partial recomputation, NLR M=4-8
tokens: 20T pretrain + 750B mid-training
```

---

## 附录 B：常见反模式（不要做）

| 反模式 | 为什么不要 | 反例代价 |
|---|---|---|
| K=1 | Llama 4 路线，公开 benchmark 偏弱 | Llama 4 Scout / Maverick 在 SWE-bench 仅 ~40 |
| 0 shared expert + V3 派路由 | Mixtral/Qwen 派不带 shared 工作良好，但你已经选 sigmoid+ALF 就跟 V3 派标配 1 shared | – |
| MoE MTP module（不是 dense） | V3 用 MoE MTP，Ling/LongCat 改 dense MTP | Ling 1T / LongCat 都说 dense MTP 更稳 |
| 半稠密 (active/total > 25%) | 2024 老一代设计，被新一代以 "更稀疏 + 更多 expert" 取代 | Mixtral 8x22B 已被市场淘汰 |
| MLA without α_q/α_kv 修正 | LongCat [[36_longcat]] §2.3.1 证明大模型不修正会不稳 | – |
| BF16 训练不加 hidden z-loss | LongCat 在 1T 段位实测必加 | OLMo 早期也踩过 |
| cosine LR 配 AdamW + reasoning data | GLM 反 WSD 但 Muon 耦合；AdamW + cosine 缺论文背书 | 信号空白 |
| Aux loss 主导 (λ > 0.01) | ALF + sigmoid 范式后，aux loss 主导被 reject | dots1 verified |

---

## 附录 C：决策依赖图

```
Step 1 (active)
  ↓
Step 2 (activation ratio) ─→ Step 3 (N_routed)
                                 ↓
                            Step 4 (K, shared)
                                 ↓
Step 5 (routing) ─────────→ Step 6 (attention)
  ↓                              ↓
Step 7 (depth/width)         Step 8 (MTP)
  ↓                              ↓
Step 9 (optimizer) ←─→ Step 10 (scheduler)
                                 ↓
                          Step 11 (precision)
                                 ↓
                      Step 12 (parallelism)
```

**关键耦合**：
- Step 6 MLA vs GQA → Step 12 EP topology（MLA + DeepSeek TileLang kernel 必须）
- Step 9 Muon → Step 10 cosine 而非 WSD（GLM 经验）
- Step 11 FP8 → Step 12 需要 fine-grained quantization 配置
- Step 7 高瘦 → Step 12 PP 更多 stage (bubble 更严重)

---

## 与本仓库其他笔记交叉引用

- **段位拓扑**：[[28_open_source_moe_catalog]] §2 全市场分布
- **理论基础**：[[17_finegrained_scaling]] / [[18_params_vs_flops]] / [[37_ling1t]] §2.3 scaling laws 三家
- **决策备忘**：[[22_FINAL_16B_design]] (16B 段) / [[38_100b_to_200b_gap]] (200B 真空)
- **路由实现**：[[03_auxloss_free]] + [[30_routing_implementation]]
- **训练系统**：[[25_node_limited_routing]] (NLR) + [[36_longcat]] (ScMoE) + [[37_ling1t]] (1F1B-overlap)
- **优化器**：[[39_muon]] (Muon 完整) + [[37_ling1t]] (AdamW 路线)
- **稳定性**：[[36_longcat]] (hidden z-loss, ε=1e-16) + [[37_ling1t]] (FP8 quant)
- **架构创新**：[[26_attention_residuals]] / [[27_mhc]] (备用 add-on)
- **MTP**：[[20_mtp_gloeckle]] + [[23_mtp_investigation]] (D=1 vs D=3 决策)
- **个体模型笔记**：[[04_deepseek_v3]] / [[06_kimi_k2]] / [[08_ling_2]] / [[24_dots1]] / [[35_glm45]] / [[36_longcat]] / [[37_ling1t]]
