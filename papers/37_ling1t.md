# Ling 2.0 / Ling-1T: Every Activation Boosted — Scaling General Reasoner to 1T Open Foundation

- **arXiv**: 2510.22115 (v2, 7 Nov 2025)
- **机构**: Ant Group / Inclusion AI (Ling Team)
- **开源**: github.com/inclusionAI/Ling-V2, huggingface.co/collections/inclusionAI/ling-v2
- **Ring 系列同源**: Ring-flash-2.0 / Ring-1T 在 Ling base 上加 RL 训练（icepop 等）

## TL;DR

Ling 2.0 是 **"统一 scaling law 驱动的 16B → 100B → 1T 三档同构家族"**，所有三个模型用**完全相同的核心配方**（256 routed + 1 shared + top-8 + 3.5% 激活率 + MTP D=1 + partial RoPE + QK-Norm + WSM scheduler），只在 layer / hidden / heads 上 scale up。三大独特贡献：

1. **Ling Scaling Law** (Tian et al. 2025a) —— EL(A, G, C) 公式，**激活率 A 是 efficiency 主驱动**，granularity G 是非线性 modulator (最优 8-12)，compute C 有 amplification effect。1000 次实验拟合，validation MSE < 0.01
2. **Ling Wind Tunnel** —— 5 个 500M-8B 模型组成的标准化实验流水，**单次 ablation 成本仅传统方法 35%**，所有新想法先走 wind tunnel 再上 1T
3. **WSM (Warmup-Stable-Merge) Scheduler** —— **完全取代 LR decay**，用 32 个 checkpoint 平均替代 annealing 阶段，理论上等价于 LR decay (Eq. 2-3)，实测 +1~2 pt 全 benchmark

**Ling-1T = 当前最大的 FP8 base 模型**（≤0.25% loss gap vs BF16 after 900B tokens）。

## 核心命题

1. **激活率 (active/total) 是 MoE efficiency 第一性指标** —— 不是 active params, 不是 total params
2. **"高稀疏 + 细粒度"路线在 1T 段位 7× 优于 dense** (efficiency leverage EL=7×, [[18_params_vs_flops]] Apple 论文的 1T 段验证)
3. **MoE 的最优 hyperparameter 可以用小模型外推到大模型** —— 通过 Ling Wind Tunnel + Scaling Law，validation loss 预测误差 < 0.01
4. **WSM 在 reasoning 模型上比 WSD 强 1-2 pt** —— 第一个**用同 base、改 scheduler** 严格对照实验
5. **MoE 的 ε 选择**：默认 AdamW (β1=0.9, β2=0.95) —— 跟 V3 / dots1 一致

## 完整架构 Spec (Table 1 直接搬运)

| 维度 | **Ling-mini-2.0** | **Ling-flash-2.0** | **Ling-1T** |
|---|---|---|---|
| Layers | **20** | **32** | **80** |
| Experts (total routed) | 256 | 256 | 256 |
| Active per Token (routed) | 8 | 8 | 8 |
| Shared Experts | 1 | 1 | 1 |
| **Activation Ratio** | **8.75%** | **5.9%** | **5.1%** |
| Attention Heads | 16 | 32 | 64 |
| **KV Heads (GQA)** | 8 | 16 | 32 |
| Dense Layers (First-K-Dense) | 1 | 1 | **4** |
| Hidden Dim | 2048 | 4096 | **8192** |
| Intermediate Dim (FFN dense) | 5120 | 9216 | 18432 |
| Expert Intermediate Dim | 512 | 1024 | 2048 |
| Attention Head Dim | **128 (固定全规模)** | 128 | 128 |
| Total Params | 16B | 103B | **1000B** |
| Activated Params | 1.4B | 6.1B | **51B** |
| Learning Rate | 3.36e-4 | 2.61e-4 | **1.86e-4** |
| Batch Size | 4400 | 8352 | **18144** |

→ **head_dim 固定 128**，attention heads 跟 layer 数同步 scaling（不像 V3 / K2 那样独立调）。

→ **First-K-Dense = 4 for 1T**（不是 3 也不是 5）—— 论文说"reduces total param count while maintaining equivalent model performance and improving routing balance"

→ **Activation Ratio 从 16B 的 8.75% 降到 1T 的 5.1%** —— scaling law 给的"越大越稀疏"路线证实。但**total experts 不变**（都是 256），所以稀疏度由 hidden/expert intermediate ratio 推出

## Ling Scaling Law 公式

### Efficiency Leverage (EL) 定义
EL = (dense 达到同一 loss 所需 compute) / (MoE 达到同一 loss 所需 compute)

### 主 scaling law (Eq. 1, §2.3.2)

$$
EL(A, G, C) = \hat{A}^{\alpha + \gamma (\log G)^2 + \beta \log G}
$$

其中：
- $A$ = activation ratio (active/total params)
- $\hat{A}$ = saturating transform of A (Clark et al. 2022 风格)
- $G$ = expert granularity = $d_{FFN}^{dense}$ / $d_{FFN}^{expert}$
- $\alpha = a + d \cdot \log C$ —— **关键：α 随 compute 增加**，这是"compute amplification effect"

### 4 个发现 (§2.3.2)

1. **Activation ratio 是主驱动**（power law），在 1/128 这种极稀疏区间仍稳
2. **Granularity 8-12 是最优** —— [[17_finegrained_scaling]] Krajewski 给 G=16-32，Ling 实证给 8-12（Ling 选 G=18432/2048≈9 是直接证据）
3. **Compute amplification**：EL 随 C 上升（compute 越大，MoE 越值）
4. **Shared expert 数 / arrangement 是次要因素**（小修正）

### 与 Apple 2501.12370 / Krajewski 2402.07871 的关系
- Apple: 给 N×D×S 的三维 scaling law，结论 "sparser+larger 永远更优"
- Krajewski: 给 N×D×G 的三维 law，结论 "G 应随 compute 单调 +3 dB 一档"
- **Ling: 把 A 和 G 合并到 EL 这个 derived metric 里**，并加 compute amplification 项 → 三家说的是同一回事，Ling 的式子最适合 production 用

## Routing 细节

### sigmoid + ALF（V3 派标配）
- routed_scaling_factor = 2.5（与 V3 / dots1 / GLM-4.5 一致 → 这是"V3 配方指纹"）

### Zero-mean ALF bias 修正（关键创新, Eq. from Liu et al. 2025a）

$$
b_i \leftarrow b_i + u \cdot (\text{sign}(e_i) - \text{mean}(\text{sign}(e)))
$$

对比 V3 原版 `b_i ← b_i + u·sign(e_i)`：减去 batch 内 sign 均值，**保证 bias 整体均值不漂移**。

→ **这是 [[03_auxloss_free]] 的 ALF 改良版**，Ling 全系（mini/flash/1T）都用。
→ V3 / dots1 / GLM-4.5 不用（保留原版 sign）。
→ 你 16B Profile B 已经决定跟 Ling 走 zero-mean —— 现在在 1T 段位有了进一步背书。

### 其他路由设置
- **Dropless routing**（不丢弃 token，dropless gating）
- **Group routing**（group balance，类似 V3 device-level loss）
- **Bias update γ = 0.001 in pre-training, 0.0001 after context extension**
  - 对比 GLM-4.5：0.001 first 15T, 0 thereafter
  - 对比 V3：similar to Ling
  - **Ling 不把 bias update 关掉**，只是降一个量级 → 论文未解释 why

## MTP 配置

- **D = 1**（单 chain）
- **MTP loss weight λ = 0.1**（GLM-4.5 用 0.3→0.1，Ling 直接用 0.1，恒定）
- **每个 model size 一个 MTP layer**
- **Fine-grained PP partitioning for MTP module within Megatron** —— 单独切片防 PP bubble
- **All parameters randomly initialized with std = 0.006** —— Ling 自己的 init 风格

→ 对你 16B：**Ling 的 λ=0.1 比 GLM 的 0.3 保守**。如果你 wind tunnel 时 MTP loss 不稳，先用 0.1 测试，不要直接跳 0.3。

## WSM Scheduler 详解 (§3.2.3)

### 核心想法
- 不做 LR decay
- 改为 stable LR 一直训到最后
- **最后 N=32 个 checkpoint 取参数平均** 作为 final model
- 论文证明（Eq. 2-3）：合适的 merge 权重 $c_j$ 数学上等价于某个 LR decay schedule

### 公式
final model = $\sum_{j=0}^{k} c_j \theta_{n+j}$，等价 gradient 加权:
$$
\hat{\theta}_{n+k} = \theta_n - \sum_{i=1}^{k} w_i g_{n+i-1}
$$

通过 $c_k = w_k$ 等关系反求 $c$，即可让 merge 复刻任意 LR decay。

### 实证结论 (Figure 7)
- WSM 全 benchmark +1~2 pt 平均
- 优势经 SFT 5 epoch 仍持续
- **不需要预先决定 decay budget**（WSD 必须预定 decay 时长，WSM 不需要）

### 对你 16B 的启示
- Ling 给 WSM **正面证据**，GLM 给 WSD **反面证据**，两者都说 cosine 不行
- 但**两家都用 sigmoid+ALF**，区别在 optimizer (Muon vs AdamW) 和 scheduler
- **你 16B 用 AdamW + Ling 系配方**，那应该跟 WSM
- 如果你**想做对照**：WSM vs WSD vs cosine 三选，你的 wind tunnel B 可以加入

## FP8 训练 (§Infrastructure)

### Quantization 粒度
- **Activations / gradients: [1, 128]** (per-row groups of 128 elements)
- **Weights: [128, 128]** (per-block 128×128)

### 精度结果
- "≤ 0.25% loss gap to BF16 **after 900B tokens**"
- 15%+ end-to-end throughput improvement
- 15%+ memory reduction

→ **这是当前已知最大 FP8 base 模型**，比 V3.1 (FP8 partial) 和 K2 (BF16) 更激进。

### 对你 16B：可以暂缓 FP8
- 16B 段位训 20T tokens 也才 ~1 周（H100×64），BF16 完全 hold
- FP8 在 1T 段位才显著节省，16B 段位 ROI 很低
- 但如果你**有 H100 / H800 集群 + 想压成本**，可以 wind tunnel 验证一下

## 异构 1F1B Pipeline (§Infrastructure)

- **Interleaved 1F1B + partial recomputation**
- 专门处理 **MTP 模块和 First-K-Dense 模块带来的异构性**（这些 layer 不是标准 MoE，PP 切片会产生气泡）
- **+40% throughput** —— 但未给绝对配置（PP/EP/TP 数）

→ 对你 100B+ 设计有直接借鉴价值。

## Pre-training 流水 (Figure 5)

```
Pre-training Substage 1 (4K, 10T tokens):
   68% general / 32% reasoning

Pre-training Substage 2 (4K, 10T tokens):
   54% general / 46% reasoning  ← 注意 reasoning 比例上升

Mid-training Stage 1 (32K, 150B tokens):
   54% general / 46% reasoning + long-context
   YaRN extension to 128K

Mid-training Stage 2 (32K, 600B tokens):
   55% general / 45% reasoning + CoT data
   "Reasoning Ability Pre-Activation"
```

总：**20T pretrain + 750B mid-training ≈ 20.75T**

→ 关键策略：**reasoning data 比例从 32% 渐升到 46%** —— 不是一开始就高。
→ Stage 2 的 600B CoT "pre-activation" 给了 RL 阶段更好起点。

## Post-training 三件套

### 1. DFT (Decoupled Fine-Tuning, §Post-Training)
- 用 differentiated system prompts 做 SFT
- 建立 "reasoning-focused" 初始化

### 2. Evo-CoT (Evolutionary Chain-of-Thought)
- 渐进深化推理能力
- **25% fewer training tokens** 比 SOTA 达到同等数学性能（AIME 2025 benchmark）

### 3. LPO (Linguistic-unit Policy Optimization)
- **句子级别**而非 token / sequence 级别的 policy optimization
- 把 sentence 当 fundamental action unit
- **复杂推理 benchmark +10%** vs token-level baseline

### 4. GAR (Group Arena Reward) for RLHF
- Group-based intra-group preference alignment
- Open-ended evaluations consistency +2~10%

## 与 16B / 100B 设计的关系

| 维度 | 你的 16B Profile B | Ling-mini-2.0 | 是否对齐 |
|---|---|---|---|
| Layers | 27 | **20** | **不一致** — 你比 Ling 深，但 Ling 更短宽 |
| Hidden | 2048 | 2048 | ✅ |
| Total / Active | 16B / 2.4B (1/6.5) | 16B / 1.4B (**1/11.4**) | **不一致** — Ling 更稀疏 |
| Experts | 64 routed + 1 shared, K=8 | **256 routed** + 1 shared, K=8 | **大差异** — Ling 走 256+8 极细粒度 |
| QK-Norm | 是 | 是 | ✅ |
| Partial RoPE | wind tunnel 待定 | **是（前 64 dims）** | **Ling 是 16B 唯一公开用 partial RoPE 的** |
| ALF zero-mean | 是 | 是 | ✅ |
| WSD vs WSM | WSD | **WSM** | **冲突信号** —— 你应该考虑切 WSM |
| MTP D=1 | wind tunnel 待定 | 是, λ=0.1 | Ling λ 是数据点 |
| First-K-Dense | 1 dense layer | 1 dense layer | ✅ |
| Group routing | 你没用 | 是（device-level group balance） | 16B EP=8 单节点不需要 |

### ⚠️ 三个值得 wind tunnel 验证的差异

1. **Expert 数 64 vs 256**：Ling 给 8.75% 激活率，你 Profile B 给 ~15%。Ling 更稀疏，但**你的 wind tunnel A2 anchor mHC 是 64 expert**，更接近你的现状。Open question：16B 段位 256 是否过度稀疏？
2. **WSD vs WSM**：Ling 强力背书 WSM，但 WSM 需要**保存 N=32 个 checkpoint** 占额外存储。如果你存储紧，WSD 仍可以。
3. **Partial RoPE 前 64 dims**：Ling 唯一在 16B 段用 partial RoPE 的开源模型。这个比 full RoPE 简单（不影响 head_dim < 64 的 head），但需要验证在你的 attention config 下不掉下游。

## Settled vs Open

### Settled
- 256 routed + 1 shared, K=8 在 16B-1T 三个量级都最优
- 激活率 ~3.5%-9% 是 100B+ 的 sweet spot
- zero-mean ALF bias 是 ALF 改良版
- FP8 在 1T 段位 ≤0.25% loss gap（已经无脑用）
- WSM > WSD on Ling base + AdamW
- MTP D=1 + λ=0.1 (Ling 派) 或 λ=0.3→0.1 (GLM 派)

### Open
- Granularity 最优值：Krajewski 给 16-32，Ling 给 8-12 → 不同实验得不同结论
- bias update 后期是否关掉（V3=关, Ling=降不关, GLM=关）
- Partial RoPE 前 64 dim 在不同 head_dim 下的迁移性
- WSM 的 N=32 是定值还是 scaling with model size
- FP8 quantization 粒度 [1,128] / [128,128] 是不是最优

### 已否决
- LR decay (Ling 明确替换为 WSM)
- Single dense layer for 1T (1T 用 4 dense, 不是 1)
- Standard ALF bias update (用 zero-mean 修正)

## 与 16B 设计的最重要单条信号

**你 16B Profile B 当前选 64 routed + 1 shared = 8.7% 激活率**。Ling-mini-2.0 用同 16B 段位选了 **256 routed + 1 shared = 8.75% 激活率**。

虽然两者**激活率几乎相同**（8.7% vs 8.75%），但 Ling 用 **256/8 = 32 倍精细化路由**，你用 **64/8 = 8 倍**。

如果按 Krajewski G_opt = E×G，假设 E=64 时 G_opt=8 → routed expert 数应该 = 64×8=512。Ling 用 256 是个折中。你用 64 偏保守（G=2-4 风格的"传统粗粒度"）。

**这是 16B 段位最大的"Ling 派 vs DeepSeekMoE 派"分歧**：
- **DeepSeekMoE-16B / V2-Lite / Moonlight**：64 routed + 1-2 shared
- **Ling-mini-2.0 / GLM-4.5-Air (128)**：256 / 128 routed + 1 shared

→ 你 16B 选 64 是跟"DeepSeekMoE 派"，跟 Ling 派的"新主流"有距离。**Wind tunnel 应该至少做一次 N_routed=64 vs 128 的对照**，不要默认就 64。

## 与其他笔记交叉引用

- [[08_ling_2]] —— Ling 2.0 论文笔记前期版本（如果存在）
- [[18_params_vs_flops]] —— Apple sparsity scaling law，与 Ling 的 EL 互补
- [[17_finegrained_scaling]] —— Krajewski granularity law，给 G_opt
- [[03_auxloss_free]] —— ALF 原理 (V3) + Ling zero-mean 修正
- [[04_deepseek_v3]] —— V3 是 Ling 系的 ancestor
- [[35_glm45]] —— GLM-4.5 反 WSD 信号 vs Ling 的 WSM 强证据
- [[36_longcat]] —— LongCat 的 hidden z-loss 在 Ling 1T FP8 里没明确用，但 FP8 训练稳定性两家都遇到
- [[22_FINAL_16B_design]] —— 应根据 Ling 1T 的新证据更新（尤其 expert count 和 scheduler 决策）
- [[38_100b_to_200b_gap]]（待写）—— Ling 直接跳过 200B 段到 1T 段，证据点
