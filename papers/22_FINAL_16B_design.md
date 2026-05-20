# 小型 MoE 模型设计建议（~16B 总参 / ~10T+ tokens 量级）

> 基于 27 篇一手论文 + 3 篇专题决策备忘综合得出（最新增量：35_glm45 / 36_longcat / 37_ling1t / 38_100b_to_200b_gap — 100B-1T 段位完整覆盖）。**16B 总参 / 16T tokens 是大致量级**，下面会给出可调整的目标区间与三种取向（reasoning / balanced / memorization）的具体推荐。每个决策都至少有 2 篇论文的实证依据；引用具体到论文章节或公式编号。

---

## 1. 一句话答案

> **总参 14–20B / 激活 1.5–3B / 27±3 层 / hidden 2048–2304 / N=64–128 routed + 1 shared / top-K=8 / d_expert ≈ hidden×0.68 / 首 1 层 dense / GQA 16Q-4KV / Aux-Loss-Free balancing (Ling 零均值变种) + α=1e-4 seq-aux / MTP D=1 / QK-Norm + Router z-loss / WSM scheduler / BF16 + AdamW(eps=1e-8) / 128K BBPE / Dropless token-choice / 10–18T tokens 课程化训练（pretrain → mid-training → long-context）。**

骨架 = **DeepSeek-V2-Lite + DeepSeek-V3 路由/balance + OLMoE 稳定性套件 + Ling-2.0 零均值 bias + Yokota top-K 调整**。

> "16B" 是甜区中位数 — 单卡 H100 / L40S 部署得下、训练 80GB GPU 也撑得住；过 22B 推理 latency 与 KV cache 都得分卡。"16T tokens" 是 over-train 上限 — 真要训 10–12T 也够；多出来的部分应该走 mid-training / reasoning pre-activation 而不是继续 raw pretrain。

---

## 2. 推荐 spec — Profile B（Balanced，默认）

> 这是直接可拿去 spec 评审的具体配置；下面 §3 给出 R/M 两个变体，§4-7 是论证、参数估算、训练计划、对比表。

| 维度 | 推荐值 | 主要依据 |
|---|---|---|
| **总参数** | **15.7 B** | V2-Lite 同体量；接近 16B 整数 |
| **激活参数 / token** | **2.4 B**（≈ 15% 激活，sparsity = 1/6.5） | V2-Lite=2.4B，DeepSeekMoE-16B=2.8B 中位 |
| **N_routed / Top-K / N_shared** | **64 / 8 / 1** | V2-Lite (64/6/2) 现代化：K 抬到 8（Yokota），shared 降到 1（V3/K2 主流） |
| **d_expert** | **1408** | V2-Lite |
| **Layers** | **27** | V2-Lite (27), DeepSeekMoE-16B (28) |
| **Hidden** | **2048** | V2-Lite, OLMoE, Ling-mini 共识 |
| **Dense 前缀** | **第 0 层 dense FFN，FFN intermediate=10944** | V2-Lite §B（DeepSeekMoE-16B 同款） |
| **Attention** | **GQA, 16 Q-heads / 4 KV-heads** | hidden=2048, head_dim=128 自然 16Q |
| **Activation** | **SwiGLU** | 全行业 |
| **Norm** | **RMSNorm pre-norm + QK-Norm**（Q/K 后各加一个 RMSNorm） | Qwen3 / Ling / OLMoE 共识；K2 用 MuonClip (post-update QK-Clip) 是另一条稳定路线 |
| **位置编码** | **RoPE, base=1e6** (后期 YaRN 扩 128K) | Qwen3-30B-A3B 同款 (1e6)；Ling-mini 用 6e5、K2 用 5e4；本 spec 选 1e6 兼顾 128K 外推 |
| **Tokenizer** | **128 K BBPE** | V3=100K, K2=160K, Ling=156K 之间 |
| **Routing gate** | **Sigmoid + Top-K + 归一化, `routed_scaling_factor=2.5`** | V3 Eq.13-16 + Ling 2.0 + dots1（三独立验证；24_dots1） |
| **Load balancing** | **ALF（bias `b_i`, γ=0.001）+ Ling 零均值更新 + α=1e-4 sequence-aux** | V3 + ALF (2408.15664) + Ling 2.0；零均值修正进 wind tunnel A2，不必默认（dots1 V3 原版在 11.2T 也稳） |
| **Node-limited routing** | **不用**（EP=8 单节点；详见 25_node_limited_routing） | V3 唯一案例，dots1/Ling/Qwen3 均不用 |
| **Gating precision** | **FP32**（即使主体 BF16/FP8） | dots1 §2 + OLMoE §4.1.7 |
| **Router z-loss** | **β=0.001** | OLMoE §4.1.7 |
| **MTP** | **D=1, V3 causal chain, λ schedule 0.3 → 0.1** | V3 §4 |
| **Init** | **Truncated normal ±0.06** | OLMoE §4.2.2 |
| **Optimizer** | **AdamW (β1=0.9, β2=0.95, wd=0.1 含 embedding, eps=1e-8)** | OLMoE §4.2.4/§4.2.6 |
| **LR schedule** | **WSM**（首选）或 WSD（fallback） | Ling 2.0 §2.4（+1~2 分 vs WSD） |
| **Peak LR** | **3.0–3.4 × 10⁻⁴** | Ling-mini-2.0 (3.36e-4) |
| **Global batch** | **~4 M tokens / step** | Ling-mini batch=4400 × seq 4096 ≈ 18M（按 GPU 数缩放） |
| **Sequence length** | **4K pretrain → 32K mid-training → 128K (YaRN)** | V3/K2/Ling 课程 |
| **Precision** | **BF16** master；router/norm/embed 始终 FP32；FP8 仅在 ops 成熟时启用 | 16B 规模 FP8 ROI 远不如 200B+ |
| **训练 tokens** | **12–14T pretrain + 1.5T mid-training + 0.5T reasoning pre-activation = 14–16T** | Ling 2.0 课程 + K2 退火 |

### 参数核算（两套 active 口径都报，避免评审时撕扯）

| 模块 | 计算 | 总参 (M) | Active 严格<br>(M) | Active V3 口径<br>(M) |
|---|---|---|---|---|
| Token embedding (untied) | 128K × 2048 | 262 | – | 262 |
| LM head (untied) | 2048 × 128K | 262 | – | 262 |
| Dense layer 0 attn (GQA 16/4) | 4 × 2048² | 10.5 | 10.5 | 10.5 |
| Dense layer 0 SwiGLU FFN (10944) | 3 × 2048 × 10944 | 67.2 | 67.2 | 67.2 |
| MoE attn × 26 layers | 26 × 10.5 | 273 | 273 | 273 |
| Shared expert × 26 layers | 26 × (3 × 2048 × 1408) | 225 | 225 | 225 |
| Routed experts (all 64) × 26 layers | 26 × 64 × 8.65 | 14,394 | – | – |
| Selected routed (K=8) × 26 | 26 × 8 × 8.65 | – | 1,799 | 1,799 |
| **Base subtotal** | | **15,494** | **2,375** | **2,899** |
| MTP module (**dense, 1 transformer block**, 训练辅助，可选) | attn + dense FFN 10944 + 投影 M_k | **~82** | – (推理可丢) | – |
| **含 MTP 训练时** | | **15,576** | **2,375** | **2,899** |

→ **Base total ≈ 15.5 B / 含 MTP ≈ 15.6 B**
→ **Active 严格口径 ≈ 2.4 B**（仅 attn + 选中的 expert FFN + norm，不含 embedding/head）
→ **Active V3/Ling 口径 ≈ 2.9 B**（含 untied embedding + LM head；对齐 V3 / V2-Lite / Ling-mini 官方公告 active）

> 两个数字**都是对的**，看你和谁对比。V2-Lite 官方 2.4B 用严格口径；V3 37B / Ling-mini 1.4B 用 V3 口径（含 head）。**评审表里建议两行都列**。

**MTP 说明**：D=1 的 MTP module = 1 个 dense transformer block（attention + 一份 dense FFN + 投影矩阵 M_k），约 82M 参数，**训练辅助、不计入 base deploy total**。推理时丢弃即可；若做 self-speculative decoding 再额外加载。

---

## 3. 三种 profile（按产品取向挑一个）

| | **Profile R**（Reasoning-leaning） | **Profile B**（Balanced，默认） | **Profile M**（Memorization / Serving） |
|---|---|---|---|
| **Total** | **~19.5 B** | **~15.5 B** (base) / 15.6 B (含 MTP) | **~16.1 B** |
| **Active (严格)** | ~3.0 B | ~2.4 B | ~0.8 B |
| **Active (V3 口径)** | **~3.3 B** | **~2.9 B** | **~1.4 B** |
| Sparsity (严格 act / total) | 1/6.5 | 1/6.7 | 1/20 |
| N_routed | 64 | 64 | 256 |
| Top-K | 8 | 8 | 8 |
| N_shared | 1 | 1 | 1 |
| d_expert | 1792 | 1408 | 512 |
| Layers | 24 | 27 | 20 |
| Hidden | 2304 | 2048 | 2048 |
| 锚点 | DeepSeekMoE-16B 路线 | V2-Lite 现代化 | Ling-mini-2.0 |

> R 实际是 **~20B total**（用户允许 size 漂移，不再硬卡 16B）；M 是严格 16B（Ling-mini 同款）。

**怎么选**：
- **R（reasoning-leaning，作者推荐）**：产品偏 reasoning / agentic / coding / math。Yokota 2025 给方向性证据 (reasoning 偏好更大 active + 更大 top-K + 不要过稀疏)，**3B active 是沿这个方向的工程取点，不是直接证明的最优**。代价：实际 total 漂到 ~20B、推理 FLOPs 大、单卡 H100 还能放下但 latency 略差。
- **B（默认 / 通用）**：通用聊天 + 中等 reasoning，单卡 H100 / L40S deploy。V2-Lite 已经在 5.7T tokens 下被 DeepSeek 与社区充分验证。
- **M（serving / memo）**：偏 long-tail 知识、多语言、对 serving QPS 敏感。Ling-mini-2.0 7× efficiency leverage 在 20T 下被实测；代价是 N=256 工程复杂度（Megablocks kernel、EP ≥ 16）。

**作者倾向**：默认 **B 稍微偏 reasoning 一些** —— 即接受 active 实际是 2.9B (V3 口径) 而不是强行压成 2.4B。如果硬件预算允许、产品偏 reasoning，直接选 R。**三套一律先在 §8 的 wind tunnel A2 上对比再固化。**

---

## 4. 关键决策的论证（精简版）

### 4.1 sparsity 选 1/6–1/8（不是 1/16 也不是 1/4）

- **Abnar 2025** (IsoFLOP, 2501.12370)：在 sparsity {0.75, 0.9, 0.95, 0.98} 网格上单调下降，S=0.85–0.94 远未到拐点；S → 1 是渐进最优但条件是 total params 也跟着无限放大。**16B-class 受总参约束**，过稀疏反而单个 expert 容量太小学不到东西。
- **Krajewski 2024** (2402.07871) Table 2：~2B active 规模 G_opt = 8–16。我们 64×top-8 给 G ≈ 8（路由到 8 个 1408-dim 小 expert ≈ 替代 1 个 11264-dim 大 FFN）。
- **Yokota 2025** (2508.18672)：**reasoning 类任务上 active 太小 + 过稀疏 + 小 top-K 一致弱**（inverted-U 形）。方向上支持 "适度密 + 大 top-K"；**注意不是直接证明 active ≥ 3B 是 peak**——Profile R 取 3B 是沿这个方向的工程取点。
- **实证锚点**：V2-Lite 1/6.5、DeepSeekMoE-16B 1/5.7、Ling-mini 1/11、OLMoE 1/5.3 — 1/6 到 1/11 是已经被多个团队跑通的合理区间。

### 4.2 N=64 / d_expert=1408（Profile B）

- **OLMoE §4.1.2** (2409.02060)：N=8 → 16 → 32 → 64 在 HellaSwag/MMLU 单调提升；32→64 已经在边际递减。
- **V2-Lite + DeepSeekMoE-16B** 实证：64 在 16B 规模有 5.7T–2T tokens 的训练验证。
- **不上 N=256** 的工程理由：N=256 需要 Megablocks 高质量 kernel + EP=16 集群；在 small EP (EP ≤ 8) 下负载均衡更难。Profile M 用 256，建议团队 infra 成熟后采纳。
- **不下 N=16**（Mixtral 风格）：Krajewski 直接证明 G=1（专家与 FFN 同维）几乎在任何 budget 下都不是最优。

### 4.3 top-K = 8（不是 2，不是 1）

- **Yokota 2025** Fig.5/7：**reasoning 任务上 K=8 显著优于 K=2/4，差距 5–15%**，且这点上 RL/TTC 不弥补。
- **OLMoE 64×K=8** 是同尺寸的稳健默认。
- **Mixtral K=2** 与 **Hunyuan K=1** 是 2024 早期路线，被新数据证明 reasoning 偏弱。

### 4.4 1 个 shared expert（不是 0，不是 2）

- **V3 / K2 / Hunyuan / Ling 全部用 1**：当 N≥64 且 K=8 时，1 shared 是低成本"通用 fallback"，对 16T over-trained 场景帮助大。
- **OLMoE §4.1.3 / Qwen3** 用 0 shared：在严格的 same active-param 约束下 1 shared 略差。但他们配套有 router z-loss + global-batch aux 等多重稳定性技术。
- **V2-Lite 用 2 shared**：偏保守，组合多样性损失更大；新模型已经普遍降到 1。
- 净判断：**1 shared 是当前最合理的风险对冲**。

### 4.5 GQA 而不是 MLA

- MLA 节省 KV cache（V3 比 GQA 8 KV-head 省 ~4×），主要价值在 100B+ 模型 + 32K+ 长上下文 serving。
- 16B 单卡 H100 部署：27 层 × head_dim 128 × 4 KV × 128K context × bf16 ≈ 1.7 GB，**KV cache 不是瓶颈**。
- MLA 工程复杂度：decoupled RoPE（V2 Eq.14-19）+ 与 FlashAttention 后端不兼容 + 需要 fused kernel。
- 16B 量级 GQA + base=1e6 RoPE + YaRN 已被 Qwen3-30B-A3B 验证 (128K context)。

### 4.6 路由：sigmoid + ALF + Ling 零均值 + α=1e-4 seq-aux

- **Sigmoid 而非 softmax**（V3 §4 Eq.13-16）：不强制 expert 间竞争，多 expert 共选时更稳。**dots1 (24_dots1) + Ling 2.0 (08_ling_2) + V3 (04_deepseek_v3) 三方独立采用**，是当前 MoE routing 的事实标准。`routed_scaling_factor = 2.5` 是三家共享的具体数值（Ling 论文称 "stabilize gate output RMS"）。
- **ALF**（2408.15664）：偏置项 `b_i` 不参与 gating 权重、不进入梯度路径，纯按 batch 后的负载差异更新。1B/3B 验证：perplexity 优于 aux-loss 0.05 个点，MaxVio 降 18×。671B/14.8T 大规模验证（V3）。**dots1 142B/11.2T** 也独立验证 V3 原版无修正可稳定收敛、无 spike、不丢 token。
- **Ling 零均值改良**（Ling 2.0）：`b_i ← b_i + u·(sign(e_i) − mean(sign(e)))` — 防止 bias 整体漂移，γ 在 context extension 后从 0.001 → 0.0001。**注意**：dots1 在 11.2T tokens 上**不用零均值修正**也跑通；该项的边际收益**未被任何论文直接消融**。建议**进 wind tunnel A2 验证后再固化**，不必默认开启。
- **α=1e-4 sequence-aux**（V3 Eq.17）：单序列粒度的极小补充 loss，防止单条长序列内部极端偏斜。
- **FP32 gating**（dots1 §2 明确强调）：即使主体 BF16/FP8，gating layer 单独跑 FP32 保证 routing 数值稳定。
- **抛弃 V2 三件套**（expert-balance α=0.003 + device-balance α=0.05 + comm-balance α=0.02）：V3 已经证明 ALF 显著更优。
- **不用 node-limited routing**（详见 25_node_limited_routing）：EP=8 单节点 setup 下 NLR 是 no-op；公开模型里**只有 V3 用了**，dots1/Ling/Qwen3/K2 全不用。

### 4.7 MTP D=1（dense 模块，训练辅助，不绑死进 base spec）

- **V3 §4 验证**：D=1 + causal chain，λ schedule 0.3 → 0.1，下游平均 +0.5-1pt，speculative decoding 1.8× 加速。
- **Gloeckle 2404.19737 Fig.3** 反例：D=4 parallel heads 在 ≤1.3B 模型上 **反而拖累**；6.7B+ 才稳定胜出。本设计 active 落在 2.4-2.9B 边界，**D=1 + causal 更稳**。
- **关键定性**：MTP module = **1 个 dense transformer block**（attention + 一份 dense FFN intermediate=10944 + 投影矩阵 M_k）≈ 82M 参数。**不要做 MoE**（head 每 token 只跑一次，sparsity 没意义）。
- **部署口径**：**MTP 是训练辅助，不计入 base deploy total**。推理时默认丢弃 (主 LM head 已足够)；要做 self-speculative decoding 再额外加载。Wind tunnel A2 没验证它在 2.4B 上正向之前，**MTP 应该作为 optional add-on 进入 spec，而不是默认开启**。

### 4.8 稳定性套件：QK-Norm + Router z-loss + Embedding wd + Truncated init

- **QK-Norm**（Qwen3 / Ling / OLMoE）：Q、K projection 后各加一个 RMSNorm，限制 attention logits 范围。OLMoE §4.2.5：减 spike 几乎免费。**K2 走另一条路**：MuonClip = Muon + QK-Clip（**post-update** weight rescale，不在 forward 加 norm）；二选一，本 spec 在 AdamW 体系下用 QK-Norm 更简洁。
- **Router z-loss β=0.001**（OLMoE §4.1.7）：约束 router logits 不爆炸；FP8 场景必备。
- **Embedding weight decay**（OLMoE §4.2.4）：GPT-2 时代把 embedding 排除在 wd 之外是历史包袱；纳入 wd 显著降 spike。
- **Truncated normal init ±0.06**（OLMoE §4.2.2）：比 Normal 更稳。
- **AdamW eps=1e-8**（OLMoE §4.2.6）：比默认 1e-5 更稳。

### 4.9 WSM > WSD

- **Ling 2.0 §2.4**：Warmup-Stable-Merge — 全程 constant LR，最后用 N=32 个 checkpoint 加权平均 ≈ 隐式 cosine decay。
- 实测 vs WSD 平均 leaderboard +1~2 分；SFT 5 epoch 后优势仍在。
- **优势**：不用预先决定何时开始 decay；预算可浮动；末段 anneal 失败可重做。
- **fallback**：K2 风格 WSD（warmup → constant → cosine decay → annealing）也成熟可用。

### 4.10 首 1 层 dense（不是 0，不是 3）

- DeepSeekMoE-16B (1), V2-Lite (1), Kimi K2 (1), Ling-mini-2.0 (1) 共识。
- V3 用 3 层是因为模型深 61 层；Ling-1T 用 4 层是 80 层。**16B / 27 层用 1 层** 是规模匹配。
- 理论解释（Sakana 2024 "Layers as Painters"）：浅层负责 embedding-to-canvas 转换，token 分布异常，MoE router 难平衡。
- 反例 OLMoE（无 dense 前缀，16 层全 MoE）配了 router z-loss + QK-Norm 等多重稳定性才稳；保留 dense 前缀更保险。

---

## 5. 16T tokens 怎么用 — 课程化训练

**核心判断**：16T tokens 在 16B 模型上 TPP_total=1000、TPP_active=6667，是**严重 over-training**。但这不代表要塞 16T tokens 进同一份 pretrain mix。借鉴 Ling 2.0 + K2 + V3 的三段式：

| 阶段 | Tokens | Context | 内容主旨 | LR 策略 |
|---|---|---|---|---|
| **S1 General pretrain** | 8T | 4K | DCLM 风格 web 60% + code 18% + math 8% + STEM 6% + multilingual 5% + books 3% | WSM constant 3e-4 |
| **S2 Reasoning ramp** | 5T | 4K | Reasoning 浓度上升到 30%（math/code/STEM 高质量子集 + 合成）；通用 web 降到 50% | continue WSM constant |
| **S3 Mid-training (32K)** | 1.5T | 32K | Long-doc + 20% 长 chunk + 10% CoT；YaRN preheat | continue WSM constant |
| **S4 Reasoning pre-activation** | 0.5T | 32K + YaRN→128K | 浓度 CoT 数据 30% + long-context retrieval 20% | WSM 收尾：merge 末段 32 个 ckpt |
| **(可选) annealing** | 100B | 32K | 高质量子集 + 极低 LR (~7e-6) | 仅当不用 WSM 时 |
| **合计** | **15.1T**（留 1T 缓冲） | | | |

**关键判断**：
- **不要训 16T 纯 pretrain**。Yokota 已证明长 over-train 在 reasoning 上损失。
- **mid-training 上 32K 的时机**：约 ⅔ pretrain 之后（13–14T 时）—— 与 Ling 2.0 (20T pretrain + 750B mid) 比例相当。
- **CoT pre-activation 的关键收益**：Ling-mini base 上 AIME25 从 2.08% → 43.75%，MATH 从 61.96% → 82.52%。
- **Rephrase 策略**：Math + STEM 用 LLM 做 10× rephrase 后单次训，比 raw repeat 10 epoch 好（K2 实测 SimpleQA 28.94% vs 23.76%）。
- **多语种**：~30 语言占 ~5%，从一开始就混入，不要末段再加。

---

## 6. 与现有同档模型对比

| 模型 | Total | Active | Sparsity | N_routed/K/shared | Hidden / Layers | Attn | Tokens | 出处 |
|---|---|---|---|---|---|---|---|---|
| Mixtral 8×7B | 47B | 13B | 1/3.6 | 8/2/0 | 4096/32 | GQA | 未公开 | 2401.04088 |
| **DeepSeek-V2-Lite** | **15.7B** | **2.4B** | **1/6.5** | **64/6/2** | **2048/27** | **MLA** | **5.7T** | 2405.04434 §B |
| DeepSeekMoE-16B | 16.4B | 2.8B | 1/5.7 | 64/6/2 | 2048/28 | MHA | 2T | 2401.06066 |
| OLMoE-1B-7B | 6.9B | 1.3B | 1/5.3 | 64/8/0 | 2048/16 | MHA | 5.1T | 2409.02060 |
| **Ling-mini-2.0** | **16B** | **1.4B** | **1/11** | **256/8/1** | **2048/20** | **GQA** | **20T+** | 2510.22115 |
| **dots.llm1** | **142B** | **14B** | **1/10** | **128/6/2** | **4096/62** | **MHA + QK-Norm** | **11.2T** | 2506.05767 |
| Qwen3-30B-A3B | 30B | 3.3B | 1/9.2 | 128/8/0 | 2048/48 | GQA | 36T | 2505.09388 |
| Phi-3.5-MoE | 42B | 6.6B | 1/6.4 | 16/2/0 | – | – | 4.9T | – |
| **本设计 (Profile B)** | **~15.5B** | **2.4B (严格) / 2.9B (V3 口径)** | **1/6.7** | **64/8/1** | **2048/27** | **GQA** | **~15T** | — |

> Active 口径说明：表中 V2-Lite (2.4B)、DeepSeekMoE-16B (2.8B) 是论文公布数字，**口径以原文为准**（V2-Lite 严格不含 head，DeepSeekMoE 计入 head 与 embedding）；Qwen3 / Ling 用 V3 口径。横向比较时**优先用 V3 口径**（含 embed+head）。

**这套 spec 处在"V2-Lite 现代化"位置**：保留其 size 与基本骨架，把 6 年累积的工程进步（ALF、QK-Norm、Router z-loss、WSM、Ling 零均值 bias）打包注入；MTP / MuonClip / N=256 等较新但实证不够厚的项一律放进 pilot。

---

## 7. 训练系统 & 成本估算

**目标硬件**：128–256 × H100 (80GB)。

**并行配置**：
- PP = 4（27 层 ÷ 4 ≈ 7 层/stage）
- **EP = 8（单节点 NVLink domain，64 routed ÷ 8 = 8 expert/rank）— 不跨节点**
- DP = 剩余（ZeRO-1，**跨节点用 IB all-reduce，与 EP 通信解耦**）
- **不用 TP**（hidden=2048 太小，TP 通信不划算 — 与 Yuan-M32 选择一致）
- **不用 node-limited routing**：EP=8 不跨节点 → NLR 是 no-op。详见 25_node_limited_routing 决策备忘 §5.1
- **All-to-all overlap**：参考 dots1 (24_dots1) 的 1F1B + grouped GEMM 方案，比 V3 DualPipe 内存友好（PP ≤ 8 场景适用）

**FLOPs 估算**：
- 训练 FLOPs ≈ 6 × N_active × D = 6 × 2.4e9 × 15e12 = **2.16 × 10²³ FLOPs**（MoE routing/all-to-all overhead 加 10-20% 后约 2.4-2.6 × 10²³）
- 对比 V3 = 3.4 × 10²⁴ FLOPs：本 spec ≈ V3 的 **6-7%**

**GPU-hours 估算**（H100 BF16，30-40% MFU，即每 GPU 实际 ~300-400 TFLOPs/s）：
- 总 GPU-hours ≈ 2.4e23 / (350e12 × 3600) ≈ **190K H100-hours**（中位估计）
- 算上失败重启、调试、退火重做等的缓冲：**~250-400K H100-hours** 是稳健预算
- 256× H100 在 30-40% MFU 下 wall-clock ≈ **30-65 天**
- 租赁价（\$2/h 量级）：**~\$500K-\$800K**

**对比 V3**：2.788M H800-hours / \$5.6M；本 spec 约为其 **1/8-1/10**。FP8 启用后还能再省 ~25%（但 16B 规模 FP8 ROI 弱、风险高，默认 BF16）。

---

## 8. 上 16T 之前必做的 pilot — Ling Wind Tunnel 思路

> 直接训 \$1M 不做 pilot 是疯狂的。借鉴 Ling 2.0 §：用 ~35% 单次 ablation 的 compute 跑一条 scaling 曲线，把架构决策的不确定性压到最小。

**5 个 anchor 模型**（power-law 间隔）：

| Anchor | Total | Active | Tokens | 目的 |
|---|---|---|---|---|
| A0 | 200M | 30M | 5B | 烧机；整套架构 sanity check（router 收敛、loss 曲线、CUDA 错误） |
| A1 | 500M | 80M | 16B | scaling law 初步拟合（loss 拟合 α 系数） |
| A2 | 1B | 200M | 25B | 中等 anchor；**做关键消融** |
| A3 | 4B | 600M | 80B | 大 anchor；验证 scaling 外推 |
| A4 | 16B | 2.4B | 320B | 目标 spec 1/50 计算量；最终 sanity check |

**A2 上做的核心消融**（每个 ~50B tokens）：

1. **Profile R vs B vs M**：active 3B / 2.4B / 1.4B 三选一的 reasoning vs memorization 取向
2. **N_routed = 64 vs 128 vs 256**（核心粒度决策）
3. **0 shared vs 1 shared expert**（仍有争议）
4. **ALF γ=0.001 sign vs Ling 零均值 vs Qwen3 global-batch**
5. **MTP D=0 vs D=1 vs D=2**（验证 D=1 在 2.4B 仍正向）
6. **WSM 末段平均 N=16 vs 32 vs 64**

**总 wind tunnel 预算**：~35K H100-hours（约 7% 主训练 budget）。

---

## 9. 不推荐 / 避开的选项

| 选项 | 否决理由 |
|---|---|
| Mixtral 风格 N=8 / d_expert=14336 | Krajewski/OLMoE 一致结论：G=1 粗粒度几乎在任何 budget 下都不是最优 |
| MLA on 16B | 工程复杂度 vs 16B 单卡 KV cache 已不是瓶颈，ROI 低 |
| Sparse Upcycling | OLMoE/Skywork：训练预算 ≥ 1.2× dense → from-scratch 反超。15T 远超此阈值 |
| Hybrid Attention（Lightning / Mamba + Softmax） | MiniMax-01/Jamba 的收益主要在 1M+ context；16B 通常 ≤ 128K，hybrid 只增工程债 |
| MFA / AFD (Step-3) | 16B 单节点部署，attention/FFN 不需 disaggregation |
| Hunyuan 风格 N=16 / top-1 + shared | OLMoE/Krajewski 都证明 fine-grained 显著更好 |
| Yuan-M32 Attention Router | test loss 仅 +0.4%，工程开销不值 |
| MTP D=4 (Gloeckle parallel) | 2.4B active 在边界（Fig.3），D=4 风险大于 D=1 |
| Muon / MuonClip | K2 在 1T 验证过，但相对新；16B 团队 ops 不成熟时 AdamW 更稳。**可在 wind tunnel A2 试** |
| V2 三件套 balance loss | V3 已抛弃，ALF 显著更优 |
| 无 dense 前缀 | 仅 OLMoE 用，且需多重稳定性补偿；保留 1 层 dense 更保险 |
| **Hyper-Connections (HC) / mHC**（27_mhc） | mHC 是 HC 的稳定性补丁，**先要承担 4× 残差宽度才有 HC 的"收益"**；I/O 34d 是 vanilla 11×；Kimi head-to-head 上不赢 Block AttnRes；DeepSeek 内部 TileLang kernel 未开源。**16B 上不解决你没有的问题** |

---

## 10. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Yokota 警示 — over-train 损害 reasoning | 中 | 中 | mid-training 加 reasoning 数据；若产品偏 reasoning 切 Profile R |
| ALF 大规模偏置漂移 | 低 | 中 | Ling 零均值更新；监控 MaxVio_global |
| Single shared expert 在严格 OLMoE 约束下退化 | 低 | 低 | A2 anchor 做 0 vs 1 shared 消融 |
| MTP D=1 在 2.4B active 边界拖累 | 低 | 低 | A2 验证；随时可关闭 |
| WSM 末段平均不稳 | 低 | 低 | fallback WSD |
| Dropless TC + EP=8 负载不均 | 中 | 中 | Megablocks + ALF + α=1e-4 seq-aux 复合 |
| 数据合成 >25% 拖累质量 | 中 | 中 | 合成限 < 20%（Hunyuan-Large 21% 是上限参考） |
| FP8 启用后训练发散 | 高（若上） | 中 | 默认 BF16；FP8 仅在 wind tunnel 通过后启用 |

---

## 11. 哪些 "16" 是可调的

> 用户原话："16B 只是一个大概的 size，包括训练 tokens"。整理可调维度：

| 参数 | 推荐 (Profile B) | 调整范围 | 调整后果 |
|---|---|---|---|
| Total params (base) | ~15.5 B | 12–22 B | <14B 偏 reasoning（密度更高）；>20B 需要双卡 serving，过 22B 应该重新评估 spec |
| Active 严格 | ~2.4 B | 1.4–3.3 B | <1.5B 推理便宜但 reasoning 弱；>3B 推理 latency 翻倍 |
| Active V3 口径 | ~2.9 B | 1.9–3.8 B | 与外部模型对比时用此数 |
| Layers | 27 | **18-30** | 更深→reasoning 略强（GLM-4.5 路线）；更浅→训练吞吐高 + upscale headroom 大（详见 32_depth_width_tradeoff） |
| Hidden | 2048 | 2048 / 2304 / 2560 | 离开 2048 需重算 head / FFN 比例；**2304 是 Ling-mini-2.0 短宽路线候选**（Option B） |
| Tokens | ~15T | 8–18T | <10T 单纯 pretrain 不饱和；>18T 边际收益陡降 |
| Mid-training token | 1.5T | 0.5–2.5T | 影响 long-context 与 reasoning 的混合质量 |

**强默认（除非 pilot 反对，否则不动）**：
- **QK-Norm + Router z-loss + Embedding weight decay** 三件套（任一去掉显著增 spike，OLMoE 多次验证；dots1 也用 QK-Norm）
- **ALF (bias `b_i`, γ=0.001) + α=1e-4 sequence-aux**（V2 三件套已被 V3 在 14.8T 训练上抛弃，无回头路；dots1 在 11.2T 上独立验证 V3 原版稳定）
- **FP32 gating**（dots1 §2 强调；OLMoE 隐含；本仓库默认采纳）
- **WSM 或 WSD**（不要用 OneCycle / 单段 cosine；dots1 走 WSD + 双段 annealing 也成熟；**Ling-1T (37_ling1t §3.2.3) 给出 WSM > WSD +1~2pt 全 benchmark 的直接对照证据**；GLM-4.5 (35_glm45) 反对 WSD 但与 Muon 耦合，未单独证明 cosine 优）
- **首 1 层 dense + 后续全 MoE**（reverse 顺序 / 0 dense 都需要更多稳定性补偿）
- **不用 node-limited routing**（详见 25_node_limited_routing；EP=8 单节点拓扑下 NLR 无收益）

**强默认但 pilot 必测**（A2 上验证后再固化）：
- **Top-K = 8**（vs 4 / vs 16）：Yokota / OLMoE 都指向 8，但 4 可能在某些 active 区间够用
- **1 shared expert**（vs 0 / vs 2）：Qwen3 / OLMoE 路线无 shared 也成功；V3 / K2 / Ling / Hunyuan 都用 1 shared；**这是当前公开文献里最有分歧的一项**，必须 A2 自己跑
- **MTP D=1**（vs 0）：2.4-2.9B active 处在 Gloeckle 论文的"刚刚显效"边界，不验证不敢绑死
- **N=64 vs 128 vs 256**：粒度选择影响吞吐与稳定性，三档都跑一遍。**重要新证据**：Ling-mini-2.0 (37_ling1t) 在 16B 段用 **256 routed + 1 shared = 8.75% 激活率**，与你 Profile B 的 64 routed + 1 shared = 15% 激活率形成"Ling派 vs DeepSeekMoE派"分歧。Ling 1T 给出 EL(A,G,C) scaling law 公式，granularity 最优 8-12，**Ling 派路线在 1T 段位已经全胜**；16B 段位是否同样优 = wind tunnel A2 T2.2 必答问题
- **Block AttnRes (N=4 或 6)**（Kimi 2603.15031；详见 26_attention_residuals）：替换标准 PreNorm residual 为块级 softmax-over-depth 聚合；I/O 5.5d（vs vanilla 3d，vs mHC 34d），推理 latency 开销 < 2%；Kimi 48B/3B + 1.4T tokens 上 GPQA-Diamond +7.5、Math +3.6、HumanEval +3.1。**16B 27 层适配**：N=4 → 7 层/块，N=6 → 4-5 层/块。代码开源但 license 限商用；**A2 判定门槛：loss 改善 ≥ 0.005 + 下游 reasoning ≥ +1pt 才纳入**
- **零均值 bias 修正**（Ling 2.0 vs V3 原版 sign）：dots1 在 11.2T 上 V3 原版稳定，零均值的边际收益从未被任何论文消融。**Ling-1T (37_ling1t) 在 1T 段位仍坚持零均值**，给出最大规模的"持续采用"证据；GLM-4.5 (35_glm45) 用 V3 原版 + bias 后期 → 0 的方案
- **Hidden z-loss (LongCat 36_longcat Eq. 10, λ=1e-7)**：λ 极小但能压住 hidden state L2 norm 不爆炸（7 个 OOM 差距）。LongCat 在 560B BF16 训练上 must-have；OLMo 2024 上 1B 已验证；**强烈建议加进 16B**（成本几乎 0，防爆 OOM 价值高）。判定门槛：开/关 loss 差 ≤ 0.001 即可固化
- **Adam ε 选择**：默认 1e-8 vs DeepSeek 1e-20 vs **LongCat 1e-16**。LongCat (36_longcat) Figure 7 给出"ε 接近 gradient RMS 时 loss 立刻劣化"的实证，推荐 ε ≪ gradient RMS。16B wind tunnel B 必加 grid {1e-8, 1e-12, 1e-16}
- **Depth-Width Tradeoff（A2 T2.6，详见 32_depth_width_tradeoff）**：baseline 27L/2048 vs Option A 24L/2048 vs **Option B 20L/2304** ⭐ vs Option C 16L/2560；同 total/active 下 latency 减 11-41%、depth upscale headroom 1.5-2.0×；reasoning quality 风险 0-3pt；**Ling-mini-2.0 验证过 20L，是 Option B 的直接 anchor**；若产品计划 12 月内 upscale 到 30B+，Option B 是 strong default candidate

**软项（可以按硬件 / 产品需求调整）**：
- Total params 14-20B
- Active params 1.4-3.3B（V3 口径）
- Layers **18-30** / Hidden **2048-2560**（含短宽 Option B/C 路线）
- Pretrain tokens 8-18T（剩余用于 mid-training / reasoning pre-activation）
- Tokenizer 100K-156K BBPE

---

## 12. Upscale Roadmap（base → 32B → 50B+）

> 16B 是起点，不是终点。这里规划三条上扩路径，明确化"未来要扩到 30B+ 时不被 spec 卡住"。完整论证见 32_depth_width_tradeoff.md。

### 12.1 三条独立上扩路径

| 路径 | 方法 | 典型 budget | 16B → next 范例 |
|---|---|---|---|
| **① Depth Upscaling** | SOLAR-DUS (Upstage 2023-12) / LLaMA Pro / LESA | 3-5T tokens continued | 16B/20L → 26B/32L |
| **② Expert Upcycling** | Sparse Upcycling (Komatsuzaki 2023) | < 1× 原 dense budget | 26B/N=64 → 50B/N=128 |
| **③ Width Upscaling** | 加 hidden + 重 init | 罕见，不推荐 | – |

### 12.2 推荐路线（基于 Option B 假设）

**Stage 1（主训练）**：16.3B / 20L / hidden 2304 / N=64 / K=8 / 1 shared / d_expert=1792，训 12-15T tokens

**↓ SOLAR-DUS (m=4)**：复制 base → 各去除中间 4 层 → 拼接 → 16+16 = 32L

- ⚠ **MoE 上 SOLAR 无公开案例**，启动前先在 1B/200M-active 上做 SOLAR proof
- 必须 verify router 不因 layer 重组发散；新拼接层的 expert 共享 router 还是各自重训是 open 问题

**Stage 2（depth-upscaled, ~3 个月后）**：~26B total / ~3.9B active / 32L / 2304 / N=64

- Continued pretrain 3-5T tokens（recovery + 新能力）
- 算力 ~150-250K H100-hr

**↓ Sparse Upcycling**：每 expert 复制 2 份 + σ=0.01 噪声 + router 全新初始化（参考 19_sparse_upcycling.md）

**Stage 3（expert-upcycled, ~6 个月后）**：~50B total / ~3.9B active / 32L / 2304 / N=128

- Continued pretrain 1-2T tokens
- 算力 ~50-100K H100-hr

### 12.3 Upscale 友好性总结

每个 Profile B 候选的 upscale 适合度：

| 起点 | Depth upscale headroom | Expert upcycle 适配 | 综合 |
|---|---|---|---|
| Baseline (27L/2048) | 1.5×（27→40，接近行业上限 GLM-4.5 92L） | ✓ | 一般 |
| Option A (24L/2048) | 1.7× | ✓ | 良 |
| **Option B (20L/2304)** | **2.0× 安全区** | ✓ | **最优** |
| Option C (16L/2560) | 2.0× 翻倍 | ✓ | 良（但 base reasoning 风险高） |

### 12.4 决策

1. **现在不要切主 spec**，先 wind tunnel A2 T2.6（4 arms，~80 H100-hr）验证 4 个 layer/hidden 组合
2. 如 Option B 通过（loss 差 ≤ 0.003 + GSM8K/MMLU 差 ≤ 1pt）→ 22_FINAL §2 Profile B 改 Layers 27→20, Hidden 2048→2304, d_expert 1408→1792
3. Stage 2 / Stage 3 是 release 后的 roadmap，**不绑定首版 release**
4. **如 T2.6 不通过**：维持 baseline 27L/2048，upscale 仍可走 Stage 2 / Stage 3，只是 headroom 略小（1.5× vs 2.0×）

### 12.5 与其他决策的交叉

- Profile B vs R vs M 的取向决策（§3）仍优先于 depth/width；先选 profile，再选 depth/width
- MTP D=1 决策（§11，wind tunnel A2 T3.1）独立于 depth/width；都需要 A2 验证
- N=64 vs 128 vs 256 决策（§11，wind tunnel A2 T2.2）与 depth/width 独立；但 N=256 + 短宽（如 Option C）的组合工程复杂度叠加，建议不一起激进

---

## 13. 一页 summary（贴给评审用）

> **目标**：~16B 量级 MoE / Active ≈ 2.4B（严格）/ 2.9B（V3 口径）/ 单卡 H100 / L40S deploy / 通用 + 中等推理
>
> **架构**：DeepSeek-V2-Lite 骨架 + 现代化（ALF zero-mean / QK-Norm / WSM / Router z-loss）
>
> - 27 层 × hidden 2048 × GQA 16Q-4KV × head_dim 128
> - 首 1 层 dense FFN (intermediate 10944)
> - 26 层 MoE：N=64 routed + 1 shared，K=8，d_expert=1408
> - RoPE base 1e6 + YaRN 后续扩 128K
> - 128K BBPE tokenizer
> - **MTP D=1（dense, ~82M）作为 optional add-on，pilot 验证后再开启**
> - **MuonClip / N=256 / 0 shared 一律放进 pilot，不绑死进主 spec**
>
> **参数账两口径都报**：
> - Base total ≈ **15.5 B**（含 MTP 训练时 15.6 B）
> - Active ≈ **2.4 B**（严格，排除 embed/head）
> - Active ≈ **2.9 B**（V3/Ling 口径，含 untied embed+head）
>
> **训练**：~15T tokens 分四段（8T pretrain → 5T reasoning ramp → 1.5T mid-training 32K → 0.5T reasoning pre-activation YaRN→128K）
>
> - AdamW (β2=0.95, wd=0.1 含 embedding, eps=1e-8) + WSM scheduler + BF16
> - 训练 FLOPs ≈ 2.2-2.5 × 10²³
> - 256× H100 @ 30-40% MFU ≈ **30-65 天 wall-clock**，**~250-400K GPU-hours**，**~\$500K-\$800K**
>
> **必做 pre-flight**：Ling wind tunnel 风格 5-anchor pilot；A2 (1B/200M/25B) 上做 R/B/M、N=64 vs 128 vs 256、0 vs 1 shared、ALF 变种、MTP D=0 vs 1 vs 2 的消融，预算 ~35K H100-hours（主训 ~8-12%）。
>
> **三个 profile**：B 默认；R（~20B / 3B active）偏 reasoning；M（16B / 1.4B active，N=256）偏 memorization / serving。**作者倾向默认 B 稍偏 reasoning，即接受 active ≈ 2.9B（V3 口径），而不是强行压回 2.4B。**

---

## 14. Caveats（必读）

1. **本设计建立在公开论文之上**；DeepSeek/Moonshot/Ant/Tencent 等公司的内部消融数据外界看不到，少数微观决策（如 Qwen3 不用 shared expert 的精确数据）只能凭论文外推。
2. **16T tokens 是非常 over-trained 的设定**。所有 scaling law (Abnar/Krajewski/Ling) 都建立在 compute-optimal 框架上，**严重 over-train 区是否依然成立尚无独立 IsoFLOP 复现**。Kimi K2 (15.5T) 和 Qwen3 (36T) 提供经验证据。
3. **Yokota 2025 是新结论**（ICLR'26）：reasoning 偏稠密 + 大 top-K，若产品偏 reasoning 强烈建议切 Profile R。
4. **N=256 (Profile M)** 是激进可选：Ling 2.0 是 2025-10 才发布的最新结果，社区独立复现还不充分；7× efficiency leverage 是承诺但需 wind tunnel 验证。
5. **MLA 工程化估计 1-2 个月**：如果团队从未做过 MLA 且需要为 256K+ 长上下文做准备，可在 200B+ 升级时再切。
6. **post-training（SFT + RL）未覆盖**：那是独立的设计空间，可参考 V3 GRPO + Ling 2.0 (DFT + Evo-CoT + LPO) + K2 agentic data synthesis。
7. **本文档不是 spec 评审通过的最终版**：建议先做 wind tunnel pilot（§8），再固化某些数字。
