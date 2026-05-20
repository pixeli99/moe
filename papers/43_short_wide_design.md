# 16B 矮胖 MoE 设计：每一条 spec 都讲清"为什么这样"

> **目的**：把"DeepSeek 那套标准设计"和"矮胖（短宽）变种"放一起对照，每一条 spec 都给"X 因为 Y"。
> **目标读者**：自己学习，不是评审 spec。
> **前置阅读**：[[32_depth_width_tradeoff]]（已有 Option B/C 设计，本文是教学版）+ [[22_FINAL_16B_design]]（你的主 spec）+ [[36_longcat]]（极短宽 560B 实例）+ [[37_ling1t]]（短宽 16B Ling-mini 实例）

---

## 0. 类比：盖大楼

想象你要盖一栋办公楼，**总建筑面积 = 16,000 平米**（= 16B 参数）。两种盖法：

**Normal（DeepSeek 派 / V2-Lite 派）**：**27 层 × 每层 600 平米**
- 每层楼像一个 transformer block
- 高瘦 → 楼内电梯往返时间长（inference latency 高），但视野好（深度抽象 → reasoning 强）
- 例：V2-Lite, Moonlight, DeepSeekMoE-16B 全 27±1 层 / hidden 2048

**矮胖（LongCat 派 / OLMoE 派）**：**16 层 × 每层 1000 平米**
- 楼层少，每层更宽敞
- 电梯快（inference latency 低），但深层抽象空间不够
- 例：OLMoE 16L/2048 (7B), LongCat 28L/6144 (560B), Ling-mini-2.0 20L/2048 (16B)

**关键问题**：16B 总参算力固定，**深度和宽度怎么取舍 = 楼怎么切**。这就是 depth/width tradeoff。

---

## 1. 一句话答案

| | Normal | 矮胖 |
|---|---|---|
| 配置 | **27L / hidden 2048 / N=64+1 / K=8** | **16L / hidden 2560 / N=64+1 / K=8** |
| 直接 anchor | V2-Lite, Moonlight, DeepSeekMoE-16B | OLMoE (16L/2048 但 7B), Ling-mini-2.0 (20L/2048) |
| L/√H | **0.60**（中庸） | **0.31**（极短宽，接近 LongCat 0.36） |
| 推理 latency vs Normal | baseline | **-41% wall-clock** |
| Reasoning quality 风险 | none | moderate (论文 anchor < 12L 才明显劣化) |
| Upscale headroom | 27→40 (1.5×) | 16→32 (2.0×) ⭐ |

→ 两个 active params 都 ~2.4B（保持 FLOPs 不变），**总参也都 ~16B**（参数补在 hidden 上）。

---

## 2. 两套 spec 并排对比

| 维度 | Normal (DeepSeek 那套) | 矮胖 | 单位 |
|---|---|---|---|
| **总参数** | 15.7 B | 16.4 B | – |
| **激活参数 (V3 口径)** | 2.97 B | 3.15 B | – |
| **激活参数 (严格 routed K)** | 2.40 B | 2.49 B | – |
| Layers | **27** | **16** | – |
| First-K-Dense | 1 | 1 | layers |
| Hidden | **2048** | **2560** | – |
| Intermediate (dense FFN, layer 0) | 11264 | 13824 | – |
| Expert FFN intermediate | **1408** | **2048** | – |
| Attention | GQA | GQA | – |
| Q heads | **16** | **20** | – |
| KV heads | **4** | **5** | – |
| head_dim | 128 | 128 | – |
| L/√H | 0.60 | 0.31 | – |
| N_routed | 64 | 64 | – |
| N_shared | 1 | 1 | – |
| Top-K | 8 | 8 | – |
| Routing | sigmoid + ALF (Ling zero-mean) | sigmoid + ALF (Ling zero-mean) | – |
| routed_scaling_factor | 2.5 | 2.5 | – |
| MTP | D=1, λ=0.1 | D=1, λ=0.1 | – |
| QK-Norm | 是 | 是 | – |
| Partial RoPE | 前 64 dim | 前 64 dim | – |
| Optimizer | AdamW | AdamW | – |
| Scheduler | WSM | WSM | – |
| Vocab | 128K BBPE | 128K BBPE | – |
| Tokens | 12-15T pretrain + 1.5T mid | 12-15T pretrain + 1.5T mid | – |
| Inference latency (单 token) | baseline | **-41%** | wall-clock |
| 32K BF16 KV cache | 864 MB | **640 MB** | per sample |
| Embedding cost | 524 M | **655 M** (+25%) | – |
| **EP topology** | EP=8 single-node | EP=8 single-node | – |

→ **不同的只有 4 个数字**：Layers (27 vs 16), Hidden (2048 vs 2560), Expert FFN (1408 vs 2048), Q/KV heads (16/4 vs 20/5)。其他 18 个维度完全一致 — 矮胖只是**形状改变**，不是配方改变。

---

## 3. 12 个维度逐一讲"为什么"

### 3.1 总参数 = 16B
**两边都选 16B。因为**：
- 16B 是 2024-2026 "**单卡 H100 / L40S 推理舒适区**" 的甜区中位数
- 22B 以上 KV cache + 模型权重压不下 80GB 单卡 → 推理必须分卡 → 部署成本 jump
- 10B 以下 active 数 < 1.5B → MoE 优势变小（Apple [[18_params_vs_flops]] 给的 N_th 阈值）
- 16B 是 V2-Lite / Moonlight / Ling-mini-2.0 / DeepSeekMoE-16B 共同选择 → **有最多 anchor 可对照**

### 3.2 激活参数 ≈ 2.4B
**两边都 2.4B。因为**：
- **同 active 才能比较 quality**：active params 决定 FLOPs，FLOPs 决定 loss（scaling law）
- 如果矮胖版偷偷把 active 拉到 3.5B，深度劣势会被 FLOPs 补偿，**对比无意义**
- 2.4B active 是 V2-Lite anchor，[[28_open_source_moe_catalog]] 验证 16B 段位最常见
- → **整个对照实验的核心控制变量**

### 3.3 Layers: 27 vs 16
**Normal = 27，因为**：
- V2-Lite 27 / DeepSeekMoE-16B 28 / Moonlight 27 三家共识
- 16B 段位"标准深度"，没有任何团队选过 < 25L 的 16B production-grade MoE
- Reasoning task (GSM8K, GPQA) 在 27L 上已经够用

**矮胖 = 16，因为**：
- OLMoE 实证 16L 在 7B 段位 work（虽然总参 7B，但 active 1.3B 接近 16B 段下界）
- **不能 < 12L**：[[32_depth_width_tradeoff]] §3.3 经验阈值，< 12L 后 reasoning quality 急剧下降
- **不能 > 20L**：那就是 Option B (20L/2304) 的位置，跟 Ling-mini-2.0 一样，**不够"矮胖"**
- **16L 是"踩 OLMoE 实证下界 + 留 reasoning 安全余量"的折中**

### 3.4 Hidden: 2048 vs 2560
**Normal = 2048，因为**：
- 16B 段位标配 (V2-Lite / DeepSeekMoE / Moonlight / OLMoE 全 2048)
- head_dim 128 × 16 head = 2048 整除关系最自然
- vocab 128K × 2048 = 524M embedding 在总参中占比 3.3% 合适

**矮胖 = 2560，因为**：
- **总参守恒**：减层后必须加宽补回来，否则总参跌到 12B
- Layer 27→16 = -41% layer 数 → 每层要多承担 ~67% 参数
- hidden 2048 → 2560 = +25%（hidden 是平方项 in FFN 参数）→ 每层 FFN 参数 +56%
- 配合 expert FFN intermediate 1408→2048 (+45%) → 综合补足 16B 总参
- **不选 3072+**：hidden 3072 attention FLOPs +50%，[[32_depth_width_tradeoff]] §3.4 显示得不偿失

### 3.5 Expert FFN intermediate: 1408 vs 2048
**Normal = 1408，因为**：
- V2-Lite 经典配比：`d_expert / hidden = 1408/2048 = 0.69`
- DeepSeekMoE-16B 一样 1408
- Krajewski [[17_finegrained_scaling]] G_opt 公式给 16B-active 段位 G≈8 → d_expert ≈ FFN_dense / 8 ≈ 11264/8 ≈ 1408 验证

**矮胖 = 2048，因为**：
- hidden 2560 配 d_expert 2048 → 比例 0.8（略高于 0.69）
- 更大 expert FFN 让每个 expert 更"厚" → 补层数减少带来的 capacity 损失
- 2048 是个**整数友好的数** (整除 EP=8)
- 也跟 V3 / Ling 1T 的 expert FFN size **2048** 巧合一致 → 这个数字在 MoE 设计里有点"自然分水岭"

### 3.6 GQA Q heads: 16 vs 20
**Normal = 16，因为**：
- hidden 2048 / head_dim 128 = **16 head 自然整除**
- V2-Lite, Moonlight 共识

**矮胖 = 20，因为**：
- hidden 2560 / head_dim 128 = **20 head 自然整除**
- **head_dim 保持 128 不变**——这是 Ling 全系（mini/flash/1T）的"固定 head_dim 跨规模"做法
- 不选 32 head：那会让 head_dim = 80，破坏 RoPE 的 64 dim partial 切分

### 3.7 GQA KV heads: 4 vs 5
**Normal = 4，因为**：
- V2-Lite 16Q/4KV (KV-LoRA 等价 4 KV head)
- Q:KV 比 4 是 GQA 经典选择（Llama 系 / V2-Lite 都用 4×压缩）

**矮胖 = 5，因为**：
- 20 Q heads / 5 KV heads = Q:KV ratio 4，跟 Normal 一致
- 不选 4 KV：那会让 20/4=5 个 Q head per group，比 Normal 多 25% → KV cache 略大
- 不选 10 KV：太接近 MHA，KV cache 翻倍

### 3.8 N_routed = 64
**两边都 64。因为**：
- V2-Lite / Moonlight / DeepSeekMoE-16B 共识
- **不跟 Ling-mini-2.0 走 256**：Ling 派 256 在 16B 段位是激进选择（[[37_ling1t]] 详述），增加 EP 通信复杂度
- 你 22_FINAL Profile B 决策是 64，所以**两个版本都不动这个变量**
- 这是 Ling 派 vs DeepSeekMoE 派的分歧，但本文比较的是 depth/width，不是 expert count → 控制变量

### 3.9 Top-K = 8
**两边都 8。因为**：
- 2025+ 共识（[[28_open_source_moe_catalog]] §3.2 实证 K=8 占 24/66 模型，远超其他）
- Yokota / OLMoE / V3 / K2 / Ling / GLM / dots1 全选 8
- K=4 太少不够 capacity，K=16 通信成本翻倍

### 3.10 N_shared = 1
**两边都 1。因为**：
- V3 派从 V2 的 2 shared 砍到 1 shared
- Ling / GLM / K2 / dots1 一致选 1
- Shared expert 是"无条件激活的容量"，1 个足够提供 baseline 表达，多了浪费 FLOPs

### 3.11 sigmoid + ALF (Ling zero-mean)
**两边都用。因为**：
- 这是 V3 派路由标配，跟 16B 主 spec [[22_FINAL_16B_design]] §11 决策一致
- Ling zero-mean 修正比 V3 原版 sign 多一个安全保障（bias 不漂移）
- routed_scaling_factor = 2.5 是 V3/dots1/GLM/Ling 共有指纹
- 跟 depth/width 决策正交 → 两边一致

### 3.12 MTP / QK-Norm / Partial RoPE / Optimizer / Scheduler
**全部跟 Normal 一致。因为**：
- 这些都是"训练动态"层面决策，跟 depth/width 正交
- 矮胖不改这些 → 减少混淆变量
- 后续 wind tunnel 比对就单纯是 layer/hidden/expert FFN 三个变量

---

## 4. 矮胖派的真实证据

### 实证 1：OLMoE 16L/2048 (7B/1.3B-active)
- 16L 在 7B 段位完全 work，HellaSwag/MMLU 跟 28L Mixtral 持平
- **但 7B 段位 reasoning task 上限本来就低**，不能完全外推到 16B

### 实证 2：Ling-mini-2.0 20L/2048 (16B/1.4B-active)
- **唯一公开的 16B 段位短宽设计**
- L/√H = 0.44，比你 Profile B 的 0.60 矮宽
- benchmark 表现：MMLU 跟 V2-Lite 持平，**Math 略弱 (Ling-coder 1B 的 GSM8K 比 Qwen2.5-Math-1.5B 弱)**
- 这是"矮宽在 reasoning 上有代价"的最直接证据

### 实证 3：LongCat-Flash 28L/6144 (560B/27B-active)
- L/√H = 0.36，更极端的矮胖
- benchmark 跟 DeepSeek-V3.1 (61L/7168, L/√H=0.72) 平打 → **大规模下矮胖能 work**
- 但 LongCat 用了 ScMoE shortcut + 异构架构补救深度损失，**16B 没法 1:1 复制**

### 实证 4：你 16B 段位的"真空带"
**没有 16B/2.4B-active 段位的 16L/2560 production-grade 模型公开过**。这是探索区：
- 上界 (Ling-mini-2.0 20L)：work
- 下界 (OLMoE 16L 但 7B)：work
- 中间 (16B/16L)：**未验证**

→ **如果你做矮胖，你是 16B 段位的第一个 16L 公开数据点**。

---

## 5. 矮胖的真实代价（5 条 risk）

### Risk 1：Reasoning task 上限略低
**因果**：CoT 需要 hierarchical abstraction，每多一层就多一层"思考粒度"。16L 跟 27L 比少 11 层，相当于"思考链短了 40%"。

**实证**：GLM-4.5 选 92L 极深就是因为 reasoning task 明确受益于深度（[[35_glm45]] §2.1 原文）。

**量化预期**：GSM8K 可能差 1-3pt，GPQA 可能差 2-5pt（vs 27L baseline）。

**缓解**：
- 加强 reasoning data 比例（你 22_FINAL §5 已经规划）
- mid-training 阶段重点喂 CoT 数据
- post-training 用 long CoT RL 补回来

### Risk 2：Attention FLOPs 增加
**因果**：attention 算力 ∝ hidden²。hidden 2048→2560 = (2560/2048)² = **1.56× attention FLOPs per layer**。

**但**：层数减少 27→16 = -41% layers → 总 attention FLOPs = 1.56 × 0.59 = **0.92×** (比 Normal **更少**)

→ 从全局看 attention FLOPs 反而**省 8%**，所以这不是问题。Risk 1 才是真问题。

### Risk 3：Embedding 浪费
**因果**：embedding + LM head = 2 × vocab × hidden = 2 × 128K × 2560 = **655M** vs Normal 524M = **+25%**

→ 矮胖在 embedding 上多用 130M 参数（占总参 0.8%）。**不算大问题但要 aware**。

**缓解**：
- 用 tied embedding（input/output 共享）→ 省一半 = 327M
- 但 tied embedding 在 reasoning 模型上有争议（V3 / K2 / Ling 都用 untied）
- 你 Profile B 默认 untied → 矮胖也 untied，多用 130M 算 baseline 决策

### Risk 4：缺少 1B+ MoE 直接走 16L 的 anchor
**因果**：OLMoE 是 7B/1.3B-active 的 16L，但 1.3B-active 跟 2.4B-active 已经差近 2×。

→ 你做 wind tunnel A2 时**没有现成 paper 可以对照 loss 曲线**，需要自己跑。

**缓解**：Wind tunnel A2 加 Option C arm（16L/2560/2.4B-active），跟 baseline 27L/2048/2.4B-active 直接对照 ~3T tokens，看 loss 差。

### Risk 5：未来 SFT/RL 阶段可能掉性能
**因果**：base 模型 reasoning capacity 上限低 → 不管怎么 SFT/RL 都难追上深模型

**实证**：GLM-4.5-Air 46L vs GLM-4.5 92L 同口径配置，AIME 24 差距明显（Air 弱）。

→ **如果你计划做 reasoning RL 后训，矮胖风险大**。如果只做 base + 通用 SFT，矮胖问题小。

---

## 6. 矮胖的 5 个好处

### 好处 1：Inference latency -41%
**因果**：每生成 1 token 要顺序经过所有 layer。L=16 vs L=27 = **每 token 少 11 步顺序计算**。

**量化**：MoE 推理时每层 latency ~ 2-3ms（H100, batch=8）→ 矮胖每 token 快 **25-35ms**。

**应用价值**：
- 实时 chat：80 token 句子在矮胖上快 ~2 秒
- agent 工作流（多轮 tool call）：每轮快 30ms × 5 轮 = 150ms 累积

### 好处 2：Depth upscale headroom 大
**因果**：未来想从 16B 扩到 30B+，**Depth Upscaling (SOLAR-DUS) 是最成熟方法**。

- Normal 27L → 翻倍 54L：接近 GLM-4.5 92L 极端，**不安全**
- 矮胖 16L → 翻倍 32L：还在 V3 (61L)、Hunyuan-A13B (32L)、Mixtral 8x7B (32L) 的安全区
- **矮胖留出 2.0× upscale headroom**（[[32_depth_width_tradeoff]] §5.3 验证）

### 好处 3：PP bubble 减小
**因果**：1F1B pipeline bubble 时间 ∝ L × micro_batch_count⁻¹。L=16 比 L=27 bubble 占比少 ~40%。

→ 训练 MFU 可提升 1-3 个百分点（如果你做 PP 训练；16B 段位 EP=8 single-node 可能不开 PP，这点收益打折）。

### 好处 4：KV cache 略小（32K 上下文）
**因果**：KV cache 大小 = `2 × L × KV_head × head_dim × seq × batch`

| | L | KV head | KV per token | 32K ctx KV |
|---|---|---|---|---|
| Normal | 27 | 4 | 27×4×128×2 = 27.6 KB | 864 MB |
| **矮胖** | 16 | 5 | 16×5×128×2 = 20.5 KB | **640 MB (-26%)** |

→ 矮胖在长上下文上**少占 224 MB 显存**，可以多塞 30% batch。

### 好处 5：训练时 expert load balance 更容易
**因果**：层数少 → expert 数总量少 → ALF bias 收敛更快（[[03_auxloss_free]] §4 经验）
- Normal: 26 MoE layer × 64 expert = 1664 expert 实例
- 矮胖: 15 MoE layer × 64 expert = 960 expert 实例

→ 矮胖训前期 expert utilization 更均匀，少 1-2 个 capacity factor 调整。

---

## 7. 何时选哪个？决策表

| 你的场景 | 推荐 | 主要理由 |
|---|---|---|
| 主要做 reasoning/coding agent | **Normal (27L)** | 深度对 CoT 上限重要 |
| 做 chat assistant + 推理速度敏感 | **矮胖 (16L)** | latency -41% 直接体感 |
| 计划 12 个月内做 SOLAR-DUS 上扩到 30B+ | **矮胖 (16L)** | 2.0× upscale headroom |
| 这是公司**第一个** 16B MoE，求稳 | **Normal (27L)** | V2-Lite 有 4 个 anchor 对照 |
| 想做差异化、走 OLMoE/LongCat 路线 | **矮胖 (16L)** | 16B 段位无人公开做过，技术声量大 |
| 训练预算紧，wind tunnel A2 跑不全 | **Normal (27L)** | 风险已知最小 |
| 后续想做 RL post-training | **Normal (27L)** | base reasoning 上限重要 |
| 目标是低成本部署 + 大量并发 | **矮胖 (16L)** | latency + KV cache 双省 |

---

## 8. 折中路线：Option B (20L/2304)

**如果你完全决定不了**，Option B 是 [[32_depth_width_tradeoff]] 当前推荐：
- L=20 (减 26%)
- Hidden 2304 (加 12%)
- L/√H = 0.42（中等短宽）
- Ling-mini-2.0 anchor 已经 work
- Inference latency -26%
- Reasoning quality risk: **low** (不像 16L 那么激进)
- Upscale headroom: 1.6×

→ **不那么矮胖，但仍能拿到 60% 矮胖收益 + 30% 矮胖风险**。

我把这个看作"**矮胖路线的入门版**"。如果你想学矮胖但不敢一次到位，Option B 是 stepping stone。

---

## 9. 训练成本对照（25T tokens, H100×64）

| | Normal (27L) | 矮胖 (16L) | Option B (20L) |
|---|---|---|---|
| Active params | 2.4 B | 2.49 B | 2.49 B |
| FLOPs / token | 6 × 2.4B = 14.4 GFLOPs | 14.94 | 14.94 |
| Total FLOPs (25T) | 3.6e23 | 3.74e23 | 3.74e23 |
| MFU (estimated) | 45% | 47% (浅模型) | 46% |
| Wall clock days (64 H100) | ~32 | ~30 | ~31 |
| 成本估算 | ~$1.5M | ~$1.4M | ~$1.45M |

→ 训练成本**几乎一样**（< 7% 差），choice 不应基于训练成本，应基于推理 + reasoning quality tradeoff。

---

## 10. 我会选哪个？

**如果是我做你这个 16B 项目**，我会选 **Option B (20L/2304)**：
- 跟 Ling-mini-2.0 直接 anchor（已公开 working）
- 拿 65% 矮胖收益（latency -26%, upscale 1.6×）
- 风险可控（Reasoning quality risk low）
- Wind tunnel A2 还是要做对照，但失败回退路径短

**但如果你的产品场景就是 chat + latency 敏感 + 没有重度 reasoning**，我会建议 **矮胖 16L/2560**：
- 拿 100% 矮胖收益
- OLMoE 给了 7B 段位 anchor，风险可量化
- 真做出来是技术亮点（16B 段位首个 16L MoE）

**Normal 27L 适合"不想冒险" + "我就是想复刻 V2-Lite + Moonlight" 的保守路线**。这是你 22_FINAL Profile B 当前位置，**没毛病但没亮点**。

---

## 11. Settled vs Open

### Settled
- 27L / 2048 在 16B 段位是**已验证的安全选择**（4 个 anchor）
- 20L / 2304 (Option B) 是已验证的**入门矮胖**（Ling-mini-2.0 anchor）
- L < 12 在任何 active params 下 reasoning 都急剧劣化
- head_dim 跨规模固定 128 是 Ling 系最佳实践
- 矮胖 + 同 active params 时 attention FLOPs 反而**少 8%**（不是 risk）

### Open
- 16B / 16L / 2560 是否 work（**没人公开做过**）
- 矮胖在 reasoning RL 后训阶段是否仍然能追上深模型
- 矮胖 + 256 expert (Ling 派) 是否会引发 expert collapse（layer 少 + expert 多）
- 矮胖在 SFT 阶段是否需要不同的 LR 配置

### 已否决
- 12L 以下（任何 hidden）
- hidden > 3072（attention FLOPs 失控）
- 不补 hidden 直接砍层（总参跌到 12B，capacity 不够）

---

## 12. 与其他笔记交叉引用

- [[32_depth_width_tradeoff]] —— 同主题的原始 spec doc，本文是教学版重组
- [[22_FINAL_16B_design]] §11 → "Depth-Width Tradeoff" 决策
- [[28_open_source_moe_catalog]] §3 → 全市场 L/√H 分布表（验证 0.31 极端、0.42 中庸、0.60 标准的分层）
- [[36_longcat]] —— 极矮胖 560B (L/√H=0.36) 的实证
- [[37_ling1t]] —— Ling-mini-2.0 20L/2048 (L/√H=0.44) 的 16B 段位实证
- [[35_glm45]] —— 反向证据：92L 极深的 reasoning rationale
- [[42_100b_cookbook]] Step 7 → "depth vs width" 在 100B 段位的三派分布
- [[09_olmoe]] —— OLMoE 16L 在 7B 段位的成功 anchor
