# Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation (Ling 2.0 Technical Report)

- **arXiv**: 2510.22115 (v1 submitted Oct 24, 2025；v2 revised Nov 7, 2025)
- **机构**: Inclusion AI / Ant Group（蚂蚁集团 Ling Team）
- **发表时间**: 2025-10-24
- **作者(代表)**: Ling Team, Inclusion AI（141+ 作者）
- **代码 / 权重**: github.com/inclusionAI/Ling-V2 ; huggingface.co/collections/inclusionAI/ling-v2

## TL;DR

Ling 2.0 是蚂蚁 Inclusion AI 的 reasoning-oriented MoE 基础系列，覆盖 16B → 1T 三档（Ling-mini-2.0 / Ling-flash-2.0 / Ling-1T），统一 "**high-sparsity + fine-grained**" 架构，所有规模 **256 routed experts + top-8 + 1 shared**（**expert-slot fraction (8+1)/(256+1) ≈ 3.5%**；active/total params 分别为 **8.75% / 5.9% / 5.1%**，见 Table 1）。核心贡献：

1. **Ling Scaling Laws + Ling Wind Tunnel**：用 ~1000 个小规模实验拟合 hyperparameter & 架构 scaling law，用 500M–8B 的 5 个 anchor 模型把 1T 训练前的验证成本压到 < 1% full-run。
2. **7× efficiency leverage**：在该稀疏档（expert-slot ≈3.5%，active/total ≈5-9%）的 MoE = 7× dense 等效 compute，理论上由 scaling law 预测、实验上由三档模型验证。
3. **WSM (Warmup-Stable-Merge) scheduler**：用 checkpoint 平均替代 LR decay，比 WSD 平均 benchmark 提升 1–2 分，**无需事前定 decay 起点**。
4. **全 FP8 训练 1T 模型**：fine-grained tile-wise quantization，900B token 后 vs BF16 gap ≤ 0.25%。
5. **DFT + Evo-CoT + LPO**：post-training 三件套：Decoupled Fine-Tuning（同一模型两种 system prompt 模式）、Evolutionary Chain-of-Thought RL、Linguistic-unit (sentence-level) Policy Optimization。

## 关键架构配置（Table 1 原表）

| 项 | Ling-mini-2.0 | Ling-flash-2.0 | Ling-1T |
|---|---|---|---|
| Total params | 16B | 103B | 1,000B |
| **Activated params** | **1.4B** | **6.1B** | **51.0B** |
| Layers | 20 | 32 | 80 |
| Hidden size | 2,048 | 4,096 | 8,192 |
| FFN intermediate (dense layers) | 5,120 | 9,216 | 18,432 |
| **Expert intermediate size** | **512** | **1,024** | **2,048** |
| **N_experts (total)** | **256** | **256** | **256** |
| **Top-K** | **8** | **8** | **8** |
| **N_shared** | **1** | **1** | **1** |
| **Active/Total params** | **8.75%** (1.4/16) | **5.9%** (6.1/103) | **5.1%** (51/1000) |
| **Expert-slot fraction** `(K+N_sh)/(N_rt+N_sh)` | ≈ 3.5% | ≈ 3.5% | ≈ 3.5% |
| Num attention heads | 16 | 32 | 64 |
| Head dim | 128 | 128 | 128 |
| Num KV heads (GQA) | 8 / 16 / 32 (one of these per Section 2.1) | 同 | 同 |
| **Dense 前缀层 (`first_k_dense_replace`)** | **1** | **1** | **4** |
| Vocab size (BBPE) | 156K | 156K | 156K |
| **Attention type** | **GQA**（标准 Grouped-Query Attention） | 同 | 同 |
| **Partial RoPE** | RoPE 仅施加到 head 的前 64 维 | 同 | 同 |
| Normalization | RMSNorm + pre-norm, **QK-Norm** | 同 | 同 |
| Activation | SwiGLU | 同 | 同 |
| **MTP** | 1 层（loss weight=0.1）| 同 | 同 |
| Init std | 0.006 | 同 | 同 |
| Peak LR | 3.36×10⁻⁴ | 2.61×10⁻⁴ | 1.86×10⁻⁴ |
| Batch size | 4,400 | 8,352 | 18,144 |
| Tokens | 20T pretrain + 750B mid-training | 同 | 同 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1, grad_clip=1.0) | 同 | 同 |
| LR schedule | **WSM (Warmup-Stable-Merge)** | 同 | 同 |
| 精度 | **全 FP8 (1×128 act/grad tile, 128×128 weight)** | 同 | 同 |

## 核心方法 / 创新点

### 1. Ling Scaling Laws

**两条 scaling law**：

**(a) Hyperparameter scaling**：用 ~1000 次 WSD 调参实验（compute ≤ 3e20 FLOPs，64 expert × 4 active × 1 shared 固定架构），拟合 optimal LR / batch size 关于 compute budget C 的 power law。关键发现：
- MoE 比 dense 在更大 compute 下倾向**更大 batch + 更小 LR**
- 原因：MoE 稀疏梯度（每个 expert 只见一部分 token）需要大 batch 稳定

**(b) Architectural Efficiency**：基于 300+ 模型（最大 28B），用 efficiency leverage (EL) 作 metric：

```
EL(A, G, C) = Â^(α + γ·(log G)² + β·log G)
其中 α = a + d·log C
```
- `A` = 激活率（active params / total params）→ **主导因子**，sparser 越高 EL 越大，1/128 仍然成立
- `G` = expert 粒度（active expert 数）→ log-polynomial 调制，**最优区间 8–12**
- `C` = compute budget → amplification factor，越大 EL 越大

→ Ling 2.0 选 **256 routed + 8 active + 1 shared**（**expert-slot ≈3.5%**；按 active/total params 主口径 mini/flash/1T 分别为 **8.75% / 5.9% / 5.1%**），scaling law **预测 > 7× EL**。三档模型实测验证：

> "Both Ling-mini-2.0-base, Ling-flash-2.0-base, and Ling-1T-base achieve performance comparable or superior to other state-of-the-art open-source models of similar scale... using less than one-seventh of their non-embedding activated parameters, confirming the 7× efficiency leverage."

### 2. Ling Wind Tunnel

**目的**：用最小 compute 验证某项新技术（feature）能否 scaling 到 1T。

**方案**：5 个 anchor 模型（500M – 8B，size 按 power law 分布），每个模型用 scaling law 给的最优超参 + 最优 token 数训到 compute-optimal。一次完整 wind tunnel run 的 compute ≈ **35% 单次 Ling-mini-2.0 ablation**，但能给出跨 scale 的 loss 差异曲线，进而判断 feature 在 100× compute extrapolation 下的行为。

**预测精度**：训练 loss 误差 ≤ 0.01。

这是该论文最有价值的方法论贡献：让大规模 MoE 设计从 "猜一个 hyperparameter 试着跑" 变成 "拟合一条曲线 → 推 100×"。

### 3. Aux-Loss-Free Load Balancing (基于 DeepSeek-V3)

继承 DeepSeek-V3 的 aux-loss-free 方案，做了**关键修改**：bias 居中保持 zero-mean：

```
b_i ← b_i + u · (sign(e_i) − mean(sign(e)))
```

其中 `u` 是 update rate（pretrain 阶段 `u = 0.001`，context extension 后 `u = 0.0001`），`b_i` 是第 i 个 expert 的 bias，`e_i` 是其 violation error。

附加：
- **Router gate scaling factor = 2.5**，稳定 gate 输出的 RMS
- **Dropless routing**（不丢 token，保证性能）
- **Group routing** 提高训练效率（无性能损失）

这是当前公开的 aux-loss-free MoE 实现里最完整的公式之一。

### 4. Multi-Token Prediction (MTP)

**1 层 MTP head**，loss weight = 0.1。论文做了 wind tunnel 验证：MTP 在不同 scale 下都对 code/math 一致提升。在 Megatron 中做了 fine-grained PP partitioning 缓解 MTP 的 PP bubble。

### 5. Dense 前缀层

- **Ling-mini-2.0 / Ling-flash-2.0**: 首 1 层 dense
- **Ling-1T**: 首 4 层 dense

> "reduces the total parameter count while maintaining equivalent model performance and improving routing balance"

直觉：前面几层主要负责低级 token feature，MoE 路由收益小，干脆全部走 dense FFN。layer 数更深的模型需要更多 dense 层来稳定。Ling-1T 选 4 而非 1 是经验法则。

### 6. QK-Norm + Partial RoPE

- **QK-Norm**: 论文强调在 **FP8 低精度训练**下显著提升稳定性。Ling-1T 全 FP8 训练 + QK-Norm 是匹配组合。
- **Partial RoPE**: RoPE 仅施加到 attention head 的**前 64 维**，剩余维度无位置编码，提升长上下文外推能力。

### 7. WSM (Warmup-Stable-Merge) Scheduler

**核心：用 checkpoint averaging 模拟 LR decay**

理论：merging `[θ_n, θ_{n+1}, ..., θ_{n+k}]` 等价于对累积梯度做加权（公式 2），从而模拟任意 LR decay schedule（公式 3 的反推）。

实践：
- 线性 warmup 2000 步 → peak LR
- **Stable phase: constant LR 直到训练结束**
- "annealing" 通过最后 N=32 个 checkpoint 加权平均完成（不动 LR）

**WSM vs WSD**:
- WSD 需要预先决定何时开始 decay、decay 用多少 token（如 400B）
- WSM 无需，更灵活
- 平均 benchmark 提升 **+1 ~ +2 分**，SFT 5 epoch 后优势仍在

### 8. 全 FP8 训练

最大的开源全 FP8 训练实践：
- Activations / gradients: 1×128 tile
- Weights: 128×128 block
- **vs BF16: 900B tokens 后差 ≤ 0.25%**
- 显存节省 > 15%，吞吐提升 ~40%（heterogeneous PP scheduling 配合 MTP）

### 9. Post-Training 三件套

**(a) DFT (Decoupled Fine-Tuning)**：同一个模型用两个 system prompt 训：
- "detailed think off" → Instant Response
- "detailed think on" → In-Depth Reasoning（`<think>...</think><answer>...</answer>`）

→ 一个模型就有两种行为，类似 Qwen3 的 thinking/non-thinking 但**只在 SFT 阶段**用 prompt 切换，不需要 mode fusion 阶段。

**(b) Evo-CoT (Evolutionary Chain-of-Thought)**：在 Instant Response mode 上做 RL，逐渐进化出 CoT 行为。Reward 组成：
- `R_correctness` (+1/0)
- `R_length` (dynamic, 难题宽容长 reasoning，简单题鼓励短)
- `R_format` (`<think>` 出现 → -0.5)
- Task-specific (例如 frontend 的 Visual Augmented Reward)

**(c) LPO (Linguistic-unit Policy Optimization)**：以**句子**为单位做 importance sampling 和 clipping（vs token-level GRPO 或 sequence-level GSPO）：

```
r_{i,k}(θ) = exp( (1/|s_{i,k}|) · Σ_t log[π_θ(y_t)/π_old(y_t)] )
```

clip ε = 0.03。比 GRPO/GSPO 训练曲线更平滑，AIME 2025 上显著领先。

## 训练 & 系统细节

### 数据 (20T pretrain + 750B mid-training)

**Pretrain (4K context, 20T tokens)**:
| Substage | Tokens | General | Reasoning |
|---|---|---|---|
| S1 | 10T | 68% | 32% |
| S2 | 10T | 54% | 46% |

**Mid-training (32K context, 750B tokens)**:
| Substage | Tokens | General | Reasoning |
|---|---|---|---|
| Long-context extension | 150B | 54% | 46% (含 20% 32K 长文) |
| Reasoning Pre-Activation | 600B | 55% | 45% (混 CoT 数据) |

随后用 **YaRN** 扩到 128K。

**关键数据集**：
- **Ling Code Corpus**: GitHub 660 语言 + GHArchive event replay + 编程竞赛
- **Ling Math Corpus**: 数学概念图扩展 + LLM-Filter/Refiner
- **多语言**: 156K vocab，~2TB 多语数据，~30 语言，占总数据 4%
- **Long-context**: ~1.2T 高质量长文

### 训练超参 (Table 1)
- AdamW: β₁=0.9, β₂=0.95, wd=0.1, grad_clip=1.0
- Warmup: 2000 步
- Batch ramp: 前 ~500B tokens 从 3024 → peak
- aux-free bias update rate: 0.001 (pretrain) → 0.0001 (after context ext.)
- MTP loss weight: 0.1

### 基础设施
- **全 FP8 训练**（fine-grained quantization）
- **Heterogeneous PP scheduling**：用 interleaved 1F1B + partial recomputation 处理 MTP 和 First-K-Dense 引入的不规则 pipeline，吞吐 +40%
- **Megatron-based** framework
- 大规模异步 reward 计算系统（40K concurrent reward requests，>99.9% 成功率）

## 关键消融与结果

### 7× Efficiency Leverage 实测
- Ling-mini-2.0 (1.4B 激活) ≈ dense Qwen3-8B-base / Seed-OSS-36B-base
- 8B / 1.4B ≈ 5.7× / 36B / 1.4B ≈ 25×（综合算 ~7×）

### MoE Architecture Sweep
- 激活率 1/128 仍然有 EL 增益（sparser 一直更好）
- Optimal active experts 数: **8–12**（→ 选 8）
- Shared expert 排列等次要因素影响小

### WSM vs WSD
- WSM 平均 leaderboard +1~2 分
- SFT 5 epoch 后优势仍在

### CoT Pre-Activation
- AIME25 base: w/o CoT 2.08% → w/ CoT 43.75%（Ling-mini-2.0）
- MATH base: 61.96% → 82.52%

### LPO vs GRPO/GSPO
- 训练曲线更平滑，AIME 25 显著高于 GRPO/GSPO

## 对 16B MoE 设计的启示

**Ling-mini-2.0 (16B/1.4B) 本身就是 16B MoE 的范式参考**。具体可学：

1. **架构配置直接 portable**：
   - hidden=2048, layers=20, FFN=5120
   - N=256 routed + top-8 + 1 shared, expert_dim=512
   - 16 attn heads, head_dim=128, GQA (8 KV heads)
   - active/total ≈ 8.75%（expert-slot 口径 ≈3.5%）
   - **首 1 层 dense**（不是 K2 的 1 层 dense 但 N=384，也不是 V3 的 3 层）

2. **Ling Wind Tunnel 方法论**：在做 16B MoE 之前，先跑 5 个小 anchor（比如 500M / 1B / 2B / 4B / 8B），用 scaling law 拟合，能把架构决策 (dense 层数、shared expert 配比、expert 粒度) 验证成本压到 < 1% full-run。这是最有 leverage 的做法。

3. **Aux-loss-free + zero-mean bias update**：完整公式 `b_i ← b_i + u·(sign(e_i) − mean(sign(e)))` 可直接复制。`u` 在 pretrain 用 0.001，context ext 后用 0.0001。Router gate scaling 2.5 是关键稳定细节。

4. **QK-Norm + FP8 是最佳搭档**。如果 16B MoE 想用 FP8 训练，QK-Norm 是必须的稳定项；vs Kimi K2 的 QK-Clip 是另一条路。

5. **WSM scheduler 把决策简化**：constant LR + 训完做 top-N (e.g., N=32) checkpoint 平均，不用提前定 decay 起点。对短 timeline 项目特别合适。

6. **MTP 1 层 + loss weight 0.1** 是稳健配方。Megatron PP partitioning 是工程关键。

7. **Partial RoPE (前 64 维)** 是一个低成本的长上下文增强。

8. **20T tokens pretrain + 750B mid-training** 是 16B MoE 的合理总量级（对照 K2 的 15.5T pretrain + 460B 退火）。

9. **DFT 双 prompt 模式** 比 Qwen3 的 4-stage mode fusion 更简单，单 SFT 阶段同时给两种 system prompt 数据即可。

10. **Ling-1T 的训练 loss curve 与 wind tunnel 预测差 ≤ 0.01**，说明小规模实验对 16B MoE 设计的预测有效。

## Caveats / 局限

- **本版论文（2510.22115 v1, 2025-10-24）的 Table 1 中三档模型都标注 GQA**（16/32/64 heads, head_dim=128 固定）。**任务描述里提到的 "Ling-1T 从 GQA 切到 MLA" 在本 revision 中没有被论文确认**。该说法可能来自更早的预告或后续 v2 修订；当前公开技术报告统一使用 GQA。**[CAVEAT: GQA→MLA 切换在本论文 v1 中未见明确陈述]**。
- **Ling-1T 训练总 tokens** 论文给出 20T pretrain（4K）+ 150B context extension（32K）+ 600B reasoning pre-activation（32K）≈ 20.75T。
- **专家 grouping 细节**：论文提到 "group routing" 但未给精确分组数。
- **三档模型的 KV heads 数**论文文字描述为 "8, 16, or 32" 但 Table 1 仅列出 attention head 总数；具体对应关系需查 HF config。
- **完整的 FP8 quantization 数值细节**（per-channel scale 还是 per-tile、e4m3 vs e5m2）需要查 GitHub 代码。
- **256 expert / top-8 在小 EP 度下推理路由开销**：论文未与 N=64 或 N=128 在 16B 规模直接对比。
- **WSM 顶部平均 N=32 的 checkpoint 来源**：要在哪些步骤保存才"等价于一段 cosine decay"，论文给出理论公式但工程上的 ckpt 间距未详。
