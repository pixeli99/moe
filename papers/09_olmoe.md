# OLMoE: Open Mixture-of-Experts Language Models

- **arXiv**: 2409.02060
- **机构**: Allen Institute for AI (AI2)、Contextual AI、华盛顿大学、普林斯顿大学
- **发表时间**: 2024 年 9 月
- **作者(代表)**: Niklas Muennighoff、Luca Soldaini、Dirk Groeneveld、Kyle Lo、Jacob Morrison、Sewon Min、Pang Wei Koh、Hannaneh Hajishirzi、Noah A. Smith、Luke Zettlemoyer 等

---

## TL;DR

OLMoE-1B-7B 是目前**完全开源**（权重、代码、训练数据、训练日志、消融脚本全部公开）的 7B 级 MoE 旗舰，核心是一个 **6.9B 总参 / 1.3B 激活** 的细粒度 MoE：16 层、2048 hidden、每层 64 个路由 expert、top-8、**0 个 shared expert**、expert FFN 中间维只有 1024。它在 5.13T tokens 上从头训练，激活 1.3B 参数却能在 MMLU/HellaSwag/ARC 等下游基准上接近 7B dense 水平（甚至超过 DCLM-1B、TinyLlama 等纯 dense 同活参对照）。论文最具价值的不是模型本身，而是 **Section 4 的系统化消融**——这是目前公开文献里最完整的一份 "~1B 激活 / ~7B 总参 MoE 设计手册"，对 16B 级 MoE 设计有直接参考意义：他们用控制变量的方式逐项验证了 expert 粒度、shared expert、Token-Choice vs Expert-Choice、Sparse Upcycling、aux-loss、router z-loss、QK-Norm 等关键设计的取舍。**对 16B MoE 设计来说，这篇是最重要的实证参考。**

---

## 关键架构配置

| 项目 | 值 |
|---|---|
| 总参 | 6.9 B |
| 激活参数/token | 1.3 B |
| Sparsity (active/total) | ~19% |
| **N_routed (每层 expert 数)** | **64** |
| **Top-K** | **8** |
| **N_shared** | **0** （论文里明确做了消融并拒绝） |
| **d_expert (expert FFN 中间维)** | **1024** （SwiGLU intermediate_size） |
| Layers | **16** |
| Hidden / model dim | **2048** |
| Attention heads | **16** （num_kv_heads = 16，未用 GQA，纯 MHA） |
| Head dim | 128 |
| 序列长度 | 4096 |
| Vocab size | 50,304 |
| Tokenizer | GPT-NeoX-20B (基本上沿用 OLMo) |
| RoPE θ | 10000 |
| Norm | **RMSNorm**（parametric）+ **QK-Norm** |
| Activation | **SwiGLU** |
| Routing | **dropless token-choice**，linear gate + softmax + top-8 |
| Aux loss 系数 α | **0.01** （load balance） |
| Router z-loss 系数 β | **0.001** |
| Dense 前缀层数 | **0**（所有层都是 MoE，第 0 层也是） |
| 初始化 | Truncated normal，上下截断 ±0.06 |
| 训练 tokens | **5.133 T** (~1.3 epoch on OLMoE-Mix 4.06T words) |
| 优化器 | AdamW，eps = 1e-8 |
| 精度 | bf16 |
| 训练硬件 | 128× H100 |
| 训练吞吐 | ~23,600 tokens/sec/GPU (MoE) vs 37,500 (1.3B dense) |
| LR schedule | Cosine + 末尾 100B token 线性退火到 0 |

参数核验：64 expert × 1024 SwiGLU × 16 层 ≈ 6.4B 路由 expert 参数；其余 0.5B 来自共享的 attention/嵌入/norm。激活时 8/64 比例下 expert 活参约 0.8B + 共享 0.5B ≈ 1.3B，与官方一致。

---

## 核心方法 / 创新点

1. **细粒度小 expert（DeepSeekMoE 风格）**：64 个 expert × 1024 hidden 远比"8 个大 expert × 8192 hidden"细，可组合数为 C(64,8) ≈ 4.4×10⁹，路由表达力更强。
2. **零 shared expert 路线**：与 DeepSeek-V2、Skywork-MoE、Qwen2-MoE 的"shared expert isolation"明显对立，他们用受控对比证明 shared expert 在同等总参 + 同 FLOPs 下表现**略差**（详见消融）。
3. **完全从头训练，不做 sparse upcycling**：他们专门做了"upcycle vs from-scratch"对照，结论是 5T tokens 这种长训练预算下，from-scratch 在 600B tokens 之后就反超并保持领先。
4. **QK-Norm + 严格的 router z-loss**：为 MoE 训练稳定性买了双重保险。
5. **完整开源链条**：权重 / 代码 / 训练数据 (OLMoE-Mix) / wandb 日志 / 中间 checkpoint / SFT + DPO 模型 / 评估管线全部释放——这是和 Mixtral、Skywork 等"weights-only"开源最大的区别。

---

## 训练 & 系统细节

- **数据**：自建 **OLMoE-Mix**，约 17.4T tokens 原料；实际训练 5.13T = 1.3 epoch。组成（words）：DCLM-Baseline 3,860B（95%）、StarCoder 101B、peS2o 57.2B、arXiv 21.1B、OpenWebMath 12.7B、Algebraic Stack 12.6B、Wikipedia 3.69B。**DCLM 占绝对主导**，这是其下游基准强的关键之一。
- **数据消融**：§4.2.1 还对比了 OLMoE-Mix vs Dolma 1.7，OLMoE-Mix 全面胜出（Dolma 1.7 偏旧、低质量比例高）。
- **退火 (annealing)**：和 OLMo 一样最后 100B tokens 用高质量子集 + 线性衰减 LR 到 0；§4.3.2 也有退火 checkpoint 选择消融，annealing 末态显著好于中段 checkpoint。
- **对齐**：SFT (Tülu 风格 instruction set) + **DPO**（论文 §4.3.3 对比 DPO vs KTO，DPO 略优、更稳定）。OLMoE-1B-7B-Instruct 在 AlpacaEval 拿到 84.0，MMLU 51.9，GSM8K 45.5——三项都超过 OLMo-7B-Instruct。

---

## 关键消融与结果（最重要！）

### §4.1.1 MoE vs Dense（同 active 参 对比）

控制变量：1.3B active param、128× H100、130B tokens。
- MoE 用 **3× 更少 tokens / FLOPs** 达到 dense 同等下游性能。
- 因 MoE 的 memory & 通信 overhead，墙钟时间仅 **2×** 加速。
- 吞吐：MoE 23,600 tok/s/GPU vs Dense 37,500 tok/s/GPU。

→ 对 16B 设计的直接结论：**~1B active 的 MoE 在等下游性能下约只需 dense 1/3 的训练 FLOPs**。

### §4.1.2 Expert 粒度 (固定总参/激活参)

测试 N ∈ {8, 16, 32, 64}，top-K 相应缩放保持 active 参不变：
- 8 expert / 1 active：baseline
- 16 expert / 2 active：HellaSwag、MMLU 上提升 ~10%
- 32 expert / 4 active：再 +1~2 pp
- **64 expert / 8 active：继续小幅提升，但开始边际递减**

结论："more granular experts improve training loss, validation loss, and downstream performance"，但 32→64 收益已经在变小。OLMoE 选 64 是因为他们认为模型规模下 64 仍在边际收益正区。

→ **16B 设计启示**：把激活参做大（比如 ~3B）的同时维持细粒度，N=64~128 / top-8 是合理区间，不要回到 N=8 的粗粒度。

### §4.1.3 Shared Experts（OLMoE 是反例）

**这是本文最有争议、也最关键的一条**。控制总参、激活参、FLOPs 一致情况下：
- "无 shared" 配置（如 64 routed / top-8）vs "有 shared" 配置（如 63 routed + 1 shared / top-7+1）。
- 结果：**两者性能相当，shared 略差**。论文图 6 给出 HellaSwag、MMLU 等多指标曲线，shared 版本一直在下方或同位。
- 解释：shared 配置把每层"可选 expert 组合"从 C(64,8)=4.4×10⁹ 砍到 C(63,7)≈4.5×10⁸（论文报 35,960→4,495 在他们的实际配置下，缩减约 90%）；**flexibility 损失 > shared 带来的"通用知识捕获"收益**。

→ **16B 设计启示**：DeepSeek-V2/Qwen-MoE 倾向 shared expert；OLMoE 倾向无 shared。两派都有理。注意 OLMoE 实验是**同 active 参约束**——shared expert 会"占用"活参预算，所以"shared = 浪费灵活性"在此约束下成立。如果允许扩大 active 总预算（比如 16B/3B 配置），shared expert 可能价值更大（因为它的成本不再占用稀缺的 routed top-K）。这是 OLMoE 结论的边界。

### §4.1.4 Expert Choice vs Token Choice (EC vs TC)

- Token-Choice (dropless)：每个 token 选 top-K experts
- Expert-Choice：每个 expert 选 top-N tokens（天然负载均衡）

结果：
- **TC 在同 token 预算下全面优于 EC**，下游基准更高。
- EC 吞吐 +20%（29,400 vs 24,400 tok/s/GPU），因为它无需 aux loss、无需 dropless 的非常规 sparse kernel。
- 但 EC 在 decoding 时**无法因果使用**（"看 batch 内所有 token 选 expert" 在推理时是非因果的），需要近似/补丁。

→ OLMoE 选 **dropless TC**。**16B 设计启示**：训推一致性 + 下游质量优先 → dropless token-choice 仍是默认；EC 只在"训练吞吐压倒一切"场景考虑。

### §4.1.5 Sparse Upcycling vs From-Scratch

- 实验：从 OLMo-1B 在 2T tokens 处的 dense checkpoint 出发，复制 FFN 8 份做 8-expert MoE，继续训 610B tokens。
- 对照：相同配置但完全 from-scratch。
- 结果：**from-scratch 在 ~500B tokens 处赶上，~600B tokens 后超过 upcycled 版本，且差距持续扩大**。
- OLMoE 称只用了 prior work (Komatsuzaki 2023) 报告 120% 预算的 **25%** 就重现了 upcycle 优势区间——也就是说 upcycle 优势区间比文献常说的更窄。

→ **16B 设计启示**：如果你打算训 **>500B tokens** 的预算，from-scratch 更优；upcycling 只在 <300B token 的"短跑"预算或快速 baseline 验证场景下值。这与 Skywork-MoE 的"CMoE ≥ 2C_dense 就应 from-scratch" 结论一致。

### §4.1.6 Load Balancing Loss

- 系数 **α = 0.01**（沿用 Switch Transformer/GShard 习惯）。
- 图 9 显示：加 aux loss 显著改善训练损失和验证损失；没有 aux loss 时 expert utilization 严重不均匀，下游性能掉。
- 没有报告 α ∈ {0.001, 0.001, 0.1} 的精细扫，但 0.01 是默认稳的。

### §4.1.7 Router Z-loss

- 系数 **β = 0.001**。
- 图 11：加 router z-loss 显著降低训练 spike（稳定性）+ 提升 final loss + 提升下游指标，代价仅 ~2% 吞吐。
- 公式：z-loss = (1/N) Σ_b (log Σ_e exp(z_b,e))²，约束 router logits 不要过大。

→ **16B 设计启示**：α=0.01、β=0.001 是稳健默认；β 没有 α 重要但提升稳定性显著。

### §4.2 一般训练设置消融

| 设置 | OLMoE 选择 | 结论 |
|---|---|---|
| §4.2.1 数据 | OLMoE-Mix (DCLM 主导) > Dolma 1.7 | OLMoE-Mix 全面胜出 |
| §4.2.2 初始化 | Truncated normal ±0.06 > Normal | Truncated 更稳，减少 spike |
| §4.2.3 Norm | RMSNorm (parametric) > non-parametric LN | 更稳定，下游略好 |
| §4.2.4 嵌入 weight decay | **包含**优于排除 | Spike 显著减少 |
| §4.2.5 QK-Norm | 必加 | 显著减少训练 spike，几乎无吞吐损失 |
| §4.2.6 AdamW eps | 1e-8 > 1e-5 | 更小 eps 训练更稳，下游略好 |

### §4.3 Adaptation 消融

- §4.3.1 SFT 期间是否保留 aux loss：**保留**略好（防止 SFT 过程把 expert 平衡破坏掉）。
- §4.3.2 annealing checkpoint 选择：用 annealing 完成的 checkpoint 做 SFT 起点最优。
- §4.3.3 偏好优化：**DPO > KTO**，更稳定收敛、final 性能更高。

### 下游基准（1B 激活组）

| Benchmark | OLMoE-1B-7B | OLMo-1B (07/24) | TinyLlama-1B | DCLM-1B | Llama3.2-1B |
|---|---|---|---|---|---|
| MMLU | **54.1** | 32.1 | 33.6 | 48.5 | 38.2 |
| HellaSwag | **80.0** | 67.5 | 60.8 | 75.1 | 67.3 |
| ARC-Challenge | **62.1** | 36.4 | 38.1 | 57.6 | 43.5 |
| ARC-Easy | **84.2** | 53.5 | 69.5 | 79.5 | 71.6 |
| PIQA | **79.8** | 74.0 | 71.7 | 76.6 | 73.7 |
| Winogrande | **70.2** | 62.9 | 60.1 | 68.1 | 62.5 |

OLMoE-1B-7B 在所有项上全面碾压同激活参 dense，**且大幅超过 OLMo-7B**（同总参 dense）很多指标——证明"MoE 在等总参 + 等数据预算下确实有质量增益"。

---

## 对 16B MoE 设计的启示

1. **粒度选择**：N=64 / top-8 / d_expert=1024 在 ~1B 激活级别仍有正收益。把模型放大到 16B/3B 激活时，可考虑 N=64~128、top-6~8、保持 d_expert 在 1024~2048 区间。**不要回到 N=8 的粗粒度**——Mixtral 风格在此规模可能浪费。
2. **Shared expert 是开放问题**：OLMoE 在同 active 参约束下证明无 shared 更优，但 16B/3B 配置可能允许 1 shared 而不显著挤压 routed 空间。建议：先 baseline 无 shared，再做 N=64-routed + 1-shared / top-7+1 的对照消融。
3. **完全 from-scratch**：训练预算 >500B tokens 时不再考虑 upcycling。OLMoE 5T tokens 是 16B-MoE 设计的合理参照。
4. **稳定性套件**：QK-Norm + 嵌入 weight decay + RMSNorm + router z-loss (β=0.001) + AdamW eps=1e-8 + truncated init ±0.06 是已验证的稳定组合，对 16B 规模同样适用，不要省。
5. **Aux loss α=0.01、router z-loss β=0.001**：直接沿用。
6. **MoE 第 0 层也用 MoE**：OLMoE 没有 dense prefix，论文里没看到对此的反例；DeepSeek-V2 / Qwen-MoE 倾向保留 dense 前缀（"early layers attend more to syntax, less benefit from MoE"），这是另一个值得做的对照点。
7. **Dropless Token-Choice + Megablocks** 是默认路线。
8. **数据预算 ≥ 4T tokens, DCLM 主导**：在等参数下 OLMoE 用 5T tokens 直接拉到 dense-7B 水平，这是数据驱动而非架构驱动的关键。

---

## Caveats / 局限

1. **基准评测范围有限**：OLMoE 主要测 commonsense + MMLU，**长上下文 / 代码 / 数学 / 多语言** 数据点较少。Math 类基准只在 OLMoE-Instruct 阶段评估（GSM8K 45.5），与 Mixtral 8x7B (GSM8K 58.4) / Yuan 2.0-M32 (92.7) 不在一个量级。
2. **序列长度只有 4096**：无 long-context 优化（无 RoPE scaling、YaRN、NTK 调整），需要后训扩展。
3. **shared expert 结论的约束**：实验在等 active 参下做，对"shared 占用预算 vs routed 占用预算"的相对成本敏感。结论不能直接外推到所有 MoE 设定。
4. **单一规模消融**：所有消融都在 1.3B/6.9B 这一档做的，未在更大规模复现。对 16B/3B 设计需要承担一定外推风险。
5. **数据强度可能掩盖架构差异**：DCLM-Baseline 数据质量极高（DCLM-1B dense 都能上 MMLU 48.5），部分"MoE 优势"可能来自数据本身。
6. **未充分对比 GQA / SWA**：OLMoE 用纯 MHA（16/16 heads），未试 GQA。16B 设计若要做长上下文 / 高 batch 推理，GQA 几乎必选，OLMoE 在此点没有给出指导。
