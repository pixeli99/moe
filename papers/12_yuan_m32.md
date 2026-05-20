# Yuan 2.0-M32: Mixture of Experts with Attention Router

- **arXiv**: 2405.17976
- **机构**: IEIT Systems (浪潮信息)
- **发表时间**: 2024 年 5 月
- **作者(代表)**: Shaohua Wu、Jiangang Luo、Xi Chen 等

---

## TL;DR

Yuan 2.0-M32 是浪潮信息 IEIT Systems 的 **40B 总参 / 3.7B 激活** MoE，结构上基于其前作 Yuan 2.0-2B dense，最大创新是**用一个"Attention-style router"代替传统的 `softmax(W·x)` 路由**——他们认为传统 router 把每个 expert 当作"独立 feature vector"，忽略了 top-K 选中的多个 expert 之间应有的协同关系；他们用 self-attention 风格的 Q/K/V 三套权重把"expert 间相关性"显式建模进路由。论文的 ablation 显示这相比 classical router 在同参规模下 **test loss 降低 ~0.4%** (2.117 → 2.109)，看起来不大但他们把它带到了完整 40B/3.7B 训练，配合 LFA (Localized Filtering-based Attention) 拿下了在 ARC-C 上 **95.8（超过 Llama3-70B 的 93.3）**、MATH 上 **55.9（超过 Llama3-70B 的 50.4）** 的结果，且**激活参数只有 1/19**。论文也额外报告 N=8/16/32 expert 的 scaling 消融。

---

## 关键架构配置

| 项目 | 值 |
|---|---|
| 总参 | **40 B** |
| 激活参数/token | **3.7 B** |
| Sparsity (active/total) | ~9.3% |
| **N_routed** | **32** |
| **Top-K** | **2** |
| **N_shared** | **0** （论文显式测过 shared-expert 路由并拒绝） |
| Router | **Attention Router** (Q/K/V 三套线性) |
| Attention | **LFA (Localized Filtering-based Attention)** —— Yuan 2.0 的特色注意力，引入局部依赖建模 |
| d (token vector 输入 router 的维度) | **2048**（即 hidden size） |
| Layers | 论文未明示，沿用 Yuan 2.0-2B 的结构（推断 24 层左右） |
| Hidden | 2048 |
| Tokenizer | 沿用 Yuan 2.0（中英双语 BPE） |
| Position encoding | RoPE，pretrain base = 10000；fine-tune 用 **NTK-aware scaling**，b' = b · s^(|D|/(|D|−2)) = 40,890（从 4096 扩展到 16384，s=4，|D|=128） |
| 训练 tokens | **2,000 B** （2T） |
| 训练数据原料 | >3,400 B tokens 候选 |
| 数据混合 | Web 25.2% / **Code 47.5%** / Math 6.36% / Book 6.4% / Specific-domain 1.93% / Translation 1.1% / Encyclopedia 1.2% / Thesis 0.84% |
| 双语 | 中英 |
| 训练并行 | DP + PP（**不用 TP、不用 optimizer parallelism**） |
| Optimizer | AdamW (推断；论文未显式说) |
| Pretrain LR | 1.0e-5 ~ 1.0e-4 (cosine decay) |
| Finetune LR | 8.0e-5 (constant) |
| Pretrain seq len | 4096 |
| Finetune seq len | 16384 |
| Pretrain batch size (global) | 1536 |
| Finetune batch size (global) | 1152 |
| Final training loss | 1.22 |
| 推理 GFlops/token | 7.4 (即 Llama3-70B 的 1/19) |
| Fine-tune GFlops/token | 22.2 |

---

## 核心方法 / 创新点

### 1. Attention Router (核心创新)

**动机**：传统 MoE router 是

$$ P = \text{Softmax}(W \cdot I), \quad W \in \mathbb{R}^{N \times d} $$

每个 expert 对应 W 的一行（一个独立 feature vector）；选 top-K 时这些 feature vectors **互相独立**。但实际推理时被选中的 top-K experts 是协同工作（输出加权求和），因此 router 决策应当考虑 expert 之间的相关性。

**公式（论文 §3）**：给定输入 token 向量 I ∈ R^d，d=2048：

$$ Q = W I, \quad W \in \mathbb{R}^{N \times d} $$
$$ K = W' I, \quad W' \in \mathbb{R}^{N \times d} $$
$$ V = W'' I, \quad W'' \in \mathbb{R}^{N \times d} $$
$$ P = \text{Softmax}(Q K^T) V, \quad P \in \mathbb{R}^N $$

然后从 P 中取 top-M 个分数对应的 expert（论文 M=2，N=32）。

**机制理解**：Q/K/V 都是 N 维向量（每维对应一个 expert）。QK^T 形成一个 N×N 矩阵，刻画 expert 之间在当前 token 上的两两相关性；再乘 V 投影出最终的 N 维路由分数。

**实证 (Table 1)**：同参数规模下，8-expert 模型 test loss：
| Router | Params (M) | Test loss |
|---|---|---|
| **Attention Router** | 826.0 | **2.109** |
| Classical router | 825.8 | 2.117 |
| Shared Expert router (2 fixed + top-2/14) | 825.8 | 2.117 (但训练慢 7.35%) |

差距小但稳定，且 Shared Expert 在此实验下**未带来任何精度提升、还更慢**——这是 Yuan 团队选择 N_shared=0 的依据。

### 2. Expert 数 scaling（Table 2）

固定 per-expert 参数、固定 top-2，训 50B tokens：
| N | Test loss |
|---|---|
| 8 | 1.820 |
| 16 | 1.787 (−2.0%) |
| 32 | 1.754 (−3.6%) |

→ 选 N=32。注意这个 trend 与 OLMoE 的 8→16→32→64 趋势一致，但 Yuan 没继续扩到 64。

### 3. LFA (Localized Filtering-based Attention)

继承自 Yuan 2.0。在标准 self-attention 前加一个局部过滤模块，让 token 显式吸收**相邻 1-2 个位置**的信息。论文称其改善了模型对局部依赖的建模、提升了下游精度。属于"前置 conv-like 增强"，与 MoE 设计正交。

### 4. NTK-aware RoPE extension

Pretrain seq_len = 4096；fine-tune 时扩展到 16384 (s=4)。用 NTK-aware base scaling：

$$ b' = b \cdot s^{\frac{|D|}{|D|-2}} = 10000 \cdot 4^{128/126} \approx 40,890 $$

他们对比了多个 base 值 (40k, 80k, 160k, ..., 10.24M) 在 needle-in-haystack 任务上的表现，**NTK-aware 的 40,890 最好**。

---

## 训练 & 系统细节

- **数据 (47.5% code !)**：这是 Yuan-M32 数据混合的最大特点——**几乎一半是代码**。这解释了它在 HumanEval (zero-shot 74.4，14-shot 78.1) 上接近 Llama3-70B (81.7) 的能力。Stack v2 + 自生成 + 中文注释翻译。
- **数学 6.36%**：proof-pile v1/v2 + AMPS + MathPile + StackMathQA + Python 数值合成。这也解释了 MATH 55.9。
- **训练 token**：**2T from scratch**（不用 upcycling）。
- **并行**：DP + PP **only**（**不用 tensor parallelism、不用 optimizer parallelism**）。论文称这是出于"通信开销 vs 性能 tradeoff"考虑；对小 hidden (2048) 来说 TP 收益不大。
- **Fine-tune**：seq 16384，LR 8e-5 constant。
- **训练计算消耗**：**仅为同参数 dense 模型的 9.25%**（一作总结）。

---

## 关键消融与结果

### Router 对比 (Table 1)
见上文：Attention > Classical = Shared Expert。

### Expert 数 (Table 2)
N=32 比 N=8 的 test loss 低 3.6%。

### NTK-aware RoPE base 扫描
40,890 最佳；远超 simple base ×100 (1M)。

### 下游基准 (Tables 3-7)

| Benchmark | Llama3-70B | Mixtral-8x22B (39B-act) | DeepSeek-V2 (21B-act) | Mixtral-8x7B (12.9B-act) | **Yuan 2.0-M32 (3.7B-act)** |
|---|---|---|---|---|---|
| HumanEval (0-shot) | 81.7 | 45.1 | 81.1 | 40.2 | **74.4** |
| HumanEval (14-shot) | – | – | – | – | **78.1** |
| GSM8K | 93.0 | 78.6 | 92.2 | 58.4 | **92.7** |
| **MATH** | 50.4 | 41.8 | 53.9 | 28.4 | **55.9 (best)** |
| MMLU | 80.3 | 77.8 | 77.8 | 70.6 | 72.2 |
| **ARC-C** | 93.3 | 91.3 | 92.3 | 85.9 | **95.8 (best)** |

**Yuan 2.0-M32 在 MATH 和 ARC-C 上拿下榜首**，超过 Llama3-70B，且激活参数只有 1/19。

### 效率 (Table 7)
- 推理 GFlops/token：Yuan 7.4 vs Llama3-70B 140 vs Mixtral-8x22B 78
- Fine-tune GFlops/token：Yuan 22.2 vs Llama3-70B 420 vs Mixtral-8x22B 234
- 平均精度 / GFlops：**Yuan 10.69 vs Llama3-70B 0.57 (18.9× 高效)**

→ 论文的核心 marketing："最高性价比"。

---

## 对 16B MoE 设计的启示

1. **Attention Router 是低成本可选项**：参数开销几乎 0（多两个 N×d 矩阵，N=32, d=2048 → 每层 ~131K 参数），训练吞吐影响小，但提供"expert 间相关性"建模能力，论文实证有 ~0.4% loss 改善。**16B 设计可以做对照消融**：传统 vs Attention Router。

2. **N=32 已被验证有效**：是 OLMoE N=64 和 Mixtral N=8 之间的中间档；对 16B 设计也可接受。但 OLMoE 的"细粒度 + 多 expert" 路线在 8→64 趋势更强，建议 N≥64。

3. **Sparsity 9% 极激进**：Yuan 用 32 expert / top-2 → 6.25% expert 活率，加上 attention/嵌入 → 整体 9% 激活率。这比 Mixtral (28%)、OLMoE (19%) 都低很多。对 16B/3B 设计这就是一个 5.3× sparsity 的目标点 (3/16 ≈ 18.75%)。

4. **数据驱动很关键**：47.5% Code + 6.36% Math 是 MATH/HumanEval 表现的直接原因。**16B 如要在 code/math 上对标 Yuan，必须显著加 code 数据比例**。

5. **DP+PP, no TP, no Opt Parallel**：在 hidden=2048 这种"较窄"模型上，**TP 不是必需**，是个反 LLaMA-style 的工程选择，对小型 MoE 启发。

6. **LFA**：本质是 conv prior 注入 attention，非 MoE 必须组件，建议先观察、不优先采纳（增加架构非标准性）。

7. **NTK-aware RoPE scaling**：长上下文扩展时的标准做法，公式 b' = b · s^(|D|/(|D|−2)) 可直接复用。

8. **Shared Expert 的另一个反例**：在 Yuan 的实验中 Shared Expert 既未提升精度又拖慢 7%，进一步加强 "shared expert 不总是必要" 的实证。

---

## Caveats / 局限

1. **MMLU 仍弱（72.2）**：尽管 MATH/ARC-C 拿了榜首，MMLU 上明显逊于 Llama3-70B (80.3)，**通用 NLP 知识覆盖不及代码/数学专精**——这是 47.5% code 数据带来的"专化"代价。如果 16B 设计目标是平衡型而非偏 code/math，Yuan 的数据 mix 不能照搬。

2. **Attention Router 收益边际**：test loss 2.117 → 2.109 仅 ~0.4%，且只在 8-expert 控制实验上验证。在 N=32 完整模型上的隔离贡献度未单独消融。**收益可能小到不值得换路由架构**。

3. **训练细节披露不全**：AdamW betas、weight decay、warmup steps、scheduler 详细公式都不在主文。Appendix A 仅给四项 (LR 范围、decay style、seq len、batch)。

4. **Layer 数 / 详细 dim 不在论文**：必须查 Yuan 2.0-2B 论文 (arXiv 2311.15786) 才能补全。

5. **没有 long-context 消融**：fine-tune 只到 16k，没测 32k/128k。

6. **路由分析单薄**：未做 Mixtral §5 那种 expert 专精化可视化。

7. **没有 upcycle 对比**：Yuan 是 from-scratch；没有 ablation 是否 upcycle 更省。

8. **基线对比不平等**：Mixtral-8x7B/8x22B 用 zero-shot HumanEval (45.1/40.2)，而很多其它论文/排行榜里 Mixtral 系列在更优 prompt 下能上 50+。Yuan 的"碾压 Mixtral on code" 在 prompt 一致时可能差距没那么大。
