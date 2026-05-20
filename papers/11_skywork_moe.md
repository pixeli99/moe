# Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

- **arXiv**: 2406.06563
- **机构**: Kunlun Inc. (Skywork)
- **发表时间**: 2024 年 6 月
- **作者(代表)**: Tianwen Wei、Bo Zhu、Liang Zhao 等

---

## TL;DR

Skywork-MoE 是 Kunlun 在 2024 年 6 月发布的 **146B 总参 / 22B 激活** MoE（**N=16 / top-2 / 52 层**），最大特色不是规模而是**两项训练技术** + **一份详细的实证决策报告**：

1. **Gating Logit Normalization**：在 softmax 之前对 router logits 做 mean-std 标准化再乘可调温度 λ，强迫 router 输出"更尖锐"的分布，避免 top-K 几乎均匀分配的退化问题；
2. **Adaptive Auxiliary Loss**：让 aux-loss 系数按 layer/step 根据 token drop rate 自适应调整，而不是固定 0.01；
3. 论文给出**实证的 upcycling vs from-scratch 决策准则**：当 MoE 训练预算 C_MoE ≥ 2× dense 训练预算 C_dense 时，应从头训练；C_MoE ≪ C_dense 时应 upcycle。Skywork-MoE 本身是从 Skywork-13B dense checkpoint **upcycled** 而来（13B dense 已经训过 ~3.2T tokens），随后再训了大约 2T tokens 的 MoE。

对 16B MoE 设计的核心价值是 **upcycling 决策、稳定性技巧和训练系统配置**。

---

## 关键架构配置

| 项目 | 值 |
|---|---|
| 总参 | **146 B** |
| 激活参数/token | **22 B** |
| Sparsity | ~15% |
| **N_routed** | **16** |
| **Top-K** | **2** |
| **N_shared** | **0** |
| **d_expert (FFN intermediate)** | **12,288** |
| Layers | **52** |
| Hidden / model dim | **4,608** |
| Attention heads | **36** |
| Head dim | **128** |
| 序列长度 | **8,192** native |
| Norm | RMSNorm |
| Position encoding | RoPE |
| Activation | SwiGLU |
| Routing | **Softmax with Gating Logit Normalization**（见下） |
| Aux loss | **Adaptive auxiliary loss**，自动按 drop rate 调整，α 上限 0.01 |
| Capacity / dropless | 有 drop（监控 drop rate）；非完全 dropless |
| Dense 前缀 | 0 |
| 训练 tokens | Skywork-13B base 已训 ~3.2T 公开 + 内部 2T；MoE 后续训练 ~2T |
| 初始化 | **Sparse Upcycling**：FFN 来自 Skywork-13B、router 随机 |
| 硬件 | 1,536× A800-80G (192× HGX-A800 nodes) |
| 并行 | 12-way PP × 4-way TP-EP × 32-way DP + ZeRO-1 |
| MFU | **38%** ≈ 690 tokens/GPU/s |

---

## 核心方法 / 创新点

### 1. Gating Logit Normalization

**动机**：原始 `softmax(W·x)` 输出的 top-K 概率经常很平（top-1 和 top-K 概率几乎相同），导致 expert 之间几乎平均加权，**失去 MoE 的"选择性"语义**——这与 dense 模型行为接近，没把 expert 多样性挖出来。

**做法**（论文公式）：

$$ \tilde{z} = \lambda \cdot \frac{z - \mu(z)}{\sigma(z)} , \quad g = \text{softmax}(\tilde{z}) $$

其中 z ∈ R^N 是 router logits（N 个 expert），μ、σ 是 mini-batch 内或单 token 内的均值、标准差。λ 是可调温度系数（论文默认 λ=1）。

**效果**：top-1 probability 比 top-K 显著拉开，gating 真正变成"选择"而不是"加权平均"。论文显示这显著提升下游表现。

→ 这是 Skywork-MoE 最具复用价值的创新，**直接可塞入任何 MoE 实现，几乎无开销**。

### 2. Adaptive Auxiliary Loss

**动机**：固定 α=0.01 在某些层 / 某些训练阶段过强（压制 expert 专精化）或过弱（导致 drop rate 飙升）。

**做法**（论文公式）：

每一步、每一层独立维护 α^(l)_i：

$$ \hat{\alpha}^{(l)}_{i+1} = f(d^{(l)}_i) $$
$$ \alpha^{(l)}_{i+1} = \beta \cdot \alpha^{(l)}_i + (1-\beta) \cdot \hat{\alpha}^{(l)}_{i+1} $$

其中 d^(l)_i 是该层当前 token drop rate，f 是单增函数（论文用 piecewise-linear f(d) = ξ·d，上限封顶 α_max）。EMA 系数 β 让 α 平滑漂移。

**论文实际配置**：ξ = 1/5、α_max = 0.01、β = 0.99。

**效果**：保证 aux-loss 始终被压在 "secondary to NTP loss" 的位置，且能动态应对不同层的不平衡。

### 3. Sparse Upcycling from Skywork-13B

**做法**：
- 取 Skywork-13B dense checkpoint。
- 每个 MoE 层的 16 个 expert 用**同一份** dense FFN 权重初始化（每个 expert 是 dense FFN 的拷贝）。
- Router 随机初始化（这一步 Skywork 论文没明说细节，但工程上是随机）。
- 在 MoE 数据 mix 上继续训练 ~2T tokens。

**关键观测**：训练过程中 expert 之间的 cosine similarity **从 1.0 逐步下降**，表明专精化在发生（但比 from-scratch 慢）。

### 4. Upcycle vs From-Scratch 的实证决策准则（论文核心结论之一）

实验：在 13B 参数规模下用三种起点 (random init / 100B-trained dense / 200B-trained dense / 300B-trained dense) 各训若干 budget。结论：

| 训练预算 C_MoE | 建议策略 |
|---|---|
| C_MoE ≪ C_dense | **Upcycle** |
| C_MoE ≈ C_dense | 取决于 dense checkpoint 质量 |
| **C_MoE ≥ 2 × C_dense** | **从头训练 (from scratch)** |

→ 在 300B token 预算（约 2× dense 训练）以上，from-scratch 全面碾压所有 upcycled 版本。这与 OLMoE §4.1.5 的结论完全一致（OLMoE 也观察到 ~500B-600B tokens 后 from-scratch 反超）。

---

## 训练 & 系统细节

- **训练数据**：Skywork SkyPile 子集 + 合成数据。英/中/代码 ≈ **7:2:1**。
- **基座**：Skywork-13B（公开 3.2T 版本 + 内部 2T 补训）。
- **MoE 继续训练**：~2T tokens。
- **优化器细节**：论文没公开 AdamW 具体 betas、peak LR、warmup（peak LR 提到 3e-3 是 dense baseline 实验，不是 MoE 最终配置）。
- **硬件 & 并行**：
  - 1,536× A800-80GB (192× HGX-A800 nodes)
  - 12-way Pipeline Parallel
  - 4-way Tensor-Expert Parallel (EDP)
  - 32-way Data Parallel + ZeRO-1
  - MFU **38%**，~690 tokens/GPU/s
- **Drop rate 监控**：贯穿训练，drop rate 上升直接触发 adaptive α 的升高。

---

## 关键消融与结果

### Gating Logit Normalization 消融

- 无 normalization：top-K 概率与非选中 expert"几乎一样"，模型退化为 dense 行为。
- 加入 normalization 后下游性能显著提升（论文用 perplexity + 下游基准多项指标证明）。
- λ 越大，gating 越尖锐；论文用 λ=1 已足够。

### Adaptive Aux Loss 消融

- 固定 α=0.01 → drop rate 在某些层飙到 30%+。
- Adaptive α → drop rate 稳定在 10% 以下，loss 曲线更平。
- α_max=0.01 设上限避免压制专精化。

### Upcycling vs From-Scratch (Figure 1 中间面板)

- **同等 300B token 总预算下，from-scratch 优于所有 upcycle 起点**。
- 不同 dense checkpoint 起点之间差距小，意味着"先训更久的 dense"对 upcycle 帮助有限。
- 启示：upcycle 的"先验"在长训练中被冲淡，from-scratch 的"diversification 优势"是关键。

### Expert similarity 衰减

- Upcycle 初始 cosine similarity ≈ 1.0。
- 持续训练后下降但**不会降到 0**，意味着 upcycled 的 expert 始终保留更多共性。
- From-scratch 的 expert similarity 起点低、终点更低 → 更强的专精化。

### 下游基准 (论文报告)

Skywork-MoE 22B-active 在通用基准上对标 Mixtral 8x22B (39B-active) 和 DeepSeek-V2 (21B-active)：
- 在中英双语下游 (C-Eval / CMMLU 等) 上 Skywork-MoE 占明显优势（因为中文数据更多）。
- 在英文 MMLU / GSM8K / HumanEval 上与 Mixtral 8x22B 大致同档。

---

## 对 16B MoE 设计的启示

1. **Gating Logit Normalization 直接采纳**：几乎零成本、显著提升 routing 选择性。这是 Skywork 给业界最具复用价值的一招，**16B MoE 应该默认开启**。

2. **Adaptive Aux Loss 值得评估**：固定 α=0.01 已是 OLMoE 默认；Skywork 的 adaptive 版本是 superset，可作为 fallback，当观察到 drop rate 异常时启用。如果训练数据均衡 + drop rate 稳定，OLMoE 的固定 α 更简单。

3. **Upcycle/Scratch 决策**：和 OLMoE 一致，**>500B-1T tokens 直接 from-scratch**。如果 16B 设计预算 ≥ 4T tokens（参照 OLMoE 5T），毫无疑问 from-scratch。Upcycle 仅在快速 prototype（<200B tokens）阶段有意义。

4. **粒度对比**：Skywork N=16 / top-2 / d_expert=12288 是 "Mixtral 路线"，**不是细粒度路线**。在 22B 激活 / 146B 总参规模下，N=16 似乎够用；但 OLMoE/DeepSeekMoE 都已实证细粒度更优。16B 设计应往细粒度走，不要照搬 Skywork 的 N=16。

5. **训练系统参考**：1536 A800 + 12 PP × 4 TP-EP × 32 DP 达到 38% MFU 是工业级标杆。**16B 规模在 A800/H100 上可用更小 PP/TP，重点关注 EP + ZeRO**。

6. **数据混合 7:2:1**：如果目标是中英双语 MoE，Skywork 的数据配比是合理参考点。

---

## Caveats / 局限

1. **基座依赖闭源**：Skywork-13B 内部版本含 2T 额外 tokens，未发布。本论文的"upcycle vs scratch"对照对外部研究者复现性差。

2. **架构粒度落后**：N=16 在 2024 中段已偏粗，**对比 DeepSeek-V2 (160 routed + 2 shared, top-6+2)** 在效率/性能比上明显落后。Skywork-MoE 22B active 才打平 DeepSeek-V2 21B active，但后者 sparsity 显著更高。

3. **训练超参未充分披露**：peak LR、batch size、AdamW betas 等关键 reproducibility 信息缺失。

4. **没有公开 capacity factor**：使用了 drop 但 capacity factor 数值未明。

5. **未做 shared expert 对比**：所以这篇论文不能直接回答"shared expert 是否值得"的问题。Skywork 自己选了 N_shared=0，但没消融。

6. **下游评测偏弱**：相比 Mixtral 论文给出完整 Llama-2 70B + GPT-3.5 表格，Skywork 论文的 benchmark 对比表更稀疏，且偏中文向。

7. **Gating Logit Normalization 的边界**：什么场景下 λ > 1 才必要？论文没给细致扫；可能在更小 N 时无效。
