# Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models

- **arXiv**: 2501.12370 (v1: 21 Jan 2025; v3: 2 Jul 2025)
- **机构**: Apple（主），MIT（Shah）
- **作者**: Samira Abnar\*, Harshay Shah\* (MIT), Dan Busbridge, Alaaeldin El-Nouby, Josh Susskind, Vimal Thilak\* (\*core contributors)
- **发表**: ICML 2025 接受

## TL;DR
本文专门研究 **MoE sparsity S 的最优值**。固定 FLOPs budget，扫描 sparsity 取值，拟合 IsoFLOP 曲面 `L(N, D, S)`。三大结论：
1. **更稀疏 + 更大 total params 在固定 compute 下更优**。
2. **最优 sparsity S\* 随 compute budget 和 total params 单调上升，并收敛到 1.0**。
3. **Active params 随 compute 缓慢减少**，total params 加速增加——MoE 把"放大模型"和"放大计算"解耦了。

## 核心命题
1. **S\*(N, C) → 1 as N, C → ∞**：足够大的模型在足够大 compute 下应该尽量稀疏。
2. **存在 N_th 阈值**：当 N > N_th 时增加 sparsity 永远有用；N < N_th 时过度稀疏反而伤害性能。这个阈值是 compute-budget-dependent 的。
3. **N_a\*（最优 active 参数）随 sparsity 单调下降**——即"越稀疏，激活的越少"，但 N\*（最优 total 参数）随 sparsity 单调上升。两者在 compute 上正交。

## 关键公式

### Sparsity 定义（§2）
$$
S = \frac{E - K}{E}
$$
- E = 总专家数（per layer），K = top-K 中的 K。
- 也即 "fraction of inactive experts"。例如 E=8, K=1 → S=0.875；E=16, K=2 → S=0.875；E=128, K=8 → S=0.9375。
- 注意：本文 sparsity **不是** active/total params 直接比，而是 expert-level。但二者数值上等价（共享 expert 除外）。

### 联合 scaling law（公式 6）
$$
L(N, D, S) = \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}} + \frac{c}{(1-S)^{\lambda}} + \frac{d}{(1-S)^{\delta} \cdot N^{\gamma}} + e
$$
- N = total 参数数。
- D = 训练 tokens。
- S = sparsity。
- `(1-S)` = fraction of active experts ≈ active / total。
- `λ` 拟合出来是**负的**：稀疏越大、loss 越小（intuition 验证：`(1-S)^负` 当 S→1 时变大，但 c 前是负值或符号约定使得贡献为减少 loss）。
- 系数拟合用 L-BFGS + Huber loss (δ=1e-3)。具体数值见附录 F Table 3（论文里没有 inline 全列，但符号性指出 λ < 0）。

### 二维边际公式
- 固定 sparsity S，求 N*(C, S)：仍是 Chinchilla 风格 N* ∝ C^a。
- 固定 N，求 S*(C, N)：parabolic 形状，沿 S 维度有 vertex 最优点。

### FLOPs 估算
- MoE：`C ≈ 6 · N_a · D`（用 active 参数代替 Chinchilla 的 N）。
- Dense：`C ≈ 6 · N · D`。

## 实验设置
- **架构**: Transformer MoE，每层都 MoE 化。
- **数据**: RedPajama，context length 2048，vocab 50432，batch 1024。
- **Sparsity 网格**: {0, 25, 50, 75, 90, 95, 98}%。即 S ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98}。
- **Compute budgets**: 3e19 → 1e21 FLOPs（约 4 个数量级跨越，但绝对量仍偏小）。
- **Top-K 设置**: 不在 paper 主表中固定 K，而是改变 E 和 K 共同决定 S。
- **模型尺寸**: Active 范围约 100M – 1B；total 上限到 ~20B。
- 拟合：3D 多项式 (2,2,2) on (log N, -log(1-S), L)。MSE on hold-out = 0.0001。

## 主要结论

### 1. Larger + Sparser 在 fixed compute 下永远更好（Fig. 2, 3）
- "When memory and communication overheads are disregarded, increasing sparsity while proportionally expanding the total number of parameters consistently leads to a lower pretraining loss, even when constrained by a fixed training compute budget."
- 即 IsoFLOP 沿 S 维度单调下降，loss 在 S=98%（最大测试值）仍未触底。

### 2. 最优 sparsity S\* 随 N 上升，收敛到 1.0（Fig. 2a, Fig. 4）
- Fig. 4: x 轴 N total，y 轴 S\*，多条曲线对应不同 C（3e19, 6e19, 1e20, 3e20, 1e21 FLOPs）。
- 趋势：N 越大，S\* 越接近 1。C 越大，曲线整体上移。
- **关键阈值 N_th**：对一个给定 compute budget，存在 N_th 使得 N < N_th 时 sparsity 不能太高（反而伤性能），N > N_th 时怎么稀疏都不亏。

### 3. Active params 随 sparsity 下降（Fig. 3b）
- "the optimal active number of parameters N_a\* decreases as sparsity level increases"。
- 即固定 compute，越稀疏 → active 越少 → FLOPs/example 越少 → 可以训更多 tokens D。
- "the optimal active number of parameters decrease more rapidly with sparsity as the training compute budget increases"。

### 4. Downstream performance（Fig. 5）
- Lambada（语言理解）、PIQA（commonsense）、Wikidata（world knowledge）：downstream error 与 upstream loss tight correlation，与 sparsity 无关。即"用 loss 就能预测 downstream，sparsity 不破坏 transfer"。
- **SQuAD（reading comprehension）例外**：相同 perplexity 下，**稀疏模型反而比稠密模型差**。猜测原因：稀疏 → active 少 → inference-time FLOPs 少 → 复杂推理任务的瓶颈。
- **CoT 在 MoE 上比 dense 上收益更大**（Appendix E）。

### 5. Parametric form
- `λ` 拟合为负，与 "sparser → lower perplexity" 直觉一致。
- 增加 sparsity 等价于 "增加 c/(1-S)^λ 这一项中的 (1-S)^|λ|" → loss 降低。

## 对 16B MoE 设计的启示

### Sparsity 1/8 是否合理？
- "1/8 sparsity" 通常指 **K/E = 1/8**，即 **S = 7/8 = 0.875 = 87.5%**。这对应本文测试网格中的 **S=87.5%**，位于 {75, 90} 之间，处于 reasonable 区。
- 本文 IsoFLOP 曲线在 S=0.875 还远未到拐点（拐点接近 S=1）。**结论：S=0.875 在 16B-total 规模下偏保守，可以更稀疏（S=0.9 或 S=0.94）**。
- 但**注意**：本文用 pretraining loss 评估。**SQuAD 等 reasoning-style 任务上稀疏模型偏弱**，对 16B 这种需要"小而强"的模型，反而不该走到极限稀疏。

### 推荐 sparsity 区间
- **基础推荐**：S ∈ [0.875, 0.9375]，即 K/E ∈ {1/8, 1/12, 1/16}。
- **激进版**：若 16B 优先 pretrain loss / cheap inference → S=0.95（E=20, K=1 或 E=40, K=2）。
- **保守版**：若 16B 关注 reasoning / SQuAD / 长链 CoT → S=0.75（E=8, K=2）。

### Total / Active 参数比
- 设 active = 2B（"16B-total"的 1/8 spec），按 Abnar 在 1e20 FLOPs 附近的趋势：N\* ≈ 16B, N_a\* ≈ 1–2B 是合理 compute-optimal 配置。
- 若把 spec 理解为 "16B active params"（即 ~128B total 在 sparsity=0.875），那 active=16B 已经比 Chinchilla optimal 偏大；除非训 >> 320B tokens，否则会 under-train。

### Top-K 选择
- 本文未单独验证 K 的影响（K 和 E 共同决定 S）。从 fine-grained MoE 视角（结合 Krajewski），**K ≥ 2 比 K=1 更稳健**（routing 平滑）。
- 与 Yokota et al. 2025 结合：**K=4 或 K=8 比 K=2 在 reasoning 上更好**，与 sparsity 选择正交。

### 训练 tokens
- 由于 active params 随 sparsity 减少，**16B-total / 2B-active 可以训到 300B–500B tokens 还在 compute-optimal 范围内**（远超 dense 2B 的 40B token 限）。
- 实务推荐：≥ 6B tokens × (1-S)^(-1) × N_a/(1B) → 16B-total, 2B-active 适合 240B+ tokens。

### 一句话推荐
**16B MoE：S = 0.875 是稳健下限，最优区间 [0.875, 0.94]；K ≥ 2；train ≥ 250B tokens；若关注 reasoning 不要走到 S ≥ 0.95**。

## Caveats / 局限

1. **完全用 FLOPs 衡量成本**：忽略 memory bandwidth、all-to-all 通信开销。S → 1 时 total params 爆炸，HBM 不够。
2. **Compute budget 上限只有 1e21 FLOPs**（约 LLaMA-3 8B 训练量的 1/100）。外推到 16B + 数百 B tokens 风险存在。
3. **SQuAD 反例提示**：reasoning-heavy 下游任务可能不遵循"越稀疏越好"。本文承认 "models with higher sparsity transfer more poorly compared to denser models"。
4. **没有研究 granularity G**：sparsity 和 granularity 是正交维度（本文与 Krajewski 互补）。
5. **没有控制 routing 类型**（dropless vs token-choice）。
6. **没有 instruct/SFT 阶段实验**。
7. **只验证了 standard MoE，不含 shared expert**（DeepSeekMoE 风格）。
8. λ exponent 的数值未在主文给出，需查附录 F Table 3。
