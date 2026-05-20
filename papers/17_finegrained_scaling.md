# Scaling Laws for Fine-Grained Mixture of Experts

- **arXiv**: 2402.07871 (v1, 12 Feb 2024)
- **机构**: University of Warsaw / IDEAS NCBR / IPPT PAN / Nomagic / TradeLink
- **作者**: Jakub Krajewski\*, Jan Ludziejewski\*, Kamil Adamczewski, Maciej Pióro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Król, Tomasz Odrzygóźdź, Piotr Sankowski, Marek Cygan, Sebastian Jaszczur\* (\*equal contribution; lead by Jaszczur, s.jaszczur@uw.edu.pl)
- **代码**: github.com/llm-random/llm-random

## TL;DR
本文系统建立了带 **granularity G** 维度的 MoE scaling law `L(N, D, G)`。核心贡献：
1. 引入 granularity G = d_ff / d_expert 作为新的一阶超参数，G > 1 即 fine-grained MoE（专家变小、专家数变多）。
2. 拟合出三变量 scaling law，并据此推出 compute-optimal 的 (N, D, G) 三元组。
3. 推翻 Clark et al. (2022) 的悲观结论："MoE 和 dense 的 efficiency gap 不仅不会闭合，反而随规模越拉越大"——在 1e25 FLOPs 时 MoE 比 dense 至少节省 40×。
4. **常用的 G=1（专家大小=FFN）几乎在所有 compute budget 下都不是最优**。

## 核心命题
1. **MoE 在任意 compute budget 下都优于 dense Transformer**（前提是 N、D、G 都 compute-optimal 取值），与 Clark et al. 2022 结论相反。
2. **Granularity G 应随 compute budget 单调增长**：从 small (1B params) 用 G=8/16，到 1T params 规模用 G=64。
3. **MoE 对训练 token 数 D 的 scaling exponent (β=0.147) 大于 dense (β=0.127)**，意味着 MoE 需要训更多 token 才能充分利用容量；但一旦给够，scaling 优势就持续放大。

## 关键公式

### 主 scaling law（公式 9，全文核心）
$$
\mathcal{L}(N, D, G) = c + \left(\frac{g}{G^{\gamma}} + a\right) \cdot \frac{1}{N^{\alpha}} + \frac{b}{D^{\beta}}
$$

其中 N = active 参数数（非 embedding），D = 训练 tokens，G = granularity。

### 拟合系数（Table 1，关键数字）
| Model | a | α | b | β | g | γ | c |
|-------|------|-------|------|-------|------|------|------|
| MoE   | 18.1 | 0.115 | 30.8 | 0.147 | 2.1 | 0.58 | 0.47 |
| Dense | 16.3 | 0.126 | 26.7 | 0.127 | -   | -    | 0.47 |

- Validation RMSE = 0.019。
- 注意 dense 的 N-exponent (0.126) 比 MoE (0.115) 略陡，但 D-exponent (0.127) 比 MoE (0.147) 浅——所以 MoE 在长训练下 token 利用率更高。
- 关键的 granularity 项：`g · G^(-γ) = 2.1 / G^0.58`，G 从 1 → 8 大约把这项削掉 3.4×，从 1 → 64 削掉 12×。

### Granularity 的定义（公式见 §4）
$$
G = \frac{d_{ff}}{d_{expert}}, \quad N_{expert} = G \cdot E, \quad E = \frac{N_{MoE}}{N_{ff}}
$$
- `E` = expansion rate = 总 MoE 参数 / 等效 dense FFN 参数（≈ 总专家数 / 激活专家数）。
- `N_expert` = 物理专家数 = G × E。
- **保持 G 增加时 active 参数不变**：因为每个专家变小到 1/G，但 token 同时被路由到 G 个细粒度专家。
- 论文里所有实验固定 **E = 64**。

### FLOPs 公式（公式 10）
$$
F = (12 \cdot d_{model}^2 \cdot c_f + d_{model} \cdot E \cdot G \cdot c_r) \cdot D \cdot n_{blocks}
$$
- 第一项是 attention + FFN 计算，第二项是 routing（与 EG 成正比）。
- 因此 G 越大 routing 开销越大；这是 G 不能无限大的原因。
- `d_model = 64 · n_blocks`（依 Kaplan 假设）。

## 实验设置
- **架构**: decoder-only Transformer，FFN 替换为 MoE。
- **Model 规模**: 129M – 3.7B 参数（active）。表 2 列出 64×3M 到 64×49M 的 scaling 网格。
- **训练 tokens D**: 16B – 130B（主网格用 16B / 33B / 66B）。
- **Granularity 范围**: G ∈ {1, 2, 4, 8, 16}（log 间隔）。
- **E 固定 = 64**。Top-K 实现上是 Expert Choice 类似的设定；增加 G 时一个 token 被路由到 G 个 fine-grained expert，激活参数总数不变。
- **数据集**: C4。
- 100+ 个独立训练 run，loss 用 Huber loss (δ=0.1) + BFGS 拟合。

## 主要结论

### 1. Compute-optimal 超参数（**Table 2，最重要的一张表**）
| N (active) | D (tokens) | G_opt | FLOPs | Loss |
|-------|--------|-----|------|------|
| 64×100M | 4.37B | **8** | 2.95e18 | 3.133 |
| 64×1B | 28.94B | **16** | 1.93e20 | 2.491 |
| 64×3B | 72.90B | **16** | 1.41e21 | 2.245 |
| 64×7B | 137.60B | **32** | 6.46e21 | 2.076 |
| 64×70B | 941.07B | **32** | 4.16e23 | 1.694 |
| 64×300B | 2.96T | **64** | 5.69e24 | 1.503 |
| 64×1T | 7.94T | **64** | 4.97e25 | 1.367 |

→ G_opt 大体随 compute 单调 +3 dB 一档。1B–10B active 规模 → G ≈ 16；70B–300B → G ≈ 32–64。

### 2. IsoFLOP 对比（Fig. 1）
- 1e20 FLOPs：MoE compute-optimal vs Dense 需要 **20×** budget 才能追平 quality。
- 1e25 FLOPs：差距扩大到 **>40×**。
- **关键定量结论原话**：*"a compute-optimal MoE model trained with a budget of 10^20 FLOPs will achieve the same quality as a dense Transformer trained with a 20× greater computing budget, with the compute savings rising steadily, exceeding 40× when budget of 10^25 FLOPs is surpassed"*。

### 3. G=1（standard MoE）几乎从不最优
- 论文原话：*"the common practice of setting the size of experts in MoE to mirror the feed-forward layer is not optimal at almost any computational budget"* (Abstract)。
- Fig. 5(b) 显示：N=64×7M, D=66B 下，G=8 是 wall-clock 最优；G=16 由于 routing 开销反而变慢。

### 4. MoE 反而需要更长训练
- β_MoE = 0.147 > β_dense = 0.127 → 同样 N、加 token，MoE 收益更大。
- 但短训练（D 小）下 MoE 可能反而输给 dense（"under-trained" 阶段）。这是 Clark 2022 错的根因——他们只看了短训练。

### 5. 极端 G 的拐点
- 当 G=64 + E=64 + d_model=256 这种 corner case 下，routing 参数已超过 expert 真实参数，性能开始下降（Discussion §7）。
- 实际 16B-scale 用 G ≤ 64 + 合理 d_model 都不会触发。

## 对 16B MoE 设计的启示（最重要的一节）

**前提**：若 16B = 16B total parameters（约 1/8 sparsity → ~2B active）；或 16B active → ~128B total。两种解读结合给出建议。

### 推荐 G 区间
- 对 **2B active**（即"16B-total spec"）：根据 Table 2 的 64×1B → 64×3B 区间 → **G_opt ≈ 16**（compute-optimal 区域）。即每个 expert 大小为 d_ff/16。
- 对 **7B active**（如稍大规模）：**G ≈ 16–32**。
- 16B-total 在 100B–500B tokens 训练下：**G = 8 仍是稳妥选择**（routing 开销可控），G=16 是激进但仍合理的选择。

### Sparsity = 1/8 是否合理？
- 1/8 sparsity 对应 K/E = 1/8，比如 K=2, E=16 → sparsity = 0.875；或 K=8, E=64 → sparsity = 0.875。
- 本文用 E = 64 fixed，对应的 sparsity 较高，与 Abnar 2025 的"S* → 1 as budget grows"方向一致。
- 但 1/8 sparsity（即 12.5% 激活）**比 Mixtral 8x7B 的 1/4 sparsity 更激进**。本文未直接给出 E 的最优值（只在附录 D 比较 E=16 vs 64），需要结合 Abnar 2025 来选 sparsity。
- **结论**：1/8 sparsity 在 fine-grained 框架下完全合理，**但前提是要配合 G > 1**——否则"1/8 sparsity + 8 个大 expert + Top-1"会留在 G=1 的低效区。建议 **(E=8, K=1, G=8)** 或 **(E=16, K=2, G=8)** 这种组合。

### Top-K 的选择
- 本文实质上是 Top-K=G（fine-grained 时 token 同时路由到 G 个小 expert）。即"用 8 个 small experts 替换 1 个 big expert"。
- 不要被 "Top-K=1 vs Top-K=2 of original experts" 的传统讨论框住——真正的语义是"激活参数总量 = N_act 不变，只是 active 被切碎"。

### 训练 tokens 怎么定
- Table 2 中 64×1B 用 ~29B tokens（TPP=29）；64×7B 用 ~138B tokens（TPP=20）；64×70B 用 941B tokens（TPP=13.4）。
- 16B-total scale 应训 **≥ 200B tokens**，最好 300–500B（contemporary 实践远超 Chinchilla optimal）。

### 是否需要 shared expert?
- 本文**没讨论 shared expert**（DeepSeekMoE 那种 architecture）。本文只研究 G 维度，正交于 shared/non-shared 设计。可与 DeepSeekMoE 的 shared expert 设计组合。

### 一句话推荐
**对 16B MoE，从 (E=16, K=2, G=8) 或 (E=8, K=1, G=8) 起步，训 200B+ tokens；scaling 上去时把 G 升到 16–32**。

## Caveats / 局限

1. **E = 64 是固定的**——E 维度没有充分扫描；附录 D 只验证 E=16 趋势一致。生产中 E 通常更小（8、16）以省 memory，需外推。
2. **没有考虑 memory / communication cost**：G 增大伴随 expert 数变多，all-to-all 通信开销在大集群上可能成为瓶颈。论文承认 "the precise selection of hyperparameters should be made considering [training setup, hardware, implementation]"。
3. **Routing 用 token choice / expert choice 哪种**：论文用类似 expert choice 路由，K_eff 隐含。Top-K(token-choice) 在 inference 时是固定的，与论文设定可能略不同。
4. **未验证 G > 64**：当 G·E·d_model · n_blocks > total active parameters 时 routing 主导，公式外推失败。
5. **只测了 16B–130B tokens**，外推到 1T+ tokens 时 β 是否仍准确未知（与 Hoffmann 一致风险）。
6. **没有 instruct/RL 数据**：纯 pretraining loss；不保证 downstream 表现按比例 transfer。
