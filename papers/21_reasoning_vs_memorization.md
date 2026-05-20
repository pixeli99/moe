# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

- **arXiv**: 2508.18672 (v3: 1 Mar 2026, 原始 v1: 26 Aug 2025)
- **发表**: ICLR 2026 accepted
- **机构**: Institute of Science Tokyo（主），NII LLMC，Tohoku University，RIKEN
- **作者**: Taishi Nakamura, Satoki Ishikawa, Masaki Kawamura, Takumi Okamoto, Daisuke Nohara, Jun Suzuki, Rio Yokota
- **代码**: github.com/rioyokotalab/optimal-sparsity

## TL;DR
本文系统揭示了 **MoE sparsity 对 reasoning vs memorization 的影响不对称**：
1. **Memorization 任务（TriviaQA, HellaSwag）** 越稀疏越好：跟随 pretrain loss 单调改善。
2. **Reasoning 任务（GSM8K, GSM-Plus, HumanEval, MBPP）** 呈现**倒 U 型**：达到一定 active params 后，**稀疏反而损害 reasoning**——稠密 MoE 反胜稀疏 MoE。
3. **TPP（tokens-per-parameter）也呈现任务分化**：memorization 偏好低 TPP（多参数），**reasoning 在 TPP ≈ 20 附近达到峰值**。
4. **Top-k 大幅影响 reasoning**：同等 active params 下，**更大 top-k 一致优于小 top-k**。
5. **GRPO 后训 / Test-time compute 都不能弥补这个 reasoning gap**——所以 sparsity 必须在 pretrain 阶段就 jointly 优化 active FLOPs 和 TPP。

## 核心命题
1. **Active FLOPs 才是 reasoning 的瓶颈**——同 train loss 下，active 大的 MoE reasoning 更强。
2. **Reasoning 任务的 TPP 最优值 ≈ 20**（与 Chinchilla 的 dense 推荐相同），而 memorization 偏好 TPP << 20。
3. **存在 inverted-U 关系**：reasoning task accuracy 随 total params 先升后降；过参化（total 太大）+ 稀疏 → 反而伤害 reasoning。

## 关键公式 / 定义

### Sparsity（§1）
$$
\text{sparsity} = 1 - \frac{\text{Top-}k}{\text{Experts}}
$$
即 fraction of inactive experts。例如 E=64, k=8 → sparsity = 0.875。

### Density（图表 x 轴）
$$
\text{MoE Density} = \frac{k}{E}
$$
即 1 - sparsity。常用值：1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128。

### TPP（tokens per parameter）
$$
\mathrm{TPP} = \frac{D}{N_{\text{total}}}
$$
即训练 tokens 除以**总**参数数（注意：不是 active）。Chinchilla 推荐值 ≈ 20。

### Active FLOPs
近似 `6 · N_active · D`，即仅 active 参数参与 forward/backward。

## 实验设置

### 架构（§3.1）
- **Mixtral 风格**：Transformer + RMSNorm + SwiGLU + RoPE。
- **L = 16 layers**（部分 ablation 用 32 layers）。
- **每层 FFN 都是 dropless token-choice top-k 路由的 MoE**。
- FFN hidden dim = 2d。

### 扫描网格
- **Model width**: `d ∈ {512, 1024, 2048}`。
- **Experts per layer**: `E ∈ {8, 16, 32, 64, 128, 256}`。
- **Top-k**: `k ∈ {2, 4, 8, 16}`。
- 对 d=512 / 1024 跑了 E×k 全网格；d=2048 限制到 E ≤ 128。
- 总训练 model 数：几十到上百个（明确数字未给）。

### 训练
- **AdamW**，peak lr = 4e-4，2k step warmup + cosine。weight decay 0.1。
- **Load-balancing loss × 1e-2，router z-loss × 1e-3**。
- **训练 tokens 固定 = 125B**（这是关键设置——基本 Chinchilla-optimal for "active params"）。
- **数据**: 43B web text + 32B math + 49B STEM literature + 1B code。

### 评估
- **Reasoning**: GSM8K (4-shot), GSM-Plus (5-shot CoT), HumanEval, MBPP。
- **Memorization / QA**: TriviaQA (4-shot), HellaSwag (4-shot)。
- **Task loss**: cross-entropy on answer tokens only（concat prompt + answer）。

### Test-Time Compute & RL
- **TTC**: Self-Consistency on GSM8K，2^r 个采样 + majority vote。
- **RL**: GRPO on GSM8K training set。

## 主要结论

### 1. Train loss 单调下降，但 reasoning task loss 反弹（Fig. 1, 2）
- TriviaQA, HellaSwag: task loss 单调跟随 train loss。
- **GSM8K, GSM-Plus**: task loss 在 train loss 降到某阈值后**反向上升**（U 形）——继续 over-fit pretrain 反而毁掉 reasoning。
- 阈值随 active params 变低（更大 active → 阈值更晚到来 → 可以更稀疏才反弹）。

### 2. Inverted-U：density 影响 reasoning（Fig. 5, **关键 figure**）

固定 active params 扫描 density：
- **Memorization (TriviaQA, HellaSwag)**：density ↓ (更稀疏) → accuracy ↑ **单调**。
- **Reasoning (GSM8K, GSM-Plus)**：accuracy 沿 density ↑ **先升后降，呈现倒 U**。
- 当 active params 较小 (≤ 2^20 = ~1M)：稀疏仍有用。
- 当 active params 较大 (≥ 2^22)：**稠密 MoE 比稀疏 MoE 在 GSM8K 上多 10-20% 准确率**。

### 3. TPP ≈ 20 是 reasoning 的甜点（Fig. 7, **关键 finding**）
- 论文原话："accuracy peaks near TPP ≈ 20, and degrades when TPP is either too low—when models have too few parameters relative to tokens—or too high—when models have too few parameters relative to tokens" (typo in source: 第二个 too high 指 tokens 不够)。
- TriviaQA, HellaSwag: accuracy 随 TPP ↑ **单调下降**（即低 TPP/多 param 更好，memorization 性质）。
- GSM8K, GSM-Plus: **TPP ≈ 20 处有 peak**。
- **同一 TPP 下，更大 top-k 一致优于小 top-k**（reasoning tasks）。

### 4. Top-k 显著影响 reasoning（Fig. 5, Fig. 7）
- "models with larger top-k values consistently outperform those with smaller top-k on reasoning tasks"。
- top-k=2 → top-k=4 → top-k=8 → top-k=16 上 GSM8K 单调改善。
- 但 **top-k 在 memorization 上效果中性**——只要 active params 配齐了。
- "changing the k in top-k routing has a negligible effect if the number of active parameters is kept constant" - 适用于 train/test loss，不适用于 reasoning accuracy。

### 5. 编码任务（HumanEval, MBPP）和 reasoning 一致（Fig. 8）
- HumanEval, MBPP 也呈现 density-optima shift：active 大时稠密 MoE 更好。
- 这进一步支持"reasoning 类任务（math + code）需要 active FLOPs"。

### 6. GRPO 和 Test-Time Compute 不能弥补（Fig. 6, §3.5）
- GSM8K + TTC (Self-Consistency, k=2^r samples): 整体精度升，但 sparsity-induced gap 不闭合。
- GSM8K + GRPO post-training: 同样，gap 保持。
- 论文原话："neither Test-Time Compute nor GRPO mitigates the GSM8K performance drop that arises when total parameters increase. In other words, although both methods consistently improve overall performance, they do not eliminate the inverted U-shaped relationship between training loss and task accuracy"。

### 7. 没有统一的 scaling law 系数
- "The coefficients of the scaling laws are not universal" - 任务依赖性强。
- 没有给单一闭式 L(N, D, S) 公式，重点是 **task-conditional 趋势**。

## 对 16B MoE 设计的启示（**最重要的一节**）

### Sparsity (1/8 = 0.875) 是否合理？

**取决于目标**：
- **若 16B MoE 目标 = memorization-heavy（QA, world knowledge）**：可以更稀疏，**S = 0.9 – 0.94 合理（E=16 k=1 ~ E=32 k=2）**。
- **若 16B MoE 目标 = reasoning-heavy（math, code, agentic）**：**S = 0.875 偏稀疏，应考虑 S = 0.75 – 0.875**，即 E=8 k=2 或 E=16 k=4。
- **如果两者都要兼顾**：**S ≈ 0.875 是合理折衷**，但要配合更大的 top-k（k ≥ 4）。

### Top-k 选择
- **强烈建议 top-k ≥ 4**（最好 8）。这是 Yokota 2025 与传统 MoE (Mixtral 用 k=2) 最大的分歧。
- 推荐组合（同 1/8 sparsity = S=0.875）：
  - **E=8, k=1**: ❌ k 太小，reasoning 弱
  - **E=16, k=2**: △ 传统但 reasoning 普通
  - **E=32, k=4**: ✓ **推荐**
  - **E=64, k=8**: ✓ **推荐**（granularity 更细）
  - **E=128, k=16**: ✓✓ 若内存允许，进一步细化

### TPP（tokens per total parameter）
- 16B-total 想 reasoning-strong → **训 16B × 20 = 320B tokens（TPP=20）**。
- 16B-total 想 memorization-strong → **训 16B × 5 = 80B tokens**（TPP=5，但风险是 reasoning 下降）。
- **现代实践通常 TPP=15-25**：训 240B-400B tokens。

### Active params 选择
- **若 reasoning 是首要**：active ≥ 3B（按 16B-total × 1/4 ~ 3-4B）。
- 1/8 sparsity 下 active = 2B 偏低；**升级到 1/4 sparsity (S=0.75) 或 increase total 到 32B-total 维持 4B active** 是更安全方案。
- 单纯 1/8 sparsity 在 reasoning 上是"够用但不强"。

### Granularity 与本文的关系
- Yokota 用 hidden dim = 2d, 没改 FFN 内部 granularity（即 G=1 in Krajewski terminology）。
- 本文的 top-k=8/16 + E=64/128 实际上**就是 fine-grained MoE 的另一种实现**——把"big expert split"替换成"小 expert + 多激活"。
- 结合 Krajewski (G=8-16 optimal) + Yokota (top-k=4-8 optimal)，**fine-grained MoE 是 reasoning 友好的设计**。

### Shared expert?
- 论文明确**排除了 shared expert**（§4 Limitations）："we excluded shared experts, as prior work reports mixed or negative results, and our preliminary tests indicated no meaningful performance changes when active and total FLOPs were matched"。
- **结论**：shared expert 不是关键变量，可加可不加。DeepSeek-V3 用 shared，性能好；Mixtral 不用，性能也好。

### MoE 层的位置
- 本文所有 layer 都 MoE。如果资源紧张，前几层可以保留 dense（如 first 1-2 layers）。

### 一句话推荐（**对 16B MoE 的最终建议**）
**16B-total / 2B-active spec（1/8 sparsity）只在 memorization 友好；若要 reasoning 强，应改为 16B-total / 4B-active（1/4 sparsity） 或保持 1/8 但 top-k=8 + E=64+；训 320B tokens (TPP=20)；不要期待 RL/CoT 弥补 pretrain 不足的 reasoning**。

### 与 Abnar 2025 的张力
| 维度 | Abnar 2025 | Yokota 2025 |
|------|------------|-------------|
| 推荐 sparsity 方向 | S\* → 1 as N, C 增长 | S\* 取决于任务；reasoning 偏保守 |
| 评估 | Pretrain loss + few-shot | 多任务 task loss + RL/TTC |
| 限制 | SQuAD 例外提示 reasoning 弱 | 直接定量 reasoning 损害 |

**调和**：Abnar 是"pretrain loss + general downstream"视角，Yokota 是"reasoning-specific"视角。对 16B：
- 若 model 主要做 base 预训练（chat 后端 / world knowledge），跟 Abnar 走 → sparser。
- 若 model 主要做 reasoning agent（math / code / agentic），跟 Yokota 走 → denser + bigger top-k。

## Caveats / 局限

1. **训练 token 数固定 125B**：对 d=512/1024 接近 Chinchilla optimal，但 d=2048 + E=128 这种大模型相对欠训。外推到 16B + 300B+ tokens 时趋势可能减弱（论文承认）。
2. **没有 shared expert**——所以不能直接 informing DeepSeekMoE 风格设计。
3. **只用了 16 层**（部分实验 32 层）：现代 LLM 更深，深度与稀疏的交互未充分研究。
4. **QK-norm 未启用**：因为 Mixtral 风格。Qwen3-MoE 用 QK-norm，行为可能略不同。
5. **GRPO 只 fine-tune on GSM8K train**：可能 over-fit；更广泛 RL（如 R1 风格 multi-domain）效果未知。
6. **Test-time compute 只测 Self-Consistency**：MCTS / best-of-N + verifier 等其他 TTC 没测。
7. **TPP=20 是基于 dense Chinchilla 的传统**：在 MoE 上"完美" peak 可能略偏 15-25 区间。
8. **没有验证 instruction tuning**：纯 pretrain 视角；instruct 后的 reasoning 可能 partially 弥补 sparsity gap。
9. **本文的 reasoning benchmarks 主要是 math/code**：MMLU 风格"知识 + 浅推理"未明确归类。
