# Better & Faster Large Language Models via Multi-token Prediction

- **arXiv**: 2404.19737 (v1: 30 Apr 2024)
- **机构**: Meta FAIR（主），CERMICS École des Ponts ParisTech（Gloeckle），LISN Université Paris-Saclay（Idrissi）
- **作者**: Fabian Gloeckle\*, Badr Youbi Idrissi\*, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve (\*equal contribution; † last authors Lopez-Paz, Synnaeve)
- **联系**: fgloeckle@meta.com, byoubi@meta.com

## TL;DR
本文主张：**让 LLM 训练时同时预测后续 n 个 token**（而非只下一个）能 (a) 提升 sample efficiency、(b) 让 13B 模型在 HumanEval 多解出 +12%、MBPP 多解出 +17%、(c) 通过 self-speculative decoding 让 inference 加速 3×。

**架构**：共享 transformer trunk + n 个**并行的** transformer-layer heads + 共享 unembedding。**所有 n 个 head 同时输出**，没有 causal chain（与 DeepSeek-V3 MTP 关键区别）。

**Loss**：n 个 head 各自独立 cross-entropy 平均（隐含权重 1/n 每个 head）。

**关键经验法则**：
- **n=4 在 7B/13B 上最优**（对 32k vocab、200B tokens 训练）。
- 小模型（≤ 1.3B）反而被 MTP 拖累——MTP 是"scaling enabler"，只有大模型才赚得到。
- 多 epoch 训练下增益保留，2-token MTP 在 500B tokens 上比 4-token 更稳定。

## 核心命题
1. **Multi-token prediction 显著改善 generative 任务（coding, summarization）**，但对 multiple-choice / NLL benchmark 几乎无影响。
2. **MTP 在大模型上才显现价值**——300M 模型用 MTP 反而比 next-token 差，6.7B 以上才稳定胜出。
3. **Self-speculative decoding 用 MTP 头作为草稿，速度 ~3× 文本、~2.7× code、~6.4× byte-level**。
4. **MTP 减少 train-test 分布 mismatch**：teacher-forcing 让模型只学短期模式；MTP 强制学长程依赖（实验：induction head 在 100M scale 就形成；polynomial reasoning out-of-domain 大幅改善）。

## 关键公式

### Loss formulation（公式 2）
对位置 t 同时预测后续 n 个 token：
$$
L_n = -\sum_t \log P_\theta(x_{t+1:t+n} \mid x_{1:t})
     = -\sum_t \sum_{i=1}^{n} \log P_\theta(x_{t+i} \mid z_{1:t}) \cdot P_\theta(z_{1:t} \mid x_{1:t})
$$

实际架构假设条件独立（given trunk 输出 z）：
$$
P_\theta(x_{t+i} \mid x_{1:t}) = \mathrm{softmax}(f_u(f_{h_i}(f_s(x_{1:t}))))
$$

其中：
- `f_s` = shared transformer trunk
- `f_{h_i}` = i-th independent prediction head（单个 transformer layer）
- `f_u` = shared unembedding matrix
- `P_θ(x_{t+1} | ...)` = next-token prediction head（即 i=1 head）

### Memory-efficient backward pass
- 朴素实现：n 个 head 的 logits（shape n×V）+ gradients 同时驻留 → memory O(nV+d)。
- **优化**：sequential backward through each head，累积 gradients on trunk，head 用完释放。Peak GPU 内存降到 O(V+d)。
- Pseudocode:
```
z = model.shared(x)
d = z.detach(); d.requires_grad = True
for i in range(n):
    p = model.heads[i](d)
    loss(p, y[i]).backward()  # release head i's grad
z.backward(gradient=d.grad)
```

### Inference
- 默认：丢弃 i>1 的 head，仅用 i=1（next-token head）做 vanilla autoregressive 解码。
- **Self-speculative decoding**：i=2..n 头并行生成 n-1 个候选 token，再由 main head 校验。block-wise parallel decoding (Stern et al. 2018) 或 Medusa-style tree attention 都兼容。

## 实验设置

### 主表（Table 1）：7B 模型
- **Tokenizer**: 32k tokens vocabulary，或 bytes (无 tokenizer)。
- **Training data**:
  - 313B bytes (0.5 epochs)
  - 200B tokens of code (0.8 epochs)
  - 1T tokens of code (4 epochs)
- **n 值测试**: n ∈ {1, 2, 4, 6, 8, 16, 32}。

### 多 scale 实验（Fig. 3）
- **Sizes**: 0.3B, 0.6B, 1.3B, 3B, 6.7B, 13B。
- **Training**: at least 91B tokens of code each。
- **Metric**: MBPP pass@1/10/100, HumanEval pass@1/10/100。

### 自然语言实验
- 200B 和 500B tokens natural language pretrain。
- 8 个 ROUGE summarization benchmarks + GSM8K (8-shot CoT)。

## 主要结论

### 1. 模型越大 MTP 收益越强（Fig. 1, Fig. 3）
- **MBPP pass@1**（相对 next-token baseline）：
  - 0.3B → -1.7（4-token 更差）
  - 0.6B → +0.0
  - 1.3B → +0.1
  - 3B → +2.0
  - 6.7B → +3.7
  - **13B → +4.5**（绝对值从 26.0 → 30.5）
- **HumanEval pass@1**：13B 上 +1.7（从 ~17 → ~19）。
- "Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale" (§3.1)。

### 2. 最优 n 值（Table 1，§3.4）
对 7B + 200B tokens of code：
- n=1: MBPP 30.0 / HE 22.8
- n=2: MBPP 30.3 / HE 22.2
- **n=4: MBPP 33.8 / HE 24.0**（最佳）
- n=6: MBPP 31.9 / HE 20.6
- n=8: MBPP 30.7 / HE 20.0

→ **n=4 是 32k vocab、code-heavy training 下的甜点**。

对 byte-level（vocab=256）训练：**n=8 才是最佳**。
"the optimal window size depends on input data distribution。"
"For the byte level models the optimal window size is more consistent (8 bytes)。"

### 3. 多 epoch 训练（§3.5）
- 1T tokens code (4 epochs) on 7B：
  - n=1: MBPP pass@1 = 40.7
  - **n=4: MBPP pass@1 = 43.1**（+2.4 绝对值）
- 但 pass@100 上 n=1 反而稍高 → MTP 让"第一次解题率"更高，"多次尝试解题率"略让步。

### 4. Self-speculative decoding（§3.2）
- 7B 4-token model on code：**3.0×** decode 加速，平均 2.5 token/3 suggestions accepted。
- 7B model on text：2.7× 加速。
- 8-byte prediction model（byte-level）：**6.4×** 加速。

### 5. NLP / multiple choice 反例（Fig. 5, §3.7）
- 6 个 multiple-choice NLP benchmarks 上：n=2 持平 baseline，**n=4 反而略差**。
- ROUGE summarization（generative）：n=2 和 n=4 都改善（Fig. 6）。
- GSM8K (math)：MTP 模型 pass@k 改善，因为 generation 是核心。
- **结论**：MTP 改善 **生成式** 能力，对 likelihood-based 评估几乎无影响。

### 6. Induction head / algorithmic reasoning（§4）
- Children's stories induction task：30M-100M scale 下 n=2 大幅胜过 n=1（induction head 形成更早）。
- Polynomial arithmetic (operations 1-10): MTP 显著提升 out-of-domain 泛化。
- **MTP 是 "learning of induction" 的诱因**——对长程依赖学习有内在效果，而不只是 inference speedup。

### 7. 训练成本中性
- "We propose a simple multi-token prediction architecture with no train time or memory overhead" (§2)。
- 通过 sequential per-head backward，peak memory 不增加。每个 head 是一个 transformer layer，但 trunk 共享 n-1 个 trunk layers → params/FLOPs 比同样 layer 数 baseline 持平。
- **公平比较**：n=4 模型 trunk 比 n=1 baseline 少 3 个 transformer layer（n-1 个 layer 被搬到 heads），保持总 params 相同。

### 8. 与 DeepSeek-V3 MTP 的差异

| 维度 | 本文 (Gloeckle et al.) | DeepSeek-V3 |
|------|------------------------|-------------|
| 架构 | n 个 head **并行**, 共享 trunk + unembedding | n=1 个 "MTP module"，causal chain |
| n 值 | 4 (typically) | 1 |
| Head 内容 | Transformer layer + projection | Transformer block 接收上一 token 的 representation |
| 数据流 | head_i 独立预测 x_{t+i} given z_{t} | MTP module 顺序生成: t+1, then t+2 给定 t+1 的 hidden state |
| Loss | n 个 head 各自 CE 求和 | 主 head CE + λ·MTP-module CE（小权重辅助） |
| Inference | self-speculative decoding | 推理时丢弃 MTP module，仅用主 head |
| 重点 | sample efficiency + inference speedup | 主要是 training auxiliary task |

DeepSeek-V3 的 MTP 是 **causal chain**（顺序，预测 t+2 时已经看到 t+1 的预测），而本文是 **parallel**（n 个 head 同时预测）。这意味着：
- 本文：head_i 之间 conditional independent（given trunk）。
- DeepSeek-V3：MTP module 利用了 chain rule（更接近真实分布，但训练更复杂）。

DeepSeek 选 D=1 是为了"轻量辅助"；本文选 n=4 是为了"主任务"。两个设计哲学不同。

## 对 16B MoE 设计的启示

### 是否对 16B MoE 加 MTP？
- **强烈推荐对 16B 加 MTP**：
  - 16B active 接近 6.7B-13B dense 等效，正好在 MTP 显效区。
  - MoE 模型推理时常 memory-bound，self-speculative decoding 收益更大（experts 加载是固定 overhead，多 token 摊薄）。
- **n 值选择**：
  - **若 16B = 16B-active**: n=4 跟随论文推荐。
  - **若 16B = 16B-total / 2B-active**: n=2 更稳健（2B active 接近论文 1.3B-3B 区间，n=4 风险大）。可起步 n=2 看曲线。
- **vocab 影响**：
  - 现代 LLM vocab 通常 64k-128k，比论文 32k 大 → 最佳 n 可能 *更小* (论文证明 byte-level 用大 n，token 越粗越用小 n)。建议 **n=2 起步，对比 n=4**。

### Loss 权重
- 论文用平均 (每 head 权重 1/n)。如果使用 DeepSeek-V3 风格的小权重辅助 (λ≈0.3) 也可行——更稳健但加速效果减弱。
- 建议：**main head 权重 1.0，i>1 head 权重 0.5–1.0**。

### 与 MoE 路由的交互
- MTP heads 应该 **是 dense layer 而不是 MoE**：因为 head 只跑一次（last layer 不需要 sparsity）。
- 但 trunk 全 MoE 是合理的。

### Self-speculative decoding 在 MoE 上的额外收益
- MoE inference 通常 memory-bound（experts 加载 + all-to-all）。一次 forward pass 出 4 个 token 摊薄掉这些开销 → MoE 上的 MTP 加速很可能 >3×。
- **强烈推荐保留 i>1 heads 用于 decoding**（不要训完丢掉）。

### 训练 budget
- 论文表明 MTP 在 **0.8-4 epochs** 范围内都有效。对 16B 训 ≥ 200B tokens 应充分。

### 与 reasoning 的关系
- MTP 改善 induction / polynomial arithmetic 这种 algorithmic 任务，但 multiple-choice benchmarks 不显著。
- 对 GSM8K（math）改善 pass@k，对 MoE-reasoning（结合 Yokota 2025）友好。
- **结合 Yokota 推荐的 K=4-8 routing + MTP n=2-4**：双管齐下改善 reasoning。

### 一句话推荐
**16B MoE 加 MTP n=2 或 n=4，shared trunk + n 个 dense head + 共享 unembedding；保留 heads 用于 self-speculative decoding，预期 inference 3× 加速**。

## Caveats / 局限

1. **n 的最佳值取决于 vocab + data distribution**，需要小规模实验确定。
2. **小模型 (≤1B) 会被 MTP 拖累**——不能盲目套用 recipe 到 < 3B 的辅助模型。
3. **Multiple-choice / NLL 评估上无收益甚至倒退**——如果主要 leaderboard 是 MMLU / Hellaswag，MTP 帮助有限。
4. **Pass@100 反而略低**——MTP 让答案多样性下降（论文 §3.5 数据）。
5. **未验证 reasoning RL stage**：本文都是 pretrain 阶段评估。
6. **未与 MoE 同时验证**：所有实验是 dense 模型。把 MTP 应用到 MoE 上需要额外验证。
7. **Head architecture**：本文用 transformer layer 作 head。head 用 MLP / linear 也可能 OK 但未测。
8. **多任务 epoch 后 n 应改小**：1T tokens (4 epoch) 下 n=4 仍是最佳，但 ROUGE 上 n=2 在 500B 后反超 n=4——长训练下 n=2 更稳。
