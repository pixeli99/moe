# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

- **arXiv**: 2506.13585
- **机构**: MiniMax (150+ 作者)
- **发表时间**: 2025-06-16

## TL;DR

MiniMax-M1 是 MiniMax-Text-01 上做大规模 RL post-training 得到的 reasoning model，**架构完全继承 01：456B / 45.9B / 32 experts / top-2 / Lightning ×7 + Softmax ×1 hybrid**。核心新贡献是 **(1) 在 hybrid-attention 上首次成功做大规模 RL；(2) 提出 CISPO 算法 — clip importance sampling weight 而不是 token update**。释放 40K 和 80K thinking budget 两个版本，1M input native，整个 RL 训练在 512×H800 上 3 周完成，租赁成本 \$534,700。M1 主要意义在长上下文 reasoning 与 RL 训练效率，对 16B 设计的直接启示有限。

## 关键架构配置

完全继承 MiniMax-01（见 13_minimax_01.md），不复述。简表：

| 参数 | 数值 |
| --- | --- |
| Total / Active | 456 B / 45.9 B |
| Layers | 80 |
| Hidden | 6144 |
| Experts / top-k | 32 / 2 |
| Hybrid pattern | 7 × Lightning + 1 × Softmax |
| Native input context | 1 M |
| Thinking budget | 40 K / 80 K (两版本) |
| RL 资源 | 512 × H800，3 周 |
| RL 成本 | \$534,700 |

## 核心方法 / 创新点

### 1. CISPO (Clipped Importance Sampling Policy Optimization)

PPO/GRPO 在 long CoT 上的痛点：很多 **反思类 token**（"wait"、"actually"、"let me reconsider"）天然 probability 低，PPO 的 ratio clip 会把它们的梯度直接砍掉，导致学不到反思行为。

CISPO 的解法：**裁剪 importance ratio 而不是 token update**：

```
J_CISPO(θ) = clip( r_{i,t}(θ), 1-ε_low, 1+ε_high ) · Â_{i,t} · log π_θ(o_{i,t})
```

- 所有 token 的梯度都保留（`log π_θ` 项不被 mask），只是大幅偏离的 token 的 ratio 被截断到 `[1-ε_low, 1+ε_high]`。
- 反思类低概率 token 不再被丢弃，模型能学到 reflective behavior。
- 实验：在 Qwen2.5-32B 上 CISPO 比 DAPO 快 **2×**。

### 2. Hybrid Attention 是 RL 训练的天然加速器

对 long CoT RL，rollout / generation 阶段占了大部分时间。Lightning Attention 在 100K 输出时 FLOPs = DeepSeek-R1 的 25%，所以同样预算下能跑更多 rollouts、更长 thinking trajectory。这是 M1 能用 \$500K 训出来的关键。

### 3. Staged Window Expansion RL

输出长度课表：**48K → 56K → 64K → 72K → 80K**，逐步扩展，每阶段保证训练稳定后再扩。40K 版本是 80K 训练中间的 checkpoint。

### 4. RL 数据 & Rewards

- **Rule-based 验证**（~153K 样本）：
  - 数学：~50K 竞赛题（AIME / IMO 级别）
  - 逻辑：~53K 合成样本，来自 SynLogic 框架，覆盖 41 类任务
  - 竞赛编程：~30K
  - SWE：几千个真实 GitHub issue，execution-based reward（sandbox 运行单测）
- **Model-based rewards**（~25K 样本）：
  - STEM / factual QA（有 ground truth）
  - 开放式任务（instruction following / creative writing）用 Generative Reward Model，5 级评分
- **课程**：先纯 reasoning + rule-based，再逐步混入 general domain。

## 训练 & 系统细节

- **基础模型**：MiniMax-Text-01（已经做过 SFT + cold-start CoT）
- **总 RL 时长**：3 周，512 × H800
- **总成本**：\$534,700（租赁价）
- **CISPO 加速**：相比 DAPO 在等价 reward 下省 ~50% 步数
- **Inference 端**：hybrid-attention 让 80K 输出在合理延迟内可服务

## 关键消融与结果

主要数字（M1-80k）：

| Benchmark | M1-40k | M1-80k | DS-R1-0528 |
| --- | --- | --- | --- |
| AIME 2024 | 83.3 | 86.0 | 91.4 |
| SWE-bench Verified | 55.6 | 56.0 | 57.6 |
| TAU-bench (airline) | 60.0 | 62.0 | 53.5 |
| OpenAI-MRCR (128k) | 76.1 | 73.4 | - |
| OpenAI-MRCR (1M) | 58.6 | 56.2 | - |
| LongBench-v2 | 61.0 | 61.5 | - |

- 长上下文（MRCR / LongBench-v2）上自称超过 o3 与 Claude 4 Opus。
- AIME 上略弱于 DeepSeek-R1-0528（91.4 vs 86.0）。
- 在工具使用 / agent 任务（TAU-bench）上明显领先。

CISPO 消融：在 Qwen2.5-32B 控制变量实验里 2× 快于 DAPO，且最终 reward 更高。

## 对 16B MoE 设计的启示

**M1 对 16B 设计的主要启示是 RL/post-training 而非 pre-training architecture。**

- **CISPO 值得直接借鉴**：
  - 是 RL 算法层面的改进，与模型规模 / 架构正交。
  - 对 reasoning / long-CoT 友好；如果 16B 模型规划走 reasoning RL 路线，CISPO 优于 GRPO/DAPO。
  - 但要注意：CISPO 的优势在 long CoT 反思类 token 上最明显，短输出 / instruction following 任务上差距不大。
- **不建议照搬 hybrid attention**：理由同 MiniMax-01 报告。16B 不需要为 1M context 做架构妥协。
- **可参考的长上下文 RL 课程**：staged window expansion (48K → 80K) 是稳健的工程做法。
- **rule-based + model-based 混合 reward 设计**：是当前 reasoning RL 的标准做法，可借鉴 SynLogic 的逻辑题合成思路。
- **重要 caveat**：M1 是 **post-training** 论文，pre-trained backbone 是 01；如果 16B 没有一个对应的强 base model，先不要急着上 RL。

## Caveats / 局限

- 与 DeepSeek-R1 在纯数学（AIME）上仍有差距，原因不完全清楚（base model 数据 / scale / RL 步数）。
- CISPO 的论文消融主要在 Qwen2.5-32B 上做，到 MoE-456B 上的 scaling 行为是否完全保持论文没有详细 ablation。
- 长上下文输出 benchmark（MRCR）上 40k 反而略好于 80k，作者归因于训练阶段，但暗示长 budget 不总是更好。
- M1 的成本数字（\$534,700）只算 RL 阶段租赁，不含基座 01 的 pre-training（那部分估计百倍以上）。
- "1M context" 是 input，不是 output；output 上限是 thinking budget（40K / 80K）。
