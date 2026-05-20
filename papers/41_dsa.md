# DeepSeek-V3.2: DSA (DeepSeek Sparse Attention) + Scaled RL

- **arXiv**: 2512.02556 (v1, 2 Dec 2025)
- **机构**: DeepSeek-AI
- **核心创新**: DSA = sparse attention on top of MLA，把 long-context attention 从 O(L²) 降到 O(Lk)
- **同源**: V3.2-Exp (2025-09 实验版) → V3.2 (2025-12 正式版) → V3.2-Speciale (gold-medal reasoning variant)

## TL;DR

V3.2 的"唯一架构修改"是在 V3.1-Terminus 基础上**通过 continued training 加入 DSA**。其他全部（MoE 配置、MLA、61 层、256 routed expert、sigmoid+ALF）**完全继承 V3.1**。

DSA 由两个组件构成：
1. **Lightning Indexer** —— 小头数 + ReLU + FP8 实现，每对 (query, key) 算一个 index score，O(L²) 但常数极小
2. **Fine-grained Token Selection** —— 对每个 query token 只取 top-k=2048 个 preceding tokens 做完整 attention

**收益**：
- Long-context (128K) **prefill 成本降 3×**, **decoding 成本降 5×** (Figure 3)
- 无质量损失（V3.1 vs V3.2 在短/长 context 都打平）
- 反而 **AA-LCR (long-context reasoning) +4 pt**（独立第三方评测）

## 核心命题

1. **Long-context 注意力不需要全连接** —— top-2048 个 preceding tokens 足够覆盖 reasoning 所需 retrieval
2. **可以用一个轻量级 "indexer head" 学到 "哪些 token 该被关注"** —— 不用人工先验（如 sliding window）
3. **DSA 可以通过 continued training 加到已有 dense attention 模型上** —— **不需要从头训**
4. **MoE RL 训练需要 "Keep Routing"** —— 训练-推理之间 expert 选择不一致会让 RL 不稳

## Lightning Indexer 数学（Eq. 1）

$$
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \text{ReLU}\big(q^I_{t,j} \cdot k^I_s\big)
$$

其中：
- $I_{t,s}$ = query token $t$ 对 preceding token $s$ 的 index score
- $H^I$ = indexer head 数（"small"，论文未公开数值，估 4-8）
- $q^I_{t,j} \in \mathbb{R}^{d^I}$ = 从 query token $h_t$ 导出的第 $j$ 个 indexer query
- $k^I_s \in \mathbb{R}^{d^I}$ = 从 preceding token $h_s$ 导出的 indexer key
- $w^I_{t,j} \in \mathbb{R}$ = 第 $j$ head 的标量权重
- **ReLU** 不是 softmax —— "for throughput consideration"
- **FP8 实现** —— indexer 是 main bottleneck 时这点非常关键

**直觉**：indexer 像个"前置 attention"，但用 ReLU + 标量加权 + 极少头 → 计算量是 main attention 的 1/100 以下。

## Token Selection (Eq. 2)

$$
u_t = \text{Attn}\big(h_t, \{c_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\big)
$$

- $\text{Top-k}(I_{t,:})$ = 取 indexer score 最大的 k 个 preceding tokens
- 训练时 **k=2048**
- 然后只对这 k 个 token 做完整 attention（MLA 风格）

→ 关键：**$c_s$ 是 MLA 的 KV latent**（512 维 compressed KV），不是 raw KV。所以 DSA 是 **MLA 之上的 sparse attention**。

## DSA × MLA 集成

> "For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA for DeepSeek-V3.2. At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency. Therefore, we implement DSA based on the MQA mode of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token."

→ 即 DSA 实现成 **MQA over MLA**：所有 query head 共享同一组 KV latent（每个 token 只取 2048 个 latent vector）。

→ 这跟原版 MLA（每 head 自己一组 KV）不同 — **DSA 强制 MQA 模式**。看附录 A 有解释。

## 训练过程（continual from V3.1-Terminus）

V3.2 不是从头训的，而是从 V3.1-Terminus 128K checkpoint 上 continued pre-training。

### Stage 1: Dense Warm-up（初始化 indexer）

- **目标**：让 indexer 输出对齐 main attention 的注意力分布
- **方法**：freeze 所有 model parameter，**只训 lightning indexer**
- **训练信号**：KL divergence between target (main attention 求和后 L1-normalized) 和 indexer 的 softmax

$$
\mathcal{L}^I = \sum_t D_{\text{KL}}\big(p_{t,:} \| \text{Softmax}(I_{t,:})\big)
$$

- LR = **1e-3**（注意：训 indexer 用大 LR）
- **1000 steps × 16 sequences × 128K tokens = 2.1B tokens**
- 单次成本 ~ $30K（H800）

### Stage 2: Sparse Training（联合训）

- **目标**：让整个模型适应 sparse pattern
- **k = 2048** tokens per query
- 同时仍然 align indexer 到 main attention（但只看 selected tokens）：

$$
\mathcal{L}^I = \sum_t D_{\text{KL}}\big(p_{t,S_t} \| \text{Softmax}(I_{t,S_t})\big), \quad S_t = \{s | I_{t,s} \in \text{Top-k}\}
$$

- LR = **7.3e-6**（注意：1.4× 小于 V3.1 的 typical LR；refer 到训了很多 token 之后的 small LR）
- **15000 steps × 480 sequences × 128K tokens = 943.7B tokens**
- **Indexer 梯度从 main model 解耦** —— 只更新 indexer 自己用 KL loss

→ 总共 ~946B tokens 的 continued training。

## 推理成本对比（Figure 3）

H800 cluster, $2/GPU-hour 计费：

| Context | V3.1 Prefill ($) | V3.2 Prefill ($) | Reduction | V3.1 Decode ($) | V3.2 Decode ($) | Reduction |
|---|---|---|---|---|---|---|
| 0K | $0.05/M | $0.05/M | 1× | $0.05/M | $0.05/M | 1× |
| 32K | $0.20/M | $0.10/M | 2× | $0.60/M | $0.12/M | 5× |
| 64K | $0.40/M | $0.15/M | **2.7×** | $1.10/M | $0.20/M | **5.5×** |
| 128K | $0.65/M | $0.22/M | **3×** | $2.10/M | $0.40/M | **5.3×** |

→ **Decoding 比 prefill 收益更大**（生成时每 token 都要 attend 所有过去 → DSA 的 Lk vs L² 优势放大）

## 复杂度分析

| 阶段 | V3.1 Attention | V3.2 DSA |
|---|---|---|
| Main attention | $O(L^2)$ per token | $O(Lk)$ per token, k=2048 |
| Indexer | – | $O(L^2)$ per token, **但 head 数极少**（small constant） |
| 总 | $O(L^2)$ × big constant | $O(Lk)$ × big + $O(L^2)$ × tiny |

→ Crossover：当 $L > 2048$ 时 main attention 主导，DSA 收益启动；$L > 32K$ 时显著。

## V3.2 vs V3.1 Quality 对照

论文 §2.2 "Parity Evaluation"：
- **Standard benchmark**：V3.2 vs V3.1 平打（无降级）
- **Human preference (ChatbotArena, 10 Nov 2025)**：closely matched
- **AA-LCR (long-context reasoning, 独立评测)**：**V3.2 +4 pt over V3.1** 在 reasoning mode
- **Fiction.liveBench**：consistently outperforms V3.1

→ **DSA 不仅不损质量，反而提升 long-context reasoning**。这是出乎意料的"sparse 比 dense 还强"现象。

**为什么？** 论文没解释，但我的猜测：
- Dense attention 在 long context 容易被 "noise tokens" 稀释
- DSA 强制选 top-k → 隐式去噪
- 类似于人脑只关注重要细节，不试图记住每个字

## MoE RL 训练的关键创新（§3.1）

V3.2 的另一半价值是 **scaled RL framework**。后训计算量已经超过预训的 10%。

### Scaling GRPO 的四个补丁

#### 1. Unbiased KL Estimate（Eq. 7）

修正 K3 estimator (Schulman 2020) 的 bias：

$$
D_{\text{KL}}\big(\pi_\theta(o_{i,t}) \| \pi_{\text{ref}}(o_{i,t})\big) = \frac{\pi_\theta}{\pi_{\text{old}}} \left( \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1 \right)
$$

→ 当 sampled tokens 的 $\pi_\theta \ll \pi_{\text{ref}}$ 时，K3 给极大梯度 → 不稳。Unbiased version 修正这个。

#### 2. Off-Policy Sequence Masking（Eq. 8-9）

引入 binary mask M：

$$
M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0, \ \frac{1}{|o_i|} \sum \log\frac{\pi_{\text{old}}}{\pi_\theta} > \delta \\ 1 & \text{otherwise} \end{cases}
$$

→ 把"高 off-policy + 负 advantage"的序列 mask 掉。**只 mask 负 advantage 的**（论文强调）—— 模型从自己错误中学习，但 highly off-policy negative samples 会 destabilize。

#### 3. **Keep Routing** —— MoE 专用稳定技巧

> "Mixture-of-Experts (MoE) models improve computational efficiency by activating only a subset of expert modules during inference. However, discrepancies between inference and training frameworks, compounded by policy updates, can result in inconsistent expert routing during inference and training even for identical inputs. Such inconsistency induces abrupt shifts in the active parameter subspace, which destabilizes optimization, which exacerbates off-policy issues. To mitigate this, we preserve the expert routing paths used during sampling in the inference framework and enforce the same routing paths during training, ensuring that identical expert parameters are optimized."

→ 这是 **DeepSeek 自 V3-0324 起就用的关键技巧**，但首次明确文档化。

→ **对你 MoE RL 设计**：必须 **保存 rollout 时的 expert routing path**，训练时 replay 同样的 routing。否则梯度直接打到错误的 expert 子集上 → 完全打乱稀疏激活的语义。

#### 4. Keep Sampling Mask

→ Top-p / top-k truncation mask 也要 preserve from rollout，避免 truncation 边界处的训练-推理不一致。

## V3.2-Speciale: 极端 reasoning 变种

V3.2-Speciale 是 "**重新 RL，去掉 length penalty**" 的 V3.2 变种：
- IOI 2025 / ICPC WF / IMO 2025 / CMO 2025 全部 **gold medal**
- AIME 2025 = **96.0** (vs V3.2 93.1, GPT-5-High 94.6, Gemini-3.0-Pro 95.0)
- 但 output token 数极大：AIME 23K tokens vs Gemini 15K

→ Speciale 用 reasoning 长度换准确度。**实用部署用 V3.2 standard 的 length-penalized 版**。

## V3 系列演进总览

| Version | 发布 | 主要变化 |
|---|---|---|
| V3 | 2024-12 | sigmoid+ALF + MLA + MTP D=1 + 256 experts 起点 |
| V3.1 | 2025-08 | hybrid thinking mode, UE8M0 FP8, +840B continual |
| V3.1-Terminus | 2025-11 | V3.1 最终 stable checkpoint, 128K ctx |
| **V3.2-Exp** | **2025-09** | **DSA 第一版实验** |
| V3.2 | 2025-12 | DSA + scaled RL + agentic task synthesis |
| V3.2-Speciale | 2025-12 | reasoning 极端变种 |

→ **架构层面 V3-V3.2 几乎不变**（除 DSA）。**DeepSeek 团队明显选择"架构 freeze + post-training 极致优化"路线**。

## 与你 100B+ 设计的关系

| 维度 | 100B+ Profile | V3.2 DSA | 建议 |
|---|---|---|---|
| Long-context (> 32K) | 必须 | DSA 是当前最优 | **如果你做 long-ctx → DSA 是 next-gen 必备** |
| Attention 架构 | 默认 MLA / 重 GQA | MLA + DSA | DSA 加在 MLA 之上 |
| RL on MoE | 大概率要做 | Keep Routing + Unbiased KL | **MoE RL 必加 Keep Routing** |
| Continued training | 可能 | DSA 仅 946B tokens 加上去 | **DSA 不需要从头训，可以 retrofit** |
| FP8 | 1T 段位用 | indexer FP8 实现 | 没影响 |

### 对你 100B 设计的两个直接行动

1. **如果你做 reasoning model → 100B base 训完后做 DSA continual** —— 成本 ~$30K，几乎免费降 long-ctx 推理 3-5×
2. **如果你做 RL post-training → 必须 Keep Routing** —— V3 团队从 V3-0324 起就这样做了，是 MoE RL 稳定的关键

## Settled vs Open

### Settled
- DSA continual training (~946B tokens) 可以让已有 dense attention 模型转 sparse
- DSA 无质量损失，反而 long-context reasoning +4 pt
- 推理成本降 3-5× at 128K
- Indexer ReLU + FP8 + 少头是最优实现
- Keep Routing 对 MoE RL 稳定必需

### Open
- DSA 的 indexer head 数 / dim 怎么 scale 到不同模型规模
- k=2048 是不是最优（更大 k 是否质量更好）
- DSA 在 32K 以下短上下文是否仍 net positive（论文用 MHA mode 模拟，未直接对比）
- DSA + 非 MLA 的 attention 形态（如 GQA）的实现方式

### 已否决（V3.2 团队明确不做）
- Sliding window attention（DSA 是 data-driven 选 top-k，而非 prefix-position 选）
- Random sparse attention（必须 indexer 学）
- 从头训 DSA（continual 更经济）

## 与其他笔记交叉引用

- [[04_deepseek_v3]] —— V3 原版 MLA + 架构基础
- [[06_kimi_k2]] —— K2 同代 1T MoE，没用 DSA
- [[42_100b_cookbook]] Step 6 —— attention 选型，DSA 是 next-gen 选项
- [[28_open_source_moe_catalog]] entry #59 —— DSV3.2-Exp 在主表
- [[40_qwen3_next]] —— hybrid attention 是另一条 long-context 路线
- [[15_jamba]] / [[13_minimax_01]] —— Mamba / Linear attention 是 DSA 的替代方案
- **[[37_ling1t]]** —— Ling-1T 用 GQA + QK-norm + partial RoPE，**没用 DSA**，长上下文走 YaRN
- [[03_auxloss_free]] —— Keep Routing 是 ALF 路由在 RL 时的扩展

## 一句话评价

> **DSA 是 100B+ 段位长上下文 attention 的 "next-gen 默认"，因为它能 retrofit、无质量损失、且大幅降推理成本。如果你做 long-context model，DSA 应该和 MLA / FP8 一道列入"3 年必采纳"清单。**
