# Muon Optimizer: Scalable Matrix-Orthogonalization for LLM Training

> **核心论文**：
> - Jordan, K. et al. 2024 — "Muon: An optimizer for hidden layers in neural networks" (blog post, kellerjordan.github.io/posts/muon/) — **原始 Muon**
> - Liu, J. et al. 2025 — "Muon is Scalable for LLM Training" (arXiv 2502.16982) — **Moonlight**, 第一个把 Muon 推到 16B/5.7T 规模的 MoE
> - Team, Kimi 2025 — Kimi K2 paper — **MuonClip**: Muon + post-QK-clip 在 1T 段位
> - GLM-4.5 Team 2025 (arXiv 2508.06471) — 第一个**整 base 训练**全程用 Muon 的 355B 模型

## TL;DR

Muon 是 2024-Q4 Keller Jordan 提出的 "**用 Newton-Schulz 迭代正交化 momentum 矩阵**" 的优化器。2025 迅速被 Moonshot (Moonlight + Kimi K2)、Z.ai (GLM-4.5) 采纳为 MoE 主流替代。三个关键事实：

1. **Muon 只对 hidden layer 的 2D 矩阵生效** —— embedding / LM head / RMSNorm / bias 仍用 AdamW
2. **Moonlight 给出 scaling law 实证：Muon 需要 ~52% AdamW FLOPs 达到同 loss** (Figure 1a, Table 3)
3. **Muon 在 MoE 上收益比 dense 大** —— 因为 router 权重 SVD entropy 提升最多（Figure 4），**MoE 路由权重最受益**

但**Ling-1T (1T MoE) 显式没用 Muon，仍用 AdamW** —— 说明 Muon 在 1T+ 段位"是否真比 AdamW 强"仍有不同选择。

## 原始 Muon 公式（Jordan 2024）

### 更新规则
$$
M_t = \mu M_{t-1} + \nabla \mathcal{L}_t(W_{t-1}) \\
O_t = \text{NewtonSchulz}(M_t) \\
W_t = W_{t-1} - \eta_t O_t
$$

其中 $M_t$ 是 momentum (零初始化)，$\mu = 0.95$ (Nesterov-style)，$\eta_t$ 是学习率。

### Newton-Schulz 迭代
设 $X_0 = M_t / \|M_t\|_F$（Frobenius norm 归一化），然后 $N=5$ 步迭代：

$$
X_k = a X_{k-1} + b (X_{k-1} X_{k-1}^T) X_{k-1} + c (X_{k-1} X_{k-1}^T)^2 X_{k-1}
$$

**系数 (a, b, c) = (3.4445, -4.7750, 2.0315)**（Jordan 2024 调出来的；目的是让多项式 $f(x) = ax + bx^3 + cx^5$ 在 1 附近有 fixed point，从而 5 步收敛）

### 数学意义
设 $M_t = U \Sigma V^T$ 为 SVD，则 $(M_t M_t^T)^{-1/2} M_t = U V^T$ → **正交矩阵，奇异值全 1**。

**直觉**：SGD-momentum 更新 $M_t$ 通常是病态矩阵（few dominant directions 主导），正交化后**所有方向有相同 magnitude**，"放大稀有方向"。

### 不优化的参数
- 1D 参数（biases, RMSNorm scales）
- Input/output 层（embeddings, LM head）—— modular norm theory 指出这些有不同的最优化动力学

→ **混合 Muon + AdamW** 是标配（不是替换 AdamW，是补充）。

## Moonlight 把 Muon 推到 LLM 规模的三大改造（arXiv 2502.16982）

Moonshot 团队发现原始 Muon "在小 scale 强、scale up 后优势消失"。他们诊断出**两个根本问题**，给出修复，让 Muon 第一次在 16B/5.7T 段位有效。

### 改造 1：加 Weight Decay（§2.2）

原始 Muon **没有 weight decay**。Moonshot 发现：
- 不加 WD → 权重和 layer 输出 RMS 持续增长
- → 最终超出 bf16 高精度范围
- → 模型 quality 受损

**修复（Eq. 3）**：
$$
W_t = W_{t-1} - \eta_t (O_t + \lambda W_{t-1})
$$

WD ratio $\lambda$ 同 AdamW (典型 0.1)。

**实证（Figure 2）**：
- AdamW (绿)：最慢
- Muon w/o WD (红)：先快后慢，**长期被 AdamW 反超**
- Muon w/ WD (蓝)：**最快且持续领先到 65k iterations**

→ **WD 是 Muon scale up 的必要补丁**。Jordan 原版只在 1.5B 段位验证，那个 scale 还没显出 WD 必要性。

### 改造 2：Consistent Update RMS via Shape-Based Scaling（§2.2 Lemma 1）

**问题**：Muon 的 update RMS 不是常数，**取决于矩阵 shape**。

**Lemma 1**：对 shape $[A, B]$ 的满秩矩阵参数，Muon 的理论 update RMS = $\sqrt{1/\max(A, B)}$。

→ 对 dense MLP $[H, 4H]$：RMS 太小 → 训练不稳，capacity 没用上
→ 对 GQA 单个 KV head $[H, H/n_{kv}]$：RMS 太大 → 不稳

**修复（Eq. 7, Adjusted LR）**：
$$
W_t = W_{t-1} - \eta_t (0.2 \cdot O_t \cdot \sqrt{\max(A, B)} + \lambda W_{t-1})
$$

**为什么 0.2？** AdamW 的典型 update RMS = 0.2-0.4。乘 0.2 让 Muon 的 update RMS 匹配 AdamW → **可以直接复用 AdamW 调好的 LR / WD**。

→ "out-of-the-box without hyper-parameter tuning" 的关键技巧。

### 改造 3：Distributed Muon for ZeRO-1（§2.3, Algorithm 1）

**问题**：原始 Muon 需要 full gradient matrix 才能算 NS 迭代。ZeRO-1 把 momentum / params 切到 DP 上后，每个 device 只有 partial → 算不了。

**修复**：先 DP-gather gradient → 算 NS → 只保留本地 partition。

伪代码：
```
g = reduce_scatter(G, dp_group)         # 1
g' = update_with_momentum(g, m, μ)      # 2
G = gather(g', dp_group)                # 3 (Muon-specific extra)
U = NewtonSchulz(G)                     # 4 (Muon-specific extra)
u = local_partition(U)
p' = apply_update(p, u)                 # 5
P = all_gather(p', dp_group)            # 6
```

蓝色行 (3, 4) 是 Muon 相对 distributed AdamW 的额外开销。

**通信开销分析**：
- distributed AdamW: 4 (fp32 G reduce-scatter) + 4 (fp32 P all-gather) = 8
- distributed Muon: 4 (fp32 G reduce-scatter) + **2 (bf16 G gather)** + 4 (fp32 P all-gather) = 10
- 比例 = (1, 1.25)，**实际多 DP 时接近 1×**

**Memory 开销分析**：Muon 只 1 个 momentum buffer，AdamW 2 个 → Muon 的额外 memory = **distributed AdamW 的一半**

→ Distributed Muon 的 wall-clock 开销仅 1-3% vs forward-backward 时间，**几乎免费**。

## Scaling Law 实证（Moonlight Figure 1a + Table 3）

### Setup
- Llama 架构 dense 模型
- 5 个模型规模：399M / 545M / 822M / 1.1B / 1.5B params (no embedding)
- Compute-optimal 训练（Kaplan 风格 grid search）
- Muon 复用 AdamW 调好的 LR (8.3e-4 - 9.5e-4) 和 WD

### 拟合 (Table 3)
| Optimizer | LM Loss (seqlen=8K) |
|---|---|
| Muon | $2.506 \cdot C^{-0.052}$ |
| AdamW | $2.608 \cdot C^{-0.054}$ |

→ Muon 的 baseline 项 (2.506) 小，但 exponent (0.052) 略浅。**两条曲线斜率几乎平行**。

### 关键结论（Figure 1a 标注）
> **Muon achieves comparable performance to AdamW trained counterparts while requiring only approximately 52% of the training FLOPs.**

→ "**~2× compute efficiency**"。

→ 注意：这是 **dense model** 验证。MoE 上的 scaling law 没在 Moonlight 论文做（数据点只在一个 model，16B Moonlight 本身）。

## Moonlight 完整训练 recipe（§3.3）

- **架构**：DeepSeek-V3-small 系（2.24B active / 15.29B total / 64 routed + 2 shared, K=6, sigmoid + ALF）
- **Tokens**: 5.7T
- **Optimizer**: Muon for 2D weights; **AdamW for embeddings / LM head / RMSNorm / bias / router**
- **Stage 1 (0→33B tokens)**: linear warmup LR 0 → **4.2e-4** in 2k steps，batch 2048
- **Stage 2 (33→5.2T tokens)**: cosine decay LR 4.2e-4 → 4.2e-5；batch 2048→4096 at 200B 处
- **Stage 3 (5.2→5.7T, cooldown)**: LR 升回 **1e-4** in 100 steps, linear decay to 0 in 500B tokens；batch 保持 4096；高质量 math/code/reasoning 数据
- **Weight decay**: 0.1（与 AdamW 一致）
- **ALF bias update rate**: 1e-3 stage 1-2, 0 stage 3
- **Seq len**: 8K

→ 这是当前**最详细公开的 Muon LLM training recipe**，可以直接复刻。

## MoE 上的 Muon 独特价值（§3.4 SVD Entropy 分析）

Moonshot 团队做了**有意思的诊断实验**：把 weight matrix 的 SVD entropy 当 "matrix 表征多样性" 指标，比较 Muon vs AdamW 训练后的权重谱。

$$
H(\sigma) = -\frac{1}{\log n} \sum_{i=1}^n \frac{\sigma_i^2}{\sum_j \sigma_j^2} \log \frac{\sigma_i^2}{\sum_j \sigma_j^2}
$$

### Figure 4 结果（按 weight group 分组）

| Group | AdamW SVD entropy | Muon SVD entropy | Gap |
|---|---|---|---|
| AttnQO | ~0.83 | ~0.90 | +0.07 |
| AttnKV | ~0.86 | ~0.95 | +0.09 |
| Experts (FFN) | ~0.92 | ~0.95 | +0.03 |
| SharedExperts | ~0.92 | ~0.94 | +0.02 |
| **Router** | **~0.7** | **~0.85** | **+0.15 (最大)** |
| Dense (first dense layer) | ~0.96 | ~0.97 | +0.01 |

→ **Router 权重的 SVD entropy 提升最大 (+0.15)** —— 即 Muon 让 router 的"专家选择维度"更丰富。

→ **MoE 比 dense 从 Muon 受益更多**，因为 MoE 的核心瓶颈之一是 router 容易塌缩。

⚠️ 论文未在主表给出 "Muon-trained MoE vs AdamW-trained MoE" 的下游 benchmark 对照（Table 4 给 Moonlight-A = Moonlight 同样 setup 但 AdamW，但 Moonlight-A 在 GSM8K 43.8 vs Moonlight Muon 45.0；MATH 16.1 vs 19.8；CMath 57.8 vs 60.2 → 数学有意义提升，**Math/Code 是 Muon 最受益的 task**）

## 各家 100B+ 段位 Muon 采用情况

| 模型 | Optimizer | Scale | Source |
|---|---|---|---|
| Moonlight | **Muon (modified)** | 16B/2.24B, 5.7T | [[06_kimi_k2]] 前作 |
| **Kimi K2** | **Muon + MuonClip (post-QK-clip)** | **1T/32B, 15.5T** | [[06_kimi_k2]] |
| **GLM-4.5 / 4.5-Air / 4.6** | **Muon (全 base training)** | 355B/32B, 23T | [[35_glm45]] |
| DeepSeek V3/V3.1/V3.2 | AdamW | 671B/37B | [[04_deepseek_v3]] |
| Qwen3 系 | AdamW (推测) | 35B/22B-235B | [[05_qwen3]] |
| dots1 | AdamW | 142B/14B, 11.2T | [[24_dots1]] |
| LongCat-Flash | AdamW (ε=1e-16) | 560B/27B, 20T | [[36_longcat]] |
| **Ling-mini/flash/1T-2.0** | **AdamW (β1=0.9, β2=0.95)** | **16B-1T, 20T+** | [[37_ling1t]] |

→ **明显的"Moonshot+Z.ai 派 (Muon)" vs "DeepSeek+Alibaba+Inclusion+Meituan 派 (AdamW)" 分歧**。

→ **特别注意：Ling-1T (Inclusion AI) 是 1T 段位最大的 AdamW 模型** —— 与 Kimi K2 (1T, Muon) 形成直接对照。两家用相同总参跑出相近能力，**说明 Muon 不是 capability 必要条件**。

## MuonClip（Kimi K2 改良，[[06_kimi_k2]] §3.2 详述）

Kimi K2 在 Muon 基础上加 **post-QK-clip**：

每个 attention layer 算完 $QK^T$ 后，clip 到一个 max 值（防止 attention logit blow up）。这是 Muon 在 1T 段位 + 长序列下的额外稳定性补丁。

→ 不是 Muon 本身的问题，而是 **Muon 优化让 attention layer logits 更激进** → 需要额外 clip。

→ 你 16B 段位如果用 Muon，**先不加 MuonClip**，只在出现 attention spike 时再加。

## 与 AdamW 的对比因果链（X 因为 Y）

### 1. Muon 在 dense 上 2× compute efficiency
**因为** orthogonalization 防止权重往 dominant direction 累积 → 更多 "rare direction" 被 update → effective capacity 提升。

### 2. Muon 在 MoE 上 router 受益最大
**因为** router 权重 shape $[H, N_{routed}]$ 经常是 [4096, 256] 这种瘦高型，AdamW 容易让某些 expert 的 router weight 主导（"expert collapse" 前兆）。Muon 的正交化强制 router 输出维度方差均匀。

### 3. Muon 需要加 WD
**因为** 原始 Muon 的 NS 把奇异值 normalize 到 1 → update magnitude 不变 → 权重持续增长 → 超 bf16 范围 → 数值精度损失。WD 把权重拉回原点。

### 4. Muon update RMS 跟 shape 有关
**因为** $\|UV^T\|_F = \sqrt{\min(A, B)}$，归一化后 $\|UV^T / \sqrt{AB}\|_{\text{RMS}} = \sqrt{1/\max(A, B)}$。

### 5. Muon-pretrained 后用 AdamW SFT 性能稍差
**因为** Muon 训出来的 weight matrix 谱结构（高 SVD entropy）与 AdamW SFT 假设的 weight 分布不匹配 → AdamW SFT 把 weight 拉回低 entropy 状态 → loss 部分 Muon 收益。

→ **如果未来 SFT 也想用 Muon，建议预训也用 Muon (Moonlight Table 6)**。

## 16B 用不用 Muon 的决策框架

### 用 Muon 的理由
1. **2× compute efficiency** (Moonlight scaling law) → 同 capability 训练成本减半
2. **MoE router 收益大** → 缓解 expert collapse 风险
3. **bf16 stable** → 不需要部分 FP32 (vs AdamW 经常需要)
4. **Memory 比 AdamW 少 50%** → 更大 batch 容得下
5. **Wall-clock 开销 < 3%** → 几乎免费

### 不用 Muon 的理由
1. **Distributed Muon kernel** 实现复杂度比 AdamW 高 — Megatron-LM 上游还没完全 merge
2. **NS 迭代需要 [A, B] 形 weight matrix**，部分 hybrid attention（如 Mamba 的 SSM 参数）不适合 → 这些参数仍要 AdamW
3. **MoE 上的 scaling law 没单独验证** — Moonlight 只有一个 16B MoE 数据点，比 dense 5 点弱
4. **生态成熟度低** — Ling 派、DeepSeek 派、Alibaba、Meituan 都不用，意味着配套（learning rate finder, gradient health monitor 等）不成熟
5. **Pretraining + SFT 都要换** — Moonlight Table 6 显示 mixed 模式（pretrain Muon, SFT AdamW）会丢部分收益。这是切换成本

### 我对你 16B 的建议

| 你的情况 | 推荐 |
|---|---|
| 第一次训 16B，有 deadline | **AdamW** (保守路线，跟 Ling/dots1) |
| 有 wind tunnel B 预算，想验证 | **Wind tunnel B 必加 Muon arm** (单 800M 模型 100B tokens 对照，~$15K) |
| 训练成本是关键约束 | **Muon (跟 Moonlight)** |
| 团队有 Muon 实现经验 | **Muon** (距 GLM-4.5/Kimi K2 路线) |
| 团队是 Megatron-LM 重度用户 | **AdamW** (Muon kernel 在 Megatron 上还在 PR 阶段) |

**我的默认建议**：**Wind tunnel B 必跑 Muon vs AdamW 800M 对照**。如果 Muon 收益 ≥ 0.005 loss + 训练稳定，则切到 Muon。否则跟 Ling 派用 AdamW。

## Settled vs Open

### Settled
- Muon 在 dense ~2× compute efficiency (Moonlight scaling law 验证)
- Muon 必须加 WD scale up
- Muon update RMS 跟 shape 有关，需要 Adjusted LR
- Distributed Muon 几乎免费（< 3% latency）
- Muon **只用于 2D hidden layer weights**，1D/embedding/LM head 用 AdamW
- Muon-pretrained + Muon-SFT > all other combos

### Open
- Muon 在 MoE 上 scaling law（只有 1 个 16B 数据点）
- Muon 在 1T+ 段位是否真比 AdamW 强（Kimi K2 vs Ling-1T 直接 contradiction）
- MuonClip 是不是 1T 段位 attention 稳定必备
- Muon 配 cosine vs WSD vs WSM 谁最优（GLM 用 cosine 但说 WSD underfits）
- Muon 在长 ctx + post-RL 阶段的稳定性

### 已否决（Jordan 原文 / Moonshot / GLM 都不做）
- Muon 替换所有参数（embedding/LM head 仍 AdamW）
- N (NS 步数) > 5（增到 10 不带来 loss 改善，只增 latency）
- 不加 WD（scale up 后必坏）

## 与本仓库其他笔记交叉引用

- [[06_kimi_k2]] —— MuonClip + 1T 段位 Muon 应用
- [[35_glm45]] —— GLM-4.5 全 base 训练用 Muon + cosine schedule
- [[37_ling1t]] —— Ling-1T 1T 段位坚持 AdamW，与 Muon 派直接对照
- [[36_longcat]] —— LongCat AdamW ε=1e-16 给出"AdamW 也可以训 560B"的反向证据
- [[03_auxloss_free]] —— ALF + Muon 组合 (Moonlight 用) 是当前最完整的 MoE 路由+optimizer 栈
- [[19_sparse_upcycling]] —— upcycling 阶段是否要切换 optimizer 是开放问题
- [[22_FINAL_16B_design]] §11 → 应该在 "强默认但 pilot 必测" 加 Muon 对照
- [[29_wind_tunnel_a2]] → wind tunnel B 应有 Muon arm
