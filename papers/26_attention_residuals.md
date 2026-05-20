# Attention Residuals (AttnRes)

- **arXiv**: 2603.15031 (v1, 2026-03-16)
- **机构**: Kimi Team / Moonshot AI
- **代码**: github.com/MoonshotAI/Attention-Residuals（CC BY-NC-ND 4.0）
- **作者**: Kimi Team 集体署名

## TL;DR

把 transformer 的标准残差 `h_l = h_{l-1} + f(h_{l-1})` 换成**沿深度的 softmax attention**：每层用一个学习的 d 维伪 query `w_l`，对所有前序层输出做注意力加权聚合。**Full AttnRes** = 每层 attend 全部 L 层；**Block AttnRes** = 把 L 层切成 N 块，块间 attend 块表示 + 块内累加。

**核心理论**：标准残差 = 沿深度的"linear attention with uniform weights"；AttnRes = 沿深度的 softmax attention。**做的事就是把 sequence 维度上 RNN→Transformer 的同一个变换搬到 depth 维度**。

**实测**：48B/3B Kimi Linear 上 1.4T tokens，Block AttnRes vs baseline 在 9 个 downstream 上**全胜**：MMLU +1.1、GPQA-Diamond **+7.5**、Math **+3.6**、HumanEval +3.1、C-Eval +2.9。Scaling law 上 **1.25× 计算优势**（5.6 PFLOP-days）。

**Block AttnRes I/O 5.5d**（vs 标准残差 3d，vs mHC 34d，vs Full AttnRes 24d），**推理 latency 开销 < 2%**，是当前最实用的残差替代。

## 关键数学（Eq. 1-4）

### Full AttnRes（论文 §3.1）

$$h_l = \alpha_{0\to l} \cdot h_1 + \sum_{i=1}^{l-1} \alpha_{i\to l} \cdot f_i(h_i)$$

attention 权重：
$$\alpha_{i\to l} = \frac{\exp(q_l^\top \cdot \text{RMSNorm}(k_i))}{\sum_j \exp(q_l^\top \cdot \text{RMSNorm}(k_j))}$$

其中：
- $q_l = w_l \in \mathbb{R}^d$ —— **layer-specific 学习 pseudo-query**（与 forward 计算解耦，可并行）
- $k_i = v_i = h_1$ if $i=0$ else $f_i(h_i)$ —— 第 $i$ 层的输出（不引入新投影）
- RMSNorm 防止 large-magnitude 输出 dominate

**关键设计选择**：
- $w_l$ **必须初始化为 0** → 训练初期 $\alpha$ 均匀 → 退化成 baseline 行为，避免 init volatility
- 每层只多 1 个 RMSNorm + 1 个 d 维向量 = 参数开销几乎为 0
- $w_l$ 与 $f_l$ 解耦 → block 内 S 个 query 可 batch 成 (S, d) GEMM，I/O 从 S 次降到 1 次

### Block AttnRes（论文 §3.2）

把 $L$ 层切成 $N$ 块、每块 $S = L/N$ 层。第 $n$ 块第 $i$ 层的 value 矩阵：

$$V = \begin{cases} [b_0, b_1, \ldots, b_{n-1}]^\top & \text{if } i=1 \text{(块内首层)}\\ [b_0, b_1, \ldots, b_{n-1}, b_n^{i-1}]^\top & \text{if } i \geq 2 \end{cases}$$

其中 $b_n = \sum_{j \in \mathcal{B}_n} f_j(h_j)$ 是块累加（partial sum），$b_0 = h_1$ 让 token embedding 始终参与。

**Block AttnRes 在做什么**：每层不再 attend 所有 L 个层输出，而是 attend $N$ 个**块表示** + 当前块内 partial sum，把 attention 的 KV cache 从 L 压到 N。

## 系统设计（论文 §4）—— 这是真正的工程亮点

### Two-phase computation（Algorithm 1）

```
Phase 1: 块间 attention（并行）
  Q = [w_l for l in B_n]      # (S, d) batched query
  K, V = [b_0; ...; b_{n-1}]  # (n, d) cached block reps
  o^(1), m^(1), ℓ^(1) = AttnWithStats(Q, K, V)  # 返回 LSE 用于后续 merge

Phase 2: 块内顺序 attention + online softmax merge
  for l in B_n:
    if i == 0: h_l = o_l^(1) / ℓ_l^(1)        # 仅块间
    else:
      o^(2), m^(2), ℓ^(2) = AttnWithStats(w_l, b_n^i, b_n^i)  # 块内
      h_l = (e^(m^(1)-m_l) o^(1) + e^(m^(2)-m_l) o^(2)) / (...)  # online softmax
    b_n^i = b_n^{i-1} + f_l(h_l)              # 更新 partial sum
```

**关键**：online softmax 让块间结果（一次 batched）和块内顺序累加可以**精确合并**而非近似。Phase 1 的 read 从 S 次摊到 1 次。

### Cross-stage caching（PP 关键，§4.1）

interleaved PP 调度下，朴素实现要每次 transition 传完整 block 历史 → $\mathcal{O}(C)$ 通信成本。**Cross-stage caching**：

$$\text{Comm}_{\text{cached}} = \underbrace{\frac{P(P-1)}{2} N_p d}_{\text{first virtual stage}} + \underbrace{(V-1) P^2 N_p d}_{\text{subsequent}}$$

把单次 transition 成本从 $\mathcal{O}(C)$ 降到 $\mathcal{O}(P)$，**$V \times$ 改进**。能与 1F1B 计算完全 overlap。

### Memory-efficient prefilling（128K 上下文 inference）

直接 storing block representations 在 128K × 8 块 × d=4096 ≈ **15 GB / device**。论文 sharding：
- Block reps 沿 sequence 维度切 P 份（TP=P）
- Phase 1 在 local shard 上跑，输出 reduce-scatter
- Phase 2 在 all-gather 后做（fused with RMSNorm）
- **每设备 15 GB → 1.9 GB**（128K context, P=8）
- 配合 chunked prefill (16K chunks) → **< 0.3 GB / device**

**推理 latency 开销 < 2%**（典型 inference workload）

## I/O 对比（Table 1，决定性证据）

每 token 每层的 residual 机制 I/O cost（不含 layer function $f_l$ 内部 I/O）：

| 方案 | 操作 | Read | Write | **总 I/O** | 典型值 |
|---|---|---|---|---|---|
| **标准 residual** | merge | 2d | d | **3d** | **3d** |
| **mHC (m=4 streams)** | compute α/β/A + apply + merge | (8m+2)d + 2m² + 4m | md | (8m+2)d + 2m² + 4m | **34d** |
| **AttnRes Full** | Phase 1 (摊销) + Phase 2 | (S+N)d + (S-1)d | 2d | (S+N)d | **24d** (S=16, N=4) |
| **AttnRes Block** | Phase 1 (摊销) + Phase 2 | (N/S)d + 3d | 2d | **(N/S + 5)d** | **5.5d** (N=8, S=16, m=4) |

**Block AttnRes 比 mHC 节省 $34/5.5 \approx 6.2\times$ I/O**，比 Full AttnRes 节省 $4.4\times$。

## 实验设置

### Scaling law（5 anchor，§5.1）

5 个模型，194M → 528M active params，4096 context，cosine LR，每 size 三 variant 共享超参。

| Active | Tokens | $L_b$ | H | $d_{model}$ | $d_{ff}$ | LR | BS | **Baseline** | **Block AttnRes** | **Full AttnRes** | **mHC(-lite)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 194M | 38.7B | 12 | 12 | 896 | 400 | 2.99e-3 | 192 | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 45.4B | 13 | 13 | 960 | 432 | 2.80e-3 | 256 | 1.895 | 1.875 | 1.874 | **1.869** |
| 296M | 62.1B | 14 | 14 | 1024 | 464 | 2.50e-3 | 320 | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 87.9B | 16 | 16 | 1168 | 528 | 2.20e-3 | 384 | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 119.0B | 17 | 17 | 1264 | 560 | 2.02e-3 | 432 | 1.719 | 1.693 | **1.692** | 1.694 |

**关键观察**：
- 三个 add-on 在 loss 上几乎打平，差距都 ≤ 0.005
- mHC(-lite) 在 528M 是 1.694，**输给 Full AttnRes 0.002**（噪声水平）
- Block AttnRes 用 1/4 I/O 拿到 ~95% 收益

**Scaling law 拟合**：
- Baseline: $\mathcal{L} = 1.891 \times C^{-0.057}$
- Block AttnRes: $\mathcal{L} = 1.870 \times C^{-0.058}$
- Full AttnRes: $\mathcal{L} = 1.865 \times C^{-0.057}$
- → Block AttnRes 在 5.6 PFLOP-days 处 **1.25× 计算优势**

### 主实验：Kimi Linear 48B/3B + 1.4T tokens

**架构**：基于 Kimi Linear（Moonshot Linear）= **Kimi Delta Attention (KDA) + MLA 3:1 比例 + MoE FFN**
- 27 transformer blocks (54 layers，attn + mlp 各算一层)
- **256 routed experts + 1 shared, top-8** → 48B total / 3B active
- Block AttnRes: **N=9 块 × 6 layers/块**（含 token embedding 是 9 + 1 = 10 个 depth source）
- Muon optimizer（vs AdamW）
- WSD scheduler，4096 context，global batch 8M tokens
- WSD pretrain 1T → mid-training 400B（Moonlight annealing recipe）→ 32K context extension
- MLA 用 **NoPE**（no positional encoding），所以 context extension 不需要 YaRN

### 训练动力学（Fig. 5）

| 信号 | Baseline | Block AttnRes |
|---|---|---|
| Validation loss | 持续高于 AttnRes | 持续较低，decay 阶段差距扩大 |
| Output magnitude (per block) | 后层指数增长 → 12 量级 | **块边界 reset，bounded periodic** |
| Gradient magnitude | 早层 disproportionately 大 | **跨层 uniform** |

→ Block AttnRes 直接证伪 PreNorm dilution（每层贡献越来越被稀释）的影响。

### 下游 benchmark（Table 3，1.4T tokens 后）

| 类别 | Benchmark | Baseline | **AttnRes** | Δ |
|---|---|---|---|---|
| **General** | MMLU | 73.5 | **74.6** | +1.1 |
| | MMLU-Pro | 52.2 | 52.2 | 0 |
| | **GPQA-Diamond** | 36.9 | **44.4** | **+7.5** ⚡ |
| | BBH | 76.3 | **78.0** | +1.7 |
| | ARC-Challenge | 64.6 | **65.7** | +1.1 |
| | HellaSwag | 83.2 | **83.4** | +0.2 |
| | TriviaQA | 69.9 | **71.8** | +1.9 |
| **Math/Code** | GSM8K | 81.7 | **82.4** | +0.7 |
| | MGSM | 64.9 | **66.1** | +1.2 |
| | **Math** | 53.5 | **57.1** | **+3.6** ⚡ |
| | CMath | 84.7 | **85.1** | +0.4 |
| | HumanEval | 59.1 | **62.2** | +3.1 |
| | MBPP | 72.0 | **73.9** | +1.9 |
| **Chinese** | CMMLU | 82.0 | **82.9** | +0.9 |
| | C-Eval | 79.6 | **82.5** | +2.9 |

**全胜，且 reasoning（GPQA, Math, HumanEval）涨幅最大**。这是 reasoning-leaning 16B 设计要看的 evidence。

## 与本仓库的契合度（16B MoE 适配）

### 推荐配置（如果 A2 通过 → 进 16B 主训练）

| 维度 | 推荐 | 理由 |
|---|---|---|
| **变体** | **Block AttnRes** | I/O 5.5d 比 Full 24d 经济 4×；scaling law 上几乎追平 Full |
| **N（块数）** | **N=4 或 6**（16B 是 27 层） | N=4 → ~7 层/块；N=6 → ~4-5 层/块。Kimi 用 N=9（54 层），按比例 27 层取 N=4 最直接 |
| **伪 query $w_l$ init** | **必须 0** | 论文明确强调，否则训练初期 attention 权重 volatile |
| **RMSNorm on key** | **必须** | 防止 large-magnitude 层输出 dominate softmax |
| **Pretrain 兼容** | ✓ 任意 | WSD / WSM / cosine 都能用；Kimi 用 WSD + Moonlight annealing |
| **PP 兼容** | ✓ | 用 cross-stage caching；vs DualPipe 共存 |
| **EP 兼容** | ✓ | AttnRes 是 depth 维操作，与 EP（width 维）正交 |
| **FP8 兼容** | 论文未明 | RMSNorm + softmax，理论上和 dots1 FP32 gating 一样可以 router-style 提到 FP32 |
| **MTP 兼容** | ✓ | 完全独立 |
| **检查点兼容** | ✓ | 标准 transformer + 每层多 1 个 d-vec |

### Wind tunnel A2 设计

借用本仓库 22_FINAL_16B_design §8 的 A2 anchor (1B/200M/25B tokens):

| arm | 配置 | 预算 |
|---|---|---|
| A2-baseline | 标准 PreNorm residual | 已有 |
| A2-attnres-N4 | + Block AttnRes, N=4 | +8% compute |
| A2-attnres-N6 | + Block AttnRes, N=6 | +8% compute |

判定：
- **Loss 改善 ≥ 0.005** + **下游 reasoning（GSM8K / GPQA-mini）涨 ≥ 1pt** → **进 16B 主训练**
- **Loss 改善 < 0.003** → **砍掉**，5.5d I/O 不白送
- **0.003 ~ 0.005** → 算 borderline，看下游和训练稳定性再决

## Caveats / 局限

- **2026-03 才发布**，独立第三方复现还几乎没有
- **N=8（Kimi 配置）是经验值**，论文 §3.2 说 "N≈8 recovers most of the benefit"，**没有 N 的系统消融**
- **Kimi Linear 是 KDA + MLA hybrid attention**，不是标准 GQA。AttnRes 在 standard GQA + MoE 上的迁移效果论文**没直接验证**（virgin GQA baseline 只在 small-scale scaling law 中跑过）
- **Muon optimizer**：Kimi 用 Muon，不是 AdamW。AttnRes 的下游收益**部分可能来自 Muon**而不是 AttnRes 本身。论文说"all variants share identical hyperparameters"，但小 anchor 是 cosine + 标准优化器，48B 主实验是 Muon —— 没有 48B baseline-only-no-AttnRes 在 Muon 下的对照
- **Sequence length 4096**，更长 context（128K+）下 cross-stage caching 的实际开销论文 §4.2 给了估算（< 0.3 GB），**但没有 wall-clock latency 测量**
- **Inference 推 latency < 2%** 是论文 claim，未独立验证
- **Pseudo-query $w_l$ 初始化 0 → 等价 baseline → 训练初期没有任何信号**：这意味着 AttnRes 的 benefit 必须**经过足够训练步**才显现。Wind tunnel A2 的 25B tokens 可能不够长，Kimi 在 1T tokens 后才看到稳定改善
- 代码 license **CC BY-NC-ND 4.0**：禁止商用 + 禁止 derivative。**实际工程使用要重写实现**，不能直接 import

## 与本仓库的交叉引用

- **22_FINAL_16B_design.md** §4.5（Attention）+ §4.8（稳定性）：AttnRes 是 PreNorm dilution 的解，应进入 §11 "强默认但 pilot 必测"
- **27_mhc.md**：直接竞争方案，I/O 6× 更贵；AttnRes 论文 Table 2 直接 head-to-head
- **08_ling_2.md**：Ling-1T 用 80 层、Ling-flash 32 层，AttnRes Block 同样适用（N=10 / N=4）
- **04_deepseek_v3.md**：V3 没用任何 residual stream 修改，AttnRes 是 V3 路线之外的可选方向
- **20_mtp_gloeckle.md** + **23_mtp_investigation.md**：MTP 与 AttnRes 完全正交，可叠加
- **18_params_vs_flops.md**：AttnRes 给定 params 下小幅提高 effective FLOPs 利用率（1.25× 等效 compute），不改 scaling law slope
