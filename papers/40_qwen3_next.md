# Qwen3-Next-80B-A3B: Hybrid Gated DeltaNet + Attention with Extreme MoE Sparsity

> **来源**：
> - Qwen 团队 blog post: qwenlm.github.io/blog/qwen3_next/
> - HuggingFace transformers docs: huggingface.co/docs/transformers/model_doc/qwen3_next
> - 配置：huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct (config.json)
> - 发布：2025-09 (无独立 arXiv 论文，主要靠 blog + 模型卡)
> - 相关基础论文：Yang et al. 2024 "Gated Linear Attention with Hardware-Efficient Training" (Gated DeltaNet 数学基础)

## TL;DR

Qwen3-Next-80B-A3B 是 **2025-Q4 hybrid attention + MoE 的最激进设计**：
1. **Hybrid 3:1 DeltaNet:Attention pattern** —— 75% layer 是 Gated DeltaNet (线性), 25% 是 Gated Attention (softmax)
2. **1/27 整体激活率** (80B total / 3B active)，**1/51 MoE 层激活率** (10/512 experts)
3. **head_dim=256** —— 比 Ling/GLM/V3 的 128 大一倍
4. **Top-K=10** —— 唯一 100B 段位用 K=10 而非 K=8 的开源模型

**Qwen 团队声称**：训练成本 < Qwen3-32B 的 1/10，长上下文 (>32K) inference throughput 10×。

## 与 hybrid attention 三家路线对比

| 模型 | 线性:Softmax | 线性变种 | 段位 |
|---|---|---|---|
| Jamba 1.5 Large | 7:1 (12.5% softmax) | Mamba SSM | 398B/94B |
| MiniMax-Text-01 / M1 | 7:1 (12.5% softmax) | Lightning Attention | 456B/45.9B |
| **Qwen3-Next-80B** | **3:1 (25% softmax)** | **Gated DeltaNet** | **80B/3B** |
| Granite 4 H | hybrid (未明确) | Mamba-2 | 32B/9B |
| Nemotron-3 Nano | hybrid | Mamba-2 | 31.6B/3.2B |
| Hunyuan-Turbo-S | hybrid | Mamba-2 | 560B/56B |

→ **Qwen3-Next 是 hybrid 流派中最"保守"的（softmax 占比最高）**。这是有意的：完全 linear 在 precise retrieval 上有限制，3:1 是 Qwen 团队对 "线性扩展 + softmax 精度" 的折中。

## 完整 Spec（从 HF config 提取）

| 维度 | Qwen3-Next-80B-A3B |
|---|---|
| Total params | **80B** |
| Active params | **3B** |
| 整体激活率 | **3.75% (1/27)** |
| 层数 | **48** |
| Hybrid 结构 | **12 × (3 × DeltaNet+MoE → 1 × Attn+MoE)** = 36 DeltaNet + 12 softmax Attention |
| Hidden | 2048 |
| Intermediate (dense FFN, 仅 dense MoE 用) | 5632 |
| **Gated Attention heads** | 16 |
| **Gated Attention KV heads** | **2** (8× GQA 压缩，极端) |
| **head_dim** | **256** (注意比 Ling/GLM 的 128 大！) |
| Gated DeltaNet conv kernel | 4 |
| DeltaNet key head dim | 128 |
| DeltaNet value head dim | 128 |
| **DeltaNet num key heads** | **16** |
| **DeltaNet num value heads** | **32** (key:value head ratio 1:2) |
| MoE expert FFN intermediate | **512** (极小！) |
| Shared expert intermediate | 512 |
| **N_routed** | **512** (最多的开源 MoE 之一) |
| Shared experts | 1 |
| **Top-K** | **10** (唯一非 8) |
| **MoE 层激活率** | **10/512 = 1/51** (论文说的 "1:50") |
| norm_topk_prob | True (Mixtral 派 softmax + aux) |
| **Router aux loss coef** | **0.001** (softmax + aux，不走 sigmoid+ALF) |
| decoder_sparse_step | 1 (每层都 MoE，没有 first-K-dense) |
| Vocab | 151936 |
| max_position_embeddings | 32768 (训练长度) |
| 推理 ctx | 1M (YaRN 扩展) |
| MTP | Yes (depth 未公开) |
| Normalization | zero-centered weight-decayed layernorm + RMSNorm (rms_norm_eps=1e-6) |

→ **L/√H = 48/√2048 = 1.06**，与 Qwen3-30B 相同 (Qwen 派高瘦传统)

→ **K=10 + 512 experts = 路由 logits 5120 维**，比 V3/K2 的 8/256 维大 2×

## 核心创新 1：Hybrid 3:1 Gated DeltaNet + Gated Attention

### 架构图（每个 hybrid block）

```
Input
  ↓
[Layer 1] Gated DeltaNet → MoE Block (10/512 expert)
  ↓
[Layer 2] Gated DeltaNet → MoE Block
  ↓
[Layer 3] Gated DeltaNet → MoE Block
  ↓
[Layer 4] Gated Attention → MoE Block      ← 唯一 softmax 层
  ↓
[下一个 hybrid block]
```

12 个这样的 block，共 48 层。

### Gated DeltaNet 数学（Yang et al. 2024 风格）

Gated DeltaNet 是 linear attention 的一种 variant，用 **outer-product memory + 门控更新** 来记忆历史：

$$
S_t = (1 - \sigma(g_t)) S_{t-1} + \sigma(g_t) k_t v_t^T \quad \text{(state update with forget gate)}
$$
$$
o_t = q_t S_t \quad \text{(output via state read)}
$$

其中：
- $S_t \in \mathbb{R}^{d_k \times d_v}$ 是 state matrix（"memory"）
- $g_t = W_g x_t$ 是 forget gate (sigmoid)
- $k_t, v_t, q_t$ 是 key/value/query 投影
- $\sigma$ = sigmoid

**关键特性**：
- O(n) 时间和空间复杂度 (vs softmax 的 O(n²))
- 可以 parallelize via chunked scan (训练时 throughput 接近 softmax attention)
- 但 **没有精确 retrieval** 能力（线性记忆压缩损失信息）

### Gated Attention（attention 层加 gate）

标准 attention 的 output 加一个 element-wise gate：
$$
\text{output} = \sigma(W_g \cdot x) \odot \text{Attention}(x)
$$

→ 这是 Hua et al. 2022 "Transformer Quality in Linear Time" 起的设计，用 gate 让 model 学会"什么时候真要 attend，什么时候 default to 0"。在 hybrid 架构里特别有用：当 attention 层处理的是 linear 已经压缩好的 sequence 时，gate 决定是否需要"重新看原始信号"。

### Why 3:1 而不是 1:7 或 7:1？

**因果链**：
- **MiniMax / Jamba 选 1:7** = 88% linear，最大化 throughput，但 retrieval 弱（论文 long-context 表现一般）
- **Qwen3-Next 选 3:1** = 75% linear, 25% softmax → 平衡 throughput 与 retrieval
- **GLM/Ling/V3 选 0:1** = 100% softmax → 最强 retrieval，但 long-context throughput 差

→ 25% softmax 比例是 Qwen 团队的判断：足以维持 GPQA / MMLU 水准，但 75% linear 给了 throughput / 长上下文 优势。

→ **MiniMax-M2 (2025-10) 把这个比例改回 0:1 (full softmax)**，说明 1:7 太激进。Qwen3-Next 的 3:1 是当前 hybrid 流派的"最稳"配置。

## 核心创新 2：Extreme MoE Sparsity (1/51)

### 配方
- 512 routed experts × 10 active = **2% expert 被激活**
- expert FFN intermediate **512** (vs Ling-flash 1024, V3 2048) → **每个 expert 极小**
- shared expert intermediate 512

### 这是 [[17_finegrained_scaling]] Krajewski G=64 的工程实现

- Krajewski 给 1B-10B active 段位 G_opt=16
- Qwen3-Next 选 G = 5632/512 ≈ 11，**接近最优**
- 但 N_routed=512 比 [[37_ling1t]] (256) 多一倍 → 总 capacity 翻倍

### Why K=10 而不是 K=8？

- 10/512 = 1/51（"1:50 activation"，论文宣传口径）
- 8/512 = 1/64（更稀疏，但每 token 路由 path 少 20%）
- **K=10 是 Qwen 团队的经验选择**，但没解释 why

→ 与 Llama 4 K=1 / V3 K=8 / LongCat K=12 一道，K 选择**仍是开放问题**。

## 核心创新 3：head_dim = 256（罕见大头）

| 模型 | head_dim | num_heads | hidden | 备注 |
|---|---|---|---|---|
| V3 / K2 | 192 | 128 / 64 | 7168 | 偏大但 hidden 也大 |
| GLM-4.5 | 128 | 96 | 5120 | 96 头反直觉 |
| Ling 全系 | 128 (固定) | 16/32/64 | 2048/4096/8192 | 标准 |
| **Qwen3-Next** | **256** | **16** | **2048** | **极少头 × 极大头** |
| gpt-oss-120b | 64 | 128 | 2880 | OpenAI 反向选择 |

**X 因为 Y**：
- Qwen3-Next 只有 12 个 softmax attention 层（其他 36 层是 linear），所以**每次 softmax attention 要"做的事更多"** → 加大 head_dim 给每个 head 更高表达力
- 16 heads × 256 dim = 4096 Q 维 vs 2 heads × 256 dim = 512 KV 维 → **8× GQA 压缩，KV cache 极小**
- 256 head_dim 配 RoPE 时，partial RoPE 应该用前 128 维（保留 128 维做 non-positional content）—— **但 Qwen3-Next 是否用 partial RoPE 未公开**

→ 这是 **"少而大"的 attention 设计哲学**，与 GLM "多而中" 的 96 头 / 128 dim 路线对立。

## 核心创新 4：稳定性优化套件

Qwen 团队提到（blog 未给细节）：
- **Zero-centered layernorm**: 跟 LongCat hidden z-loss 类似精神，防止 hidden state mean drift
- **Weight-decayed layernorm**: layernorm 参数也加 WD（OLMo 早期发现的稳定技巧）
- **Gated Attention**: 已述

→ 这些都是 **2025-Q3 之后 100B+ 段位的标配稳定性补丁**，单独效果都小，叠加显著。

## 路由路线：Mixtral 派 (softmax + aux)，不是 V3 派

Qwen 全系（30B / 235B / Coder / Next）都用 **softmax router + aux loss**，与 DeepSeek-V3 派的 **sigmoid + ALF** 路线对立。

| 路由细节 | Qwen3-Next | V3 派 (Ling/GLM/dots1/V3) |
|---|---|---|
| Gate 函数 | softmax | sigmoid |
| Balance | aux loss (coef 0.001) | aux-loss-free bias |
| norm_topk_prob | True (top-K 后 renormalize) | False |
| Routed scaling factor | 无 (softmax 自归一) | 2.5 |
| Group routing | 否 | 是 (V3 device-level) |

→ **Qwen 团队是 100B+ 段位仍坚持 softmax + aux 的最大势力**（GLM/Ling 都切到 V3 派）。这是 routing 流派分歧的最后阵营。

## 1M 上下文机制

Qwen 团队声称 1M context 支持，机制：
1. **训练 32K**（max_position_embeddings = 32768）
2. **推理 YaRN 扩展**到 256K / 1M
3. **Hybrid 架构帮助** —— linear layers 不受 ctx 长度影响，softmax layer 只占 25%

→ **K V cache 估算**：
- 1M ctx × 2 KV heads × 256 head_dim × 12 softmax layers × 2 bytes = **12 GB per sample**
- 完全可行（H100 80GB 单卡能塞）
- 对比 V3 (61 layer × 128 KV × 192 dim, MLA 压缩到 latent 512) 1M ctx KV ≈ 60 GB

## 训练成本声明（未公开数据）

Qwen 团队 blog 说："训练成本 < Qwen3-32B 的 1/10"

推断：
- Qwen3-32B (dense) 训练成本估 ~$8M (假设 8T tokens, H100)
- Qwen3-Next-80B-A3B < $800K
- → 应该训了 ~3-5T tokens（active 3B × 3-5T × 6 ≈ 6e22 FLOPs）

**这与 Qwen3-235B 训 36T 形成 ~50× 差距** —— Qwen3-Next 是**用极稀疏 + 极少 token 试探 efficiency 上限**的实验性路线。

## 与你 16B 设计的对照

| 维度 | 16B Profile B | Qwen3-Next | 借鉴价值 |
|---|---|---|---|
| Attention 类型 | GQA 16Q/4KV | Hybrid (3:1) | **不直接借鉴** —— hybrid 在 16B 段位 ROI 不明 |
| head_dim | 128 (推测) | 256 | Qwen 反向选择，16B 段位用 256 没意义 |
| Expert 数 | 64 | 512 | Qwen 跟 Ling 派同样选大 N，但 K 选 10 |
| Top-K | 8 | 10 | 你的 wind tunnel A2 T2.2 可以加 K=10 arm |
| 路由 | sigmoid + ALF | softmax + aux | **你不会换路由路线** |
| Activation ratio | 15% | 3.75% | Qwen3-Next 远比你稀疏 |
| Zero-centered layernorm | 没考虑 | 是 | **可以加进你的 wind tunnel**（成本几乎 0） |
| Gated Attention | 没考虑 | 是 | 16B 段位收益不明 |

### 唯一值得偷的：Zero-centered layernorm

这是一个 ~30 行 PyTorch 代码就能加的稳定性补丁：

```python
class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, weight_decay=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))  # 零初始化
        self.eps = eps
    def forward(self, x):
        # x: (B, L, D)
        x = x - x.mean(dim=-1, keepdim=True)  # zero-center
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * (1.0 + self.weight)  # 残差形式
```

→ **建议加进 wind tunnel B**（开/关对照，看 loss 改善 ≥ 0.001 即固化）

## 与 Hybrid attention 流派的关系

[[15_jamba]] (Mamba 7:1) + [[13_minimax_01]] / [[14_minimax_m1]] (Lightning 7:1) 是 hybrid 早期。

**Qwen3-Next 是 hybrid 流派的最新一代** —— 选 3:1 比例 + Gated DeltaNet 替换 Mamba/Lightning。

但 **MiniMax-M2 (2025-10) 反向退回 full softmax** —— 说明 hybrid 派内部还在分裂。

**对你 16B 的启示**：
1. **如果你不做 long-context (< 32K) → 不要碰 hybrid**（增加 infra 复杂度 + scaling law 不确定）
2. **如果你做长上下文 (128K+) → hybrid 才有 ROI**
3. **MiniMax-M2 的退回是重要信号** —— hybrid 在 2025 仍未 settled

## Settled vs Open

### Settled (Qwen3-Next 团队内部)
- Hybrid 3:1 比例（Qwen 选这个）
- 512 routed + 1 shared + K=10
- head_dim 256
- Zero-centered layernorm

### Open（全行业仍在分裂）
- Hybrid 比例 (1:7 vs 3:1 vs 0:1) 谁最优
- Gated DeltaNet vs Mamba-2 vs Lightning Attention 谁更适合 MoE
- 1M context 推理质量是否真到位（NIAH 之外的实测）
- Softmax + aux 路由是否会被 V3 派 (sigmoid + ALF) 完全取代

### 已否决（Qwen3-Next 明确不做）
- sigmoid + ALF 路由（Qwen 全系坚持 softmax + aux）
- MLA（Qwen 全系坚持 GQA）
- 1 个 MoE block 多 attn（Qwen3-Next 每层都 MoE）

## 与其他笔记交叉引用

- [[05_qwen3]] —— Qwen3 系前作（Qwen3-30B-A3B, Qwen3-235B 都不用 hybrid）
- [[13_minimax_01]] + [[14_minimax_m1]] + [[15_jamba]] —— hybrid attention 流派老成员
- [[28_open_source_moe_catalog]] entry #26 —— Qwen3-Next 在主表里
- [[42_100b_cookbook]] Step 6 —— attention 选型决策树
- [[17_finegrained_scaling]] —— G_opt 推 Qwen3-Next 的 K=10 与 N=512 配置
- [[36_longcat]] —— zero-experts vs Qwen3-Next 稀疏路线对比
- [[37_ling1t]] —— Ling 全系不用 hybrid，是 Qwen3-Next 的"100B+ 段位 full softmax 派"对照
