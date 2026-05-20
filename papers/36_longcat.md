# LongCat-Flash Technical Report

- **arXiv**: 2509.01322 (v2, 2025-09)
- **机构**: Meituan LongCat Team
- **开源**: huggingface.co/meituan-longcat, github.com/meituan-longcat
- **同源**: longcat.ai (chat)

## TL;DR

LongCat-Flash 是 **560B/27B-avg 的"极短宽 + 动态 active"** MoE 模型，三个独立的反主流创新：

1. **零专家 (Zero-Computation Experts)** —— 256 个 identity-routing slot 加入 expert 池，让 token 可以"跳过"FFN 计算，**让 active params 从 18.6B 动态变化到 31.3B**（27B 均值，但单 token 极差 1.7×）
2. **Shortcut-Connected MoE (ScMoE)** —— 跨层 shortcut 让前 block 的 dense FFN 与当前 block 的 all-to-all dispatch/combine **并行执行**，TPOT 比 V3 减半
3. **28 层 / hidden 6144 / 512 experts 极短宽**（L/√H = 0.36，全市场最短宽）+ **每层 2 个 MLA block + 多个 FFN block 异构组合**

训 20T+ tokens / 30 天 / 数万张 H800 / 98.48% 时间在线（zero manual intervention for fault resolution）。推理 100+ TPS / $0.7 per 1M output tokens。

## 核心命题

1. **不是所有 token 都需要相同 active params** —— 简单 token（function words, punctuation）路由到 zero-experts 几乎不损质量；困难 token 自动多激活 FFN expert
2. **MoE 通信瓶颈可以靠 cross-layer shortcut 隐藏** —— 不需要等 all-to-all 完成再算 FFN
3. **极短宽 (28L) + 极多 expert (512+256) + 极高 top-K (12) 在 560B 段是可行的** —— 反 GLM-4.5 派
4. **MLA 在大模型上需要 variance correction**，否则 init 阶段 attention 不稳定 —— 第一篇明确提出这个问题的论文

## 完整 Spec（§2.4 Model Information）

| 维度 | LongCat-Flash | DeepSeek-V3.1 | Kimi-K2 | Llama-4-Maverick |
|---|---|---|---|---|
| Total Params | **560B** | 671B | 1043B | 400B |
| Activated Params | **18.6B-31.3B (avg 27B)** | 37B | 32B | 17B |
| Layers | **28** | 61 | 61 | 48 |
| Hidden | **6144** | 7168 | 7168 | 5120 |
| **L / √Hidden** | **0.36** (极短宽) | 0.72 | 0.72 | 0.67 |
| MoE Intermediate (expert FFN) | **2048** | 2048 | 2048 | 8192 |
| Dense FFN Intermediate | **12288** | 18432 | 18432 | 16384 |
| **每层 MLA block 数** | **2** | 1 | 1 | 1 |
| Attention Type | **MLA + α 校正** | MLA | MLA + MuonClip | GQA + iRoPE |
| Attention Heads | **64** | 128 | 64 | 40 |
| Head Dim | **128** | 192 | 192 | 128 |
| KV Compression (MLA) | **512** | 512 | 512 | – |
| Query Compression (MLA) | **1536** | 1536 | 1536 | – |
| Routed Experts | **512** | 256 | 384 | 128 |
| **Zero-Computation Experts** | **256** | 0 | 0 | 0 |
| **Top-K (含 zero)** | **12** | 8 | 8 | 1 |
| Shared Experts | 0 (但 dense FFN 起类似作用) | 1 | 1 | 1 (shared MLP) |
| Vocab | 131072 | 129280 | 163840 | 202048 |
| MTP | **1 dense layer**（不是 MoE！） | 1 MoE | 0 | 0 |

→ **L/√H = 0.36 是全市场最短宽**（OLMoE 0.35 在 7B 段，LongCat 在 560B 段独此一份）

→ 64 head × 128 dim = 8192 Q 维度（V3 是 128×192=24576），KV 压缩到 512 → KV cache 是 V3 的 ~1/2

→ **每层 2 个 MLA block** 是反常规设计（Figure 2 架构图），不是 attention-FFN-attention-FFN 重复，而是 attention-FFN-attention-MoE+shortcut 这种异构

## 三大创新详解

### 1. Zero-Computation Experts（§2.1）

**核心想法**：在常规 N=512 个 FFN expert 旁边，加 Z=256 个 "什么都不做" 的虚拟 expert，输出 = 输入：

$$
\text{MoE}(x_t) = \sum_{i=1}^{N+Z} g_i E_i(x_t), \quad
E_i(x_t) = \begin{cases} \text{FFN}_i(x_t), & 1 \le i \le N \\ x_t, & N < i \le N+Z \end{cases}
$$

Router 是 softmax over N+Z=768 个 logits + bias，top-12 选择。如果 token 落在 zero-expert，等价于 identity（这一槽不产生 FFN FLOPs）。

**结果**：average active FFN expert 数 = 8（不是 12），即 ~33% 的 routing slot 跳过计算。但**std = 3**（Figure 3c），意味着不同 token 之间 active expert 数差异极大（"4 个" 到 "12 个" 都常见）。Active params 因此动态：18.6B-31.3B。

#### 1.1 Computational Budget Control (PID controller)

为了让 model **学会** 给重要 token 多分配 FFN，必须控制 zero-expert 的平均 selection ratio。LongCat 用 PID controller 调 bias：

$$
\Delta b_i = \begin{cases}
\mu \cdot (K_e/K \cdot 1/N - T_i / (K T_{\text{all}})), & 1 \le i \le N \\
0, & N < i \le N+Z
\end{cases}
$$

- $K_e$ = 目标平均 FFN 激活数（< K=12）
- $T_i$ = 第 $i$ expert 收到的 token 数，$T_{\text{all}}$ = global batch tokens
- $\mu$ = bias adaptation rate（论文用 decay schedule）

⚠️ **关键**：**zero-experts 自身不参与 bias 更新**（identity 性质自动满足均匀分布约束）。这是 LongCat 团队的独创发现。

PID 把 V3 的 "静态 bias rate" 升级为 "negative-feedback controller"。这是 100B+ MoE 路由里**第一次引入 control theory**。

#### 1.2 Load Balance Loss（§2.1.2）

device-level group balance loss，把 D 个 device group 各算一项：

$$
\mathcal{L}_{\text{LB}} = \alpha \sum_{j=1}^{D+1} f_j P_j
$$

其中 zero-experts 单独归一组 $j=D+1$，频率 $f_{D+1} = \frac{1}{(K-K_e)T} \sum \mathbb{1}(\text{select zero})$。

→ 比 V3 device-level loss 多一组 zero-expert，让 zero/FFN 比例稳定收敛到 $K_e/(K-K_e)$。

### 2. Shortcut-Connected MoE (ScMoE, §2.2)

**问题**：MoE 层的 all-to-all dispatch/combine 是顺序阻塞 —— 必须先把 token 路由到 expert 所在的 device 才能算 FFN。导致 device 在等通信时空闲。

**ScMoE 的解法**：用 cross-layer shortcut 重排执行顺序，让 **前 block 的 dense FFN 与当前 block 的 dispatch/combine 并行**。

Figure 2 架构（一层 = 两个 ScMoE block）：

```
Input
  ↓
MLA → FFN ──→ ┐ (dense FFN 算)
         ↓     │
         ↓    并行
       MoE ←──┘ (all-to-all + expert FFN)
         ↓
       MLA → FFN ──→ ┐ (dense FFN 算)
              ↓       │
              ↓      并行
            MoE ←────┘
         ↓
       Output
```

效果（§2.2 末尾原文）：
- "**reducing the theoretical Time-Per-Output-Token (TPOT) by nearly 50%** compared to leading models such as DeepSeek-V3"
- "allows for the **concurrent execution of distinct communication operations**: intra-node TP (NVLink) and inter-node EP (RDMA)"

→ 这是 LongCat 能跑 **100+ TPS 推理**的核心（同 H800 跑 V3 大约 50 TPS）。

⚠️ 但 ScMoE 不开源 kernel，社区无法复刻。

### 3. Variance Alignment Design（§2.3）—— MLA 大规模时的稳定性补丁

LongCat 团队发现：MLA 的 query/key 在 init 阶段方差不匹配，导致大模型容易不稳。原因：
- $\sigma^2(q^C_t), \sigma^2(q^R_t) \propto d_q$（compressed query 维度）
- $\sigma^2(k^C_t) \propto d_{kv}$（compressed KV 维度）
- $\sigma^2(k^R_t) \propto d_{\text{model}}$（rotary key 用 full hidden）
- → 三者维度不同 → init 方差不匹配 → attention score 不稳

**解法**：在 MLA compression 步骤前加 scale factor：
$$
\alpha_q = \sqrt{d_{\text{model}}/d_q}, \quad \alpha_{kv} = \sqrt{d_{\text{model}}/d_{kv}}
$$

把 compressed Q/KV 的方差 rescale 到 $d_{\text{model}}$ 同量级。

→ Figure 5(a) 显示在 1B active MoE 上 loss 从 2.65 降到 2.55（有意义的 absolute 改进）。

**额外补丁**：fine-grained expert init compensation
$$
\text{MoE}(x_t) = \gamma \sum_{i=1}^{mN} g_i E_i(x_t), \quad \gamma = m
$$
当一个 expert 被切成 $m$ 个细粒度子 expert 时，gating dilution（gate prob /= m）+ dimensional reduction（每 expert FFN 维度 /= m）双重让 init 方差 /= $m \cdot m$。补一个 $\gamma = m$ 还原。

## 训练稳定性 (§3.1.3)

三个独立技巧：

### Router stability：监控 gradient norm ratio
$$
R_g = \frac{\| \alpha \nabla_{\vec{P}} \mathcal{L}_{\text{LB}} \|_2}{\| \nabla_{\vec{P}} \mathcal{L}_{\text{LM}} \|_2}
$$
保持 $R_g < 0.1$。如果 LB loss 梯度太大压过 LM 梯度，所有 expert 路由权重会收敛到同一点（"expert collapse"）。

### Hidden z-loss：抑制 massive activation
$$
\mathcal{L}_Z = \frac{\lambda}{T} \sum_{t=1}^{T} \big(\log \sum_{i=1}^{|z_t|} \exp(\text{abs}(z_t^i))\big)^2
$$
$\lambda = 10^{-7}$（极小系数），但能压住 hidden state L2 norm 不爆炸（Figure 6 直接对比 norm 7 个 OOM 差距）。

→ **这是 100B+ BF16 训练 must-have**。OLMo 团队、DeepSeek 团队都遇到过 massive activation 但用不同 hack 解（DeepSeek 用 FP32 部分计算，OLMo 用 z-loss）。LongCat 选 z-loss 路线。

### Adam ε = 1e-16（不是 1e-8）

OLMo 论文已经发现 ε=1e-8 优于 1e-5（默认），LongCat 进一步推到 **1e-16**。原因：

- Figure 7 实证：大模型 gradient RMS 极小（10⁻⁵ 量级）
- 当 ε 接近 gradient RMS → Adam 的 second-moment 校正失效 → loss 立刻劣化
- ε 比 gradient RMS 小几个 OOM 就够，不要担心数值下溢

→ **这是你 16B wind tunnel 里 ε 选择的有力锚点**：1e-16 ≫ 1e-20 (DeepSeek) ≫ 1e-8 (OLMo) 的频谱中 LongCat 选最右侧。

## Hyperparameter Transfer + Model Growth

### width scaling factor s=8 transfer
proxy model 用 width 768，target model 用 width 6144 → scaling factor s=8。然后：
- $\sigma^2_{\text{target}} = \sigma^2_{\text{proxy}} / s$
- $\eta_{\text{target}} = \eta_{\text{proxy}} / s$（hidden/unembedding）
- $\sigma^2, \eta$ for embedding 不变

→ 在小模型上做 grid search，把最优 LR / init variance 通过理论规则 transfer 过去。**这是 μP / Tensor Programs IV 派的工程化应用**。

### Model growth (r=2)
- 先训 **14 层** 模型 on tens of B tokens
- 然后 **stack 2 倍** 变 28 层
- 保留所有训练状态（sample counter, LR schedule）继续训

→ Figure 5(b)：6B active 实验显示 growth init 初期 loss 反弹然后超过 random init baseline。

⚠️ 注意论文警告："over-optimizing predecessor models may negatively impact token efficiency in target models" —— small model 不能训太久，否则 stacking 后 capacity 被锁死。

## 训练 / 推理基础设施 (§5-6)

- **Deterministic computation** —— SDC（Silent Data Corruption）detection 必备
- **FP8 inference**, **BF16 training**
- 没有公开 TP/PP/EP 数值，但说支持 "tens of thousands of accelerators"（H800）
- 推理用 **Single Batch Overlap pipeline**（ScMoE 启用）

### 长上下文扩展
1. 8K → 32K: 80B tokens, RoPE base 1e6 → 5e6
2. 32K → 128K: 20B tokens, RoPE base 5e6 → 1e7

## 与 16B / 100B 设计的对照

| 维度 | 你的 16B Profile B | LongCat 选择 | 借鉴价值 |
|---|---|---|---|
| 深度 vs 宽度 | 27L / 2048 (L/√H=0.60) | 28L / 6144 (L/√H=0.36) | LongCat 路线**只在 500B+ 才合理** —— 16B 不要学 |
| Zero-experts | 没考虑 | 256 个 | **可作为 wind tunnel D 候选** —— 容易实现，但 PID controller 难调 |
| MLA variance correction | 你不用 MLA | α 校正必备 | **如果你未来上 MLA 必须加** |
| Hidden z-loss | 没用 | λ=1e-7 | **强烈建议加进 16B**（成本几乎 0，防爆 OOM） |
| Adam ε | 默认（1e-8？） | 1e-16 | **wind tunnel 里加进选择网格** |
| MTP | wind tunnel 待验 | 1 dense layer | 反 V3 的 "MoE MTP"，LongCat 走 dense → 启示 |
| Model growth | 没考虑 | 14L → 28L | **wind tunnel B 候选** |
| ScMoE | 不适用 | 跨层 shortcut | 16B EP=8 单节点不需要（all-to-all 几乎免费） |

## 设计哲学：efficiency-first 而非 capability-first

LongCat 团队的**核心信念**（论文 §1）："algorithmic design, underlying system optimizations, and data strategy all play equally critical roles in further pushing the frontier of scalable intelligence"。

具体表现：
- **不追总参数**（560B < V3 的 671B）
- **不追 active params**（27B < V3 的 37B）
- **追每美元的能力**（$0.7/M output token，业界 SOTA 之一）
- **追 zero-intervention 训练**（30 天 98.48% uptime）

→ 对 **production-grade MoE** 团队是最值得对标的论文。
→ 对 **research-grade** 团队（追 benchmark）可能不那么对口。

## Settled vs Open

### Settled
- Hidden z-loss + Adam ε 极小（应被 100B+ 团队普遍采用）
- MLA variance correction（如果用 MLA 就必须）
- Model growth (r=2) 在 large MoE 上有效

### Open
- Zero-experts 在中等规模（16B-100B）有没有正收益（论文只在 560B 段验证）
- PID controller 的稳定性 hyperparams 怎么 transfer 到不同规模
- ScMoE 在 PP=4-8 拓扑下 throughput 收益（论文没分 PP 数据）
- 28L 极短宽是不是真的 reasoning 没退化（论文 GPQA 51.09 跟 V3 47.16 持平 → 看似没退化，但 GPQA 不代表所有 reasoning）

### 已否决（LongCat 团队明确不做）
- Shared experts（用 dense FFN + ScMoE 替代）
- 标准 1 attn + 1 FFN block 结构（用异构 2 MLA + N FFN）
- MoE MTP（用 dense MTP）

## 与其他笔记交叉引用

- [[28_open_source_moe_catalog]] entry #56
- [[32_depth_width_tradeoff]] §2 表格中 LongCat 是最短宽极端
- [[03_auxloss_free]] —— ALF 是 LongCat PID controller 的前身
- [[04_deepseek_v3]] —— MLA / EP 配置的对标
- [[27_mhc]] —— LongCat 跟 mHC 都是想稳大模型，路线完全不同（z-loss vs Sinkhorn）
- [[38_100b_to_200b_gap]]（待写）—— LongCat 是"跳过 200B 直接做 560B 的"代表
