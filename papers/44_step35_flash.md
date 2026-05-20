# Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters

- **arXiv**: 2602.10604 (v2, 23 Feb 2026)
- **机构**: StepFun (阶跃星辰)
- **开源**: github.com/stepfun-ai, huggingface.co/stepfun-ai/Step-3.5-Flash
- **前作**: [[16_step3]] Step-3 (321B/38B, 61L, MFA attention)
- **训练规模**: 17.2T tokens / 4096 NVIDIA H800

## TL;DR

Step 3.5 Flash 是 **2026-Q1 "200B 段位 + 极致推理 latency"** 的代表作。三个独特设计：

1. **196B 总 / 11B active / 45L (3 dense + 42 MoE)** —— **真正 200B 段位的 production model**（之前我在 [[38_100b_to_200b_gap]] 标的"200B 真空带"被打破！）
2. **S3F1 Hybrid Attention** —— 3 个 Sliding Window Attention (SWA, W=512) + 1 个 Full Attention 交替，但用 **Augmented Query Heads (64→96)** + **Head-wise Gated Attention** 补回 3:1 SWA 的 quality 损失
3. **Steptron infra + Muon Polar Express + 三大稳定性诊断**（Muon 数值精度 / Expert Collapse / Localized Activation Blow-up）—— 17.2T tokens 只出现 1 次 loss spike

→ **架构对照本文用户问"200B/45L"完全匹配**：196.81B total params，45 layers。

## 核心命题（StepFun 团队的设计哲学）

1. **Inference latency 是 agent 时代的第三个 first-class 约束**（除 intelligence + cost）
2. **SWA 比 Linear Attention 更适合 speculative decoding** —— SWA 保留标准 attention semantics，可以并行 verify；linear attention 的 state-update 让 draft tree generation 复杂
3. **Hardware-aligned GQA-8** —— 8 KV heads 完美匹配 8-GPU 节点的 TP 切分
4. **Hybrid 3:1 SWA + augmented heads 是 "free lunch"** —— prefill/decode FLOPs 几乎不变，但能达到 full attention 水平
5. **大 MoE 的 collapse 有两种**：routing collapse（已知）+ **expert collapse**（新发现，paper 第一次明确）

## 完整 Spec 表

| 维度 | Step 3.5 Flash |
|---|---|
| Total params | **196.81B** |
| Active params | **11B** |
| 激活率 | **5.6% (1/18)** |
| **Layers** | **45** (3 dense + 42 MoE) |
| Hybrid attention layout | **1 Full Attention (leading) + 11 × (3 SWA + 1 Full Attention) = S3F1** |
| 总 attention layer 分布 | **33 SWA + 12 Full = 45** |
| SWA window size | **W = 512** |
| GQA-8 KV heads | **8** (全部 attention layer) |
| **SWA Query heads** | **96** (augmented from 64) |
| Full Attention Query heads | 64 |
| Head_dim | 128 (推断) |
| Hidden | ~5120 (推断) |
| **Routed experts** | **288** (新颖数字，不是 128/160/256) |
| Shared experts | 1 |
| Top-K | **8** |
| Routing | softmax + **loss-free balancing + EP-Group balanced MoE routing** (Eq. 1) |
| Aux loss | EP-level balancing loss (新颖) |
| Normalization | **Zero-centered RMSNorm** (全模型) |
| **MTP heads** | **MTP-3** (3 个 MTP head, 仅 MTP-1 在主训) |
| MTP module 内部 | SWA + dense FFN（不是 MoE） |
| MTP params | 0.81B (~0.41% of total) |
| Vocab | (推断 128K+) |
| FP precision | BF16 + 部分 fp16 (Muon Polar Express) |
| Optimizer | **Muon + Polar Express orthogonalization, N=6 steps** |
| Tokens | **17.2T** |
| Loss spike count | **1** (在 17.2T 全程) |

## 架构核心 1：S3F1 Hybrid Attention 详解

### 4 层重复 motif

```
[Layer 1] Full Attention (leading) ← 唯一开头
[Layer 2] SWA (W=512)  ←┐
[Layer 3] SWA           │ S3F1 block #1
[Layer 4] SWA           │
[Layer 5] Full Attention ←┘
[Layer 6] SWA  ←┐
[Layer 7] SWA   │ S3F1 block #2
... 总共 11 个 S3F1 block
```

→ **Full attention 占 12 layers (27%)，SWA 占 33 layers (73%)**。

### 为什么 3:1 而不是 1:7 (Jamba/MiniMax) 或 7:1？

| 比例 | 代表 | softmax 占比 | retrieval 能力 | throughput |
|---|---|---|---|---|
| 1:7 | Jamba (Mamba) | 12.5% | 弱（论文说 "underperforms"） | 最高 |
| 3:1 | **Step 3.5 Flash** | **27%** (12 full layer) | **接近 full attention** | 中等 |
| 1:0 | V3/K2/Ling | 100% | 最强 | 最低 |

→ Step 3.5 Flash 团队选 3:1 的理由（论文 §2.1）：
- "absence of robust empirical evidence that linear attention yields superior long-context modeling for agentic tasks"
- SWA W=512 已经 "strikes a favorable balance between kernel efficiency and capturing local dependencies"
- **27% full attention 足够维持 retrieval**

⚠️ **跟 Qwen3-Next 3:1 (DeltaNet:Attention) 同比例但不同性质**：
- Qwen3-Next: **linear** attention (DeltaNet) + softmax
- Step 3.5 Flash: **sliding window** softmax + full softmax

→ SWA 是"被截断的标准 attention"，DeltaNet 是"线性 attention" — **两者技术路线完全不同，巧合用同比例**。

### Augmented Query Heads (64→96)：关键的 "free lunch"

**问题**：朴素 S3F1 替换会让 quality 显著掉。

**解法（Table 1）**：把 SWA 层的 query heads 数从 64 升到 96。

| Layout | SWA Heads | Rel. FLOPs Decode/Prefill | Pre-train Avg | LongCtx |
|---|---|---|---|---|
| FFFF (all full) | 32 | 2.68 / 2.90 | 54.1 | 28.8 |
| S1F1 (1:1) | 32 | 1.58 / 1.65 | 54.6 | 29.6 |
| S3F1 (3:1) | 32 | **1.00 / 1.00** | 53.6 | 27.5 |
| **S3F1+Head** | **48** | **1.01 / 1.02** | **55.7** | **28.2** |

→ S3F1+Head 比 FFFF 在 pretrain avg 上 **+1.6 pt**（"surpasses FFFF"），且 FLOPs 几乎一样！

→ **加 SWA query heads 是 "近乎免费的午餐"**：
- attention 一直是 memory-bandwidth bound（不是 FLOPs bound）
- 加 query heads 只增加 compute 但不增 memory traffic → wall-clock 基本不变

**这是本论文最大的工程发现**：S3F1+augmented heads 让 Step 团队真正 "**用 SWA 的代价拿到 full attention 的质量**"。

### Head-wise Gated Attention（替代 sink tokens）

**问题**：SWA 在 input window 没有有用信息时，attention 权重无处安放 → unstable

**经典解法（Sink Tokens, OpenAI 2024）**：人为加入 learnable sink token

**Step 3.5 Flash 解法**：**Head-wise Gated Attention** — 给每个 head 加一个 data-dependent gate，等价于 "可学习 + 数据自适应的 sink token"

实证（Table 2，100B-A10B 控制实验）：
| Method | BBH | MMLU | GPQA | MBPP | C-EVAL | CMMLU | Avg |
|---|---|---|---|---|---|---|---|
| Sink Token | 70.6 | 65.1 | 27.2 | 61.2 | 76.2 | 74.6 | 62.5 |
| **Head-wise Gate** | **73.7** | **67.0** | **28.1** | **62.6** | **77.9** | **77.1** | **64.4** |

→ **+1.97 pt avg**，且 FLOPs/latency 都 negligible。

## 架构核心 2：288 routed experts (非主流数字)

| Routed expert 数 | 模型 |
|---|---|
| 64 | DeepSeekMoE-16B, V2-Lite, Moonlight |
| 128 | dots1, GLM-4.5-Air, gpt-oss-120b |
| 160 | GLM-4.5, V2-236B, Qwen3-Coder |
| 256 | V3, Ling 全系, MiniMax-M2, LongCat (FFN) |
| **288** | **Step 3.5 Flash** |
| 384 | Kimi K2 |
| 512 | Qwen3-Next, LongCat (含 zero) |

→ **288 是 2026 第一个跳出 power-of-2 / 64×N 套路的数字**

**推断 why 288**：
- 288 = 32 × 9 = 8 × 36 = EP=8 时每 rank 36 个 expert
- 288 / 8 (top-K) = 36 → 平均每个 active expert 服务 36 个 token slot
- 不是 256 (V3) 也不是 384 (K2) → **可能在 scaling law sweet spot 上**
- StepFun 团队大概率做了 wind tunnel 找到的最优值（论文未明确）

## 架构核心 3：EP-Group Balanced MoE Routing (Eq. 1)

**问题**：device-level balance loss 在 micro-batch 内强制均匀 → 但 EP 跨 rank 时可能 stragglers（少数 rank 负载重）→ 拖慢整体训练

**Step 解法**：在 EP-level 加一个 group balance loss

$$
p_e = \frac{1}{T} \sum_t p_{t,e}, \quad f_e = \frac{1}{TK} \sum_t s_{t,e}, \quad p_g = \sum_{e \in \mathcal{E}_g} p_e, \quad f_g = \sum_{e \in \mathcal{E}_g} f_e, \quad \mathcal{L}_{EP} = G \sum_g f_g p_g
$$

其中：
- 把 expert 按 EP rank 分成 G 个 group
- $p_g, f_g$ = group g 的总路由概率/频率
- $\mathcal{L}_{EP}$ 强制 group-level 均匀

→ 比 device-level loss 更"粗粒度"，允许 expert 内部专精，但 group 整体平衡 → **既避免 expert collapse，又避免 EP stragglers**

→ 这是 [[36_longcat]] PID controller + [[37_ling1t]] zero-mean bias 之外的**第三条 ALF balance 改良**。

## 训练稳定性：三大新发现

### 4.1.1 Numerical Sensitivity of Muon

**问题**：Muon 用 Newton-Schulz iteration 正交化，**Polar Express 是更快的 NS 变种**（N=6 steps 而非 5）但在 bf16 下偶发不可恢复 loss spike

**Step 团队诊断**：simulations 显示 bf16 Polar Express 在某些 update 统计下会产生 extreme intermediate outliers（cumulative error in addition）

**解法**：**仅把 Polar Express 迭代部分 cast 到 fp16**，其他保持 bf16

→ 修复后 17.2T tokens 全程**只剩 1 个 loss spike**（Figure 3 标注 ④）

→ 这是 [[39_muon]] 笔记里 "Muon 1T+ 段位风险"的实证 + 解法。

### 4.1.2 Expert Collapse Beyond Routing Collapse（新概念！）

**Step-3 (前作 [[16_step3]]) 报告**："dead experts" — expert 收不到 token

**Step 3.5 Flash 新发现**：**Expert-side collapse** 即使 router dispatch 看起来 OK 也会发生：
- Routed-expert aggregation 与 shared expert 之间缺乏 explicit scaling factor → routed expert contribution 被压制
- Micro-batch level balance loss 太严格 → 阻碍 expert specialization

**解法**：
1. **Broader-scope balancing**（global-batch statistics 而非 micro-batch）→ 跟 V3 / Ling 一致
2. **Loss-free bias adjustment** + 监测 **per-expert activation norm 和 parameter norm**（不是只看 router stats）
3. **早期 warning signal**：min-to-median ratio 下降 = expert 在死亡

→ **对你 16B 设计的启示**：wind tunnel 不能只监控 router dispatch（你 22_FINAL §10 当前规划），要加监 expert FFN output norm

### 4.1.3 Localized Activation Blow-up in MoE Layers（更新的发现）

**Step 团队观察**：在**深层 MoE layer**（如 Layer 38, 45）一小部分 expert 的 activation norm 急剧增长，**而 training loss 看起来稳定** → 没人发现

**直接证据 (Figure 4)**：
- Layer 38 (中间层): max & median activation 稳定
- **Layer 45 (深层): max activation 与 median 差距随训练**指数级**扩大**

**两种 mitigation**：
1. **Weight clipping on expert projections**: 如果 $\max_x \|Wx\| > \tau$，rescale $W \leftarrow W \cdot \frac{\tau}{\max_x \|Wx\|}$。**类似 MuonClip in attention** ([[06_kimi_k2]]) 但用在 expert FFN。**Offline checkpoint clipping**, 不是 on-the-fly
2. **Activation clipping inside experts**: 在 MoE FFN intermediate activation 上 element-wise clip

→ **对 1T 段位 MoE 训练这几乎是 must-have**。Step 自己的实证。

→ 这对你 16B 段位也是 free 防护：加 weight clipping 几乎 0 成本，但能避免后期突发训练崩溃。

## 训练系统

| 维度 | 值 |
|---|---|
| GPU 数 | 4096 H800 |
| 节点拓扑 | 8 GPU/node × 512 nodes |
| 节点内互联 | NVLink + NVSwitch |
| 节点间互联 | **8 × 200 Gbps RoCE** (RoCE 而不是 InfiniBand！) |
| Framework | **Steptron** (StepFun 自研, 基于 PyTorch + Megatron-LM) |
| Parallelism | **PP=8 (VPP) + EP=8 + ZeRO-1 DP** |
| Attention/MoE 并行 | **Decoupled** — attention 与 MoE 用不同 DP group |
| Communication | **fabric-aware scheduling**（NVLink intra-node + RoCE inter-node 分相）+ **rank placement optimization** |
| Optimizer | Muon (Polar Express N=6) + AdamW for 非 expert params |
| **Muon ZeRO-1 resharding** | **整 parameter 给 DP rank** + reduce-scatter（避免 naive all-reduce 双倍通信） |
| Iteration time improvement | ~5% 来自 communication optimization |

→ **RoCE 而非 IB** 是 Step 3.5 Flash 的特别之处 — 中国 H800 cluster 因出口管制部分用 RoCE。论文 explicitly 优化这一点。

→ **PP=8 + EP=8 + ZeRO-1 DP** 是 200B 段位的合理拓扑（[[42_100b_cookbook]] Step 12 推荐）

## MTP 设计（不同于 V3 / Ling）

- **3 个 MTP head** (MTP-1, MTP-2, MTP-3)
- 每个 MTP head = 1 个 SWA + 1 个 dense FFN（**不是 MoE！**）
- **训练时只优化 MTP-1**（主训阶段）
- 后期 light fine-tune 阶段：把 MTP-2/3 从 MTP-1 clone，联合训
- 总 MTP params: 0.81B (~0.41% of total)
- **Position-dependent loss reweighting**（Fast-MTP 风格）防止过度优化远距 token

→ 跟 V3 (D=1) 和 GLM (D=1) 不同 — **Step 3.5 Flash D=3 是少数派**（之前只有 MiniMax-M2）

→ 跟 LongCat ([[36_longcat]]) 一样选 **dense MTP** 而非 MoE MTP。

## RL 算法：MIS-PO（新算法）

**Metropolis Independence Sampling-Filtered Policy Optimization (MIS-PO)** —— GRPO 改进

**核心想法**：用 **discrete distributional filtering** 替代 continuous importance weighting，把样本限制在 stable trust region。

→ 跟 V3.2 ([[41_dsa]]) 的 Off-Policy Sequence Masking 思想类似，但用 Metropolis sampling 框架。

→ 对 long-horizon reasoning RL 更稳定（因为 token-level discrepancy 累积成 high-variance gradients 在 MoE 上特别严重）

## 200B 段位的位置（颠覆 [[38_100b_to_200b_gap]] 部分结论）

**我之前在 [[38_100b_to_200b_gap]] 说 "200B 段位是真空带"**，给出 4 个原因。**Step 3.5 Flash 部分推翻这个判断**：

| 我之前的论点 | Step 3.5 Flash 的反例 |
|---|---|
| 200B 跟 100B benchmark 拉不开差距 | **96.4% IMO + 85.4% AIME 25 + 86.4% LiveCodeBench** — 同时打过 GLM-4.5 (355B) 和 K2 (1T)|
| 200B 训练成本翻倍但价格不能翻倍 | Step 自家定价能力够大（StepFun 是腾讯+上海政府背景） |
| 200B EP/PP 拓扑跳档 | **真证明了 PP=8+EP=8 在 196B 上 work**（之前预测） |
| Qwen3-235B 被放弃 = 段位被弃 | **Step 3.5 Flash 证明 200B 可以做出 SOTA** |

→ **更新理解**：200B 段位**没有死**，而是**等到有团队把 "200B + agent latency 优化" 路线想透**才能 work。Step 3.5 Flash 是这个段位的破局者。

→ 我应该更新 [[38_100b_to_200b_gap]] 备注新证据。

## 与 Step-3 ([[16_step3]]) 的对比

| 维度 | Step-3 (前作) | **Step 3.5 Flash** |
|---|---|---|
| 总参 | 321B | **196.81B** (砍 39%) |
| Active | 38B | **11B** (砍 71%) |
| 激活率 | 11.8% | **5.6%** (1/18, 更稀疏) |
| Layers | 61 | **45** (砍 26%) |
| Hidden | 7168 | ~5120 (减 28%) |
| L/√H | 0.72 | ~0.63 |
| Attention | MFA (Multi-Matrix Attention, 新颖) | **S3F1 SWA Hybrid + Augmented Heads** |
| Routed experts | 48 | **288** (6×) |
| Top-K | 3 | **8** |
| Shared | 1 | 1 |
| MTP | 无 | **D=3** |
| 训练 tokens | ? | **17.2T** |

→ Step-3 → Step 3.5 Flash 的演进是 **"更稀疏 + 更细粒度 + 更短宽 + 更优 attention"** —— 完全符合 [[42_100b_cookbook]] 趋势

→ **MFA attention 被 SWA 3:1 hybrid 取代** —— Step 团队自己放弃了前作的核心创新

## 与你 16B 设计的关系

| 维度 | 16B Profile B | Step 3.5 Flash | 借鉴价值 |
|---|---|---|---|
| 段位 | 16B | 196B | 不可比 |
| Attention | GQA 16Q/4KV | S3F1 SWA Hybrid | 16B 段位 hybrid ROI 不明 |
| Head-wise Gated Attention | 没考虑 | 是 | **强烈建议加进 wind tunnel B**（+2pt 几乎免费） |
| Zero-centered RMSNorm | 没考虑 | 全模型用 | **可以加** (跟 [[40_qwen3_next]] 一样推荐) |
| Weight clipping on expert | 没考虑 | offline | **wind tunnel C 候选** — 防 late-training expert blow-up |
| Activation norm monitoring | 没考虑 | 是 | **必加 wind tunnel monitoring** (防 expert collapse) |
| EP-Group balancing | 不需要 (EP=8 单节点) | 是 | 16B EP 内不跨 node，不必加 |
| Muon Polar Express | 没考虑 | 是 (fp16 cast) | 如果你用 Muon, fp16 cast 必须 |

## Settled vs Open

### Settled (Step 团队自己验证)
- S3F1 + 96 SWA heads + Head-wise Gate = "free lunch" 拿到 FFFF 质量（Table 1）
- Head-wise Gate > Sink Tokens (Table 2, +1.97pt)
- EP-Group balance loss 比 device-level 更适合 fine-grained MoE
- 17.2T tokens 单 loss spike = Muon Polar Express + fp16 cast 是工作组合
- Expert collapse 监测必须包括 expert-side norms（不只 router stats）

### Open
- 288 routed expert 是否真比 256 / 384 优 (Step 没给 ablation 单独验证 288)
- S3F1 比 1:7 (Qwen3-Next/Jamba) 哪个更好（不同评测，没直接对比）
- Polar Express vs 原始 Newton-Schulz 在 long-training 是否真值（Step 选 N=6 vs Moonlight N=5）
- MIS-PO 在不是 long-horizon reasoning 上是否仍有收益

### 已否决（Step 3.5 Flash 明确不做）
- Linear attention (DeltaNet/Mamba/Lightning) — speculative decoding 不兼容
- MoE MTP module（用 dense MTP）
- Naive S3F1 (没有 Augmented Heads + Head-wise Gate)
- Micro-batch level LBL (用 EP-Group + loss-free balancing)
- MFA attention (前作创新被放弃)

## 一句话总结

> **Step 3.5 Flash 是 "200B + Agent Latency 优化" 的破局者**：用 S3F1 SWA Hybrid + Augmented Heads + Head-wise Gate 拿到 full attention 的质量，但 prefill/decode FLOPs 砍 60%；用 288 routed + 1 shared + EP-Group balance 拿到 sparse MoE 的容量；用 Muon Polar Express + fp16 cast + 三大 stability 诊断让 17.2T tokens 只出 1 个 loss spike。**这是 2026 Q1 性价比最高的 production-grade reasoning MoE**。

## 与其他笔记交叉引用

- [[16_step3]] —— Step-3 前作（架构完全不同）
- [[38_100b_to_200b_gap]] —— "200B 真空带"判断被 Step 3.5 Flash 部分推翻
- [[40_qwen3_next]] —— 同样 3:1 hybrid 但用 DeltaNet 而非 SWA
- [[15_jamba]] / [[13_minimax_01]] —— 1:7 hybrid 是另一个极端
- [[39_muon]] —— Muon 优化器，Step 用 Polar Express 变种
- [[06_kimi_k2]] —— MuonClip post-QK-clip，Step 3.5 Flash 类似思想 (weight clipping on expert)
- [[36_longcat]] —— 同样用 dense MTP + 同样 hidden z-loss 思想（Step 用 activation norm 监控）
- [[37_ling1t]] —— EP-Group balance vs Ling zero-mean bias 是两条 ALF 改良路线
- [[42_100b_cookbook]] Step 6 (attention) + Step 8 (MTP) + Step 12 (parallelism) 都需要根据 Step 3.5 Flash 更新
- [[03_auxloss_free]] —— ALF + EP-Group balance 是新组合
- [[41_dsa]] —— MIS-PO vs Off-Policy Sequence Masking 是不同的 RL 稳定路线
