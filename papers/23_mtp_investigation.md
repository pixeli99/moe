# MTP for ~16B MoE：是否纳入主 spec 的深度调研

> 调研目的：本设计目标 ~16B total / ~2.4B active 落在 Gloeckle (2024) 验证的"边界区"（≤1.3B 拖累、≥6.7B 显效），而 V3 / Ling 2.0 / GLM-4.5 等 MoE 的 MTP 验证都在更大 active scale 上。本文盘点公开证据、明确**未知**与**已知**，给出 pilot-first 的决策框架。

---

## TL;DR

1. **MTP 在 2-3B active 是 boundary case**。Gloeckle Fig.3 直接证据；V3/Ling 实证都在 ≥30B active。
2. **MoE 模型采用 MTP 的情况是分裂的**：V3 / Ling 2.0 / GLM-4.5 (推测) 采用；**Qwen3 / Kimi K2 / Hunyuan-Large / Mixtral / MiniMax 都不采用**。两条路线都有 SOTA 模型背书。
3. **训练成本约 +5–10%**（额外一个 transformer block + projection），推理可丢、零部署成本；保留则做 self-speculative decoding，V3 报告 1.8× 加速。
4. **下游 gain 在 V3 (37B active) 报告 +0.5–1pt**，2.4B active 上无直接证据。
5. **建议**：**移出主 spec，放进 wind tunnel A2 (1B/200M active/25B tokens) anchor 做 D=0 vs D=1 对照**；决策树见 §8。

---

## 1. MTP 是什么 — 两种主流变体

### Gloeckle 2024 ([2404.19737](https://arxiv.org/abs/2404.19737)) — n 个 parallel heads

- **架构**：共享 trunk + n 个并行的 transformer-layer heads + 共享 unembedding。所有 n 个 head **同时** 输出 t+1...t+n 的预测。
- **Loss**：n 个 head 各自 CE 求和（每 head 权重 1/n）。
- **典型 n**：32K vocab + code → n=4；byte-level → n=8。
- **推理**：默认丢 i>1 头；可保留做 block-wise speculative decoding（3× on text，6.4× on byte-level）。

### V3 风格 ([2412.19437](https://arxiv.org/abs/2412.19437) §2.2) — D=1 causal chain

- **架构**：D 个 sequential MTP module，每个 = 1 个 Transformer block `TRM_k` + 投影矩阵 `M_k ∈ R^(d×2d)` + 共享 embedding & output head。
- **关键差异**：每个 module 用 **上一 token 的预测 + 下一个 token 的 embedding** 拼接，**保持 causal chain**（而不是并行独立预测）。
- **Loss**：`L_MTP = (λ/D) Σ_k L_MTP^k`，**V3 λ schedule：前 10T tokens λ=0.3，后 4.8T λ=0.1**。
- **D=1**：V3 的实际选择；只预测 t+2（"下下个 token"）。
- **推理**：默认丢 MTP module；也可保留做 speculative decoding，论文宣称 **80%+ acceptance rate、1.8× 端到端加速**。

### 两者对比

| 维度 | Gloeckle (D=4 parallel) | V3 (D=1 causal) |
|---|---|---|
| 并行 vs 顺序 | n 个 head 并行 | n 个 module 顺序（causal chain） |
| 预测目标 | 同时 t+1...t+n | sequential，每步 conditioned on 上一步 |
| 训练复杂度 | 中（n 个 head 同时 backward，需 memory-efficient trick） | 高（每 module 1 个 transformer block） |
| 训练 overhead | ~少（heads 是单层） | ~5-10%（每 module 是完整 block） |
| 推理 spec decoding | 3-6× | 1.8× |
| 推理参数开销 | 较小（n-1 个 head 各 1 层） | 较大（D 个 transformer block） |
| 主要用途 | sample efficiency + inference speed | 训练 auxiliary task + 可选 spec decoding |

**对 16B MoE 关心的是 V3 风格**。Gloeckle 没在 MoE 上单独验证；V3 风格已在 671B MoE 上工程化。

---

## 2. 公开 MoE 模型的 MTP 采用盘点

| 模型 | Total / Active | MTP? | 配置 | 出处 |
|---|---|---|---|---|
| DeepSeek-V3 | 671B / 37B | ✅ | D=1, λ=0.3→0.1, causal chain | 2412.19437 §2.2 |
| DeepSeek-R1 | 671B / 37B | ✅ (继承 V3) | 同 V3 | 2501.12948 |
| Ling-mini-2.0 | 16B / 1.4B | ✅ | D=1, dense head, weight=0.1 | 2510.22115 |
| Ling-flash-2.0 | 103B / 6.1B | ✅ | D=1, dense head, weight=0.1 | 同 |
| Ling-1T | 1T / 51B | ✅ | D=1, dense head, weight=0.1 | 同 |
| GLM-4.5 (355B Air / Large) | 355B+ | ❓ | 摘要未提，PDF 全文未确认 | 2508.06471 |
| Qwen3-30B-A3B | 30B / 3.3B | ❌ | 无 MTP head | 2505.09388；HF config |
| Qwen3-235B-A22B | 235B / 22B | ❌ | 无 MTP head | 同 |
| Kimi K2 | 1T / 32B | ❌ | HF config 无 `num_nextn_predict_layers` | 2507.20534 |
| Hunyuan-Large | 389B / 52B | ❌ | 论文未提 | 2411.02265 |
| Mixtral 8×7B / 8×22B | 47B/13B, 141B/39B | ❌ | 无 | 2401.04088 |
| MiniMax-01 / M1 | 456B / 45.9B | ❌ | 无 | 2501.08313 |
| Skywork-MoE | 146B / 22B | ❌ | 无 | 2406.06563 |
| Yuan 2.0-M32 | 40B / 3.7B | ❌ | 无 | 2405.17976 |
| Jamba | 52B / 12B | ❌ | 无 | 2403.19887 |
| OLMoE | 6.9B / 1.3B | ❌ | 无 | 2409.02060 |

**观察**：
- **DeepSeek 系（V3 / R1 / 衍生）+ Ling 系（蚂蚁）一致用 D=1 MTP**。
- **Qwen / Moonshot / Mistral / Tencent / MiniMax / OLMo 一致不用**。
- 这不是技术对错的问题，是路线分歧。**两条路线都跑出 SOTA**。

---

## 3. 实证证据按 model scale 排序

### Gloeckle Fig.3（dense baseline，最关键的 scale-dependent 证据）

200B tokens of code，n=4 vs n=1 baseline，MBPP pass@1 的**相对差异**：

| Dense size | Δ vs n=1 baseline |
|---|---|
| 0.3 B | **−1.7**（MTP 拖累） |
| 0.6 B | 0.0 |
| 1.3 B | +0.1 |
| **3 B** | **+2.0**（刚开始正向） |
| 6.7 B | +3.7 |
| 13 B | **+4.5**（绝对 26.0 → 30.5） |

**结论**：boundary 在 ~1.3B–3B 之间。小于 1B 时 MTP 反而拖累；大于 6.7B 时稳定正向。

> ⚠️ **我们 spec 的 2.4B active 直接落在这个 boundary**。Gloeckle 没有 MoE 验证，但 MoE 的"等效 dense 容量"普遍被认为接近 active params（OLMoE / Krajewski 论文都用此假设）。

### V3 §4.5.1 (Ablation Studies for MTP)

- 论文存在这张表，公开摘要写 "MTP enhance the overall performance"。
- 我尝试 WebFetch 直接拉具体数字 — **arxiv HTML 在 §3.3.2 处截断**，没能拉到 §4.5.1 完整数字。
- 此前我的笔记（来自完整 PDF 通读）记录：**"V3 D=1：next-token loss 差 < 0.005；下游平均 +0.5–1 个点"** —— 这是 671B/37B active 上的 ablation 结果。
- **未公开的关键信息**：
  - V3 是否在更小的 ablation 模型（15.7B、228B 等）上分别测过 MTP？
  - 不同 active scale 上 MTP 的 gain 是否单调？
  - D=2 / D=4 是否对比过？

### Ling 2.0 wind tunnel

- 论文宣称 "MTP 在不同 scale 下都对 code/math 一致提升"（§3.1 wind tunnel）。
- **但具体 scale gradient 数字（如 0.5B 上 +x%，8B 上 +y%）未在论文中给出**。
- Ling 2.0 在 5 个 anchor (500M–8B) 上跑 wind tunnel，最小 500M 已经在 Gloeckle 的"边界"以下；他们的实证证据**可能** 比 Gloeckle 更乐观（dense 路线小模型也有正收益）。

### 反例：Qwen3 / K2 的选择

- Qwen3 团队明确不要 MTP，配套用 thinking budget + reasoning RL；技术报告未给精确论据，但他们的 reasoning 性能（AIME 24 = 85.1）证明**不用 MTP 也能拿 SOTA**。
- Kimi K2 不要 MTP，靠 MuonClip + 高数据效率 (10× rephrase) 拿到稳定训练。
- **结论**：MTP **不是 SOTA 必需品**，是一种 sample-efficiency + inference-speed 的可选优化。

---

## 4. 我们 16B / 2.4B active 落在哪里

把上面的证据按"对我们 spec 适用性"排序：

| 证据类型 | 适用度 | 结论 |
|---|---|---|
| Gloeckle Fig.3 (dense, ~3B) | ★★★★ | 我们正处 boundary，+2.0 pts MBPP gain，但是 dense |
| V3 §4.5.1 (37B MoE active) | ★★ | 大 5-6×，外推风险大；笔记记录 +0.5-1pt 但 ablation 模型 scale 未在摘要披露 |
| Ling 2.0 wind tunnel (multi-scale) | ★★★ | 跨 500M-8B 都正向，但具体数字未公开 |
| Qwen3 / K2 不用 MTP | ★★★ | 证明非必需，但他们用别的稳定/效率技术补偿 |

**净判断**：2.4B active 上 MTP **预期收益 = -0.5 ~ +1.0 pts 下游平均**（基于 Gloeckle 趋势 + V3 外推 + Ling 乐观主义）。

**置信度低**。需要在我们的 wind tunnel 上直接测。

---

## 5. MoE + MTP 的特殊交互（值得单独说）

### 5.1 推理时 MoE 是 memory-bound，spec decoding 收益放大

- Dense 模型推理 bottleneck 通常是 attention KV cache 访存。
- **MoE 推理 bottleneck 是 expert 权重加载** —— 每 token 要从 HBM 把选中的 expert weight 调出来。
- 一次 forward pass 一次性出 D+1 个 token 的草稿（spec decoding），expert 加载摊薄了 D+1 倍。
- **理论上 MoE 上的 MTP 加速 > dense**。V3 报告的 1.8× 是个保守数字（D=1，acceptance ~80%），D=2 应能 2.5×+。
- **如果产品定位 serving QPS 重要 → MTP 价值显著高于训练时的 sample efficiency 提升**。

### 5.2 训练时 MTP 与 MoE PP partitioning 冲突

- Megatron PP 调度上，MTP module 加在最后 layer，是个不规则的"尾巴"。
- Ling 2.0 报告需要 **fine-grained heterogeneous PP partitioning + 部分 recomputation** 才能让 MTP 不引入 pipeline bubble。这是工程成本。
- V3 用 DualPipe，本来就是 fine-grained 调度，MTP 嵌入相对容易。
- **我们 PP=4 的相对小集群上，这个工程成本是可控的**（V2-Lite 同尺寸用 simple ZB-H1 就能 partition）。

### 5.3 router + MTP module 的交互

- MTP module 默认是 dense FFN（Ling 2.0、V3 都是），不参与路由。
- 但 MTP module 的 trunk 复用主模型的 hidden state，所以 router 收到的输入受 MTP 训练梯度的影响（通过 backward 流回）。
- **未公开**：是否 MTP 影响 router 的 load balancing？V3 在 ALF + MTP 同时启用下没报告异常。

---

## 6. 工程成本明细

### 6.1 训练成本

| 项 | 估算 |
|---|---|
| MTP module 参数（D=1，1 transformer block + 投影 M_k） | ~82M（attn 10.5M + dense FFN 67.2M + M_k 8.4M） |
| 占 base model active | ~3.4% |
| 占 base model total | ~0.5% |
| 训练时 forward 增量（1 个 block 多一遍） | ~3-5% FLOPs |
| 训练时 backward + 通信增量（PP partition 不规则） | ~3-5% |
| **总训练 overhead** | **~5-10%** |

对应到 §7 的成本估算：原 ~300-500K H100-hours，**加 MTP 约 +20-50K H100-hours = +\$40K–\$100K**。

### 6.2 推理成本

- **不启用 spec decoding（默认）**：丢弃 MTP module，**零部署开销**。
- **启用 spec decoding**：保留 MTP module，~82M 额外参数 + ~3% 额外 forward FLOPs（draft + verify），**换 1.8× 端到端加速**（V3 数字）。
- **结论**：纯训练用，丢弃；要 serving 加速则保留。两种模式之间运行时切换无开销。

### 6.3 实现复杂度

- 训练代码：+1 个 transformer block + 投影矩阵 + λ schedule + PP partition 调整。**估 1-2 周工程时间**。
- 推理框架：spec decoding 需要 vLLM/SGLang 等 backend 支持，**MoE 的 spec decoding 比 dense 更复杂**（需要保证 draft 路径与主路径走同一 expert 子集，否则失效）。
- 开源 inference framework 对 MTP-MoE spec decoding 的支持仍**不成熟**（vLLM 2025-Q1 才开始有原型 PR）。

---

## 7. 推荐：MTP 移出主 spec，放进 pilot

**理由综合**：

1. **下游 gain 在 2.4B active 不确定**（boundary case）。
2. **训练 overhead 5-10% 非零**，需要明确价值才值得。
3. **推理加速 (1.8×) 是真实价值**，但仅在 spec decoding 启用时；且依赖 inference framework 成熟度。
4. **不用 MTP 不影响 SOTA**（Qwen3 / K2 反例）。
5. **加 MTP 的工程债大于其他可选项**（如 MuonClip 只是优化器一行代码、N=256 主要影响 EP 拓扑）。

→ **MTP 是"高度推荐的 pilot 项，但不应该绑定主 spec"**。

---

## 8. 决策树（pilot 之后用）

**Wind tunnel A2 (1B total / 200M active / 25B tokens) 上做 D=0 vs D=1 对照**：

```
A2 D=0 vs D=1 ablation 结果：
│
├─ Δ 下游均值 ≥ +0.5 pt (HumanEval, MBPP, GSM8K, MMLU 4 项均值)
│    │
│    ├─ 训练成本预算余 ≥ 10% → ✅ 加入主 spec，D=1 + λ=0.3→0.1
│    └─ 预算紧 → 跳过，留作 v2 升级项
│
├─ Δ ∈ [0, +0.5 pt)
│    │
│    ├─ 产品定位 = serving QPS 重要 → ✅ 加入，主要为 spec decoding
│    │   （前提：inference framework 支持 MoE-MTP spec decoding）
│    └─ 产品定位 = chat / 通用 → ❌ 跳过，收益不抵成本
│
└─ Δ < 0
     │
     └─ ❌ 跳过（Gloeckle boundary 警示成真）
```

**A2 测试配置**（保持其他变量不变）：
- 同 architecture (1B / 200M active / 16 routed / K=4 / d_expert=448 / 16 layers / hidden 1024)
- 同 ALF balancing + QK-Norm + Router z-loss
- 同 WSD schedule + AdamW + BF16
- 同数据 mix（webtext + code + math）
- 训练 25B tokens
- A1: D=0（baseline，无 MTP）
- A2: D=1（V3 风格 causal chain，λ=0.3→0.1）
- A3 (可选): D=2（验证 D 是否单调；Gloeckle dense 上 n=4 最佳，MoE 上可能更小）
- **评测**：HumanEval / MBPP / GSM8K / MATH / MMLU / TriviaQA / HellaSwag 7 项均值

**Pilot 预算**：约 3K H100-hours per condition × 3 = **~9K H100-hours total**（< 主训练 budget 3%）。

---

## 9. 公开数据空缺与风险

### 我无法从公开论文中确认的关键问题：

1. **V3 §4.5.1 MTP ablation 在多大 scale 模型上做？** WebFetch 拉到的 V3 paper HTML 在 §3.3.2 截断。我的笔记记 "+0.5-1pt 下游" 但没有 ablation 模型尺寸。**这是最重要的未知数**。
2. **Ling 2.0 wind tunnel 中 MTP 的精确 per-scale 数字**？论文只说"一致提升"，未给数字。
3. **GLM-4.5 是否真的用了 MTP？** Abstract 不提，PDF 全文不可访问。
4. **D=2 / D=4 在 MoE 上是否单调更好？** 无公开数据。Gloeckle 在 dense 上 n=4 最佳，但 MoE 下没人验证过。
5. **MoE-MTP spec decoding 的 acceptance rate 在 2-3B active 上是多少？** V3 报 80%+ 是 37B active，外推到 2.4B 应该更低（小模型预测准确率低）。

### 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 2.4B active 上 MTP 拖累 | 中（Gloeckle 边界） | 中（-0.5 pt） | A2 pilot 决定 |
| Inference framework 不支持 MoE-MTP spec decoding | 高 | 低（默认丢弃就行） | 推理时关闭 MTP |
| MTP PP partition 引入 5%+ bubble | 低 | 中（训练时间膨胀） | fine-grained PP；fallback 关 MTP |
| λ schedule 0.3→0.1 在不同数据 mix 下未必最优 | 低 | 低 | 沿用 V3 schedule，不重新调 |

---

## 10. 一句话结论

> **MTP 是低风险、潜在中等收益的训练辅助 + 推理加速选项；2.4B active 落在 Gloeckle 的边界区，没有直接证据保证正收益。建议从主 spec 移出，A2 (1B/200M active/25B tokens) anchor 做 D=0 vs D=1 对照，3% 主训练 budget 决定是否纳入；若 nailed down spec decoding 是产品诉求则尽量保留。**
