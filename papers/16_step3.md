# Step-3 is Large yet Affordable: Model-System Co-design for Cost-effective Decoding

- **arXiv**: 2507.19427
- **机构**: StepFun (198 作者)
- **发表时间**: 2025-07-25

## TL;DR

Step-3 是一篇 **"模型-系统协同设计、成本驱动"** 的 MoE 论文，是这 4 篇里最贴近 16B 部署成本思考的一篇。**321B total / 38B active / 61 层 / 7168 hidden**（LLM 版 316B，VLM 版 321B），两个核心创新都瞄准 decoding 阶段硬件效率：
1. **MFA (Multi-matrix Factorization Attention)**：一种新的低 KV cache + 适中算术强度的 attention 方案，算术强度 128，介于 DeepSeek-MLA (512) 与 Qwen3-GQA (32) 之间。
2. **AFD (Attention-FFN Disaggregation)**：把 attention 和 FFN 部署到两个独立的 GPU 子集群，分别按各自的 bottleneck 选硬件，3-stage 流水。

最终在 Hopper 上达到 **4039 tok/s/GPU**（DeepSeek-V3 是 2324），1M tokens 解码成本 **\$0.055**（DSv3 \$0.068），是开源 MoE 里少见的"明确把成本算到小数点后三位"的论文。

## 关键架构配置

| 参数 | 数值 |
| --- | --- |
| Total params (LLM) | 316 B |
| Total params (VLM) | 321 B |
| Active params / token | 38 B |
| Layers | 61 |
| Hidden dim | 7168 |
| Attention heads (query) | 64 |
| Head dim | 256 |
| KV heads | 1 shared K + 1 shared V (head_dim 256) |
| Q down-projection rank | 7168 → 2048 → 64×256 |
| MoE 位置 | 除前 4 层和最后 1 层外的所有层 |
| Shared experts | 1 |
| MoE sparsity | ~0.08 (active / total in MoE 层) |
| Decoding throughput | 4039 tok/s/GPU on H100/H800, FP8 |
| 50ms TPOT SLA |  4 K context |

## 核心方法 / 创新点

### 1. MFA (Multi-matrix Factorization Attention)

MFA 是这篇的 attention 设计。关键参数：

- 64 query heads，共享 1 个 Key head 和 1 个 Value head（head_dim 都是 256）
- Q 投影：**7168 → 2048（low-rank）→ norm → 64×256（up-project）**
- KV 投影直接 7168 → 256（每种 1 个 head）

可以理解为：
- 它是 **MQA（multi-query attention）+ low-rank Q factorization** 的组合：KV 像 MQA 那样只有 1 个 head，Q 则用 low-rank bottleneck（2048）压缩。
- 与 **MLA**（DeepSeek-V3）对比：MLA 把 KV 也低秩压缩（512 维 latent），KV cache 更小；MFA 直接共享 1 个 KV head，cache 同样很小但实现更简单。
- 与 **GQA** 对比：GQA 有多个 KV group（典型 8），MFA 极端到 1 个 KV head。

**算术强度比较**（论文 Table 给出的 GEMM 算术强度，越接近 GPU sweet spot 越好）：

| 方案 | Arith intensity (FP8) | KV cache / token |
| --- | --- | --- |
| MLA (DeepSeek-V3) | ~512 | 中 |
| **MFA (Step-3)** | ~128 | 小 |
| GQA (Qwen3) | ~32 | 大 |

论文论点：MFA 的 128 算术强度在 H100/H800/H20 等不同 GPU 上都接近 sweet spot，**硬件可移植性最好**，避免了 MLA 在 H20 那种低算力卡上"算力打不满"的尴尬。

### 2. AFD (Attention-FFN Disaggregation)

核心思想：**attention 是 memory-bound（KV cache 访存），FFN 是 compute-bound (MoE GEMM)，二者不应该挤在同一组 GPU 上**。

AFD 设计：
- 把 attention 和 FFN 拆到两个 disaggregated GPU 子集群：**A-cluster（attention specialists）** 和 **F-cluster（FFN specialists）**。
- 每一层的执行变成 3-stage pipeline：
  - Stage 1: A-cluster 算 attention
  - Stage 2: F-cluster 算 FFN（MoE 路由 + expert FFN）
  - Stage 3: 二者间的 AllToAll / 通信
- 每个 stage 约 16.6 ms，3 stage 流水起来后整层达到 50ms TPOT SLA。
- 允许 **A 和 F 用不同 GPU 型号**：例如 A 用 H20（大显存适合存 KV cache），F 用 H100（高算力适合 MoE GEMM），按 bottleneck 各买各的。

### 3. MoE 设计

- 除首 4 层和最后 1 层外，所有 56 层用 MoE
- 1 个 shared expert（always-on，类似 DeepSeek-V3 的 shared expert）
- Sparsity ~0.08（active expert FLOPs / total expert FLOPs in MoE 层）
- 具体 num experts / top-k 论文摘要未明示，但从 38B active / 316B total 推断与 DSv3 同量级稀疏度

## 训练 & 系统细节

- **训练硬件**：未完全披露，但优化目标显然是 H100/H800/H20 国产卡混用场景
- **量化**：FP8 训练 + FP8 inference (与吞吐数字配套)
- **AFD 实现**：3-stage pipeline，A 和 F 间靠 high-bandwidth interconnect (NVLink / RDMA)；论文给出了通信 overlap 与显存分配策略

## 关键消融与结果

**Decoding 吞吐**（FP8、4K context、50ms TPOT SLA）：

| Model | tok/s/GPU |
| --- | --- |
| **Step-3** | 4039 |
| DeepSeek-V3 | 2324 |

Step-3 比 DSv3 高 ~74%。

**1M tokens decoding 成本**：

| Context | Step-3 | DeepSeek-V3 | Qwen3 MoE |
| --- | --- | --- | --- |
| 8K | \$0.055 | \$0.068 | \$0.062 |
| 32K | \$0.129 | \$0.211 | - |

在 32K context 上 Step-3 比 DSv3 便宜 ~39%，差距随 context 长度扩大（因为 MFA 的 KV cache 更小）。

**模型能力**：论文摘要主推 cost，没强调通用 benchmark 大幅领先，但表示与 DSv3、Qwen3-235B 等同级 MoE 在通用任务上可比。

## 对 16B MoE 设计的启示

**结论：MFA 在 16B 上要不要采用？— 大概率不采用，但其方法论值得借鉴。**

### MFA 是否值得在 16B 上采用？

**答：不推荐。**

- MFA 的核心收益是 **大模型 long-context 下减小 KV cache + 算术强度跨硬件均衡**。16B 规模下：
  - KV cache 体量本身就小（如果 head_dim 128、layers 30、hidden 4096，128K context 也才几 GB）
  - 算术强度的"跨硬件均衡"在 16B 部署上不是主要矛盾 —— 16B 通常单卡 / 双卡部署，硬件就一种
- MFA 的 Q 低秩 down-project + up-project 引入额外 layer norm + 两次 matmul 的实现复杂度，对 16B 收益太低
- 16B 推荐 **直接用 GQA**（简单、kernel 成熟、与 FlashAttention 完美兼容）；如果对长 context 真有顾虑，可考虑 **MLA**（已经被 DeepSeek-V3 大规模验证）

### AFD (Attention-FFN Disaggregation) 是否值得借鉴？

**答：16B 部署一般不需要 AFD。**

- AFD 的前提是 **多节点部署**，把 attention 和 FFN 切到不同节点。16B 通常单节点就能跑，AFD 没有用武之地。
- 如果 16B 要在 **大规模并发服务** 下追求极致 TPOT，那 AFD 的思想（把 memory-bound 和 compute-bound 算子分到不同硬件）值得参考，但工程复杂度极高。

### 真正值得借鉴的方法论

1. **明确成本目标 + 硬件目标 → 反推架构**：Step-3 是这 4 篇里唯一一篇把"每百万 token 多少美元"写成核心指标的。16B 设计也应该这么做 — 例如目标"1×H100 上 ≥ 8000 tok/s decoding"，反推 attention / MoE 设计。
2. **算术强度概念**：评估任何 attention 变体时，算 GEMM 算术强度 → 看是否匹配目标 GPU 的 FLOPs:HBM 比。16B 也适用。
3. **Shared expert**：1 个 shared expert + 多个 sparse expert 的组合（同 DSv3 / Step-3），对 16B MoE 是稳健默认。
4. **"前几层和末层不用 MoE"**：Step-3 前 4 层和最末层是 dense MLP。这个保守做法在小规模 MoE 上也常见（embedding 后第 1-2 层稳定性敏感），16B 可考虑前 2-3 层保持 dense。

## Caveats / 局限

- 论文是 **cost-aware engineering paper**，benchmark / 能力对比不是重点；与 DSv3 / Qwen3 的通用能力差距不完全清晰。
- MFA 的算术强度 128 是针对当代 H100/H800/H20 优化的；下一代硬件（B200 等）的 FLOPs:HBM 比变化后可能需要重新调整 Q rank。
- AFD 假设有高带宽 A↔F 互联；如果用普通以太网就跑不起来。
- 论文未完全公开训练数据量 / 总训练 FLOPs。
- "316B vs 321B"差别在 VL 模块（vision encoder），LLM 部分应该是 316B。
- Active 38B 比同等级模型（DSv3 37B、Qwen3-235B 22B）略高，"compute per token" 不是最低；省的是 attention + KV cache + 系统层效率，不是 expert 稀疏度。
