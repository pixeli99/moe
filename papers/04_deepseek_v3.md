# DeepSeek-V3 Technical Report

- **arXiv**: 2412.19437
- **机构**: DeepSeek-AI
- **发表时间**: 2024 年 12 月
- **作者(代表)**: DeepSeek-AI 团队署名（涵盖 Aixin Liu, Bei Feng, Bing Xue, Chenggang Zhao, Damai Dai, Daya Guo, Dejian Yang 等百余人）

## TL;DR
DeepSeek-V3 在 V2 架构基础上做了五件大事：(1) 把 MoE 扩到 **671B 总参 / 37B 激活**，N_routed=**256**、N_shared=**1**、top-K=**8**、d_expert=**2048**，hidden=**7168**、61 层；(2) 用 **auxiliary-loss-free**（论文 3）+ 极小 sequence-wise α=1e-4 的复合策略替换 V2 的三件套 balance loss；(3) 引入 **Multi-Token Prediction (MTP)** D=1，预测下一个 token 的同时预测下下个 token，作为训练阶段的辅助目标，**推理时丢弃 MTP head**；(4) 工程层面祭出 **FP8 混合精度训练**（1×128 activation tile + 128×128 weight block 量化）和 **DualPipe** 双向流水让 all-to-all 与计算几乎完全重叠；(5) 训练 **14.8T tokens** 总计 **2.788M H800 GPU-hours**（≈ $5.6M），是当时性价比最高的前沿开源模型，post-training 从 DeepSeek-R1 蒸馏推理能力。

## 关键架构配置

| 项 | 值 |
|---|---|
| 总参 / 激活参数 | **671B / 37B**（≈ 5.5% 激活） |
| Layers | **61** |
| Hidden | **7168** |
| Dense 前缀层 | **前 3 层**为 dense FFN，**其余 58 层全 MoE**（Section 4.2） |
| Attention | **MLA**：n_h=128, d_h=128 |
| q-lora rank d_c' | **1536** |
| kv-lora rank d_c | **512** |
| decoupled RoPE d_h^R | **64** |
| MoE | N_routed=**256**, K_r=**8**, N_shared=**1**, d_expert=**2048** |
| 路由 gating | **Sigmoid affinity + Top-K + normalization**（Eq. 13, 15-16） |
| Node-Limited Routing | **M=4**（每 token 至多送 4 个节点） |
| 训练 tokens | **14.8T** |
| 训练成本 | **2.788M H800 GPU-hours**（≈ \$5.576M @ \$2/h） |
| 上下文 | 4K → 32K → 128K 三段扩展 |
| MTP | **启用，D=1**；shared embedding+head；λ=0.3 (前 10T) → 0.1 (后 4.8T) |
| 精度 | **FP8 训练（GEMM）**，BF16/FP32 关键路径 |

## 核心方法 / 新点

### 1. 路由：Sigmoid Affinity + Top-K + Normalization（Eq. 13-16）
亲和度改为 sigmoid（与 V2 的 softmax 不同）：
$$
s_{i,t}=\sigma(\mathbf{u}_t^\top \mathbf{e}_i)
$$
带偏置项的 top-K：
$$
g'_{i,t}=
\begin{cases}
s_{i,t}, & \text{if } s_{i,t}+b_i\in \text{TopK}(\{s_{j,t}+b_j\}_{j=1}^{N_r},K_r)\\
0,&\text{otherwise}
\end{cases}
$$
（与论文 3 完全一致；b_i 不参与权重计算。）

最终 gating 经归一化：
$$
g_{i,t}=\frac{g'_{i,t}}{\sum_j g'_{j,t}}
$$
归一化使得激活的 8 个专家加权和约等于 1（vs V2 的 softmax 已自带归一化），sigmoid 路径让多专家共选时不强制竞争。

### 2. Auxiliary-Loss-Free Load Balancing（论文 3 的工程化）
- 偏置更新：b_i ← b_i ± γ，**γ = 0.001**（V3 称为 bias update speed），方向与是否过载/欠载相反。
- **γ 调度**：前 **14.3T tokens 用 γ=0.001**，最后 0.5T tokens **γ=0**（停止更新偏置，让模型适应 deployment 期固定 routing）。
- **Complementary Sequence-Wise Balance Loss**（Eq. 17）：
  $$
  \mathcal{L}_{\text{Bal}}=\alpha\sum_{i=1}^{N_r}f_i P_i,\quad \alpha=\mathbf{1\times 10^{-4}}
  $$
  仅在单序列粒度上加一个极小的辅助 loss，防止序列内极端偏斜；α 比 V2 的 expert-balance loss 小 30×，几乎不干扰梯度。
- **完全删除** V2 中的 device-balance loss 与 communication-balance loss。

### 3. Node-Limited Routing M=4（Section 4.2）
> "we ensure that each token will be sent to at most M nodes, which are selected according to the sum of the highest K_r/M affinity scores."

256 个 routed expert 均匀分布在 **64 张 GPU × 8 节点**：每节点 32 个专家。每 token 选 8 个专家 → 至多 4 个节点 × 2 个专家/节点。**IB 跨节点流量被严格限制**，使 IB 通信与 NVLink 同节点通信可在 DualPipe 中重叠。

### 4. Multi-Token Prediction (MTP)（Eq. 21-25）
- **结构**：MTP module 与主模型**共享 embedding 与 output head**；每个 module 内含 **1 个 Transformer block** TRM_k 和投影矩阵 M_k。
- **训练目标**：每个 module 预测向后 k 步的 token；总 loss:
  $$
  \mathcal{L}_{\text{MTP}}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{\text{MTP}}^k
  $$
- **D=1**：只预测下一个+下下个，结构最简。
- **λ schedule**：前 10T tokens **λ=0.3**，后 4.8T tokens **λ=0.1**（降低 MTP 权重，让模型最后阶段更专注 next-token loss）。
- **推理**：MTP module 通常**丢弃**，只走主模型 next-token head；也可保留用于 speculative decoding，论文报告 MTP head 在 80% 以上 token 上预测下一步正确，可作 draft model 加速 1.8×。

### 5. FP8 混合精度训练（Section 3.3 / Figure 6-7）
- **走 FP8 的算子**：**所有 GEMM**（Fprop / Dgrad / Wgrad），格式 **E4M3**。
- **保留 BF16/FP32 的算子**：
  - Embedding module
  - Output head（最终 LM head）
  - **MoE gating module（router）**
  - Normalization
  - **所有 attention 算子**（含 RoPE）
  - Master weights、weight gradients、optimizer states（AdamW first/second moment 用 BF16；master 用 FP32）
- **细粒度量化**：
  - **Activations**：1×128 **tile-wise**（每 token × 每 128 channel 一个 scale）
  - **Weights**：128×128 **block-wise**
  - 兼顾局部 outlier，避免全张量 scale 导致精度坍塌。
- **FP32 累加促进**：在 GEMM 内部每 N_C=128 个元素将累加器从 FP8 partial 提升到 FP32，缓解 NVIDIA H800 Tensor Core 的 14-bit 累加精度问题。
- **反向特殊处理**：attention 反向的某些激活用 **E5M6** 格式（5-bit 指数 6-bit 尾数）以保留尾巴信息。
- **训练精度结果**：相比 BF16 baseline 的 loss 偏差 < 0.25%，无明显发散。

### 6. DualPipe + 通信优化（Section 3.2.1-3.2.2）
- **DualPipe**：双向流水，把 forward / backward 拆成更细 chunk 并对称调度，bubble 数从 1F1B 的 (PP−1)(F+B) 降到 (PP/2−1)(F&B + B − 3W)。
- **总并行**：
  - PP = **16-way Pipeline Parallelism**
  - EP = **64-way Expert Parallelism**（跨 8 节点）
  - DP = **ZeRO-1**
  - **无 TP**
- **All-to-All kernel**：
  - 把 **20 个 SM** 切成 **10 个通信 channel**，warp specialization：
    1. IB 发送
    2. IB → NVLink 转发
    3. NVLink 接收
  - 动态调整每类 warp 数量；与计算 kernel 并发执行，做到 **all-to-all 几乎完全被算掩盖**。
- 配合 Node-Limited Routing，将跨节点流量上限固定 → IB 带宽永不饱和。

### 7. 训练成本分解（Section 4.6 / Table 1）
| Stage | GPU-Hours |
|---|---|
| Pre-training (14.8T tokens) | **2,664K** |
| Context extension (4K→32K→128K) | **119K** |
| Post-training (SFT + RL) | **5K** |
| **Total** | **2,788K H800-hours** |

折算：每 T token 约 **180K GPU-hours**，在 2048 卡 H800 集群 ≈ **3.7 天/T**。

## 训练 & 系统细节

- **数据**: 14.8T 高质量 token，相比 V2 增加数学/编程比例，扩展多语种；用 **document packing** 提升 GPU 利用；**Fill-In-Middle (PSM)** 以 10% 概率应用。
- **Tokenizer**: 在 V2 的 100K BBPE 基础上做了微调；新词表大小相近。
- **优化器**: AdamW（β₁=0.9, β₂=0.95, weight_decay=0.1）；grad clip = 1.0。
- **学习率**: 峰值 **2.2e-4**，cosine decay。
- **Batch size**: 前 469B token 从 **3072 ramp 到 15360**，之后保持 15360；序列 4K。
- **稳定性**: 无显式提到 QK-Norm/MuonClip；FP8 + 细粒度量化 + FP32 promote 是核心稳定性手段。
- **Post-training**:
  - SFT 数据约 1.5M 条（推理 + 通用 + 代码 + 数学 + 中文 + 多轮对话）。
  - **R1 蒸馏**：从 DeepSeek-R1 抽取 long-CoT 推理数据，做 SFT 注入 reasoning pattern。
  - **多阶段 RL**：基于 **GRPO（Group Relative Policy Optimization）**；reward 含 **rule-based RM**（数学/代码可验证）与 **model-based RM**（开放回答）。

## 关键消融与实验结果（Section 5 / Table 8）

### Base 模型 vs 同档开源 / 闭源
| Benchmark | DeepSeek-V3 Base | LLaMA-3.1 405B | Qwen2.5 72B |
|---|---|---|---|
| MMLU (5-shot) | **87.1** | 84.4 | 85.0 |
| MMLU-Pro | **64.4** | 51.1 | 58.3 |
| BBH | **87.5** | 81.7 | 80.5 |
| GPQA-Diamond | **41.3** | 33.3 | 38.9 |
| HumanEval | **65.2** | 65.2 | 64.6 |
| MATH | **61.6** | 49.0 | 56.5 |
| C-Eval | **90.1** | – | 84.0 |

### Chat / Instruct 关键
- AIME 2024: **39.2** (V3 chat)，远超同档开源；蒸馏 R1 后进一步提升。
- LiveCodeBench: **40.5%**。
- SWE-Bench Verified: **42.0%**。
- 全榜在当时多个推理 / 代码项目逼近或超过 GPT-4o，是首个把 sparse MoE 推到 GPT-4 同档的开源工作。

### MoE / 训练消融（Table 5-6）
- **Loss-Free vs Loss-Controlled**：在 V3 内部规模上重复论文 3 的实验，Loss-Free 仍领先 0.04-0.06 perplexity 且 MaxVio 显著更稳。
- **MTP D=1 vs no MTP**：next-token loss 几乎不变（差 < 0.005），但下游平均提升约 +0.5–1 个点；推理 speculative decoding 加速 1.8×。
- **FP8 vs BF16**：相对 baseline loss 差 < 0.25%，下游 MMLU/GSM8K 等差距 < 0.3pt；同步**节省 ~30% 训练成本**（相对纯 BF16）。

## 对 16B MoE 设计的启示

1. **基础配方直接用 V3-style，但缩比例**：
   - 总参 16B、激活 ~2.5B 比例（≈ 16%，比 V3 的 5.5% 高），原因是 16B-class 仍受激活下限影响。
   - **N_routed 缩到 64**（V2-Lite 同款），保留 N_shared=1 或 2，K=6-8。
   - **保留 dense 前缀层 1-3 层**（V3 用 3 层，16B 至少 1 层）。
   - MLA：在 16B 下 hidden 通常 ≤ 3072，可以不要 q-lora（V2-Lite 即如此），kv-lora rank d_c=512、d_h^R=64 直接抄。
2. **强烈推荐采纳的项**：
   - **Sigmoid + Top-K + 归一化**的路由（Eq. 13-16）。
   - **Auxiliary-Loss-Free** + 极小 α=1e-4 sequence-wise loss。
   - **MTP D=1**：付出极少额外计算就能拿 +0.5pt 下游 + speculative decoding 加速。
   - **Node-Limited Routing**：16B 通常 ≤ 4 节点，可设 M=2。
3. **要谨慎的项**：
   - **FP8 训练**对 16B 必要性较低（成本本身就低），实现复杂；如果团队 infra 不成熟，建议 BF16 起步。
   - **DualPipe** 在 PP > 8 才显著见效；16B 大多 PP ≤ 4，普通 ZB1P 已足够。
   - V3 的 R1 蒸馏依赖已有 R1 模型；16B 无法直接复现，需要替代 reasoning 数据源（如 open-source long-CoT）。
4. **明确反例**：
   - **不要**把 V2 的 device-balance / communication-balance loss 抄进 16B；V3 已经证明 Loss-Free 更好。
   - **不要**在最终阶段保留 γ=0.001 不衰减；按 V3 在最后 ~3% 训练量把 γ 调到 0。

## Caveats / 局限

- 报告级别的论文，正文公式与 schedule 给得相对克制，部分细节（如 MTP TRM_k 的隐藏维度、PSM FIM 模板）需查代码（开源 weights 配文件）。
- **FP8 训练稳定性**仅在 H800 / E4M3 上验证；E4M3 之外（如 H100 用 E5M2 default）需重新调 N_C 与 tile size。
- 不同 WebFetch 版本对"前 3 层 dense"还是"前 1 层 dense"出现过描述差异；v2 paper html 与官方 config 均确认 **3 层 dense**（V3 与 V2 不同；V2 是 1 层）。
- 训练成本 2.788M GPU-hours 是 official；不含**数据准备 / 失败 / 调试**成本，真实"从零到 V3"投入应更高。
- Post-training 的 R1 蒸馏与 GRPO 细节相对简略（reward 函数权重、batch 配比等），社区复现存在 gap。
- **DeepSeek-V3 没有显式 MuonClip 或 QK-Norm 等近期被普遍认为重要的稳定性技术**；这意味着团队靠数据质量 + FP8 量化策略 + 路由设计本身就够稳，但其他团队复现到 700B 规模可能仍需要这些 trick。
- MTP D=1 是上限保守值；论文未公开 D>1（D=2/3/4）的对比。
- 256 个专家在 256K 词表 + 8K 序列 + 64-way EP 下的吞吐和均衡靠 Node-Limited Routing 严格依赖物理拓扑（8 节点 × 8 卡）；不同集群拓扑（如 32 卡/节点）需重新优化。
