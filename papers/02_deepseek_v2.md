# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- **arXiv**: 2405.04434
- **机构**: DeepSeek-AI
- **发表时间**: 2024 年 5 月
- **作者(代表)**: DeepSeek-AI（团队署名），含 Damai Dai, Chenggang Zhao, R.X. Xu, Daya Guo, Aixin Liu 等

## TL;DR
DeepSeek-V2 在 DeepSeekMoE（论文 1）基础上把两件大事做到了大规模量产：(1) 提出 **Multi-head Latent Attention (MLA)** —— 用 low-rank KV 联合压缩到 d_c=512 + decoupled RoPE 通道 d_h^R=64，**KV cache 比 DeepSeek 67B 减少 93.3%**，吞吐提升 5.76×；(2) 将 DeepSeekMoE 扩到 **N_routed=160, K=6, N_shared=2**、总参 **236B / 激活 21B**，在 **8.1T tokens** 上完成预训练，整体每 T token 训练用 172.8K GPU 小时（较 67B Dense 节省 42.5%）。MLA 与 DeepSeekMoE 的组合从此成为 DeepSeek 系列基础架构，V2 还配套开源了 **DeepSeek-V2-Lite (15.7B / 2.4B 激活)** 供社区研究复用。

## 关键架构配置

### DeepSeek-V2（主模型）
- **总参 / 激活**: 236B / 21B（≈ 8.9% 激活）
- **Layers**: 60；**Hidden**: 5120
- **Attention**: MLA，n_h = 128, d_h = 128
  - q-lora rank d_c' = **1536**
  - kv-lora rank d_c = **512**
  - decoupled RoPE 维度 d_h^R = **64**（每头）
- **MoE**: N_routed=**160**, K_r=**6**, N_shared=**2**, d_expert=**1536**（intermediate dim）
- **Dense 前缀**: 第 1 层为 dense FFN，其余 59 层均为 MoE
- **位置编码 / Norm / Activation**: 部分 RoPE（仅 d_h^R 通道）/ RMSNorm / SwiGLU
- **Context window**: 4K 预训练，YaRN 扩到 **128K**
- **Tokenizer**: byte-level BPE, 100K 词表
- **训练 tokens**: **8.1T**（中文 token 比英文多约 12%）
- **MTP**: 本文 **未使用 MTP**（V3 才引入）

### DeepSeek-V2-Lite（Appendix B；HF 开源）
- 总参 / 激活: **15.7B / 2.4B**
- Layers: **27**（即 1 dense + 26 MoE；HF config 也为 27）
- Hidden: **2048**（注意：早期 WebFetch 给出 2560/32 层为错误数据；以 HF deepseek-ai/DeepSeek-V2-Lite 官方 config 为准）
- Attention heads: 16，d_h = 128
- MLA: q-lora rank = **未启用**（V2-Lite 直接用 full-rank Q），kv-lora rank d_c = 512，d_h^R = 64
- MoE: N_routed=**64**, K_r=**6**, N_shared=**2**, d_expert=**1408**
- 训练 tokens: 5.7T（V2-Lite 单独训练；与主模型语料同源但 token 数较少）
- 推理友好：单卡 40GB 可跑

> 说明：论文正文 Appendix B 简要描述了 V2-Lite，但 WebFetch 抓取版本之间字段存在轻微出入；上述 V2-Lite 数字以**官方 HuggingFace config (deepseek-ai/DeepSeek-V2-Lite)** 为权威。

## 核心方法 / 新点

### 1. Multi-head Latent Attention (MLA)（Eq. 9-19）
关键思路：把 K/V 联合压成低秩 latent `c^{KV}_t ∈ R^{d_c}`，推理时只缓存 `c^{KV}_t` 与位置相关的 `k^R_t`，避开缓存 n_h × d_h 维的完整 K/V。

**KV 联合压缩**（Eq. 9-11）：
$$
\mathbf{c}^{KV}_t = W^{DKV}\mathbf{h}_t,\quad W^{DKV}\in\mathbb{R}^{d_c\times d_{model}}\;(512\times 5120)
$$
$$
\mathbf{k}^C_t = W^{UK}\mathbf{c}^{KV}_t,\quad \mathbf{v}^C_t = W^{UV}\mathbf{c}^{KV}_t
$$
W^{UK}, W^{UV} 输出维度均为 n_h × d_h = 16384。

**Query 低秩压缩**（Eq. 12-13）：
$$
\mathbf{c}^Q_t = W^{DQ}\mathbf{h}_t \in \mathbb{R}^{d_c'},\quad \mathbf{q}^C_t = W^{UQ}\mathbf{c}^Q_t
$$
d_c' = 1536；这一步主要省训练激活/反向显存，对 KV cache 无影响。

**Decoupled RoPE**（Eq. 14-19）：因为 W^{UK} 与 RoPE 不可交换，作者把 RoPE 单独分到一组维度上：
$$
\mathbf{q}^R_{t,i} = \text{RoPE}(W^{QR}\mathbf{c}^Q_t),\quad \mathbf{k}^R_t = \text{RoPE}(W^{KR}\mathbf{h}_t)
$$
W^{KR} ∈ R^{64×5120}（即只有 d_h^R=64 维带位置编码，且对所有 head 共享同一 k^R_t）。最终注意力使用拼接：
$$
\mathbf{q}_{t,i}=[\mathbf{q}^C_{t,i};\mathbf{q}^R_{t,i}],\quad \mathbf{k}_{t,i}=[\mathbf{k}^C_{t,i};\mathbf{k}^R_t]
$$
$$
\mathbf{o}_{t,i}=\sum_j \text{softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h+d_h^R}}\right)\mathbf{v}^C_{j,i}
$$

**KV cache 大小**（Table 1）：每 token 每层只缓存 `(d_c + d_h^R) = 576` 个元素 ≈ 9/2 · d_h，相当于 **GQA with 2.25 groups** 的体量，但表现优于 full MHA。

### 2. DeepSeekMoE in V2（Eq. 20-22）
沿用论文 1 的 fine-grained + shared expert：
$$
\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s}\text{FFN}^{(s)}_i(\mathbf{u}_t)+\sum_{i=1}^{N_r}g_{i,t}\text{FFN}^{(r)}_i(\mathbf{u}_t)
$$
gate 仍是 softmax 后 top-K：
$$
g_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\text{TopK}(\{s_{j,t}\},K_r)\\0,&\text{otherwise}\end{cases},\quad s_{i,t}=\text{Softmax}_i(\mathbf{u}_t^\top \mathbf{e}_i)
$$

### 3. Device-Limited Routing + 三层 balance loss
- **Device-Limited Routing**：每个 token 先选 **M=3** 个亲和度最高的设备，再在这些设备的专家中做 top-K；极大降低跨设备 all-to-all 流量。
- **Expert-Level Balance**（Eq. 23-25）：α_1 = **0.003**。
- **Device-Level Balance**（Eq. 26-28）：α_2 = **0.05**；目标是均衡各 device group 的总流量。
- **Communication Balance**（Eq. 29-31）：α_3 = **0.02**；目标是均衡"每张卡接收的 token 数"。
- **Token-Dropping**：在每个设备上按 affinity 从低到高丢弃 token 直到打到 capacity factor=1.0；并保证 ~10% 的训练序列**永不被丢**，让模型学到完整序列分布。

## 训练 & 系统细节

- **数据**: 8.1T tokens 中英文混合；中文略多于英文 ~12%；BPE 100K 词表。
- **优化器**: AdamW（β₁=0.9, β₂=0.95, weight_decay=0.1）。
- **学习率**: 峰值 **2.4e-4**；前 2K steps 线性 warmup；走完 ~60% token 后 ×0.316，~90% token 后再 ×0.316（两段 step decay）。
- **Batch size**: 起步 2304，前 225B token 内线性升到 9216，之后保持 9216；序列长度 4K。
- **初始化**: 所有可学参数 std=0.006 正态初始化。
- **并行**:
  - 16-way **zero-bubble pipeline parallelism**（基于 ZB-H1/H2 思路）
  - 8-way **expert parallelism**（D=8）
  - **ZeRO-1** data parallel
  - **不使用 tensor parallel**（激活参数小，tp 通信不划算）
  - 把共享专家计算与专家并行 all-to-all **重叠**
- **硬件**: NVIDIA H800 GPUs（8 卡/节点 NVLink+NVSwitch；节点间 InfiniBand）。
- **训练成本**: 172.8K GPU-hours / T token（V2 总训练 ≈ 172.8K × 8.1 ≈ 1.4M GPU-hours），较 DeepSeek 67B 节省 **42.5%**。
- **长上下文**: 预训练完后用 **YaRN** 把 4K → 128K，64K SFT 验证。
- **稳定性**: 没有公布 muonClip / QK-Norm 等技巧；走的是 BF16 + 标准 LayerNorm/RMSNorm，且 router 用 FP32 计算 softmax。

## 关键消融与实验结果

### MLA vs MHA / GQA / MQA（Table 8/9）
- 同等 7B-class 训练，MLA 在 MMLU/CMRC/CMath 等基准上**优于 MHA**，同时 KV cache 量近似 **GQA(2.25 groups)**。
- 部署吞吐：相对 67B Dense，V2 单序列推理 **吞吐 5.76× 提升**，KV cache **减小 93.3%**。

### 主要 benchmark（Base 模型，5-shot 等标准 setting）
| Benchmark | DeepSeek-V2 (236B/21B) |
|---|---|
| MMLU (5-shot) | **78.5** |
| BBH (3-shot) | 78.9 |
| HumanEval (0-shot) | 48.8 |
| GSM8K (8-shot) | 79.2 |
| MATH | 43.6 |
| C-Eval | 81.7 |
| CMMLU | 84.0 |

整体在 Open LLM 之中处于第一梯队，且训练成本远低于同量级 Dense / 其他 MoE。

### 消融（Section 4）
- **MLA vs MHA**：相同训练，MLA 在多数任务 +0.5~1.5 个点，且 KV cache 显著降低。
- **Device-Limited Routing M=3**：当 M=∞（无限制）相比，MMLU 几乎一致但跨节点 IB 流量大幅下降；M=3 是 8-way EP × 2-token-per-device 配置下的甜点。
- **Communication balance loss**：去掉后训练后期出现 device 偏斜 → throughput 下降 ~5%。
- **Token-dropping**：drop 后保留 ~90% tokens 完整无丢损失差距 < 0.01 perplexity，但吞吐显著好转。

## 对 16B MoE 设计的启示

1. **可直接复用 V2-Lite 当作起点**：
   - 27 层，hidden 2048，64 routed + 2 shared，K=6，d_expert=1408，总参 15.7B / 激活 2.4B 几乎就是 16B-class 的官方推荐配置。
   - MLA（无 q-lora，因为 hidden=2048 已经够小）+ d_c=512 + d_h^R=64 直接抄。
   - 节省 KV cache 是 16B-class 在端侧/单卡部署的核心卖点。
2. **配方可复用**：
   - Optimizer/LR/初始化 std=0.006 与 batch ramp-up 全套可继承。
   - balance loss 三件套（α_1=0.003, α_2=0.05, α_3=0.02）仍然适用；但若上 16B 建议优先用 V3 的 auxiliary-loss-free + α_seq=1e-4 复合（见论文 3 / 4）。
   - Device-Limited Routing M=3 是 EP=8 时甜点；若 16B 用 EP=4 则 M=2 即可。
3. **不要直接复用的点**：
   - V2 总参 236B 的 N_routed=160 不适合 16B-class，应该缩到 N_routed=64（与 V2-Lite 一致）；否则单专家容量太低，路由噪声变大。
   - V2 没有 MTP，V3 才引入；新建 16B 时建议直接采纳 MTP D=1（论文 4）。
   - V2 用 RMSNorm + LayerNorm 标配，未采纳 QK-Norm / muonClip；如果训练规模 < 16B 且数据干净，可保持简洁。

## Caveats / 局限

- V2-Lite 在论文正文 Appendix B 提及但参数细节抓取在不同 WebFetch 版本间略有出入，最终需以 HF config 为准（论文内文也存在两处口径稍差）。
- MLA 的 W^{UK} 在推理时可以被吸收进 W^{UQ}（math identity），论文提到这一点，但对训练梯度的具体处理需查代码实现。
- 论文未公开 YaRN 长上下文扩展的精确 scaling 系数。
- 三层 balance loss 系数（0.003 / 0.05 / 0.02）对训练有副作用——V3 把这套体系大改为 auxiliary-loss-free，意味着 V2 的 balance 设计**不是最优**。
- 256K vocabulary（早期 DeepSeek-LLM）vs 100K（V2）切换是否会影响 token 效率，论文未做对照。
- Token-dropping 在 SFT/inference 阶段不会启用，会有 train/inference gap；这一点论文只用 "10% 永不 drop" 部分缓解。
