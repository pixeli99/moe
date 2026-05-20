# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

- **arXiv**: 2401.06066
- **机构**: DeepSeek-AI
- **发表时间**: 2024 年 1 月 (v1)
- **作者(代表)**: Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Daya Guo, Meng Li, Y. Wu, Zhenda Xie 等

## TL;DR
DeepSeekMoE 提出两项核心架构创新使 MoE 走向"专家特化(expert specialization)"的极致：(1) **Fine-Grained Expert Segmentation**（细粒度专家分割），把每个标准 FFN 切成 m 份得到 mN 个小专家、激活 mK 个，在保持总参数与激活参数不变的前提下显著提升专家组合数；(2) **Shared Expert Isolation**（共享专家隔离），把 K_s 个专家强制对所有 token 激活，用于学习公共知识、减少路由专家间的冗余。论文以 2B、16B、145B 三个尺度验证：DeepSeekMoE-2B 在 0.3B 激活下 Pile loss 1.808，超 GShard 0.3B（1.867）并逼近 GShard×1.5；DeepSeekMoE-16B 仅用约 40% 计算量即匹配 DeepSeek 7B Dense 与 LLaMA2 7B。该论文奠定了后续 DeepSeek-V2/V3 的 MoE 基本范式。

## 关键架构配置

### DeepSeekMoE-16B
- **总参 / 激活参数**: 16.4B / 2.8B（约 17% 激活）
- **Layers**: 28
- **Hidden**: 2048
- **Attention heads**: 16（标准 MHA，未引入 MLA）
- **N_routed / Top-K / N_shared**: 64 / 6 / 2（即 mN=64, mK=6, K_s=2）
- **细粒度分割 m**: 4（每个专家 FFN intermediate ≈ 标准的 1/4）
- **训练 tokens**: 2T
- **位置编码 / Norm / Activation**: RoPE / RMSNorm / SwiGLU（沿用 LLaMA 习惯）
- **Dense 前缀层数**: 第 1 层为 dense FFN，其余 27 层均替换为 MoE

### DeepSeekMoE-2B（消融/对比基线）
- 总参 / 激活: 2.0B / 0.3B
- Layers: 9，Hidden: 1280，Heads: 10
- N_routed=63, K_r=7, N_shared=1（mN=64 fine-grained experts, m=4）
- 训练 tokens: 100B
- 用于本文绝大多数消融

### DeepSeekMoE-145B（preliminary）
- 总参 ≈ 145B
- Layers: 62, Hidden: 4096, Heads: 32
- 训练 tokens: 245B（初步实验，未训练完整）

## 核心方法 / 新点

### 1. Fine-Grained Expert Segmentation（Eq. 6-8）
基线 GShard：N 个专家，激活 top-K。
DeepSeekMoE 把每个专家 FFN 在中间维度切成 m 份，得到 mN 个细粒度专家，相应将激活专家数提高到 mK：
$$
\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{mN} g_{i,t}\,\text{FFN}_i(\mathbf{u}_t)
$$
其中 g_{i,t} 在 mN 个专家中做 top-mK 选择。

**直觉**：提高组合数（combinatorial flexibility）。论文给出对比：GShard 16-experts top-2 仅 C(16,2)=120 种组合；DeepSeekMoE 64-experts top-8 有 C(64,8)≈4.4B 种组合。专家细分后每个专家可承担更窄的"知识子域"，缓解一个专家被强行学多种异质知识的问题。

### 2. Shared Expert Isolation（Eq. 9-11）
将其中 K_s 个专家固定为"共享专家"，对所有 token 必激活；其余 mN-K_s 仍走 top-(mK-K_s) 路由：
$$
\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{K_s}\text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=K_s+1}^{mN} g_{i,t}\,\text{FFN}_i^{(r)}(\mathbf{u}_t)
$$
**动机**：把"通用知识"集中到共享专家，让路由专家专注差异化知识，从而降低参数冗余、提高激活效率。

### 3. Expert-Level + Device-Level Balance Loss（Eq. 12-17）
- 专家级负载均衡：
$$
\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i
$$
  其中 f_i 是 token 选中专家 i 的频率，P_i 是平均路由概率。α_1：2B 用 0.01，16B 用 0.001。
- 设备级负载均衡（Eq. 15-17）：将 mN-K_s 个路由专家划分到 D 个设备组，定义组内平均频率 f'_i 和累计概率 P'_i：
$$
\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f'_i P'_i
$$
  α_2 比 α_1 大（论文中 α_2=0.05 这一量级，明确不同尺度）。
- **16B 用 pipeline parallelism**，每层所有专家放在同一设备上，因此使用 device-level balance loss 而非 token-dropping。

## 训练 & 系统细节

- **数据**: 与 DeepSeek 67B 同源的中英语料（DeepSeek-LLM 数据管线）。
- **优化器**: AdamW（β₁=0.9, β₂=0.95, weight_decay=0.1）。
- **学习率**: warmup-then-step-decay（与 DeepSeek-LLM 系列一致）。
- **精度**: BF16 训练。
- **并行**: pipeline parallelism + expert parallelism（每层专家放同设备组）；16B 阶段未启用 tensor parallel 以减少通信。
- **稳定性**: 使用 RMSNorm；balance loss 系数随规模缩小（2B→0.01, 16B→0.001）。

## 关键消融与实验结果

### Table 1 — 2B/0.3B 激活下各 MoE 方法对比（Pile loss，越低越好）
| Method | Pile loss |
|---|---|
| Dense baseline | – |
| Hash Layer | 1.932 |
| Switch Transformer | 1.881 |
| GShard | 1.867 |
| **DeepSeekMoE** | **1.808** |

DeepSeekMoE 在同等激活成本下大幅领先。

### Table 2 — DeepSeekMoE-2B vs GShard×1.5（专家数 1.5×）
DeepSeekMoE 2B 与 GShard 1.5× 表现相当，意味着达到 GShard 同效果只需 ~67% 的专家计算预算。

### Figure 3 — Fine-Grained Segmentation 消融（曲线持续下降）
保持总参与激活参数恒定，把 16 experts top-2 → 32 experts top-4 → 64 experts top-8 → 128 experts top-16，loss 单调下降；论文措辞 "continuous refinement of expert segmentation granularity corresponds to continuous enhancement"，但实际选择 m=4 作为性能/工程平衡点。

### Figure 4-5 — Shared Expert 与冗余度
- 禁用共享专家后 Pile loss 从 **1.808 → 2.414**，说明共享专家承担的功能不可被路由专家替代（论文意义上的 "irreplaceability"）。
- 把激活路由专家从 6 减到 4，DeepSeekMoE 仍可逼近 GShard×1.5 同样的 loss，说明 fine-grained 设计降低了专家间冗余。

### Table 3 / Table 5 — DeepSeekMoE-16B 评测
Base 模型（vs DeepSeek 7B Dense）：
| Benchmark | DeepSeekMoE-16B | DeepSeek 7B Dense |
|---|---|---|
| MMLU | 45.0 | 48.2 |
| HumanEval Pass@1 | 26.8 | – |
| GSM8K EM | 18.8 | – |

Chat 模型（vs LLaMA2-7B SFT）：
| Benchmark | DeepSeekMoE-16B Chat |
|---|---|
| MMLU | 47.2 |
| BBH | 42.2 |
| HumanEval | 45.7 |
| GSM8K | 62.2 |

总体来看 16B 在约 **40% 计算量**下与 DeepSeek 7B Dense / LLaMA2-7B 相当。

## 对 16B MoE 设计的启示

1. **可直接复用的配方**：
   - 细粒度分割 m=4 是被多次验证的甜点；激活比例约 12-17%（2.8/16.4）。
   - 1 第一层 dense FFN + 其余 MoE 的"前缀 dense"做法（V2/V3 也沿用，V3 进一步扩到 3 层）。
   - 共享专家数 K_s=2、路由 top-K=6 是 16B-class 上的稳健默认。
   - balance loss 系数随规模缩小（16B 用 α_1=0.001）。

2. **要注意的差异**：
   - 本文的 attention 还是普通 MHA，不是 MLA；若做 16B-class 模型并希望低 KV-cache 推理，应直接采纳 V2 的 MLA（论文 2 的发现）。
   - 用了 explicit expert-balance loss + device-balance loss；但 V3 转向 auxiliary-loss-free（论文 3/4），后者训练稳定性与最终性能更优；16B 设计建议跳过这一步直接用 loss-free + 极小 sequence-wise α=1e-4。
   - 145B 仅 245B token 是"初步实验"，不能据此外推大规模训练曲线；要参考 V2/V3 的 14.8T 配方。

3. **反例提醒**：
   - 表面上"继续切得更细"看似越好（Figure 3），但 m 过大会导致 all-to-all 通信压力和 router 不稳定；m=4 是大多数后续 DeepSeek 模型的默认。
   - 若 K_s=0（无共享专家），论文显示 loss 从 1.808 退化到 2.414，幅度巨大，**不要去掉共享专家**。

## Caveats / 局限

- 145B 实验未跑完整训练，对 16B→更大规模的外推证据有限；论文的强大消融仅在 2B 上做。
- 没有 MLA，KV cache 与推理效率显著差于 V2/V3，做 16B 部署友好模型应不沿用本文 attention 配置。
- 没有公开训练数据混合比例细节；仅说"DeepSeek-LLM 同源中英文"。
- balance loss 即使较小仍可能干扰梯度方向；DeepSeek 自己在 8 个月后（2408.15664）转向了 auxiliary-loss-free，表明本文方法不是终局。
- 论文未给出 m 与 d_expert 的精确表达式（如 d_expert = d_FFN_baseline / m）；需要从开源 config 反推（HF: deepseek-ai/deepseek-moe-16b-base）。
- 论文中没有显式给出 sequence-wise balance loss，只是 token-level；后续 V2/V3 才加入 device-level / communication / sequence-wise 三层补充。
