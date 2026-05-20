# Qwen3 Technical Report

- **arXiv**: 2505.09388 (v1 submitted May 14, 2025)
- **机构**: Qwen Team, Alibaba Cloud (阿里巴巴通义实验室)
- **发表时间**: 2025-05-14
- **作者(代表)**: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, ... (Qwen Team，共 60+ 作者)
- **代码 / 权重**: huggingface.co/Qwen, 包含 6 个 dense + 2 个 MoE 版本

## TL;DR

Qwen3 是 Qwen 系列的第 3 代基础模型，覆盖 0.6B–235B 的 dense 与 MoE 模型，最大模型 **Qwen3-235B-A22B**（235B 总参 / 22B 激活）。两个核心创新：
1. **思考 / 非思考统一模型**：单一模型同时支持 `/think` 与 `/no_think` 两种推理风格，并可通过 "thinking budget" 控制推理 token 数。
2. **多语言扩展**：从 Qwen2.5 的 29 语言扩展到 **119 语言/方言**，预训练 **36T tokens**。

MoE 设计上明确选择**不使用 shared expert**（与 Qwen2.5-MoE、DeepSeek、Kimi 路线不同），并用 **global-batch load balancing loss** 替代 per-micro-batch aux loss，鼓励专家专业化。

## 关键架构配置

### Qwen3-30B-A3B (MoE, 中型)

| 项 | 值 |
|---|---|
| 总参 / 激活 | 30B / 3.3B |
| Layers | 48 |
| Hidden size | 2,048 |
| FFN intermediate (dense layers) | 6,144 |
| MoE intermediate (expert dim) | 768 |
| Num attention heads (Q) | 32 |
| Num KV heads (GQA) | 4 |
| Head dim | 128 |
| N_routed experts | 128 |
| Top-K | 8 |
| **N_shared experts** | **0 (无 shared expert)** |
| 激活率 | 8/128 ≈ 6.25% |
| Context length (config) | 32,768 (40,960 max), 原生 128K |
| Vocab size | 151,936 (BBPE, 151,669 实际 token) |
| Tied embeddings | No |
| RoPE θ | 1,000,000 |

### Qwen3-235B-A22B (MoE, 旗舰)

| 项 | 值 |
|---|---|
| 总参 / 激活 | 235B / 22B |
| Layers | 94 |
| Hidden size | 4,096 |
| FFN intermediate (dense layers) | 12,288 |
| MoE intermediate (expert dim) | 1,536 |
| Num attention heads (Q) | 64 |
| Num KV heads (GQA) | 4 |
| Head dim | 128 |
| N_routed experts | 128 |
| Top-K | 8 |
| **N_shared experts** | **0 (无 shared expert)** |
| 激活率 | 8/128 ≈ 6.25% |
| Context length (config) | 32,768 → 128K |
| Vocab size | 151,936 |
| Tied embeddings | No |
| RoPE θ | 1,000,000 |

### 共通设计要点

- **Attention**: Grouped-Query Attention (GQA)；**QK-Norm**（替换 Qwen2 中的 QKV-bias）以稳定训练；移除 QKV bias。
- **Normalization**: RMSNorm + pre-normalization。
- **Activation**: SwiGLU (SiLU)。
- **Position encoding**: RoPE，使用 ABF 把 base frequency 从 10K 提升到 **1,000,000**。
- **Dense 前缀层**: 论文未明确说明有 dense pre-layer（与 DeepSeek-V3 / Ling-1T 不同）。HF config 中也无 `first_k_dense_replace` 类设置；所有 transformer block 看起来都用 MoE FFN。
- **MTP**: ❌ Qwen3 **没有** Multi-Token Prediction head（与 DeepSeek-V3、Kimi K2 后续工作、Ling 2.0 不同）。
- **Tokens / 优化器 / 精度**: 36T tokens 预训练；BF16 精度（HF 权重为 bfloat16）；具体 batch size、peak LR、warmup 未公开。

### dense 兄弟模型（用于知识蒸馏到小模型）

包含 0.6B / 1.7B / 4B / 8B / 14B / 32B（6 个 dense）+ 30B-A3B / 235B-A22B（2 个 MoE）。小尺寸 dense 使用 **tied embeddings**，大尺寸不使用。

## 核心方法 / 创新点

### 1. 不使用 shared expert (vs. Qwen2.5-MoE / DeepSeek)

> "Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts."

论文未给出公开的消融论证。结合 global-batch load balancing 推断，团队认为在大 batch 下 router 已能自然学到 "通用专家"，再显式增加 shared expert 会减少专家专业化的空间。这是和 DeepSeek-V3 / Kimi K2 / Hunyuan-Large / Ling 2.0 的明显分歧。

### 2. Global-Batch Load Balancing Loss

> "we adopt the global-batch load balancing loss (global_balance) to encourage expert specialization."

核心思想：传统 aux loss 在 **micro-batch**（GPU local 上的 mini-batch）内强制专家均衡；当某个 micro-batch 数据分布很窄（比如全是 Python 代码）时，会迫使所有专家都被均匀使用，**反而抑制了专家的专业化**。Global-batch aux loss 在更大的 batch 维度（DP 维度聚合）上做平衡，允许 micro-batch 内 router 高度倾斜到少数专家，但全局上仍然均衡。

论文未给出精确公式；该策略原始来源于近期 router 文献（如 OLMoE、DeepSeek 系列工作中的讨论）。直观上，aux loss 项是：

```
L_aux = α · Σ_i f_i · P_i
```

其中 f_i 是全局聚合的 token-to-expert 频率，P_i 是 router 输出的全局 mean gating probability。与 per-micro-batch 相比，f_i 和 P_i 都跨 DP rank 做 all-reduce 才参与 loss 计算。

### 3. QK-Norm

在 attention 的 query/key projections 后插入 RMSNorm，即：
```
Q' = RMSNorm(Q_proj(x)),  K' = RMSNorm(K_proj(x))
```
作用：限制 attention logits 的数值范围，防止 attention 在长 context / 高 LR 下出现极端值导致 NaN/Inf。论文未给出精确公式但说明 "introduce QK-Norm to the attention mechanism to ensure stable training"。

（对比：Kimi K2 用 **MuonClip** 中的 QK-clip（post-update weight rescaling）代替 QK-Norm 这种 forward-pass normalization。）

### 4. 双模式（thinking / non-thinking）的训练

四阶段 post-training：

1. **Long-CoT Cold Start SFT**：在 math、code、STEM 题目上做长链式思考冷启动，严格过滤掉 "不需推理就能答对" 的问题。
2. **Reasoning RL (GRPO)**：3,995 道题目 + verifier 对，170 步 RL training，AIME'24 从 70.1 → 85.1。
3. **Thinking Mode Fusion SFT**：把 thinking 和 non-thinking 数据混合 SFT，用 chat template 设计 `/think` 与 `/no_think` 切换。non-thinking 样本保留一个空的 `<think></think>` 块。
4. **General RL**：跨多个 downstream task 的通用 RL，进一步对齐。

### 5. Thinking Budget

> 用户给定一个 token 上限。当 thinking 达到上限时，模型被注入一个 stop 指令："Considering the limited time by the user, I have to give the solution based on the thinking directly now"，然后模型立即基于已有思考生成最终答案。

论文强调：**"this capability is not explicitly trained but emerges naturally as a result of applying Thinking Mode Fusion"**。

### 6. 强→弱蒸馏

对 ≤8B 的小模型，直接从 Qwen3-235B-A22B 和 Qwen3-32B teacher 蒸馏 output logits，**比走完整 4 阶段后训省 ~10× GPU hours，且效果更好**。

## 训练 & 系统细节

### 预训练数据 (36T tokens, 119 语言)

3 stage progressive curriculum：

| Stage | Tokens | Seq Len | 描述 |
|---|---|---|---|
| S1 General | 30T | 4K | 大规模通用知识，119 语言 |
| S2 Reasoning | ~5T | 4K | STEM / code / reasoning 高质量数据浓度上升 |
| S3 Long-Context | 0.1T+ | 32K | 75% 数据为 16K-32K 长 chunk |

### 优化器与稳定性
- 论文未公开 batch size、peak LR、warmup 步数
- 未提到训练中的 loss spike 事件
- Scaling laws "for optimal hyper-parameters" 在三个 stage 都做了

### Tokenizer
- BBPE，词表 151,669（HF config 中 vocab_size pad 到 151,936）
- 多语言扩展：从 Qwen2.5 的 29 语言到 119 语言/方言

## 关键消融与结果

### 性能（论文报告）

- Qwen3-235B-A22B 与 DeepSeek-V3、Llama 3.1 405B、Claude 3.5 Sonnet、GPT-4o 在主流 benchmark 上对标
- Qwen3-30B-A3B 是中小型 MoE 的强基线，3.3B 激活打出 30B+ dense 的水平

### Thinking mode 增益
- AIME 24: 70.1 → 85.1（仅 170 步 RL）
- 各 reasoning benchmark 都有大幅增益

### 蒸馏效率
- 4-stage full post-training vs. logit distillation: 后者 1/10 GPU hours 且效果更好（对小模型）

## 对 16B MoE 设计的启示

1. **不一定要 shared expert**。Qwen3 是当前主流大型 MoE 中唯一坚定不用 shared expert 的路线（DeepSeek、Kimi K2、Hunyuan-Large、Ling 2.0 都用了 1 个 shared）。设计 16B MoE 时应将此作为一个实际可比的 ablation 维度。

2. **Global-batch aux-loss 的工程实现**。如果用 aux loss 而非 aux-loss-free 路线（DeepSeek-V3 / Ling 2.0 的方案），应跨 DP 做 all-reduce 后再算 aux loss，micro-batch 内不强制均衡。

3. **GQA + QK-Norm 是稳健组合**。head dim 128，KV heads 4，Q heads = hidden/128。对 16B / hidden_size=2048 的模型，对应 16 Q heads + 4 KV heads 是合理起点。

4. **专家粒度**：N=128, top-K=8（6.25% 激活）、moe_intermediate_size = hidden_size × 0.375 (768 vs 2048) 是一个 fine-grained 设计。对于 16B 模型，N=128/top-8/expert_dim=768 直接 portable。

5. **MTP 并非强制**。Qwen3 没有 MTP 也能拿到 SOTA reasoning 性能；MTP 主要影响 inference speed (speculative decoding) 和 code/math 数据效率。

6. **Thinking budget 是 emergent property**。在 SFT 阶段混合两种模式 + 简单的 stop instruction 注入即可实现可控推理深度，不需要显式训练 budget 控制头。

7. **强→弱蒸馏对 16B 极其有效**。如果有 Qwen3-235B-A22B 这种 teacher，直接做 logit distillation 比从零跑 4 阶段 post-training 更省、更好。

## Caveats / 局限

- **大量细节未公开**：精确 batch size、peak LR、warmup、weight decay、weight init、loss spike、global-batch aux loss 的具体公式与 α、训练 FLOPs、训练硬件 GPU/集群规模、数据数据混合配比等都未在 v1 中披露。需要参考 HF 配置文件或后续社区分析。
- **No shared expert 决策无消融**：论文未给出 "with/without shared expert" 的对照实验数据，只是陈述事实。
- **QK-Norm 公式未给出**：只声明使用，没有方程。
- **MoE scaling law 缺少**：与 Hunyuan-Large、Ling 2.0 不同，Qwen3 没有显式发表自己的 MoE scaling law 曲线（只提到 "develop scaling laws for optimal hyper-parameters"）。
- **统一思考/非思考之 trade-off**：论文承认双模式 "may degrade the model's performance" if not carefully fused，提示这种 unified design 不是免费午餐。
