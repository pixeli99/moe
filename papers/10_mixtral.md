# Mixtral of Experts

- **arXiv**: 2401.04088
- **机构**: Mistral AI
- **发表时间**: 2024 年 1 月
- **作者(代表)**: Albert Q. Jiang、Alexandre Sablayrolles、Arthur Mensch、Guillaume Lample、Devendra Singh Chaplot、William El Sayed 等

---

## TL;DR

Mixtral 8x7B 是第一个真正"开源权重 + 商业可用"的 SOTA MoE，**47B 总参 / 13B 激活**，架构上完全延续 Mistral 7B（同样 dim=4096、32 层、SwiGLU、RoPE、GQA、SWA），只是把 FFN 替换为 "**N=8 experts, top-K=2**" 的 MoE 层。没有 shared expert。Routing 是最朴素的 `softmax(TopK(x · W_g))`。在 13B 激活下全面打平/超越 Llama 2 70B，并在多语言、Math、Code 上显著领先。后续推出的 **Mixtral 8x22B (141B 总 / 39B 激活)** 沿用同设计，是更大尺度的版本。论文的最大贡献不在"创新"——其架构是 Switch + GShard + Mistral 的直接组合——而在 **"用极小 active 参数（13B）证明 MoE 路线在工业级表现的可行性"**。Section 5 的 routing 分析也是少数公开的 MoE expert 专精化研究（结论：**专精化主要按句法/位置而非按主题**）。

---

## 关键架构配置

### Mixtral 8x7B (主模型)

| 项目 | 值 |
|---|---|
| 总参 | 46.7 B (常被取整为 47B) |
| 激活参数/token | 12.9 B (常被取整为 13B) |
| Sparsity (active/total) | ~28% |
| **N_routed** | **8** |
| **Top-K** | **2** |
| **N_shared** | **0** |
| **d_expert (FFN intermediate)** | **14,336** （= Mistral 7B 的 hidden_dim） |
| Layers | **32** |
| Hidden / model dim | **4096** |
| Attention heads (Q) | **32** |
| KV heads (GQA) | **8** （GQA 8:1） |
| Head dim | **128** |
| 序列长度 (context) | **32,768** |
| Vocab size | **32,000** （Mistral SentencePiece BPE） |
| RoPE θ | 1,000,000 (Mistral 7B 也用此值) |
| Norm | RMSNorm |
| Activation | **SwiGLU** |
| Sliding Window Attention | Mistral 7B 用 SWA=4096；Mixtral 论文说"fully dense context length of 32k"，**取消了 SWA** |
| Routing | `Softmax(TopK(x · W_g))`，linear gate + softmax 仅在 top-2 上归一化 |
| Aux loss | 论文中**未明确给出系数**；基础设施实现常用 ~0.01 |
| Capacity factor | dropless（用 Megablocks 内核） |
| Dense 前缀 | 0（每层 FFN 都被替换） |
| Tokenizer | Mistral BPE (32k) |
| 训练 tokens | **未公开** |
| 优化器 | **未公开** |
| 精度 | 训练 bf16 |

参数核验：8 expert × 2 × (4096 × 14336) ≈ 8 × 117M × 2 (SwiGLU 双路) ≈ 1.87 B per layer 的 FFN 总量 × 32 层 ≈ 60 B 仅 FFN。算上 attention/嵌入约 47 B 总参；激活时 2/8 比例 ≈ 13 B。

### Mixtral 8x22B（论文未单独发，但是 Mistral 后续以同 family 发布）

| 项目 | 值 |
|---|---|
| 总参 | **141 B** |
| 激活/token | **39 B** |
| N_routed / Top-K | 8 / 2 |
| Layers | 56 |
| Hidden | 6144 |
| FFN | 16,384 |
| Attention heads (Q/KV) | 48 / 8 (GQA) |
| Head dim | 128 |
| Context | **65,536** |
| Vocab | 32,768 (升级 tokenizer，多语言更优) |

---

## 核心方法 / 创新点

1. **直接复用 Mistral 7B 架构**：FFN 8 份并并联 + 一个 router。
2. **Sparse MoE 的 routing 公式（论文 §2.1）**：

   $$ y = \sum_{i=0}^{n-1} \text{Softmax}(\text{Top2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x) $$

   即先取 top-2 logits，对这 2 个值做 softmax 归一化（其余视为 −∞），所以**softmax 只在被选中的 K 个 expert 之间归一化**，而不是先 softmax 全部 N 个再 top-K——这是 Mixtral 的一个具体实现细节。
3. **没有 shared expert，没有专门 aux loss 讨论**：论文里完全没 detail aux 配方，把这些当成"工程默认"。
4. **Megablocks 内核 + Expert Parallelism**：明确把 MoE 实现作为系统级关键（贡献到 vLLM），是后续 MoE 推理普及的关键工程基础。
5. **§5 Routing 分析（少有的实证研究）**：
   - 在 The Pile 多领域 (ArXiv/PubMed/Github/Wikipedia/...) 上看 expert 分布，**没看到明显的"主题专家"**。
   - 唯一例外：DM Mathematics 因为是合成数据、不自然语言，分布偏离均匀。
   - 但 **token 级有强 positional locality**：相邻两个 token 用同一 expert 的概率远高于随机（layer 15 first-choice 重复率 ~27% vs 随机 12.5%；first-or-second 重复率 ~62% vs 随机 46%）。
   - 例子：Python 中 `self`、英文 `Question` 之类的多 token 词序列总是路由到同一 expert，**专精化按句法/词形而非主题**。这是后续 MoE 优化（caching、speculative routing）的实证基础。

---

## 训练 & 系统细节

- **训练数据**：未公开（Mistral 政策）。论文只说"multilingual data, upsample relative to Mistral 7B"。
- **训练 token 量**：未公开。
- **优化器/超参**：未公开。
- **硬件支持鸣谢**：CoreWeave、Scaleway。
- **推理栈**：vLLM + Megablocks CUDA kernels；NVIDIA 协助集成 TensorRT-LLM + Triton。
- **License**：Apache 2.0（学术 + 商用均可）。
- **后训**：SFT + DPO 得到 Mixtral 8x7B Instruct，MT-Bench 8.30，Arena Elo 1121（2023-12 数据），全面超过 GPT-3.5 Turbo / Claude-2.1 / Gemini Pro / Llama-2-70B-chat。

---

## 关键消融与结果

Mixtral 论文**几乎没有架构消融**——这是它和 OLMoE/DeepSeek-V2 最大的差别。其结果章节几乎纯是与其它模型对比：

### vs Llama-2 系列 (Table 2)

| Benchmark | Llama 2 7B | Llama 2 13B | Llama 2 70B | Mistral 7B | **Mixtral 8x7B (13B active)** |
|---|---|---|---|---|---|
| MMLU | 44.4 | 55.6 | 69.9 | 62.5 | **70.6** |
| HellaSwag | 77.1 | 80.7 | 85.4 | 81.0 | 84.4 |
| Arc-c | 43.2 | 48.8 | 56.5 | 54.9 | **59.7** |
| TriviaQA | 56.6 | 64.0 | 73.0 | 62.5 | 71.5 |
| HumanEval | 11.6 | 18.9 | 29.3 | 26.2 | **40.2** |
| MBPP | 26.1 | 35.4 | 49.8 | 50.2 | **60.7** |
| Math | 3.9 | 6.0 | 13.8 | 12.7 | **28.4** |
| GSM8K | 16.0 | 34.3 | 69.6 | 50.0 | **74.4** |

→ **13B 激活打平/超越 70B dense**，Math/Code 上 **2× 提升**。这是 MoE 路线"5× 推理效率"的最强 marketing point。

### vs GPT-3.5 + Llama-2 70B (Table 3)

MMLU、ARC-Challenge、MBPP、GSM-8K 上 Mixtral 全部领先；HellaSwag、Winogrande 略逊。

### 多语言 (Table 4)

法/德/西/意 ARC-c、HellaSwag、MMLU 上 Mixtral 8x7B (13B active) 全面 **大幅超过** Llama-2-70B（最大差距 +10 pp on ARC-c French）。

### Long Context (Figure 4)

- Passkey retrieval：32k 范围内 100% 准确（位置 + 长度无关）。
- proof-pile 困惑度：单调下降到 32k，证明真支持 32k context。

### Routing 局部性 (Table 5)

| 数据集 | Layer 0 first-choice 重复率 | Layer 15 重复率 | Layer 31 重复率 |
|---|---|---|---|
| ArXiv | 14.0% | 27.9% | 22.7% |
| Github | 14.9% | 28.1% | 19.7% |
| Wikipedia | 14.4% | 23.6% | 25.3% |
| 随机基线 | 12.5% | 12.5% | 12.5% |

第 0 层重复率几乎等于随机；中后层显著高于随机——意味着 MoE 缓存策略在中后层更有效。

---

## 对 16B MoE 设计的启示

1. **粗粒度也能工作，但是不是最优**：Mixtral 用 N=8 / top-2 / d_expert=14336 的"巨大 expert"路线，效果好。但同时 DeepSeekMoE/OLMoE/Qwen-MoE 都已实证"细粒度更好"。**16B 设计应往细粒度走（N≥32），不学 Mixtral 的 N=8**。
2. **完全沿用 dense 架构剩余部分**：除了 FFN→MoE，attention/norm/RoPE 都和 dense 兄弟一样。**这是工程上最简单的方案**——所有 dense 优化（GQA、KV cache、SWA、FlashAttention）即插即用。
3. **没有 shared expert**：与 OLMoE 一致。在 N 较小（8）+ top-2 较小时，shared expert 占用相对更多预算，不值得。
4. **dropless token-choice + Megablocks**：已成行业默认，无需重新评估。
5. **GQA 8:1 必备**：Mixtral 用 32 Q / 8 KV，让 KV cache 在 32k context 下还能装得下。**16B 设计的长上下文 path 必然走 GQA。**
6. **Routing 专精化几乎无主题信号、但有强位置局部性**：意味着 token-level caching、speculative decoding、专家 affinity prefetching 都能挖到收益。**16B 服务架构应预留这些优化空间。**
7. **routing 公式细节**：先 TopK 再 softmax（不是先 softmax 再 TopK）——能确保被选中的 top-K 概率之和恰好 = 1，避免 dead expert 的概率泄漏。

---

## Caveats / 局限

1. **训练细节几乎全黑箱**：tokens、数据、优化器、batch、LR、aux loss 系数全部未公开。无法用 Mixtral 论文做"复现指南"。OLMoE / Skywork / DeepSeek-V2 在这点上完胜。
2. **没有任何架构消融**：N=8 vs 16 vs 32 不知道；top-2 vs top-4 不知道；shared vs no-shared 不知道；upcycle vs from-scratch 不知道——一切只能从其它论文交叉推断。
3. **Routing 分析样本量小**：只看了 The Pile 一个数据集，3 个 layers，比较粗。
4. **Sparsity 偏低 (28%)**：vs OLMoE (19%) / DeepSeek-V2 (~9%) / Yuan-M32 (9%) 都明显高。"激活参与总参之比"较高意味着 MoE 的"压缩率"较低，但好处是单 expert 内特征更稠密、更稳定。这是工程取舍点。
5. **8x22B 没有独立论文**：只能从 model card / 配置文件推断。
6. **长上下文测试单一**：只有 passkey + proof-pile 困惑度，对真实 RAG / 代码 / 多轮场景未深测。
