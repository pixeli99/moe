# dots.llm1 Technical Report

- **arXiv**: 2506.05767 (v1, 2025-06-06)
- **机构**: rednote-hilab（小红书 RedNote AI 实验室）
- **发表时间**: 2025-06-06
- **作者**: 27 名（rednote-hilab 集体署名）
- **代码 / 权重**: github.com/rednote-hilab/dots.llm1 ; huggingface.co/rednote-hilab

## TL;DR

dots.llm1 是 **142B 总参 / 14B 激活**的 MoE 基础模型，单节点 8×H800（40/80GB）就能 inference。论文核心定位是"**性价比 SOTA**"：11.2T 高质量 token、**完全不用合成数据**，pretrain 总成本只要 **1.456M H800-hours**（Qwen2.5-72B 的 ¼）。架构是 **DeepSeekMoE 框架 + V3 路由**的直接继承，再加 **MHA + QK-Norm**（不是 MLA、不是 GQA）和 rednote 自研的 **Cybertron framework + 1F1B all-to-all overlap** infra。

性能上 base model **均分接近 Qwen2.5-72B、超过 DeepSeek-V2 base**，但**整体落后 DeepSeek-V3 base** 几个点，post-training 后差距相似。**没有 MTP、没有 FP8、没有 node-limited routing**——三件 V3/Ling 的"高级"项 dots1 都没碰，靠基础配方 + 数据质量打到这个位置。

**对本仓库的最大价值**：dots1 是 **Ling 2.0 routing 配方的祖代**（`routed_scaling_factor=2.5` 是确凿指纹），证明 V3 风格 ALF 不带 Ling 零均值修正在 11.2T tokens 下也能稳定收敛——loss 曲线无不可恢复 spike、训练全程不丢 token。

## 关键架构配置

来源：HF `config.json`（rednote-hilab/dots.llm1.base）+ 论文 §2 + §4.2。

| 项 | 值 |
|---|---|
| 总参 / 激活参数 | **142B / 14B**（≈ 9.9% 激活） |
| Layers | **62** |
| Hidden | **4096** |
| **Attention** | **MHA**（vanilla multi-head；32 Q heads = 32 KV heads，**没有 GQA**） |
| Head dim | **128** |
| **QK-Norm** | **✓**（RMSNorm applied to Q/K projections, §2.Attention Layer） |
| Dense FFN intermediate | **10944**（仅第 0 层） |
| **MoE expert intermediate** | **1408** |
| **N_routed / Top-K / N_shared** | **128 / 6 / 2** → 8 active experts/token |
| **Active/Total params** | **≈ 9.9%** (14B/142B) |
| **Expert-slot fraction** `(K+N_sh)/(N_rt+N_sh)` | **≈ 6.2%** (8/130) |
| **Routed-only fraction** `K/N_rt` | **≈ 4.7%** (6/128) |
| **Dense 前缀层** (`first_k_dense_replace`) | **1** |
| Routing gate | **Sigmoid + Top-K + 归一化** (`scoring_func=sigmoid`, `topk_method=noaux_tc`, `norm_topk_prob=true`) |
| **`routed_scaling_factor`** | **2.5** ⚡（**与 Ling 2.0 完全一致**，是继承指纹） |
| Load balancing | **Aux-loss-free (Wang 2024)** + sequence-wise balance loss；**无 NLR** |
| Gating numerics | **FP32**（即使主体 BF16，gating 单独提到 FP32） |
| MTP | **无** |
| Vocab size | **152064**（BBPE，与 Qwen2 系列规模相同） |
| Position encoding | **RoPE, base = 1e7** |
| Normalization | RMSNorm pre-norm + QK-Norm；`rms_norm_eps=1e-5` |
| Activation | SwiGLU (silu) |
| Init | std = 0.006（与 Ling 2.0 相同） |
| Tied embeddings | ✗（untied）|
| Max seq | 32768 |
| 精度 | **BF16**（无 FP8） |

### 与 Ling 2.0 / V3 的指纹级对比

| 项 | dots1 (2025-06) | Ling 2.0 (2025-10) | DeepSeek V3 (2024-12) |
|---|---|---|---|
| 总参 / 激活 (active/total) | 142B / 14B (**9.9%**) | 16B/103B/1T (**8.75% / 5.9% / 5.1%**；expert-slot ≈3.5%) | 671B / 37B (**5.5%**) |
| 路由 gate | sigmoid + noaux_tc | sigmoid + noaux_tc | sigmoid + noaux_tc |
| `routed_scaling_factor` | **2.5** | **2.5** | 2.5（V3 也用此值） |
| Bias update | V3 sign(e_i) | **Ling 零均值** sign−mean(sign) | V3 sign(e_i) |
| Sequence aux loss | ✓（α 未给具体值） | ✓ | ✓ α=1e-4 |
| Node-limited routing | **✗** | **✗** | ✓ M=4 |
| First k dense | 1 | 1 (mini/flash) / 4 (1T) | 3 |
| Attention | **MHA** | GQA | MLA |
| QK-Norm | ✓ | ✓ | ✗ |
| MTP | ✗ | D=1, weight 0.1 | D=1, λ schedule 0.3→0.1 |
| LR scheduler | **WSD** (warmup-stable-**decay**) | **WSM** (warmup-stable-**merge**) | cosine |
| 精度 | BF16（gating FP32） | **FP8** (1T 全 FP8) | FP8 + BF16 critical path |
| 推理框架 | vLLM | — | — |

> **指纹判断**：dots1 → Ling 2.0 的继承是 routing 配方层面（sigmoid + noaux_tc + 2.5 scaling + first_k_dense + QK-Norm 这一整套）。Ling 2.0 在此之上的**新动作**是：(1) zero-mean bias 修正、(2) 把架构推向更稀疏（**dots1 active/total 9.9% → Ling-flash-2.0 5.9%**；**expert-slot fraction 6.2% → 3.5%**）、(3) FP8 全程、(4) WSD → WSM。**所以"ling 2.0 用 dots1 的策略"准确说法是**："Ling 2.0 在 dots1/V3 的 routing 配方上多加了零均值修正，并把架构推向更稀疏 + FP8"。

## 核心方法 / 创新点

### 1. MoE Layer & Routing（§2）

**完全继承 DeepSeekMoE / V3**：

> "we replace the FFN with a Mixture-of-Experts (MoE) layer comprising both shared and isolated experts. Our implementation features 128 routed experts and 2 shared experts activated for all tokens, with each expert implemented as a fine-grained, two-layer FFN utilizing SwiGLU activation. For each token, the router selects the top-6 isolated experts in addition to the 2 shared experts, resulting in 8 active experts per token. Notably, **we employ FP32 precision for the gating layer** computations rather than BF16 to ensure numerical stability and more accurate expert selection during the routing process." (§2.MoE Layer)

→ FP32 gating 是一个常被忽略但很重要的稳定性 trick；本仓库其他论文里只有 OLMoE 明确强调过。

### 2. Load Balancing（§2 Load Balancing）

> "we adopt an auxiliary-loss-free approach (Wang et al., 2024a), **as also employed in DeepSeek-AI et al. 2024**. It introduces a bias term for each expert, which is added to the corresponding affinity scores to determine the top-k routing. This bias term is **dynamically adjusted** during training to maintain a balanced load across experts. **In addition, we also employ a sequence-wise balance loss** to prevent extreme imbalance within any single sequence."

**重要**：
- 论文没给 bias update 公式，明确是引用 V3 的版本，**所以不是 Ling 2.0 的零均值修正**
- 没给 sequence aux loss 的 α 值（V3 是 1e-4）
- "dots1 maintains good load balance throughout training and **does not drop any tokens** during training" → 是 dropless

### 3. Cybertron Framework + 1F1B All-to-all Overlap（§3）

- 训练框架：**Cybertron**（rednote 自研，基于 Megatron-Core）
- **Interleaved 1F1B-based all-to-all overlap**：在 warmup phase 多加一步使 all-to-all 与计算在 steady 1F1B 阶段完全重叠
- vs DeepSeek **DualPipe**：dots1 自评 "**notable advantage in memory consumption, albeit with a marginally higher bubble rate**"
- **Grouped GEMM 自研实现** (Table 1)：vs NVIDIA TE v2.1，forward 平均 +14%，backward 平均 +6.7%（H800）
- 关键 trick：M_i 对齐到 M-multiple 的 fixed block size，让一个 threadblock 内 warpgroup 处理同一个 expert 的 tokens

### 4. WSD Scheduler + 两段 annealing（§4.2）

| 阶段 | Tokens | Batch | LR | 内容 |
|---|---|---|---|---|
| Warmup | 4000 步 | 64M ramp | 0 → 3e-4 | — |
| **Stable** | **10T** | 64M → 96M (6T) → 128M (8.3T) | constant 3e-4 | general data |
| **Annealing 1** | **1T** | 128M | 3e-4 → 3e-5 | reasoning + knowledge **占比 90%** |
| **Annealing 2** | **200B** | 128M | 3e-5 → 1e-5 | code + math + reasoning |
| Long context (32K) | 128B | — | constant | UtK strategy |
| **Total pretrain** | **11.2T** | | | |

**Optimizer**: AdamW (β₁=0.9, β₂=0.95, wd=0.1, grad_clip=1.0), init std 0.006

**关键观察**：annealing stage 1 把 reasoning/knowledge 数据占比拉到 **90%** —— 这是 dots1 在 math/code 接近 Qwen2.5-72B 的关键。

### 5. UtK Long Context（§4.3）

128B tokens 把 context 从 8K 扩到 32K。**UtK (Untie the Knot, Tian 2024b)**：把训练文档**拆 chunk + 打乱 → 让模型重建相关 segments**。不修改数据集本身，只改 sequence packing。

### 6. Pretraining Data（§4.1 + Appendix C）

三阶段：document preparation → rule-based → model-based。
- **Web Clutter Removal Model**：line-level 轻量模型去 boilerplate
- **Category Balancing**：200-class 分类器 balance web data 类别比例
- **完全无合成数据**
- 中英 1:1
- TxT360 SOTA 对照实验：dots1 web data 在 1.5B dense baseline 上**全面超过 TxT360**（Fig. 4）

### 7. Post-training（§5）

**两阶段 SFT**:
- Stage 1: 400K instances 上下采样 + multi-session concat，2 epochs
- Stage 2: math/code domain enhancement 用 **rejection sampling fine-tuning (RFT) + verifier system**
- Cosine LR 5e-6 → 1e-6

**论文不强调 RL**——只在 §3 概述提到 "encapsulated separate trainers for pretraining, SFT, and RL"，但正文没展开 RL 细节。

## 关键消融与结果

### Pretrain Loss & 稳定性（Fig. 3）
- 训练 loss 1.4 区间稳定下降，**全程无不可恢复 spike，无 rollback**
- 6T 时 batch 64M → 96M，8.3T 时 96M → 128M，均平滑

### Base Model 对比（Table 2）

| Domain (avg) | Qwen2.5-32B | Qwen2.5-72B | DSv2 base | DSv3 base | dots1.base |
|---|---|---|---|---|---|
| Chinese | 89.5 | 90.3 | 86.0 | 89.5 | **91.3** |
| English | 73.2 | 76.3 | 71.8 | **78.0** | 75.7 |
| MATH | 79.9 | 77.3 | 64.8 | **82.1** | 78.3 |
| Code | 56.9 | 59.0 | 50.9 | **62.5** | 59.6 |

→ **中文最强**（数据 pipeline 优势），**英文/math/code 落后 V3**，**整体接近 Qwen2.5-72B**。注意：dots1 base 用 **14B active**，Qwen2.5-72B 是 **72B dense** —— **5× efficiency leverage**。

### Long Context（Table 3, RULER）

| Context | Qwen2.5-72B | dots1 |
|---|---|---|
| 4K | 96.5 | 94.7 |
| 8K | 94.3 | **94.9** |
| 16K | 93.1 | 92.6 |
| 32K | **92.7** | 87.7 |

→ 8K 持平、32K 输 5 分；UtK 在 32K 边界还是不够强。

### Instruct Model 对比（Table 5）
- MMLU: dots1.inst 82.1（vs DSv3 87.9, gpt4o 86.7） — **明显落后**
- AIME24: dots1.inst 33.1（vs DSv3 34.0） — **打平 V3**！
- MATH500: 84.8（vs DSv3 88.9）
- C-Eval: **92.2**（**超过 DSv3 86.3 和 gpt4o 77.8**）
- IFEval: 82.1（vs DSv3 86.1, gpt4o 85.2） — 落后

→ 整体 instruct 在中文/math 强、英文/code/alignment 弱。

### 训练成本（Table 4）

| Model | GPU-hours / 1T tokens | Pretrain tokens | Total GPU-hours |
|---|---|---|---|
| Qwen2.5-72B | 340K (×1.0) | 18.0T | 6,120K (×1.0) |
| **dots1** | **130K (×0.38)** | **11.2T** | **1,456K (×0.24)** |

→ 整体 **¼ Qwen2.5-72B 成本**，关键是 14B active 的 MoE 推理友好性。

### MoE 专家专门化（§6, Fig. 5）
- 专家激活在 specific domains（如 DM Mathematics）显著超 random selection
- Wikipedia 这类"知识图谱"型数据上专家更均衡
- 跨层观察到 stronger expert specialization

## 对 16B MoE 设计的启示

1. **Routing 配方信心增强**：sigmoid + noaux_tc + scaling 2.5 + 1 层 dense 前缀这套，dots1 在 11.2T 上无 spike 跑通；本仓库 22_FINAL_16B_design Profile B 已采用，**dots1 + Ling 2.0 + V3 三重独立验证**。

2. **不一定要 zero-mean bias 修正**：dots1 用 V3 原版 sign(e_i) 在 11.2T 上没有 bias drift 问题。**16T tokens 区间内零均值修正可能不是 must-have**——可以放进 wind tunnel A2 而不是默认开启。Ling 论文也没给"零均值 vs 不修正"的直接消融。

3. **MHA on 16B 可行但略贵**：dots1 用 MHA 是因为 142B 量级 KV cache 不是单卡瓶颈、做 32K 也够。**16B 用 GQA 16Q-4KV 仍是更优 ROI**（22_FINAL_16B_design §4.5），dots1 不构成 MHA 翻案证据。

4. **FP32 gating** 是 dots1 单独强调的细节：默认 gating 跑 FP32 而非 BF16/FP8。本仓库其他论文里只有 OLMoE 明确强调过类似（router z-loss + careful precision）。**16B 设计应明确写出 "router/gating 始终 FP32"**——本仓库 22_FINAL_16B_design Spec 表已写入此条。

5. **WSD vs WSM 选择**：dots1 用经典 WSD（4000 步 warmup + 10T constant + 1T+200B 双段 annealing）也能稳。**WSM 不是绝对优于 WSD**，是 +1~2 分的差。22_FINAL_16B_design 推荐 WSM 但 WSD 仍是 fallback；dots1 给 WSD 提供完整 schedule 模板。

6. **annealing reasoning 占比 90%** 是激进但有效的 recipe。22_FINAL_16B_design §5 Reasoning ramp 阶段可以借鉴这个数字（目前是 30-46%）。

7. **不用 NLR 是合理选择**：dots1 142B + 128 expert + EP 任意，**完全没用 NLR**。这强烈支持本仓库 25_node_limited_routing 备忘的判断：**NLR 是 V3 特定 8×8 拓扑下的工程优化，不是普适必需项**。

8. **Cybertron + 1F1B overlap 是 DualPipe 的轻量替代**：声称 vs DualPipe 内存友好但 bubble 略高。如果团队 PP ≤ 8、不想做 DualPipe 工程，这是公开的可参考方案。

## Caveats / 局限

- **Bias update 公式没在论文里写出**，只说"as in DeepSeek-AI 2024"。但论文没声明任何修改，**默认它就是 V3 原版 sign(e_i)**。
- **Sequence-wise balance loss 的 α 值未给**。可能就是 V3 的 1e-4，但论文没写。
- **没有 wind tunnel / scaling law**：dots1 直接训了一次 142B/11.2T，**没有 Ling 2.0 那种小规模 anchor 的 scaling 验证**。这是 dots1 整体效果不如 Ling-mini-2.0 在 dense baseline 比较中亮眼的原因之一。
- **Dropless** 但论文没给 capacity factor 配置（Megablocks 风格还是 ScatterMoE 风格）。
- **没有 RL 细节**：§3 提到有 RL trainer 但 §5 完全不展开，只描述 SFT + RFT。吴建民提到的"1.0 用 MHA RL 效率低"对得上——MHA 在 RL inference 阶段 KV cache 比 GQA 大很多，rollout 慢。
- **Long context 32K 上 RULER 落后 Qwen2.5-72B 5 分**（Table 3），UtK 在边界还是不够。
- **GPU hours 是 rednote 自己的优化框架估计**——Qwen2.5-72B 的 340K/T 也是 rednote 在自己 framework 上的 reproduce，**不是 Qwen 团队官方数字**。
- **完全没用合成数据**是论文卖点，但同时也是与 K2 / Ling / Qwen3 reasoning 数据策略的最大分歧。dots1 在 reasoning 上落后 V3/Qwen3 与此可能有关。
- **没开源 RL 数据 / SFT 数据**，只开源了权重和 intermediate ckpts (every 1T tokens)。

## 与本仓库的交叉引用

- **08_ling_2.md**：Ling 2.0 routing 配方继承自这里；零均值修正是 Ling 在此基础上的增量
- **04_deepseek_v3.md**：dots1 显式声称继承 V3 的 ALF；但**没有继承 NLR 和 MTP**
- **03_auxloss_free.md**：dots1 引用此论文 (Wang 2024a) 作为 ALF 来源
- **22_FINAL_16B_design.md**：本论文是 §4.6（路由）和 §4.8（稳定性套件 / FP32 gating）的额外独立证据
- **25_node_limited_routing.md**：dots1 = "不用 NLR 也能跑通 11.2T MoE 训练"的关键证据
