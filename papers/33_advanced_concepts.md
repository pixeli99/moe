# 老师提到的概念 — 系统梳理

> **背景**：老师给的 9 个点（宽高 vs 瘦高 / FFN:Attn 2:1 / head 维度 / hidden 底层考虑 / 翻倍 2 点 / sparse attn / linear attn / KV cache / 宽度 upscale / DSA / 压缩），每一条都不是孤立的，背后有完整的因果链。本文按"对你 16B Profile B 的影响"排序，必懂的在前。
> **怎么读**：每节末尾有"一句话"，是你应该形成的 mental model。如果一句话能复述出来，那这个点你就过关。

---

## 1. 宽高 vs 瘦高（depth vs width）

**老师在重复问这件事**，因为这是你 16B 设计的 #1 开放问题。

完整答案在 `32_depth_width_tradeoff.md` + `depth_width.html`（交互页）。简版：

- 短宽（Option B 20L/2304）：latency 减 26%，upscale headroom 2.0×，**reasoning 风险 1-2pt**
- 中庸（Baseline 27L/2048）：稳妥
- 深窄（GLM-4.5 路线 92L/5120）：reasoning 上限高但难 upscale

**A2 T2.6 验证后再切主 spec**。Ling-mini-2.0 已经在 16B 量级上验证过 20L，是 Option B 的直接 anchor。

→ **一句话**：短宽对 inference + upscale 友好，深窄对 reasoning 友好，需要 wind tunnel 验证后决定。

---

## 2. FFN : Self-Attn 参数比 2:1 — 这条经典规律的真伪

这是一个**经典 dense transformer 的设计原则**，但在 MoE 下完全不成立。先把数学推清楚。

### 2.1 经典 dense MHA + 8/3 FFN（Llama 1 / 2 / Chinchilla 风格）

单层参数账：

| 模块 | 公式 | 数量（H = hidden） |
|---|---|---|
| 自注意力 MHA Q+K+V+O | 4 × H × H | **4 H²** |
| FFN SwiGLU (intermediate = 8H/3) | 3 × H × (8H/3) | **8 H²** |
| **比值 FFN : Attn** | | **2 : 1** ✓ |

为什么 SwiGLU 用 8H/3 不是 4H？因为 SwiGLU 有 3 个矩阵（gate, up, down），相比 GELU 的 2 个矩阵（up, down），保持总参数一致就要把 intermediate 缩到 2/3。GELU intermediate=4H，对应 SwiGLU intermediate=8H/3 ≈ 2.67H。

这就是 **2:1 的来源**：经典 GELU/MHA dense LLM 的"教科书"比例。

### 2.2 现代 GQA + 加宽 FFN（Llama 3 路线）

Llama 3 70B 用 GQA + intermediate = 3.5H：

| 模块 | 公式 | 数量 |
|---|---|---|
| GQA 8Q/1KV: Q+O = 2H², K+V = 2 × H × (H/8) = H²/4 | 2H² + H²/4 | **2.25 H²** |
| FFN SwiGLU (intermediate = 3.5H) | 3 × H × 3.5H | **10.5 H²** |
| **比值 FFN : Attn** | | **~4.7 : 1** |

GQA 把 attention 缩了 ~45%；intermediate 加宽到 3.5H 把 FFN 加了 30%。**经典 2:1 已经偏离到 ~5:1**。

### 2.3 你的 MoE 16B Profile B 单层（不严格 2:1）

Profile B：hidden=2048, GQA 16Q/4KV, N_routed=64, K=8, N_shared=1, d_expert=1408。

单 MoE 层参数账：

| 模块 | 数量 (M) | 占比 |
|---|---|---|
| GQA attn | 10.5 | 1.7% |
| Shared expert FFN | 8.65 | 1.4% |
| Routed experts FFN (全 64 个) | 554 | 89.5% |
| **Total / 层** | **573** | |
| Active K=8 routed FFN | 69.2 | 12.1% (of total) |
| Active 全部 (attn + shared + K=8) | 88 | – |

**比值 (per layer, total)**：FFN (shared + all routed) : Attn = (554 + 8.65) / 10.5 = **53.6 : 1**

**比值 (per layer, active)**：FFN : Attn = (69.2 + 8.65) / 10.5 = **7.4 : 1**

→ **MoE 下经典 2:1 完全不适用**。MoE 把 FFN 容量"吹大"了几十倍但只激活其中一小部分。

### 2.4 为什么老师提这个

老师可能在测两件事：

1. **你知不知道这个经典比例**——证明你读了 dense LLM 基础
2. **你会不会无脑把 2:1 移植到 MoE**——MoE 的"等价 active"才该和 attn 比，且仍偏向 FFN

**真正应该关心的问题**（这才是 senior engineer 视角）：

- **Active FFN : Active Attn ≈ 7-10 : 1**（你 Profile B 是 7.4:1）—— attention 在 active FLOPs 里占比 12%，**长 context 时这个比例会变**（attention 加 seq 项）
- **Total FFN : Total Attn ≈ 50-100 : 1**（你 Profile B 是 53.6:1）—— MoE 的总参绝大头在 FFN expert，attn 反而像"配角"
- **如果你切到 Option B（20L/2304）**：hidden 加宽 ↑ → attention FLOPs 上升（attn ∝ H²）；同时 d_expert 也加到 1792 → FFN 也上升。新比值仍在 7:1 附近。**短宽不改变 FFN:Attn 比值的"量级"，只是绝对值都涨**。

### 2.5 极端反例：当 attn 占比变高

- **Long context**：attention 加 4×H×seq 项；32K seq 时 attention FLOPs ≈ 4×H²+4×H×32K = 与 H² 同级。**比值改变**
- **MLA**：DeepSeek-V3 的 MLA 把 attention 参数缩到只有 ~2H²，FFN expert 总参 100×。比值更偏 FFN
- **MFA (Step-3)**：multi-matrix attention 反向，把 attention 加宽

→ **一句话**：经典 dense LLM FFN:Attn ≈ 2:1（MHA + 8H/3 SwiGLU）；现代 GQA + 加宽 FFN 偏到 5:1；MoE active 是 7:1，total 是 50:1。**MoE 设计不要被 2:1 锁住，要看 active 路径上的比值**。

---

## 3. Head dim & Hidden size — infra 对齐要求

老师说"hidden size 考虑底层 / infra 相关 head 维度" —— 这是讲**硬件友好的数值选择**。不是凭空挑，是有约束的。

### 3.1 Head dim 的硬件约束

**FlashAttention（FA2 / FA3）只优化了少数 head_dim**：

| head_dim | FA2 优化？ | FA3 优化？ | 出处 |
|---|---|---|---|
| 32, 48, 56 | 部分 | – | 老模型 |
| **64** | ✓ | ✓ | gpt-oss, GPT-2 |
| 80 | △ | △ | Hunyuan-Large（非标准） |
| **128** | ✓✓ 最佳 | ✓✓ 最佳 | Llama, Qwen, DeepSeek 主流 |
| 192 | ✓ | ✓ | Qwen3-Coder |
| **256** | ✓ | ✓ | Qwen3-Next, MFA Step-3 |

**为什么 128 是甜区**：

1. **GPU 寄存器 / shared memory 友好**：128 × 4 (bytes BF16) = 512 字节，对齐 1 个 wavefront
2. **WMMA / MMA 指令原生支持**：CUDA 的 tensor core 在 head_dim = 16/32/64/128 上吃 native MMA
3. **质量上不输 64**：每 head 表达力够丰富
4. **行业默认**：Llama 全系、Qwen 全系、DeepSeek 全系都 128，kernel 复用度高

**何时选别的 head_dim**：

- 64：gpt-oss 选了，配合 hidden=2880 = 45×64 → num_heads=45。理由可能是更小 head 让 attention 更"密集"
- 256：Qwen3-Next 用，配合 hidden=2048, num_heads=8。head 大但少 → 适合超稀疏 MoE（节省 attn 算力）
- 80：Hunyuan-Large 非标准，复现成本高

→ **你的 16B Profile B 用 head_dim=128** 是行业默认，不需要消融。

### 3.2 Hidden size 的硬件约束

**Hidden size 不能瞎选**，要满足：

1. **必须是 128 的倍数**（BF16 tensor core 对齐）；最好是 256 的倍数（FP8 tensor core）
2. **必须能被 num_q_heads × head_dim 整除**：hidden = Q heads × head_dim
3. **必须能被 num_kv_heads × head_dim 整除**：hidden = KV heads × head_dim × (Q/KV ratio)
4. **必须能被 TP shard 数整除**：如果 TP=2，hidden / 2 仍需 ≥ 128 对齐
5. **必须能被 EP shard 数整除（对 FFN 而言）**：N_routed / EP 应是整数

**合法值（head_dim=128 时）**：1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096 ...

**不合法 / 不友好**：1500, 1600, 1700, 2100, 2200, 2400, 2500 — 都不是 128 的倍数

**Profile B 候选**：
- 2048 = 16 × 128 ✓（baseline）
- 2304 = 18 × 128 ✓（Option B；恰好够 18 个 Q heads）
- 2560 = 20 × 128 ✓（Option C）
- 3072 = 24 × 128 ✓（更宽）

**Q heads 选择**：
- 2048 / 128 = 16 → GQA 16Q/4KV（baseline）
- 2304 / 128 = 18 → GQA 18Q/6KV（Option B，6 是 18/3 的自然分组）
- 2560 / 128 = 20 → GQA 20Q/5KV（Option C，5 是 20/4 的自然分组）

### 3.3 FFN intermediate 的对齐

SwiGLU FFN intermediate 也要对齐：

- 一般是 hidden × 4 ~ 5.4，向上取整到 64 / 128 的倍数
- Llama 3 70B: hidden=8192, intermediate=28672 = 3.5×（取整到 448 的倍数）
- Llama 2 7B: hidden=4096, intermediate=11008 = 2.69×（≈ 8/3）
- 你的 16B baseline: 第 0 层 dense intermediate 取 10944 = hidden × 5.34 ✓ 64 倍数

**d_expert（expert FFN intermediate）也要对齐**：

- baseline: 1408 = hidden 2048 × 0.687，11 × 128 ✓
- Option B: 1792 = hidden 2304 × 0.778，14 × 128 ✓
- Option C: 2048 = hidden 2560 × 0.8，16 × 128 ✓

### 3.4 为什么这些都重要

不对齐会发生什么：

1. **CUDA kernel 性能跌 30-60%**：tensor core 无法发挥
2. **TP 时形状不能均分**：要 padding，浪费显存 + 算力
3. **EP 时 expert 数 / EP 不整 → 负载不均
4. **FlashAttention 部分 path 走 fallback kernel**（慢 10×）

**老师的意思**：选 spec 时**先考虑硬件能跑得高效**，再考虑参数最优。否则纸面 spec 漂亮，实际 MFU 跌一半。

→ **一句话**：head_dim = 128 是行业默认硬件甜区；hidden 必须是 128 的倍数 + 能被 head_dim × Q_heads 整除 + TP/EP 友好。你的 Option B 2304 = 18×128 是合法的。

---

## 4. 每翻一倍 +2 个点 — Chinchilla scaling rule of thumb

这是个**经验法则**，不是严格定律。源自 Chinchilla scaling law (DeepMind 2022) + 后续观察。

### 4.1 大致规律

| 参数翻倍 | downstream 提升（粗略） |
|---|---|
| 7B → 14B (active 翻倍) | MMLU +5-10 pt |
| 14B → 28B | +3-5 pt |
| 28B → 56B | +2-3 pt |
| 56B → 112B | +1-2 pt |
| 112B → 250B | +1 pt |

**关键观察**：边际收益递减。每次翻倍的"+2 个点"是高端模型的经验值；中小模型一次翻倍能 +5-10 pt。

### 4.2 IsoFLOPs（同算力下不同分配）

Chinchilla 的真正贡献：**给定 FLOPs，参数和 tokens 应当 1:20 比例**。

- 7B 模型 → 应训 140B tokens
- 70B 模型 → 应训 1.4T tokens
- 700B 模型 → 应训 14T tokens

如果违反这个比例，效率下降。

**Over-training**：训 > Chinchilla 推荐 tokens 仍有提升，但每翻倍 tokens 只 +1 pt 左右（远不如翻倍参数）。

### 4.3 reasoning vs knowledge 的差异

Yokota 2025 观察：
- Knowledge tasks (MMLU)：参数翻倍 +2 pt
- Reasoning tasks (GPQA, MATH)：**翻倍 active params** > 翻倍 total，约 +3-4 pt
- Long-tail memo：翻倍 total > 翻倍 active

→ **MoE 的 active vs total 在 reasoning vs memo 上效果不同**。这是你 Profile R/B/M 三档存在的根本原因。

### 4.4 老师说这个的意思

可能是：
1. **设 baseline 期望**：你 16B → 32B upscale 后预期 +2-4 pt MMLU
2. **检查 spec 是否在 sweet spot**：如果某个超参选错，可能损失"一个翻倍"的收益（5-10 pt）
3. **理解为什么 over-train**：边际 +1 pt 看起来小，但累积 5 个翻倍就是 +5 pt，这是 V3 / K2 / Ling 都 over-train 20T+ 的理由

→ **一句话**：经验法则——翻倍参数（或 active）+2 pt downstream，边际递减；reasoning 偏 active 翻倍，knowledge 偏 total 翻倍。

---

## 5. Sparse Attention vs Linear Attention — 不是一回事

老师把它们并列提，但**这两个是不同范畴**。常被混淆。

### 5.1 Linear Attention（线性注意力）

**机制**：把 softmax(QK^T) 换成 φ(Q) × φ(K)^T × V 形式（核函数近似），让计算从 O(N²) 变成 O(N)。

**特点**：
- 内存 / 计算从 O(N²) 降到 O(N)
- 失去 softmax 的"选择性聚焦"
- 长序列友好，**retrieval / 精确召回弱**

**代表**：
- Mamba / SSM (Gu, Dao 2023)
- Lightning Attention (MiniMax)
- DeltaNet (Yang et al.)
- RWKV

**用法**：通常**与 softmax attention 混合**，避免单独使用导致 retrieval 失败。Jamba 7:1, Qwen3-Next 3:1, MiniMax 7:1。

### 5.2 Sparse Attention（稀疏注意力）

**机制**：仍是 softmax，但**只对部分位置算 attention**。剩下的位置假设无关。

**特点**：
- 内存 / 计算从 O(N²) 降到 O(N × k)（k = 看的位置数）
- 保留 softmax 的选择性
- 关键：**怎么选哪些位置看**（这是各方法的差异）

**代表**：
- **Sliding window**（gpt-oss, Mistral）：每 token 只看前 W=2048 个邻居
- **Block-sparse**（Longformer, BigBird）：固定 block + global tokens
- **Dilated**（Longformer）：跳步采样
- **DSA (DeepSeek Sparse Attention)** — 见 §8
- **Top-K** (MoBA, NSA)：动态选 top-K 相关 token

### 5.3 区别表

| 维度 | Linear Attention | Sparse Attention |
|---|---|---|
| 数学形式 | 替换 softmax | 保留 softmax 但只算一部分 |
| 复杂度 | O(N) | O(N × k) |
| Retrieval 精度 | **差** | **好**（保 softmax） |
| 训练稳定性 | 略难 | 与 dense softmax 相似 |
| 是否能完全替代 softmax | 不能（需混合） | 视方法而定（sliding 不能，DSA 可以） |
| 与 MoE 关系 | 正交 | 正交 |

### 5.4 与 MoE 的关系

**两者都与 MoE 正交**——MoE 决定 FFN 怎么算，attention 怎么改不直接影响 MoE 路由。

但混合架构可以叠加：
- MiniMax-01：Lightning Attn (linear) + softmax 每 8 层 + MoE FFN
- Qwen3-Next：DeltaNet (linear) + softmax 每 4 层 + MoE FFN
- DeepSeek-V3.2-Exp：DSA (sparse) + MLA + MoE FFN

### 5.5 你的 16B 用不用？

**不用**。理由：
1. **目标 context 128K**，softmax 仍撑得住（KV cache ~1GB）
2. **工程复杂度高**：hybrid attention kernel 不成熟（除非走 Jamba / MiniMax 的成熟方案）
3. **Linear / sparse 主要为 1M+ context 服务**，与你的 16B 不匹配

→ **一句话**：Linear 用核函数把 O(N²) 降到 O(N)，retrieval 差；Sparse 保留 softmax 但只算选定位置，retrieval 好。两个不是一回事，都与 MoE 正交。

---

## 6. KV cache & 压缩（MLA 的核心机制）

老师说"压缩的部分" —— 我理解是讲 **MLA 怎么压缩 KV cache** 的具体机制。

### 6.1 标准 GQA 的 KV cache

每生成 token 需要前面所有 K, V。每层每 KV head 存一份：

```
KV cache size = 2 × L_layers × num_kv_heads × head_dim × seq_len × precision
```

例：你的 Profile B (27 层, 4 KV heads, 128 head_dim, 32K seq, BF16):
2 × 27 × 4 × 128 × 32768 × 2 bytes = **906 MB**

128K seq 时：~3.6 GB。

### 6.2 MLA 的核心想法 — 低秩压缩

DeepSeek V2 (2024-05) 发现：K 和 V **存在低秩结构**。不存原始 K, V，存它们的低维 latent。

```
传统：cache 中存 K_i ∈ R^(num_kv × head_dim) 和 V_i ∈ R^(num_kv × head_dim)
MLA：只存 c_kv_i ∈ R^d_c，d_c = 512（远小于 num_kv × head_dim ≈ 1024-4096）
      推理时再从 c_kv_i 解压出 K_i, V_i
```

**详细公式（V2 §3.2）**：

1. **压缩**（每个 token 进来时）：
   ```
   c_kv = W_DKV @ x           # x ∈ R^H, c_kv ∈ R^d_c
   ```
2. **缓存**：只存 c_kv（每 token 一个 d_c=512 向量）
3. **解压**（attention 计算时）：
   ```
   K = W_UK @ c_kv            # 展开回 K
   V = W_UV @ c_kv            # 展开回 V
   ```

**关键 trick**：`W_UK @ c_kv = K`，而 attention 算 `Q @ K^T`，可以提前合并 `Q @ W_UK^T`，让 Q 在 d_c 维空间和 c_kv 算点积。**等价于在低维空间做 attention**。

### 6.3 节省多少

| 配置 | 每 token cache 大小（BF16）|
|---|---|
| Llama 2 70B (MHA) | 2 × 64 × 128 × 2 = 32 KB |
| Llama 3 70B (GQA 8:1) | 2 × 8 × 128 × 2 = 4 KB |
| V3 (MLA, d_c=512) | 1 × 512 × 2 = 1 KB |

**MLA 比 GQA 还小 4×**，比 MHA 小 32×。这就是 V3 能 128K context 单卡推理的关键。

### 6.4 但 MLA 有 RoPE 兼容问题

RoPE 把位置信息打到 K, V 的某些维度上。如果 K 都从低维 latent 解压出来，怎么应用 RoPE？

DeepSeek 的解法：**分离 K**
- K 一部分是 RoPE-applied K_rope（直接存）
- 一部分是 from-latent K_nope（从 c_kv 解压）
- attention 时把两部分拼起来

这个设计让 MLA + RoPE 兼容，但实现复杂。**这是 MLA 难复现的根本原因**——非 DeepSeek 团队 kernel 适配难度高。

### 6.5 其他压缩方法

KV cache 减小的其他思路：

| 方法 | 机制 | 节省 |
|---|---|---|
| GQA | 多 Q heads 共享 K/V heads | 4-8× |
| MQA | 1 个 KV head | 32-64× |
| **MLA** | 低秩 latent | 100× |
| Token eviction (StreamingLLM) | 丢老 token | 视策略 |
| H2O | 选关键 token 保留 | 视策略 |
| **量化 (KIVI, AWQ-KV)** | INT4/INT8 KV | 4× |
| Sparse Attention (DSA) | 只算 top-K | 10×+ |

**你的 16B Profile B**：用 GQA 16Q/4KV，cache ~1GB @ 32K，**不需要 MLA**。

→ **一句话**：MLA 把 K, V 投影到 d_c=512 维 latent 存进 cache，比 GQA 缩 100×。代价是 RoPE 兼容需要"分离 K"的复杂设计，非 DeepSeek 团队复现门槛高。

---

## 7. Width Upscaling — 为什么很少有人这么做

老师说"宽度的 upscale" —— 这是 model 上扩的第 3 条路径（很少用）。

### 7.1 Net2Net（Chen et al. 2015, ICLR）

最早系统化讲 width upscaling：**function-preserving expansion**。

**核心想法**：给现有 layer 加新 neuron / channel，初始化时让网络 output 不变（恒等）。

具体做法（FFN 加宽）：
1. 原 FFN：W1 ∈ R^(H × d), W2 ∈ R^(d × H)
2. 加宽到 d' > d：
   - W1' = [W1, W1_copy]，复制某些列（带小噪声）
   - W2' = [W2; W2_scaled]，相应缩放新增行使总和不变
3. Output 在第 0 步与原模型相同，随后训练分化

### 7.2 为什么 LLM 上很少用

1. **Embedding / LM head 也要扩宽**：vocab × H → vocab × H'，新增维度难初始化
2. **Attention 需要重新设计 head**：head_dim 改变 vs heads 数改变都难
3. **比 depth / expert upscale 收益小**：宽度增加导致 attention FLOPs ↑↑（hidden²），FFN ↑↑，但 quality 提升不显著
4. **modern 替代品**：直接 from-scratch 训一个更宽的，配合精炼 distill；或加 expert（MoE 路径）

### 7.3 真实案例（罕见）

- **Mixtral 8x7B → 8x22B**：宽度 + 深度 + expert 同时扩，不算纯 width upscale
- **某些 stage-wise growth 方法**（论文级，未广泛工业用）
- **Net2Net 在 vision 还有零星使用**（GAN 训练）

### 7.4 你的 16B 用不用 width upscale

**不用**。Stage 2 / Stage 3 走 depth + expert，不走 width。理由：
- 你的 16B → 50B 路径里 hidden 应该固定（节省 kernel 重适配）
- Width 改变需要重新调 RoPE base、调 head 数、调 FFN intermediate
- 工程成本太高，收益不明显

→ **一句话**：Width upscaling = 给 layer 加新 neuron，Net2Net 是经典方法；LLM 几乎不用，因为 embedding 重 init、attention 重设计成本高，不如 depth / expert upscale 划算。

---

## 8. DSA 的复现 — DeepSeek Sparse Attention（V3.2-Exp）

老师说"DSA 的复现" —— DeepSeek 最新的 sparse attention 设计。值得专门讲。

### 8.1 DSA 是什么

DeepSeek-V3.2-Exp（2025-09 发布）在 V3.1 基础上把 MLA 升级成 **Sparse Attention**。

**机制**（论文 V3.2 §2-3，简化）：

1. **Lightning Indexer**（轻量 indexer 模块）—— 一个小的 attention 模块，先用 ~1/10 算力扫一遍所有 token，给每个 query 找出 top-K 个最相关的 key（K=256 左右）
2. **Selected Attention**—— 然后正经的 MLA attention 只对这 K 个 key 计算
3. **缓存结构**——仍是 MLA 的 c_kv latent；indexer 用独立投影

数学上：
```
relevance_scores = lightning_indexer(Q, K_all)  # 快速近似
top_K_indices = argmax_K(relevance_scores)
output = softmax(Q @ K_selected^T / √d) @ V_selected  # 只算 K=256 个 token
```

### 8.2 比 V3 (MLA) 强在哪

| 维度 | V3 MLA | V3.2 DSA |
|---|---|---|
| KV cache | 100× 缩 (vs MHA) | 同 V3 |
| 长 context 计算 | O(N²) softmax | **O(N × K) 选 top-K** |
| 1M context FLOPs | 不可行 | **可行** |
| Retrieval 精度 | 完整 softmax | 略损（依赖 indexer） |

DSA 把 **attention 计算量** 从 N² 降到 N×K，K=256 远小于 N=128K，省 500×。

### 8.3 开源 / 复现状况

**好消息**：DeepSeek 已开源 V3.2-Exp 权重（HF: deepseek-ai/DeepSeek-V3.2-Exp）

**难点**：
1. **Lightning Indexer 的训练**：与主 attention 联合训练，loss balancing 不明
2. **Indexer kernel**：非标准操作，TileLang 内部 kernel
3. **MLA 已经难复现**，DSA 在 MLA 上叠加，门槛更高

**目前公开复现**：几乎没有。社区在等 DeepSeek 自己发布 kernel 或更详细的训练 recipe。

### 8.4 你的 16B 用不用 DSA

**不用**。理由：
- 16B 目标 128K context，softmax 已够
- DSA 主要为 1M+ context 服务（V3.2 用例）
- 你团队没有 MLA 基础，直接上 DSA 是技术债爆炸

**未来上扩到 100B+ 时再考虑** — DSA 是 long-context 上的关键技术，但放在 Stage 4 以后。

→ **一句话**：DSA = DeepSeek V3.2 的 sparse attention（lightning indexer + selected attention），把 attention 算量从 N² 降到 N×K，专为 1M+ context；权重开源但 kernel 没开源，社区复现还在起步。

---

## 9. 一句话总结清单

老师 9 个点，每个一句话：

| # | 点 | 一句话 |
|---|---|---|
| 1 | 宽高 vs 瘦高 | 同 total/active 下短宽利 inference + upscale，深窄利 reasoning；A2 T2.6 验证 |
| 2 | FFN:Attn = 2:1 | 经典 dense（MHA + 8H/3）的比例；GQA/现代偏到 5:1；MoE active 7:1, total 50:1 |
| 3 | Head dim | 128 是硬件甜区（FlashAttention 最优），64 / 256 是少数特例 |
| 4 | Hidden 底层 | 必须 128 倍数 + 能被 head_dim × num_heads 整除 + TP/EP 友好 |
| 5 | 翻倍 +2 pt | Chinchilla 经验法则，边际递减；reasoning 偏 active 翻倍 |
| 6 | Sparse vs Linear | Linear 替换 softmax → O(N) retrieval 差；Sparse 保留 softmax 但选位置算 → O(N×K) retrieval 好 |
| 7 | KV cache | GQA 缩 4-8×，MLA 缩 100×（低秩 latent），DSA 进一步对长 context 缩 |
| 8 | 宽度 upscale | Net2Net function-preserving 加 neuron；LLM 上几乎不用，被 depth/expert upscale 取代 |
| 9 | DSA 复现 | DeepSeek V3.2 sparse attention，1M+ context 关键；weights 开源但 kernel 未开源，复现门槛高 |

---

## 10. 你 16B Profile B 的决策清单（哪些直接影响）

| 决策 | 当前 spec | 是否受这 9 点影响 |
|---|---|---|
| Hidden 2048 vs 2304 vs 2560 | 待 T2.6 | **直接相关**：1, 3 |
| Head_dim 128 | ✓ 已定 | **直接相关**：3（默认值，不消融） |
| GQA 16Q/4KV | ✓ 已定 | **直接相关**：3, 6 |
| Layers 27 vs 20 vs 16 | 待 T2.6 | **直接相关**：1 |
| MLA 不用 | ✓ 已定 | **直接相关**：6, 7 |
| Hybrid attention 不用 | ✓ 已定 | **直接相关**：5, 7 |
| DSA 不用 | ✓ 已定 | **直接相关**：8 |
| Width upscale 不计划 | ✓ 已定 | **直接相关**：7 |
| FFN:Attn 比值 | 隐含 ~7:1 active | **间接相关**：2（不需消融，但要 aware） |
| 训练 tokens 14-16T | ✓ 已定 | **直接相关**：4（over-train 决策） |

**结论**：9 个点中 8 个已经在 22_FINAL spec 里有对应决策；剩下的 1 个（hidden 选择）正在 wind tunnel A2 T2.6 验证。**老师的 9 个点其实在测你"知不知道每个决策是为什么"**。

---

## 11. 老师可能下一步会问的

如果他追问，可能是：

1. **"你的 attn FLOPs / token 怎么算？"** → 4 × H² + 4 × H × seq_len；Profile B 在 4K seq 下每层 ~17M FLOPs
2. **"GQA Q:KV 为什么是 4:1 不是 8:1？"** → 4:1 是 Llama 主流；8:1 (Mixtral) 牺牲略多 quality；Profile B 16Q/4KV = 4:1
3. **"如果用 MLA 你的 16B 能省多少？"** → KV cache 906MB → ~50MB（18×）；但 kernel 复杂度涨 10×，ROI 在 16B 不划算
4. **"为什么不用 MoBA / NSA 这些新 sparse attn？"** → 太新（2025-Q4）；产品级未验证；A2 不消融
5. **"Width upscale 真不能做吗？"** → 能但收益不值；depth + expert 已经够
6. **"V3.2 DSA 你打算引入吗？"** → 长 context 上扩到 100B 之后再考虑

---

## 12. 与其他笔记的交叉

- 宽高 vs 瘦高：`32_depth_width_tradeoff.md`（深度版）+ `depth_width.html`（交互版）
- MLA：`02_deepseek_v2.md` §3
- Sparse Upcycling：`19_sparse_upcycling.md`
- 入门概念：`31_foundations.md` §7（attention 字母汤）
- 全市场对比：`28_open_source_moe_catalog.md`（depth/width 分布）
- 交互概念学习：`concepts.html`（新加 5 个 Upscale tier 概念）

## 13. 参考资料

- Chinchilla scaling: Hoffmann et al. 2022, arXiv 2203.15556
- FlashAttention: Dao 2022/2023, arXiv 2205.14135 / 2307.08691
- MLA (DeepSeek-V2): arXiv 2405.04434 §3.2
- DSA (DeepSeek-V3.2): https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Net2Net: Chen et al. 2015, arXiv 1511.05641
- Mamba: Gu, Dao 2023, arXiv 2312.00752
- Sliding window: Mistral 7B paper / gpt-oss model card
- 你的 22_FINAL_16B_design.md（主 spec）
