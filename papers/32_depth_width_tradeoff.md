# 16B MoE 短宽设计 — Depth/Width Tradeoff 与 Upscale 友好性

> **问题**：22_FINAL Profile B 当前是 27 层 / hidden 2048。如果想做成"矮胖"（fewer layers, wider hidden）以便后续 upscale，该怎么设计？风险在哪？怎么落实？
> **结论先讲**：推荐 **Option B（20L / hidden 2304 / d_expert 1792 / 64×8+1）**——比 baseline 减 26% 深度，加宽 12% hidden 补回参数 + 给后续 SOLAR 式 depth upscale 留出 1.6× 翻倍空间，attention FLOPs 增加可控（+27%）。Option C（16L / 2560）是激进版，留作 Profile R 极端变种。

---

## 1. 为什么"矮胖"对 upscale 友好

### 三条 upscale 路径

MoE 模型上扩到下一档规模（比如 16B → 32B → 64B）有三条独立的路径：

```
                  ┌─ Depth Upscaling (加层) ─────┐
                  │   SOLAR / LLaMA Pro / LESA  │
                  │                              │
   16B base ──────┼─ Expert Upcycling (加 N) ───┼─→ 32B / 64B / 100B+
                  │   Sparse Upcycling (Komatsuzaki 2023)
                  │                              │
                  └─ Width Upscaling (加 hidden) ┘
                      （罕见、难做）
```

### 短宽 起点的优势

1. **Depth upscale headroom 大**：从 16L 翻倍到 32L 还在"合理深度"区间；从 27L 翻倍到 54L 接近 GLM-4.5 92L 极端
2. **Inference latency 直接收益**：每生成 1 token 要顺序经过所有层，层数减少 = latency 减少（不能并行的部分）
3. **Pipeline bubble 减少**：PP 并行下 bubble 时间 ∝ 层数，浅模型 bubble 占比小
4. **梯度流更顺**："Depth Delusion" 2026 实证浅 + 宽在 7B 上反而比深 + 窄 quality 好
5. **Mixtral-style 也走过**：Mixtral 8x7B (32L/4096) → 8x22B (56L/6144) 同时长深 + 长宽，证明 base 短一点不影响后续放大

### 短宽 起点的代价

1. **同 active 算力下 reasoning 可能略弱**：GLM-4.5 (92L) 显式选 depth > width，因 reasoning 偏 deep；OpenAI 系（Switch / GLaM）也历史性偏 deep。短宽是有 quality 风险的。
2. **Attention FLOPs 上升**：hidden 越大，attention FLOPs ∝ hidden²。hidden 2048→2560 = 1.56× attention 算力（虽然 attention 在 16B 总 FLOPs 只占 ~20%，但仍是真实增量）
3. **Embedding 浪费**：hidden 越宽，embedding + LM head 占比越大（128K vocab × 2560 = 655M，比 2048 × 128K = 524M 多 25%）
4. **没有 1B+ MoE 直接走 16L 验证过**：OLMoE 是 16L 但只到 6.9B/1.3B-active；纯短宽 16B-MoE 是探索区

---

## 2. 全市场 depth/width 分布（从 28_open_source_moe_catalog 抽取）

> 按 Layers / sqrt(Hidden) 比值排序（小 = 短宽，大 = 高瘦）。这个比值有理论依据：相同参数预算下，dense 网络"形状"通常用 L/√H 衡量。

| Model | Total | Active | Layers | Hidden | **L/√H** | 取向 |
|---|---|---|---|---|---|---|
| LongCat-Flash | 560B | 27B | 28 | 6144 | **0.36** | 极短宽（Meituan 2025-09） |
| OLMoE | 6.9B | 1.3B | 16 | 2048 | **0.35** | 短宽（AI2 2024） |
| Ling-mini-2.0 | 16B | 1.4B | 20 | 2048 | **0.44** | 短宽（Inclusion 2025-09） |
| Phi-3.5-MoE | 41.9B | 6.6B | 32 | 4096 | 0.50 | 偏短宽 |
| Mixtral 8x7B | 47B | 13B | 32 | 4096 | 0.50 | 中庸 |
| WizardLM-2 8x22B | 141B | 39B | 56 | 6144 | 0.71 | 偏深 |
| Llama 4 Scout | 109B | 17B | 48 | 5120 | 0.67 | 偏深 |
| GLM-4.5-Air | 106B | 12B | 46 | 4096 | 0.72 | 偏深 |
| **DeepSeek-V2-Lite** | **15.7B** | **2.4B** | **27** | **2048** | **0.60** | **中庸（你的 baseline）** |
| Moonlight-16B | 15.3B | 2.24B | 27 | 2048 | 0.60 | 中庸 |
| DeepSeekMoE-16B | 16.4B | 2.8B | 28 | 2048 | 0.62 | 中庸 |
| dots.llm1 | 142B | 14B | 62 | 4096 | 0.97 | 深 |
| Qwen3-30B-A3B | 30B | 3B | 48 | 2048 | **1.06** | 高瘦 |
| Qwen3-Next-80B | 80B | 3B | 48 | 2048 | 1.06 | 高瘦 |
| **GLM-4.5** | **355B** | **32B** | **92** | **5120** | **1.29** | **极高瘦（reasoning 路线）** |

**关键观察**：
- 16B 量级的 SOTA 都集中在 L/√H = 0.4-0.62（V2-Lite / Moonlight / DeepSeekMoE / Ling-mini-2.0 / OLMoE）
- **Ling-mini-2.0 是 16B 量级"短宽"的最直接 anchor**（20L/2048, L/√H=0.44）
- 高瘦极端是 GLM-4.5 系（92L）和 Qwen3-30B/Next（48L）—— 都是 reasoning-leaning 选择
- 极短宽是 LongCat-Flash（28L/6144）和 OLMoE（16L/2048）
- **你的 baseline V2-Lite 路线（27L/2048）是中庸**

---

## 3. 短宽 vs 高瘦的因果链

### 3.1 推理 latency

每生成一个 token 需要顺序经过所有 L 层。**每层 latency 约等于 attention + FFN 的 GPU 计算时间**（受 NVLink / HBM 带宽限制）。

- L=27（baseline）：每 token 27 步顺序计算
- L=20（Option B）：每 token 20 步 — 减 26%
- L=16（Option C）：每 token 16 步 — 减 41%

**注**：单 step 的计算量在短宽下变大（hidden 更大），但 GPU 并行性吃满 → 实际 wall clock 收益主要来自层数。

### 3.2 训练吞吐

PP 并行下，bubble 时间 ∝ L × micro_batch_count⁻¹。L 减少直接缩 bubble，提升 MFU 1-3 个百分点（具体取决于 PP 配置）。

### 3.3 Reasoning quality

GLM-4.5 论文 (arXiv 2508.06471) 团队明确选 92L/5120 的"depth > width"路线，理由是 reasoning task 受益于更深的层级抽象。

但 **Depth Delusion** (2026-01, arXiv 2601.20994) 反方观察：在 7B 上 deep + 160M 多参数比 wide 配置 quality 反而差。论文用了"zombies"形容多出的 deep 参数。

**直觉**：
- Reasoning task（GSM8K, GPQA, MATH）：浅模型上限略低，但 16L+ 都够（不要 < 12L）
- Knowledge task（MMLU, C-Eval）：深度敏感度低，主要看 total params
- Code task（HumanEval）：中等敏感

### 3.4 Attention 算力

Per token attention FLOPs ≈ 4 × hidden² + 2 × hidden × seq_len。

| 配置 | hidden | 单层 attn FLOPs | 27→20 层（B 路线） | vs baseline |
|---|---|---|---|---|
| Baseline | 2048 | 16.8M | × 27 = 453M | 1.00× |
| Option A | 2048 | 16.8M | × 24 = 403M | 0.89× |
| **Option B** | 2304 | 21.2M | × 20 = 425M | **0.94×** |
| Option C | 2560 | 26.2M | × 16 = 419M | 0.92× |

意外的好结果：**短宽 + 略加 hidden 下 attention 总 FLOPs 不增反降**——因为 hidden² 上升被层数下降抵消。

### 3.5 KV cache（推理显存）

KV cache size = 2 × L × num_kv_heads × head_dim × seq_len × precision。

| 配置 | L | kv_heads | head_dim | 32K seq KV cache (BF16) |
|---|---|---|---|---|
| Baseline | 27 | 4 | 128 | **864 MB** |
| **Option B** | **20** | **6** | **128** | **960 MB** (+11%) |
| Option C | 16 | 5 | 128 | 640 MB (-26%) |

Option B 因 kv_heads 多 (6 vs 4) 略增；Option C 因层少显著减。但都在 1GB 以内，**16B 32K 推理上不是瓶颈**。

### 3.6 综合 - L/√H 路线选择

短宽 (L/√H < 0.5)：偏 inference-friendly + upscale-friendly
中庸 (L/√H ≈ 0.6)：业界 16B 主流，稳妥
高瘦 (L/√H > 0.9)：偏 reasoning + 高质量上限，但难 upscale

---

## 4. 三个具体配方（精确参数账）

> 共同 spec：N_routed=64, K=8, N_shared=1, vocab=128K, GQA, head_dim=128, MTP D=1 (训练辅助), 第 0 层 dense FFN。

### Option A — 温和短宽（V2-Lite 浅版）

| 维度 | 值 | 说明 |
|---|---|---|
| Layers | **24** | -11% vs baseline 27 |
| Hidden | 2048 | 不变 |
| d_expert | **1664** | 加宽以补 layer 减少 |
| GQA | 16Q / 4KV | 同 baseline |
| 第 0 层 dense FFN intermediate | 11008 | hidden × 5.4 |

**参数账**：

| 模块 | 算式 | 总参 (M) | Active 严格 (M) |
|---|---|---|---|
| Embedding (untied) | 128K × 2048 | 262 | – |
| LM head (untied) | 128K × 2048 | 262 | – |
| Layer 0 attn (GQA 16/4) | 4 × 2048² 等价 | 10.5 | 10.5 |
| Layer 0 dense FFN | 3 × 2048 × 11008 | 67.6 | 67.6 |
| 23 MoE layers attn | 23 × 10.5 | 241.5 | 241.5 |
| 23 MoE layers shared expert | 23 × 3 × 2048 × 1664 | 234.5 | 234.5 |
| 23 MoE layers all routed | 23 × 64 × 3 × 2048 × 1664 | 15,058 | – |
| 23 MoE layers active K=8 routed | 23 × 8 × 3 × 2048 × 1664 | – | 1,882 |
| **Base total** | | **16,136** | **2,436** |
| MTP module (1 transformer block) | | ~85 | – |
| **含 MTP 训练时** | | **16,221** | **2,436** |

→ **Base ≈ 16.1B / Active 严格 ≈ 2.44B / V3 口径 ≈ 2.96B**

### Option B — 中等短宽（Ling-mini-2.0 升级）⭐ **推荐默认**

| 维度 | 值 | 说明 |
|---|---|---|
| Layers | **20** | -26% vs baseline 27；与 Ling-mini-2.0 一致 |
| Hidden | **2304** | +12.5% vs 2048 |
| d_expert | **1792** | 加宽补层减少 |
| GQA | **18Q / 6KV** | head_dim=128, 2304/128=18 自然 |
| 第 0 层 dense FFN intermediate | 12288 | hidden × 5.3 |

**参数账**：

| 模块 | 算式 | 总参 (M) | Active 严格 (M) |
|---|---|---|---|
| Embedding (untied) | 128K × 2304 | 295 | – |
| LM head (untied) | 128K × 2304 | 295 | – |
| Layer 0 attn (GQA 18Q/6KV) | (5.31+1.77+1.77+5.31) | 14.2 | 14.2 |
| Layer 0 dense FFN | 3 × 2304 × 12288 | 84.9 | 84.9 |
| 19 MoE layers attn | 19 × 14.2 | 269.8 | 269.8 |
| 19 MoE layers shared | 19 × 3 × 2304 × 1792 | 235.3 | 235.3 |
| 19 MoE layers routed all | 19 × 64 × 3 × 2304 × 1792 | 15,067 | – |
| 19 MoE layers routed K=8 active | 19 × 8 × 3 × 2304 × 1792 | – | 1,883 |
| **Base total** | | **16,266** | **2,487** |
| MTP module | | ~95 | – |
| **含 MTP 训练时** | | **16,361** | **2,487** |

→ **Base ≈ 16.3B / Active 严格 ≈ 2.49B / V3 口径 ≈ 3.08B**

### Option C — 激进短宽（LongCat 极端 + OLMoE 风格）

| 维度 | 值 | 说明 |
|---|---|---|
| Layers | **16** | -41% vs baseline；OLMoE 同款 |
| Hidden | **2560** | +25% vs 2048 |
| d_expert | **2048** | 进一步加宽 |
| GQA | **20Q / 5KV** | head_dim=128, 2560/128=20 |
| 第 0 层 dense FFN intermediate | 13824 | hidden × 5.4 |

**参数账**：

| 模块 | 算式 | 总参 (M) | Active 严格 (M) |
|---|---|---|---|
| Embedding (untied) | 128K × 2560 | 328 | – |
| LM head (untied) | 128K × 2560 | 328 | – |
| Layer 0 attn (GQA 20Q/5KV) | (6.55+1.64+1.64+6.55) | 16.4 | 16.4 |
| Layer 0 dense FFN | 3 × 2560 × 13824 | 106.1 | 106.1 |
| 15 MoE layers attn | 15 × 16.4 | 245.7 | 245.7 |
| 15 MoE layers shared | 15 × 3 × 2560 × 2048 | 236.0 | 236.0 |
| 15 MoE layers routed all | 15 × 64 × 3 × 2560 × 2048 | 15,099 | – |
| 15 MoE layers routed K=8 active | 15 × 8 × 3 × 2560 × 2048 | – | 1,887 |
| **Base total** | | **16,359** | **2,491** |
| MTP module | | ~120 | – |
| **含 MTP 训练时** | | **16,479** | **2,491** |

→ **Base ≈ 16.4B / Active 严格 ≈ 2.49B / V3 口径 ≈ 3.15B**

### 三方对比

| | Baseline (V2-Lite) | Option A (温和) | **Option B (推荐)** | Option C (激进) |
|---|---|---|---|---|
| Layers | 27 | 24 | **20** | 16 |
| Hidden | 2048 | 2048 | **2304** | 2560 |
| d_expert | 1408 | 1664 | **1792** | 2048 |
| L/√H | 0.60 | 0.53 | **0.42** | 0.31 |
| Total params (base) | 15.5 B | 16.1 B | **16.3 B** | 16.4 B |
| Active 严格 | 2.4 B | 2.44 B | **2.49 B** | 2.49 B |
| Inference latency (层 fewer 收益) | baseline | -11% | **-26%** | -41% |
| Attention FLOPs total | 1.00× | 0.89× | **0.94×** | 0.92× |
| KV cache 32K BF16 | 864 MB | 768 MB | 960 MB | 640 MB |
| Reasoning quality risk | none | low | **moderate** | high |
| Depth upscale headroom | 27→40 (1.5×) | 24→40 (1.7×) | **20→40 (2.0×)** | 16→32 (2.0×) |
| Expert upcycle headroom | 同 | 同 | 同 | 同 |
| 已有 16B SOTA 验证 | V2-Lite, DeepSeekMoE-16B, Moonlight | 无 | **Ling-mini-2.0 (20L/2048)** | 无 |

---

## 5. Upscale 路径详细规划

### 5.1 Depth Up-Scaling 方法学（按成熟度排序）

**SOLAR-DUS (Upstage 2023-12, [arXiv 2312.15166](https://arxiv.org/abs/2312.15166))** ⭐ 最成熟
- 步骤：(a) 复制 base model → (b) 各去掉 m 层 → (c) 拼接
- 例：Mistral 7B 32L → 复制成 2 个 → 各去 8 层 → 24+24=48L
- 关键：去掉中间层而非首尾，保留 embedding ↔ output 的对齐
- 后续 continued pretrain ~3B tokens 修复 grad-flow gap
- 已在 dense 上充分验证；**MoE 上未公开验证案例**

**LLaMA Pro Block Expansion (Wu et al. 2024)**
- 在 base 中插入新层，新层初始化为"恒等映射"（≈ skip connection）
- 训练时新层 unfreeze，base 层 frozen 一段时间
- 比 SOLAR 更 clean，但收敛慢
- Dense LLM 上有效；MoE 上 open

**LESA (ACL 2025, [aclanthology.org/2025.acl-long.1095](https://aclanthology.org/2025.acl-long.1095.pdf))** — 2025 新方法
- Learnable Layer Scaling-Up：用 meta-net 学习如何加层
- 比 SOLAR/LLaMA Pro 更智能，但工程复杂
- 太新，复现案例少

**Depth Tiling (Rae 2021, Sparse Upcycling 论文 §3 baseline)**
- 简单复制 layer 直接堆叠
- Grad flow 差，被 SOLAR 取代
- 不推荐

### 5.2 Expert Upcycling（已在 19_sparse_upcycling.md 详述）

- N_routed: 64 → 128 / 256 / ...
- 每个 expert 复制 + 加小噪声 (σ=0.01) 分化
- Router 全新初始化
- 适合：< 1× 原 dense budget 的"快速扩容"场景

### 5.3 三种短宽 base + upscale roadmap

#### Scenario 1：Option B 起手 → SOLAR-DUS 加深到 32B

```
Stage 1 (主训练)
  Spec: 20L / 2304 / N=64 / K=8 / 16.3B / 2.49B active
  训练: 12-15T tokens (你的主 spec)

Stage 2 (depth upscale, 3 个月后)
  方法: SOLAR-DUS (m=4)
  从 20L 复制 → 各去 4 层 → 16+16 = 32L
  新 spec: 32L / 2304 / N=64 / K=8 / ~26B / ~3.9B active
  Continued pretrain: 3-5T tokens (recovery + 新能力)
  
Stage 3 (expert upcycle, 6 个月后)
  方法: Sparse Upcycling
  N=64 → 128（每个 expert 复制 2 份 + σ=0.01 噪声）
  新 spec: 32L / 2304 / N=128 / K=8 / ~50B / ~3.9B active
  Continued pretrain: 1-2T tokens
```

→ 6-9 个月内一个 16B base 演化到 50B + 同 active；deploy 成本几乎不变。

#### Scenario 2：Option B → 直接 SOLAR + Expert 同步

```
Stage 1: 16.3B / 20L / 2304 / N=64 → 训完
Stage 2: 同时 depth 加到 30L (m=5) + expert 加到 N=96
  新 spec: 30L / 2304 / N=96 / ~32B / ~3.7B active
  Continued pretrain: 5T tokens
```

更激进但风险更大（同时改两个变量）。

#### Scenario 3：Option C → 翻倍 depth

```
Stage 1: 16.4B / 16L / 2560 / N=64 → 训完
Stage 2: SOLAR-DUS (m=4): 16L → 12+12 = 24L
  新 spec: 24L / 2560 / N=64 / ~23B / ~3.3B active
Stage 3: 再来一次 24L → 18+18 = 36L
  新 spec: 36L / 2560 / N=64 / ~34B / ~4.7B active
```

Option C 的"翻翻"潜力最大，但每次都要 3-5T tokens recovery training，总成本高。

### 5.4 Upscale 友好性总结

| Option | Depth Upscale 友好 | Expert Upcycle 友好 | 综合 |
|---|---|---|---|
| Baseline 27L | △（27→40 是 1.5×，但 40L 已接近行业上限） | ✓ | 一般 |
| A 24L | ✓ | ✓ | 良 |
| **B 20L** | **✓✓**（20→40 是 2× 安全区） | ✓ | **最优** |
| C 16L | ✓✓✓（16→32 是 2× 翻倍） | ✓ | 良（但 base reasoning 风险高） |

---

## 6. 风险登记与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Option B 在 GPQA/Math 上输 baseline ≥ 2pt | 中 | 中 | A2 T2.6 验证；准备 fallback 到 24L |
| SOLAR-DUS 在 MoE 上没公开案例 → upscale 翻车 | **中** | 高 | Stage 2 之前先用 1B/200M-active 跑 SOLAR-DUS proof，验证 router 不发散 |
| 加宽 hidden 后 RoPE base 需要调（128K 外推） | 低 | 低 | hidden 2304 与 Qwen3-30B 同；RoPE base=1e6 不变 |
| Attention head 数变化（16→18→20）训练不稳 | 低 | 中 | A0/A1 sanity check 阶段就发现 |
| Embedding 占总参比例上升 (4.4% → 5.7%)  | 低 | 低 | 接受；现代 MoE 主流都这个比例 |
| MTP head 在更宽 hidden 下需要重新调 λ | 低 | 低 | T3.1 消融时同步验证 |
| Continued pretrain budget（3-5T tokens × 2 stage = 6-10T）总成本超预算 | **中** | 中 | upscale 路径列入 roadmap 但 release 不绑定 |

---

## 7. Wind tunnel A2 调整 — 新增 T2.6 消融

> 把 depth-width 消融加入 29_wind_tunnel_a2 T 系列。

### T2.6 — Depth-Width Tradeoff

A2 等比缩小后的 1B 对应规模：

| Arm | A2 配置 (1B/200M-active) | 对应 16B Profile |
|---|---|---|
| **A** (baseline) | 12L / 1024 / d_expert=704 | Baseline 27L/2048 |
| B (Option A 缩) | 11L / 1024 / d_expert=832 | Option A 24L/2048 |
| **C** (Option B 缩) | **10L / 1152 / d_expert=896** | **Option B 20L/2304** ⭐ |
| D (Option C 缩) | 8L / 1280 / d_expert=1024 | Option C 16L/2560 |

每 arm 25B tokens / ~21 H100-hr → 4 arms = 84 H100-hr。

**决策指标**：
- 主：valid loss + GSM8K + MMLU + HumanEval
- 副：单 token inference latency (步数 × per-step 时间) + attention all-time MFU

**Accept/reject thresholds**：
- 若 C arm（Option B）在 GSM8K + MMLU 上输 A arm ≥ 1.5pt → reject Option B，回退 Option A
- 若 C arm 与 A arm 持平或更好（< 1pt 差）→ 切换 Profile B 主 spec 到 Option B
- 若 D arm（Option C）在 reasoning bench 上输 ≥ 3pt → 弃用 Option C 作为 Profile R 候选
- 若 C / D arm 在 throughput 上比 A arm 高 ≥ 5% 且 quality 持平 → 强采纳

**算力总账更新**：A2 从 27 arms → 31 arms，从 ~580 H100-hr → ~660 H100-hr。仍 < wind tunnel 总预算的 2%。

---

## 8. 推荐：分两步走

### 第一步 — 走标准路径，但 spec 留 short-wide 接口

**主 spec 维持 27L / 2048（Profile B baseline）**作为 A4 final sanity 的 main run，**但 wind tunnel A2 添加 T2.6 消融**。

理由：
- baseline 是经过 V2-Lite / Moonlight / DeepSeekMoE-16B / Ling-mini-2.0 多次验证的安全区
- Option B (20L/2304) 是 Ling-mini-2.0 同款 layer，但更宽 hidden，**没有任何 paper 直接验证**
- A2 用 ~80 H100-hr 跑 T2.6 即可知道相对差异

### 第二步 — 根据 A2 结果选择主 spec

**T2.6 通过条件**：Option B (1B 缩比) loss 差 baseline ≤ 0.003 且 GSM8K + MMLU 差 ≤ 1pt → **切 Profile B 主 spec 为 Option B**

**主 spec 切换后的连锁更新**：
- 22_FINAL §2 Profile B 表：Layers 27→20, Hidden 2048→2304, d_expert 1408→1792
- 22_FINAL §10 风险登记：新增"Option B 在 reasoning 下游的潜在 -1pt 风险"
- 22_FINAL §11 可调维度：Layers 范围 22-32 改为 18-30
- 28_open_source_moe_catalog 新增 Option B/C anchor 标注

### 长线 upscale roadmap

不论 A2 是否通过 Option B，都建议在 22_FINAL 后加一节 §12 "Upscale Roadmap"：
1. Stage 1：16B base（Profile B 或 Option B）训 12-15T
2. Stage 2 (可选)：SOLAR-DUS 加深到 30-32L → ~26-32B total
3. Stage 3 (可选)：Sparse Upcycling 加 expert 到 N=128 → ~50B total
4. 每阶段 continued pretrain 3-5T tokens

Roadmap 不绑定 release，只是把"未来 upscale 时不被 spec 卡住"明文化。

---

## 9. 一句话总结

> **想做"矮胖"的话推荐 Option B（20L / 2304 / d_expert 1792），有 Ling-mini-2.0 同 layer 验证、+12.5% 加宽 hidden 把参数补回，attention FLOPs 不增反降（0.94×），depth upscale 翻倍空间 2.0× 安全区。但 reasoning quality 风险需 A2 T2.6 验证后才能切主 spec；不通过就回退 Option A 或 baseline。**

---

## 10. 与其他笔记的交叉

- 主 spec：22_FINAL_16B_design.md §2（待 A2 T2.6 通过后更新 Layers/Hidden/d_expert）
- 已有 upscale 调研：19_sparse_upcycling.md（expert 路径）
- Wind tunnel：29_wind_tunnel_a2.md（添加 T2.6 arm A/B/C/D）
- Reasoning 评估：21_reasoning_vs_memorization.md（depth 偏 reasoning 的证据）
- 实施细节：30_routing_implementation.md（n_q_heads / kv_heads 变化下 router 不需改）
- 入门概念：31_foundations.md §12（"22_FINAL 每个数字的因果"会需要更新）

## 11. Open question 留给社区 / 未来研究

1. **MoE 上 SOLAR-DUS 的 router 行为**：dense SOLAR 把 layer 拼接后 router 怎么变？复制的 layer 用同 router 还是各自重训？无公开案例
2. **Ling-mini-2.0 选 20L 的官方 reasoning**：Ling 团队没有正式 paper 说明为什么选 20 而不是 27 —— 可能纯工程原因，也可能有质量考量
3. **"Depth Delusion" 是否成立 in MoE**：那篇 paper 用 dense；MoE 的 router + expert 加深和加宽机制可能不同
4. **Option B (20L/2304) 是否会在 long-context (128K+) 时 underperform**：浅模型在长上下文 retrieval 上可能弱（attention 深度对 long-range 重要）→ A3 阶段做 needle-in-haystack benchmark 验证

## 12. 参考资料

- SOLAR-10.7B paper: [arXiv 2312.15166](https://arxiv.org/abs/2312.15166), [ACL Anthology](https://aclanthology.org/2024.naacl-industry.3/)
- LESA (2025): [aclanthology.org/2025.acl-long.1095](https://aclanthology.org/2025.acl-long.1095.pdf)
- Depth Delusion (2026): [arXiv 2601.20994](https://www.arxiv.org/pdf/2601.20994)
- Generalizing Scaling Laws for Dense and Sparse Large Language Models (2025): [arXiv 2508.06617](https://arxiv.org/pdf/2508.06617)
- 本仓库 19_sparse_upcycling.md（expert 路径 upscale）
- 本仓库 28_open_source_moe_catalog.md §3（depth/width 分布统计的源数据）
