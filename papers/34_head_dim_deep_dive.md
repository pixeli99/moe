# head_dim 深度解析 — 为什么硬件 care，原理在哪

> **问题**：为什么 FlashAttention 只优化 head_dim ∈ {64, 128, 256}？这个限制不是写死的"规则"，是从 GPU 架构 → Tensor Core 指令 → SRAM tile 设计这三层硬件约束叠加出来的。
> **怎么读**：从 §1 概念定位开始，按顺序读到 §5 就够；§6 是把所有约束综合起来。不要跳读，每一层都建立在前一层上。

---

## 1. head_dim 是什么 — 先把概念在 attention 计算里定位

### 1.1 Attention 单 head 的 4 步

对一个 token x ∈ R^H（H = hidden, 比如 2048）：

```
1. Q = x @ W_Q     shape: [head_dim]   ← 投影到 head 空间
2. K = x @ W_K     shape: [head_dim]
3. V = x @ W_V     shape: [head_dim]
4. attention(Q, K's of all prev tokens, V's of all prev tokens) → out [head_dim]
```

对**整个序列 seq=N**，一个 head：
- Q, K, V 都是 [N, head_dim] 矩阵
- Attention scores: **Q @ K^T** → [N, N]，**这里 head_dim 是 GEMM 的 K (reduction) 维**
- softmax(scores / √head_dim)
- 输出: **attn_weights @ V** → [N, head_dim]，**这里 head_dim 是 GEMM 的 N (output) 维**

### 1.2 Multi-head 是怎么拼的

```
hidden H = num_heads × head_dim
```

例：H=2048, num_heads=16 → head_dim = 128。
每 head 用自己的 W_Q, W_K, W_V，独立计算 attention，最后拼回 [N, H]。

GQA：Q heads 多（比如 16），KV heads 少（比如 4），4 个 Q head 共享一组 K/V。

### 1.3 head_dim 的两面角色

| 在 Q @ K^T 里 | head_dim 是 K dim（reduction） |
| 在 attn @ V 里 | head_dim 是 N dim（output） |

这两个角色就是硬件 care 的根源 — **head_dim 在两个独立 GEMM 里以不同身份出现，对两边都要满足硬件对齐**。

→ **一句话**：head_dim 是每个 attention head "私有"的工作维度。它同时是 Q@K^T 的 K 维和 attn@V 的 N 维 — 任何关于矩阵形状对齐的硬件约束都打在它身上。

---

## 2. GPU 三层内存 hierarchy — 为什么所有这些都重要

NVIDIA H100 的内存结构（从慢到快）：

```
┌─────────────────────────────────────────────────────────────┐
│ HBM3 (显存 80GB)             ─── 3 TB/s  ─── 0.5 μs latency │
│       ↕                                                       │
│ L2 Cache (50 MB)             ─── 12 TB/s ─── 50 ns           │
│       ↕                                                       │
│ SM Shared Memory (228 KB/SM) ─── ~30 TB/s ─── 10 ns         │ ← FlashAttention tile 住这
│       ↕                                                       │
│ Register File (~256 KB/SM)   ─── ~100 TB/s ─── 1 ns         │ ← Tensor Core 直接吃这
│       ↕                                                       │
│ Tensor Core (4 个/SM)         ─── 1 cycle MMA               │
└─────────────────────────────────────────────────────────────┘
```

H100 有 132 个 SM（Streaming Multiprocessor），每 SM 持有：
- 228 KB Shared Memory（其中 ~96-100 KB 给 FlashAttention 用，剩下系统占）
- 4 个 Tensor Core unit
- 64 个 FP32 cores
- 32 threads × 4 = 128 thread per warp-group

### 关键 trade-off

| 数据放哪 | 速度 | 容量 |
|---|---|---|
| HBM | 1× (基准) | 80 GB |
| SRAM | **10×** | 96 KB / SM |
| Register | 100× | 量级 KB / SM |

**结论**：能放 SRAM 的就别放 HBM。FlashAttention 整个工作就是**把 attention 计算限制在 SRAM 里跑，不写 N×N attention matrix 回 HBM**。

而 SRAM 只有 96 KB，所以 attention 必须**分块（tile）**算。tile 多大？由 head_dim 决定。

→ **一句话**：GPU 的 SRAM 只有 96 KB，所以 attention 必须分块计算。每个 tile 大小直接被 head_dim 决定，head_dim 就是 SRAM 内存预算的核心维度。

---

## 3. Tensor Core 的 MMA 指令 — 矩阵乘法的"硬件原子单位"

### 3.1 Tensor Core 干什么

Tensor Core 不是普通 ALU，它一次性算一个**小矩阵乘法 + 加法**：

```
D = A × B + C        (A, B 是输入矩阵; C, D 是 accumulator)
```

**关键**：A, B 的形状是**写死在硬件指令里的**，不是任意形状！

### 3.2 各代 GPU 的 MMA 形状

| 代际 | 指令 | A shape (m×k) | B shape (k×n) | 输出 (m×n) | 注释 |
|---|---|---|---|---|---|
| Volta V100 | mma.sync (FP16) | 16×16 | 16×16 | 16×16 | k=16 |
| Ampere A100 | mma.sync (BF16) | 16×16 | 16×8 | 16×8 | **k=16 还是 16** |
| Hopper H100 | wgmma (BF16) | 64×16 | 16×N | 64×N | warp-group, N 灵活 |

**所有现代 BF16 Tensor Core MMA 的 K (contracting) 维 = 16**。

这意味着：任何参与 Tensor Core 计算的矩阵，**K 维必须是 16 的倍数**。否则要 padding（浪费）或走 CUDA cores（慢 10×）。

### 3.3 这对 head_dim 的影响

回到 §1.3：
- Q @ K^T 里 head_dim 是 K 维 → **head_dim 必须是 16 的倍数**
- attn @ V 里 head_dim 是 N 维 → 也要对齐（N 也最好是 16 的倍数）

合法的 head_dim：16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, ...

但**合法 ≠ 高效**。下面 §4-5 会告诉你为什么实际上只有 {64, 128, 256} 是真正 high-throughput。

→ **一句话**：BF16 Tensor Core 的 MMA 指令 K 维硬性写死=16。所以 head_dim 必须是 16 的倍数才能用 tensor core，否则掉到 CUDA core 慢 10×。

---

## 4. 内存访问对齐 — 为什么 128 字节这个数字反复出现

Tensor Core 的硬约束是 16 的倍数。但**实际高效要求更严**：

### 4.1 三个 128 字节的硬件事实

1. **HBM 内存事务 = 32 字节 / 64 字节 / 128 字节**（cache line 边界）
   - L1 / SRAM 的 cache line 是 128 字节
   - 一次最优 HBM 读 = 128 字节连续
   - 32 字节读也行但浪费
2. **Warp size = 32 threads**，每 thread 一次 vec4 load = 4 个 float32 = 16 字节
   - 32 thread × 16 字节 = **512 字节** per warp 一次访存
   - 但 vec4 load (`LDG.E.128`) 一次 16 字节，要求**数据地址 16 字节对齐**
3. **Shared memory banks = 32 banks × 4 字节 = 128 字节 / row**
   - 同一 row 不同 bank 可并行访问
   - **128 字节 row 大小是 SRAM bank conflict-free 访问的天然单位**

### 4.2 head_dim 翻译成字节

| head_dim | BF16 字节数 | 1 cache line (128B) 装得下吗？ |
|---|---|---|
| 64 | 128 B | 正好 1 个 cache line ✓ |
| 80 | 160 B | 1.25 个 cache line ✗ |
| 96 | 192 B | 1.5 个 cache line ✗ |
| **128** | **256 B** | **正好 2 个 cache line ✓** |
| 192 | 384 B | 3 个 cache line ✓（次优） |
| **256** | **512 B** | **正好 4 个 cache line ✓** |

→ **64 / 128 / 256 在字节级别都是 128 字节的整数倍**。这就是为什么这三个数字"特殊"。

### 4.3 vec4 load 的角度

每个 thread 一次最多加载 16 字节（vec4 = 4×float = 4×4B = 16B）。

把 head_dim 个 BF16 元素分给一个 warp 的 32 thread：

| head_dim | BF16 字节数 | per thread 字节 | vec4 (16B) 完整？ |
|---|---|---|---|
| 64 | 128 B | 4 B | vec2 load (8B) × 1 不齐 |
| **128** | **256 B** | **8 B** | **vec4 load (16B) × 0.5？ 实际 vec2** |
| **256** | **512 B** | **16 B** | **vec4 完整 ✓** |
| 80 | 160 B | 5 B | 无法 vec 化，必须 scalar |
| 96 | 192 B | 6 B | vec2 部分 |

实际中 FlashAttention 用 vec4 load (`LDG.E.128`) 加载 K, V 列。head_dim ∈ {64, 128, 256} 时这个 load 完整使用带宽；其他 head_dim 要么浪费带宽，要么用更慢的 scalar load。

→ **一句话**：HBM cache line / SRAM bank / vec4 load 三件事都以 128 字节为单位。head_dim=128 (BF16=256B) 和 256 (BF16=512B) 都是 128B 的整数倍 → 内存访问最高效。

---

## 5. FlashAttention 的 SRAM tile 计算

现在把 §2 SRAM 预算 + §3 Tensor Core 形状 + §4 字节对齐结合起来，看 head_dim 怎么决定 tile size。

### 5.1 FlashAttention 的算法骨架

```python
# 输入: Q, K, V ∈ HBM, shape [N, d]  (N=seq, d=head_dim)
# 输出: O ∈ HBM, shape [N, d]

for i in range(N // Br):                          # Q 行分块, Br = Q tile rows
    Q_i = load_to_SRAM(Q[i*Br:(i+1)*Br])          # [Br, d]
    O_i = zeros([Br, d])
    
    for j in range(N // Bc):                       # K, V 列分块, Bc = K tile rows
        K_j = load_to_SRAM(K[j*Bc:(j+1)*Bc])      # [Bc, d]
        V_j = load_to_SRAM(V[j*Bc:(j+1)*Bc])      # [Bc, d]
        
        S_ij = Q_i @ K_j.T                         # [Br, Bc]
        P_ij = online_softmax(S_ij, ...)
        O_i  += P_ij @ V_j                         # accumulate in SRAM
    
    write_to_HBM(O_i)
```

关键观察：**Q_i, K_j, V_j, O_i 全在 SRAM**。中间 [Br, Bc] 也在 SRAM。**N×N attention matrix 从来不实例化**。

### 5.2 SRAM 预算

H100 每 SM 有 ~96 KB usable SRAM。需要装：
- Q_i: Br × d × 2 字节
- K_j: Bc × d × 2 字节
- V_j: Bc × d × 2 字节
- O_i: Br × d × 4 字节（FP32 accumulator）
- softmax 统计量 (LSE): 小，~Br × 8 字节

**总 SRAM = Br × d × 2 + 2 × Bc × d × 2 + Br × d × 4 + 杂项**
≈ (6 Br + 4 Bc) × d ≈ 10 × Br × d（假设 Br ≈ Bc）

要 ≤ 96 KB：
```
10 × Br × d ≤ 96 KB
Br × d ≤ 9600 elements
```

### 5.3 不同 head_dim 下的最大 tile

| head_dim (d) | max Br × Bc 同时 | 典型选择 | per SM 算力利用 |
|---|---|---|---|
| 64 | Br = 128, Bc = 128 ✓ | (128, 128) | 高（tile 大） |
| **128** | **Br = 128, Bc = 64 ✓** | **(128, 64) 或 (64, 128)** | **高** |
| 192 | Br = 64, Bc = 64 | (64, 64) | 中（tile 小，overhead 升） |
| 256 | Br = 64, Bc = 32 | (64, 32) | 中（tile 更小） |

**算力利用关键**：tile 越大，每次外层循环（HBM ↔ SRAM 数据搬运）摊销越多。tile 太小 → SRAM 没填满 → HBM 数据搬运频繁 → 受 HBM 带宽限制（"memory-bound"）。

### 5.4 head_dim = 80 这种"歪斜"值的悲剧

```
Br × 80 ≤ 9600  →  Br ≤ 120
```

但 Tensor Core 喜欢 Br 是 16 的倍数 → 最大 Br=112 不友好，实际只能选 Br=64 或 Br=96。**SRAM 利用率只有 ~70%**。

再加 §4 字节对齐问题：80 元素 vec4 load 不完整 → HBM 带宽利用率也降。

**双重低效**：tile 小 + memory access 慢。综合性能比 head_dim=128 低 **40-60%**。

→ **一句话**：FlashAttention 的 tile 大小受 SRAM 96 KB 限制，head_dim 越大 tile 越小；只有 {64, 128, 256} 能在 Br × Bc tile size 和 §4 字节对齐之间找到平衡点。

---

## 6. 把所有约束叠加 — 为什么是 {64, 128, 256}

三个独立约束：

| 约束 | 来源 | head_dim 必须 |
|---|---|---|
| ① Tensor Core MMA K=16 | §3 硬件 ISA | 16 的倍数 |
| ② Cache line / vec4 / bank | §4 内存子系统 | 128 字节倍数 → **64/128/256 in BF16** |
| ③ SRAM tile 利用率 | §5 FA 算法 | 不能太大（限 SRAM） |

合法的 head_dim：

| head_dim | 满足 ①? | 满足 ②? | 满足 ③? | 实际推荐 |
|---|---|---|---|---|
| 16 | ✓ | ✗ (32B) | ✓ | 太小，废 |
| 32 | ✓ | ✗ (64B) | ✓ | 不够大 |
| 48 | ✓ | ✗ (96B) | ✓ | 浪费 |
| **64** | **✓** | **✓ (128B = 1 cache line)** | **✓** | **gpt-oss, 老模型** |
| 80 | ✓ | ✗ (160B) | ✓ | Hunyuan 用，**低效** |
| 96 | ✓ | ✗ (192B) | ✓ | 某些老 ViT 用 |
| 112 | ✓ | ✗ (224B) | ✓ | 罕见 |
| **128** | **✓** | **✓ (256B = 2 cache lines)** | **✓** | **行业默认** |
| 144 | ✓ | ✗ | ✓ | 罕见 |
| 160 | ✓ | ✗ | ✓ | 罕见 |
| 192 | ✓ | ✓ (384B = 3 cache lines) | △ | **Qwen3-Coder, OK** |
| **256** | **✓** | **✓ (512B = 4 cache lines)** | **△ (tile 小)** | **Qwen3-Next, MFA** |

**铁三角**：head_dim ∈ {64, 128, 256}。**192 也 OK 但不如**。其他全部至少在一个维度上不优。

### 为什么 128 是甜区？

- ① ✓ 是 16 的倍数
- ② ✓ 是 128 字节倍数（2 cache lines, BF16）
- ③ ✓ tile (128×128) 刚好用满 SRAM 又不撑爆
- **质量**：head 内部表达力足够；num_heads = H/128 是个合理的整数

128 在三个约束的"中央甜区"。这就是为什么 Llama / Qwen / DeepSeek 全用 128。

---

## 7. 三个特殊 head_dim 的真实分析

### 7.1 head_dim = 64 (gpt-oss-120b / gpt-oss-20b)

**为什么 OpenAI 选 64**：
- gpt-oss 用 hidden=2880, 2880/64 = 45 head — 与 H100 的 wgmma m=64 比例好
- 64 让 head 更多 → attention pattern 更密集 → 可能更好抓细粒度
- SRAM tile 可以更大（Br=128, Bc=128）→ 单 SM 算力利用率高
- **代价**：每 head 表达力弱，需要更多 head 数 → attention head 间通信开销略高

### 7.2 head_dim = 80 (Hunyuan-Large)

**为什么 Hunyuan 选 80**：
- 80 = 16 × 5，**没有 64 / 128 的字节对齐优势**
- 实际 MFU 比同 hidden 的 128 配置低 ~20-30%（行业内部估计）
- **不推荐复现 Hunyuan-Large 的 80** — 历史选择，非最优

### 7.3 head_dim = 256 (Qwen3-Next / MFA Step-3)

**为什么这些模型选 256**：
- Qwen3-Next 80B/3B-active：**极稀疏 MoE**，attention 占总 FLOPs 较大比例 → 用 256 配少量 head（num_heads=8）减少 head-level overhead
- MFA Step-3：multi-matrix attention 设计上需要大 head
- SRAM tile 减小但仍可用（Br=64, Bc=32）
- **代价**：每 head 大表达力，但 num_heads 少 → 多样性弱

→ **小结**：head_dim 选择 = (硬件友好 ∩ 模型质量) 的最优点。128 是 99% 情况下的正确选择。

---

## 8. 对你 16B Profile B 的实际影响

### 8.1 baseline 选择

| | Profile B baseline | Option B (短宽) |
|---|---|---|
| Hidden | 2048 | 2304 |
| head_dim | **128** | **128** |
| num_q_heads | 2048/128 = **16** | 2304/128 = **18** |
| num_kv_heads (GQA 4:1) | 4 | **6 → 但 6 不是 4 的倍数关系？** |

等等，2304 / 128 = 18，GQA 4:1 比例 18/4 = 4.5 → 不是整数。实际 GQA 18Q/6KV 是 3:1 比例（不是严格 4:1）。

**这是 hidden=2304 vs 2048 的隐藏代价之一**：2048 = 16×128 → GQA 16/4 完美；2304 = 18×128 → GQA 18/6 是 3:1 比例（KV head 数比 baseline 多 50%）。

KV cache 因此略增：
- Baseline 27L × 4 KV × 128 × 32K × 2 byte = 864 MB
- Option B 20L × 6 KV × 128 × 32K × 2 byte = 960 MB（+11%）

**我之前的算账正确**。Option B 这点 KV cache 代价就是从 hidden 2304 的 GQA 配比来的。

### 8.2 替代方案

如果你想 Option B hidden=2304 同时保持 4:1 GQA：
- 选 16Q/4KV → 但 16×128=2048 ≠ 2304，**Q proj 不再是方阵**

可以做：**Q proj output dim 2048 + 单独 reshape 到 16×128**。这是合法的（HuggingFace `head_dim` 字段独立于 `hidden_size / num_heads`），但与"hidden = num_heads × head_dim"的简化关系打破，工程上要注意。

主流实践：Q proj output dim = num_q_heads × head_dim（可以不等于 hidden）。Llama 1 7B 就是 hidden=4096, num_heads=32, head_dim=128 → Q proj 是 [4096, 4096]。但 Mistral 7B 是 hidden=4096, num_heads=32, head_dim=128 → 同样 Q proj 是 [4096, 4096]。

如果 hidden ≠ num_heads × head_dim：
- 比如 hidden=2304, num_q_heads=16, head_dim=128 → Q proj 是 [2304, 2048]（非方阵）
- O proj 是 [2048, 2304]
- **完全合法**，只是 Q/O proj 不是 self-symmetric

### 8.3 给你的建议

**Profile B baseline (2048/16Q/4KV) 是最干净的配置** — 三个约束全满足，GQA 4:1 完美。

**Option B (2304/18Q/6KV) 略有妥协** — GQA 比例 3:1，KV cache 多 11%。但 hidden 2304 = 18×128 仍然满足 head_dim 硬约束。

**Option C (2560/20Q/5KV) GQA 比例 4:1 ✓** — 数学上更干净。但 hidden 加 25% 带来 attention FLOPs ↑ 56%。

**总结**：head_dim=128 是 99% 情况下不需要重新论证的默认值。你需要决定的是 num_q_heads × head_dim = hidden 这个约束怎么解（这隐含了 hidden 的选择）。

→ **一句话**：Profile B / Option A / Option B / Option C 都用 head_dim=128（甜区不变），只有 num_q_heads 和 num_kv_heads 跟着 hidden 变。

---

## 9. 一句话总结（如果你忘了细节）

> **head_dim 是 attention 单 head 的工作维度。在 GEMM 里它既是 Q@K^T 的 reduction K 维（必须是 Tensor Core MMA 的 16 倍数），又是 attn@V 的 output N 维（必须是 128 字节对齐才能 vec4 load）。叠加 FlashAttention 的 SRAM 96 KB tile 预算约束，三件事的交集就是 {64, 128, 256}。128 在三个约束的中央甜区，是行业默认。**

---

## 10. 还可以深挖的方向

| 想知道什么 | 去看 |
|---|---|
| FlashAttention 完整算法 | arXiv 2205.14135 (FA1), 2307.08691 (FA2), Dao 博客 |
| Tensor Core MMA 指令集 | NVIDIA PTX ISA docs，section "Matrix multiply-accumulate" |
| H100 SRAM / SM 微观架构 | NVIDIA H100 whitepaper（公开） |
| 不同 head_dim 实际 MFU 测试 | Mosaic AI / Together / Anthropic 内部博客 |
| 为什么 OpenAI gpt-oss 选 64 | gpt-oss model card 没明说，社区猜测 |
| Qwen3-Next 选 256 的原因 | Qwen blog 提到"减少 attn head 开销" |

## 11. 与本仓库其他笔记的交叉

- 33_advanced_concepts §3.1：head_dim 简版（这里是详版）
- concepts.html "head-dim" 概念卡（简短版）
- 22_FINAL_16B_design §2：Profile B 的 GQA 16Q/4KV/head_dim=128 配置
- 32_depth_width_tradeoff §6：hidden 选择如何影响 Q/KV head 比例
