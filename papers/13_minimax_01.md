# MiniMax-01: Scaling Foundation Models with Lightning Attention

- **arXiv**: 2501.08313
- **机构**: MiniMax (90 作者，alphabetical order; submitter Yiran Zhong)
- **发表时间**: 2025-01-14

## TL;DR

MiniMax-01 是 MiniMax 第一代开源旗舰，**456B total / 45.9B active / 32 experts / top-2** 的 MoE，但与 DeepSeek-V3 这类 dense-softmax-attention MoE 显著不同，它采用了 **Lightning Attention（线性注意力变体）× 7 + Softmax Attention × 1 的混合（hybrid）注意力块**。这套设计使得它能在训练阶段原生支持 1M tokens、推理时外推到 4M tokens，且 100K 生成时 FLOPs 仅为 DeepSeek-R1 的 25%。是目前规模最大、研究最深入的开源 hybrid-attention MoE。

## 关键架构配置

| 参数 | 数值 |
| --- | --- |
| Total params | 456 B |
| Active params / token | 45.9 B |
| Layers | 80 |
| Hidden dim | 6144 |
| Attention heads | 64 × 128 (head_dim 128) |
| Softmax 层 GQA group | 8 |
| Experts (N) | 32 |
| Top-k | 2 |
| Expert FFN hidden dim | 9216 |
| Hybrid pattern | 7 × Lightning + 1 × Softmax |
| → Lightning attention 层 | 70 |
| → Softmax attention 层 | 10 |
| RoPE base (long-context) | 1e7 |
| Train context | 1 M tokens |
| Inference context (extrapolation) | 4 M tokens |
| 训练 token 数 (VL) | 512 B |

## 核心方法 / 创新点

### 1. Lightning Attention（线性注意力）

Lightning Attention 是 TransNormer 系一脉的线性注意力变体（无 softmax），核心写法：

```
O = Norm( Q (KᵀV) )
```

- 训练 / 推理都用 **tiling**（块化）来兼顾 GPU 算力与显存：
  - **Intra-block**（块内）：左乘 `[(QKᵀ) ⊙ M] V`，保留因果 mask 的稀疏特性
  - **Inter-block**（块间）：右乘 `Q (KᵀV)`，复用前缀状态
- 复杂度从 softmax 的 `O(N²d)` 降到 `O(Nd² + nBd)`（B 为 block size），prefill 阶段线性可扩展。
- 推理时 attention "state" 与序列长度无关，对 1M+ context 是关键。

### 2. Hybrid Attention Pattern — 7 Lightning + 1 Softmax

每 8 层中 7 层 Lightning + 1 层 Softmax（在 80 层中即 70 + 10）。这是论文里反复消融过的核心配置：

- **Lightning 层**承担绝大部分序列建模负载，提供 O(N) 的吞吐 / 显存收益；
- **Softmax 层**（每 8 层 1 次）提供精确 retrieval / in-context-learning 能力，弥补线性 attention 在精确 token-level 检索上的弱点。论文将其类比为"long-term memory checkpoint"。
- Softmax 层使用 GQA（group size 8）+ RoPE，RoPE 只作用于一半 head dimension。

为什么不是全 Lightning？纯线性 attention 在 needle-in-haystack 等检索任务上明显不如 softmax，hybrid 等于"用 1/8 的 softmax 成本买到几乎完整的 retrieval 能力"。

### 3. MoE — 32 experts / top-2 + Global Router

```
h_t = Σ_{i=1..E} Softmax_i( TopK(x_t · W_g) ) · FFN_i(x_t)
```

- 32 experts、top-2、token-drop with capacity（即丢弃 overflow token）
- 辅助负载均衡 loss：`L_aux = α_aux · (1/E) Σ f_i · m_i`
- **Global Router**：在 Expert Parallel（EP）通信中加入一次额外 allgather，让 router 看到 EP 组全局 token 分布并据此重路由，避免某些 GPU "过载、其他 GPU 空闲"。这是它与 DeepSeek-V3 的 device-limited routing / bias-based loss-free balance 的主要差异。

### 4. DeepNorm + PostNorm

- 用 PostNorm + DeepNorm 缩放（α = (2N)^0.25, β = (8N)^(-0.25), N=80）取代主流的 PreNorm。
- 论文论点：80 层很深，PreNorm 会让 effective depth 衰减，PostNorm 配 DeepNorm 在深层稳定性更好。

## 训练 & 系统细节

- **三阶段长上下文训练**：
  1. 128K context：300B tokens
  2. 512K context：32B tokens
  3. 1M context：26B tokens
- 推理外推到 4M tokens，靠 RoPE base = 1e7 + Lightning 的常数状态。
- 单节点 8×80GB GPU + 8-bit 量化即可跑 1M context 推理。
- VL 版本额外训练了 512B vision tokens。

## 关键消融与结果

- 性能：在多项基准上对标 GPT-4o / Claude-3.5-Sonnet，长上下文（RULER 4M、LongBench-v2 等）大幅领先 — 20×~32× 的 context 优势。
- FLOPs：100K 生成长度下相比 DeepSeek-R1 仅 25% FLOPs（M1 论文里的对照数据）。
- Hybrid pattern 消融：作者在小模型上验证，hybrid (7+1) 在通用基准上几乎不弱于 full softmax，但显存 / 吞吐显著占优；纯 Lightning 在长程精确检索任务上显著掉点。
- Global Router 对训练稳定性 / load balance 有正面效果（具体数字论文表中）。

## 对 16B MoE 设计的启示

**主要结论：16B 规模上 hybrid attention 收益有限，建议保持 dense softmax attention。**

- **不建议直接照搬 hybrid (7 + 1) Lightning + Softmax**：
  - MiniMax 之所以做 hybrid，是因为 80 层 × 6144 hidden × 1M context 下，softmax KV cache / 算力都顶不住；16B 规模不存在这个压力。
  - Lightning Attention 自带工程包袱（自定义 kernel、tiling 状态管理、量化兼容性），16B 不值这个复杂度。
  - 16B 规模主流 context（32K~128K）下 softmax + GQA / MLA 完全够用。
- **可以借鉴的点**：
  1. **Global Router 思路**：在 EP 通信中加 allgather 让 router 看到全局分布，对 expert 负载均衡有帮助 —— 但 DeepSeek-V3 的 bias-based loss-free balance 更简洁，建议优先后者。
  2. **DeepNorm + PostNorm 在深层** 的稳定性论点对 16B 影响小（层数通常 27~32 层），无需切换。
  3. **三阶段长上下文训练课表**（128K → 512K → 1M）的 token 配比（10 : 1 : 1）是可参考的工程经验，如果 16B 要做 128K+ 长上下文。
- **核心警示**：MiniMax-01 的设计是"为 1M+ context 服务"的；如果 16B 模型目标 context ≤ 128K，hybrid attention 几乎没有正收益，只有复杂度成本。

## Caveats / 局限

- Lightning Attention 在精确检索（needle-in-haystack）等任务上不如 softmax；hybrid 是补丁不是免费午餐，对 1/8 比例的选择论文未做特别详细的扫描（也许 1/4、1/16 都行）。
- Token-drop 策略（capacity-based）在 100% capacity 下会丢 token，对 RL / SFT 阶段长输出可能掉点；DeepSeek-V3 的 no-token-drop 路线相比更稳。
- 训练 FLOPs / 总 token 数论文未完全公开（VL 部分给了 512B，但语言部分 token 数不完全清晰）。
- Hybrid 设计带来的工程债（自定义 attention kernel、量化、推理 framework 适配）是真实成本，开源社区落地的实际门槛比 dense MoE 高。
