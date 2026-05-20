# Jamba: A Hybrid Transformer-Mamba Language Model

- **arXiv**: 2403.19887
- **机构**: AI21 Labs
- **发表时间**: 2024-03-28 (v1) / 2024-07-03 (revised)

## TL;DR

Jamba 是首个 production-scale 的 **Transformer-Mamba hybrid MoE**：在一个 8 层的 block 内交错 **1 层 Attention + 7 层 Mamba**，并在其中 **每隔 1 层（即每 2 层一次）把 MLP 换成 MoE**。原始论文释放的是 **52B total / 12B active** 版本，能在 80GB 单 GPU 上跑 256K context。AI21 在 follow-up Jamba-1.5（arXiv 2408.12570）中把规模扩到 **398B / 94B (Large)** 和 **52B / 12B (Mini)**。与 MiniMax-01 的"linear-attention + softmax hybrid"路线形成有趣对比 — Jamba 走的是"SSM (Mamba) + attention hybrid"路线。

## 关键架构配置

### Jamba (original, 2403.19887)

| 参数 | 数值 |
| --- | --- |
| Total params | 52 B |
| Active params | 12 B |
| Block 层数 | 8 (`l = 8`) |
| Attention : Mamba ratio | 1 : 7 (`a:m = 1:7`) |
| Block 数 | 4 (共 32 层) |
| MoE 频率 | 每 `e = 2` 层一次 |
| Experts (N) | 16 |
| Top-k | 2 |
| Attention | GQA |
| Activation | SwiGLU |
| Vocab | 64 K |
| 训练 context | 1 M tokens（成功训练过） |
| 发布 context | 256 K |
| 部署 | 单卡 80GB GPU |

### Jamba-1.5 (2408.12570, 不在主任务但用户提到)

| 模型 | Total | Active |
| --- | --- | --- |
| Jamba-1.5-Large | ~398 B | 94 B |
| Jamba-1.5-Mini | ~52 B | 12 B |

两个 1.5 版本都保留 1:7 hybrid + 256K context，主要是规模扩展 + instruction tuning。

## 核心方法 / 创新点

### 1. Jamba Block — 1 Attention + 7 Mamba

```
Block (l=8 layers):
  Mamba → Mamba → Mamba → Mamba → Attention → Mamba → Mamba → Mamba
```

- 4 个这样的 block 串联 = 32 层。
- **Mamba 层**：Selective State Space Model（Mamba-1），O(N) 复杂度，constant-size hidden state，对长序列零 KV 增量。
- **Attention 层**：GQA + RoPE，承担精确 token-level retrieval。
- 论文论点：**纯 Mamba 在 in-context learning 和 needle-in-haystack 上有明显短板**，加 1/8 比例的 attention 就能补回来，且保持显存友好。

### 2. MoE — 每 2 层一次

```
Layer 0 (Mamba) → MLP
Layer 1 (Mamba) → MoE  ← every 2 layers
Layer 2 (Mamba) → MLP
Layer 3 (Mamba) → MoE
...
```

- 16 experts, top-2，平均 ~8 experts 激活在 forward graph 中
- 与 DeepSeek-V3（每层都 MoE）不同，Jamba 是 **稀疏 MoE 层** 的设计（每两层一次），这样可以把 attention 层和 MoE 层错开，部分缓解 MoE 通信瓶颈。

### 3. 单卡 80GB 部署目标

Jamba 的工程目标之一就是 **single-GPU 部署**。Mamba 的常数状态 + 稀疏 MoE 让 52B/12B 在 80GB 内能跑 256K context — 这是 dense transformer 同规模做不到的（KV cache 会爆）。

## 训练 & 系统细节

- 训练过的最大 context：1M tokens；发布时限制到 256K（推理稳定性 / SFT data 长度限制）。
- MoE 实现：标准 token-choice + auxiliary load balancing loss
- Mamba kernel：使用原 Mamba-1 的 selective scan CUDA kernel
- 训练数据 / token 数论文未完全公开（AI21 商业模型）

## 关键消融与结果

- **Hybrid vs Pure Mamba**：作者明确做了消融 — 纯 Mamba 在 IBC（in-context learning benchmark）上显著低于 hybrid；加 1 个 attention 层就能补上。
- **Attention ratio 扫描**：a:m = 1:7 是论文里最优点（也试过 1:3、1:15）。
- **MoE 频率消融**：每 2 层一次比每 1 层一次（更密集 MoE）效果接近但成本更低。
- 在 LongBench / Needle-in-Haystack 上能稳定到 256K，pure-transformer 同规模显存不够跑。
- 标准 reasoning / 通用任务（MMLU、HumanEval 等）与同 active-param 的 Mixtral 8×7B 接近。

## 对 16B MoE 设计的启示

**主要结论：Jamba 的 hybrid (Mamba + Attention) 路线在 16B 规模上不推荐采用。**

- **不建议引入 Mamba**：
  - Mamba kernel / training stack 仍属于"半小众"，工程债比 softmax attention 大得多（量化、长 context 训练稳定性、与 FlashAttention/SDPA 生态不兼容）。
  - 16B 规模上 KV cache 压力远没有 50B+ 那么大；GQA / MLA 已经够。
  - Mamba 在 reasoning / agent 任务上的 reliability 仍不如纯 transformer，开源社区已有不少 case 显示长 CoT 时 SSM 路线掉点。
- **可借鉴的设计点**：
  1. **MoE 每 N 层一次（而不是每层）**：Jamba 的 "MoE every 2 layers" 是一个有效的稀疏化手段，能降低通信成本。16B 规模可以考虑 — 但 DeepSeek-V3 / Qwen3 路线（每层都 MoE + 少 active expert）已经验证更好，主流仍是后者。
  2. **Attention ratio < 1**：如果未来真要做 hybrid，1:7 是一个被验证过的比例。
  3. **单卡部署目标**：Jamba 提醒：架构选择应该围绕"目标部署形态"做。16B 应该明确目标卡（如 1×H100 fp8、2×A100 等）再决定 attention / KV cache 设计。
- **与 MiniMax-01 的对比**（重要）：
  - MiniMax-01：Lightning Attention (线性 attention) + Softmax，7:1，80 层 / 456B
  - Jamba：Mamba (SSM) + Attention，7:1，32 层 / 52B
  - 共同点：都是 **"长 context 友好的 sub-quadratic 模块 ×7 + 1 个 softmax 锚点"**，比例一致很有意思 —— 暗示 1/8 ~ 1/4 的 softmax 是 retrieval 的最小代价点。
  - 差异：Mamba 状态固定大小（不随 seq 长扩），Lightning 状态也是常数；二者对长 context 都友好。MiniMax 选 Lightning 主要是它和 transformer 生态更兼容（kv-cache 接口、tensor parallel 更直接）。

## Caveats / 局限

- 原论文 (2403.19887) 只发布了 52B/12B；398B/94B 是 Jamba-1.5 (2408.12570) 才出现。本任务编号是原论文，但用户提到的 398B/94B 数字来自 1.5。
- Mamba-1 在某些 retrieval / multi-hop reasoning 上仍弱；Mamba-2 / SSD 已出现但 Jamba 没切换。
- Training token 数、数据组成未公开（商业模型）。
- AI21 的工程实现 / inference framework 未完全开源，复现门槛比 DeepSeek 系列高。
- MoE 路由是经典 top-k + aux loss，没有 DeepSeek-V3 的 loss-free balance 创新；可能在 expert utilization 上有改善空间。
