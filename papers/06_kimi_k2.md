# Kimi K2: Open Agentic Intelligence

- **arXiv**: 2507.20534 (v1 submitted Jul 28, 2025；v2 revised Feb 3, 2026)
- **机构**: Moonshot AI (月之暗面 / Kimi 团队)
- **发表时间**: 2025-07-28
- **作者(代表)**: Kimi Team（共 199+ 作者），包括 Yifan Bai、Yiping Bao、T.Y. Liu 等
- **代码 / 权重**: huggingface.co/moonshotai/Kimi-K2-Base & Kimi-K2-Instruct

## TL;DR

Kimi K2 是 Moonshot AI 推出的开源 **1.04T 总参 / 32.6B 激活** MoE 基础模型，专注 agentic intelligence。核心贡献：

1. **MuonClip 优化器** = Muon + QK-clip，把 Muon 的 token-efficiency 优势延伸到 1T 参数规模，做到 **15.5T tokens 零 loss spike**。
2. **超高稀疏度**：384 experts、top-8、1 shared，稀疏度 48（vs DeepSeek-V3 的 32）；通过 scaling-law 验证 "稀疏越高，FLOPs 越省"。
3. **Attention 减半**：64 heads (vs V3 的 128)，节省 128K context 下 ~83% 的 inference FLOPs，仅 0.5–1.2% loss 损失。
4. **Agentic 后训练**：大规模合成工具调用数据 (3000+ 真实 MCP + 20000+ 合成 tool)，验证 RL + self-critique rubric。

## 关键架构配置

| 项 | 值 |
|---|---|
| 总参 / 激活 | 1.04T / 32.6B |
| Layers | 61 |
| Hidden size | 7,168 |
| FFN intermediate (dense layers) | 18,432 |
| MoE intermediate (expert dim) | 2,048 |
| Num attention heads (Q) | 64 |
| Num KV heads | 64（MLA，KV 实际由 latent 投影出，所以 KV heads 与 Q heads 同数） |
| **Attention 类型** | **MLA (Multi-head Latent Attention)** |
| q_lora_rank | 1,536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| N_routed experts | 384 |
| Top-K | 8 |
| N_shared experts | 1 |
| 稀疏度 (sparsity ratio) | 48（384÷8） |
| 激活比例 | 8/384 ≈ 2.1% |
| **首层 dense** (`first_k_dense_replace`) | **1**（对比 DeepSeek-V3 = 3） |
| Expert grouping | **No**（取消 V3 的 expert grouping） |
| Vocab size | 163,840 |
| Context length (pretrain → extend) | 4K → 32K → 128K（YaRN, factor 32×） |
| RoPE θ | 50,000 (with YaRN) |
| Position encoding | RoPE (partial; 64 dim) |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| **MTP** | **❌ 论文未提，HF config 中也无 MTP head**（与 V3 不同） |
| Tokens | 15.5T pretrain + 460B 退火 |
| 优化器 | **MuonClip** (Muon + QK-Clip) |
| 精度 | BF16 params, FP32 grad buffer, **FP8-E4M3 activation in 1×128 tiles with FP32 scales** |

## 核心方法 / 创新点

### 1. MuonClip = Muon + QK-Clip

#### Muon 优化器（基础）

Muon (Jordan et al. 2024) 用 Newton-Schulz iteration 对动量 `M_t` 做正交近似，再做参数更新：

```
M_t  = μ·M_{t-1} + G_t                          # 标准动量
O_t  = NewtonSchulz(M_t)                        # 正交化（SVD 近似）
W_t  = W_{t-1} - η·(O_t + λ·W_{t-1})            # 加 weight decay
```

为了让不同形状矩阵更新幅度一致，乘上 `max(n,m)·0.2` 的 RMS scaling（匹配 Adam 的 RMS）。

> "Muon substantially outperforms AdamW under equivalent compute and model size"

→ Moonshot 选 Muon 而不是 AdamW 是为了在数据稀缺时拿到更高 token efficiency。

#### QK-Clip（修复 Muon 的稳定性 bug）

实验发现：vanilla Muon 在大模型下 **attention max logit 会迅速涨到 >1000**，导致训练崩溃。QK-Clip 是一个 **post-update weight rescaling**：

设每个 head h 的最大 attention logit 是 `S_max^h`，阈值 `τ = 100`。每步更新后计算：

```
γ_h = min(1, τ / S_max^h)
```

把该 head 的 Q/K projection 权重 rescale：

- `q^C`, `k^C`（标准 head 部分）：weight ×√γ_h
- `q^R`（head-specific rotary 部分）：weight ×γ_h
- `k^R`（shared rotary 部分）：**不动**

特点：
- 不修改 forward/backward 计算图，只在 weight 上做事后 clip
- 只在某 head 真的超过 τ=100 时才生效（γ_h<1），稳定后 γ_h=1 等价于 noop
- 训练曲线：max logit 在前 30% 步数保持 ~100，之后自然衰减，**全程 15.5T tokens 零 loss spike**

#### 设计取舍

QK-Norm（forward-time normalize Q/K，如 Qwen3、Ling 2.0）也能控制 logit，但会引入 forward overhead；QK-Clip 是事后修正、零 forward cost、只在异常时生效。

### 2. 稀疏度 scaling law → N=384

K2 系统性验证了 **"稀疏度 = N_routed / top-K"** 对 compute efficiency 的影响：

| Sparsity | FLOPs at validation loss 1.5 |
|---|---|
| 8 | 1.69× (baseline) |
| 16 | 1.39× |
| 32 | 1.15× |
| **48 (K2)** | **1.0× (best)** |

→ K2 选 sparsity=48 → 384 experts / top-8。
> "Increasing the total number of experts (i.e., increasing sparsity) consistently lowers both the training and validation loss."

DeepSeek-V3 是 256/8 = sparsity 32；K2 直接把这个旋钮拧到 48。

### 3. Attention heads 从 128 → 64

DeepSeek-V3 用 128 heads，K2 把它减半到 64。论文说明：

- 把 heads 从 64 → 128 在 128K 上下文下 **推理 FLOPs +83%**
- 但 validation loss 仅改善 **0.5%–1.2%**
- 对 agentic 长上下文场景，这点 loss 不值得这么大的推理成本

→ K2 的最终选择：64 heads。Hidden=7168，head_dim=128 → 64 heads 是自然的整除。

### 4. 取消 Expert Grouping

V3 把 256 个 experts 分成 8 组 × 32，top-K 路由先选 4 组再在组内选 8 个 expert。K2 直接取消分组，384 个 expert 走 flat top-8 routing。

理由：EP (expert parallelism) 用 16-way，每个 EP rank 装 24 个 expert，已经足够小，不需要 grouping 来限制跨节点通信。同时取消 group 让 router 选择更灵活。

### 5. Dense 前缀只 1 层（vs V3 的 3 层）

`first_k_dense_replace = 1`。原因：减少 inference 开销，剩余 60 层全部走 MoE 给更多稀疏 capacity。

## 训练 & 系统细节

### 数据 (15.5T tokens 主训 + 460B 退火)

四大领域：web text / code / math / knowledge。关键数据策略：

- **学习笔记式 (learning-note) rephrasing**: math
- **Chunk-wise autoregressive 重生成 + 保真度检查**: knowledge

数据效率消融（SimpleQA Acc）：

| 策略 | Accuracy |
|---|---|
| Raw 重复 10 epoch | 23.76% |
| 1× rephrase + 10 epoch | 27.39% |
| **10× rephrase + 1 epoch** | **28.94%** |

→ 多次 rephrase 单次训 > 单次 rephrase 多次训 > 直接重复。

### WSD 学习率 schedule

| 阶段 | Tokens | LR |
|---|---|---|
| Warmup | 500 step | linear → 2e-4 |
| Stable (Phase 1) | 10T | constant 2e-4 |
| Cosine decay (Phase 2) | 5.5T | 2e-4 → 2e-5 |
| Annealing | 460B | 2e-5 → 7e-6 |

- Global batch size: **67M tokens**
- Weight decay: **0.1** 全程
- Context: 4K pretrain → 32K (60B tokens) → 128K (YaRN, factor 32)

### 精度
- Params: BF16
- Grad buffer: FP32
- Activations: FP8-E4M3, 1×128 tiles + FP32 scales

### Expert Parallelism
- EP=16，每 rank 装 24 expert
- 取消 expert grouping，因 EP 已足够小

## 关键消融与结果

### 稳定性
- **零 loss spike 训完 15.5T tokens**（Figure 3 显示 unsmoothed per-step loss curve）
- Vanilla Muon 跑同设置 → max attention logit > 1000，崩溃
- QK-Clip (τ=100) 后 → max logit 平稳，30% 步后自然降下来

### Token Efficiency
- 同 compute / 同模型，Muon > AdamW（论文说 "substantially outperforms"）
- Rephrase 策略让 1.04T MoE 用 15.5T tokens 就够（vs Llama 3 405B 用 15T，K2 拿到更好 reasoning）

### Sparsity Scaling
- Sparsity 48 比 32 在同 valid loss 上 FLOPs −13%
- 比 sparsity 8 省 FLOPs 41%

### Post-training（Agentic）
- **SFT 也用 Muon**（不是 AdamW）
- 合成 agentic data：3,000+ 真实 MCP tools + 20,000+ synthetic tools，生成数万条 tool-use trajectories
- **RL recipe**: verifiable rewards (math/code) + self-critique rubric rewards (subjective)；budget control + PTX loss

## 对 16B MoE 设计的启示

1. **MuonClip 是 16B MoE 训练的强候选**。Muon 的 token efficiency 优势在小模型同样适用；QK-Clip 是低成本的稳定性保险，几乎零 overhead。如果想用 Muon，QK-Clip 几乎必带。

2. **稀疏度可以推到 48 甚至更高**。对 16B/1.x B 激活的 MoE（比如 16B 总/ 1.5B 激活，sparsity = 100+），K2 的实验证明 "稀疏越高越省 FLOPs" 在 1T 规模仍成立，对 16B 同样大概率有效。

3. **MLA 在 1T 规模仍是合理选择**。K2 把 V3 的 MLA 完整继承下来（q_lora=1536, kv_lora=512, qk_nope=128, qk_rope=64, v=128），证明 MLA 在 trillion 规模可行。但对 16B 模型，MLA 的复杂性可能不划算，GQA 更简单。

4. **Attention heads 选择上要权衡推理 FLOPs**。对推理为主的 16B 模型，head 数量不一定越多越好；K2 的 "64 heads 比 128 heads 推理省 83%、loss 只差 1%" 是关键 lesson。对 16B/hidden=2048，建议 head_dim=128 → 16 Q heads。

5. **Rephrase 数据 > 重复数据**。SimpleQA 实验直接可移植：宁可多花 LLM rephrase 成本生成多个版本，也不要 raw repeat 10 次。

6. **WSD schedule + 大 stable phase 是 MoE 友好**。K2 的 10T constant LR + 5.5T cosine decay 是 MoE 训练的好模板。

7. **Expert grouping 不是必须**。当 EP 度小（≤16）时可以取消分组，flat top-K 路由更灵活。

8. **MTP 在 K2 中不存在**。和 V3 不同，K2 选择不带 MTP head。对 16B MoE，MTP 是 inference acceleration 选项，可单独考虑。

## Caveats / 局限

- **MLA 实现细节**（具体 norm 位置、low-rank decomposition 公式）在 v1 中沿用 V3，未独立详述。
- **Post-training data 配比 / RL 详细 reward 公式**部分披露但未给完整数据集。
- **MTP 是否被刻意排除**：论文未明说原因，HF config 中也无 `num_nextn_predict_layers`。这可能是因为 K2 主要面向 agentic / chat，对 MTP 的吞吐增益不如 V3 那么关键。
- **训练硬件 / 集群规模未披露**（推测 H800/H100 大集群，但论文未明确说）。
- **稀疏度 vs 路由复杂度**：N=384 的路由开销在小模型上未必划算（K2 实测在 1T 总参才显著）。
- **QK-Clip 阈值 τ=100 来源**：论文没有给出 τ 选择的精确公式或 ablation，只说 "vanilla 跑下来会到 >1000"，所以选 100 作为上限。
