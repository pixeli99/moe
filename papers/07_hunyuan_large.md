# Hunyuan-Large: An Open-Source MoE Model with 52B Activated Parameters by Tencent

- **arXiv**: 2411.02265 (v1 submitted Nov 4, 2024)
- **机构**: Tencent Hunyuan（腾讯混元）
- **发表时间**: 2024-11-04
- **作者(代表)**: Tencent Hunyuan Team
- **代码 / 权重**: huggingface.co/tencent/Tencent-Hunyuan-Large

## TL;DR

Hunyuan-Large 是腾讯混元在 2024 年底发布的**当时最大开源 Transformer MoE 模型**（389B 总参 / 52B 激活）。技术亮点：

1. **粗粒度 + top-1 routing**：仅 16 个 specialized expert，每 token top-1 + 1 shared expert（极简路由，与 DeepSeek/Kimi 的 fine-grained 路线相反）。
2. **CLA (Cross-Layer Attention)**：每 2 层共享 KV cache，叠加 GQA，整体把 KV cache 压缩到原 MHA 的 ~5%。
3. **Expert-specific learning rate**：shared / specialized experts 用不同 LR（比例 ~0.31）来纠正 effective batch size 差异。
4. **Recycle routing**：替代 token dropping，让超载 expert 的 token 被随机分配到其它有 capacity 的 expert。
5. **MoE scaling law**：自有公式 `C ≈ 9.59·N·D + 2.3×10⁸·D`，确定 52B 激活 + 7T tokens 的设计点。

## 关键架构配置

| 项 | 值 |
|---|---|
| 总参 / 激活 | 389B / 52B |
| Layers | 64 |
| Hidden size | 6,400 |
| FFN intermediate (dense layers) | 18,304 |
| Expert intermediate dim | 论文未明确披露（推测同 18,304 或类似）|
| Num attention heads (Q) | 80 |
| Num KV heads (GQA) | 8 |
| Head dim | 80 (= 6400/80) |
| **Attention 类型** | **GQA + CLA（Cross-Layer Attention）** |
| **CLA share factor** | **2 (每 2 层共享一次 KV)** |
| N_routed (specialized) experts | **16** |
| Top-K | **1** |
| N_shared experts | **1** |
| 激活比例 | (1+1)/17 ≈ 11.8% |
| Vocab size | 129,024 |
| Context length | 256K |
| RoPE θ | 10,000 (with dynamic scaling, α=1000) |
| Position encoding | RoPE |
| Normalization | RMSNorm (eps=1e-5) + **QK-Norm (`use_qk_norm: true`)** |
| Activation | SiLU/SwiGLU |
| MTP | ❌ 未提 |
| Routing | `use_mixed_mlp_moe: true`（mixed routing strategy） |
| Capacity factor | 1.0 |
| Drop tokens | **false**（用 recycle routing 代替） |
| Tokens | **7T** 主训 (含 ~1.5T 合成) + 长上下文阶段 |
| 优化器 | AdamW（论文未给完整超参） |
| 精度 | BF16（HF 权重） |

## 核心方法 / 创新点

### 1. Cross-Layer Attention (CLA)

**动机**：长上下文（256K）下 KV cache 主导显存。CLA 思想：相邻两层的 KV 完全共享，只在偶数层计算 KV、奇数层重用。

**叠加 GQA 后的 KV 内存**：
- 原 MHA: `2·n_h·d_h·L = 2·80·80·64 = 819,200` 参数/token
- GQA-only: `2·n_g·d_h·L = 2·8·80·64 = 81,920`（÷10）
- **GQA + CLA**: `2·n_g·d_h·(L/2) = 40,960`（再÷2）
- **总压缩比**: ~5% 的 MHA KV cache → 节省 ~95%

实现上：偶数层正常计算 Q/K/V，奇数层只算 Q，K/V 直接从上一层复用。注意 Q 不能跨层共享（否则破坏 token-level diversity）。

### 2. Mixed routing strategy（shared + specialized）

| 类型 | 数量 | 触发 |
|---|---|---|
| **Shared expert** | 1 | 所有 token 都会过 |
| **Specialized experts** | 16 | top-1 routing |

每个 token 实际经过 1 shared + 1 specialized = 2 个 expert，总激活 ~52B（含 attention/dense 部分）。

**与 DeepSeek-V3 / Kimi K2 对比**：
- HY-Large: 16 expert × top-1 = 粗粒度
- V3: 256 × top-8 = 细粒度
- K2: 384 × top-8 = 超细粒度

粗粒度的优势：路由开销小、推理 batch 内 expert 复用率高；劣势：专家专业化深度不够。这也解释了为什么 HY-Large 需要 shared expert 提供 "通用 capacity"。

### 3. Recycle Routing

普通 MoE 在 expert 达到 capacity 上限后，多余 token 会被 drop。Hunyuan-Large 不丢弃：

> "an additional random allocation for tokens originally routed to overloaded experts to other specialized experts which have not exceeded their capacity"

机制：超载 expert 的剩余 token 被**随机重新分配**到其它仍有 capacity 的 expert。论文：
> "preserves vital information while simultaneously optimizing training efficiency"

`moe_drop_tokens=false`、`moe_random_routing_dropped_token=false`（HF config），与论文一致。

### 4. Expert-Specific Learning Rate

观察：shared expert 被所有 token 用 → effective batch size = B；specialized expert 平均只见 B/n 个 token（n=16）→ effective batch size 不同。按经典 LR-batch scaling，需要给两者不同的 LR：

- Shared expert: `ε_opt(B)` （optimal LR at full batch）
- Specialized expert: `ε_opt(B/n) ≈ ε_opt(B) × 0.31` （n=16）

比例 ~0.31 来自他们的 LR scaling law 实验。这个策略**实际上是 aux-loss-free 的另一种形式**：通过 LR 调节而非 bias 调节专家利用率。

### 5. MoE Scaling Law

HY-Large 提出自己的 compute budget 公式（结合 attention + MoE 项）：

```
C ≈ 9.59 · N_act · D  +  2.3 × 10⁸ · D
```

其中 N_act 是激活参数（不含 embedding），D 是训练 tokens。

通过 10M–1B 激活、10B–100B tokens 的 scaling 实验，拟合出：
- 系数 N_c = 5.9×10⁻³，指数 α = 0.5305
- D 指数 β = 0.50
- **预测最优激活参数: 58.1B → 选 52B**（"smoothness" 取整）
- **预测最优训练 tokens: 5.6T → 选 7T**（"cost-efficiency"）

### 6. 大规模合成数据

7T 总 tokens 中 ~1.5T (~21%) 是合成的。4 步流程：

1. **Instruction generation**：从种子源生成 instruction
2. **Instruction evolution**：改 clarity / 扩 domain / 提 difficulty
3. **Response generation**：用专业 teacher model 生成回答
4. **Response filtering**：用 critic model + consistency check 过滤

合成数据重点：**数学、代码、低资源语言、高教育价值**领域。

## 训练 & 系统细节

### 数据 (7T tokens)
- ~75% 普通长度 + ~25% 长文本 (books, code)
- ~1.5T 合成
- 长上下文分两阶段：32K → 256K，每阶段约 10B tokens

### 训练硬件 & 优化器
- 论文未公开 GPU 数 / FLOPs
- 优化器: AdamW + load balance losses（具体 α 未给）
- 精度: BF16（HF 配置确认）
- 论文未提 FP8

### Long-Context
- 32K → 256K，约 10B tokens / 阶段
- RoPE dynamic scaling type, α=1000

### Post-Training
- 大规模 SFT + RLHF（论文较少披露 RL 细节）

## 关键消融与结果

### KV cache 压缩
- GQA + CLA → KV cache 减少 ~95%（vs full MHA）
- 在 256K 长上下文显存预算上有决定性意义

### Scaling Law 预测
- 最优 N_act = 58.1B（实选 52B），D_opt = 5.6T（实选 7T）
- 拟合误差极小，证明 MoE scaling law 在 ~400B 总参规模仍成立

### 性能对标
- **打过 Llama 3.1 70B**：在 language understanding / reasoning / math / coding 上
- **与 Llama 3.1 405B 相当**：用 52B 激活打 405B 全激活，是 "MoE > dense" 的强证据

### Recycle routing
- 论文称比 drop-token 在训练稳定性和性能上都更好（但未给精确数字）

## 对 16B MoE 设计的启示

1. **粗粒度路由的极简方案**。Hunyuan-Large 用 16 expert + top-1 + 1 shared 证明粗粒度路由也能 work。对于 16B MoE，如果不想用 256+ 细粒度路由，N=16 / top-1 / 1 shared 是另一条路线（路由开销极小、kernel 简单）。

2. **CLA 是显著的长上下文显存优化**。如果 16B MoE 设计目标包括长上下文 (32K+) 部署，CLA 共享每 2 层 KV → 显存减半，几乎免费（论文 ablation 表示精度损失很小）。可作为 GQA 的正交增强。

3. **Expert-specific LR 是 aux-loss-free 的替代视角**。从优化器角度纠正 effective batch size 差异，shared:specialized = 1:0.31。对 16B MoE 中同时有 shared expert 和 specialized expert 的设计，可以采纳这种 LR 分组策略。

4. **Recycle routing > token dropping**。不丢 token 的实现简单（超载 token 随机分到非超载 expert），且训练效率/精度更好。比设置 capacity_factor + drop 简单。

5. **粗粒度路由对 shared expert 几乎是必须**。N=16 的 specialized 专家容易因为 routing 不均、单 expert 容量过大而过拟合细分 domain；shared expert 提供 "通用 fallback"。这正是 Qwen3 "no shared expert" 路线只在 N=128 细粒度下成立的原因。

6. **MoE Scaling Law 的实际形式**：`C ≈ 9.59·N·D + 2.3e8·D` 是一个可以直接拟合的模板。16B MoE 项目里跑 ~10 个小规模 anchor（如 50M、200M、1B 激活 × 5–10B tokens）就能验证 scaling 趋势。

7. **合成数据 ~20–25% 是可承受比例**。Hunyuan-Large 1.5T/7T ≈ 21% 合成，证明大比例合成仍能稳定训练（需要 4 步生成-evolve-生成-过滤 pipeline）。

## Caveats / 局限

- **Expert intermediate dim 未公开**：论文 + HF config 中难以直接读到 expert 内部 FFN 维度。
- **Load balance loss 具体公式未给**：只说 "load balance losses"。
- **CLA 的精度损失消融数据有限**：论文给出 GQA+CLA 节省 ~95% KV cache，但 ablation 表中具体 perplexity 退化未充分披露。
- **训练硬件 / FLOPs / GPU hours 未披露**。
- **粗粒度路由可能不是最优**：对比后续 Kimi K2 的 sparsity-48 scaling，HY-Large 的 sparsity-8 路线在 frontier 上可能不是最优；论文出版时间 (2024 Nov) 早于 sparsity-scaling 文献，可视为一种保守工程选择。
- **没有 MTP / FP8**：这些新技巧在 2024-11 还不流行；最新 MoE (Kimi K2 / Ling 2.0) 都会用。HY-Large 没用。
- **Mixed routing 实现细节简略**：`use_mixed_mlp_moe=true` 的具体语义在论文中未对应到公式，只能从 HF code 反推。
