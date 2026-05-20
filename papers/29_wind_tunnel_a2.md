# Wind Tunnel A2 实验矩阵 — 1B total / 200M active / 25B tokens

> **目的**：把 22_FINAL_16B_design §8 提到的 A2 消融具体到 **每个 arm 的超参表、决策门槛、算力账、依赖顺序**，外加这次调研新发现的 3 个开放变量（ε / 优化器 / 路由派系）。
> **范围**：仅 A2 (1B/200M/25B)；A0 / A1 / A3 / A4 在 §10 给出最小骨架。
> **算力**：A2 一个 arm ≈ 21 H100-hours（200M active × 25B tokens × 6 FLOPs/param × 40% MFU on H100）。9 个消融 × 3 arms 平均 ≈ **27 arms ≈ 570 H100-hrs**，占 wind tunnel 总预算 35K hrs 的 **1.6%**。

---

## 1. TL;DR — 9 个消融、3 个 tier、3 周完成

| Tier | 消融 | Arms | Δ-tokens | Δ-cost (H100-hr) | 决定什么 |
|---|---|---|---|---|---|
| **T1** | T1.1 优化器 ε（1e-8 vs 1e-20） | 2 | 25B | 42 | AdamW 数值精度路线 |
| **T1** | T1.2 优化器（AdamW vs Muon） | 2 | 25B | 42 | 是否走 Moonshot/K2 系 |
| **T1** | T1.3 WSM 末段平均 N（16 / 32 / 64） | 3 | 25B | 63 | scheduler final 段稳定性 |
| **T2** | T2.1 **路由派系**（sigmoid+ALF vs softmax+aux-loss） | 2 | 25B | 42 | **架构最大分水岭** |
| **T2** | T2.2 N_routed（64 / 128 / 256） | 3 | 25B | 63 | 粒度决策 |
| **T2** | T2.3 Profile R/B/M（active 300M / 200M / 100M\* A2-scale） | 3 | 25B | 63 | 取向决策 |
| **T2** | T2.4 0 shared vs 1 shared expert | 2 | 25B | 42 | 共享专家 |
| **T2** | T2.5 ALF variant（V3 sign / Ling 零均值 / Qwen3 global-batch） | 3 | 25B | 63 | 仅 T2.1 sigmoid 派胜出后跑 |
| **T3** | T3.1 MTP（D=0 / D=1 / D=2） | 3 | 25B | 63 | active 2.4B 边界确认 |
| **T3** | T3.2 Block AttnRes（off / N=4 / N=6） | 3 | 25B | 63 | Kimi 残差替代 |
| | **A2 合计** | **27** | **675B** | **~570** | |

\* A2-scale 是把 16B Profile R/B/M 等比缩到 1B 总参的对应 active 配比。

**3 周时间表（8×H100 节点 × 单 arm 串行）**：
- W1：T1（7 arms / 147 H100-hr）→ 锁定 ε / 优化器 / scheduler
- W2：T2.1-T2.4（10 arms / 210 H100-hr）→ 锁定派系 + 粒度 + shared + profile
- W3：T2.5 + T3（9 arms / 189 H100-hr）→ ALF 变种 + 训练辅助

并行化（4 节点 × 8 H100，~7 arms 并发）可压到 1 周。

---

## 2. 共同 A2 baseline（所有 arm 起点）

> 这是"如果不消融就直接训的 1B 模型" — 9 个消融每个都从这套配置出发，只动 1-2 个变量。

| 维度 | A2 baseline | 来源 |
|---|---|---|
| Total params | **1.0 B** (base, 不含 MTP module) | scaling 自 16B Profile B |
| Active params (严格) | **200 M** | 1/5 稀疏比例（与 16B 的 1/6.5 略稠密，因 A2 总参小） |
| Layers | **12** | mHC 3B anchor 同款，缩到 1B |
| Hidden | **1024** | sqrt-scaling from 2048 / hidden_2048_for_16B |
| Head dim | **64** | gpt-oss 同款，A2 不验证 head_dim |
| Attention | **GQA, 8 Q-heads / 2 KV-heads** | hidden 1024 / 64 = 16 Q 全开太多，缩到 8/2 |
| FFN expert intermediate | **704** | 1024×0.69，对齐 V2-Lite 1408/2048 比例 |
| N_routed / Top-K / N_shared | **64 / 6 / 1** | A2 默认（其中 N_routed 在 T2.2 消融） |
| Dense 前缀 | 第 0 层 | 与 16B spec 一致 |
| Vocab | **128 K BBPE** | 与 16B spec 一致（节省一次 sweep） |
| Sequence | **4096** pretrain | 不在 A2 验证 long-context |
| Routing | **sigmoid + ALF** (V3 默认) | T2.1 消融对照 |
| Aux balance | **bias-based ALF γ=0.001** + α=1e-4 seq-aux | T2.5 消融变种 |
| Routed scaling factor | **2.5** | V3 / dots1 / GLM-4.5 共识 |
| Norm | RMSNorm pre-norm + QK-Norm | OLMoE / Qwen3 / dots1 共识 |
| Optimizer | **AdamW (β=0.9/0.95, ε=1e-8, wd=0.1)** | T1.1/T1.2 消融对照 |
| Peak LR | **1.5e-3** | 1B scale，OLMoE / Ling-mini 同档位经验值 |
| Schedule | WSM (warmup 1500 / stable 50% / merge 50%) | T1.3 消融对照 |
| Batch tokens | 1.0 M / step | 25B / 1M = 25K steps |
| Precision | BF16 master | A2 不验证 FP8 |
| MTP | D=0 (off) | T3.1 消融对照 |

**预算**：200 M × 25 B × 6 ≈ **3 × 10¹⁹ FLOPs**；H100 BF16 @ 400 TFLOPs effective → 21 H100-hr/arm。

---

## 3. Tier 1 消融 — 训练机制（先跑、固化）

### T1.1 — 优化器 ε（**这次调研新增**）

| Arm | ε | 备注 |
|---|---|---|
| **A** (baseline) | **1e-8** | OLMoE §4.2.6 推荐；dots1 / Qwen3 / Hunyuan-Large 均用 |
| B | **1e-20** | mHC 3B/9B/27B 实测；V3 / V2 全家一致；Ling-mini-2.0 亦用 |

**Hypothesis**：ε 小 12 个数量级在大 active scale (V3 37B) 下让 AdamW 二阶矩更稳定；在 200M active scale 上是否显著未知。
**决策指标**：25B token 处 loss + 最后 5B token gradient norm 标准差
**Decision threshold**：
- 若 |loss\_A − loss\_B| < 0.005 且 grad\_std 差异 < 10% → 选 A (ε=1e-8，社区主流，不引入风险)
- 若 loss\_B − loss\_A ≤ -0.01 或 grad\_std\_B/grad\_std\_A < 0.7 → 选 B (ε=1e-20)
**风险**：ε=1e-20 在 fp32 master 下接近 denormal 边界；监控 NaN
**依赖**：无（最先跑）

### T1.2 — 优化器选择

| Arm | Optimizer | 备注 |
|---|---|---|
| **A** (baseline) | **AdamW** β=0.9/0.95 | 行业默认 |
| B | **Muon** (Moonshot/K2) | K2/Moonlight 验证；OLMo-2 已开源 Muon kernel |

**Hypothesis**：Muon 在 K2 1T 训练上展示了显著优势（论文 Table 2，pretrain loss -0.03），200M active 上是否仍正向是开放问题。
**决策指标**：25B token 处 loss
**Decision threshold**：
- 若 loss\_B − loss\_A ≤ -0.01 → 切 Muon（同时引入 MuonClip post-update QK-Clip 替代 QK-Norm）
- 若 ≥ -0.005 → 保留 AdamW（成熟度高、ops 风险低）
**风险**：Muon orthogonal-momentum 实现复杂度；引入需替换 QK-Norm 为 MuonClip
**依赖**：在 T1.1 之后用 T1.1 winner 的 ε 跑

### T1.3 — WSM 末段平均 N

| Arm | N (末段平均权重数) | 备注 |
|---|---|---|
| **A** (baseline) | **N=32** | Ling 2.0 §2.4 推荐 |
| B | N=16 | 短窗，更快响应 |
| C | N=64 | 长窗，更稳但延迟 |

**Hypothesis**：WSM 末段 weight-averaging 窗口直接决定 final ckpt 的 loss / downstream 表现，Ling 报告 N=32 比 WSD 多 1-2 分。
**决策指标**：merge 段最后 10B token loss 的 EMA + HellaSwag dev set acc
**Decision threshold**：选 |loss| 最低 + acc 最高的；如果三者差异 < 0.003 loss / 0.5% acc，按算力倒序选 N=32。
**依赖**：T1.1 + T1.2 winner

---

## 4. Tier 2 消融 — 架构核心（最关键的 5 个）

### T2.1 — 路由派系：sigmoid+ALF vs softmax+aux-loss（**这次调研新增**）

| Arm | Gate | Balance | 灵感来源 |
|---|---|---|---|
| **A** (baseline) | **sigmoid** + scaling=2.5 | **ALF bias** (γ=0.001) + α=1e-4 seq-aux | V3 / K2 / Ling 2.0 / GLM-4.5 |
| B | **softmax** + scaling=1.0 | aux-loss α=0.01 | Mixtral / Qwen3-30B / Hunyuan-Large / OLMoE / gpt-oss |

**Hypothesis**：根据 28_open_source_moe_catalog §3.4 统计，sigmoid+ALF 占 2025+ 新模型 21%（且占增量主体），但 softmax+aux 占总量 53% 且包括 OLMoE / Qwen3 / gpt-oss 等强模型。**两条路线都被 SOTA 验证**，1B/25B 上谁更优是开放问题。
**决策指标**：
- 主指标：25B 处 valid loss
- 副指标：路由熵 H(g)、active expert 占比、MaxVio (max load / min load - 1)、HellaSwag acc
**Decision threshold**：
- loss 差 ≥ 0.005 → 选低者
- loss 差 < 0.005 但 sigmoid 派 MaxVio 比 softmax 派低 ≥ 20% → 选 sigmoid（balance 更好）
- 否则按 default 选 sigmoid+ALF（与 16B spec 一致）
**风险**：softmax+aux 的 dropless+EP=8 实现略不同；要确保 kernel 都对
**依赖**：T1 winner

### T2.2 — N_routed（粒度决策）

| Arm | N_routed | Top-K | Active (严格) | 备注 |
|---|---|---|---|---|
| **A** (baseline) | **64** | 6 | 200M | V2-Lite / Moonlight / DeepSeekMoE-16B 同款 |
| B | 128 | 12 | 200M | Qwen3-30B-A3B 同档 |
| C | 256 | 24 | 200M | Ling-mini-2.0 路线（fine-grained 极致） |

**关键**：每个 arm 调整 `d_expert` 使 active 严格保持 200M（B arm `d_expert ≈ 352`，C arm `d_expert ≈ 176`）。这样三个 arm 同 active 同 total（1B），只在 expert 粒度上变化。
**Hypothesis**：Krajewski 2024 fine-grained scaling + Ling 2.0 都指向更细更好；但 256 在 EP=1 单节点上 dropless kernel 开销更大。1B 规模上 256 是否仍正向是工程问题。
**决策指标**：loss + MMLU acc + MaxVio + throughput (tok/s)
**Decision threshold**：
- 若 (A→B→C) loss 单调下降且 |Δ| ≥ 0.005 per step → 选最细的 C
- 若 throughput\_C / throughput\_A < 0.85 → 不接受 C
- 主推 B (128/12)；选 A 的条件是 B/C 都不显著优于 A 且 throughput 差异 < 5%
**依赖**：T2.1 winner

### T2.3 — Profile R/B/M（active scale）

| Arm | Active | N_routed | Top-K | Total | 对应 16B 取向 |
|---|---|---|---|---|---|
| **A** (Profile B / baseline) | **200 M** | 64 | 6 | 1.0 B | Balanced (V2-Lite 等比缩) |
| B (Profile R) | **280 M** | 64 | 8 | 1.2 B | Reasoning-leaning (DeepSeekMoE 16B/2.8B 等比) |
| C (Profile M) | **80 M** | 256 | 8 | 1.0 B | Memorization (Ling-mini-2.0 等比) |

**Hypothesis**：Yokota 2025 的方向性证据 — reasoning 偏好更大 active + 更大 K + 不要过稀疏。1B/25B 是否还能复现 reasoning vs memorization 的分裂是开放问题。
**决策指标**：综合得分 = 0.5×loss + 0.5×(MMLU + GSM8K + HumanEval avg)
**Decision threshold**：
- 若 Profile R 在 reasoning bench 上 ≥ +1.5pt 且 loss 不差超过 0.005 → 16B 切 Profile R
- 若 Profile M 在 (loss + 综合) 上不输 ≥ -0.002 且 throughput 高 ≥ 10% → 16B 切 Profile M
- 否则默认 Profile B
**依赖**：T2.1 winner

### T2.4 — 0 vs 1 shared expert

| Arm | N_shared | d_expert (调整以保 active 不变) | 备注 |
|---|---|---|---|
| **A** (baseline) | **1** | 704 | V3 / K2 / Ling 2.0 / GLM-4.5 |
| B | 0 | 768 (稍大补齐 active) | Mixtral / OLMoE / Qwen3-30B / gpt-oss |

**Hypothesis**：28_catalog 统计 1-shared 与 0-shared 各占 36%，是公开文献最大分歧。OLMoE §3 主张 0 shared 在 1B/7B 上更好；DeepSeek/Ling 系坚持 1 shared 因 V3 在 14.8T 上表现优秀。**直接 head-to-head 没有任何论文做**。
**决策指标**：loss + MMLU + path coverage (每个 routed expert 至少被 ≥ 1% token 选择)
**Decision threshold**：
- |loss\_A − loss\_B| < 0.003 → 选 A (1 shared，与 16B spec 一致)
- loss\_B 更低且 ≥ 0.005 → 切 0 shared
- 若 0 shared 出现 ≥ 5 个 dead expert (token < 0.1%) → 不接受 B
**依赖**：T2.1 winner

### T2.5 — ALF variant（仅在 T2.1 sigmoid 派胜出后跑）

| Arm | Bias 更新规则 | 备注 |
|---|---|---|
| **A** (baseline) | **V3 sign rule**：`b_i ← b_i + γ·sign(load_i - target)` | V3 §4.2 / dots1 同款 |
| B | **Ling 零均值**：`b_i ← b_i + u·(sign(e_i) − mean(sign(e)))` | Ling 2.0 §3 |
| C | **Qwen3 global-batch**：aux α 在 batch 级别（非 sequence）+ ALF bias | Qwen3 系 |

**Hypothesis**：dots1 在 11.2T tokens 上不用零均值修正也稳，说明 V3 原版充分；Ling 零均值是否带来边际改善 ≥ 0.002 loss 未知。
**决策指标**：loss + MaxVio_global 漂移幅度（最后 2B token bias 均值的标准差）
**Decision threshold**：
- bias 漂移 |mean(b_i)| > 1.0 at 25B tokens 的 arm 拒绝（说明 V3 原版漂得严重 → 选 Ling）
- 若三个 arm bias 都收敛 (|mean| < 0.5) 且 loss 差 < 0.003 → 选 A (V3 简单)
**依赖**：T2.1 sigmoid+ALF arm 胜出

---

## 5. Tier 3 消融 — 训练辅助（最后跑、可丢）

### T3.1 — MTP D 维度

| Arm | D (MTP depth) | 训练 loss 加权 λ | 推理形态 |
|---|---|---|---|
| **A** (baseline) | **0** (off) | – | 标准 decoding |
| B | 1 (V3 causal chain) | 0.3 → 0.1 (token 60% 处衰减) | 主 head + 可丢 / self-spec |
| C | 2 (V3 chain D=2) | 0.3 → 0.1 → 0.05 | 同上 + 二级丢弃 |

**Hypothesis**：23_mtp_investigation 已结论 2.4B active 是 boundary case。A2 的 200M active 比 boundary 还低 12×，MTP 大概率在 A2 拖累 loss。**A2 跑 MTP 的目的不是验证它在 A2 有效，而是建立 D=0/1/2 在 200M 上的 baseline gap，外推到 16B 时校准期望**。
**决策指标**：25B 处 main-head loss（MTP head 不算）
**Decision threshold**：
- 若 main-head loss B/C - A ≤ 0.005 → 16B 可以放心带 D=1
- 若 ≥ 0.015 → 16B 不要带 MTP (跨 active 12× 后差距不一定收敛)
- 若 0.005-0.015 → 需要 A3 (600M active) 再确认
**依赖**：T2 全部 winner（baseline 锁定后）

### T3.2 — Block AttnRes（Kimi 残差替代）

| Arm | 残差形态 | 备注 |
|---|---|---|
| **A** (baseline) | **标准 PreNorm** | 行业默认 |
| B | **Block AttnRes N=4** (3 层/块) | Kimi 2603.15031 Block 版 |
| C | Block AttnRes N=6 (2 层/块) | 同上更细 |

**Hypothesis**：26_attention_residuals 在 Kimi 48B/3B + 1.4T tokens 上展示 1.25× scaling + reasoning bench 全胜。1B/25B 上是否能复现是开放问题。
**决策指标**：loss + GSM8K + 推理 latency overhead
**Decision threshold**（22_FINAL §11 已给）：
- loss 改善 ≥ 0.005 **且** GSM8K (或同类 reasoning) ≥ +1.0pt → 16B 纳入
- 否则不纳入（多一层抽象、license CC BY-NC-ND 限商用）
**风险**：Kimi license 限商用，需 legal 确认是否能内部用
**依赖**：T2 全部 winner

---

## 6. 完整实验矩阵（27 arms × 25B tokens × 21 H100-hr/arm）

| ID | Tier | 消融 | Arm | 核心差异 | 算力 (H100-hr) |
|---|---|---|---|---|---|
| 01 | T1.1 | ε | A | ε=1e-8 (默认) | 21 |
| 02 | T1.1 | ε | B | ε=1e-20 (DeepSeek) | 21 |
| 03 | T1.2 | Opt | A | AdamW (默认) | 21 |
| 04 | T1.2 | Opt | B | Muon | 21 |
| 05 | T1.3 | WSM | A | N=32 (默认) | 21 |
| 06 | T1.3 | WSM | B | N=16 | 21 |
| 07 | T1.3 | WSM | C | N=64 | 21 |
| 08 | T2.1 | Route | A | sigmoid+ALF (默认) | 21 |
| 09 | T2.1 | Route | B | softmax+aux | 21 |
| 10 | T2.2 | N\_rt | A | 64 (默认) | 21 |
| 11 | T2.2 | N\_rt | B | 128 | 21 |
| 12 | T2.2 | N\_rt | C | 256 | 21 |
| 13 | T2.3 | Profile | A | B (默认 200M) | 21 |
| 14 | T2.3 | Profile | B | R (280M) | 25 |
| 15 | T2.3 | Profile | C | M (80M, sparser) | 12 |
| 16 | T2.4 | Shared | A | 1 (默认) | 21 |
| 17 | T2.4 | Shared | B | 0 | 21 |
| 18 | T2.5 | ALF | A | V3 sign (默认) | 21 |
| 19 | T2.5 | ALF | B | Ling 零均值 | 21 |
| 20 | T2.5 | ALF | C | Qwen3 global-batch | 21 |
| 21 | T3.1 | MTP | A | D=0 (默认) | 21 |
| 22 | T3.1 | MTP | B | D=1 | 23 (+10%) |
| 23 | T3.1 | MTP | C | D=2 | 25 (+20%) |
| 24 | T3.2 | AttnRes | A | 标准 (默认) | 21 |
| 25 | T3.2 | AttnRes | B | Block N=4 | 22 |
| 26 | T3.2 | AttnRes | C | Block N=6 | 22 |
| 27 | – | Final | – | T1+T2+T3 winners 合一组 + 50B tokens 复跑 | 42 |
| | | | | **Total** | **~582** |

**Note**：每个消融的 baseline arm（A）共享：可以只跑一次，作为所有 ablation 的 anchor。这样实际 arm 数 ≈ 19，算力降到 ~420 H100-hr。**但建议每个消融独立跑 baseline 以避免 cross-contamination**（不同 ablation 在 baseline 上的小差异要明显高于消融 effect 时需要警惕）。

---

## 7. 决策树（消融顺序与门控）

```
┌─────────────────────────────────────────────────┐
│ A0 (200M/30M/5B) smoke test                     │
│ - 路由收敛？loss 曲线无 NaN？kernel 跑通？        │
└────────────────────────┬────────────────────────┘
                         │ pass
                         ▼
┌─────────────────────────────────────────────────┐
│ T1.1 ε  +  T1.2 Optimizer  +  T1.3 WSM           │
│ (并行 3 个，独立)                                 │
│ → 锁定 (ε*, Opt*, WSM_N*)                        │
└────────────────────────┬────────────────────────┘
                         │ T1 winners
                         ▼
┌─────────────────────────────────────────────────┐
│ T2.1 路由派系                                    │
│ → 锁定 sigmoid+ALF 或 softmax+aux                │
└────────────────────────┬────────────────────────┘
                         │
                         ├──── sigmoid → T2.5 ALF variant
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│ T2.2 N_routed  +  T2.3 Profile  +  T2.4 Shared   │
│ (3 个并行)                                       │
│ → 锁定 (N*, Profile*, N_sh*)                     │
└────────────────────────┬────────────────────────┘
                         │ T2 winners
                         ▼
┌─────────────────────────────────────────────────┐
│ T3.1 MTP  +  T3.2 Block AttnRes                  │
│ (2 个并行；都是 optional add-on)                 │
│ → 决定 16B 是否带 MTP / AttnRes                   │
└────────────────────────┬────────────────────────┘
                         │ T3 winners
                         ▼
┌─────────────────────────────────────────────────┐
│ Arm 27 — winners 合一组复跑 50B tokens 验证      │
│ (确保 winning combo 在更长 horizon 仍占优)        │
└────────────────────────┬────────────────────────┘
                         │ confirm
                         ▼
                       A3 (600M)
                         ▼
                       A4 (16B)
```

**关键 gate**：
- A0 不通过：fix bug → 重跑 A0
- T2.1 选错派系 → T2.5 (sigmoid 配套) 不适用，需要回头跑 softmax 派的 aux α 调整消融
- T3 任一负面：当作 optional 不进 16B，但记录回 22_FINAL §11

---

## 8. Telemetry checklist（每个 arm 必须落盘）

> 没记下来的就当没跑过 — 这些指标必须每 500 step 落盘。

| 类别 | 指标 | 用途 |
|---|---|---|
| **训练动态** | loss (train/val) | 主决策指标 |
| | grad norm | 稳定性 |
| | grad clip rate (% of steps) | 触发 clip 比例 → spike 预警 |
| | LR (当前) | scheduler 验证 |
| | optimizer state norm | Muon vs AdamW 对比 |
| **路由健康** | MaxVio = max(load)/min(load) − 1 | balance 质量 |
| | router entropy H(g) | 路由清晰度 |
| | bias `b_i` 均值 + std (仅 ALF) | bias 漂移 |
| | top-1 hit rate per expert | dead expert 检测 |
| | gate score 分布 (FP32) | softmax/sigmoid 区分 |
| **效率** | tokens/sec/GPU | throughput |
| | MFU (model FLOPs utilization) | 算力效率 |
| | EP all-to-all 时间占比 | comm overhead |
| **下游 (每 5B token 一次)** | HellaSwag acc | reasoning lite |
| | MMLU 5-shot | knowledge |
| | GSM8K 8-shot | math reasoning |
| | HumanEval pass@1 | code |
| | C-Eval (中文 CN model only) | 中文能力 |

**Final ckpt 必跑**：MMLU-Redux / GPQA-Diamond / BBH / IFEval（A2 规模上 GPQA 可能噪声大，仅作参考）

---

## 9. 显式不在 A2 验证（避免 scope creep）

> 这些已在 22_FINAL §9 或本调研中决定不引入，A2 不浪费 budget 验证：

| 不验证项 | 在哪里决定的 |
|---|---|
| MLA (DeepSeek 多潜在头注意力) | 22_FINAL §9 — 工程复杂度 vs 收益不划算 |
| FP8 训练 | 22_FINAL §10 — 16B ROI 远不如 200B+；BF16 baseline 优先 |
| Node-limited routing | 25_node_limited_routing — EP=8 单节点拓扑下无意义 |
| Hybrid attention (Mamba / Lightning) | 22_FINAL §9 — 收益主要在 1M+ context |
| HC / mHC (hyper-connections) | 27_mhc §"不推荐用于 16B" — 5 个原因；kernel 不开源 |
| Sparse upcycling | 22_FINAL §9 — 15T tokens 远超 1.2× from-scratch 阈值 |
| MFA / Step-3 AFD | 22_FINAL §9 — 16B 单节点部署不需要 disaggregation |
| Mixtral 风格 N=8 粗粒度 | 22_FINAL §9 + 28_catalog Pattern B 已被取代 |
| LongCat zero-experts | 28_catalog §4 G — 太新（25-09），独立复现不足 |
| `routed_scaling_factor` 数值 (1.0 vs 2.5) | 隐含在 T2.1 派系对比中 |
| RoPE base (1e4 vs 1e6 vs 1e7) | A2 不验证 long-context，固定 1e6 |
| Tokenizer (vocab size / BBPE 变种) | A2 固定 128K BBPE |
| MuonClip vs QK-Norm | 隐含在 T1.2 Muon arm（Muon 胜出自动带 MuonClip） |

---

## 10. A0 / A1 / A3 / A4 简表（A2 上下游）

> 完整 ladder 见 22_FINAL §8；这里给最小骨架。

| Anchor | Total | Active | Tokens | 算力 (H100-hr) | 主要目的 |
|---|---|---|---|---|---|
| **A0** | 200 M | 30 M | 5 B | ~3 | 烧机 / sanity check / kernel 验证 |
| **A1** (5 arms scaling sweep) | 200M→1B | 30M→200M | 16 B 等 | ~50 | scaling law α 系数拟合 |
| **A2** (27 arms 消融) | 1 B | 200 M | 25 B | **~582** | **本文档主题** |
| **A3** (6 arms — A2 winners + alt) | 4 B | 600 M | 80 B | ~1,200 | scaling 外推验证 |
| **A4** (1-2 arms — final spec) | 16 B | 2.4 B | 320 B | ~3,200 | 1/50 main-run sanity check |
| | | | **Total** | **~5,037** | 占 35K wind tunnel 预算 14% |

**剩余 30K H100-hr buffer 用途**：
1. 任一 tier 出现意外 surge / divergence 重跑（保险 10K）
2. A3 上跑 MTP B vs C 二次确认（约 400）
3. A4 单 arm 第二次复跑（3200，仅在主 A4 与 A3 外推不一致时启动）
4. 团队学习曲线 / debug 时间（剩余）

---

## 11. 与其他笔记的交叉

- 主 spec：22_FINAL_16B_design §8（A0-A4 anchor 表）+ §11（强默认但 pilot 必测）
- 决策依据：
  - T1.1 ε ← 27_mhc §"实验模型配置"（DeepSeek 用 ε=1e-20）+ 09_olmoe §4.2.6（OLMoE 用 1e-8）
  - T1.2 Muon ← 06_kimi_k2 + Moonlight HF page
  - T2.1 派系 ← 28_open_source_moe_catalog §3.4 + §4 Pattern A vs B
  - T2.2 N_routed ← 17_finegrained_scaling + 08_ling_2
  - T2.3 Profile ← 21_reasoning_vs_memorization + 22_FINAL §3
  - T2.4 shared ← 22_FINAL §11"强默认但 pilot 必测"列表
  - T2.5 ALF variants ← 03_auxloss_free + 08_ling_2 + 05_qwen3
  - T3.1 MTP ← 23_mtp_investigation §8 决策框架
  - T3.2 AttnRes ← 26_attention_residuals §3
- 不验证项依据：22_FINAL §9 + 25_node_limited_routing + 27_mhc

---

## 12. 开放问题（A2 后仍未答 → A3 接力）

| 问题 | 为何 A2 答不了 | A3 怎么补 |
|---|---|---|
| MTP 在 600M active 是否仍正向？ | A2 200M 离 boundary 太远，外推噪声大 | A3 (600M) 跑 D=0 vs D=1 单对照 |
| 1/32 极稀疏 (Profile M) 是否 scale up 到 16B | A2 80M active 太小，dead expert 风险 ≠ 16B | A3 单跑 Profile M 等比放大 |
| QK-Norm vs MuonClip 在长训下的差异 | A2 25B 不够长 | A3 80B + grad norm 时序对比 |
| Mid-training 数据混合（数学/code 比例） | A2 完全 pretrain corpus | A3 后接 mid-training rehearsal |
| Long-context (32K+) 行为 | A2 固定 4K | A3 / A4 上做 YaRN 外推 |
| FP8 启用门槛 | A2 默认 BF16 | A4 BF16 跑通后用 same data run FP8 二次 |
