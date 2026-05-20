# Node-Limited Routing 决策备忘 — 16B MoE 要不要用？

> **TL;DR**：默认 **不用**。NLR 不是 load-balancing 策略，是**EP 跨节点时**的通信拓扑约束。16B / 64 routed experts / EP=8 单节点的 default 拓扑下完全多余。
> 仅当 (a) 走 Profile M（256 experts）+ EP ≥ 16 跨节点，或 (b) 集群 NVLink domain < 8 GPU、被迫 EP 跨节点 时，按 V3 风格设 **M=2** 作为零成本保险。
> **Multi-machine training ≠ EP 跨节点**，多机 DP + 单节点 EP 是绝大多数 16B 实践，这种情况 NLR 是 no-op。

---

## 1. NLR 是什么，不是什么

### 1.1 定义（DeepSeek V3 §4.2 原文）

> "we ensure that each token will be sent to at most M nodes, which are selected according to the sum of the highest K_r/M affinity scores."

V3 配置：256 routed experts / EP=64 / 8 节点 × 8 GPU；K=8；**M=4**。

### 1.2 NLR **不**做什么 ❌

- ❌ NLR **不是** load balancing —— 跟 expert load 均衡度无关
- ❌ NLR **不**改善 perplexity / 模型质量
- ❌ NLR **不**替代 ALF / sequence-aux loss
- ❌ NLR **不**影响 dropless 性质

### 1.3 NLR 实际做什么 ✓

- ✓ **bound cross-node IB 流量上限**：每 token 至多 M 节点 → all-to-all 数据量 ≤ M/N_nodes × full
- ✓ 让 IB 通信能与 NVLink intra-node + 计算在 DualPipe 里**完全重叠**（V3 §3.2.2）
- ✓ 与 expert placement strategy 配合，把 K 个 expert 在物理拓扑上聚到少数节点

**核心**：NLR 是 V3 把 IB 当成 first-class bottleneck 的产物。它的价值**完全取决于你的 EP 拓扑是否跨节点 + IB/NVLink 带宽比**。

---

## 2. 决策流程图

```
是否做 EP？
├── 否 (PP+TP only / 全 dense replicated experts) → NLR 无意义
└── 是 (做 EP) →
    │
    EP 是否跨节点？
    ├── 否 (EP 在一个 NVLink domain 内) → NLR 无意义 ✓ 默认
    └── 是 (EP 跨 ≥ 2 节点) →
        │
        K_r / N_nodes_in_EP 是几？
        ├── ≤ 1（每节点最多 1 个 expert，K 全部在 ≤ K 个节点） → NLR 不影响
        └── > 1 →
            │
            IB 带宽是否在 profiler 中显著拥塞？
            ├── 否 → NLR 仍无意义
            └── 是 → 用 NLR，M = ceil(K / max_experts_per_node)
```

**对 16B Profile B**（64 experts / EP=8）：
- 8 GPUs 一个节点（H100 NVLink domain）→ EP=8 不跨节点 → **第二个分支就退出，NLR 无意义**

**对 16B Profile M**（256 experts / EP=16 → 2 节点）：
- EP 跨 2 节点；每节点 16 experts
- K=8 → 最坏 8 个 expert 全分到 2 节点（M=2 即上限自然满足）
- → **NLR M=2 是 trivial constraint，等价不加**

→ **16B 的所有 sensible EP 配置下 NLR 都几乎是 no-op**。

---

## 3. 公开模型里 NLR 的实际使用情况

| 模型 | Total / Experts | EP 配置 | 节点 × GPU/node | NLR | 为什么 |
|---|---|---|---|---|---|
| **DeepSeek V3** | 671B / 256 | EP=64 | **8 × 8** (H800) | **M=4** | EP 跨 8 节点；K=8 / 8 节点最坏全跨节点；M=4 强制至多 4 节点 |
| **DeepSeek V2** | 236B / 160 | — | — | **device-balance + comm-balance loss** | V2 用 loss 控制，不用硬约束 |
| **Kimi K2** | 1T / 384 | EP=384 | — | **不用 NLR** | 用 token-choice + ALF 直接 |
| **Ling-1T** | 1T / 256 | EP=? | — | **不用 NLR** | 论文未提；config 也无 |
| **Ling-mini-2.0** | 16B / 256 | — | — | **不用 NLR** | 同 |
| **dots.llm1** | 142B / 128 | 单节点 H800 | — | **不用 NLR** | 单节点能跑，不需要 |
| **Qwen3-30B-A3B** | 30B / 128 | — | — | **不用 NLR** | 同 |
| **OLMoE-1B-7B** | 7B / 64 | — | — | **不用 NLR** | 16 层小模型 |
| **Mixtral 8×7B** | 47B / 8 | — | — | N/A | K=2 / N=8 太粗 |
| **Hunyuan-Large** | 389B / 16 | — | — | **不用 NLR** | K=1 / N=16 |

**关键观察**：
- **唯一公开使用 NLR 的就是 V3**
- V3 之外的所有团队（包括 V3 的衍生路线 dots1/Ling）都不用
- V4 是否砍掉 NLR 无定论（聊天里张一鸣说"v4 好像删掉了这个限制"）—— 但即使没砍，**从公开证据看 NLR 不是行业普遍做法**

---

## 4. 为什么"多机器训练"不等于"需要 NLR"

聊天里"肯定是多机器训练"这句话，对 NLR 决策实际上**没有信息量**。理由：

### 4.1 多机训练有三种典型 parallel topology

| 拓扑 | 说明 | NLR 相关性 |
|---|---|---|
| **DP-only multi-node** | FSDP / ZeRO-3 跨节点 DP，模型完整复制 | **无关**（无 EP） |
| **DP + 单节点 EP** | EP 装在 NVLink domain；DP 跨节点 | **无关**（EP 不跨节点） |
| **DP + 跨节点 EP** | EP 跨 ≥ 2 节点；token 的 K experts 可能落到不同节点 | **相关** |

对 16B + 64 experts：
- 单 H100 80GB 能放 64 个 expert 全套（64 × 1408 × 2048 × 3 × 2B ≈ 1.1 GB）→ **EP=1 都行**
- 但 EP=1 意味着每个 step 每张卡都跑全部 64 experts 的 GEMM，吞吐烂；所以**实践 EP=8 把 64 experts 分到 8 卡** → 一个 NVLink domain
- → DP × EP=8 是自然选择，无论你用多少节点

→ **想到达"NLR 相关"的 topology 必须主动选 EP > 8**。这只在两种情况下发生：
1. Profile M (256 experts) + 想把 expert 进一步分散到 EP=16 → 跨节点
2. 集群是 4-GPU/node（如某些 L40S 配置） → EP=8 被迫跨 2 节点

### 4.2 "多机训练" 的常见误解

- **误解**："我用了 100 张 GPU 跨 12 节点训 16B，所以需要 NLR" → **错**。如果 EP=8 装在节点内、DP=12.5 跨节点，**all-to-all 只在节点内发生**，跨节点是 all-reduce（DP gradient sync），那是另一个通信原语，跟 NLR 无关
- **正确判断**：看 `EP_size > 节点内 GPU 数` 是否成立

---

## 5. 16B 设计的最终建议

### 5.1 Profile B（64 experts，默认）

| 项 | 推荐 | 理由 |
|---|---|---|
| EP | **8**（单节点 NVLink domain） | 64 experts ÷ 8 GPU = 8 experts/GPU；H100 内存充裕 |
| NLR | **不用** | EP 不跨节点，NLR 是 no-op |
| 多机配置 | DP × ZeRO-1 跨任意节点数 | DP 跨节点用 IB all-reduce，与 EP 解耦 |
| Spec 写法 | "EP=8 single-node; **no node-limited routing** by design" | 评审时主动声明 |

### 5.2 Profile M（256 experts）

| 项 | 推荐 | 理由 |
|---|---|---|
| EP | **16** 或 **8** | 256/16 = 16 experts/GPU 略高但可接受；或 EP=8 容忍 32/GPU |
| NLR | **EP=8 不需要**；**EP=16 跨节点时设 M=2 作为 trivial 保险** | M=2 在 EP=16/2 节点的 setup 下其实是自动满足的 |
| 关键风险 | **不是 NLR**，而是 256 expert 的 dropless kernel（Megablocks / GroupedGEMM）成熟度 | dots1 自研 grouped GEMM 比 NVIDIA TE +14% 是参考 |

### 5.3 如果集群是 4-GPU/node 的小节点（如 L40S / 部分 4090 集群）

| 项 | 推荐 |
|---|---|
| EP | **8 跨 2 节点**（不可避免） |
| NLR | **M=2**（强制 K=8 的 8 个 expert 都落在这 2 节点 — 与拓扑约束吻合） |
| DualPipe / 1F1B-overlap | **必须开启**，否则 IB 通信成本 dominates |
| 替代方案 | 切回 Profile R（19.5B）或 Profile B 但 EP=4 跨更多节点 — **均不优** |

### 5.4 Wind tunnel 中是否消融 NLR？

**不必专门测**。理由：
- A2 anchor (1B/200M) 用 EP=1 或 EP=2 单节点，NLR 无意义
- A4 anchor (16B target) 已经跑目标拓扑，profiler 直接看 IB 利用率，**事后开关 NLR 1 次足够**，不必专门做 controlled ablation
- NLR 与 model quality 无关，只与 throughput 有关

---

## 6. 张一鸣"V4 删掉 NLR" 的可能解释

聊天里"v4 好像删掉了这个限制 但考虑 deepseek 经常内藏一些训练细节 也可以考虑先用 v3 做"。三种解释：

1. **拓扑变了**：V4 可能用更大 NVLink domain（如 GB200 NVL72 = 72-GPU NVLink），EP 更容易装进单 domain，NLR 不再需要
2. **通信优化更彻底**：FlashOverlap / 更激进的 all-to-all kernel + grouped GEMM 让 IB 不再是 bottleneck（dots1 的 1F1B-overlap 就是这个方向）
3. **DeepSeek 内藏细节**：V4 仍在用 NLR 但论文没强调

**对 16B 的指导**：
- 这三种解释**都不影响 16B 的判断**
- 16B 不是 V3/V4 那种 256 expert × 8 节点的 setup，参考意义有限
- "V4 砍了 → 我们也砍" 不是有效推理；"V3 用 → 我们也用"也不是
- **基于自己的 EP 拓扑判断**才是正确路径（见 §2 决策树）

---

## 7. 给评审的一行 spec

> **Routing topology**: EP=8 single-node × DP-N multi-node × ZeRO-1; **no node-limited routing** (EP 不跨节点，NLR 是 no-op)；如未来切 Profile M 且 EP ≥ 16 跨节点，可按 V3 风格加 M=2 — 当前不进 spec。

---

## 8. 与本仓库的交叉引用

- **04_deepseek_v3.md** §4.2：NLR 原始定义，M=4 + 8 节点 setup
- **02_deepseek_v2.md**：V2 用 device-balance loss + comm-balance loss（V3 全部抛弃）
- **24_dots1.md**：142B / 128 expert / 单节点 H800，**不用 NLR** 也跑通 11.2T
- **08_ling_2.md**：256 expert × 三档规模，**不用 NLR**
- **22_FINAL_16B_design.md** §7：训练系统配置（PP=4 / EP=8 / DP=ZeRO-1 / 无 TP）—— **EP=8 单节点是默认前提**
