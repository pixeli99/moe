# T2.1 路由实施细节：sigmoid+ALF vs softmax+aux 两条 kernel 路径

> **范围**：把 29_wind_tunnel_a2 T2.1 (路由派系消融) 在 dropless + EP=8 + Megablocks 路径下的实现差异、共享组件、容易踩的坑全部展开。**两条路径要在 A2 上做公平对比，必须在共享组件上完全等价，仅在标星的 4 个点上不同**。
> **基线**：PyTorch 2.4+ / Megablocks 0.6+ / FSDP2 + TP=1 / EP=8 单节点；不依赖 DualPipe，1F1B 调度。

---

## 1. TL;DR

两条路径在 99% 代码路径上相同（dispatch / all-to-all / grouped GEMM / combine），仅在 **4 个点** 上分叉：

| 分叉点 | sigmoid+ALF (V3 派) | softmax+aux (Mixtral 派) |
|---|---|---|
| **① Gate 函数** | `sigmoid(logits)` per-expert independent | `softmax(logits, dim=-1)` cross-expert competition |
| **② Top-K weights** | gather raw gate → **renormalize × scale (2.5)** | gather softmax → optional renormalize (× 1.0) |
| **③ Balance 机制** | `bias[i] ← bias[i] − γ·sign(load_i − target)` (在 autograd 外) | `L_aux = α · N · Σ f_i · P_i` (加到 main loss) |
| **④ 辅助 seq-aux** | α=1e-4 seq-level（保险丝） | 无（aux-loss 已足够） |

**踩坑根源**：① ② ③ 三个变量同时被 V3 论文一起改动了，社区复现时经常误把"sigmoid+aux-loss"或"softmax+ALF"当成有效组合。**这些 hybrid 组合 没有任何论文验证过**，T2.1 必须严格按 V3 派 / Mixtral 派两套 self-consistent 配置跑。

---

## 2. 共享组件（两条路径完全相同，A2 公平对比的前提）

> 这些代码路径必须 byte-identical，否则对比会有 noise。

### 2.1 FP32 router compute

```python
# input x: [B, S, D] in BF16
x_fp32 = x.to(torch.float32)               # CRITICAL: gating always FP32
logits = F.linear(x_fp32, W_router_fp32)   # [B, S, N], FP32
# W_router 本身也 FP32 master，不走 mixed precision wrapper
```

**为什么**：dots1 §2 + OLMoE §4.1.7 共识 — gate 走 BF16 会在 N 大时出现 top-K 不稳，影响 expert path 选择。**单次实验 budget 21 H100-hr 是 expert 计算的 30 倍，router FP32 不是瓶颈**。

### 2.2 Dropless token dispatch (Megablocks `permute`)

```python
# 输入：topk_idx [B*S, K] (int32) + topk_g [B*S, K] (BF16)
# 输出：x_permuted [num_local_tokens, D] sorted by expert id
#       expert_offsets [N+1] (前缀和，每个 expert 的 token 范围)
permuted_x, permuted_g, bin_ids = megablocks.permute(
    x.view(-1, D),
    topk_idx.view(-1),
    topk_g.view(-1),
    num_experts=N,
)
expert_offsets = cumsum_per_expert(bin_ids, N)  # [N+1]
```

**特性**：dropless = 所有 token 都送（没有 capacity factor / drop policy）。MaxVio 高时 expert 之间 token 数不均匀，Megablocks 的 grouped GEMM 用 ragged sparse matmul 处理。**两条路径完全共享此 kernel**。

### 2.3 EP=8 all-to-all（单节点 8 H100 NVLink）

```python
# 把 [num_tokens, D] 按 (expert_id % EP) 切分到 8 个 EP rank
# Pattern: AllToAll on token dim, gather on expert dim
sent_tokens = all_to_all(permuted_x, expert_to_rank_map)
sent_g = all_to_all(permuted_g, expert_to_rank_map)
```

**带宽**：NVLink 4.0 单方向 450 GB/s，单 step 200M token × D=1024 × 2 (BF16) = 0.4 GB 数据，远低于 NVLink ceiling。**A2 规模 EP all-to-all 不是 throughput 瓶颈**。

### 2.4 Grouped GEMM (Megablocks `gmm`)

```python
# 每个 EP rank 持有 N/EP = 8 个 expert
# expert 之间 token 数可能不均 → grouped GEMM
expert_out = megablocks.gmm(
    sent_tokens,
    W_expert_local,  # [N/EP, 2*intermediate, D] (SwiGLU)
    expert_offsets_local,
)
# expert_out: [num_local_tokens, D]
```

**特性**：dropless 下 grouped GEMM 退化为多个不同形状的 dense GEMM 的拼接；CUTLASS 内核支持。**两条路径完全共享**。

### 2.5 Combine（反向 all-to-all + weighted sum）

```python
out_pieces = all_to_all_back(expert_out)        # 返回每个 token 它的 K 份 expert 输出
weighted = (out_pieces * topk_g_BF16.unsqueeze(-1)).sum(dim=1)  # [B*S, D]
```

**关键**：`topk_g` 在这里 cast 回 BF16；FP32 → BF16 的舍入误差被 Σ 平均掉。**两条路径完全共享**。

### 2.6 Router z-loss（两条路径都加）

```python
L_z = (torch.logsumexp(logits, dim=-1) ** 2).mean() * 0.001  # β=0.001 (OLMoE)
total_loss = L_ce + L_z + (L_aux_if_softmax_arm)
```

**为什么两条都加**：z-loss 是 logits magnitude 控制器，与 gate 函数无关；OLMoE / V3 / Ling / Qwen3 都加。**A2 不消融 z-loss**。

---

## 3. 路径 A：sigmoid + ALF (V3 派, T2.1 arm A)

### 3.1 完整 forward

```python
# Module: SigmoidALFRouter
class SigmoidALFRouter(nn.Module):
    def __init__(self, dim, n_experts, top_k, scaling=2.5):
        super().__init__()
        self.W = nn.Parameter(trunc_normal(n_experts, dim, std=0.02))
        # bias 不是 nn.Parameter — 不进 autograd、不进 optimizer
        self.register_buffer('bias', torch.zeros(n_experts, dtype=torch.float32))
        self.top_k = top_k
        self.scaling = scaling

    def forward(self, x):
        x32 = x.to(torch.float32)
        logits = F.linear(x32, self.W.to(torch.float32))  # [B, S, N]
        gate = torch.sigmoid(logits)                       # [B, S, N], FP32, each in [0,1]

        # *** 派系核心 ***：bias 只用于 top-K 选择，不进入权重
        scores = gate + self.bias  # [B, S, N]
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)  # [B, S, K]

        # gather raw gate (NO bias) for weights
        top_gate = gate.gather(-1, top_idx)                # [B, S, K]
        # renormalize + scale
        top_gate = top_gate / top_gate.sum(-1, keepdim=True)
        top_gate = top_gate * self.scaling                 # × 2.5

        return top_idx, top_gate.to(x.dtype), gate  # gate 返回用于 seq-aux
```

### 3.2 Bias update（在 autograd 外，每 step）

```python
# 放在 backward 之后、optimizer.step 之前
@torch.no_grad()
def update_alf_bias(router, gate, top_idx, gamma=0.001, mode='v3_sign'):
    # gate: [B, S, N] from forward
    # top_idx: [B, S, K]
    N = router.bias.shape[0]
    K = top_idx.shape[-1]
    total_tokens = top_idx.numel()  # B*S*K

    # 统计每个 expert 的 load
    flat_idx = top_idx.view(-1)  # [B*S*K]
    load = torch.zeros(N, dtype=torch.float32, device=gate.device)
    load.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    load_norm = load / total_tokens  # [N], 总和 = 1
    target = K / N  # 每个 expert 期望负载

    if mode == 'v3_sign':
        # V3 / dots1 默认
        error = load_norm - target  # 正=overloaded, 负=underloaded
        router.bias -= gamma * error.sign()  # overloaded → 减 bias → 下次更少被选

    elif mode == 'ling_zero_mean':
        # Ling 2.0：保 bias 均值不漂移
        error = load_norm - target
        signed = error.sign()
        router.bias -= gamma * (signed - signed.mean())

    elif mode == 'qwen3_global_batch':
        # Qwen3 系：用 global batch（跨 DP rank）的 load
        all_reduce(load_norm, op=SUM, group=DP_GROUP)
        load_norm /= DP_SIZE
        error = load_norm - target
        router.bias -= gamma * error.sign()
```

**关键**：bias 是 buffer 不是 parameter，**不进 AdamW 状态**、**不被 weight decay 衰减**。这是 V3 路径的"feature, not bug" — bias 是 controller variable，不是 learned weight。

### 3.3 辅助 seq-aux loss（α=1e-4，保险丝）

**严格按 V3 paper Eq. 16-20 实现**：

- Eq. 16（实际路由）：`Topk({s_{j,t} + b_j}, K)` — 用 **biased** 分数
- Eq. 18（seq-aux 的 f_i）：`Topk({s_{j,t}}, K)` — 用 **unbiased** raw sigmoid 分数 ⚠️
- Eq. 19-20：P 用 per-token L1 归一化 `s'_{i,t} = s_{i,t} / Σ_j s_{j,t}`，再序列内 mean

**关键**：seq-aux 内部**自己重新算一次 unbiased Top-K**，**不能**接 forward 路径的 `top_idx`（那个是 biased 的）。

```python
# 加到 main loss
def seq_aux_loss(gate_raw, K, alpha=1e-4):
    """V3 paper §2.1.2 Eq. 16-20 严格实现。

    Args:
        gate_raw: [B, S, N] 不带 bias 的 sigmoid 输出（FP32）
        K:        top-K 数（路由用的同一个 K）
        alpha:    V3=1e-4, Ling=1e-4, GLM-4.5=1e-4

    Returns:
        scalar loss
    """
    B, S, N = gate_raw.shape

    # ---- f_i: Eq. 18, top-K over UNBIASED 分数 ----
    # ★ 注意: 这里重新做一次 topk, 不能复用 forward 的 biased top_idx
    _, unbiased_top_idx = gate_raw.topk(K, dim=-1)  # [B, S, K]
    onehot = F.one_hot(unbiased_top_idx, num_classes=N).sum(dim=-2).float()  # [B, S, N]
    counts = onehot.sum(dim=1)  # [B, N]
    f = counts * (N / (K * S))  # scaled so mean(f) = 1 per sequence

    # ---- P_i: Eq. 19-20, per-token L1 normalize then mean over seq ----
    s_prime = gate_raw / (gate_raw.sum(dim=-1, keepdim=True) + 1e-20)
    P = s_prime.mean(dim=1)  # [B, N]

    # ---- L_Bal: Eq. 17, per-sequence then batch-mean ----
    per_seq_loss = (f.detach() * P).sum(dim=-1)  # [B]
    return alpha * per_seq_loss.mean()
```

**为什么 f_i 用 unbiased Top-K**（V3 paper Eq. 18 的设计选择）：

- seq-aux 是要逼 router **自己**学到平衡 —— 不是检查 "bias 是否把不平衡修好了"
- 用 unbiased Top-K → loss 测的是 router 原始倾向，bias 不掺和 → router 学到本征均衡
- 用 biased Top-K（错误做法）→ router 可以摆烂依赖 bias 兜底

**注意**：Ling 2.0 在 8B+ 上用 α=1e-4；本 T2.1 arm A 不消融这个值（α 在 T2.5 不在 T2.1）。GLM-4.5 paper §2.4 明确同样 α=1e-4。

---

## 4. 路径 B：softmax + aux-loss (Mixtral 派, T2.1 arm B)

### 4.1 完整 forward

```python
class SoftmaxAuxRouter(nn.Module):
    def __init__(self, dim, n_experts, top_k, alpha=0.01):
        super().__init__()
        self.W = nn.Parameter(trunc_normal(n_experts, dim, std=0.06))
        self.top_k = top_k
        self.alpha = alpha

    def forward(self, x):
        x32 = x.to(torch.float32)
        logits = F.linear(x32, self.W.to(torch.float32))  # [B, S, N]
        gate = F.softmax(logits, dim=-1)                  # [B, S, N], FP32, sum=1 per token

        # *** 派系核心 ***：top-K 直接在 softmax probability 上
        top_gate, top_idx = gate.topk(self.top_k, dim=-1)  # [B, S, K]
        # OLMoE / Mixtral 不再 renormalize；K=8 时已接近 1（≈0.8-0.95）
        # 也不乘 scaling factor (隐含 = 1.0)

        return top_idx, top_gate.to(x.dtype), gate
```

### 4.2 Aux loss（Fedus 2021 / Switch Transformer）

```python
def switch_aux_loss(gate, top_idx, alpha=0.01):
    # gate: [B, S, N] (full softmax), top_idx: [B, S, K]
    B, S, N = gate.shape
    K = top_idx.shape[-1]

    # f_i: fraction of tokens routed to expert i (基于 top-K 选择)
    flat_idx = top_idx.view(-1)
    f = torch.zeros(N, device=gate.device)
    f.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    f = f / flat_idx.numel()  # [N], 和=1

    # P_i: avg gate prob per expert (在全部 token 上 — 进入 autograd)
    P = gate.mean(dim=(0, 1))  # [N], FP32, 和=1

    # 经典 Switch loss: N * Σ f_i * P_i
    # f 不进 autograd (硬选择)，P 进 autograd
    L_aux = alpha * N * (f.detach() * P).sum()
    return L_aux

# 调用
total_loss = L_ce + L_z + switch_aux_loss(gate, top_idx, alpha=0.01)
total_loss.backward()
```

**关键**：aux loss 通过 P 给 W_router 加梯度，**把路由从"质量最优"拉向"负载均衡"**。V3 抛弃此机制的理由：α=0.01 太强（影响 quality），α 太小（无效）。ALF 用 bias 解耦了这两个目标。

**Mixtral 实际数值**：Mixtral 论文 §2.1 用 α=0.01；OLMoE 用 α=0.01；Switch 用 α=0.01。**T2.1 arm B 用 α=0.01 是社区共识，不在 T2.1 范围内消融**。

---

## 5. 四个分叉点的并列对比表

| 维度 | V3 派 (arm A) | Mixtral 派 (arm B) |
|---|---|---|
| Gate function | sigmoid (per-expert independent) | softmax (cross-expert) |
| Gate sum constraint | 无（每 logit 独立 0-1） | 强制 sum=1 |
| Top-K selection 用什么 | `sigmoid_gate + bias` (评分含 bias) | softmax_gate 直接 (无 bias) |
| Top-K weights 用什么 | `sigmoid_gate.gather(top_idx)`（**不含 bias**） | `softmax_gate.gather(top_idx)` |
| Top-K renormalize | ✓ (sum to 1)（必须，sigmoid 不保 sum=1） | ✗ (OLMoE) / ✓ (Mixtral) — A2 选不 renorm |
| routed_scaling_factor | **2.5** | 1.0 |
| Balance 机制 | bias update (autograd 外) | aux loss (autograd 内) |
| Balance 触发频率 | 每 step | 每 step (与 main loss 同步) |
| Balance 强度参数 | γ=0.001 | α=0.01 |
| Seq-level safety | ✓ α=1e-4 | ✗ (aux loss 已足够) |
| Router W init std | 0.02 (V3) | 0.06 (OLMoE) |
| Bias init | 0 | n/a |
| Bias 是否 nn.Parameter | ✗ (buffer) | n/a |
| Bias 是否进 AdamW | ✗ | n/a |
| Bias 是否进 weight decay | ✗ | n/a |
| Logits magnitude 监控 | z-loss β=0.001 | z-loss β=0.001 |

---

## 6. 实施常见坑（15 个）

### A. 跨派系混合（最大类）

1. **❌ Sigmoid + aux-loss**：sigmoid gate 不 sum=1，aux loss 公式 `N·Σf_i·P_i` 的 normalization 假设崩塌 → loss 爆炸或 noop。**没论文做过，不要尝试**。
2. **❌ Softmax + ALF bias**：softmax 已经 cross-expert 竞争，加 bias 会破坏 sum=1 → 后续 top-K 选择不再可解释。也没论文做过。
3. **❌ ALF 但 bias 是 nn.Parameter**：bias 会被 AdamW 优化 + weight decay 衰减，与论文 spec 不符，balance 失效。**初学者最常见错误**。
4. **❌ ALF + routed_scaling_factor=1.0**：sigmoid 输出 mean ~0.5，topK 后 sum ~K·0.5 = 4 (K=8) → 与 dense FFN 输出尺度不匹配。**必须用 2.5（V3）或 6.0（LongCat）**。
5. **❌ aux-loss 但 P 用 detach()**：那 W_router 收不到 balance 梯度，aux loss 沦为 monitoring。Switch 论文是 P 进梯度、f detach。

### B. FP precision

6. **❌ Router gate 用 BF16**：dots1 §2 / OLMoE §4.1.7 都强调必须 FP32；BF16 下 sigmoid 在 logits>10 时饱和，top-K 不稳。
7. **❌ Bias 用 BF16**：γ=0.001 量级，bias 累积步数多，BF16 下尾数精度不足 → bias 长期摆动。bias buffer **必须 FP32**。
8. **❌ top_gate.to(BF16)** 在 renormalize **之前**：renormalize 应在 FP32 下完成，最后一步 cast 回 BF16。

### C. EP / 通信

9. **❌ ALF bias update 在 DP 之间不同步**：bias 是 controller variable，必须 all-reduce 跨 DP rank 后再 update（Qwen3 global-batch 模式）。**V3 默认就这么做**；如果只在单 rank 更新 → 不同 rank 路由不一致，模型发散。
10. **❌ aux-loss 用 local batch 而非 global batch P**：OLMoE 默认 local（per-DP-rank）；Qwen3 用 global（all-reduce P 后求 loss）。**T2.1 arm B 用 local 与 OLMoE 一致**；如果要做 Qwen3 风格，放 T2.5 而不是 T2.1。
11. **❌ All-to-all 的 token 数没 padding**：dropless 下每 rank 收到的 token 数不等，需要 dynamic-shape all-to-all（Megablocks 已支持）；naive impl 用固定 capacity factor 会 drop token → 与 dropless spec 不符。

### D. Top-K 与 weight

12. **❌ 用 (sigmoid + bias) 当 top-K weight**：bias 是 controller，混进 weight 会让 expert output 直接被 bias 拉偏。V3 §4.2 + Eq. 15 明确：bias **only for selection, not for weighting**。
13. **❌ Top-K renormalize 之后忘乘 scaling**：sigmoid path 必须 `topk_g = (topk_g / sum) * 2.5`，少乘 2.5 → expert 输出尺度只有 dense FFN 的 1/2.5。

### E. Bias update 时机

14. **❌ Bias update 在 forward 之内**：bias 是 controller，update 应该在 backward 完成、optimizer.step 之前。在 forward 内 update → 同一 step 的 forward 看到不同 bias，破坏 step 一致性。
15. **❌ Bias 累积到 NaN**：γ=0.001 × sign(error) 单步最多 0.001 改动，1M step 累积 1000，FP32 仍稳；但如果 error 不是 sign 而是 raw value，单步可能 100×，需要 clip。**用 sign 是关键**。

---

## 7. T2.1 公平对比 checklist

> A 和 B arm 必须**仅在 §5 表格的 4 行（gate function / top-K weights / balance / seq-aux）上差异**，其他全部 byte-identical。

| 共享设置 | 值 | 理由 |
|---|---|---|
| Data | 同 25B token 同 seed 同 shuffle | 否则差异淹没在数据 noise |
| Init seed | 同（router W 除外） | 全局 RNG 一致 |
| Router W init | A: std=0.02 / B: std=0.06 | 各派系自家最优；不在 T2.1 范围 |
| Bias init | A: 0 / B: n/a | – |
| Top-K | 6 | A2 baseline，不变 |
| N_routed | 64 | A2 baseline，不变 |
| N_shared | 1 | A2 baseline，不变 |
| Optimizer | AdamW β=0.9/0.95 ε=1e-8 wd=0.1 | T1.1/T1.2 已锁 |
| LR | 1.5e-3 peak | A2 baseline |
| Batch | 1M tokens/step, 25K steps | 同 |
| Precision | BF16 master + FP32 router | 同 |
| EP | 8 | 同 |
| z-loss | β=0.001 | 都加 |
| QK-Norm | ✓ | 都加 |
| First layer dense | ✓ | 都加 |
| Sequence length | 4096 | 同 |
| MTP | D=0 | T3.1 不在 T2.1 中 |

**孤立变量**：仅 ① gate function、② top-K weight 路径、③ balance（bias vs aux loss）、④ seq-aux α=1e-4 vs none —— 4 件事打包成 "派系" 一起切换，**不拆开**。拆开（5 中提到的 hybrid 组合）没有论文背书且大概率失败。

---

## 8. Telemetry — T2.1 必抓的对比指标

### 8.1 训练动态
- `train/loss` — 主决策
- `train/loss_aux` (B arm only) / `train/loss_alf_seq` (A arm only) — 监控 balance overhead
- `train/loss_z` — 都监控
- `train/grad_norm` 全局
- `train/router_grad_norm` 单独 — A arm 路由梯度 ≈ B arm 的 5-10%（A 没有 aux gradient 拉）

### 8.2 路由健康
- `route/MaxVio_local` = max(load_per_expert) / min(load_per_expert) − 1, per DP rank
- `route/MaxVio_global` = 同上但 all-reduce 后
- `route/active_experts` = 至少被 1% token 选中的 expert 数（dead expert 检测）
- `route/router_entropy` = `-Σ gate · log(gate)` 平均 — 主观上 A arm > B arm（sigmoid 没强制 sum=1）
- `route/topk_gate_sum` 平均 — A arm 应 ≈ 2.5（scaling）；B arm 应 ≈ 0.85-0.95（softmax topK partial sum）
- `route/bias_mean` (A arm only) — 应在 ±0.5 内（V3 §4.2）
- `route/bias_std` (A arm only) — 应 < 1.0
- `route/W_router_norm` — 都监控；A arm 应 stable，B arm 因 aux loss 推动会缓慢漂移

### 8.3 通信 / 效率
- `comm/a2a_time_ms` per step — 两条路径应相同
- `comm/imbalance_factor` = max_tokens_per_rank / mean — A arm 因 ALF 应 < 1.05，B arm 因 aux 应 < 1.10
- `perf/MFU` — A arm 可能略高（少一个 aux loss compute），但差异 < 1%

### 8.4 下游（每 5B tokens）
- HellaSwag / PIQA / WinoGrande / SciQ — sanity check
- MMLU 5-shot — A2 200M active 可能噪声大但仍有信号
- GSM8K 8-shot — reasoning 试探

---

## 9. PyTorch 完整 forward 伪代码（两条对比）

```python
class MoELayer(nn.Module):
    def __init__(self, dim, n_routed, top_k, n_shared, expert_dim,
                 mode='sigmoid_alf'):
        super().__init__()
        self.mode = mode
        if mode == 'sigmoid_alf':
            self.router = SigmoidALFRouter(dim, n_routed, top_k, scaling=2.5)
        elif mode == 'softmax_aux':
            self.router = SoftmaxAuxRouter(dim, n_routed, top_k, alpha=0.01)

        self.experts = GroupedExperts(n_routed, dim, expert_dim)  # SwiGLU
        if n_shared > 0:
            self.shared = nn.Sequential(  # standard dense SwiGLU
                SwiGLU(dim, expert_dim * n_shared)
            )
        else:
            self.shared = None

    def forward(self, x):
        # Router
        top_idx, top_g, full_gate = self.router(x)  # full_gate for seq-aux/aux-loss

        # Dispatch + grouped GEMM + combine (Megablocks)
        routed_out = megablocks_moe(x, top_idx, top_g, self.experts)

        # Shared expert
        if self.shared is not None:
            shared_out = self.shared(x)
            out = routed_out + shared_out
        else:
            out = routed_out

        # Stash for loss / bias update
        self._gate_cache = full_gate
        self._idx_cache = top_idx
        return out

    def post_step_update(self, gamma=0.001):
        if self.mode == 'sigmoid_alf':
            update_alf_bias(self.router, self._gate_cache, self._idx_cache,
                           gamma=gamma, mode='v3_sign')

    def get_aux_loss(self):
        if self.mode == 'softmax_aux':
            return switch_aux_loss(self._gate_cache, self._idx_cache,
                                   alpha=0.01)
        elif self.mode == 'sigmoid_alf':
            # ★ 不传 self._idx_cache (那是 biased top-K); seq_aux_loss 内部
            # 用 unbiased gate_raw 重新算 Top-K, 严格按 V3 paper Eq. 18
            K = self._idx_cache.shape[-1]
            return seq_aux_loss(self._gate_cache, K, alpha=1e-4)
        return 0.0

# Training loop integration
for batch in data:
    loss = model(batch).loss
    for layer in model.moe_layers:
        loss = loss + layer.get_aux_loss()
    loss = loss + z_loss(...)
    loss.backward()

    # *** sigmoid+ALF special: bias update OUTSIDE autograd ***
    if model.cfg.mode == 'sigmoid_alf':
        for layer in model.moe_layers:
            layer.post_step_update(gamma=0.001)

    optimizer.step()
    optimizer.zero_grad()
```

---

## 10. 实施 sprint 计划（与 T2.1 启动绑定）

| Step | 内容 | Owner | 工期 |
|---|---|---|---|
| 1 | 把 Megablocks `permute / gmm / combine` 在 8×H100 + EP=8 跑通 baseline (任一派系即可) | infra | 3d |
| 2 | 实现 `SigmoidALFRouter` + bias update; 单元测试 bias 随 imbalance 收敛 | research | 2d |
| 3 | 实现 `SoftmaxAuxRouter` + aux loss; 验证 aux loss 数值范围 (~0.01-0.05) | research | 1d |
| 4 | A2 baseline 配置打通；end-to-end loss 曲线无 NaN 100B token | infra+research | 3d |
| 5 | 跑 T2.1 arm A: sigmoid+ALF | – | 21 H100-hr (= 4hr on 8 H100s) |
| 6 | 跑 T2.1 arm B: softmax+aux | – | 21 H100-hr |
| 7 | telemetry dashboard：把 §8 指标拉通 | infra | 与训练并行 |
| 8 | 决策会：根据 §29_wind_tunnel_a2 T2.1 决策门槛裁定 winner | research | 1d |

**Total**：~2 周 (含 infra setup) ~10 H100-day = 1 整节点 1.25 day

---

## 11. 与其他笔记的交叉

- 上游决策：29_wind_tunnel_a2 §4 (T2.1) 给出消融格式 + 决策门槛
- 派系统计：28_open_source_moe_catalog §3.4 + §4 Pattern A vs B
- ALF 原始论文：03_auxloss_free.md (2408.15664) + V3 §4.2
- Ling 零均值变种：08_ling_2.md + 29_wind_tunnel_a2 T2.5
- Switch aux loss：10_mixtral.md 引用 Fedus 2021 + 09_olmoe.md §4.1.7
- Router z-loss：09_olmoe §4.1.7（β=0.001 来源）
- dropless / EP / Megablocks 工程细节：22_FINAL §6 (硬件 / 部署)

---

## 12. 后续想做但不在 T2.1 范围

| 项 | 何时做 | 备注 |
|---|---|---|
| Sigmoid + group routing (Ling-mini-2.0 风格 n_group=8 topk_group=4) | T2.2 (N_routed=256 arm) 时一起带 | 仅 256 expert 上有意义 |
| `routed_scaling_factor` 2.5 vs 6.0 (LongCat) | A3 上做 | A2 N_routed=64 太小，6.0 过强 |
| Bias update γ 调整 (0.001 vs 0.0001) | T2.5 包含 (Ling 零均值 arm 用 0.001 default) | – |
| Expert grouping (Pangu MoGE) | 不做 | 25_node_limited_routing 已结论 NLR 不引入 |
| GRIN gradient routing (Phi-3.5) | 不做 | 工程复杂度高、独立验证不足 |
| Attention router (Yuan-M32) | 不做 | 22_FINAL §9 已否决 |
