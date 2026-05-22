# `calc/` — MoE Architecture Calculator

输入 MoE spec（hidden / layers / N / K / d_expert / ...），算出：

- **参数账**：总参 + 严格 active + V3 口径 active + 每模块占比
- **不变量校验**：attn:FFN 比、d_dense_int = (K+N_sh)×d_expert、GQA 整除、head_dim RoPE 兼容、N_routed EP 友好
- **三种"激活率"**：active/total、expert-slot、routed-only（防止 [[feedback_unit_precision]] 这种口径混用）
- **KV cache**：per-token / per-layer / @ context length
- **FLOPs**：per-token forward / training total
- **训练成本**：H100-hours / wall-clock days / $
- **并行可行性**：EP/PP/TP 整除 + NLR 必要性
- **跟 anchor 对照**：12 个 production model 直接横比（V2-Lite / V3 / K2 / Ling 全系 / GLM 全系 / dots1 / Mixtral / OLMoE）

## 快速开始

```bash
# 列所有 anchor
python3 -m calc list

# 看 V3 spec breakdown
python3 -m calc info v3

# 校验你的 L100 spec
python3 -m calc check user-l100-v1

# 你的 16B vs V2-Lite vs Ling-mini-2 side-by-side
python3 -m calc compare user-16b v2-lite ling-mini-2

# 训练成本估算 (22T tokens / 256 H100 / MFU 0.45 / $3/hr)
python3 -m calc cost user-l100-v2 22 --gpus 256

# 并行拓扑可行性
python3 -m calc parallel v3 64 16 1

# 全 anchor summary 一页
python3 -m calc all
```

## Python API

```python
from calc import MoEArchSpec, get_anchor, compare_specs

# 用 anchor
spec = get_anchor("v3")
print(spec.summary())

# 自定义 spec
my = MoEArchSpec(
    name="My 100B",
    hidden=4096, num_layers=32,
    num_q_heads=64, num_kv_heads=8, head_dim=128,
    attn_type="gqa",
    n_routed=128, top_k=8, n_shared=1, d_expert=2048,
    first_k_dense=1, dense_intermediate=0,  # 0 = auto compute-equivalent
    vocab_size=128256,
)

# 直接拿数字
print(my.total_params())          # 总参 (int)
print(my.active_params())          # strict active (no embedding)
print(my.attn_ffn_ratio())         # 0.37 = 1:2.7
print(my.activation_rate())        # {'active_over_total': 0.066, ...}
print(my.kv_cache_bytes_total(32768))  # 32K ctx KV cache 字节数

# 校验
for r in my.check_invariants():
    print(r)
for r in my.check_parallelism(ep=16, pp=2, tp=1):
    print(r)

# 训练成本
print(my.training_cost(22e12, num_gpus=256))

# 对比
print(compare_specs(my, get_anchor("v3"), get_anchor("ling-flash-2")))
```

## 数字准确度（vs paper）

| Anchor | 我算的 (V3 口径 active) | Paper 报告 active | 误差 |
|---|---|---|---|
| **DeepSeek-V3** | 37.55B | 37B | **+1.5%** ✓ |
| **Ling-1T** | 55.06B | 51B | +8% (口径差，paper 可能排除 router 或 shared) |
| **GLM-4.5** | (待 verify) | 32B | – |
| **K2** | (待 verify) | 32B | – |
| **V2-Lite** | 2.66B | 2.4B | +11% (V2-Lite paper 用 strict 口径) |
| **User 16B Profile B'** | 3.11B | 2.45B strict / 3.11B V3 口径 | **0%** ✓ (用户自验过) |

→ **总参普遍 ±2% 内**，**active 因 paper 口径差异 ±10% 内**。设计探索够用，做 marketing material 时验证 paper 具体定义。

## 已知限制

1. **Hybrid attention 不支持**：Step 3.5 Flash (SWA:Full 3:1)、Qwen3-Next (DeltaNet:Attn 3:1)、Jamba (Mamba:Attn) 这种异构 attention 模型只能用"平均每层"近似，spec 精度差。这些模型在 anchor DB 暂未收录。
2. **MTP 模块大小是估算**：不同模型 MTP 实现差异大（V3 是 MoE MTP，LongCat/Ling/Step 是 dense MTP），我用统一近似式
3. **Sequence-aux loss / router z-loss 不算 param**：这些是 loss 项不是 weight，calc 不涉及
4. **不算 RMSNorm/bias**：通常 < 0.1% 总参，可忽略
5. **FLOPs 估算用 Chinchilla 6× 公式**：忽略 KV cache 重用带来的优化

## 文件结构

```
calc/
├── __init__.py         # 公开 API
├── spec.py             # MoEArchSpec dataclass + 全部公式
├── anchors.py          # 12 个 anchor + user 设计
├── compare.py          # 横向对比表
├── cli.py              # 命令行
└── __main__.py         # `python -m calc` 入口
```

## 添加新 anchor

每个 anchor 必须有可验证 source：HF `config.json` 是首选，paper Table 1 次之。

```python
# anchors.py
"my-model": MoEArchSpec(
    name="My Model Name (123B/4.5B)",
    # 必填
    hidden=..., num_layers=...,
    num_q_heads=..., num_kv_heads=...,
    attn_type="gqa",  # or "mha", "mla"
    n_routed=..., top_k=..., n_shared=..., d_expert=...,
    first_k_dense=..., dense_intermediate=0,  # 0=auto
    vocab_size=...,
    # 可选
    mtp_depth=0,
    mtp_is_moe=False,
    notes="Source: <link>; Paper: arXiv ...",
),
```

然后跑 `python3 -m calc info my-model` 看跟 paper 公开数字差多少。

## 跟仓库其他笔记关系

- [[28_open_source_moe_catalog]] —— anchor 来源
- [[22_FINAL_16B_design]] —— user-16b spec 来自这里
- [[42_100b_cookbook]] —— Step 1-12 决策树，calc 验证每个 step 的不变量
- [[43_short_wide_design]] —— user-16b vs Option B/C 对照
- [[45_l100_l200_design]] (待写) —— L100/L200 spec 跟 anchor 对照
- [[03_auxloss_free]] —— ALF bias 不在 param count 里（buffer）
- [[39_muon]] —— Muon 不改 param count，但改 cost 估算（Muon 给 ~2× compute efficiency）
- [[memory: feedback_unit_precision]] —— 为什么 active 有 3 种口径

## TODO

- [ ] 支持 hybrid attention（SWA + Full / Mamba + Attn / DeltaNet + Full）
- [ ] 加 inference latency 估算（layer-by-layer wall clock）
- [ ] 加 expert load balance simulator (输入 ALF u 看 bias 收敛速度)
- [ ] 加 upscale 路径模拟器（SOLAR-DUS / sparse upcycle 给目标 spec）
- [ ] streamlit web UI 可选
