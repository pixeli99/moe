"""Wind tunnel experiment planning.

Given a target spec (e.g., L100 or L200), generate a set of small
scaled-down anchor experiments + total compute cost. Inspired by
Ling 2.0 paper §2.3.3 (Ling Wind Tunnel — 5 anchors at 500M-8B, total
cost ~35% of one full L100 ablation).

Use case: before training a 100B model from scratch (or doing 100B → 200B
upscale), run 4-5 wind tunnel models at smaller scale to verify scaling
laws + invariants + new technique (Muon, AttnRes, hybrid attention, etc.).

Strategy:
  - Keep architecture shape (N_routed, K, N_shared, attn_type, etc.)
  - Shrink hidden / num_layers proportionally (power law)
  - Train each to compute-optimal (Chinchilla 20× active tokens for dense;
    25-35× for MoE, see Ling Scaling Law)
"""
from __future__ import annotations
from dataclasses import replace
from .spec import MoEArchSpec


def scale_down(
    base: MoEArchSpec,
    scale_factor: float,
) -> MoEArchSpec:
    """Shrink a spec while preserving architecture identity.

    For total params ∝ N_routed × hidden × d_expert × num_layers and we keep
    N_routed fixed (preserving sparsity/granularity), we scale each of the
    other 3 dims by cube-root: c = scale_factor^(1/3).

    Critical alignments:
      - hidden is aligned to head_dim so num_q_heads × head_dim == hidden exactly
      - For hybrid models, each AttnLayerSpec's heads are also scaled by
        hidden_ratio so Q-promote relationships are preserved
      - d_expert is aligned to 32 for EP-friendliness
      - num_layers is rescaled by cube-root, with hybrid layer counts
        proportionally rescaled to sum exactly to new num_layers

    N_routed, K, N_shared, attn_type, head_dim, vocab — PRESERVED.
    """
    c = scale_factor ** (1/3)

    head_dim = base.head_dim
    # Align hidden to head_dim (not 64) so q_heads × head_dim == hidden exactly
    new_hidden = max(head_dim, round(base.hidden * c / head_dim) * head_dim)
    new_layers = max(6, round(base.num_layers * c))
    new_d_expert = max(128, round(base.d_expert * c / 32) * 32)

    hidden_ratio = new_hidden / base.hidden

    # Re-derive Q/KV heads
    if base.attn_type in ("mha", "gqa"):
        gqa_ratio = max(1, base.num_q_heads // base.num_kv_heads)
        new_q = new_hidden // head_dim  # exact since hidden is head_dim-aligned
        new_kv = max(1, new_q // gqa_ratio)
    else:  # MLA — scale heads by hidden_ratio
        new_q = max(4, round(base.num_q_heads * hidden_ratio))
        new_kv = new_q

    # Rescale hybrid_attn: layer counts AND each layer's heads
    new_hybrid = None
    if base.hybrid_attn:
        L = base.num_layers
        # Step 1: rescale layer counts (largest-remainder so sum == new_layers)
        raw = [(cnt * new_layers / L, ls) for cnt, ls in base.hybrid_attn]
        floors = [(int(v), v - int(v), ls) for v, ls in raw]
        leftover = new_layers - sum(f for f, _, _ in floors)
        order = sorted(range(len(floors)), key=lambda i: -floors[i][1])
        counts = [f for f, _, _ in floors]
        for i in order[:leftover]:
            counts[i] += 1

        # Step 2: also scale each AttnLayerSpec's heads by hidden_ratio
        new_hybrid = []
        for i, (cnt, _, ls) in enumerate(floors):
            new_cnt = counts[i]
            if new_cnt == 0:
                continue
            scaled_q = max(1, round(ls.num_q_heads * hidden_ratio))
            scaled_kv = max(1, round(ls.num_kv_heads * hidden_ratio))
            # Scale linear-attn value heads too if applicable
            scaled_lv_heads = (
                max(1, round(ls.linear_num_value_heads * hidden_ratio))
                if ls.linear_num_value_heads else None
            )
            new_ls = replace(
                ls,
                num_q_heads=scaled_q,
                num_kv_heads=scaled_kv,
                linear_num_value_heads=scaled_lv_heads,
            )
            new_hybrid.append((new_cnt, new_ls))

    return replace(
        base,
        name=f"{base.name} ↓{scale_factor:.3f}×",
        hidden=new_hidden,
        num_layers=new_layers,
        num_q_heads=new_q,
        num_kv_heads=new_kv,
        d_expert=new_d_expert,
        dense_intermediate=0,  # auto-recompute from (K+N_sh)*d_expert
        hybrid_attn=new_hybrid,
        notes=f"Wind tunnel anchor at {scale_factor:.3f}× of {base.name}",
    )


def wind_tunnel_plan(
    target: MoEArchSpec,
    num_anchors: int = 5,
    smallest_total_b: float = 1.0,
    largest_total_b: float = 10.0,
    token_multiple: float = 100.0,
) -> list[tuple[MoEArchSpec, int]]:
    """Generate wind tunnel anchor schedule.

    Args:
        target: the final spec you want to verify scaling behavior for
        num_anchors: how many points (Ling uses 5)
        smallest/largest_total_b: scale range in billions of total params
        token_multiple: tokens per active param (MoE Chinchilla ≈ 25-35×)

    Returns:
        List of (anchor_spec, training_tokens) tuples, increasing in size.
    """
    import math
    target_total = target.total_params_no_mtp() / 1e9  # B
    # Logarithmic spacing of anchor sizes
    log_lo = math.log(smallest_total_b)
    log_hi = math.log(largest_total_b)
    anchor_sizes_b = [math.exp(log_lo + (log_hi - log_lo) * i / (num_anchors - 1))
                      for i in range(num_anchors)]

    plan = []
    for size_b in anchor_sizes_b:
        scale = size_b / target_total
        anchor = scale_down(target, scale)
        # Compute-optimal tokens: token_multiple × active_params
        tokens = int(token_multiple * anchor.active_params())
        plan.append((anchor, tokens))
    return plan


def wind_tunnel_summary(
    target: MoEArchSpec,
    plan: list[tuple[MoEArchSpec, int]] | None = None,
    num_gpus: int = 64,
    gpu_peak_tflops: float = 989.0,
    mfu: float = 0.45,
    price_per_gpu_hour: float = 3.0,
    **plan_kwargs,
) -> str:
    """Pretty-print wind tunnel plan with per-anchor cost + total."""
    if plan is None:
        plan = wind_tunnel_plan(target, **plan_kwargs)

    lines = [f"═══ WIND TUNNEL PLAN for {target.name} ═══", ""]
    lines.append(f"Target:  total={target.total_params_no_mtp()/1e9:.0f}B  "
                 f"active={target.active_params()/1e9:.1f}B")
    lines.append(f"GPU:     {num_gpus} H100 @ MFU {mfu} @ ${price_per_gpu_hour}/hr")
    lines.append("")
    lines.append(f"{'Anchor':<10} {'Total':>8} {'Active':>8} {'Tokens':>10} "
                 f"{'Days':>6} {'Cost ($)':>12}")
    lines.append("─" * 70)
    total_cost = 0.0
    total_days = 0.0
    for i, (anchor, tokens) in enumerate(plan, 1):
        c = anchor.training_cost(tokens, gpu_peak_tflops=gpu_peak_tflops,
                                 mfu=mfu, num_gpus=num_gpus,
                                 price_per_gpu_hour=price_per_gpu_hour)
        total_cost += c["estimated_cost_usd"]
        total_days += c["wall_clock_days"]
        lines.append(
            f"W{i:<9} "
            f"{anchor.total_params_no_mtp()/1e9:>7.2f}B "
            f"{anchor.active_params()/1e9:>7.2f}B "
            f"{tokens/1e9:>9.1f}B "
            f"{c['wall_clock_days']:>6.2f} "
            f"${c['estimated_cost_usd']:>11,.0f}"
        )
    lines.append("─" * 70)
    lines.append(f"{'TOTAL':<10} {' ':<8} {' ':<8} {' ':<10} "
                 f"{total_days:>6.1f} ${total_cost:>11,.0f}")
    full_train_cost = target.training_cost(25e12, num_gpus=256)['estimated_cost_usd']
    lines.append("")
    lines.append(f"Cost context:")
    lines.append(f"  Wind tunnel total:           ${total_cost:>11,.0f}")
    lines.append(f"  One full target training:    ${full_train_cost:>11,.0f}  (25T tokens, 256 GPU)")
    lines.append(f"  Wind tunnel = {total_cost/full_train_cost*100:.2f}% of one full training run")
    lines.append("")
    lines.append("⚠️ Cost depends heavily on token_multiple (default 100×active = compute-optimal).")
    lines.append("   Ling paper §2.3.3 uses ~50-100× compute-optimal per anchor for full scaling-law fit;")
    lines.append("   for quick validation (Muon proof, AttnRes test, etc.) 25-50× is enough.")
    return "\n".join(lines)
