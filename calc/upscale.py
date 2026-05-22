"""MoE upscale path simulator.

Given a base spec (e.g. L100) + an upscale method, compute the target spec
(e.g. L200) and the recovery training cost.

Three supported methods:
  1. SOLAR-DUS (depth doubling, Upstage 2023, arXiv 2312.15166)
     - Copy base → drop middle layers from each copy → concat
     - 32L → 48L: drop 8 middle layers per copy
     - Layer count up, expert count fixed
     - Active params scale up linearly with layer count

  2. Sparse Upcycling (expert replication, Komatsuzaki 2023, arXiv 2212.05055)
     - Duplicate some experts + small noise (σ=0.01)
     - Re-initialize router for new experts
     - N_routed up, layer count fixed
     - Active params nearly unchanged (K still selects same number)

  3. Hybrid = SOLAR-DUS then Sparse Upcycle (no public 100B+ precedent;
     recommended for L100 → L200 based on 38_100b_to_200b_gap.md analysis)

Recovery training token estimates from literature:
  - SOLAR-DUS recovery (dense, paper): ~3B tokens for 7B base
  - Sparse upcycling recovery (Komatsuzaki §3): ~50% of original budget
  - For 100B+ MoE we conservatively estimate 3T / 1.5T tokens per stage

These are heuristics; no public MoE → bigger MoE upscale has been published.
Wind tunnel at smaller scale (1B → 2B) is strongly recommended before doing
this for real (see 38_100b_to_200b_gap.md §7).
"""
from __future__ import annotations
from dataclasses import replace
from .spec import MoEArchSpec


def solar_dus(base: MoEArchSpec, target_layers: int) -> MoEArchSpec:
    """SOLAR-DUS depth upscale.

    Args:
        base: source spec
        target_layers: desired final layer count. Must equal 2(L - m) for some
                      integer m ≥ 0 (i.e., target_layers ≤ 2 × base.num_layers
                      and (2L - target) is non-negative even).

    Returns:
        New spec with num_layers = target_layers, all other fields preserved.

    Notes:
        Active params per token scale up by (target_layers / base.num_layers).
        Total params scale similarly except embedding/MTP head are unchanged.
    """
    L = base.num_layers
    if target_layers > 2 * L:
        raise ValueError(f"target_layers={target_layers} exceeds 2 × {L} = {2*L} (SOLAR-DUS upper bound)")
    if target_layers < L:
        raise ValueError(f"target_layers={target_layers} < base {L} — use layer drop, not SOLAR-DUS")
    drop_total = 2 * L - target_layers
    if drop_total % 2 != 0:
        raise ValueError(f"target_layers must be 2L - 2k for integer k; got drop_total={drop_total}")
    drop_per_copy = drop_total // 2

    return replace(
        base,
        name=f"{base.name} [SOLAR-DUS to {target_layers}L]",
        num_layers=target_layers,
        notes=(
            f"SOLAR-DUS: copy base → drop {drop_per_copy} middle layers each → concat. "
            f"2 × ({L}L − {drop_per_copy}L) = {target_layers}L. "
            f"Active params ↑ {target_layers/L:.2f}×. Recovery ~3T tokens."
        ),
    )


def sparse_upcycle(base: MoEArchSpec, target_n_routed: int) -> MoEArchSpec:
    """Sparse Upcycling — duplicate experts to grow N_routed.

    Args:
        base: source spec
        target_n_routed: new routed expert count. Should be ≥ base.n_routed.
                        Cleanest when (target % base) divides evenly, e.g.
                        128 → 192 (+50%), 128 → 256 (×2), 256 → 320 (+25%).

    Returns:
        New spec with n_routed = target_n_routed; K, hidden, layers unchanged.

    Notes:
        Active params per token essentially unchanged (top-K still picks K).
        Total params grow proportional to routed expert count.
    """
    if target_n_routed < base.n_routed:
        raise ValueError(f"target_n_routed={target_n_routed} < base {base.n_routed}")
    if target_n_routed == base.n_routed:
        return base

    growth = target_n_routed / base.n_routed
    return replace(
        base,
        name=f"{base.name} [upcycle N={base.n_routed}→{target_n_routed}]",
        n_routed=target_n_routed,
        notes=(
            f"Sparse Upcycle (Komatsuzaki 2023): {base.n_routed} → {target_n_routed} experts "
            f"({growth:.2f}×). Each base expert duplicated + σ=0.01 noise. "
            f"Router re-init for {target_n_routed - base.n_routed} new experts. "
            f"Active params ~unchanged (K still {base.top_k}). Recovery ~1.5T tokens."
        ),
    )


def hybrid_upscale(
    base: MoEArchSpec,
    target_layers: int,
    target_n_routed: int,
) -> MoEArchSpec:
    """Hybrid: SOLAR-DUS depth then sparse upcycle experts.

    Convenience wrapper: solar_dus(base, target_layers) then sparse_upcycle.
    """
    step1 = solar_dus(base, target_layers)
    step2 = sparse_upcycle(step1, target_n_routed)
    L = base.num_layers
    return replace(
        step2,
        name=f"{base.name} [hybrid → {target_layers}L × N={target_n_routed}]",
        notes=(
            f"Hybrid upscale: (1) SOLAR-DUS {L}L → {target_layers}L "
            f"(2) Upcycle N={base.n_routed} → {target_n_routed}. "
            f"Total recovery ~4.5T tokens (3T depth + 1.5T expert). "
            f"WARNING: no public 100B+ MoE precedent."
        ),
    )


def upscale_summary(
    base: MoEArchSpec,
    target: MoEArchSpec,
    depth_recovery_tokens: float = 3e12,
    expert_recovery_tokens: float = 1.5e12,
    gpu_peak_tflops: float = 989.0,
    mfu: float = 0.45,
    num_gpus: int = 384,
    price_per_gpu_hour: float = 3.0,
) -> str:
    """Pretty-print before/after comparison + cost estimate."""
    from .compare import compare_specs

    # Detect what changed
    stages = []
    if target.num_layers != base.num_layers:
        stages.append(("Depth SOLAR-DUS",
                       f"{base.num_layers}L → {target.num_layers}L",
                       depth_recovery_tokens))
    if target.n_routed != base.n_routed:
        stages.append(("Expert Sparse Upcycle",
                       f"N_routed {base.n_routed} → {target.n_routed}",
                       expert_recovery_tokens))

    # Cost: use TARGET active (post-upscale) for recovery FLOPs estimate
    total_recovery_tokens = sum(t for _, _, t in stages)

    # Per-stage cost (active grows progressively, but we approximate with target active)
    lines = ["═══ UPSCALE PLAN ═══", ""]
    lines.append(f"BASE:   {base.name}")
    lines.append(f"        total={base.total_params_no_mtp()/1e9:.1f}B  "
                 f"active={base.active_params()/1e9:.1f}B  "
                 f"L={base.num_layers}  N={base.n_routed}")
    lines.append(f"TARGET: {target.name}")
    lines.append(f"        total={target.total_params_no_mtp()/1e9:.1f}B  "
                 f"active={target.active_params()/1e9:.1f}B  "
                 f"L={target.num_layers}  N={target.n_routed}")
    lines.append("")
    lines.append(f"Param growth: total ×{target.total_params_no_mtp()/base.total_params_no_mtp():.2f}, "
                 f"active ×{target.active_params()/base.active_params():.2f}")
    lines.append("")
    lines.append("── Upscale stages ──")

    cumulative_cost = 0.0
    cumulative_days = 0.0
    for i, (stage_name, change, tokens) in enumerate(stages, 1):
        c = target.training_cost(
            tokens,
            gpu_peak_tflops=gpu_peak_tflops,
            mfu=mfu,
            num_gpus=num_gpus,
            price_per_gpu_hour=price_per_gpu_hour,
        )
        lines.append(f"  Stage {i}: {stage_name}")
        lines.append(f"           {change}")
        lines.append(f"           {tokens/1e12:.1f}T recovery tokens · {c['wall_clock_days']:.1f} days · "
                     f"${c['estimated_cost_usd']:,.0f}")
        cumulative_cost += c["estimated_cost_usd"]
        cumulative_days += c["wall_clock_days"]
    lines.append("")
    lines.append(f"  TOTAL upscale: {total_recovery_tokens/1e12:.1f}T tokens · "
                 f"{cumulative_days:.1f} wall-clock days · ${cumulative_cost:,.0f}")
    lines.append(f"        (on {num_gpus} H100 @ MFU {mfu} @ ${price_per_gpu_hour}/GPU-hr)")
    lines.append("")
    lines.append(target.notes)
    lines.append("")
    lines.append("── Side-by-side ──")
    lines.append(compare_specs(base, target))
    return "\n".join(lines)
