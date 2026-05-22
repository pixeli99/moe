"""Command-line interface for the MoE calculator.

Usage:
  python -m calc info <anchor>                 # show full spec breakdown
  python -m calc check <anchor>                # invariant checks
  python -m calc compare <a> <b> [<c> ...]     # side-by-side
  python -m calc list                          # list known anchors
  python -m calc all                           # summary table of all anchors
  python -m calc cost <anchor> <tokens_T>      # training cost estimate
  python -m calc parallel <anchor> <ep> <pp> <tp>  # parallelism check
  python -m calc upscale <base> [--layers L] [--n N] [--method solar|upcycle|hybrid]
                                               # simulate L100 → L200 upscale
  python -m calc latency <anchor> [--ctx N] [--accept R]
                                               # decode latency + MTP speedup
  python -m calc wind-tunnel <target> [--num N] [--gpus G]
                                               # generate scaled-down anchors + cost
"""
import sys
from .anchors import ANCHORS, get_anchor, list_anchors


def _parse_flags(argv: list[str], start: int, flags: dict) -> dict:
    """Robust flag parser with bounds checking.

    Args:
        argv: full argv
        start: index from which to parse flags
        flags: dict mapping flag name (e.g. "--gpus") to (kwarg_name, converter)
                e.g. {"--gpus": ("num_gpus", int), "--mfu": ("mfu", float)}

    Returns: kwargs dict. Exits with error if a flag lacks a value.
    """
    out = {}
    i = start
    while i < len(argv):
        a = argv[i]
        if a in flags:
            if i + 1 >= len(argv):
                print(f"Error: flag '{a}' requires a value", file=sys.stderr)
                sys.exit(1)
            name, conv = flags[a]
            try:
                out[name] = conv(argv[i + 1])
            except (ValueError, TypeError) as e:
                print(f"Error: '{a} {argv[i+1]}' invalid ({e})", file=sys.stderr)
                sys.exit(1)
            i += 2
        else:
            i += 1
    return out
from .compare import compare_specs, summary_table
from .upscale import solar_dus, sparse_upcycle, hybrid_upscale, upscale_summary
from .wind_tunnel import wind_tunnel_summary


def main(argv: list[str] = None):
    argv = argv or sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    cmd = argv[0]

    if cmd == "list":
        for n in list_anchors():
            print(f"  {n:20s} {ANCHORS[n].name}")
        return 0

    if cmd == "all":
        specs = [get_anchor(n) for n in list_anchors()]
        print(summary_table(specs))
        return 0

    if cmd == "info":
        if len(argv) < 2:
            print("usage: calc info <anchor>"); return 1
        spec = get_anchor(argv[1])
        print(spec.summary())
        return 0

    if cmd == "check":
        if len(argv) < 2:
            print("usage: calc check <anchor>"); return 1
        spec = get_anchor(argv[1])
        print(f"═══ Invariant checks: {spec.name} ═══\n")
        for r in spec.check_invariants():
            print(f"  {r}")
        return 0

    if cmd == "compare":
        if len(argv) < 3:
            print("usage: calc compare <anchor1> <anchor2> [<anchor3> ...]"); return 1
        specs = [get_anchor(n) for n in argv[1:]]
        print(compare_specs(*specs))
        return 0

    if cmd == "cost":
        if len(argv) < 3:
            print("usage: calc cost <anchor> <tokens_T> [--gpus N] [--mfu M] [--price P]"); return 1
        spec = get_anchor(argv[1])
        tokens = float(argv[2]) * 1e12
        kwargs = _parse_flags(argv, 3, {
            "--gpus": ("num_gpus", int),
            "--mfu": ("mfu", float),
            "--price": ("price_per_gpu_hour", float),
        })
        c = spec.training_cost(tokens, **kwargs)
        print(f"═══ Training cost: {spec.name} ═══\n")
        print(f"  Train tokens:    {tokens/1e12:.2f}T")
        print(f"  Total FLOPs:     {c['total_flops']:.2e}")
        print(f"  GPU-hours:       {c['total_gpu_hours']:.0f}")
        print(f"  Wall-clock days: {c['wall_clock_days']:.1f}  (on {kwargs.get('num_gpus', 256)} GPUs)")
        print(f"  Estimated $:     ${c['estimated_cost_usd']:,.0f}  (@ ${kwargs.get('price_per_gpu_hour', 3.0)}/GPU-hr)")
        return 0

    if cmd == "latency":
        if len(argv) < 2:
            print("usage: calc latency <anchor> [--ctx N] [--accept R] [--bw GBPS] [--bw-frac F]"); return 1
        spec = get_anchor(argv[1])
        parsed = _parse_flags(argv, 2, {
            "--ctx": ("kv_seq_len", int),
            "--bw": ("hbm_bandwidth_gbps", float),
            "--bw-frac": ("effective_bw_fraction", float),
            "--accept": ("__accept", float),
        })
        accept_rate = parsed.pop("__accept", 0.75)
        kwargs = parsed
        d = spec.decode_latency_ms(**kwargs)
        mtp_x = spec.mtp_speedup(accept_rate=accept_rate)
        print(f"═══ Inference latency: {spec.name} ═══\n")
        print(f"  Single-stream decode (batch=1, memory-bound):")
        print(f"     active params:    {d['active_params_GB']:.2f} GB ({d['active_latency_ms']:.2f} ms)")
        print(f"     KV cache traversed: {d['kv_cache_GB']:.2f} GB ({d['kv_latency_ms']:.2f} ms)")
        print(f"     per-token total:  {d['total_per_token_ms']:.2f} ms")
        print(f"     base TPS:         {d['tokens_per_sec']:.1f} tok/s")
        if spec.mtp_depth:
            print(f"  MTP speedup (D={spec.mtp_depth}, accept {accept_rate:.0%}):")
            print(f"     effective TPS:    {d['tokens_per_sec'] * mtp_x:.1f} tok/s "
                  f"({mtp_x:.2f}× single-token)")
            print(f"     effective ms/tok: {d['total_per_token_ms'] / mtp_x:.2f} ms")
        ctx = kwargs.get("kv_seq_len", 4096)
        prefill = spec.prefill_throughput(seq_len=ctx)
        print(f"  Prefill (compute-bound, seq_len={ctx}):")
        print(f"     time per token:   {prefill['time_per_token_ms']:.3f} ms")
        print(f"     prefill TPS:      {prefill['tokens_per_sec']:,.0f} tok/s")
        print(f"     attn quadratic:   {prefill['attn_quad_pct']:.1f}% of FLOPs")
        return 0

    if cmd == "wind-tunnel":
        if len(argv) < 2:
            print("usage: calc wind-tunnel <target> [--num N] [--gpus G] [--small SIZE_B] [--large SIZE_B]"); return 1
        target = get_anchor(argv[1])
        kwargs = _parse_flags(argv, 2, {
            "--num": ("num_anchors", int),
            "--gpus": ("num_gpus", int),
            "--small": ("smallest_total_b", float),
            "--large": ("largest_total_b", float),
        })
        print(wind_tunnel_summary(target, **kwargs))
        return 0

    if cmd == "upscale":
        if len(argv) < 2:
            print("usage: calc upscale <base> [--layers L] [--n N] [--method solar|upcycle|hybrid] "
                  "[--gpus N] [--mfu M] [--depth-tokens T] [--expert-tokens T]"); return 1
        base = get_anchor(argv[1])
        parsed = _parse_flags(argv, 2, {
            "--layers": ("__layers", int),
            "--n": ("__n", int),
            "--method": ("__method", str),
            "--gpus": ("num_gpus", int),
            "--mfu": ("mfu", float),
            "--depth-tokens": ("__depth_t", float),
            "--expert-tokens": ("__expert_t", float),
        })
        target_layers = parsed.pop("__layers", base.num_layers)
        target_n = parsed.pop("__n", base.n_routed)
        method = parsed.pop("__method", None)
        if "__depth_t" in parsed:
            parsed["depth_recovery_tokens"] = parsed.pop("__depth_t") * 1e12
        if "__expert_t" in parsed:
            parsed["expert_recovery_tokens"] = parsed.pop("__expert_t") * 1e12
        kwargs = parsed
        # Auto-detect method
        if method is None:
            if target_layers != base.num_layers and target_n != base.n_routed:
                method = "hybrid"
            elif target_layers != base.num_layers:
                method = "solar"
            elif target_n != base.n_routed:
                method = "upcycle"
            else:
                print("Nothing to do: target_layers and target_n both match base"); return 1
        # Build target
        if method == "solar":
            target = solar_dus(base, target_layers)
        elif method == "upcycle":
            target = sparse_upcycle(base, target_n)
        elif method == "hybrid":
            target = hybrid_upscale(base, target_layers, target_n)
        else:
            print(f"Unknown method: {method}"); return 1
        print(upscale_summary(base, target, **kwargs))
        return 0

    if cmd == "parallel":
        if len(argv) < 5:
            print("usage: calc parallel <anchor> <ep> <pp> <tp>"); return 1
        spec = get_anchor(argv[1])
        ep, pp, tp = int(argv[2]), int(argv[3]), int(argv[4])
        print(f"═══ Parallelism check: {spec.name} (EP={ep}, PP={pp}, TP={tp}) ═══\n")
        for r in spec.check_parallelism(ep, pp, tp):
            print(f"  {r}")
        return 0

    print(f"Unknown command: {cmd}")
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
