"""Command-line interface for the MoE calculator.

Usage:
  python -m calc info <anchor>                 # show full spec breakdown
  python -m calc check <anchor>                # invariant checks
  python -m calc compare <a> <b> [<c> ...]     # side-by-side
  python -m calc list                          # list known anchors
  python -m calc all                           # summary table of all anchors
  python -m calc cost <anchor> <tokens_T>      # training cost estimate
  python -m calc parallel <anchor> <ep> <pp> <tp>  # parallelism check
"""
import sys
from .anchors import ANCHORS, get_anchor, list_anchors
from .compare import compare_specs, summary_table


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
        kwargs = {}
        for i, a in enumerate(argv[3:]):
            if a == "--gpus": kwargs["num_gpus"] = int(argv[3+i+1])
            if a == "--mfu": kwargs["mfu"] = float(argv[3+i+1])
            if a == "--price": kwargs["price_per_gpu_hour"] = float(argv[3+i+1])
        c = spec.training_cost(tokens, **kwargs)
        print(f"═══ Training cost: {spec.name} ═══\n")
        print(f"  Train tokens:    {tokens/1e12:.2f}T")
        print(f"  Total FLOPs:     {c['total_flops']:.2e}")
        print(f"  GPU-hours:       {c['total_gpu_hours']:.0f}")
        print(f"  Wall-clock days: {c['wall_clock_days']:.1f}  (on {kwargs.get('num_gpus', 256)} GPUs)")
        print(f"  Estimated $:     ${c['estimated_cost_usd']:,.0f}  (@ ${kwargs.get('price_per_gpu_hour', 3.0)}/GPU-hr)")
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
