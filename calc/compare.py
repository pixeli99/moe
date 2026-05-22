"""Side-by-side comparison of multiple specs."""
from .spec import MoEArchSpec


def _fmt(n):
    if isinstance(n, float):
        return f"{n:.3f}" if abs(n) < 100 else f"{n:.1f}"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return str(n)


def compare_specs(*specs: MoEArchSpec) -> str:
    """Side-by-side comparison table."""
    def _attn_label(s):
        if s.hybrid_attn:
            return "hybrid (" + ", ".join(f"{c}×{ls.kind}" for c, ls in s.hybrid_attn) + ")"
        return s.attn_type

    def _qkv_label(s):
        if s.hybrid_attn:
            return "(hybrid; see Attn type row)"
        return f"{s.num_q_heads}/{s.num_kv_heads}"

    def _headdim_label(s):
        if s.hybrid_attn:
            dims = sorted(set(ls.head_dim for _, ls in s.hybrid_attn))
            return "/".join(str(d) for d in dims)
        return s.head_dim

    def _kv_per_tok_label(s):
        if s.hybrid_attn:
            # Show effective KV per token at large ctx (no SWA capping)
            total = s.kv_cache_bytes_total(1, 1, "bf16")  # per token = bytes at ctx=1
            return f"{total} (mixed)"
        return s.kv_cache_bytes_per_token_per_layer("bf16") * s.num_layers

    rows = [
        ("Layers", lambda s: s.num_layers),
        ("Hidden", lambda s: s.hidden),
        ("Attn type", _attn_label),
        ("Q / KV heads", _qkv_label),
        ("head_dim", _headdim_label),
        ("N_routed", lambda s: s.n_routed),
        ("Top-K", lambda s: s.top_k),
        ("N_shared", lambda s: s.n_shared),
        ("d_expert", lambda s: s.d_expert),
        ("first_k_dense", lambda s: s.first_k_dense),
        ("dense_int", lambda s: s.dense_intermediate),
        ("vocab", lambda s: s.vocab_size),
        ("MTP D", lambda s: s.mtp_depth),
        ("───────────", lambda s: ""),
        ("Total params", lambda s: s.total_params_no_mtp()),
        ("Active strict", lambda s: s.active_params(False)),
        ("Active V3 口径", lambda s: s.active_params(True)),
        ("active/total", lambda s: s.activation_rate()["active_over_total"]),
        ("expert-slot frac", lambda s: s.activation_rate()["expert_slot"]),
        ("L/√H", lambda s: s.l_over_sqrt_h()),
        ("d_expert/hidden", lambda s: s.d_expert_over_hidden()),
        ("Attn:FFN ratio", lambda s: f"1:{1/s.attn_ffn_ratio():.2f}"),
        ("───────────", lambda s: ""),
        ("KV @ 32K BF16 (MB)", lambda s: s.kv_cache_bytes_total(32768, 1, "bf16") / 1024 / 1024),
        ("KV @ 128K BF16 (GB)", lambda s: s.kv_cache_bytes_total(128*1024, 1, "bf16") / 1024**3),
        ("KV @ 1M BF16 (GB)", lambda s: s.kv_cache_bytes_total(1024*1024, 1, "bf16") / 1024**3),
    ]

    name_w = 22
    col_w = max(28, max(len(s.name) for s in specs) + 2)

    header = f"{'':<{name_w}}" + "".join(f"{s.name:<{col_w}}" for s in specs)
    lines = [header, "─" * (name_w + col_w * len(specs))]
    for label, fn in rows:
        vals = []
        for s in specs:
            try:
                v = fn(s)
                vals.append(_fmt(v) if not isinstance(v, str) else v)
            except Exception as e:
                vals.append(f"<err: {type(e).__name__}>")
        lines.append(f"{label:<{name_w}}" + "".join(f"{v:<{col_w}}" for v in vals))
    return "\n".join(lines)


def summary_table(specs: list[MoEArchSpec]) -> str:
    """Compact single-line-per-spec summary (≤ 6 columns)."""
    lines = [
        f"{'Name':<40s} {'Total':>8s} {'Active':>8s} {'A/T%':>6s} {'L/√H':>6s} {'Attn:FFN':>10s}",
        "─" * 86,
    ]
    for s in specs:
        rate = s.activation_rate()["active_over_total"] * 100
        lines.append(
            f"{s.name:<40s} "
            f"{_fmt(s.total_params_no_mtp()):>8s} "
            f"{_fmt(s.active_params()):>8s} "
            f"{rate:>5.1f}% "
            f"{s.l_over_sqrt_h():>6.2f} "
            f"1:{1/s.attn_ffn_ratio():>8.2f}"
        )
    return "\n".join(lines)
