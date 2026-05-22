"""MoE architecture spec + parameter/FLOPs/memory accounting.

All formulas follow:
- SwiGLU FFN: 3 matrices (gate, up, down), each shape (hidden, intermediate)
- Standard MHA/GQA: Q + K + V + O projections only (no bias)
- MLA (DeepSeek-V2/V3): low-rank Q/KV compression + RoPE-decoupled K + V uncompress
- MoE per-token active = (K_routed + N_shared) × per-expert SwiGLU
- Dense layer 0..first_k_dense use ordinary SwiGLU with `dense_intermediate`

Numbers verified against DeepSeek-V3 paper (Eq. 1-25), Ling 2.0 paper Table 1,
GLM-4.5 paper Table 1, Step 3.5 Flash paper §2.2, K2 paper, dots1 paper.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

AttnType = Literal["mha", "gqa", "mla"]
AttnKind = Literal["softmax", "swa", "linear", "mamba"]
DTypeBytes = {"bf16": 2, "fp16": 2, "fp8": 1, "fp32": 4, "int8": 1, "int4": 0.5}


@dataclass
class AttnLayerSpec:
    """Per-layer attention spec for hybrid models.

    Use cases:
      - Step 3.5 Flash: alternating SWA (W=512) and Full softmax layers
      - Qwen3-Next: alternating Gated DeltaNet (linear) and Gated Attention
      - Jamba / Granite 4: Mamba state-space layers mixed with softmax
    """
    kind: AttnKind = "softmax"
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    # SWA-specific
    window_size: Optional[int] = None  # e.g. 512 for Step 3.5 Flash

    # Linear attention specific (Gated DeltaNet)
    linear_value_head_dim: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    conv_kernel: int = 0  # 1D conv kernel before linear attention

    # Mamba state dim (rough approximation)
    state_dim: Optional[int] = None

    def params(self, hidden: int) -> int:
        """Per-layer attention parameter count."""
        if self.kind in ("softmax", "swa"):
            q = hidden * self.num_q_heads * self.head_dim
            k = hidden * self.num_kv_heads * self.head_dim
            v = hidden * self.num_kv_heads * self.head_dim
            o = self.num_q_heads * self.head_dim * hidden
            return q + k + v + o
        elif self.kind == "linear":
            v_heads = self.linear_num_value_heads or self.num_q_heads
            v_d = self.linear_value_head_dim or self.head_dim
            q = hidden * self.num_q_heads * self.head_dim
            k = hidden * self.num_kv_heads * self.head_dim
            v = hidden * v_heads * v_d
            o = v_heads * v_d * hidden
            gate = hidden * v_heads * v_d  # output gate
            conv = hidden * self.conv_kernel  # 1D conv weights
            return q + k + v + o + gate + conv
        elif self.kind == "mamba":
            # Approximate: Mamba ≈ 3× hidden² + state contribution
            return 3 * hidden * hidden + hidden * (self.state_dim or 16) * 2
        return 0

    def kv_cache_bytes_at_ctx(self, ctx: int, dtype_bytes: float) -> int:
        """KV cache or state memory at the given context length."""
        if self.kind == "softmax":
            return int(2 * self.num_kv_heads * self.head_dim * dtype_bytes * ctx)
        elif self.kind == "swa":
            eff = min(ctx, self.window_size or ctx)
            return int(2 * self.num_kv_heads * self.head_dim * dtype_bytes * eff)
        elif self.kind == "linear":
            v_heads = self.linear_num_value_heads or self.num_q_heads
            v_d = self.linear_value_head_dim or self.head_dim
            return int(self.num_kv_heads * v_heads * v_d * dtype_bytes)
        elif self.kind == "mamba":
            return int((self.state_dim or 16) * dtype_bytes * 2)
        return 0

    def attn_quadratic_flops(self, seq_len: int) -> int:
        """TOTAL prefill attention compute (QK^T + softmax + V) over N tokens.

        Coefficient 4 = 2 (QK^T multiply-add) + 2 (Attn × V multiply-add).
        For SWA, capped at window_size. For linear/mamba, O(N) not O(N²).
        """
        if self.kind == "softmax":
            return 4 * self.num_q_heads * self.head_dim * seq_len * seq_len
        elif self.kind == "swa":
            eff = min(seq_len, self.window_size or seq_len)
            return 4 * self.num_q_heads * self.head_dim * seq_len * eff
        elif self.kind == "linear":
            v_d = self.linear_value_head_dim or self.head_dim
            # Linear attention: O(N × d_k × d_v) instead of O(N²)
            return 2 * self.num_q_heads * self.head_dim * v_d * seq_len
        elif self.kind == "mamba":
            return 2 * seq_len * (self.state_dim or 16) * 8
        return 0

    def attn_per_token_flops(self, ctx_len: int) -> int:
        """Per-token decode-style attention compute (this token vs past ctx_len).

        Coefficient 4 = same QK + AV breakdown but only for 1 query.
        SWA caps at window. Linear is O(d²) constant per token.
        """
        if self.kind == "softmax":
            return 4 * self.num_q_heads * self.head_dim * ctx_len
        elif self.kind == "swa":
            eff = min(ctx_len, self.window_size or ctx_len)
            return 4 * self.num_q_heads * self.head_dim * eff
        elif self.kind == "linear":
            v_d = self.linear_value_head_dim or self.head_dim
            return 2 * self.num_q_heads * self.head_dim * v_d
        elif self.kind == "mamba":
            return 16 * (self.state_dim or 16)
        return 0


@dataclass
class CheckResult:
    """One invariant/feasibility check outcome."""
    name: str
    ok: bool
    message: str
    severity: Literal["error", "warn", "info"] = "info"

    def __str__(self):
        icon = {"error": "❌", "warn": "⚠️ ", "info": "ℹ️ "}[self.severity if not self.ok else "info"]
        if self.ok:
            icon = "✅"
        return f"{icon} {self.name}: {self.message}"


@dataclass
class MoEArchSpec:
    """A MoE language model architecture specification.

    Required fields for any spec; MLA fields only needed when attn_type=='mla'.
    Default values match V3/Ling派 conventions where possible.
    """

    # Identity
    name: str

    # Backbone shape
    hidden: int
    num_layers: int           # total transformer blocks (including dense + MoE)
    head_dim: int = 128       # standard for GQA/MHA; MLA uses qk_nope+qk_rope

    # Attention
    num_q_heads: int = 32
    num_kv_heads: int = 8
    attn_type: AttnType = "gqa"

    # MLA fields (only used when attn_type=='mla')
    q_lora_rank: Optional[int] = None       # None means no Q LoRA (V2-Lite style)
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None  # content part of QK
    qk_rope_head_dim: Optional[int] = None  # decoupled RoPE part of QK
    v_head_dim: Optional[int] = None        # value head dim (no RoPE)

    # MoE
    n_routed: int = 64
    top_k: int = 8                          # routed experts per token
    n_shared: int = 1                       # always-active shared experts
    d_expert: int = 1408                    # per-expert SwiGLU intermediate

    # Dense first-K layers
    first_k_dense: int = 1                  # number of dense (non-MoE) layers at start
    dense_intermediate: int = 0             # FFN intermediate for dense layers
                                            # 0 means "auto = (K+N_sh)*d_expert"

    # Tokenizer
    vocab_size: int = 128256
    tied_embedding: bool = False

    # MTP
    mtp_depth: int = 0                      # 0 means no MTP
    mtp_is_moe: bool = True                 # V3 style; LongCat / Step set False

    # Hybrid attention layer schedule
    # If set, list of (count, AttnLayerSpec) tuples whose counts sum to num_layers.
    # Overrides the uniform num_q_heads/num_kv_heads/attn_type fields above.
    hybrid_attn: Optional[list[tuple[int, AttnLayerSpec]]] = None

    # Notes
    notes: str = ""

    def __post_init__(self):
        if self.dense_intermediate == 0:
            # Default to compute-equivalent (= MoE layer per-token compute)
            self.dense_intermediate = (self.top_k + self.n_shared) * self.d_expert
        if self.attn_type == "mla":
            assert self.kv_lora_rank, "MLA needs kv_lora_rank"
            assert self.qk_nope_head_dim and self.qk_rope_head_dim, "MLA needs qk_nope/rope dims"
            assert self.v_head_dim, "MLA needs v_head_dim"
        if self.hybrid_attn is not None:
            total_count = sum(c for c, _ in self.hybrid_attn)
            assert total_count == self.num_layers, (
                f"hybrid_attn counts sum to {total_count}, but num_layers={self.num_layers}"
            )

    # ---------------------- Parameter accounting ----------------------

    def attn_params_total(self) -> int:
        """Total attention params summed across all layers (hybrid-aware)."""
        if self.hybrid_attn is not None:
            return sum(count * spec.params(self.hidden) for count, spec in self.hybrid_attn)
        return self.attn_params_per_layer() * self.num_layers

    def attn_params_per_layer(self) -> int:
        """Per-layer attention params (no bias).

        For hybrid: returns weighted AVERAGE across layer types.
        For uniform: returns single-layer count.

        GQA/MHA: Q + K + V + O projection.
        MLA: q_lora down + q_lora up + kv down (kv_a) + k_pe + kv up + O.
        """
        if self.hybrid_attn is not None:
            return self.attn_params_total() // self.num_layers

        h = self.hidden
        if self.attn_type in ("mha", "gqa"):
            q = h * self.num_q_heads * self.head_dim
            k = h * self.num_kv_heads * self.head_dim
            v = h * self.num_kv_heads * self.head_dim
            o = self.num_q_heads * self.head_dim * h
            return q + k + v + o
        else:  # MLA
            q_nope = self.qk_nope_head_dim
            q_rope = self.qk_rope_head_dim
            v_dim = self.v_head_dim
            nh = self.num_q_heads
            kv_r = self.kv_lora_rank

            # Q path
            if self.q_lora_rank:
                q_a = h * self.q_lora_rank
                q_b = self.q_lora_rank * nh * (q_nope + q_rope)
                q_total = q_a + q_b
            else:
                # V2-Lite: no Q LoRA; direct Q
                q_total = h * nh * (q_nope + q_rope)

            # KV path
            kv_a = h * (kv_r + q_rope)   # kv_a combines compressed_kv + k_pe (key with RoPE shared across heads)
            kv_b = kv_r * nh * (q_nope + v_dim)  # up-projection for K_nope and V

            # O
            o = nh * v_dim * h

            return q_total + kv_a + kv_b + o

    def ffn_dense_per_layer(self) -> int:
        """Per-layer FFN params for a DENSE layer (SwiGLU 3 matrices)."""
        return 3 * self.hidden * self.dense_intermediate

    def ffn_moe_total_per_layer(self) -> int:
        """Per-layer total MoE FFN params (all routed experts + all shared)."""
        per_expert = 3 * self.hidden * self.d_expert
        return (self.n_routed + self.n_shared) * per_expert

    def ffn_moe_active_per_layer(self) -> int:
        """Per-token active MoE FFN params (K routed + N_shared, SwiGLU)."""
        per_expert = 3 * self.hidden * self.d_expert
        return (self.top_k + self.n_shared) * per_expert

    def router_per_layer(self) -> int:
        """Router weight matrix (just a small linear)."""
        return self.hidden * self.n_routed

    def embedding_params(self) -> int:
        """Input embedding + LM head (untied -> 2x)."""
        emb = self.vocab_size * self.hidden
        return emb if self.tied_embedding else 2 * emb

    def mtp_params(self) -> int:
        """MTP module(s) per V3-style chain."""
        if self.mtp_depth == 0:
            return 0
        # Each MTP module ≈ 1 transformer block + a linear projection + (shared embedding/head)
        attn = self.attn_params_per_layer()
        if self.mtp_is_moe:
            ffn = self.ffn_moe_total_per_layer() + self.router_per_layer()
        else:
            ffn = self.ffn_dense_per_layer()
        # Linear projection layer M_k: 2d -> d
        proj = 2 * self.hidden * self.hidden
        # 2 RMSNorm in projection (negligible)
        per_module = attn + ffn + proj
        return self.mtp_depth * per_module

    def params_breakdown(self) -> dict[str, int]:
        """Detailed per-module total params (hybrid-aware on attention)."""
        n_moe_layers = self.num_layers - self.first_k_dense
        out = {
            "embedding": self.embedding_params(),
            "attn_total": self.attn_params_total(),  # hybrid-aware
            "dense_ffn": self.ffn_dense_per_layer() * self.first_k_dense,
            "moe_ffn_routed": self.n_routed * 3 * self.hidden * self.d_expert * n_moe_layers,
            "moe_ffn_shared": self.n_shared * 3 * self.hidden * self.d_expert * n_moe_layers,
            "router": self.router_per_layer() * n_moe_layers,
            "mtp": self.mtp_params(),
            # RMSNorm / bias / etc. are sub-MB and ignored
        }
        return out

    def total_params(self) -> int:
        """Total model parameters (excluding MTP if you want base-only; include here)."""
        return sum(self.params_breakdown().values())

    def total_params_no_mtp(self) -> int:
        b = self.params_breakdown()
        b.pop("mtp", None)
        return sum(b.values())

    def active_params(self, include_embedding: bool = False) -> int:
        """Per-token active params.

        Conventions:
        - V3 paper / Ling paper report "activated params" INCLUDING embedding lookup.
        - Strict (compute-relevant) excludes embedding (lookup is O(1) memory only).

        We default to strict (no embedding); set include_embedding=True for V3 口径.
        """
        n_moe = self.num_layers - self.first_k_dense
        a = {
            "attn": self.attn_params_total(),  # hybrid-aware
            "dense_ffn": self.ffn_dense_per_layer() * self.first_k_dense,
            "moe_active": self.ffn_moe_active_per_layer() * n_moe,
            "router": self.router_per_layer() * n_moe,  # small but counted
        }
        if include_embedding:
            a["embedding"] = self.embedding_params()
        return sum(a.values())

    # ---------------------- Ratios & invariants ----------------------

    def attn_ffn_ratio(self) -> float:
        """Per-layer attention params / per-layer MoE-active FFN params."""
        return self.attn_params_per_layer() / self.ffn_moe_active_per_layer()

    def activation_rate(self) -> dict[str, float]:
        """Three rates discussed in 28_open_source_moe_catalog & feedback_unit_precision."""
        return {
            "active_over_total": self.active_params() / self.total_params_no_mtp(),
            "expert_slot": (self.top_k + self.n_shared) / (self.n_routed + self.n_shared),
            "routed_only": self.top_k / self.n_routed,
        }

    def l_over_sqrt_h(self) -> float:
        """Depth/width ratio (Krajewski-style)."""
        return self.num_layers / (self.hidden ** 0.5)

    def d_expert_over_hidden(self) -> float:
        return self.d_expert / self.hidden

    # ---------------------- KV cache ----------------------

    def kv_cache_bytes_per_token_per_layer(self, dtype: str = "bf16") -> int:
        """Per-token, per-layer KV cache bytes (uniform; not for hybrid)."""
        b = DTypeBytes[dtype]
        if self.attn_type == "mla":
            return int((self.kv_lora_rank + self.qk_rope_head_dim) * b)
        return int(2 * self.num_kv_heads * self.head_dim * b)

    def kv_cache_bytes_total(self, seq_len: int, batch: int = 1, dtype: str = "bf16") -> int:
        """Total KV cache bytes across ALL layers at given context length.

        Hybrid-aware: SWA layers cap at window size; linear/Mamba layers have
        fixed state size (independent of seq_len).
        """
        b = DTypeBytes[dtype]
        if self.hybrid_attn is not None:
            total = 0
            for count, layer_spec in self.hybrid_attn:
                total += count * layer_spec.kv_cache_bytes_at_ctx(seq_len, b)
            return total * batch
        per = self.kv_cache_bytes_per_token_per_layer(dtype)
        return per * seq_len * batch * self.num_layers

    # ---------------------- FLOPs ----------------------

    def per_token_forward_flops(self, seq_len: int = 4096) -> int:
        """Approximate per-token forward FLOPs (decode-style: this token vs ctx).

        Hybrid-aware: each AttnLayerSpec contributes its own quadratic term.
        """
        # Projection FLOPs (linear): 2 × params for forward
        attn_proj = 2 * self.attn_params_total()  # hybrid-aware

        # Attention QK + AV (per token, this token vs seq_len past)
        if self.hybrid_attn is not None:
            attn_qk = sum(
                count * ls.attn_per_token_flops(seq_len)
                for count, ls in self.hybrid_attn
            )
        else:
            if self.attn_type == "mla":
                head_d = self.qk_nope_head_dim + self.qk_rope_head_dim
            else:
                head_d = self.head_dim
            attn_qk = 4 * self.num_q_heads * head_d * seq_len * self.num_layers

        n_moe = self.num_layers - self.first_k_dense
        ffn_active = 2 * (
            self.ffn_dense_per_layer() * self.first_k_dense +
            self.ffn_moe_active_per_layer() * n_moe +
            self.router_per_layer() * n_moe
        )
        lm_head = 2 * self.vocab_size * self.hidden
        return attn_proj + attn_qk + ffn_active + lm_head

    def training_flops(self, train_tokens: int) -> int:
        """Total training FLOPs ≈ 6 × (active + lm_head + MTP) × tokens.

        Includes:
        - active params (strict): attn, dense_ffn, moe_active, router
        - LM head projection (vocab × hidden, fires per token)
        - MTP modules (trained alongside main model in V3/Ling/Step)

        Excludes embedding lookup (O(1) memory, not compute).
        """
        lm_head = self.vocab_size * self.hidden
        mtp = self.mtp_params()  # full MTP modules trained per step
        total_train_active = self.active_params(include_embedding=False) + lm_head + mtp
        return 6 * total_train_active * train_tokens

    def training_cost(
        self,
        train_tokens: int,
        gpu_peak_tflops: float = 989.0,  # H100 BF16 dense; H800 is 75% of this
        mfu: float = 0.45,
        num_gpus: int = 256,
        price_per_gpu_hour: float = 3.0,
    ) -> dict[str, float]:
        flops = self.training_flops(train_tokens)
        sec = flops / (gpu_peak_tflops * 1e12 * mfu)
        return {
            "total_flops": flops,
            "total_gpu_seconds": sec,
            "wall_clock_days": sec / num_gpus / 86400,
            "total_gpu_hours": sec / 3600,
            "estimated_cost_usd": sec / 3600 * price_per_gpu_hour,
        }

    # ---------------------- Inference latency ----------------------

    def decode_latency_ms(
        self,
        dtype: str = "bf16",
        hbm_bandwidth_gbps: float = 3350.0,  # H100 HBM3 = 3.35 TB/s
        effective_bw_fraction: float = 0.40,  # realistic Megatron / vLLM
        kv_seq_len: int = 4096,
    ) -> dict[str, float]:
        """Per-token decode latency (single-stream, batch=1).

        MoE decode is memory-bandwidth-bound: every token must read all
        active params + the KV cache of all past tokens once. We compute:
            latency = (active_param_bytes + kv_cache_read_bytes) / effective_BW

        Theoretical lower bound; real systems achieve 30-50% of HBM BW.

        Returns dict with breakdown.
        """
        b = DTypeBytes[dtype]
        active_bytes = self.active_params() * b
        # KV cache traversed per decode step: full cache at current ctx position.
        # MUST use hybrid-aware total (SWA caps at window, linear has fixed state).
        kv_total = self.kv_cache_bytes_total(kv_seq_len, batch=1, dtype=dtype)

        effective_bw_bps = hbm_bandwidth_gbps * 1e9 * effective_bw_fraction
        active_ms = active_bytes / effective_bw_bps * 1000
        kv_ms = kv_total / effective_bw_bps * 1000
        total_ms = active_ms + kv_ms

        return {
            "active_params_GB": active_bytes / 1e9,
            "kv_cache_GB": kv_total / 1e9,
            "active_latency_ms": active_ms,
            "kv_latency_ms": kv_ms,
            "total_per_token_ms": total_ms,
            "tokens_per_sec": 1000 / total_ms if total_ms > 0 else 0,
        }

    def mtp_speedup(self, accept_rate: float = 0.75) -> float:
        """Speculative decoding speedup from MTP heads.

        Accept rate = fraction of MTP-predicted tokens that match the main
        model's would-be output. V3 paper reports ~85-90% acceptance for
        MTP D=1; LongCat / Ling report similar.

        Effective tokens per decode step = 1 + accept_rate * mtp_depth.
        """
        if self.mtp_depth == 0:
            return 1.0
        return 1.0 + accept_rate * self.mtp_depth

    def prefill_throughput(
        self,
        seq_len: int = 4096,
        gpu_peak_tflops: float = 989.0,  # H100 BF16
        mfu: float = 0.50,  # prefill MFU usually higher than decode
    ) -> dict[str, float]:
        """Prefill throughput: tokens/sec for a single sequence of seq_len.

        Prefill is compute-bound (process seq_len tokens in parallel).
        Total prefill FLOPs = 2 × active_params × seq_len  (projections, FFN)
                            + 2 × num_q × head_dim × seq_len² × num_layers  (attn QK^T+V)
        """
        n = seq_len
        # Linear projections (Q, K, V, O, FFN gate/up/down, router) + LM head:
        # active params include embedding/lm_head (V3 口径)
        active = self.active_params(include_embedding=True)
        proj_flops = 2 * active * n

        # Attention quadratic (hybrid-aware: SWA caps at window, linear is O(N))
        if self.hybrid_attn is not None:
            attn_quad = sum(
                count * layer_spec.attn_quadratic_flops(n)
                for count, layer_spec in self.hybrid_attn
            )
        else:
            if self.attn_type == "mla":
                head_d = self.qk_nope_head_dim + self.qk_rope_head_dim
            else:
                head_d = self.head_dim
            # Coefficient 4 = 2 (QK^T) + 2 (Attn × V); matches AttnLayerSpec.attn_quadratic_flops
            attn_quad = 4 * self.num_q_heads * head_d * n * n * self.num_layers

        total_flops = proj_flops + attn_quad
        effective_flops_per_sec = gpu_peak_tflops * 1e12 * mfu
        time_sec = total_flops / effective_flops_per_sec
        return {
            "total_flops": total_flops,
            "proj_flops": proj_flops,
            "attn_quad_flops": attn_quad,
            "attn_quad_pct": attn_quad / total_flops * 100,
            "time_sec": time_sec,
            "tokens_per_sec": n / time_sec,
            "time_per_token_ms": time_sec * 1000 / n,
        }

    # ---------------------- Invariants ----------------------

    def check_invariants(self) -> list[CheckResult]:
        """All design rules that should hold for V3-派 MoE."""
        checks = []

        # Hybrid attention: skip uniform attention checks; instead check each layer kind
        if self.hybrid_attn is not None:
            for count, ls in self.hybrid_attn:
                if ls.kind in ("softmax", "swa"):
                    q_total = ls.num_q_heads * ls.head_dim
                    checks.append(CheckResult(
                        f"hybrid layer ({ls.kind}, ×{count}): Q×head = hidden",
                        q_total == self.hidden,
                        f"{ls.num_q_heads}×{ls.head_dim}={q_total} vs hidden={self.hidden}"
                        + (" (this is OK for some Q-promoted designs, e.g. Qwen3-Next)" if q_total != self.hidden else ""),
                        "info",
                    ))
        else:
            # 1. head_dim divides hidden (for GQA/MHA Q proj)
            if self.attn_type in ("gqa", "mha"):
                q_total = self.num_q_heads * self.head_dim
                checks.append(CheckResult(
                    "head_dim × num_q_heads = hidden",
                    q_total == self.hidden,
                    f"{self.num_q_heads}×{self.head_dim}={q_total} vs hidden={self.hidden}",
                    "warn" if q_total != self.hidden else "info",
                ))

            # 2. GQA ratio (num_q / num_kv) is integer
            if self.attn_type == "gqa":
                ratio = self.num_q_heads / self.num_kv_heads
                checks.append(CheckResult(
                    "GQA Q:KV ratio integer",
                    ratio == int(ratio),
                    f"Q/KV = {self.num_q_heads}/{self.num_kv_heads} = {ratio:.2f}",
                    "error" if ratio != int(ratio) else "info",
                ))

        # 3. dense_intermediate = (K + N_sh) × d_expert (compute-equivalent)
        # Only applicable if there ARE dense layers
        if self.first_k_dense > 0:
            expected = (self.top_k + self.n_shared) * self.d_expert
            deviation = abs(self.dense_intermediate - expected) / expected
            checks.append(CheckResult(
                "dense_int = (K+N_sh) × d_expert",
                deviation < 0.05,
                f"dense_int={self.dense_intermediate}, expected={expected}, dev={deviation*100:.1f}%",
                "warn" if deviation >= 0.05 else "info",
            ))

        # 4. Attn:FFN ≈ 1:2 (V3 派 100B+ convention; hybrid models exempt)
        ratio = self.attn_ffn_ratio()
        # 1:2 means ratio = 0.5; allow 0.3-0.7 as "in band"
        in_band = 0.30 <= ratio <= 0.70
        is_hybrid = self.hybrid_attn is not None
        checks.append(CheckResult(
            "Attn:FFN ratio in 1:2 band (V3 派 convention)",
            in_band or is_hybrid,
            f"attn/ffn = 1:{1/ratio:.2f} (r={ratio:.3f})"
            + (" — hybrid model exempt (SWA/linear layers cheap, more attn budget OK)" if is_hybrid else ""),
            "info" if (in_band or is_hybrid) else "warn",
        ))

        # 5. d_expert / hidden ≈ 0.30 (V3 派 convention for K=9)
        rho = self.d_expert / self.hidden
        target = 4 / (3 * (self.top_k + self.n_shared))
        checks.append(CheckResult(
            "d_expert/hidden ≈ 4/(3(K+N_sh))",
            abs(rho - target) / target < 0.5,  # ±50% tolerance
            f"d_expert/hidden = {rho:.3f}, ideal = {target:.3f}",
            "info",
        ))

        # 6. MoE expert count power-of-2 friendly (for EP topology)
        n = self.n_routed
        good = (n in {32, 64, 128, 160, 192, 256, 288, 320, 384, 512, 640})
        checks.append(CheckResult(
            "N_routed is EP-topology-friendly (∈ standard set)",
            good,
            f"N_routed={n}",
            "info",
        ))

        # 7. RoPE partial division (head_dim should be ≥ 64 and divisible by 2)
        eff_head_dim = self.head_dim if self.attn_type != "mla" else self.qk_rope_head_dim
        checks.append(CheckResult(
            "head_dim ≥ 64 and even (RoPE compat)",
            eff_head_dim >= 64 and eff_head_dim % 2 == 0,
            f"head_dim (rope) = {eff_head_dim}",
            "warn" if eff_head_dim < 64 else "info",
        ))

        return checks

    def check_parallelism(self, ep: int, pp: int = 1, tp: int = 1) -> list[CheckResult]:
        """Verify TP/PP/EP divisibility constraints."""
        checks = []

        checks.append(CheckResult(
            "num_kv_heads % TP == 0",
            self.num_kv_heads % tp == 0,
            f"kv_heads={self.num_kv_heads}, TP={tp}",
            "error" if self.num_kv_heads % tp != 0 else "info",
        ))
        checks.append(CheckResult(
            "num_q_heads % TP == 0",
            self.num_q_heads % tp == 0,
            f"q_heads={self.num_q_heads}, TP={tp}",
            "error" if self.num_q_heads % tp != 0 else "info",
        ))
        checks.append(CheckResult(
            "N_routed % EP == 0",
            self.n_routed % ep == 0,
            f"N_routed={self.n_routed}, EP={ep}",
            "error" if self.n_routed % ep != 0 else "info",
        ))
        checks.append(CheckResult(
            "num_layers % PP == 0",
            self.num_layers % pp == 0,
            f"L={self.num_layers}, PP={pp}",
            "warn" if self.num_layers % pp != 0 else "info",
        ))

        if ep > 8:
            checks.append(CheckResult(
                "EP > 8 → NLR (M=4) recommended",
                False,
                f"EP={ep} crosses node boundary (single H100 node = 8 GPU); enable NLR",
                "warn",
            ))
        return checks

    # ---------------------- Pretty print ----------------------

    def summary(self) -> str:
        b = self.params_breakdown()
        total = self.total_params_no_mtp()
        active = self.active_params(include_embedding=False)
        active_v3 = self.active_params(include_embedding=True)
        rates = self.activation_rate()

        def fmt(n: int) -> str:
            if n >= 1e9:
                return f"{n/1e9:.2f}B"
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            return f"{n}"

        lines = [
            f"═══ {self.name} ═══",
            f"  Backbone:  {self.num_layers}L × hidden {self.hidden} ({self.first_k_dense} dense + {self.num_layers - self.first_k_dense} MoE)",
        ]
        if self.hybrid_attn is not None:
            lines.append(f"  Attention: HYBRID")
            for count, ls in self.hybrid_attn:
                extra = ""
                if ls.kind == "swa":
                    extra = f" W={ls.window_size}"
                elif ls.kind == "linear":
                    extra = f" linear K={ls.linear_num_value_heads or ls.num_q_heads}h×{ls.linear_value_head_dim or ls.head_dim}d"
                lines.append(f"             {count}× {ls.kind:>7s} {ls.num_q_heads}Q/{ls.num_kv_heads}KV d={ls.head_dim}{extra}")
        else:
            lines.append(f"  Attention: {self.attn_type.upper()} {self.num_q_heads}Q/{self.num_kv_heads}KV head_dim {self.head_dim}")
            if self.attn_type == "mla":
                lines.append(f"             MLA: q_lora={self.q_lora_rank}, kv_lora={self.kv_lora_rank}, "
                             f"qk_nope={self.qk_nope_head_dim}, qk_rope={self.qk_rope_head_dim}, v={self.v_head_dim}")
        lines.append(
            f"  MoE:       {self.n_routed} routed + {self.n_shared} shared, K={self.top_k}, d_expert={self.d_expert}"
        )
        lines.append(f"  Vocab:     {self.vocab_size} ({'tied' if self.tied_embedding else 'untied'})")
        if self.mtp_depth:
            lines.append(f"  MTP:       D={self.mtp_depth} ({'MoE' if self.mtp_is_moe else 'dense'})")
        lines.append("")
        lines.append("  ── Params ──  (% of base total, excludes MTP)")
        mtp_v = b.get("mtp", 0)
        base_total = total  # total = total_params_no_mtp() returned above
        for k, v in b.items():
            if k == "mtp":
                continue
            lines.append(f"     {k:18s} {fmt(v):>10s}  ({100*v/base_total:5.2f}%)")
        lines.append(f"     {'BASE TOTAL':18s} {fmt(base_total):>10s}  (no MTP)")
        if mtp_v > 0:
            grand = base_total + mtp_v
            lines.append(f"     {'+ MTP module':18s} {fmt(mtp_v):>10s}  (+{100*mtp_v/base_total:5.2f}% over base)")
            lines.append(f"     {'TOTAL incl. MTP':18s} {fmt(grand):>10s}")
        lines.append(f"     {'ACTIVE (strict)':18s} {fmt(active):>10s}  (per token, no embedding)")
        lines.append(f"     {'ACTIVE (V3 口径)':18s} {fmt(active_v3):>10s}  (per token, w/ embedding)")
        lines.append("")
        lines.append("  ── Rates ──")
        for k, v in rates.items():
            lines.append(f"     {k:20s} {v*100:6.2f}%   (1/{1/v:.1f})")
        lines.append(f"     L/√H                 {self.l_over_sqrt_h():.3f}")
        lines.append(f"     d_expert/hidden      {self.d_expert_over_hidden():.3f}")
        lines.append(f"     Attn:FFN per layer   1:{1/self.attn_ffn_ratio():.2f}")
        lines.append("")
        lines.append("  ── KV cache (BF16) ──")
        if self.hybrid_attn is None:
            per = self.kv_cache_bytes_per_token_per_layer("bf16")
            lines.append(f"     per token per layer: {per} B")
        total_kv_32k = self.kv_cache_bytes_total(32768, 1, "bf16")
        total_kv_128k = self.kv_cache_bytes_total(128 * 1024, 1, "bf16")
        total_kv_1m = self.kv_cache_bytes_total(1024 * 1024, 1, "bf16")
        lines.append(f"     32K ctx, batch=1:    {total_kv_32k/1024/1024:.0f} MB")
        lines.append(f"     128K ctx, batch=1:   {total_kv_128k/1024/1024/1024:.2f} GB")
        lines.append(f"     1M ctx, batch=1:     {total_kv_1m/1024/1024/1024:.2f} GB")
        if self.hybrid_attn is not None:
            lines.append(f"     (hybrid: SWA layers capped at window; linear/mamba use fixed state)")
        return "\n".join(lines)
