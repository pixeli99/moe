"""Database of anchor MoE models.

Each spec is verified against:
- HuggingFace config.json (primary source)
- Paper Table 1 or equivalent
- 28_open_source_moe_catalog.md cross-reference

When a model uses hybrid attention (Step 3.5 Flash SWA:Full, Qwen3-Next
DeltaNet:Full, Jamba Mamba:Attn), only the "average per-layer" approximation
is provided; calc.py does NOT model hybrid attention explicitly.

Sources cited inline.
"""
from .spec import MoEArchSpec, AttnLayerSpec


ANCHORS = {
    # ─────────────────── 16B segment (V2-Lite anchor) ───────────────────
    "v2-lite": MoEArchSpec(
        name="DeepSeek-V2-Lite (15.7B/2.4B, MLA)",
        # Source: deepseek-ai/DeepSeek-V2-Lite/config.json
        hidden=2048, num_layers=27,
        head_dim=128,  # informational; MLA uses qk_nope+qk_rope
        num_q_heads=16, num_kv_heads=16,
        attn_type="mla",
        q_lora_rank=None,  # V2-Lite has no Q LoRA
        kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed=64, top_k=6, n_shared=2, d_expert=1408,
        first_k_dense=1, dense_intermediate=10944,
        vocab_size=102400, tied_embedding=False,
        mtp_depth=0,
        notes="The 16B-segment anchor model. Paper: arXiv 2405.04434",
    ),
    "moonlight": MoEArchSpec(
        name="Moonlight-16B-A3B (Moonshot, Muon-trained, MLA)",
        # Source: moonshotai/Moonlight-16B-A3B/config.json; uses V3-small (MLA) arch
        # CORRECTED 2026-05: attn_type was gqa, should be mla per xlsx + V3-small lineage
        hidden=2048, num_layers=27, head_dim=128,
        num_q_heads=16, num_kv_heads=16,
        attn_type="mla",
        q_lora_rank=None,  # V2-Lite style (no Q LoRA)
        kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed=64, top_k=6, n_shared=2, d_expert=1408,
        first_k_dense=1, dense_intermediate=11264,
        vocab_size=163840, tied_embedding=False,
        mtp_depth=0,
        notes="V3-small arch + Muon optimizer. HF config intermediate_size=11264. Paper: arXiv 2502.16982",
    ),

    "ling-mini-2": MoEArchSpec(
        name="Ling-mini-2.0 (16B/1.4B)",
        # Source: inclusionAI/Ling-mini-2.0/config.json; Ling 2.0 paper Table 1 for family context
        hidden=2048, num_layers=20,
        head_dim=128,
        num_q_heads=16, num_kv_heads=4,
        attn_type="gqa",
        n_routed=256, top_k=8, n_shared=1, d_expert=512,
        first_k_dense=1, dense_intermediate=5120,
        vocab_size=157184, tied_embedding=False,
        mtp_depth=0, mtp_is_moe=False,
        notes="Short-wide 16B; 1/32 expert-slot. HF config num_nextn_predict_layers=0.",
    ),

    "deepseekmoe-16b": MoEArchSpec(
        name="DeepSeekMoE-16B (16.4B/2.8B, MHA, original MoE paper)",
        # Source: deepseek-ai/deepseek-moe-16b-base/config.json; arXiv 2401.06066
        hidden=2048, num_layers=28, head_dim=128,
        num_q_heads=16, num_kv_heads=16,
        attn_type="mha",
        n_routed=64, top_k=6, n_shared=2, d_expert=1408,
        first_k_dense=1, dense_intermediate=10944,
        vocab_size=102400, tied_embedding=False,
        mtp_depth=0,
        notes="Origin of fine-grained + shared expert MoE paradigm. Paper: arXiv 2401.06066",
    ),

    # ─────────────────── 200B-300B segment ───────────────────
    "v2-236b": MoEArchSpec(
        name="DeepSeek-V2 236B (21B, MLA legacy)",
        # Source: deepseek-ai/DeepSeek-V2/config.json; arXiv 2405.04434
        hidden=5120, num_layers=60, head_dim=128,
        num_q_heads=128, num_kv_heads=128,
        attn_type="mla",
        q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed=160, top_k=6, n_shared=2, d_expert=1536,
        first_k_dense=1, dense_intermediate=12288,  # = (6+2) × 1536
        vocab_size=102400, tied_embedding=False,
        mtp_depth=0,
        notes="V3's predecessor; MLA + device-limited routing (M=3). Paper: arXiv 2405.04434",
    ),
    "mimo-v2-flash": MoEArchSpec(
        name="MiMo-V2-Flash 309B (15B, Hybrid SWA+GA, GQA 64Q/8KV)",
        # Source: xlsx + GQA inferred (KV=8 makes active match xlsx 15B exactly)
        hidden=4096, num_layers=48, head_dim=192,
        num_q_heads=64, num_kv_heads=8,
        attn_type="gqa",
        n_routed=256, top_k=8, n_shared=0, d_expert=2048,
        first_k_dense=1, dense_intermediate=16384,  # 8 × 2048
        vocab_size=128256, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="Reasoning-focused; 27T tokens; sequence aux 1e-5. KV=8 inferred from active=15B",
    ),
    # ─────────────────── 100B segment ───────────────────
    "ling-flash-2": MoEArchSpec(
        name="Ling-flash-2.0 (103B/6.1B)",
        # Source: inclusionAI/Ling-flash-2.0/config.json; Ling 2.0 paper Table 1 for family context
        hidden=4096, num_layers=32,
        head_dim=128,
        num_q_heads=32, num_kv_heads=4,
        attn_type="gqa",
        n_routed=256, top_k=8, n_shared=1, d_expert=1024,
        first_k_dense=1, dense_intermediate=9216,
        vocab_size=157184, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="100B base, ~5.9% active. HF config num_key_value_heads=4; paper: arXiv 2510.22115",
    ),
    "glm-4.5-air": MoEArchSpec(
        name="GLM-4.5-Air (106B/12B)",
        # Source: GLM-4.5 paper Table 1; arXiv 2508.06471
        hidden=4096, num_layers=46,  # 1 dense + 45 MoE
        head_dim=128,
        num_q_heads=96, num_kv_heads=8,  # 96 heads, not 32 (GLM quirk)
        attn_type="gqa",
        n_routed=128, top_k=8, n_shared=1, d_expert=1408,
        first_k_dense=1, dense_intermediate=10944,
        vocab_size=151552, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="Depth>width 100B. Paper: arXiv 2508.06471",
    ),
    "dots1": MoEArchSpec(
        name="dots.llm1 (142B/14B, MHA)",
        # Source: rednote-hilab/dots.llm1.base/config.json; arXiv 2506.05767
        hidden=4096, num_layers=62,
        head_dim=128,
        num_q_heads=32, num_kv_heads=32,  # full MHA
        attn_type="mha",
        n_routed=128, top_k=6, n_shared=2, d_expert=1408,
        first_k_dense=1, dense_intermediate=10944,
        vocab_size=152064, tied_embedding=False,
        mtp_depth=0,
        notes="MHA + QK-Norm; routing inherited from V3. Paper: arXiv 2506.05767",
    ),

    # ─────────────────── 200B-400B segment ───────────────────
    "glm-4.5": MoEArchSpec(
        name="GLM-4.5 (355B/32B)",
        # Source: GLM-4.5 paper Table 1
        hidden=5120, num_layers=92,  # 3 dense + 89 MoE
        head_dim=128,
        num_q_heads=96, num_kv_heads=8,
        attn_type="gqa",
        n_routed=160, top_k=8, n_shared=1, d_expert=1536,
        first_k_dense=3, dense_intermediate=12288,
        vocab_size=151552, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="Depth>width extreme L/√H=1.29. Paper: arXiv 2508.06471",
    ),

    # ─────────────────── 600B-1T segment ───────────────────
    "v3": MoEArchSpec(
        name="DeepSeek-V3 (671B/37B, MLA)",
        # Source: deepseek-ai/DeepSeek-V3/config.json; paper arXiv 2412.19437
        hidden=7168, num_layers=61,
        head_dim=192,  # informational
        num_q_heads=128, num_kv_heads=128,
        attn_type="mla",
        q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed=256, top_k=8, n_shared=1, d_expert=2048,
        first_k_dense=3, dense_intermediate=18432,
        vocab_size=129280, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=True,
        notes="The reference 600B+ MoE. Paper: arXiv 2412.19437",
    ),
    "k2": MoEArchSpec(
        name="Kimi K2 (1T/32B, MLA + MuonClip)",
        # Source: moonshotai/Kimi-K2-Instruct/config.json
        hidden=7168, num_layers=61,
        head_dim=192,
        num_q_heads=64, num_kv_heads=64,  # K2 has only 64 heads (vs V3's 128)
        attn_type="mla",
        q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed=384, top_k=8, n_shared=1, d_expert=2048,
        first_k_dense=1, dense_intermediate=18432,
        vocab_size=163840, tied_embedding=False,
        mtp_depth=0,
        notes="1T MoE, 1/31 sparsity, Muon-trained. K2 paper",
    ),
    "ling-1t": MoEArchSpec(
        name="Ling-1T (1T/51B, GQA + FP8)",
        # Source: inclusionAI/Ling-1T/config.json; Ling 2.0 paper Table 1 for family context
        hidden=8192, num_layers=80,
        head_dim=128,
        num_q_heads=64, num_kv_heads=8,
        attn_type="gqa",
        n_routed=256, top_k=8, n_shared=1, d_expert=2048,
        first_k_dense=4, dense_intermediate=18432,
        vocab_size=157184, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="Largest FP8 base. HF config num_key_value_heads=8; paper: arXiv 2510.22115",
    ),

    # ─────────────────── Hybrid attention (2025-2026) ───────────────────
    "step-3.5-flash": MoEArchSpec(
        name="Step 3.5 Flash (196B/11B, SWA:Full 3:1 hybrid)",
        # Source: arXiv 2602.10604 + xlsx ground truth (corrected 2026-05)
        # 45L = 3 dense + 42 MoE; attention: 1 leading Full + 11 × (3 SWA + 1 Full)
        # CORRECTED: hidden=4096 (was 5120), d_expert=1280 (was 1024)
        hidden=4096,
        num_layers=45, head_dim=128,
        num_q_heads=64, num_kv_heads=8,  # nominal Full-attn config (overridden by hybrid_attn)
        attn_type="gqa",
        hybrid_attn=[
            (12, AttnLayerSpec(kind="softmax", num_q_heads=64, num_kv_heads=8, head_dim=128)),
            (33, AttnLayerSpec(kind="swa", num_q_heads=96, num_kv_heads=8, head_dim=128,
                               window_size=512)),
        ],
        n_routed=288, top_k=8, n_shared=1, d_expert=1280,
        first_k_dense=3, dense_intermediate=11264,
        vocab_size=128896, tied_embedding=False,
        mtp_depth=3, mtp_is_moe=False,
        notes="Hybrid SWA+Full; Augmented 96Q SWA heads; Head-wise Gated Attention. "
              "HF config/model card: intermediate_size=11264, vocab_size=128896, nextn=3.",
    ),
    "qwen3-next": MoEArchSpec(
        name="Qwen3-Next-80B (80B/3B, DeltaNet:Attn 3:1 hybrid)",
        # Source: HF config Qwen/Qwen3-Next-80B-A3B-Instruct
        # 48L = 12 × (3 Gated DeltaNet + 1 Gated Attention)
        # = 36 DeltaNet (linear) + 12 Full attention
        hidden=2048,
        num_layers=48, head_dim=256,  # head_dim 256 for Full attn
        num_q_heads=16, num_kv_heads=2,  # Full attn config
        attn_type="gqa",
        hybrid_attn=[
            (36, AttnLayerSpec(
                kind="linear",
                num_q_heads=16, num_kv_heads=16, head_dim=128,  # linear_key_head_dim
                linear_num_value_heads=32, linear_value_head_dim=128,
                conv_kernel=4,
            )),
            (12, AttnLayerSpec(
                kind="softmax",
                num_q_heads=16, num_kv_heads=2, head_dim=256,
            )),
        ],
        n_routed=512, top_k=10, n_shared=1, d_expert=512,
        first_k_dense=0, dense_intermediate=5632,  # only relevant if first_k_dense > 0
        vocab_size=151936, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="Hybrid 3:1 DeltaNet (linear) + Full softmax. head_dim=256 reverse default. "
              "Zero-centered weight-decayed layernorm. 1M ctx via YaRN.",
    ),

    # ─────────────────── Legacy / reference points ───────────────────
    "mixtral-8x7b": MoEArchSpec(
        name="Mixtral 8x7B (46.7B/12.9B, GQA + softmax+aux)",
        # Source: mistralai/Mixtral-8x7B-v0.1/config.json
        hidden=4096, num_layers=32,
        head_dim=128,
        num_q_heads=32, num_kv_heads=8,
        attn_type="gqa",
        n_routed=8, top_k=2, n_shared=0, d_expert=14336,
        first_k_dense=0, dense_intermediate=14336,  # all-MoE
        vocab_size=32000, tied_embedding=False,
        mtp_depth=0,
        notes="Mixtral派老一代; K=2 + 0 shared; d_expert huge",
    ),
    "olmoe": MoEArchSpec(
        name="OLMoE-1B-7B (6.9B/1.3B)",
        # Source: allenai/OLMoE-1B-7B-0924/config.json
        hidden=2048, num_layers=16,
        head_dim=128,
        num_q_heads=16, num_kv_heads=16,
        attn_type="mha",
        n_routed=64, top_k=8, n_shared=0, d_expert=1024,
        first_k_dense=0, dense_intermediate=1024,
        vocab_size=50304, tied_embedding=False,
        mtp_depth=0,
        notes="Fully open MoE; 16L short. Paper: arXiv 2409.02060",
    ),

    # ─────────────────── User's 16B design ───────────────────
    "user-16b": MoEArchSpec(
        name="USER 16B Profile B' (20L/2560/64×8+1 GQA 20Q5KV)",
        # Per discussion 2026-05-20 with user
        hidden=2560, num_layers=20,
        head_dim=128,
        num_q_heads=20, num_kv_heads=5,
        attn_type="gqa",
        n_routed=64, top_k=8, n_shared=1, d_expert=1536,
        first_k_dense=1, dense_intermediate=13824,  # = 9 × 1536
        vocab_size=128256, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="User's 16B base spec (20L short-wide); see 22_FINAL & 43_short_wide_design",
    ),

    # ─────────────────── User's L100 / L200 plan ───────────────────
    "user-l100-v1": MoEArchSpec(
        name="USER L100 v1 (105B/6.5B, original recommendation, 1:5.4 attn:ffn)",
        hidden=4096, num_layers=32,
        head_dim=128,
        num_q_heads=32, num_kv_heads=8,
        attn_type="gqa",
        n_routed=128, top_k=8, n_shared=1, d_expert=2048,
        first_k_dense=1, dense_intermediate=18432,
        vocab_size=128256, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="L100 v1 — attn:FFN 1:5.4 (violates 1:2 rule); see 45_l100_l200_design",
    ),
    "user-l100-v2": MoEArchSpec(
        name="USER L100 v2 (107B/7B, fixed: 64Q like K2, 1:2.7 attn:ffn)",
        hidden=4096, num_layers=32,
        head_dim=128,
        num_q_heads=64, num_kv_heads=8,  # bumped from 32 → 64
        attn_type="gqa",
        n_routed=128, top_k=8, n_shared=1, d_expert=2048,
        first_k_dense=1, dense_intermediate=18432,
        vocab_size=128256, tied_embedding=False,
        mtp_depth=1, mtp_is_moe=False,
        notes="L100 v2 — closer to 1:2 attn:FFN; matches K2 Q head count",
    ),
}


def get_anchor(name: str) -> MoEArchSpec:
    if name not in ANCHORS:
        raise KeyError(f"Unknown anchor '{name}'. Available: {', '.join(sorted(ANCHORS))}")
    return ANCHORS[name]


def list_anchors() -> list[str]:
    return sorted(ANCHORS.keys())
