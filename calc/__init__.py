"""MoE Architecture Calculator.

Compute params / FLOPs / KV cache / training cost / invariants for MoE LLMs.
Validate against anchor models (V3, K2, Ling, GLM, dots1, Step 3.5 Flash, etc.).
"""
from .spec import MoEArchSpec, CheckResult
from .anchors import ANCHORS, get_anchor
from .compare import compare_specs, summary_table
from .upscale import solar_dus, sparse_upcycle, hybrid_upscale, upscale_summary
