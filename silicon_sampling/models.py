"""The models this project samples with, and where their runs are filed.

A run key is a short directory-safe name (``qwen25_7b``) standing for a Hugging
Face id (``Qwen/Qwen2.5-7B``).  Outputs live under
``data/<study>/silicon_sampling/<key>/`` so two models' samples sit side by side
and the analysis can compare them without either run moving.
"""

from __future__ import annotations

#: Run key -> Hugging Face model id.
MODELS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B",
    "v4_flash": "deepseek-ai/DeepSeek-V4-Flash-Base",
}

#: Human-readable names for report tables and plot legends.
LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "v4_flash": "DeepSeek-V4-Flash",
}

#: The run every existing path and report already refers to.
DEFAULT_RUN = "qwen25_7b"

#: Engine settings a model *requires*, applied unless the caller overrides them.
#:
#: ``v4_flash`` needs ``kv_cache_dtype="fp8_ds_mla"``.  On an H200 vLLM selects
#: DeepSeek-V4's FlashMLA attention, whose paged layout *is* the fp8 format —
#: UE8M0 block-scaled fp8 packed as uint8, 576 B per token — and it refuses to
#: start with anything else ("FlashMLA fp8 layout only supports fp8 kv-cache").
#:
#: This is not the fidelity compromise that ``kv_cache_dtype="fp8"`` is on a
#: bf16-KV model: the checkpoint ships the UE8M0 scales, so this is the precision
#: DeepSeek built the model to run at.  It is still an asymmetry with the
#: Qwen2.5-7B run, which used bf16 KV, and the comparison reports say so.
ENGINE_DEFAULTS = {
    "v4_flash": {"kv_cache_dtype": "fp8_ds_mla"},
}


def engine_defaults(run: str) -> dict:
    """Engine settings this run needs, as keyword overrides."""
    return dict(ENGINE_DEFAULTS.get(run, {}))


def model_id(run: str) -> str:
    """The Hugging Face id for a run key."""
    try:
        return MODELS[run]
    except KeyError:
        raise SystemExit(
            f"unknown run key {run!r}; known keys: {', '.join(sorted(MODELS))}"
        ) from None


def label(run: str) -> str:
    """The name this run goes by in a report."""
    return LABELS.get(run, run)
