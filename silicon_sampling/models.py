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
