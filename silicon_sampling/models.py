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
    "qwen25_72b": "Qwen/Qwen2.5-72B",
    "v4_flash": "deepseek-ai/DeepSeek-V4-Flash-Base",
    # Generation replicates: the same respondents with the same demographics, drawn
    # again from offset seeds.  Averaging a replicate with its parent halves our own
    # sampling noise, which is what attenuates the pooled correlation the benchmark
    # sorts on -- a separate lever from averaging across *models*, and a safer one,
    # since it cannot import a worse model's ordering.
    "qwen25_7b_seed2": "Qwen/Qwen2.5-7B",
    "qwen25_72b_seed2": "Qwen/Qwen2.5-72B",
    # Second template render for ICPC and Goldwert.  Their 0-100 sliders printed
    # the endpoint labels the respondent saw and never the numeric range, so the
    # models answered on a small scale: 80-94% of every slider answer came back at
    # 10 or below against 8-31% for real participants, and control-arm level error
    # ran 20-47 points.  The `_v2` runs are the same instruments with the range
    # stated; the originals are kept because they are the evidence for the defect.
    "qwen25_7b_v2": "Qwen/Qwen2.5-7B",
    "qwen25_72b_v2": "Qwen/Qwen2.5-72B",
    "v4_flash_v2": "deepseek-ai/DeepSeek-V4-Flash-Base",
    # Third render, after the full fidelity audit: the block randomiser permuting
    # the wrong blocks, piped correction text and photographs rendered as
    # placeholders, media-only screens rendered as bare brackets, a scored video
    # nobody was shown, a blank control arm where participants watched a
    # five-minute video, opt-outs printed but refused, and a letter box whose
    # prompt told the model to enter a ZIP code.  `_v2` was the slider-range fix
    # alone and is superseded.
    "qwen25_7b_v3": "Qwen/Qwen2.5-7B",
    "qwen25_72b_v3": "Qwen/Qwen2.5-72B",
    "v4_flash_v3": "deepseek-ai/DeepSeek-V4-Flash-Base",
    # A third independent draw of respondents for each ranker.  Roughly 13% of
    # Qwen2.5-7B's Pfander effect variance and 26% of Qwen2.5-72B's is sampling
    # noise, and the submission averages the runs, so each extra seed buys a
    # little more of the signal: measured reliability goes 0.866 at two runs to
    # 0.928 at four, and Spearman-Brown puts six at about 0.951.
    "qwen25_7b_seed3": "Qwen/Qwen2.5-7B",
    "qwen25_72b_seed3": "Qwen/Qwen2.5-72B",
}

#: Human-readable names for report tables and plot legends.
LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "qwen25_72b": "Qwen2.5-72B",
    "v4_flash": "DeepSeek-V4-Flash",
    "qwen25_7b_seed2": "Qwen2.5-7B (replicate)",
    "qwen25_72b_seed2": "Qwen2.5-72B (replicate)",
    "qwen25_7b_v2": "Qwen2.5-7B (scale stated)",
    "qwen25_72b_v2": "Qwen2.5-72B (scale stated)",
    "v4_flash_v2": "DeepSeek-V4-Flash (scale stated)",
    "qwen25_7b_seed3": "Qwen2.5-7B (replicate 3)",
    "qwen25_72b_seed3": "Qwen2.5-72B (replicate 3)",
    "qwen25_7b_v3": "Qwen2.5-7B (audited)",
    "qwen25_72b_v3": "Qwen2.5-72B (audited)",
    "v4_flash_v3": "DeepSeek-V4-Flash (audited)",
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
#:
#: ``qwen25_72b`` needs nothing special.  It is a dense bf16 model, so its KV
#: cache is ordinary GQA — 80 layers x 8 kv heads x 128 dims x 2 (k and v) x 2 B
#: = 320 KB per token, three times Qwen2.5-7B's 57 KB and a third of
#: V4-Flash's ~1 MB.  That middle position is the whole reason it is worth
#: sampling with: 145 GB of weights and a cheap cache buy a far larger resident
#: group than V4-Flash's 295 GB and hybrid MLA cache manage, which is what
#: throughput is actually made of here.
ENGINE_DEFAULTS = {
    "v4_flash": {"kv_cache_dtype": "fp8_ds_mla"},
    "v4_flash_v2": {"kv_cache_dtype": "fp8_ds_mla"},
    "v4_flash_v3": {"kv_cache_dtype": "fp8_ds_mla"},
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


#: Studies whose questionnaire the ``_v3`` audit actually changed.
#:
#: The slider-range defect — endpoint labels printed without the 0-100 range —
#: existed only in the ICPC and Goldwert conversions.  Voelkel and Pfänder state
#: their ranges in the source questionnaire's own prose, so the audit found
#: nothing to change in them and they were not re-sampled.
REVISED_STUDIES = ("ICPC", "Goldwert")


def base_run(run: str) -> str:
    """The pre-audit run key that *run* is a re-sample of.

    ``qwen25_72b_v3`` -> ``qwen25_72b``; anything else is returned unchanged.
    """
    for suffix in ("_v3", "_v2"):
        if run.endswith(suffix):
            return run[: -len(suffix)]
    return run


def resolve_run(samples_dir, run: str) -> str | None:
    """The run key to *score* for this study, or ``None`` if it has no sample.

    A study with no ``_v3`` directory is usually not missing data.  Only ICPC and
    Goldwert were re-sampled, because only their templates changed; Voelkel's v1
    sample was taken on a template the audit would have left alone, so it is the
    audited sample for that study and belongs in the comparison.  Treating it as
    absent is what silently cut leave-one-study-out from three folds to two, and
    two folds cannot show that a parameter is stable.

    Falls back only when the fallback exists, and never crosses models.
    """
    if (samples_dir(run) / "samples.csv").exists():
        return run
    fallback = base_run(run)
    if fallback != run and (samples_dir(fallback) / "samples.csv").exists():
        return fallback
    return None
