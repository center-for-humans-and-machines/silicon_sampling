"""The model registry, and the constraints the cluster puts on it."""

from __future__ import annotations

import sys

from silicon_sampling import models as M


def test_muse_glimmer_is_registered_with_its_own_container():
    """The default DAIS image cannot load it, so the container is part of the config.

    vLLM 0.23.0 there does not register ``MuseGlimmerForConditionalGeneration`` and
    transformers 5.12.1 does not know ``muse_glimmer``; the checkpoint ships no
    ``auto_map``, so there is no ``trust_remote_code`` path either.  Only
    ``MUSE_GLIMMER_CONTAINER`` works, and it has no pandas — which is what
    ``silicon_sampling.lazy`` exists for.
    """
    assert M.MODELS["muse_glimmer_30b"] == "meta-models/Muse-Glimmer-30B"
    assert M.MUSE_GLIMMER_CONTAINER.endswith("vllm-openai-muse-glimmer.sif")
    assert "muse_glimmer_30b" in M.MULTIMODAL_CAPABLE


def test_every_study_entry_point_imports_without_pandas_or_scipy():
    """The Muse-Glimmer image ships neither, and the sampler must still start.

    Covers ``cli`` as well as ``run``, because **the cluster wrapper invokes
    ``python -m silicon_sampling.<study>.cli sample``** — and an earlier version of
    this test checked only ``run``.  It passed, the jobs were submitted, and two of
    the five studies then burned twelve retry attempts each on the cluster before
    failing at ``cli.py: import pandas``.  Testing the module the entry point
    actually is, rather than the one underneath it, is the whole point.
    """
    import builtins
    import importlib

    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name in ("pandas", "scipy") or name.startswith(("pandas.", "scipy.")):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real(name, *args, **kwargs)

    studies = ("pfander", "voelkel", "icpc", "goldwert", "ccc")
    for study in studies:
        for module in ("cli", "run"):
            for name in [
                m for m in list(sys.modules) if m.startswith("silicon_sampling")
            ]:
                del sys.modules[name]
            builtins.__import__ = guard
            try:
                importlib.import_module(f"silicon_sampling.{study}.{module}")
            finally:
                builtins.__import__ = real
