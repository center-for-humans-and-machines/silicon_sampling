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


def test_no_cli_touches_pandas_before_dispatching_a_subcommand():
    """``sample`` must not pay for pandas, and the lazy import is not enough.

    Two of the CLIs called ``pd.set_option("display.width", 220)`` immediately
    after ``parse_args`` — on *every* code path, including ``sample``. Making the
    module-level import lazy did not help: the first attribute access still
    resolved it, so the sampler still died in the Muse-Glimmer container, which
    ships no pandas.

    That cost twelve retry attempts per study on the cluster, twice, because the
    earlier guard test only checked that the module *imported*. Importing is not
    running. This checks the region between argument parsing and the first
    subcommand branch, which is the region ``sample`` always executes.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "silicon_sampling"
    offenders = {}
    for study in ("pfander", "voelkel", "icpc", "goldwert", "ccc"):
        source = (root / study / "cli.py").read_text()
        start = source.find("parser.parse_args(")
        if start < 0:
            continue
        first_branch = source.find("args.command ==", start)
        if first_branch < 0:
            continue
        preamble = source[start:first_branch]
        used = re.findall(r"\bpd\.\w+", preamble)
        if used:
            offenders[study] = sorted(set(used))
    assert not offenders, (
        "pandas is touched before subcommand dispatch, so `sample` will fail "
        f"wherever pandas is absent: {offenders}"
    )
