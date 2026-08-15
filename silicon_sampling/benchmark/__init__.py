"""The Silicon Sample Benchmark's scoring pipeline, in Python.

Study-independent: give it human and synthetic respondent-level frames sharing a
``condition`` column and a set of outcomes, and it produces the benchmark's
metrics — including the human replication reference that makes them readable.

Written for this validation exercise, but it is equally the tool for scoring a
Pfänder submission before filing it.
"""

from __future__ import annotations

from .distributions import (
    compare_distributions,
    compute_ks,
    compute_ovl,
    compute_w1,
    variance_ratio,
)
from .metrics import (
    adjusted_metrics,
    cluster_bootstrap,
    directional_score,
    pearson_within_outcome,
    pooled_metrics,
    run_calibration_pooled,
    signed_metrics,
)
from .reference import ate_pairs, half_split, treatment_effects

__all__ = [
    "adjusted_metrics",
    "ate_pairs",
    "cluster_bootstrap",
    "compare_distributions",
    "compute_ks",
    "compute_ovl",
    "compute_w1",
    "directional_score",
    "half_split",
    "pearson_within_outcome",
    "pooled_metrics",
    "run_calibration_pooled",
    "signed_metrics",
    "treatment_effects",
    "variance_ratio",
]
