"""The Silicon Sample Benchmark's scoring pipeline, in Python.

Study-independent: give it human and synthetic respondent-level frames sharing a
``condition`` column and a set of outcomes, and it produces the benchmark's
metrics — including the human replication reference that makes them readable.

Written for this validation exercise, but it is equally the tool for scoring a
Pfänder submission before filing it.

``metrics``, ``distributions`` and ``reference`` hold the pooled leaderboard row
and its inputs; ``scored`` holds everything else the benchmark scores — subgroup
interactions, per-outcome and per-intervention breakdowns, within-group response
shape, demographic baselines, the parity gap and the stereotyping regressions —
and assembles all of it into one flat row per submission
(:func:`~silicon_sampling.benchmark.scored.leaderboard_row`).
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
from .scored import (
    ReferenceSides,
    ScoredDesign,
    ScoredReport,
    align_submission_levels,
    assert_full_grid,
    ate_side,
    binary_marginal_effects,
    breakdowns,
    build_ate_pairs,
    build_subgroup_pairs,
    compare_demographic_baselines,
    compare_demographic_predictability,
    demographic_parity_gap,
    leaderboard_row,
    metrics_by_group,
    reference_sides,
    response_distributions,
    run_moderator_model,
    score_submission,
    subgroup_distributions,
    subgroup_metrics,
    subgroup_side,
    summarise_field,
)

__all__ = [
    "ReferenceSides",
    "ScoredDesign",
    "ScoredReport",
    "adjusted_metrics",
    "align_submission_levels",
    "assert_full_grid",
    "ate_pairs",
    "ate_side",
    "binary_marginal_effects",
    "breakdowns",
    "build_ate_pairs",
    "build_subgroup_pairs",
    "cluster_bootstrap",
    "compare_demographic_baselines",
    "compare_demographic_predictability",
    "compare_distributions",
    "compute_ks",
    "compute_ovl",
    "compute_w1",
    "demographic_parity_gap",
    "directional_score",
    "half_split",
    "leaderboard_row",
    "metrics_by_group",
    "pearson_within_outcome",
    "pooled_metrics",
    "reference_sides",
    "response_distributions",
    "run_calibration_pooled",
    "run_moderator_model",
    "score_submission",
    "signed_metrics",
    "subgroup_distributions",
    "subgroup_metrics",
    "subgroup_side",
    "summarise_field",
    "treatment_effects",
    "variance_ratio",
]
