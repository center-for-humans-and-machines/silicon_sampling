"""The effect transforms, checked on the invariances that make them worth using.

The point of these tests is not that the arithmetic runs.  It is that each
transform touches exactly the metrics it is supposed to and leaves the others
alone — which is the entire basis for combining them.  A shrinkage that quietly
reordered the effects, or a profile substitution that quietly changed the message
ranking, would be useless even though every number it produced looked plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from silicon_sampling.benchmark import metrics as M
from silicon_sampling.calibration import effects as E


def effect_table(
    seed: int = 0, n_outcomes: int = 5, n_conditions: int = 8
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in range(n_outcomes):
        level = rng.normal(2.0, 2.0)
        for condition in range(n_conditions):
            rows.append(
                {
                    "outcome": f"out_{outcome}",
                    "condition": f"arm_{condition}",
                    "estimate": level + rng.normal(0, 3.0),
                    "se": abs(rng.normal(1.0, 0.2)),
                }
            )
    return pd.DataFrame(rows)


def test_global_shrink_preserves_every_scale_invariant_metric():
    """The claim that shrinkage cannot buy a leaderboard place, as a test."""
    table = effect_table(1)
    rng = np.random.default_rng(2)
    human = table["estimate"] * 0.2 + rng.normal(0, 1.0, len(table))
    pairs = pd.DataFrame(
        {
            "outcome": table["outcome"],
            "condition": table["condition"],
            "estimate_h": human,
            "se_h": table["se"],
        }
    )

    def scored(frame: pd.DataFrame) -> dict:
        joined = pairs.assign(estimate_l=frame["estimate"], se_l=frame["se"])
        return M.pooled_metrics(joined) | M.run_calibration_pooled(joined)

    base = scored(table)
    for factor in (0.5, 0.2, 0.05):
        moved = scored(E.global_shrink(table, factor))
        for metric in (
            "directional_pct",
            "spearman_rho",
            "pearson_r",
            "pearson_within",
            "pearson_adj",
            "alpha",
        ):
            assert moved[metric] == pytest.approx(base[metric], abs=1e-9), metric
        # beta scales inversely, which is the whole point of doing it
        assert moved["beta"] == pytest.approx(base["beta"] / factor, rel=1e-6)


def _pairs_for(table: pd.DataFrame, human, shrunk: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": table["outcome"],
            "condition": table["condition"],
            "estimate_h": human,
            "se_h": table["se"],
            "estimate_l": shrunk["estimate"],
            "se_l": shrunk["se"],
        }
    )


def test_slope_matching_factor_drives_beta_to_exactly_one():
    """beta is a with-intercept slope, so this is the factor that lands it on 1."""
    table = effect_table(3)
    rng = np.random.default_rng(4)
    human = table["estimate"] * 0.15 + rng.normal(0, 0.8, len(table))
    factor = E.slope_matching_factor(table["estimate"], human)
    pairs = _pairs_for(table, human, E.global_shrink(table, factor))
    assert M.run_calibration_pooled(pairs)["beta"] == pytest.approx(1.0, abs=1e-9)


def test_optimal_shrinkage_minimises_rmse_and_is_not_the_slope_factor():
    """The two factors answer different questions; conflating them is the bug."""
    table = effect_table(3)
    rng = np.random.default_rng(4)
    human = table["estimate"] * 0.15 + rng.normal(0, 0.8, len(table))
    best = E.optimal_shrinkage(table["estimate"], human)

    def rmse(factor: float) -> float:
        shrunk = E.global_shrink(table, factor)
        return M.pooled_metrics(_pairs_for(table, human, shrunk))["rmse"]

    assert rmse(best) < rmse(best * 0.8)
    assert rmse(best) < rmse(best * 1.25)
    # An intercept sits between them, so they must not be interchangeable here.
    assert best != pytest.approx(
        E.slope_matching_factor(table["estimate"], human), rel=1e-3
    )


def test_shrink_within_outcome_keeps_the_profile_and_the_ranking():
    table = effect_table(5)
    before = E.outcome_profile(table)
    shrunk = E.shrink_within_outcome(table, 0.4)
    after = E.outcome_profile(shrunk)
    for outcome in before.index:
        assert after[outcome] == pytest.approx(before[outcome], abs=1e-9)
    # Within each outcome the ordering of conditions must be untouched.
    for outcome, group in table.groupby("outcome"):
        mine = shrunk.loc[group.index, "estimate"]
        assert stats.spearmanr(group["estimate"], mine).statistic == pytest.approx(1.0)


def test_substitute_profile_replaces_the_profile_and_keeps_deviations():
    table = effect_table(6)
    anchor = pd.Series(
        {name: 1.0 for name in table["outcome"].unique()}, name="estimate"
    )
    swapped = E.substitute_profile(table, anchor, weight=1.0)
    for outcome, value in E.outcome_profile(swapped).items():
        assert value == pytest.approx(1.0, abs=1e-9)
    original = table["estimate"] - table.groupby("outcome")["estimate"].transform(
        "mean"
    )
    moved = swapped["estimate"] - swapped.groupby("outcome")["estimate"].transform(
        "mean"
    )
    assert np.allclose(original, moved)


def test_substitute_profile_leaves_unanchored_outcomes_alone():
    """A missing anchor is missing information, not a claim that the effect is zero."""
    table = effect_table(7)
    named = table["outcome"].unique()[0]
    anchor = pd.Series({named: 9.0})
    swapped = E.substitute_profile(table, anchor, weight=1.0)
    profile_before = E.outcome_profile(table)
    profile_after = E.outcome_profile(swapped)
    assert profile_after[named] == pytest.approx(9.0, abs=1e-9)
    for outcome in named_others(named, profile_before.index):
        assert profile_after[outcome] == pytest.approx(
            profile_before[outcome], abs=1e-9
        )


def named_others(name, index):
    return [value for value in index if value != name]


def test_weight_zero_is_the_identity():
    table = effect_table(8)
    anchor = pd.Series({name: 42.0 for name in table["outcome"].unique()})
    assert np.allclose(
        E.substitute_profile(table, anchor, weight=0.0)["estimate"], table["estimate"]
    )


def test_true_effect_sd_floors_when_spread_is_pure_noise():
    """An outcome whose spread is all sampling noise must read zero, not small."""
    rng = np.random.default_rng(9)
    rows = []
    for condition in range(12):
        rows.append(
            {
                "outcome": "noise_only",
                "condition": f"arm_{condition}",
                "estimate": rng.normal(0, 1.0),
                "se": 1.0,
            }
        )
        rows.append(
            {
                "outcome": "real_signal",
                "condition": f"arm_{condition}",
                "estimate": rng.normal(0, 6.0),
                "se": 1.0,
            }
        )
    table = pd.DataFrame(rows)
    spread = E.true_effect_sd(table)
    assert spread["noise_only"] < 1.0
    assert spread["real_signal"] > 4.0


def test_flatten_outcomes_only_touches_the_named_ones():
    table = effect_table(10)
    target = table["outcome"].unique()[0]
    flattened = E.flatten_outcomes(table, {target}, factor=0.1)
    untouched = table["outcome"] != target
    assert np.allclose(
        flattened.loc[untouched, "estimate"], table.loc[untouched, "estimate"]
    )
    picked = table["outcome"] == target
    assert flattened.loc[picked, "estimate"].std(ddof=1) < table.loc[
        picked, "estimate"
    ].std(ddof=1)


def test_variance_match_hits_the_requested_spread():
    table = effect_table(11)
    factor = E.variance_match(table["estimate"], target_sd=1.1)
    assert (table["estimate"] * factor).std(ddof=1) == pytest.approx(1.1, abs=1e-9)


def test_profile_agreement_is_one_against_itself():
    table = effect_table(12)
    profile = E.outcome_profile(table)
    agreement = E.profile_agreement(profile, profile)
    assert agreement["r"] == pytest.approx(1.0)
    assert agreement["n"] == table["outcome"].nunique()
