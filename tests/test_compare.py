"""Checks for the two-model comparison.

These cover the pieces where a sign or a pairing could silently invert a
conclusion.  The heavier end-to-end paths need the study data and the human
responses, so they are exercised by running the reports, not from here.

Runs under plain ``python tests/test_compare.py`` as well as pytest.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from silicon_sampling.pfander import compare as pfc
from silicon_sampling.voelkel.score import COMPARISON_METRICS, share_better


def test_share_better_flips_for_error_metrics():
    # A correlation that rose in 97% of resamples improved in 97% of resamples.
    assert share_better(0.97, higher_is_better=True) == 0.97
    # An RMSE that rose in 97% of resamples *worsened* in 97% of them.
    assert abs(share_better(0.97, higher_is_better=False) - 0.03) < 1e-12
    # The metric table must actually mark RMSE as lower-is-better, or the flip
    # above never gets applied where it matters.
    assert COMPARISON_METRICS["rmse"] is False
    assert COMPARISON_METRICS["pearson_r"] is True


def _effect_table(estimates: dict, outcome: str = "trust_multidimensional"):
    """The columns `effect_agreement` reads, and nothing else."""
    return pd.DataFrame(
        [
            {"outcome": outcome, "condition": condition, "estimate": value}
            for condition, value in estimates.items()
        ]
    )


def test_effect_agreement_is_perfect_for_an_identical_sample():
    values = {f"arm{i}": float(i) - 2.0 for i in range(6)}
    tables = {"a": _effect_table(values), "b": _effect_table(values)}
    row = pfc.effect_agreement(tables, "a", "b").iloc[0]
    assert abs(row["pearson_r"] - 1.0) < 1e-12
    assert abs(row["sd_ratio"] - 1.0) < 1e-12
    assert row["sign_agreement"] == 1.0
    assert row["n_interventions"] == 6


def test_effect_agreement_reports_a_rescaled_sample_as_agreeing_but_wider():
    """Ordering preserved, magnitudes doubled — the case that matters.

    A model can rank the interventions perfectly and still be unusable on levels,
    which is exactly what the Voelkel scoring found. So r must stay at 1 while
    sd_ratio reports the exaggeration.
    """
    values = {f"arm{i}": float(i) - 2.0 for i in range(6)}
    doubled = {key: 2.0 * value for key, value in values.items()}
    tables = {"a": _effect_table(values), "b": _effect_table(doubled)}
    row = pfc.effect_agreement(tables, "a", "b").iloc[0]
    assert abs(row["pearson_r"] - 1.0) < 1e-12
    assert abs(row["sd_ratio"] - 2.0) < 1e-12


def test_effect_agreement_matches_on_condition_not_row_order():
    """The two tables need not list the arms in the same order."""
    values = {"a1": 1.0, "a2": -2.0, "a3": 3.0, "a4": 0.5}
    shuffled = dict(reversed(list(values.items())))
    tables = {"x": _effect_table(values), "y": _effect_table(shuffled)}
    row = pfc.effect_agreement(tables, "x", "y").iloc[0]
    assert abs(row["pearson_r"] - 1.0) < 1e-12


def test_partisan_gap_recovers_a_planted_gap():
    rng = np.random.default_rng(4)
    party = ["Republican"] * 400 + ["Democrat"] * 400
    values = np.concatenate(
        [rng.normal(50, 5, 400), rng.normal(80, 5, 400)]  # a planted -30 gap
    )
    sample = pd.DataFrame({"party": party, "belief_post": values})
    row = pfc.partisan_gap({"qwen25_7b": sample}).iloc[0]
    assert abs(row["gap"] - -30.0) < 1.5
    assert row["republican"] < row["democrat"]


def test_partisan_gap_skips_a_sample_without_the_columns():
    assert len(pfc.partisan_gap({"qwen25_7b": pd.DataFrame({"x": [1, 2]})})) == 0


def _control_arm(gaps: dict, n: int = 400, seed: int = 9):
    """A control-arm sample with planted Republican-minus-Democrat gaps."""
    rng = np.random.default_rng(seed)
    party = ["Republican"] * n + ["Democrat"] * n
    frame = {"condition": ["Null_Control"] * (2 * n), "party_gen": party}
    for outcome, gap in gaps.items():
        frame[outcome] = np.concatenate(
            [rng.normal(50 + gap, 3, n), rng.normal(50, 3, n)]
        )
    return pd.DataFrame(frame)


def test_party_gaps_recover_planted_gaps_and_rank_the_models():
    from silicon_sampling.voelkel import compare as vc

    planted = {"PA": 10.0, "ADA": -6.0, "SUC": 0.0}
    human = _control_arm(planted, seed=1)
    flat = _control_arm({k: 0.3 * v for k, v in planted.items()}, seed=2)
    loud = _control_arm({k: 3.0 * v for k, v in planted.items()}, seed=3)

    table = vc.party_gaps(human, {"qwen25_7b": flat, "v4_flash": loud})
    covered = table.dropna(subset=["human"])
    assert set(covered["outcome"]) == set(planted)
    for outcome, gap in planted.items():
        got = float(covered.loc[covered["outcome"] == outcome, "human"].iloc[0])
        assert abs(got - gap) < 1.0, (outcome, got, gap)

    summary = vc.party_gap_summary(table).set_index("model")
    # The flat model understates the gaps, the loud one overstates them, and both
    # track the human ordering — which is exactly the shape the report claims.
    assert (
        summary.loc["Qwen2.5-7B", "mean_abs_gap"]
        < summary.loc["human (Human 1)", "mean_abs_gap"]
    )
    assert (
        summary.loc["DeepSeek-V4-Flash", "mean_abs_gap"]
        > summary.loc["human (Human 1)", "mean_abs_gap"]
    )
    assert summary.loc["human (Human 1)", "corr_with_human_gaps"] == 1.0


def test_party_gaps_ignore_treated_arms():
    """Gaps are a control-arm quantity; an intervention arm must not leak in."""
    from silicon_sampling.voelkel import compare as vc

    human = _control_arm({"PA": 8.0}, seed=4)
    treated = _control_arm({"PA": -40.0}, seed=5).assign(condition="Party_Overlap")
    mixed = pd.concat([human, treated], ignore_index=True)
    only_control = vc.party_gaps(human, {"qwen25_7b": human})
    with_treated = vc.party_gaps(human, {"qwen25_7b": mixed})
    a = float(only_control.loc[only_control["outcome"] == "PA", "Qwen2.5-7B"].iloc[0])
    b = float(with_treated.loc[with_treated["outcome"] == "PA", "Qwen2.5-7B"].iloc[0])
    assert abs(a - b) < 1e-9


def main() -> int:
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
