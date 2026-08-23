"""Checks for the ported benchmark metrics, against hand-computed cases."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from silicon_sampling.analysis.ols import design_matrix, ols
from silicon_sampling.benchmark import distributions as dist
from silicon_sampling.benchmark import metrics as M
from silicon_sampling.benchmark.reference import (
    ate_pairs,
    half_split,
    treatment_effects,
)


def test_directional_score_gives_zeros_half_credit():
    # 2 of 3 signs agree, and the zero scores 0.5 -> (1 + 0 + 0.5) / 3
    assert np.isclose(
        M.directional_score([1, -1, 5], [2, 3, 0]), 100 * (1 + 0 + 0.5) / 3
    )
    # An all-zero predictor makes no directional claim anywhere.
    assert M.directional_score([1, -1, 5], [0, 0, 0]) == 50.0
    # A perfect predictor.
    assert M.directional_score([1, -1, 5], [2, -3, 1]) == 100.0


def test_noise_correction_inflates_correlation_toward_truth():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 5, 200)
    se = np.full(200, 3.0)
    reference = truth + rng.normal(0, 3.0, 200)  # a noisy measurement of truth
    pairs = pd.DataFrame(
        {"estimate_h": reference, "estimate_l": truth, "se_h": se, "outcome": "x"}
    )
    raw = M.pearson(pairs["estimate_h"], pairs["estimate_l"])
    adjusted = M.adjusted_metrics(pairs)["pearson_adj"]
    # The prediction *is* the truth, so the only thing deflating r is reference
    # noise; correcting for it should move r decisively toward 1.
    assert adjusted > raw + 0.1
    assert adjusted <= 1.0


def test_adjusted_rmse_floors_at_zero():
    pairs = pd.DataFrame(
        {
            "estimate_h": [1.0, 2.0, 3.0],
            "estimate_l": [1.0, 2.0, 3.0],
            "se_h": [2.0, 2.0, 2.0],
        }
    )
    out = M.adjusted_metrics(pairs)
    assert out["rmse_adj"] == 0.0 and out["rmse_adj_at_floor"]


def test_calibration_recovers_a_known_line():
    lo = np.linspace(-5, 5, 50)
    pairs = pd.DataFrame(
        {"estimate_l": lo, "estimate_h": 2.0 + 0.5 * lo, "outcome": "x"}
    )
    fit = M.run_calibration_pooled(pairs)
    assert np.isclose(fit["alpha"], 2.0) and np.isclose(fit["beta"], 0.5)


def test_calibration_slope_correction_undoes_prediction_noise():
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 4, 400)
    noisy = truth + rng.normal(0, 2, 400)
    pairs = pd.DataFrame(
        {
            "estimate_h": truth,
            "estimate_l": noisy,
            "se_l": np.full(400, 2.0),
            "outcome": "x",
        }
    )
    fit = M.run_calibration_pooled(pairs)
    # Noise in the predictions drags the raw slope below 1; the correction lifts
    # it back toward the latent slope of 1.
    assert fit["beta"] < 0.9
    assert abs(fit["beta_adj"] - 1.0) < abs(fit["beta"] - 1.0)


def test_within_outcome_correlation_strips_outcome_level_signal():
    # Two outcomes on very different levels, but no within-outcome signal at all.
    pairs = pd.DataFrame(
        {
            "outcome": ["a"] * 4 + ["b"] * 4,
            "estimate_h": [10, 10, 10, 10, -10, -10, -10, -10],
            "estimate_l": [5, 5, 5, 5, -5, -5, -5, -5],
        }
    )
    assert M.pearson(pairs["estimate_h"], pairs["estimate_l"]) == 1.0
    assert np.isnan(M.pearson_within_outcome(pairs))


def test_ovl_and_w1_agree_on_identical_and_disjoint_samples():
    rng = np.random.default_rng(2)
    a = rng.normal(50, 10, 500)
    assert dist.compute_ovl(a, a.copy(), lo=0, hi=100) > 0.95
    assert dist.compute_w1(a, a.copy(), lo=0, hi=100) < 1.0
    far = rng.normal(50, 10, 500) + 60
    assert dist.compute_ovl(a, far, lo=0, hi=200) < 0.2
    assert dist.compute_w1(a, far, lo=0, hi=200) > 30


def test_bandwidth_matches_silvermans_rule():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    expected = 0.9 * min(sd, iqr / 1.349) * 5 ** (-0.2)
    assert np.isclose(dist.bw_nrd0(x), expected)


def test_variance_ratio_is_one_for_a_perfect_match():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 7, 400)
    assert np.isclose(dist.variance_ratio(a, a.copy()), 1.0)


def test_hc2_differs_from_hc1_and_both_recover_the_mean_difference():
    rng = np.random.default_rng(4)
    group = np.array([0] * 300 + [1] * 60)
    y = np.where(group == 1, rng.normal(10, 8, 360), rng.normal(4, 2, 360))
    X, names = design_matrix({"g": group.tolist()})
    hc1 = ols(X, y, names, robust="HC1")
    hc2 = ols(X, y, names, robust="HC2")
    diff = y[group == 1].mean() - y[group == 0].mean()
    assert np.isclose(hc1.beta[1], diff) and np.isclose(hc2.beta[1], diff)
    # With cells this uneven the leverage correction is not cosmetic.
    assert hc2.se[1] > hc1.se[1]


def test_half_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame({"i": range(1000)})
    a1, b1 = half_split(frame, seed=42)
    a2, b2 = half_split(frame, seed=42)
    assert list(a1["i"]) == list(a2["i"]) and list(b1["i"]) == list(b2["i"])
    assert set(a1["i"]).isdisjoint(set(b1["i"]))
    assert len(a1) + len(b1) == 1000


def test_half_split_can_draw_an_exact_half():
    frame = pd.DataFrame({"id": range(101)})
    left, right = half_split(frame, exact=True)
    # floor(n/2) in the reference half, the remainder in the replication half —
    # the benchmark's own sample(ids, floor(n/2)) draw.
    assert len(left) == 50 and len(right) == 51
    assert set(left.index) | set(right.index) == set(frame.index)
    # The default is a coin flip, so it lands near half but not on it.
    default_left, _ = half_split(frame)
    assert len(default_left) != 50


def test_treatment_effects_recover_a_planted_effect_and_pair_up():
    rng = np.random.default_rng(5)
    rows = []
    for condition, shift in (
        ("control", 0.0),
        ("a", 5.0),
        ("b", -3.0),
        ("c", 1.5),
        ("d", -8.0),
    ):
        for value in rng.normal(50 + shift, 10, 800):
            rows.append({"condition": condition, "y": value})
    frame = pd.DataFrame(rows)
    effects = treatment_effects(frame, {"y": 100.0}, control="control")
    planted = {"a": 5.0, "b": -3.0, "c": 1.5, "d": -8.0}
    for _, row in effects.iterrows():
        assert abs(row["estimate"] - planted[row["condition"]]) < 1.5
    pairs = ate_pairs(effects, effects)
    assert len(pairs) == 4
    assert M.pooled_metrics(pairs)["pearson_r"] == 1.0


def test_cluster_bootstrap_widens_with_few_clusters():
    rng = np.random.default_rng(6)
    pairs = pd.DataFrame(
        {
            "condition": np.repeat([f"c{i}" for i in range(6)], 9),
            "outcome": list(range(9)) * 6,
            "estimate_h": rng.normal(0, 3, 54),
            "estimate_l": rng.normal(0, 3, 54),
        }
    )
    out = M.cluster_bootstrap(
        pairs,
        lambda p: {"pearson_r": M.pearson(p["estimate_h"], p["estimate_l"])},
        draws=300,
    )
    assert out["n_clusters"] == 6
    assert out["pearson_r_hi"] > out["pearson_r_lo"]


def _paired_submissions(seed: int = 11):
    """Two models scored against the same humans, sharing most of their error.

    Both miss each intervention cluster in the same direction — same instrument,
    same interventions, same reference — and differ only slightly.  This is the
    situation the paired bootstrap exists for.
    """
    rng = np.random.default_rng(seed)
    left, right = [], []
    for cluster in range(8):
        shared_bias = rng.normal(0, 1.5)
        for outcome in range(6):
            truth = rng.normal(0, 2)
            human = truth + rng.normal(0, 0.3)  # identical in both submissions
            common = truth + shared_bias
            base = {
                "condition": f"c{cluster}",
                "outcome": outcome,
                "se_h": 0.3,
                "se_l": 0.5,
            }
            left.append(
                {
                    **base,
                    "estimate_h": human,
                    "estimate_l": common + rng.normal(0, 0.15),
                }
            )
            right.append(
                {
                    **base,
                    "estimate_h": human,
                    "estimate_l": common * 0.97 + rng.normal(0, 0.15),
                }
            )
    return pd.DataFrame(left), pd.DataFrame(right)


def _pooled(pairs):
    return {**M.pooled_metrics(pairs), **M.run_calibration_pooled(pairs)}


def test_pairing_resolves_a_difference_separate_intervals_call_a_tie():
    left, right = _paired_submissions()
    alone_l = M.cluster_bootstrap(left, _pooled, draws=600)
    alone_r = M.cluster_bootstrap(right, _pooled, draws=600)
    paired = M.paired_cluster_bootstrap(left, right, _pooled, draws=600)

    # Scored separately the two are indistinguishable: the intervals overlap.
    assert alone_l["rmse_lo"] < alone_r["rmse_hi"]
    assert alone_r["rmse_lo"] < alone_l["rmse_hi"]

    # Paired, the difference is real and far more precisely estimated than either
    # level — which is the whole point of resampling one set of clusters for both.
    width_paired = paired["rmse_delta_hi"] - paired["rmse_delta_lo"]
    width_alone = alone_l["rmse_hi"] - alone_l["rmse_lo"]
    assert width_paired < width_alone / 5
    assert paired["rmse_delta_hi"] < 0  # the contender really has lower error
    assert paired["rmse_p_gt0"] < 0.05


def test_paired_delta_is_the_difference_of_the_point_estimates():
    left, right = _paired_submissions(seed=3)
    paired = M.paired_cluster_bootstrap(left, right, _pooled, draws=200)
    for metric in ("rmse", "pearson_r", "beta"):
        expected = _pooled(right)[metric] - _pooled(left)[metric]
        assert abs(paired[f"{metric}_delta"] - expected) < 1e-9
    assert paired["n_clusters"] == 8


def test_paired_bootstrap_needs_shared_clusters():
    left, right = _paired_submissions(seed=5)
    right = right.assign(condition=right["condition"] + "_other")
    assert M.paired_cluster_bootstrap(left, right, _pooled, draws=50) == {}


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
