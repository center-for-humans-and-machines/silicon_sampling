"""Applying a calibration to a Tier-1 file without invalidating it.

Three format properties can each silently break a submission, and each gets a
test: the composite must stay consistent with its twelve items, the binary
outcome must stay binary, and effects expressed in percentage points must not be
applied at ten times their size to the 0-10 donation scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.calibration import components as C
from silicon_sampling.calibration import tier1 as T1

CONTROL = "control"
ARMS = (CONTROL, "message_a", "message_b")


def tier1_frame(n: int = 3000, seed: int = 5) -> pd.DataFrame:
    """A minimal but structurally faithful Tier-1 frame."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "profile_id": [f"p{index:05d}" for index in range(n)],
            "condition": rng.choice(ARMS, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "age_band": rng.choice(["18-29", "30-44", "45-59", "60+"], size=n),
            "race": rng.choice(
                ["White / Caucasian", "Black / African American"], size=n
            ),
            "education": rng.choice(
                ["Less than high school", "Bachelor's degree"], size=n
            ),
            "income": rng.choice(["Less than $30,000", "$168,000 or more"], size=n),
            "party": rng.choice(["Republican", "Democrat"], size=n),
        }
    )
    shift = {CONTROL: 0.0, "message_a": 6.0, "message_b": -4.0}
    base = 55.0 + frame["condition"].map(shift).to_numpy()
    for item in T1.TRUST_ITEMS:
        frame[item] = np.clip(base + rng.normal(0, 20, n), 0, 100)
    frame["trust_multidimensional"] = frame[list(T1.TRUST_ITEMS)].mean(axis=1)
    for outcome in ("trust_post", "belief_post", "concern_mean"):
        frame[outcome] = np.clip(base + rng.normal(0, 18, n), 0, 100)
    frame["donation_ams"] = np.clip(
        3.0 + frame["condition"].map(shift).to_numpy() / 10 + rng.normal(0, 2, n), 0, 10
    )
    frame["newsletter_signup"] = (
        rng.random(n) < 0.3 + frame["condition"].map(shift).to_numpy() / 200
    ).astype(int)
    return frame


def targets_for(frame: pd.DataFrame, outcomes, factor: float) -> pd.DataFrame:
    """Our own effects in pp, scaled — the shape a real calibration produces."""
    rows = []
    for outcome in outcomes:
        scale = T1.scale_of(outcome)
        for arm, value in C.condition_effects(frame, outcome, CONTROL).items():
            rows.append(
                {
                    "outcome": outcome,
                    "condition": arm,
                    "estimate": float(value) * (100.0 / scale) * factor,
                    "se": 1.0,
                }
            )
    return pd.DataFrame(rows)


OUTCOMES = (
    "trust_multidimensional",
    "trust_post",
    "belief_post",
    "concern_mean",
    "donation_ams",
    "newsletter_signup",
)


def test_effects_land_on_target_and_composite_stays_consistent():
    frame = tier1_frame()
    factor = 0.2
    calibrated, drift = T1.calibrate(
        frame, targets=targets_for(frame, OUTCOMES, factor), outcomes=OUTCOMES
    )
    # continuous outcomes are exact; the binary one is limited by its 0/1 grain
    continuous = drift[drift["outcome"] != "newsletter_signup"]
    assert continuous["max_abs_effect_drift"].max() < 1e-6
    assert T1.composite_consistency(calibrated) < 1e-6

    before = C.condition_effects(frame, "trust_multidimensional", CONTROL)
    after = C.condition_effects(calibrated, "trust_multidimensional", CONTROL)
    for arm in ("message_a", "message_b"):
        assert after[arm] == pytest.approx(before[arm] * factor, rel=1e-6)


def test_items_at_the_ceiling_do_not_eat_the_shift():
    """The bug this module was fixed for: pinned items must not absorb the effect."""
    frame = tier1_frame()
    # pin a fifth of respondents' items at the top of the scale
    pinned = frame.index[: len(frame) // 5]
    for item in T1.TRUST_ITEMS:
        frame.loc[pinned, item] = 100.0
    frame["trust_multidimensional"] = frame[list(T1.TRUST_ITEMS)].mean(axis=1)

    calibrated, drift = T1.calibrate(
        frame,
        targets=targets_for(frame, ("trust_multidimensional",), 0.2),
        outcomes=("trust_multidimensional",),
    )
    row = drift.set_index("outcome").loc["trust_multidimensional"]
    assert row["max_abs_effect_drift"] < 1e-6
    assert T1.composite_consistency(calibrated) < 1e-6
    for item in T1.TRUST_ITEMS:
        assert calibrated[item].between(0, 100).all()


def test_binary_outcome_stays_binary_and_moves_toward_target():
    frame = tier1_frame()
    calibrated, _ = T1.calibrate(
        frame,
        targets=targets_for(frame, ("newsletter_signup",), 0.2),
        outcomes=("newsletter_signup",),
    )
    assert set(np.unique(calibrated["newsletter_signup"])) <= {0, 1}
    before = C.condition_effects(frame, "newsletter_signup", CONTROL)
    after = C.condition_effects(calibrated, "newsletter_signup", CONTROL)
    # shrunk toward zero, and by roughly the requested amount
    assert abs(after["message_a"]) < abs(before["message_a"])
    assert after["message_a"] == pytest.approx(before["message_a"] * 0.2, abs=0.01)


def test_percentage_points_are_converted_back_to_the_native_scale():
    """A pp effect applied raw would be ten times too large on the 0-10 donation."""
    table = pd.DataFrame(
        [
            {
                "outcome": "trust_post",
                "condition": "message_a",
                "estimate": 5.0,
                "se": 1.0,
            },
            {
                "outcome": "donation_ams",
                "condition": "message_a",
                "estimate": 5.0,
                "se": 1.0,
            },
            {
                "outcome": "newsletter_signup",
                "condition": "message_a",
                "estimate": 5.0,
                "se": 1.0,
            },
        ]
    )
    converted = T1.pp_to_raw(table).set_index("outcome")["estimate"]
    assert converted["trust_post"] == pytest.approx(5.0)
    assert converted["donation_ams"] == pytest.approx(0.5)
    assert converted["newsletter_signup"] == pytest.approx(0.05)


def test_outcomes_without_a_target_keep_their_own_effects():
    """A partial calibration is a valid input, not a silent zeroing."""
    frame = tier1_frame()
    partial = targets_for(frame, ("trust_post",), 0.2)
    calibrated, _ = T1.calibrate(frame, targets=partial, outcomes=OUTCOMES)
    before = C.condition_effects(frame, "concern_mean", CONTROL)
    after = C.condition_effects(calibrated, "concern_mean", CONTROL)
    for arm in ARMS:
        assert after[arm] == pytest.approx(before[arm], abs=1e-6)


def test_an_external_level_anchor_moves_the_control_mean_only():
    """How ground truth enters for a study with no human data of its own."""
    frame = tier1_frame()
    anchor = 40.0
    calibrated, drift = T1.calibrate(
        frame, levels={"trust_post": anchor}, outcomes=("trust_post",)
    )
    assert calibrated.loc[calibrated.condition == CONTROL, "trust_post"].mean() == (
        pytest.approx(anchor, abs=1e-6)
    )
    # the effects are untouched by a level move
    before = C.condition_effects(frame, "trust_post", CONTROL)
    after = C.condition_effects(calibrated, "trust_post", CONTROL)
    for arm in ARMS:
        assert after[arm] == pytest.approx(before[arm], abs=1e-6)
    assert drift["max_abs_effect_drift"].max() < 1e-6


def test_all_values_stay_inside_their_native_ranges():
    frame = tier1_frame()
    calibrated, _ = T1.calibrate(
        frame, targets=targets_for(frame, OUTCOMES, 3.0), outcomes=OUTCOMES
    )
    for outcome in OUTCOMES:
        assert calibrated[outcome].min() >= -1e-9
        assert calibrated[outcome].max() <= T1.scale_of(outcome) + 1e-9


def other_study_instrument() -> T1.Instrument:
    """A study with a different control label, different scales and no composite."""
    return T1.Instrument(
        scales={"attitude": 100.0, "support": 100.0, "spend": 20.0},
        control="null_arm",
        moderators=("party",),
        binary=(),
        composites={},
    )


def other_study_frame(n: int = 2000, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "condition": rng.choice(["null_arm", "treat"], size=n),
            "party": rng.choice(["Republican", "Democrat"], size=n),
        }
    )
    lift = np.where(frame["condition"] == "treat", 5.0, 0.0)
    frame["attitude"] = np.clip(45 + lift + rng.normal(0, 15, n), 0, 100)
    frame["support"] = np.clip(60 + lift + rng.normal(0, 18, n), 0, 100)
    frame["spend"] = np.clip(8 + lift / 10 + rng.normal(0, 3, n), 0, 20)
    return frame


def test_calibrate_works_for_a_study_that_is_not_pfander():
    """The path that validates a calibration must be the path that applies it.

    This module was briefly hard-wired to Pfänder's control label, scale ranges
    and trust battery, so a calibration could only be *validated* by calling the
    layer underneath — which is exactly where a bug survives review.
    """
    design = other_study_instrument()
    frame = other_study_frame()
    targets = pd.DataFrame(
        [
            {
                "outcome": outcome,
                "condition": "treat",
                "estimate": float(
                    C.condition_effects(frame, outcome, "null_arm")["treat"]
                )
                * (100.0 / design.scales[outcome])
                * 0.25,
                "se": 1.0,
            }
            for outcome in design.outcomes
        ]
    )
    calibrated, drift = T1.calibrate(frame, targets=targets, instrument=design)
    assert (drift["max_abs_effect_drift"] < 1e-9).all()
    for outcome in design.outcomes:
        before = C.condition_effects(frame, outcome, "null_arm")["treat"]
        after = C.condition_effects(calibrated, outcome, "null_arm")["treat"]
        assert after == pytest.approx(before * 0.25, rel=1e-6)
        assert calibrated[outcome].between(0, design.scales[outcome]).all()


def test_a_level_anchor_works_on_another_study_too():
    design = other_study_instrument()
    frame = other_study_frame()
    calibrated, drift = T1.calibrate(
        frame, levels={"attitude": 70.0}, instrument=design
    )
    control = calibrated[calibrated.condition == "null_arm"]
    assert control["attitude"].mean() == pytest.approx(70.0, abs=1e-6)
    assert (drift["max_abs_effect_drift"] < 1e-9).all()


def test_scale_conversion_uses_the_instrument_it_was_given():
    """A 0-20 scale must not be converted with Pfänder's 0-10 donation range."""
    design = other_study_instrument()
    table = pd.DataFrame(
        [{"outcome": "spend", "condition": "treat", "estimate": 10.0, "se": 1.0}]
    )
    converted = T1.pp_to_raw(table, instrument=design)["estimate"].iloc[0]
    assert converted == pytest.approx(2.0)
