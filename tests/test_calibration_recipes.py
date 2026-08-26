"""Recipes: the named chain from runs on disk to a calibrated, format-valid file.

The properties worth pinning down are the ones that were wrong at some point and
whose failure is silent: a recipe that swaps nothing must return its input
untouched, a binary outcome must survive, a composite must stay consistent with
its items, and averaging several runs' effects must actually average them rather
than quietly take the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.calibration import components as C
from silicon_sampling.calibration import offsets as OFF
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1

CONTROL = "control"
ARMS = (CONTROL, "arm_a", "arm_b")


def design() -> T1.Instrument:
    return T1.Instrument(
        scales={"attitude": 100.0, "spend": 10.0, "signup": 1.0},
        control=CONTROL,
        moderators=("party",),
        binary=("signup",),
        composites={},
    )


def run_frame(lift: float, level: float, seed: int, n: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "profile_id": [f"p{i:05d}" for i in range(n)],
            "condition": np.tile(np.array(ARMS), n // len(ARMS)),
            "party": rng.choice(["Republican", "Democrat"], size=n),
        }
    )
    shift = frame["condition"].map({CONTROL: 0.0, "arm_a": lift, "arm_b": -lift})
    frame["attitude"] = np.clip(level + shift + rng.normal(0, 15, n), 0, 100)
    frame["spend"] = np.clip(5 + shift / 10 + rng.normal(0, 2, n), 0, 10)
    frame["signup"] = (rng.random(n) < 0.3 + shift / 100).astype(int)
    return frame


def two_runs() -> dict[str, pd.DataFrame]:
    return {"ranker": run_frame(6.0, 45.0, 1), "grounded": run_frame(2.0, 62.0, 2)}


def test_a_recipe_that_swaps_nothing_returns_its_input():
    """Recomposition draws fresh residuals, so a no-op must skip it entirely."""
    runs = two_runs()
    out, drift = R.apply(R.uncalibrated("ranker"), runs=runs, instrument=design())
    assert (drift["max_abs_effect_drift"] < 1e-9).all()
    for column in ("attitude", "spend", "signup"):
        assert np.allclose(out[column], runs["ranker"][column])


def test_averaging_effects_actually_averages_them():
    runs = two_runs()
    spec = design()
    recipe = R.Recipe(name="avg", effects_from=("ranker", "grounded"))
    averaged = R.averaged_effects(recipe, runs, spec)
    left = R.effect_table(runs["ranker"], spec).set_index(["outcome", "condition"])
    right = R.effect_table(runs["grounded"], spec).set_index(["outcome", "condition"])
    got = averaged.set_index(["outcome", "condition"])
    for key in got.index:
        expected = (left.loc[key, "estimate"] + right.loc[key, "estimate"]) / 2
        assert got.loc[key, "estimate"] == pytest.approx(expected, rel=1e-9)


def test_an_averaged_recipe_lands_on_the_averaged_effects():
    runs = two_runs()
    spec = design()
    recipe = R.Recipe(name="avg", effects_from=("ranker", "grounded"), shrink=None)
    out, drift = R.apply(recipe, runs=runs, instrument=spec)
    continuous = drift[drift["outcome"] != "signup"]
    assert (continuous["max_abs_effect_drift"] < 1e-6).all()

    wanted = R.averaged_effects(recipe, runs, spec)
    wanted = wanted[wanted.outcome == "attitude"].set_index("condition")["estimate"]
    got = C.condition_effects(out, "attitude", CONTROL)
    for arm in ("arm_a", "arm_b"):
        assert got[arm] == pytest.approx(wanted[arm], abs=1e-6)


def test_the_first_effect_run_supplies_the_respondents():
    runs = two_runs()
    recipe = R.Recipe(name="avg", effects_from=("ranker", "grounded"))
    assert recipe.template_run == "ranker"
    out, _ = R.apply(recipe, runs=runs, instrument=design())
    assert list(out["profile_id"]) == list(runs["ranker"]["profile_id"])


def test_a_component_swap_takes_the_level_from_the_named_run():
    runs = two_runs()
    spec = design()
    out, drift = R.apply(
        R.Recipe(
            name="hybrid",
            effects_from="ranker",
            level_from="grounded",
            offsets_from="grounded",
            residuals_from="grounded",
        ),
        runs=runs,
        instrument=spec,
    )
    assert C.control_level(out, "attitude", CONTROL) == pytest.approx(
        C.control_level(runs["grounded"], "attitude", CONTROL), abs=1e-6
    )
    wanted = C.condition_effects(runs["ranker"], "attitude", CONTROL)
    got = C.condition_effects(out, "attitude", CONTROL)
    for arm in ARMS:
        assert got[arm] == pytest.approx(wanted[arm], abs=1e-6)


def test_the_binary_outcome_survives_a_component_swap():
    """Recomposing it additively once collapsed a 0.311 signup rate to 0.003."""
    runs = two_runs()
    out, _ = R.apply(
        R.Recipe(
            name="hybrid",
            effects_from="ranker",
            level_from="grounded",
            offsets_from="grounded",
            residuals_from="grounded",
        ),
        runs=runs,
        instrument=design(),
    )
    assert set(np.unique(out["signup"])) <= {0, 1}
    assert out["signup"].mean() == pytest.approx(
        runs["ranker"]["signup"].mean(), abs=0.02
    )


def test_shrinkage_holds_the_reference_condition_at_zero():
    """The control row is not an effect; shrinking toward a mean including it
    gave the control arm a spurious effect and moved the baseline."""
    runs = two_runs()
    spec = design()
    out, _ = R.apply(
        R.Recipe(name="shrunk", effects_from="ranker", shrink=0.2, flatten_noise=True),
        runs=runs,
        instrument=spec,
    )
    for outcome in ("attitude", "spend"):
        assert C.control_level(out, outcome, CONTROL) == pytest.approx(
            C.control_level(runs["ranker"], outcome, CONTROL), abs=1e-6
        )


def test_describe_names_every_moving_part():
    text = R.describe(R.hybrid_default(effects_from=("a", "b"), grounded="c"))
    assert "a+b" in text and "levels=c" in text and "shrink" in text
    assert "shrink" not in R.describe(
        R.hybrid_default(effects_from=("a", "b"), grounded="c", shrink=None)
    )


def test_the_default_shrinks_at_the_measured_ratio():
    """Shrinkage is on by default, and the two factors ship as a matched pair.

    Absolute human effect magnitudes differ 4.5-fold across the reference studies,
    so no absolute effect *target* transfers -- but the fitted *ratio* does.

    The number moved twice.  It was first fitted partly on the ICPC and Goldwert
    samples taken before the fidelity audit, whose questionnaires printed slider
    endpoint labels without their 0-100 range; those models answered on an
    implicit 0-10 scale and their effects came out compressed about five-fold, so
    the shrinkage fitted against them was too aggressive.  Re-fitted on the
    audited samples it roughly doubled.

    It then moved again because the global factor and the within-outcome factor
    interact: out-of-fold the best global k falls from 0.475 to 0.250 as the
    within factor rises from 0.2 to 1.0.  Pairing the no-within value with
    within-shrinkage 0.5 overshoots to beta 1.53, which is what this test guards.
    """
    assert R.hybrid_default().shrink == pytest.approx(R.GLOBAL_SHRINK)
    assert R.hybrid_default().within_shrink == pytest.approx(R.WITHIN_SHRINK)
    assert 0.25 < R.GLOBAL_SHRINK < 0.55
    # The pair has to stay on the fitted line k ~= 0.525 - 0.275 * within; a
    # constant changed on its own is the failure mode this catches.
    expected = 0.525 - 0.275 * R.WITHIN_SHRINK
    assert R.GLOBAL_SHRINK == pytest.approx(expected, abs=0.06)
    scale = R.HUMAN_EFFECT_SCALE
    # the absolute magnitudes really do disagree; it is the ratio that does not
    assert max(scale.values()) / min(scale.values()) > 4.0


def test_party_gaps_move_toward_the_external_anchors():
    """The party calibration closes the gap it is supposed to, and only that.

    Party is the one Pfander moderator the model is not told: it is elicited at
    Q16, before almost every outcome.  The real gap is strongly topic-dependent --
    4.0 pp on the trust battery TISP measures item-for-item, 27 pp on climate
    policy -- and the models apply a roughly uniform one, so the anchors pull the
    per-outcome profile into shape rather than inflating everything.
    """
    weight = R.PARTY_GAP_WEIGHT
    assert 0.0 < weight <= 1.0
    # Half, not all: two of the three public sources contrast ideology, not party.
    assert weight < 1.0
    anchors = R.PARTY_GAP_ANCHORS
    # The topic spread is the whole point; a flat set of anchors would do nothing
    # that the donor model does not already do.
    assert max(anchors.values()) / min(anchors.values()) > 5.0
    assert anchors["trust_multidimensional"] < anchors["policy_general"]

    levels = pd.Series(0.0, index=["Democrat", "Republican", "Independent"])
    shares = pd.Series({"Democrat": 400, "Republican": 350, "Independent": 250})
    imposed = OFF.impose_gap(levels, 20.0, "Democrat", "Republican", shares)
    assert imposed["Democrat"] - imposed["Republican"] == pytest.approx(20.0)
    # Offsets are deviations from the arm mean, so they must stay centred or the
    # calibration would move the level while fixing the gap.
    assert float((imposed * shares).sum() / shares.sum()) == pytest.approx(0.0)


def test_impose_gap_leaves_a_moderator_it_cannot_place_alone():
    """A missing level means missing information, not a finding of no gap."""
    levels = pd.Series({"Yes": 1.0, "No": -1.0})
    same = OFF.impose_gap(levels, 20.0, "Democrat", "Republican")
    pd.testing.assert_series_equal(same, levels)


def test_residual_scaling_touches_only_the_within_cell_spread():
    """It must reach the variance ratio without disturbing the effects.

    Level, condition effects and demographic offsets are separate terms of the
    decomposition, so scaling residuals cannot move them.  That is the property
    that makes this safe to stack on top of an already-calibrated effect vector,
    and it is worth a test because the obvious alternative -- scaling the whole
    outcome around its mean -- would silently shrink every effect too.
    """
    assert 0.8 < R.RESIDUAL_SCALE < 1.3
    assert R.hybrid_default().residual_scale == pytest.approx(R.RESIDUAL_SCALE)
    # The uncalibrated baseline must stay untouched, or it stops being a baseline.
    assert R.uncalibrated("qwen25_7b").residual_scale == pytest.approx(1.0)
