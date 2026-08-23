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
    assert "a+b" in text and "levels=c" in text
    assert "shrink" not in text, "the default no longer shrinks; see hybrid_default"
    assert "shrink=0.46" in R.describe(
        R.hybrid_default(effects_from=("a", "b"), grounded="c", shrink=0.46)
    )


def test_the_default_recipe_does_not_shrink():
    """Shrinkage is a variant, not a default.

    The factor is the ratio of real effects to ours, and real effects differ 4.5x
    between the reference studies (Voelkel 1.125 pp, Goldwert 2.967, ICPC 5.035).
    Against our 2.46 pp the implied factor spans 0.46 to 2.04 — across 1.0 — so the
    direction of the correction is undetermined and defaulting to Voelkel's 0.159
    would under-predict a climate target by 8-13x.
    """
    assert R.hybrid_default().shrink is None
    scale = R.HUMAN_EFFECT_SCALE
    assert max(scale.values()) / min(scale.values()) > 4.0
