"""Demographic offset rescaling, tested on the behaviour that makes it safe.

The transform is only worth having if it shrinks moderators we know nothing about
and keeps the ones we do, without disturbing the condition effects — those three
properties are what the tests below pin down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.calibration import components as C
from silicon_sampling.calibration import offsets as OFF

CONTROL = "control"
MODS = ("party", "age")
OUTCOMES = {f"out_{index}": 100.0 for index in range(4)}


def paired_runs(n: int = 4000):
    """Party gaps that point the right way but too small; age gaps that invert.

    Both moderators are given *real* structure rather than noise, because a
    moderator with genuinely random offsets gives a correlation over a handful of
    cells that is itself random — on three age levels and one outcome it came out
    at 0.999 by chance, which tests nothing.  Inverting the sign instead makes
    "this moderator is worse than useless" a deterministic property, and shrinking
    it to zero the deterministic right answer.
    """

    def build(party_gap: float, age_sign: float, seed: int) -> pd.DataFrame:
        local = np.random.default_rng(seed)
        frame = pd.DataFrame(
            {
                "condition": local.choice([CONTROL, "arm"], size=n),
                "party": local.choice(["Republican", "Democrat"], size=n),
                "age": local.choice(["18-29", "30-44", "60+"], size=n),
            }
        )
        for index, outcome in enumerate(OUTCOMES):
            scale = 1.0 + 0.4 * index
            age_shift = {
                "18-29": age_sign * 6.0 * scale,
                "30-44": 0.0,
                "60+": -age_sign * 6.0 * scale,
            }
            frame[outcome] = (
                50.0
                + np.where(
                    frame["party"] == "Democrat", party_gap * scale, -party_gap * scale
                )
                + frame["age"].map(age_shift)
                + local.normal(0, 12, n)
            ).clip(0, 100)
        return frame

    human = build(party_gap=15.0, age_sign=+1.0, seed=101)
    synthetic = build(party_gap=4.0, age_sign=-1.0, seed=202)
    return synthetic, human


def test_compare_offsets_pairs_only_cells_present_on_both_sides():
    synthetic, human = paired_runs()
    synthetic = synthetic.copy()
    synthetic.loc[synthetic.index[:20], "party"] = "Libertarian"
    paired = OFF.compare_offsets(synthetic, human, MODS, OUTCOMES, CONTROL)
    assert "Libertarian" not in set(paired["level"])
    assert {"party", "age"} == set(paired["moderator"])


def test_offset_recovery_finds_the_signal_and_the_noise():
    synthetic, human = paired_runs()
    paired = OFF.compare_offsets(synthetic, human, MODS, OUTCOMES, CONTROL)
    table = OFF.offset_recovery(paired, by_moderator=True).set_index("moderator")
    # party points the same way as the humans, so it carries signal
    assert table.loc["party", "r"] > 0.9
    # and it is too small, so it wants scaling UP
    assert table.loc["party", "scale_variance_match"] > 2.0
    # age points the wrong way, so it is worse than useless
    assert table.loc["age", "r"] < -0.9
    assert table.loc["age", "scale_rmse_optimal"] < 0


def test_fit_offset_scales_floors_uninformative_moderators_at_zero():
    synthetic, human = paired_runs()
    paired = OFF.compare_offsets(synthetic, human, MODS, OUTCOMES, CONTROL)
    scales = OFF.fit_offset_scales(paired, objective="rmse", floor=0.0)
    assert scales["party"] > 0.5
    # an inverted moderator must be flattened, never left to point the wrong way
    assert scales["age"] == pytest.approx(0.0, abs=1e-9)


def test_variance_objective_inflates_where_rmse_objective_shrinks():
    """The two objectives genuinely disagree, and the caller must choose."""
    synthetic, human = paired_runs()
    paired = OFF.compare_offsets(synthetic, human, MODS, OUTCOMES, CONTROL)
    rmse_scales = OFF.fit_offset_scales(paired, objective="rmse")
    variance_scales = OFF.fit_offset_scales(paired, objective="variance")
    # Variance matching only looks at magnitudes, so it happily keeps a moderator
    # that points the wrong way; the error-minimising fit flattens it. That
    # disagreement is the reason the objective is a caller's choice.
    assert variance_scales["age"] > 0.5
    assert rmse_scales["age"] == pytest.approx(0.0, abs=1e-9)
    assert rmse_scales["age"] < variance_scales["age"]


def test_rescaling_offsets_leaves_condition_effects_untouched():
    """The whole point of the decomposition: one term moves, the others do not."""
    synthetic, human = paired_runs()
    paired = OFF.compare_offsets(synthetic, human, MODS, OUTCOMES, CONTROL)
    scales = OFF.fit_offset_scales(paired, objective="variance")

    outcome = next(iter(OUTCOMES))
    before = C.condition_effects(synthetic, outcome, CONTROL)
    part = C.decompose(synthetic, outcome, MODS, CONTROL)
    rebuilt, drift = C.recompose_frame(
        synthetic,
        {
            outcome: C.Decomposition(
                outcome,
                part.level,
                part.effects,
                OFF.rescale_offsets(part.offsets, scales),
                part.residuals,
            )
        },
        CONTROL,
        bounds={outcome: (0.0, 100.0)},
    )
    after = C.condition_effects(rebuilt, outcome, CONTROL)
    assert drift["max_abs_effect_drift"].max() < 1e-9
    for arm in before.index:
        assert after[arm] == pytest.approx(before[arm], abs=1e-9)
    # and the party gap really did grow
    grew = C.cell_offsets(rebuilt, "party", outcome, CONTROL)
    original = C.cell_offsets(synthetic, "party", outcome, CONTROL)
    assert abs(grew["Democrat"] - grew["Republican"]) > abs(
        original["Democrat"] - original["Republican"]
    )


def test_rescale_leaves_moderators_without_a_factor_alone():
    synthetic, _ = paired_runs()
    part = C.decompose(synthetic, next(iter(OUTCOMES)), MODS, CONTROL)
    rescaled = OFF.rescale_offsets(part.offsets, {"party": 2.0})
    assert np.allclose(rescaled["age"], part.offsets["age"])
    assert np.allclose(rescaled["party"], part.offsets["party"] * 2.0)


def test_blend_offsets_moves_toward_an_external_anchor():
    synthetic, human = paired_runs()
    outcome = next(iter(OUTCOMES))
    ours = C.decompose(synthetic, outcome, MODS, CONTROL).offsets
    theirs = C.decompose(human, outcome, MODS, CONTROL).offsets
    blended = OFF.blend_offsets(ours, theirs, weight=1.0)
    assert np.allclose(blended["party"].sort_index(), theirs["party"].sort_index())
    half = OFF.blend_offsets(ours, theirs, weight=0.5)
    expected = 0.5 * theirs["party"] + 0.5 * ours["party"]
    assert np.allclose(half["party"].sort_index(), expected.sort_index())


def test_blend_keeps_our_value_where_the_anchor_says_nothing():
    synthetic, human = paired_runs()
    ours = C.decompose(synthetic, next(iter(OUTCOMES)), MODS, CONTROL).offsets
    partial = {"party": pd.Series({"Democrat": 20.0})}
    blended = OFF.blend_offsets(ours, partial, weight=1.0)
    assert blended["party"]["Democrat"] == pytest.approx(20.0)
    assert blended["party"]["Republican"] == pytest.approx(ours["party"]["Republican"])
