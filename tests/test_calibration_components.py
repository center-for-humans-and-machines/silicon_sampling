"""The decomposition has to be lossless before it can be trusted to be useful.

Two properties carry the whole design and are easy to break silently:

*Exactness* — a rebuilt sample's refit condition means must land on the targets,
or a calibration aimed at one term perturbs the others.  An earlier version lost
0.07 of pooled Pearson r to nothing but reconstruction noise.

*Dispersion* — recomposing a run from its own parts must return the run's own
spread.  Measuring residuals against thin interacted cells made a
self-recomposition under-disperse by a quarter, which is precisely the pathology
the benchmark's variance ratio is designed to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.calibration import components as C

CONTROL = "control"
MODERATORS = ("party", "gender")
OUTCOMES = {"trust": 100.0, "belief": 100.0}


def synthetic_run(seed: int, n: int = 3000, party_effect: float = 12.0) -> pd.DataFrame:
    """A sample with known level, condition and demographic structure."""
    rng = np.random.default_rng(seed)
    conditions = rng.choice([CONTROL, "arm_a", "arm_b"], size=n)
    frame = pd.DataFrame(
        {
            "condition": conditions,
            "party": rng.choice(["Republican", "Democrat"], size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
        }
    )
    shift = pd.Series({CONTROL: 0.0, "arm_a": 4.0, "arm_b": -3.0})
    for outcome in OUTCOMES:
        frame[outcome] = (
            50.0
            + frame["condition"].map(shift)
            + np.where(frame["party"] == "Democrat", party_effect, -party_effect)
            + rng.normal(0, 15, n)
        ).clip(0, 100)
    return frame


def test_decompose_recovers_the_structure_it_was_built_from():
    frame = synthetic_run(1)
    part = C.decompose(frame, "trust", MODERATORS, CONTROL)
    assert part.effects[CONTROL] == pytest.approx(0.0, abs=1e-9)
    assert part.effects["arm_a"] == pytest.approx(4.0, abs=1.5)
    assert part.effects["arm_b"] == pytest.approx(-3.0, abs=1.5)
    gap = part.offsets["party"]["Democrat"] - part.offsets["party"]["Republican"]
    assert gap == pytest.approx(24.0, abs=2.0)
    assert part.offsets["gender"].abs().max() < 2.0


def test_recompose_puts_condition_means_exactly_on_target():
    """The exactness guarantee, stated as a test rather than as a hope."""
    frame = synthetic_run(2)
    part = C.decompose(frame, "trust", MODERATORS, CONTROL)
    target = pd.Series({CONTROL: 0.0, "arm_a": 9.0, "arm_b": -6.0})
    rebuilt = frame.copy()
    rebuilt["trust"] = C.recompose(
        frame,
        "trust",
        level=part.level,
        effects=target,
        offsets=part.offsets,
        residuals=part.residuals,
        control=CONTROL,
        bounds=None,
    )
    realised = C.condition_effects(rebuilt, "trust", CONTROL)
    for arm, wanted in target.items():
        assert realised[arm] == pytest.approx(wanted, abs=1e-9)
    assert C.control_level(rebuilt, "trust", CONTROL) == pytest.approx(
        part.level, abs=1e-9
    )


def test_self_recomposition_preserves_effects_levels_and_dispersion():
    """Rebuilding a run from its own parts must change nothing that is scored."""
    frame = synthetic_run(3)
    runs = {"only": frame}
    rebuilt, drift = C.hybrid(runs, OUTCOMES, MODERATORS, CONTROL, effects_from="only")

    assert (drift["max_abs_effect_drift"] < 1e-6).all()
    for outcome in OUTCOMES:
        source = frame[frame.condition == CONTROL][outcome]
        copy = rebuilt[rebuilt.condition == CONTROL][outcome]
        assert copy.mean() == pytest.approx(source.mean(), abs=1e-6)
        # Dispersion is the property the thin-cell bug destroyed.
        assert copy.var(ddof=1) / source.var(ddof=1) == pytest.approx(1.0, abs=0.12)


def test_hybrid_takes_each_term_from_the_run_it_was_asked_to():
    """Effects from one run, level and demographic offsets from another."""
    ranker = synthetic_run(4, party_effect=1.0)  # good effects, flat demographics
    ranker["trust"] = ranker["trust"] - 20.0  # and a badly wrong level
    ranker["belief"] = ranker["belief"] - 20.0
    grounded = synthetic_run(5, party_effect=18.0)  # right level, strong demographics

    runs = {"ranker": ranker, "grounded": grounded}
    mixed, drift = C.hybrid(
        runs,
        OUTCOMES,
        MODERATORS,
        CONTROL,
        effects_from="ranker",
        level_from="grounded",
        offsets_from="grounded",
        residuals_from="grounded",
    )

    assert (drift["max_abs_effect_drift"] < 1e-6).all()

    # the effects are the ranker's
    wanted = C.condition_effects(ranker, "trust", CONTROL)
    got = C.condition_effects(mixed, "trust", CONTROL)
    for arm in wanted.index:
        assert got[arm] == pytest.approx(wanted[arm], abs=1e-6)

    # the level is the grounded run's, not the ranker's 20-point-low one
    assert C.control_level(mixed, "trust", CONTROL) == pytest.approx(
        C.control_level(grounded, "trust", CONTROL), abs=1e-6
    )

    # and the party gap is the grounded run's, not the ranker's flat one
    gap = C.cell_offsets(mixed, "party", "trust", CONTROL)
    assert gap["Democrat"] - gap["Republican"] > 25.0


def test_thin_groups_are_dropped_rather_than_estimated_badly():
    frame = synthetic_run(6, n=400)
    frame.loc[frame.index[:5], "party"] = "Libertarian"
    offsets = C.cell_offsets(frame, "party", "trust", CONTROL, min_n=30)
    assert "Libertarian" not in offsets.index
    assert {"Republican", "Democrat"} <= set(offsets.index)


def test_clipping_keeps_values_in_range_and_reports_any_drift_it_causes():
    """A target that cannot fit on the scale must be visible, not silently absorbed."""
    frame = synthetic_run(7)
    part = C.decompose(frame, "trust", MODERATORS, CONTROL)
    impossible = pd.Series({CONTROL: 0.0, "arm_a": 60.0, "arm_b": -60.0})
    parts = {
        "trust": C.Decomposition(
            outcome="trust",
            level=part.level,
            effects=impossible,
            offsets=part.offsets,
            residuals=part.residuals,
        )
    }
    rebuilt, drift = C.recompose_frame(
        frame, parts, CONTROL, bounds={"trust": (0.0, 100.0)}
    )
    assert rebuilt["trust"].between(0, 100).all()
    assert drift["max_abs_effect_drift"].iloc[0] > 1.0
