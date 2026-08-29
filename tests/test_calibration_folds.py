"""The reference studies, described the way the benchmark scorer needs them.

These are the guards on the seam that the four-study cross-validation used to
cross without noticing: a fold is only informative about Pfänder if both sides of
it are measured the same way, and four separate defects lived in the gap between
"the humans' frame" and "the silicon sample's frame".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.benchmark import scored as SC
from silicon_sampling.calibration import folds as F
from silicon_sampling.calibration import tier1 as T1


def test_every_fold_describes_a_complete_scored_grid():
    """Each study's design must derive its own pair count and carry a control."""
    for study in F.load_folds():
        design = study.design
        assert design.control in set(design.conditions)
        assert design.expected_pairs == len(design.outcomes) * (
            len(design.conditions) - 1
        )
        assert len(design.moderators) >= 2
        # The instrument and the scored design have to describe one study, or a
        # recipe is applied under one set of facts and graded under another.
        assert set(study.instrument.scales) == set(design.outcomes)
        assert study.instrument.control == design.control
        assert set(study.instrument.binary) == set(design.binary)


def test_voelkels_composite_is_out_of_the_scored_grid():
    """It is the exact mean of the other eight, so scoring it double-counts them."""
    study = F.BUILDERS["Voelkel"]()
    assert "Composite" in study.dropped
    assert "Composite" not in study.design.outcomes
    human = study.load_humans()
    components = [c for c in study.design.outcomes]
    rebuilt = human[components].mean(axis=1)
    assert float((human["Composite"] - rebuilt).abs().max()) == pytest.approx(0.0)


def test_a_composite_of_scored_outcomes_is_refused_rather_than_calibrated():
    """The trap that dropped the Voelkel fold from 0.576 to 0.132.

    ``calibrate`` hands every item of a composite the *composite's* effect
    vector.  That is right for Pfänder, whose twelve trust items exist only to
    carry ``trust_multidimensional`` and are never scored, and catastrophic when
    the items are themselves the outcomes: their own targets are replaced and
    nothing says so.
    """
    frame = pd.DataFrame(
        {
            "condition": ["control"] * 40 + ["treat"] * 40,
            "a": np.linspace(10, 60, 80),
            "b": np.linspace(20, 70, 80),
        }
    )
    frame["whole"] = frame[["a", "b"]].mean(axis=1)
    instrument = T1.Instrument(
        scales={"a": 100.0, "b": 100.0},
        control="control",
        composites={"whole": ("a", "b")},
    )
    with pytest.raises(ValueError, match="themselves scored outcomes"):
        T1.calibrate(frame, instrument=instrument)


def test_a_binary_outcome_keeps_its_holes_instead_of_inventing_answers():
    """``astype(int)`` on a NaN is -2**63, which then reads as a real answer."""
    values = [1.0, 0.0, 1.0, 0.0, np.nan] * 8
    frame = pd.DataFrame(
        {"condition": ["control"] * 20 + ["treat"] * 20, "signed": values}
    )
    instrument = T1.Instrument(
        scales={"signed": 1.0}, control="control", binary=("signed",)
    )
    rebuilt, _ = T1.calibrate(frame, instrument=instrument)
    got = rebuilt["signed"]
    assert got.isna().sum() == frame["signed"].isna().sum()
    present = pd.to_numeric(got.dropna())
    assert set(present.unique()) <= {0, 1}
    assert present.min() >= 0


def test_the_binary_rate_is_computed_over_the_rows_that_have_an_answer():
    """Counting missing rows in the denominator drives the rate off target."""
    rng = np.random.default_rng(0)
    n = 400
    answered = rng.random(n) < 0.5
    values = np.where(answered, (rng.random(n) < 0.4).astype(float), np.nan)
    frame = pd.DataFrame(
        {"condition": ["control"] * (n // 2) + ["treat"] * (n // 2), "y": values}
    )
    # Ask for a +20pp effect on the treated arm.
    targets = pd.Series({"control": 0.0, "treat": 0.20})
    moved = T1.calibrate_binary(frame, "y", targets, seed=1, control="control")
    treated = moved[frame["condition"].to_numpy() == "treat"]
    control = moved[frame["condition"].to_numpy() == "control"]
    gap = np.nanmean(treated) - np.nanmean(control)
    assert gap == pytest.approx(0.20, abs=0.02)


def test_ccc_donation_is_put_back_on_the_budget_the_survey_enforced():
    from silicon_sampling.ccc import score as CS

    boxes = {f"Donation_{i}": [0.0] * 3 for i in range(1, 7)}
    frame = pd.DataFrame({"condition": ["Control"] * 3, **boxes})
    # A respondent who obeys the budget, one who spends five times it, and one
    # who allocates nothing at all.
    frame.loc[0, [f"Donation_{i}" for i in range(1, 7)]] = [20, 20, 20, 20, 10, 10]
    frame.loc[1, [f"Donation_{i}" for i in range(1, 7)]] = [100, 100, 100, 100, 50, 50]
    frame.loc[2, [f"Donation_{i}" for i in range(1, 7)]] = [0, 0, 0, 0, 0, 0]
    frame["Donation"] = frame[[f"Donation_{i}" for i in range(1, 6)]].sum(axis=1)

    fixed = CS.normalise_donation(frame)
    # The two allocating respondents make the same relative choice, so they must
    # land on the same repaired value; the empty one has nothing to preserve.
    assert fixed.loc[0, "Donation"] == pytest.approx(90.0)
    assert fixed.loc[1, "Donation"] == pytest.approx(90.0)
    assert np.isnan(fixed.loc[2, "Donation"])
    assert float(pd.to_numeric(fixed["Donation"]).max()) <= 100.0 + 1e-9


def test_ccc_moderator_labels_are_one_vocabulary_on_both_sides():
    from silicon_sampling.ccc import score as CS

    synthetic = pd.DataFrame(
        {
            "education": [
                "Bachelor's degree",
                "Some college or Associate's degree",
                "High school diploma / GED",
                "Doctorate degree / Ph.D.",
            ],
            "party": ["Democrat", "Republican", "Independent", "Other"],
        }
    )
    got = CS.harmonise_moderators(synthetic)
    assert set(got["education"]) <= {
        "Bachelor or Postgraduate",
        "Some college",
        "HS or less",
    }
    assert set(got["party"]) <= {"Democrat", "Republican", "Neither"}
    # Democrats and Republicans must survive untouched: the party gap is their
    # contrast, and folding either one would move every gap in the project.
    assert got["party"].tolist()[:2] == ["Democrat", "Republican"]


def test_a_stringified_hole_is_not_a_demographic_group():
    """ICPC carried 44 interaction estimates for a group called ``"nan"``."""
    values = pd.Series(["Male", "Female", "nan", "None", "<NA>", ""])
    got = SC._labels(values)
    assert got.notna().sum() == 2
    assert set(got.dropna()) == {"Male", "Female"}


def test_the_subgroup_grid_joins_on_every_fold():
    """A grid that cannot be joined is a test set that silently shrank."""
    from silicon_sampling import models as MODELS
    from silicon_sampling.benchmark.reference import half_split

    for study in F.load_folds():
        human = study.prepare(study.load_humans())
        human1, _ = half_split(human)
        key = MODELS.resolve_run(study.samples_dir, "qwen25_7b_v3")
        if key is None:  # pragma: no cover - every study has this run today
            continue
        sample = study.prepare(
            pd.read_csv(study.samples_dir(key) / "samples.csv", low_memory=False)
        )
        design = study.design
        left = set(human1.columns)
        for moderator in design.moderators:
            assert moderator in left
            human_levels = set(SC._labels(human1[moderator]).dropna().unique())
            model_levels = set(SC._labels(sample[moderator]).dropna().unique())
            missing = human_levels - model_levels
            assert not missing, f"{study.name}/{moderator}: {sorted(missing)}"
