"""Does borrowing a level actually help?  The one place the question can be answered.

Pfänder publishes no human data, so an anchor applied to a Pfänder submission
cannot be checked against anything.  Voelkel can: it has real respondents, two
silicon samples of the same instrument, and the same additive decomposition.  So
the rehearsal is run there and the conclusion carried across.

The experiment has three steps and the third is the one that decides anything.

1. **Ceiling.**  Take each outcome's true control-arm level from Human 1,
   re-level the silicon sample onto it, and score the four distribution-shape
   metrics before and after.  This is the most any level anchoring can buy, since
   it uses the truth rather than a borrowed proxy.

2. **Exactness.**  Check the recomposition's drift audit is zero.  Re-levelling
   that perturbed the condition effects would be buying distribution metrics with
   leaderboard metrics, which is not a trade anyone agreed to.  It comes out at
   1e-14.

3. **Break-even.**  Repeat with the true level plus a deliberate error, and find
   the error at which the gain disappears.  That number is the specification an
   external anchor has to meet, and without it the ceiling in step 1 is just an
   encouraging figure with no decision attached.

## Why this does not call ``tier1.calibrate``

``calibration.tier1.calibrate`` is the production route for a Pfänder submission
and takes exactly the ``levels`` dict this package produces.  It cannot run on
Voelkel: it hard-wires the control arm's label to ``"control"`` (Voelkel's is
``"Null_Control"``) and looks every outcome's scale up in Pfänder's
``SCALE_RANGE``, which raises ``KeyError`` on ``PA``.  Monkeypatching two module
constants to borrow the wrapper would test the patch rather than the calibration,
so the rehearsal calls the same components ``tier1`` delegates to — ``decompose``,
``Decomposition``, ``recompose_frame`` — with Voelkel's control label.  The steps
``tier1`` adds on top are Pfänder format constraints with no Voelkel counterpart:
the twelve-item trust composite and the binary newsletter outcome.  What is being
validated here is the level substitution, which is entirely in the shared part.

## What it found

Re-levelling to the truth cuts mean W1 from 23.3 to 6.6 (Qwen2.5-7B) and 10.0 to
4.2 (DeepSeek-V4-Flash), and on that metric the gain survives an anchor error of
about 24 points for Qwen but only about 9 for DeepSeek — roughly each model's own
mean absolute level error, which is the intuitive answer and worth having measured
rather than assumed.  On KS, the strictest of the four, the break-evens are 12.8
and 4.7 points.  The variance ratio is the one metric anchoring does not reliably
help: for Qwen it moves from 1.13 to 0.85, further from 1 than it started.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..benchmark import metrics
from ..benchmark.reference import half_split
from ..calibration.components import Decomposition, decompose, recompose_frame
from ..voelkel import outcomes as voelkel_outcomes
from ..voelkel import paths as voelkel_paths
from ..voelkel import score as voelkel_score

CONTROL = voelkel_score.CONTROL
MODERATORS = voelkel_score.VISIBLE_MODERATORS + voelkel_score.INVISIBLE_MODERATORS
RUNS = ("qwen25_7b", "v4_flash")

#: Nominal anchor errors to probe, in outcome points.
ERRORS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

#: Each scored quantity a level anchor can move, and whether larger is better.
#: ``w1``/``ovl``/``ks`` are the response-distribution metrics; ``baseline_r`` and
#: ``baseline_rmse`` are the demographic-baseline pair over control cell means,
#: which the benchmark scores in raw outcome points.  The variance ratio is
#: reported but deliberately left out of the break-even: it has a two-sided target
#: of 1, so "worse" is not a direction.
SCORED_METRICS = {
    "w1": False,
    "ovl": True,
    "ks": False,
    "baseline_r": True,
    "baseline_rmse": False,
}


def load_run(run: str) -> pd.DataFrame:
    return pd.read_csv(voelkel_paths.samples_dir(run) / "samples.csv", low_memory=False)


def true_levels(human1: pd.DataFrame) -> dict[str, float]:
    """Each outcome's real control-arm mean, the target an anchor is trying to hit."""
    control = human1[human1["condition"] == CONTROL]
    return {
        outcome: float(control[outcome].mean()) for outcome in voelkel_outcomes.OUTCOMES
    }


def relevel(
    frame: pd.DataFrame, levels: dict[str, float], seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move the control-arm means onto ``levels``, holding every effect fixed.

    Outcomes absent from ``levels`` keep their own level, which makes a partial
    anchor — the realistic case, since only three Pfänder outcomes are anchorable
    — a valid input rather than an error.
    """
    parts = {}
    for outcome in voelkel_outcomes.OUTCOMES:
        part = decompose(frame, outcome, MODERATORS, CONTROL)
        parts[outcome] = Decomposition(
            outcome=outcome,
            level=levels.get(outcome, part.level),
            effects=part.effects,
            offsets=part.offsets,
            residuals=part.residuals,
        )
    bounds = {
        outcome: (0.0, scale) for outcome, scale in voelkel_outcomes.OUTCOMES.items()
    }
    return recompose_frame(frame, parts, CONTROL, bounds=bounds, seed=seed)


def shape_summary(human1: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """The four shape metrics averaged over condition x outcome cells."""
    table = voelkel_score.distribution_table(human1, synthetic)
    return {
        "cells": int(len(table)),
        "w1": float(table["w1"].mean()),
        "ovl": float(table["ovl"].mean()),
        "ks": float(table["ks"].mean()),
        "variance_ratio": float(table["variance_ratio"].mean()),
        "abs_mean_error": float(
            (table["mean_synthetic"] - table["mean_human"]).abs().mean()
        ),
    }


def baseline_summary(human1: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """The demographic-baseline pair and the parity gap, both driven by levels.

    The benchmark correlates human against predicted control-cell means across
    moderator levels and outcomes, reports the RMSE between them in raw outcome
    points, and reads the spread of per-group absolute error as a parity gap.  All
    three are functions of the level plus the demographic offsets, so a level
    anchor moves them even though it leaves every treatment effect alone — which is
    the whole argument for anchoring, and is worth measuring rather than asserting.
    """
    baselines = voelkel_score.baseline_means(human1, synthetic)
    if baselines.empty:
        return {
            "baseline_cells": 0,
            "baseline_r": float("nan"),
            "baseline_rmse": float("nan"),
            "dpd_mean": float("nan"),
        }
    gaps = voelkel_score.parity_gap(baselines)
    return {
        "baseline_cells": int(len(baselines)),
        "baseline_r": metrics.pearson(
            baselines["human_mean"], baselines["synthetic_mean"]
        ),
        "baseline_rmse": float(
            np.sqrt(
                np.mean((baselines["synthetic_mean"] - baselines["human_mean"]) ** 2)
            )
        ),
        "dpd_mean": float(gaps["dpd"].mean()) if len(gaps) else float("nan"),
    }


def level_error(frame: pd.DataFrame, levels: dict[str, float]) -> float:
    """Mean absolute distance between a frame's control levels and ``levels``."""
    control = frame[frame["condition"] == CONTROL]
    return float(
        np.mean(
            [
                abs(float(control[outcome].mean()) - level)
                for outcome, level in levels.items()
            ]
        )
    )


def ceiling(human1: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Before, self-recomposed, and anchored-to-truth, for both silicon samples.

    The self-recomposition row is not padding: it separates the reconstruction's
    own cost from the anchoring's benefit, and without it any change in the
    variance ratio could be attributed to either.
    """
    truth = true_levels(human1)
    rows = []
    for run in RUNS:
        sample = load_run(run)
        rows.append(
            {
                "run": run,
                "step": "raw sample",
                "level_error": level_error(sample, truth),
                "effect_drift": 0.0,
                **shape_summary(human1, sample),
                **baseline_summary(human1, sample),
            }
        )
        for label, levels in (("self-recomposed", {}), ("anchored to truth", truth)):
            rebuilt, drift = relevel(sample, levels, seed=seed)
            rows.append(
                {
                    "run": run,
                    "step": label,
                    "level_error": level_error(rebuilt, truth),
                    "effect_drift": float(drift["max_abs_effect_drift"].max()),
                    **shape_summary(human1, rebuilt),
                    **baseline_summary(human1, rebuilt),
                }
            )
    return pd.DataFrame(rows)


#: Sign patterns each error magnitude is applied under.  Two uniform and two
#: mixed, because the choice is not cosmetic: a uniform shift preserves the
#: *ordering* of the outcomes, and the demographic-baseline correlation is pooled
#: across outcomes and so mostly measures that ordering.  Degrading with a uniform
#: sign alone leaves ``baseline_r`` at 0.98 even 30 points from the truth, which
#: would read as an anchor that cannot be bad enough to hurt.  A real anchor's
#: errors do not share a sign, so mixed patterns are what that metric is judged on.
SIGN_PATTERNS = ("+", "-", "mixed:1", "mixed:2")


def _signs(pattern: str, outcomes: tuple[str, ...]) -> dict[str, int]:
    if pattern == "+":
        return {outcome: 1 for outcome in outcomes}
    if pattern == "-":
        return {outcome: -1 for outcome in outcomes}
    rng = np.random.default_rng(int(pattern.split(":")[1]))
    return {
        outcome: int(sign)
        for outcome, sign in zip(outcomes, rng.choice((-1, 1), size=len(outcomes)))
    }


def degradation_sweep(
    human1: pd.DataFrame, errors: tuple[float, ...] = ERRORS, seed: int = 0
) -> pd.DataFrame:
    """Anchor to the truth plus a deliberate error, under four sign patterns.

    The four are averaged at each magnitude.  Each shape metric is computed per
    condition x outcome cell and so depends only on that cell's own error, which
    makes the sign pattern nearly irrelevant to them — but not to the pooled
    demographic-baseline correlation, and not to clipping at the ends of the scale.
    The *realised* error is reported alongside the nominal one because an outcome
    whose true level is 10.8 cannot be moved 30 points down, and reporting the
    nominal figure there would overstate how much error the anchor survived.
    """
    truth = true_levels(human1)
    outcomes = tuple(truth)
    rows = []
    for run in RUNS:
        sample = load_run(run)
        rows.append(
            {
                "run": run,
                "anchor": "none (raw sample)",
                "pattern": "",
                "nominal_error": float("nan"),
                "realised_error": level_error(sample, truth),
                "effect_drift": 0.0,
                **shape_summary(human1, sample),
                **baseline_summary(human1, sample),
            }
        )
        for error in errors:
            for pattern in SIGN_PATTERNS if error else ("+",):
                signs = _signs(pattern, outcomes)
                levels = {
                    outcome: float(np.clip(level + signs[outcome] * error, 0.0, 100.0))
                    for outcome, level in truth.items()
                }
                rebuilt, drift = relevel(sample, levels, seed=seed)
                rows.append(
                    {
                        "run": run,
                        "anchor": f"truth {pattern}{error:g}",
                        "pattern": pattern,
                        "nominal_error": error,
                        "realised_error": float(
                            np.mean([abs(levels[o] - truth[o]) for o in truth])
                        ),
                        "effect_drift": float(drift["max_abs_effect_drift"].max()),
                        **shape_summary(human1, rebuilt),
                        **baseline_summary(human1, rebuilt),
                    }
                )
    return pd.DataFrame(rows)


def break_even(sweep: pd.DataFrame) -> pd.DataFrame:
    """The anchor error at which each metric stops preferring the anchor.

    Linear interpolation between the two probed magnitudes that bracket the raw
    sample's own score, averaging the sign patterns at each magnitude.  Reported
    per metric rather than pooled because they disagree by a factor of two or
    more, and the tightest of them is the one an anchor has to satisfy.  ``inf``
    means no crossing up to the largest error probed; ``nan`` means the metric was
    already worse at the true level than in the raw sample, so there is nothing to
    break even on.
    """
    rows = []
    for run, group in sweep.groupby("run"):
        raw = group[group["anchor"] == "none (raw sample)"].iloc[0]
        probed = (
            group[group["anchor"] != "none (raw sample)"]
            .groupby("nominal_error")[list(SCORED_METRICS) + ["realised_error"]]
            .mean()
            .sort_index()
        )
        for metric, higher_is_better in SCORED_METRICS.items():
            # Signed so that "worse than raw" is always positive.
            deficit = (
                raw[metric] - probed[metric]
                if higher_is_better
                else probed[metric] - raw[metric]
            )
            crossing = _first_crossing(probed.index.to_numpy(float), deficit.to_numpy())
            rows.append(
                {
                    "run": run,
                    "metric": metric,
                    "raw": float(raw[metric]),
                    "at_truth": float(probed[metric].iloc[0]),
                    "break_even_error": crossing,
                }
            )
    return pd.DataFrame(rows)


def _first_crossing(x: np.ndarray, deficit: np.ndarray) -> float:
    """Where ``deficit`` first turns positive, linearly interpolated."""
    for index in range(1, len(deficit)):
        if deficit[index] > 0 >= deficit[index - 1]:
            span = deficit[index] - deficit[index - 1]
            if span == 0:
                return float(x[index])
            share = -deficit[index - 1] / span
            return float(x[index - 1] + share * (x[index] - x[index - 1]))
    # ``inf`` rather than the largest probed error, which would read as a measured
    # crossing; ``nan`` rather than 0 when even the true level scores worse than
    # the raw sample, which is the variance ratio's situation and a real answer.
    return float("nan") if deficit[0] > 0 else float("inf")


def per_outcome(human1: pd.DataFrame, run: str, seed: int = 0) -> pd.DataFrame:
    """Per-outcome W1 and level error, before and after anchoring to the truth."""
    truth = true_levels(human1)
    sample = load_run(run)
    anchored, _ = relevel(sample, truth, seed=seed)
    before = voelkel_score.distribution_table(human1, sample)
    after = voelkel_score.distribution_table(human1, anchored)
    rows = []
    for outcome in voelkel_outcomes.OUTCOMES:
        control = sample[sample["condition"] == CONTROL][outcome]
        rows.append(
            {
                "outcome": outcome,
                "human_level": truth[outcome],
                "sample_level": float(control.mean()),
                "level_error": float(control.mean()) - truth[outcome],
                "w1_before": float(
                    before.loc[before["outcome"] == outcome, "w1"].mean()
                ),
                "w1_after": float(after.loc[after["outcome"] == outcome, "w1"].mean()),
                "ovl_before": float(
                    before.loc[before["outcome"] == outcome, "ovl"].mean()
                ),
                "ovl_after": float(
                    after.loc[after["outcome"] == outcome, "ovl"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    humans = voelkel_score.load_humans()
    human1, _ = half_split(humans)
    print(
        f"Human 1: n = {len(human1)}, control n = {(human1['condition'] == CONTROL).sum()}"
    )
    print("\n== true control levels ==")
    print(pd.Series(true_levels(human1)).round(2).to_string())
    print("\n== ceiling: raw vs self-recomposed vs anchored to truth ==")
    print(ceiling(human1).round(4).to_string(index=False))
    sweep = degradation_sweep(human1)
    print("\n== degraded anchors ==")
    print(sweep.round(4).to_string(index=False))
    print("\n== break-even anchor error ==")
    print(break_even(sweep).round(3).to_string(index=False))
    for run in RUNS:
        print(f"\n== per outcome, {run} ==")
        print(per_outcome(human1, run).round(3).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover - a report driver
    main()
