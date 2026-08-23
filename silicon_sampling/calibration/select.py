"""Choosing a calibration when there are only two or three studies to choose on.

The honest constraint first, because it shapes everything else: after Voelkel,
ICPC and Goldwert there are **three** studies with both a silicon sample and real
participant responses.  Leave-one-study-out with three folds is not
cross-validation in any useful sense — it is a three-point transfer check, and no
mean across three folds has a standard error worth reporting.  This module is
built so that the weakness is visible in the output rather than hidden by it: it
reports per-fold results and a fold-count of wins, and it deliberately does not
compute a confidence interval across folds.

What three folds *can* establish is narrow and worth stating:

- whether one calibration **dominates** another in every fold;
- whether a fitted parameter is **stable** to within a factor of two across folds,
  which is the substitute for an interval that this fold count allows;
- the **sign** of an effect, when the folds agree on it.

What it cannot establish: a precise value for any parameter, or a ranking between
two calibrations whose held-out scores differ by less than the noise between
folds.

## The fold family has to preserve pooling

A trap worth naming, because it fails silently and looks like a result.  Most of
these candidates change pooled Pearson r *only* by re-weighting outcomes against
each other.  So a fold has to contain several outcomes for the transform to be
visible inside it: leave-one-condition-out gives each outcome a single
observation and makes within-outcome shrinkage the identity, while
leave-one-outcome-out gives the fold one outcome and makes it an affine map,
which Pearson r ignores.  Both were run, and both returned bit-identical
held-out scores for every candidate — which reads as "no calibration helps" and
actually means "this fold family cannot see any calibration".  Folds must be
whole studies, or groups of outcomes.

## The pre-commitment

Written down here so it constrains the code rather than being asserted afterwards:

1. **At most two free parameters, total.** A per-outcome parameter is rejected a
   priori — the *oracle* gain from per-outcome shrinkage factors was +0.003 pooled
   r, which is inside the rounding.
2. **K-of-K.** A calibration is adopted only if it beats the simpler alternative
   it nests inside, on the target metric, in *every* fold.
3. **Parameter stability instead of an interval.** Out-of-fold estimates must all
   fall within a factor of two of each other.
4. **No metric shopping.** The target metric is declared before fitting, and the
   others are reported as constraints, not as alternative ways to win.
5. **Nothing is tuned on Pfänder itself**, except the two knobs that need no
   ground truth: flattening outcomes whose true effect SD floors at zero, and
   cross-model agreement.

:func:`evaluate` scores every candidate on every fold, and :func:`decide` applies
the rules above.  Both refuse to collapse a fold table into a single number,
because that is the step where this fold count would start lying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

#: A candidate takes an effect table and returns a transformed one.
Transform = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class Candidate:
    """One calibration, with its cost in degrees of freedom made explicit."""

    name: str
    transform: Transform
    n_parameters: int = 0
    #: The candidate this one nests inside, and must beat to be adopted.
    nests_inside: str | None = None
    notes: str = ""


@dataclass
class Fold:
    """One held-out study: our effects and the human effects to score against."""

    name: str
    #: (outcome, condition, estimate, se) for our sample.
    predicted: pd.DataFrame
    #: (outcome, condition, estimate_h, se_h) for the human reference.
    reference: pd.DataFrame


def _pairs(predicted: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Join a transformed effect table to its human reference."""
    left = predicted.rename(columns={"estimate": "estimate_l", "se": "se_l"})
    merged = reference.merge(
        left[["outcome", "condition", "estimate_l", "se_l"]],
        on=["outcome", "condition"],
        how="inner",
    )
    return merged.dropna(subset=["estimate_h", "estimate_l"])


def score_pairs(pairs: pd.DataFrame) -> dict:
    """Every Section-1 number the benchmark computes, for one fold."""
    from ..benchmark import metrics as M

    if len(pairs) < 3:
        return {}
    return M.pooled_metrics(pairs) | M.run_calibration_pooled(pairs)


def evaluate(
    candidates: Sequence[Candidate],
    folds: Sequence[Fold],
    fit: Callable[[Sequence[Fold], Candidate], Transform] | None = None,
) -> pd.DataFrame:
    """Score every candidate on every held-out fold.

    ``fit`` receives the *training* folds — everything except the held-out one —
    and returns the transform to apply.  Passing it is what makes this
    out-of-sample: a candidate whose parameter is fitted on the fold it is scored
    on will look far better than it is.  Candidates with no free parameters can
    ignore it, and ``fit=None`` uses each candidate's own transform unchanged,
    which is only honest for zero-parameter candidates.
    """
    rows = []
    for candidate in candidates:
        for held_out in folds:
            training = [fold for fold in folds if fold.name != held_out.name]
            transform = (
                fit(training, candidate) if fit is not None else candidate.transform
            )
            scored = score_pairs(
                _pairs(transform(held_out.predicted), held_out.reference)
            )
            if not scored:
                continue
            rows.append(
                {
                    "candidate": candidate.name,
                    "fold": held_out.name,
                    "n_parameters": candidate.n_parameters,
                    "nests_inside": candidate.nests_inside,
                    **scored,
                }
            )
    return pd.DataFrame(rows)


def fold_wins(
    results: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    tie: float = 1e-9,
) -> pd.DataFrame:
    """Per candidate, how many folds it beat the candidate it nests inside on.

    The K-of-K rule reads this table.  A candidate with no ``nests_inside`` is a
    baseline and gets no win count — there is nothing it was required to beat.

    Ties are counted separately, and ``tie`` is not a cosmetic tolerance.  Some
    transforms are *provably* invariant on some metrics — a global rescale cannot
    change Pearson r — so their fold differences are floating-point dust.  Scoring
    that dust as wins and losses turns a mathematical identity into apparent
    evidence: without this the global shrinkage candidate reported "beat raw in
    1 of 3 folds", which is meaningless. A candidate that ties everywhere is
    neutral on this metric, which is a different verdict from failing on it.
    """
    out = []
    by_name = {
        (row.candidate, row.fold): getattr(row, metric)
        for row in results.itertuples()
        if pd.notna(getattr(row, metric, np.nan))
    }
    for name, group in results.groupby("candidate"):
        parent = group["nests_inside"].iloc[0]
        folds = sorted(group["fold"].unique())
        if not isinstance(parent, str):
            out.append(
                {
                    "candidate": name,
                    "nests_inside": None,
                    "n_folds": len(folds),
                    "wins": np.nan,
                    "ties": np.nan,
                    "k_of_k": np.nan,
                    "neutral": np.nan,
                }
            )
            continue
        wins = ties = comparable = 0
        for fold in folds:
            mine = by_name.get((name, fold))
            theirs = by_name.get((parent, fold))
            if mine is None or theirs is None:
                continue
            comparable += 1
            if abs(mine - theirs) <= tie:
                ties += 1
            elif (mine > theirs) if higher_is_better else (mine < theirs):
                wins += 1
        decisive = comparable - ties
        out.append(
            {
                "candidate": name,
                "nests_inside": parent,
                "n_folds": comparable,
                "wins": wins,
                "ties": ties,
                # A tie is not a loss, so K-of-K asks about the decisive folds.
                "k_of_k": decisive > 0 and wins == decisive,
                "neutral": comparable > 0 and ties == comparable,
            }
        )
    return pd.DataFrame(out)


def parameter_stability(estimates: Iterable[float], tolerance: float = 2.0) -> dict:
    """Whether out-of-fold parameter estimates agree to within a factor.

    Used in place of a confidence interval, which three folds cannot support.  A
    parameter whose fold estimates span more than ``tolerance``-fold is treated as
    unidentified, and the calibration that depends on it is capped at a
    conservative default rather than fitted.
    """
    values = np.asarray(
        [value for value in estimates if np.isfinite(value)], dtype=float
    )
    if len(values) < 2:
        return {"n": len(values), "stable": False, "ratio": float("nan")}
    positive = values[np.abs(values) > 0]
    if len(positive) < 2 or np.ptp(np.sign(positive)) != 0:
        return {
            "n": len(values),
            "stable": False,
            "ratio": float("inf"),
            "min": float(values.min()),
            "max": float(values.max()),
            "note": "estimates change sign across folds",
        }
    ratio = float(np.abs(positive).max() / np.abs(positive).min())
    return {
        "n": len(values),
        "stable": ratio <= tolerance,
        "ratio": ratio,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def decide(
    results: pd.DataFrame,
    metric: str = "pearson_r",
    higher_is_better: bool = True,
    max_parameters: int = 2,
) -> pd.DataFrame:
    """Apply the pre-commitment and report a verdict per candidate.

    The verdict column is the decision; ``reason`` says which rule decided it.
    Deliberately returns a table rather than a single winner, because with this
    many folds the difference between the top two candidates is usually not
    something the data can resolve — and that should be visible to whoever reads
    it, not hidden behind an ``argmax``.
    """
    wins = fold_wins(results, metric, higher_is_better).set_index("candidate")
    summary = (
        results.groupby("candidate")
        .agg(
            n_parameters=("n_parameters", "first"),
            n_folds=("fold", "nunique"),
            metric_mean=(metric, "mean"),
            metric_min=(metric, "min"),
            metric_max=(metric, "max"),
        )
        .join(wins[["nests_inside", "wins", "ties", "k_of_k", "neutral"]])
    )

    verdicts, reasons = [], []
    for name, row in summary.iterrows():
        if row["n_parameters"] > max_parameters:
            verdicts.append("rejected")
            reasons.append(
                f"{int(row['n_parameters'])} parameters exceeds the ceiling of {max_parameters}"
            )
        elif not isinstance(row["nests_inside"], str):
            verdicts.append("baseline")
            reasons.append("nothing it was required to beat")
        elif row["neutral"] is True:
            verdicts.append("neutral")
            reasons.append(
                f"identical to {row['nests_inside']} on {metric} in every fold; "
                "judge it on the metrics it does move"
            )
        elif row["k_of_k"] is True:
            verdicts.append("adopt")
            reasons.append(
                f"beat {row['nests_inside']} on {metric} in {int(row['wins'])}/"
                f"{int(row['n_folds']) - int(row['ties'])} decisive folds"
            )
        else:
            verdicts.append("hold")
            reasons.append(
                f"beat {row['nests_inside']} in only {int(row['wins'])}/"
                f"{int(row['n_folds']) - int(row['ties'])} decisive folds; "
                "adopt the direction at a conservative default, not the fitted magnitude"
            )
    summary["verdict"] = verdicts
    summary["reason"] = reasons
    return summary.reset_index().sort_values(
        ["verdict", "metric_mean"], ascending=[True, not higher_is_better]
    )
