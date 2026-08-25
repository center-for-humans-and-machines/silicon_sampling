"""Rescaling how much a synthetic respondent's demographics move their answers.

Base models do not condition on who they have been told they are.  Measured on
the Pfänder sample, Qwen2.5-7B puts synthetic Republicans and Democrats
**1.1 points apart** on belief in human-caused climate change, where real US
survey data puts them tens of points apart, and the largest variance any
moderator explains beyond condition is R^2 = 0.002.  DeepSeek-V4-Flash fails the
other way: a 12.4-point gap and R^2 = 0.037, overshooting rather than flattening.

Neither is right, and both are wrong in a *structured* way — which is what makes
this correctable.  Measured on Voelkel by the functions in this module, over the
control arm's (moderator, level, outcome) cells:

| moderator | cells | r, Qwen | r, V4-Flash | sd_synth Qwen / V4 | sd_human |
| --- | --- | --- | --- | --- | --- |
| party | 27 | +0.117 | **+0.425** | 2.79 / 5.17 | 3.04 |
| race | 45 | -0.011 | +0.229 | 1.93 / 2.02 | 3.62 |
| gender | 18 | +0.325 | -0.233 | 0.36 / 0.47 | 1.25 |
| education | 36 | +0.002 | +0.053 | 0.96 / 1.09 | 2.40 |
| age | 36 | -0.169 | -0.023 | 0.56 / 0.88 | 3.88 |
| **pooled** | 162 | **+0.027** | **+0.190** | 1.60 / 2.44 | 3.21 |

(Correlations pooled over cells within a moderator.  An earlier pass averaged
per-outcome correlations instead and read higher for party — 0.330 and 0.502 —
which is a different estimand and not the one a scale should be fitted on.)

Three things in that table set the design.

**The signal is concentrated and differs by model.**  Party is V4-Flash's one
informative moderator at +0.425, and race is weakly informative at +0.229;
everything else is noise.  Qwen has almost nothing anywhere, its best being
gender on 18 cells.  So a *single* global rescale would amplify four moderators'
worth of noise to fix one moderator's worth of signal, and factors are fitted
**per moderator** instead.

**Uncorrelated moderators should be shrunk, not inflated.**  The error-minimising
scale for a predictor with correlation r is ``r * sd_human / sd_synth``, which for
r near zero is near zero — the formal version of "we know nothing about age, so
predict the grand mean".  Fitting to *variance* instead makes the sample look
realistically varied and scores worse: on Qwen it takes offset RMSE from 3.563 to
4.229, against 3.301 for the error-minimising fit.

**Both are under-dispersed relative to humans, but that is not the problem.**
Every ``sd_synth`` is below its ``sd_human``, so the naive read is "inflate
everything".  The measured outcome of doing that is worse on both models.  The
deficit is not too little variation, it is variation pointing the wrong way.

Measured effect of the error-minimising fit, with condition effects held exactly
fixed (drift < 1e-13):

| | offset r | offset RMSE |
| --- | --- | --- |
| Qwen, raw | +0.027 | 3.563 |
| Qwen, rescaled | **+0.105** | **3.301** |
| V4-Flash, raw | +0.190 | 3.698 |
| V4-Flash, rescaled | +0.043 | **3.450** |

Qwen improves on both counts.  V4-Flash trades correlation for error, because the
fit shrinks the two moderators that were genuinely carrying signal.  So this is a
per-model decision and not a universal improvement, which is exactly why it has
to clear held-out scoring before being adopted.

## What this can and cannot buy

It cannot move the leaderboard's sort key.  All four scored subgroup metrics
(directional, Spearman, Pearson, ``pearson_adj`` on the condition x moderator
interactions) are scale-invariant, so rescaling interaction terms changes none of
them — and the Voelkel finding that moderators the model *could see* predict its
subgroup effects no better than ones it could not (r 0.236 against 0.237) says
there is little signal there to rescue.

What it does reach: the **stereotyping coefficients**, the **demographic parity
gap**, the **demographic baselines** and the **within-subgroup response
distributions** — four reported analyses, all in raw outcome points, that no
effect-level calibration touches at all.  Those are graded on being close to the
human gaps, and closeness is exactly what a fitted scale delivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .components import MIN_CELL, cell_offsets


def offset_table(
    frame: pd.DataFrame,
    moderators: tuple[str, ...],
    outcomes: tuple[str, ...] | dict[str, float],
    control: str,
    min_n: int = MIN_CELL,
) -> pd.DataFrame:
    """Every (moderator, level, outcome) offset in the control arm, long form."""
    rows = []
    for moderator in moderators:
        if moderator not in frame.columns:
            continue
        for outcome in outcomes:
            for level, value in cell_offsets(
                frame, moderator, outcome, control, min_n
            ).items():
                rows.append(
                    {
                        "moderator": moderator,
                        "level": level,
                        "outcome": outcome,
                        "offset": float(value),
                    }
                )
    return pd.DataFrame(rows)


def compare_offsets(
    synthetic: pd.DataFrame,
    human: pd.DataFrame,
    moderators: tuple[str, ...],
    outcomes: tuple[str, ...] | dict[str, float],
    control: str,
    min_n: int = MIN_CELL,
) -> pd.DataFrame:
    """Our demographic offsets against the humans', cell by cell.

    Inner-joined, so a level either side dropped for thinness is dropped from the
    comparison rather than counted as a zero — treating an unmeasured group as a
    group with no effect would flatter us on exactly the cells we know least
    about.
    """
    mine = offset_table(synthetic, moderators, outcomes, control, min_n)
    theirs = offset_table(human, moderators, outcomes, control, min_n)
    return mine.merge(
        theirs,
        on=["moderator", "level", "outcome"],
        suffixes=("_synth", "_human"),
        how="inner",
    )


def offset_recovery(paired: pd.DataFrame, by_moderator: bool = False) -> pd.DataFrame:
    """How well our offsets track the human ones, pooled or per moderator.

    ``r`` is the quantity that decides whether a moderator's offsets are worth
    keeping at all; ``scale_rmse_optimal`` is what to multiply them by if they
    are.  A moderator with ``r`` near zero has an optimal scale near zero, which
    is the formal way of saying "we know nothing about this one, so predict the
    grand mean".
    """
    groups = paired.groupby("moderator") if by_moderator else [("pooled", paired)]
    rows = []
    for name, group in groups:
        mine = group["offset_synth"].to_numpy(float)
        theirs = group["offset_human"].to_numpy(float)
        if len(mine) < 3 or mine.std(ddof=1) == 0:
            rows.append({"moderator": name, "n_cells": len(mine), "r": float("nan")})
            continue
        correlation = float(np.corrcoef(mine, theirs)[0, 1])
        sd_synth = float(mine.std(ddof=1))
        sd_human = float(theirs.std(ddof=1))
        rows.append(
            {
                "moderator": name,
                "n_cells": len(mine),
                "r": correlation,
                "sd_synth": sd_synth,
                "sd_human": sd_human,
                "rmse": float(np.sqrt(np.mean((mine - theirs) ** 2))),
                # variance matching: makes our spread look right
                "scale_variance_match": (
                    sd_human / sd_synth if sd_synth else float("nan")
                ),
                # RMSE-optimal: shrinks by how much we actually know
                "scale_rmse_optimal": (
                    correlation * sd_human / sd_synth if sd_synth else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_offset_scales(
    paired: pd.DataFrame, objective: str = "rmse", floor: float = 0.0
) -> dict[str, float]:
    """Per-moderator factors to multiply our demographic offsets by.

    ``objective`` picks what the factor is for.  ``"rmse"`` gives the
    error-minimising scale ``r * sd_human / sd_synth``, which shrinks hard where we
    have no signal — the right choice when the metric grades closeness to the human
    gaps.  ``"variance"`` matches the human spread instead, which looks more
    realistic and scores better on anything reading dispersion, at the cost of
    amplifying noise where ``r`` is low.

    ``floor`` clips factors from below.  Left at zero, an uncorrelated moderator
    is flattened completely; a small positive floor keeps some variation so that
    metrics needing non-degenerate groups stay defined.
    """
    column = "scale_rmse_optimal" if objective == "rmse" else "scale_variance_match"
    table = offset_recovery(paired, by_moderator=True)
    scales = {}
    for row in table.itertuples():
        value = getattr(row, column)
        scales[row.moderator] = float(max(value, floor)) if np.isfinite(value) else 1.0
    return scales


def rescale_offsets(
    offsets: dict[str, pd.Series], scales: dict[str, float]
) -> dict[str, pd.Series]:
    """Scale each moderator's offsets, for feeding back through ``recompose``.

    A moderator absent from ``scales`` is left alone rather than zeroed, on the
    same principle as elsewhere: a missing factor is missing information, not a
    finding that the moderator does nothing.
    """
    return {
        moderator: table * scales.get(moderator, 1.0)
        for moderator, table in offsets.items()
    }


def blend_offsets(
    offsets: dict[str, pd.Series],
    anchor: dict[str, pd.Series],
    weight: float,
) -> dict[str, pd.Series]:
    """Move our offsets toward externally known ones.

    For Pfänder this is how ground truth about US demographic structure enters at
    all: the study publishes no human data, but the gaps between Republicans and
    Democrats on climate belief, worry and policy support are known from
    nationally representative survey data.  ``weight`` 1 adopts the anchor
    outright; levels the anchor does not cover keep our own value.
    """
    out = {}
    for moderator, table in offsets.items():
        target = anchor.get(moderator)
        if target is None:
            out[moderator] = table
            continue
        aligned = target.reindex(table.index)
        out[moderator] = table.where(
            aligned.isna(), weight * aligned + (1 - weight) * table
        )
    return out


def impose_gap(
    offsets: pd.Series,
    gap: float,
    high: str,
    low: str,
    shares: pd.Series | None = None,
) -> pd.Series:
    """Rebuild one moderator's offsets so *high* sits ``gap`` above *low*.

    **This replaces our own offsets rather than rescaling them, and that is the
    point.** Rescaling multiplies whatever signal is there; when the signal is
    near zero, as the models' party offsets are, the factor needed to reach a
    realistic gap is 20-50x and it amplifies noise far faster than signal.
    Measured against real Goldwert participants, rescaling the best model's party
    offsets to match the human spread cut RMSE only from 15.28 to 14.08 -- while
    substituting a single externally known constant cut it to 6.08.

    So the external number is used as the offsets, not as a target to stretch
    toward. Levels other than *high* and *low* are placed at zero, and the whole
    series is then centred on the population shares so it stays a set of
    deviations from the arm mean rather than shifting the level.

    ``gap`` is in the same units as ``offsets``.
    """
    out = pd.Series(0.0, index=offsets.index, dtype=float)
    if high not in out.index or low not in out.index:
        return offsets
    out[high] = gap / 2.0
    out[low] = -gap / 2.0
    if shares is None:
        weights = pd.Series(1.0, index=out.index)
    else:
        weights = shares.reindex(out.index).fillna(0.0)
    total = float(weights.sum())
    if total > 0:
        out = out - float((out * weights).sum() / total)
    return out


def party_offsets_from_gaps(
    frame: pd.DataFrame,
    gaps: dict[str, float],
    scales: dict[str, float],
    moderator: str = "party",
    high: str = "Democrat",
    low: str = "Republican",
) -> dict[str, dict[str, pd.Series]]:
    """Externally anchored party offsets, one entry per outcome in *gaps*.

    ``gaps`` are Democrat-minus-Republican differences in **pp of scale range**,
    which is how every external source in this project reports them; ``scales``
    converts each outcome back to its own units.
    """
    if moderator not in frame.columns:
        return {}
    shares = frame[moderator].value_counts()
    built: dict[str, dict[str, pd.Series]] = {}
    for outcome, gap in gaps.items():
        if outcome not in frame.columns or outcome not in scales:
            continue
        levels = pd.Series(0.0, index=shares.index, dtype=float)
        built[outcome] = {
            moderator: impose_gap(
                levels, gap / 100.0 * scales[outcome], high, low, shares
            )
        }
    return built
