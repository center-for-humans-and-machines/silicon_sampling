"""Decompose a silicon sample into parts, fix each part, put it back together.

Everything this project can do to a Tier-1 submission has to be expressed as a
change to respondent-level numbers, because that is the only thing the benchmark
reads: it refits ``lm(outcome ~ condition)`` itself and computes response
distributions and demographic tables from the rows we hand it.  So a calibration
is not a correction applied to an estimate, it is a reconstruction of the data
the estimate will be fitted to.

The decomposition used throughout is additive and, for our purposes, complete:

    y_ijc = level_j + effect_jc + offset_j(m_i) + residual_i

*level* is the control-arm mean, *effect* the shift a condition applies to it,
*offset* what a respondent's demographics add, and *residual* the individual
deviation left over.  The reason to write it out this way is that the benchmark's
scored analyses read **different terms**, and almost disjointly:

| term | what reads it |
| --- | --- |
| ``effect`` | ATE recovery, the calibration regression, RMSE — the leaderboard sort key |
| ``level`` | response distributions (OVL, KS, W1), demographic baselines |
| ``offset`` | stereotyping coefficients, parity gap, subgroup effects, baselines |
| ``residual`` | variance ratio, and the standard errors behind ``beta_adj`` |

Which means a term can be replaced without disturbing what reads the others —
and that turns out to matter a great deal, because our two samplers are good at
different terms.  Measured on Voelkel, the one study with real responses:
Qwen2.5-7B ranks interventions far better (pooled r 0.408 against 0.190) while
DeepSeek-V4-Flash sits three times closer on levels (8.3 pp against 23.7) and
carries demographic signal Qwen simply does not (cell-offset r 0.190 against
0.027).  Taking ``effect`` from one and ``level``/``offset`` from the other beats
both, on their own metrics, at no sampling cost.

## The exactness guarantee

:func:`recompose` adds the condition term *last* and centres everything else
within condition, so

    mean(y | condition c) == level + effect(c)

holds to floating point.  This is not a nicety.  Without it a reconstruction
perturbs the very effects it was supposed to leave alone: the first version of
this code sampled residuals without centring them, and pooled Pearson r fell
from 0.408 to 0.337 purely as reconstruction noise.  Any transform that claims to
change one term and preserve another has to be checked this way, and
:func:`recompose` is tested against exactly that.

## Preserving dispersion

The residual is drawn from the donor run's own within-cell spread, and *how* that
spread is measured decides whether the variance ratio survives.  Residuals taken
against fully interacted ``condition x every moderator`` cells are far too small:
that partition has thousands of cells, many holding one or two respondents, so
each deviation is measured against a mean it largely determines.  Recomposing a
run from itself that way returned a variance ratio of 0.758 — the reconstruction
invented under-dispersion, which is the single failure mode the benchmark's
Section 3 exists to catch.  So residuals are taken against the *additive*
prediction (condition plus offsets), which is the model actually being fitted, and
a self-recomposition then reproduces the source's dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Minimum cell size before a demographic group's mean is trusted.  The
#: benchmark uses the same threshold to decide which groups it reports at all, so
#: matching it keeps our offsets defined exactly where its metrics look.
MIN_CELL = 30


def cell_offsets(
    frame: pd.DataFrame,
    moderator: str,
    outcome: str,
    control: str,
    min_n: int = MIN_CELL,
) -> pd.Series:
    """What each level of one moderator adds to an outcome, in the control arm.

    Deviations from the control arm's grand mean rather than from a reference
    level, because these are used additively across several moderators at once and
    reference-level coding would count the reference's own shift many times.
    Groups thinner than ``min_n`` are dropped rather than estimated badly; callers
    treat a missing level as a zero offset.
    """
    rows = frame[frame["condition"] == control][[moderator, outcome]].dropna()
    if rows.empty:
        return pd.Series(dtype=float)
    sizes = rows.groupby(moderator)[outcome].size()
    means = rows.groupby(moderator)[outcome].mean()
    keep = sizes[sizes >= min_n].index
    return means.loc[keep] - rows[outcome].mean()


def condition_effects(frame: pd.DataFrame, outcome: str, control: str) -> pd.Series:
    """Each condition's shift away from the control mean, control itself at zero."""
    means = frame.groupby("condition")[outcome].mean()
    baseline = means.get(control)
    if baseline is None:
        raise KeyError(f"no {control!r} condition in this frame")
    return means - baseline


def control_level(frame: pd.DataFrame, outcome: str, control: str) -> float:
    """The control arm's mean — the point every effect is measured from."""
    return float(frame.loc[frame["condition"] == control, outcome].mean())


@dataclass
class Decomposition:
    """One run's terms for one outcome, ready to be swapped between runs."""

    outcome: str
    level: float
    effects: pd.Series
    offsets: dict[str, pd.Series]
    residuals: np.ndarray = field(repr=False)

    @property
    def residual_sd(self) -> float:
        return float(np.std(self.residuals, ddof=1)) if len(self.residuals) > 1 else 0.0


def decompose(
    frame: pd.DataFrame,
    outcome: str,
    moderators: tuple[str, ...],
    control: str,
    min_n: int = MIN_CELL,
) -> Decomposition:
    """Split one outcome into level, condition effects, demographic offsets, residuals.

    Residuals are taken against the additive fit — condition effect plus the sum of
    the row's demographic offsets — and not against fully interacted cells.  See
    the module docstring: interacted cells are so thin that the residuals measured
    against them are systematically too small, and a reconstruction built from them
    silently under-disperses.
    """
    level = control_level(frame, outcome, control)
    effects = condition_effects(frame, outcome, control)
    offsets = {
        moderator: cell_offsets(frame, moderator, outcome, control, min_n)
        for moderator in moderators
    }
    predicted = level + frame["condition"].map(effects).fillna(0.0).to_numpy(float)
    for moderator, table in offsets.items():
        predicted = predicted + frame[moderator].map(table).fillna(0.0).to_numpy(float)
    residuals = frame[outcome].to_numpy(float) - predicted
    return Decomposition(
        outcome=outcome,
        level=level,
        effects=effects,
        offsets=offsets,
        residuals=residuals[~np.isnan(residuals)],
    )


def recompose(
    template: pd.DataFrame,
    outcome: str,
    level: float,
    effects: pd.Series,
    offsets: dict[str, pd.Series],
    residuals: np.ndarray,
    control: str,
    bounds: tuple[float, float] | None = (0.0, 100.0),
    seed: int = 0,
    donors: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild one outcome so its condition means land exactly on ``level + effects``.

    ``template`` supplies the rows: their condition assignment and their
    demographics.  Everything that is not the condition term is centred within
    condition before the condition term is added, which is what makes the refit
    ATEs equal the targets rather than merely resemble them.

    ``bounds`` clips to the response scale — a rebuilt slider answer outside
    0-100 would be rejected by the submission's range check.  Clipping fights the
    centring, so it is done by :func:`_clip_preserving_means`, which iterates and
    hands each correction only to rows that are not already pinned at a boundary.
    A target the scale genuinely cannot hold stalls rather than diverging, and the
    leftover gap is reported by :func:`recompose_frame` so it cannot pass
    unnoticed.
    """
    rng = np.random.default_rng(seed)
    inner = np.zeros(len(template))
    for moderator, table in offsets.items():
        if moderator in template.columns and len(table):
            inner = inner + template[moderator].map(table).fillna(0.0).to_numpy(float)
    pool = np.asarray(residuals, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        pool = np.zeros(1)
    if donors is None:
        drawn = rng.choice(pool, size=len(template), replace=True)
    else:
        # Every outcome reads the same donor row, so one synthetic respondent
        # inherits one real respondent's whole residual vector.
        drawn = pool[np.mod(donors, len(pool))]
    inner = inner + drawn

    condition = template["condition"].to_numpy()
    inner = inner - _group_mean(condition, inner)
    target = level + template["condition"].map(effects).fillna(0.0).to_numpy(float)
    values = target + inner
    if bounds is not None:
        values = _clip_preserving_means(values, target, condition, bounds)
    return values


def _clip_preserving_means(
    values: np.ndarray,
    target: np.ndarray,
    condition: np.ndarray,
    bounds: tuple[float, float],
    rounds: int = 40,
) -> np.ndarray:
    """Clip to the response scale while holding each condition mean on target.

    Clipping and then re-centring does not work: shifting a group to repair its
    mean pushes the values that were just clipped back outside the scale, and
    clipping them again reopens the same gap.  The fix is to give the correction
    only to rows that can absorb it — those not already pinned at the boundary in
    the direction the group needs to move — and to iterate, since each pass pins a
    few more.

    This converges in a handful of rounds for any target the scale can actually
    hold.  For one it cannot, it stalls at the best achievable answer rather than
    diverging, and the leftover gap surfaces in :func:`recompose_frame`'s audit,
    which is where an unreachable target is supposed to become visible.
    """
    low, high = bounds
    values = np.clip(values, low, high)
    for _ in range(rounds):
        gap = target - _group_mean(condition, values)
        if np.abs(gap).max() < 1e-12:
            break
        # Only rows with headroom in the required direction can carry the shift.
        movable = np.where(gap > 0, values < high, values > low)
        share = _group_mean(condition, movable.astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(share > 0, gap / share, 0.0)
        values = np.clip(values + np.where(movable, step, 0.0), low, high)
    return values


def _shared_donors(
    parts: dict[str, Decomposition], n_rows: int, seed: int
) -> np.ndarray:
    """One donor row index per rebuilt respondent, reused across every outcome.

    Indices are drawn against the *shortest* residual pool so that the same index
    is valid for every outcome, which is what keeps a donor's residuals aligned
    across outcomes.  Pools differ in length only through per-outcome missingness,
    so in practice they are nearly equal and nothing is meaningfully truncated.
    """
    lengths = [len(part.residuals) for part in parts.values() if len(part.residuals)]
    if not lengths:
        return np.zeros(n_rows, dtype=int)
    return np.random.default_rng(seed).integers(0, min(lengths), size=n_rows)


def _group_mean(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each element's group mean, aligned to the input order."""
    frame = pd.DataFrame({"k": keys, "v": values})
    return frame.groupby("k")["v"].transform("mean").to_numpy(float)


def recompose_frame(
    template: pd.DataFrame,
    parts: dict[str, Decomposition],
    control: str,
    bounds: dict[str, tuple[float, float]] | None = None,
    seed: int = 0,
    couple_residuals: bool = True,
    resample_residuals: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild every outcome in ``parts`` onto ``template``'s rows.

    ``couple_residuals`` decides whether a synthetic respondent is one person or
    several, and it matters more than it looks.  Drawing each outcome's residual
    independently gives every rebuilt row a fresh personality per outcome, which
    destroys the cross-outcome correlation structure of whatever run the residuals
    came from.  That structure is a real and measurable property: on the Pfänder
    control arm, DeepSeek-V4-Flash's respondents correlate -0.270 between the
    twelve-item trust composite and distrust, and +0.637 between the two trust
    measures, while Qwen2.5-7B manages +0.000 and +0.307 — the same respondent
    reporting high trust *and* high distrust.  A hybrid meant to inherit
    V4-Flash's coherence gets none of it if the residuals are drawn per outcome.

    So with ``couple_residuals`` the donor rows are chosen **once** and every
    outcome reads the same donor, carrying that donor's whole residual vector
    across.  Set it to ``False`` only to reproduce the older independent-draw
    behaviour.

    ``resample_residuals`` should be true only when the residuals come from a
    *different* run than the template.  When they are the template's own, pairing
    each row with its own residual makes the reconstruction an exact inverse of the
    decomposition, and that matters most exactly where it is easiest to miss: on a
    saturated outcome.  ``belief_post`` has 36.5% of its answers at 100 and
    ``donation_ams`` 29% at zero, so a residual drawn from another row and added to
    this row's prediction clips asymmetrically at the bound, and the achievable
    mean falls short of the target.  That put 0.272 raw points of drift on
    ``belief_post`` — about three quarters of a shrunk effect — for a calibration
    that never intended to touch the residuals at all.

    Returns the rebuilt frame and a per-outcome audit of how far the realised
    condition means ended up from their targets.  The audit is the point: a
    calibration is only meaningful if it moved what it claimed to move, and the
    clip in :func:`recompose` is the one step that can quietly break that.
    """
    out = template.copy()
    drift = []
    if not resample_residuals:
        donors = np.arange(len(template))
    elif couple_residuals:
        donors = _shared_donors(parts, len(template), seed)
    else:
        donors = None
    for index, (outcome, part) in enumerate(parts.items()):
        limits = (bounds or {}).get(outcome, (0.0, 100.0))
        out[outcome] = recompose(
            template,
            outcome,
            part.level,
            part.effects,
            part.offsets,
            part.residuals,
            control=control,
            bounds=limits,
            seed=seed + index,
            donors=donors,
        )
        realised = condition_effects(out, outcome, control)
        shared = realised.index.intersection(part.effects.index)
        gap = (realised.loc[shared] - part.effects.loc[shared]).abs()
        drift.append(
            {
                "outcome": outcome,
                "max_abs_effect_drift": float(gap.max()) if len(gap) else float("nan"),
                "mean_abs_effect_drift": (
                    float(gap.mean()) if len(gap) else float("nan")
                ),
                "level_drift": abs(control_level(out, outcome, control) - part.level),
            }
        )
    return out, pd.DataFrame(drift)


def hybrid(
    runs: dict[str, pd.DataFrame],
    outcomes: dict[str, float],
    moderators: tuple[str, ...],
    control: str,
    effects_from: str,
    level_from: str | None = None,
    offsets_from: str | None = None,
    residuals_from: str | None = None,
    seed: int = 0,
    couple_residuals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take each term of the decomposition from whichever run predicts it best.

    Defaults fall back to ``effects_from``, so calling this with one run name is a
    self-recomposition — which is the test that the machinery is lossless, and is
    exercised as such in the test suite.

    The rows come from ``effects_from``: its condition assignment and its
    demographic composition define who the synthetic respondents are.
    """
    level_from = level_from or effects_from
    offsets_from = offsets_from or effects_from
    residuals_from = residuals_from or effects_from
    template = runs[effects_from]
    parts = {}
    for outcome in outcomes:
        effect_part = decompose(runs[effects_from], outcome, moderators, control)
        parts[outcome] = Decomposition(
            outcome=outcome,
            level=control_level(runs[level_from], outcome, control),
            effects=effect_part.effects,
            offsets=decompose(runs[offsets_from], outcome, moderators, control).offsets,
            residuals=decompose(
                runs[residuals_from], outcome, moderators, control
            ).residuals,
        )
    bounds = {outcome: (0.0, scale) for outcome, scale in outcomes.items()}
    return recompose_frame(
        template,
        parts,
        control,
        bounds=bounds,
        seed=seed,
        couple_residuals=couple_residuals,
        resample_residuals=residuals_from != effects_from,
    )
