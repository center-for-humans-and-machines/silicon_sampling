"""Transforms on the fitted effects, and what each one can and cannot buy.

These act on a table of ATEs — one row per (outcome, condition) — and are the
half of calibration that targets the benchmark's Section 1.  Applying them to a
Tier-1 submission means feeding the transformed effects back through
:func:`~silicon_sampling.calibration.components.recompose`, which is what makes
the refit ATEs equal the targets.

The reason to keep them separate and named is that the benchmark's metrics
respond to them very differently, and the differences are not intuitive.  All of
the following is measured on the Voelkel pairs, not argued:

**A single global rescale ``d -> k*d`` cannot move six of the ten Section-1
numbers.**  ``directional_pct``, ``spearman_rho``, ``pearson_r``,
``pearson_within``, ``pearson_adj`` and ``alpha`` come out bit-identical at
k = 0.5, 0.159 and 0.05.  It moves ``beta`` (0.159 -> 1.001 at k = 0.159),
``beta_adj``, ``rmse`` (3.620 -> 1.402) and ``rmse_adj``.  So shrinkage is
mandatory hygiene worth a large slice of RMSE, and provably worth zero on the
leaderboard's sort key.  Note that ``pearson_adj`` survives shrinkage because it
is corrected with the *reference's* standard errors, not ours — a detail worth
stating because getting it wrong makes shrinkage look far more dangerous than it
is.

**Shrinking only the within-outcome deviations does move ``pearson_r``.**
Because it re-weights the pooled correlation toward the per-outcome profile,
which we predict better than we predict the message ranking: Qwen goes 0.408 ->
0.439 and Spearman 0.311 -> 0.402 at ``factor`` 0.5.  It is not free of risk — on
V4-Flash the same factor nudges r up but pulls Spearman *down* (0.186 -> 0.135) —
so the factor is fit per model rather than adopted as a constant.

**Replacing the per-outcome profile is the largest single lever, and has a
break-even.**  Substituting the true per-outcome mean effect takes Qwen to 0.491
and V4-Flash to 0.465.  But a *transferred* profile is not the truth, and the
crossover is high: an anchor correlated 0.6 with the truth scores 0.392, below
the 0.408 of leaving our own profile alone.  The anchor has to reach roughly 0.7
before it pays.  :func:`substitute_profile` therefore takes an explicit weight,
and the weight is something to measure rather than assume.

The structural reason all of this works the way it does: on the Voelkel human
effects, **56.6% of the variance across (outcome, condition) pairs is
between-outcome**, and an oracle predicting only the per-outcome mean human
effect — with no information at all about which message works — scores a pooled r
of 0.752, against a fresh human replication's 0.514.  Most of what the pooled
correlation rewards is knowing which outcomes move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The columns an effect table is expected to carry.
EFFECT_KEYS = ("outcome", "condition")


def _require(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise KeyError(
            f"effect table needs a {column!r} column; has {list(frame.columns)}"
        )


def global_shrink(
    effects: pd.DataFrame, factor: float, column: str = "estimate"
) -> pd.DataFrame:
    """Scale every effect by one number.

    Fixes ``beta`` and ``rmse`` and touches nothing that is scale-invariant.  The
    RMSE-optimal factor is :func:`optimal_shrinkage` fit on a study with real
    responses; on Voelkel it came out at 0.11-0.18 across leave-one-out folds,
    stable enough to transfer as a single parameter.
    """
    out = effects.copy()
    out[column] = out[column] * factor
    if "se" in out.columns:
        out["se"] = out["se"] * factor
    return out


def shrink_within_outcome(
    effects: pd.DataFrame,
    factor: float,
    column: str = "estimate",
    reference: str | None = None,
) -> pd.DataFrame:
    """Shrink each effect toward its own outcome's mean, leaving the profile alone.

    This is the one shrinkage that moves the leaderboard's sort key, because it
    shifts weight from the component we predict badly (which message) to the one
    we predict better (which outcome).  Fit ``factor`` per model — the direction
    held in 6 of 6 leave-one-condition-out folds on Voelkel but the magnitude in
    only 3 of 6, so a conservative 0.5 is preferred to an optimised value.

    ``reference`` names the control condition, if the table carries a row for it.
    That row is not an effect — it is identically zero by construction — so it
    must be excluded from the outcome mean and left alone.  Including it gave the
    control arm a spurious effect of +1.71 pp on ``belief_post``, which then moved
    the control *level* of the rebuilt file: a transform meant to touch only the
    message ranking silently relocated the baseline every effect is measured from.
    A table with no control row (the shape :func:`ate_pairs` produces) is
    unaffected either way.
    """
    _require(effects, "outcome")
    out = effects.copy()
    movable = _movable(out, reference)
    means = (
        out.loc[movable]
        .groupby("outcome")[column]
        .mean()
        .reindex(out["outcome"])
        .to_numpy()
    )
    shrunk = np.where(movable, means + factor * (out[column].to_numpy() - means), 0.0)
    out[column] = np.where(np.isnan(shrunk), out[column], shrunk)
    if "se" in out.columns:
        out.loc[movable, "se"] = out.loc[movable, "se"] * factor
    return out


def _movable(effects: pd.DataFrame, reference: str | None) -> np.ndarray:
    """Rows that are genuine effects, i.e. everything but the reference condition."""
    if reference is None or "condition" not in effects.columns:
        return np.ones(len(effects), dtype=bool)
    return (effects["condition"] != reference).to_numpy()


def substitute_profile(
    effects: pd.DataFrame,
    anchor: pd.Series | pd.DataFrame,
    weight: float,
    column: str = "estimate",
    reference: str | None = None,
) -> pd.DataFrame:
    """Blend our per-outcome mean effect toward an externally supplied one.

    ``anchor`` maps outcome -> mean effect, in the same units as ``column``
    (percentage points of scale range).  ``weight`` 0 leaves us untouched, 1
    adopts the anchor's profile entirely; our within-outcome deviations — the
    message ranking — survive either way.

    Outcomes absent from ``anchor`` keep their own profile rather than being
    shrunk toward zero, because a missing anchor is missing information, not
    evidence of a null effect.
    """
    _require(effects, "outcome")
    if isinstance(anchor, pd.DataFrame):
        anchor = anchor.set_index("outcome")[column]
    out = effects.copy()
    movable = _movable(out, reference)
    ours = (
        out.loc[movable]
        .groupby("outcome")[column]
        .mean()
        .reindex(out["outcome"])
        .to_numpy()
    )
    theirs = out["outcome"].map(anchor).to_numpy(dtype=float)
    blended = np.where(np.isnan(theirs), ours, weight * theirs + (1 - weight) * ours)
    moved = blended + (out[column].to_numpy() - ours)
    out[column] = np.where(movable & ~np.isnan(moved), moved, out[column])
    out.loc[~movable, column] = 0.0
    return out


def flatten_outcomes(
    effects: pd.DataFrame,
    outcomes: set[str] | tuple[str, ...],
    factor: float = 0.2,
    column: str = "estimate",
    reference: str | None = None,
) -> pd.DataFrame:
    """Shrink named outcomes' *within-outcome* spread hard, keeping their profile.

    For use on outcomes whose observed between-message spread is entirely
    sampling noise.  On the Pfänder sample ``belief_post`` and ``donation_ams``
    both have a noise-corrected true effect SD of 0.000, so their apparent
    message ranking is variance with no covariance behind it; removing most of it
    raises pooled r by shrinking ``sd(l)`` without touching ``cov(h, l)``.

    This needs no ground truth — the diagnostic comes from our own sample — which
    is why it is safe to apply to Pfänder, where no human data exists.
    """
    _require(effects, "outcome")
    picked = effects["outcome"].isin(set(outcomes))
    if not picked.any():
        return effects.copy()
    shrunk = shrink_within_outcome(
        effects.loc[picked], factor, column=column, reference=reference
    )
    out = effects.copy()
    out.loc[picked, column] = shrunk[column]
    if "se" in out.columns:
        out.loc[picked, "se"] = shrunk["se"]
    return out


def true_effect_sd(effects: pd.DataFrame, column: str = "estimate") -> pd.Series:
    """Per-outcome effect SD with sampling noise removed, as a flatten diagnostic.

    ``sqrt(var(observed) - mean(se^2))``, floored at zero.  An outcome that floors
    is one whose between-message differences are indistinguishable from noise at
    this sample size, and therefore a candidate for :func:`flatten_outcomes`.
    """
    _require(effects, "outcome")
    if "se" not in effects.columns:
        raise KeyError("true_effect_sd needs a 'se' column")

    def one(group: pd.DataFrame) -> float:
        observed = group[column].var(ddof=1)
        noise = float(np.mean(group["se"].to_numpy(float) ** 2))
        return float(np.sqrt(max(observed - noise, 0.0)))

    return effects.groupby("outcome")[[column, "se"]].apply(one)


def optimal_shrinkage(
    predicted: np.ndarray | pd.Series, reference: np.ndarray | pd.Series
) -> float:
    """The factor minimising RMSE against a reference: ``sum(hl) / sum(ll)``.

    A regression through the origin, because RMSE is measured against zero rather
    than against a fitted intercept.  This is the factor to use when RMSE is the
    target.

    It is **not** the same as :func:`slope_matching_factor`, which drives the
    benchmark's ``beta`` to 1 — that one regresses *with* an intercept.  The two
    coincide only when the intercept is zero, which on the Voelkel pairs it very
    nearly is (alpha = 0.014, and the two factors agree to 0.3%), so the
    distinction is invisible on real data and glaring on synthetic data with a
    shifted mean.  Keeping them separate is the only way that difference does not
    turn into a silent bug later.
    """
    lo = np.asarray(predicted, dtype=float)
    hi = np.asarray(reference, dtype=float)
    keep = np.isfinite(lo) & np.isfinite(hi)
    denominator = float(np.sum(lo[keep] ** 2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(hi[keep] * lo[keep]) / denominator)


def slope_matching_factor(
    predicted: np.ndarray | pd.Series, reference: np.ndarray | pd.Series
) -> float:
    """The factor that drives the benchmark's calibration ``beta`` to exactly 1.

    The benchmark fits ``h = alpha + beta * l`` with an intercept, so its slope is
    ``cov(h, l) / var(l)``.  Scaling our effects by ``k`` divides that slope by
    ``k``, which means setting ``k`` to the currently-fitted slope lands ``beta``
    on 1 exactly.  Use this when ``beta`` is the target and
    :func:`optimal_shrinkage` when RMSE is.
    """
    lo = np.asarray(predicted, dtype=float)
    hi = np.asarray(reference, dtype=float)
    keep = np.isfinite(lo) & np.isfinite(hi)
    variance = float(np.var(lo[keep], ddof=1))
    if variance <= 0:
        return float("nan")
    return float(np.cov(lo[keep], hi[keep], ddof=1)[0, 1] / variance)


def variance_match(predicted: np.ndarray | pd.Series, target_sd: float) -> float:
    """The factor that makes our effect spread equal ``target_sd``.

    An alternative to :func:`optimal_shrinkage` that does not require paired
    references, only a belief about how spread out real effects of this kind are.
    It is the more defensible choice when the pairing is thin: least squares on
    54 pairs across 6 intervention clusters is a noisy objective, while "real
    effects in megastudies of this shape have an SD of about 1 pp" is a claim that
    transfers.
    """
    spread = float(np.nanstd(np.asarray(predicted, dtype=float), ddof=1))
    if spread <= 0:
        return float("nan")
    return target_sd / spread


def outcome_profile(effects: pd.DataFrame, column: str = "estimate") -> pd.Series:
    """The per-outcome mean effect — the 13 numbers that carry most of pooled r."""
    _require(effects, "outcome")
    return effects.groupby("outcome")[column].mean()


def profile_agreement(left: pd.Series, right: pd.Series) -> dict:
    """How well one study's outcome profile predicts another's.

    The number that decides whether :func:`substitute_profile` is worth using at
    all.  ``r`` below roughly 0.7 means a transferred anchor scores worse than
    leaving our own profile in place, so this is a gate, not a diagnostic.
    """
    shared = sorted(set(left.index) & set(right.index))
    if len(shared) < 3:
        return {"n": len(shared), "r": float("nan"), "rank_r": float("nan")}
    a = left.loc[shared].to_numpy(float)
    b = right.loc[shared].to_numpy(float)
    from scipy import stats

    return {
        "n": len(shared),
        "r": float(np.corrcoef(a, b)[0, 1]),
        "rank_r": float(stats.spearmanr(a, b).statistic),
        "sd_left": float(a.std(ddof=1)),
        "sd_right": float(b.std(ddof=1)),
    }
