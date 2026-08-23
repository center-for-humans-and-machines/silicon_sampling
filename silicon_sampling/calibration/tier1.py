"""Apply a calibration to a Tier-1 submission without breaking its format.

The generic machinery in :mod:`~silicon_sampling.calibration.components` treats
every outcome as an unbounded continuous number.  A real Pfänder submission is
not that, and three of its properties will silently invalidate a calibrated file
if they are ignored.

**The primary outcome is a composite of twelve submitted items.**
``trust_multidimensional`` is the mean of four subscale means, each the mean of
three items — which, because every subscale holds exactly three, is just the
plain mean of all twelve.  The submission carries all twelve alongside the
composite, and the format check warns when they disagree by more than 0.5.  So
moving the composite means moving the items with it; this module shifts all
twelve by the same amount, which reproduces the composite exactly and leaves the
subscale structure and inter-item correlations untouched.

**One outcome is binary.**  ``newsletter_signup`` is 0/1 and the check rejects
anything else, so a condition mean cannot be moved by adding a continuous offset.
It is moved by flipping the smallest number of rows that lands the arm's rate on
target.  Flips are seeded and drawn from whichever class has rows to spare, which
preserves most of the existing pattern but does perturb the demographic
composition of the signups slightly — a limitation worth knowing rather than a
hidden one, and the only honest option at a 0/1 grain.

**The scales are not all 0-100.**  ``donation_ams`` runs 0-10 and everything else
0-100.  The benchmark converts to percentage points of scale range at pair-build
time, so a calibration expressed in pp has to be converted back per outcome
before it touches the data.  Getting this wrong would scale the donation
calibration by ten.

Everything else is delegated: condition means land exactly on target because
:func:`~silicon_sampling.calibration.components.recompose` puts them there, and
the audit it returns is passed through so a clip that could not reach its target
is visible rather than absorbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..pfander import outcomes as pfander_outcomes
from .components import Decomposition, condition_effects, decompose, recompose_frame

CONTROL = "control"

#: The twelve items whose plain mean is ``trust_multidimensional``.
TRUST_ITEMS = tuple(
    f"trust_{facet}_{index}"
    for facet in ("competence", "integrity", "benevolence", "openness")
    for index in (1, 2, 3)
)

#: Outcomes that are not continuous and cannot be recomposed additively.
BINARY_OUTCOMES = ("newsletter_signup",)


def scale_of(outcome: str) -> float:
    """The outcome's native range, used to convert pp back to raw points."""
    return float(pfander_outcomes.SCALE_RANGE[outcome])


def pp_to_raw(effects: pd.DataFrame, column: str = "estimate") -> pd.DataFrame:
    """Convert an effect table from pp of scale range back to raw outcome points.

    The benchmark scores in pp, so calibrations are fitted and expressed there.
    The data is in raw points.  For the eleven 0-100 outcomes the conversion is
    the identity, which is exactly why forgetting it is easy: it is only wrong on
    ``donation_ams`` (by 10x) and ``newsletter_signup`` (by 100x).
    """
    out = effects.copy()
    factor = out["outcome"].map(lambda name: scale_of(name) / 100.0)
    out[column] = out[column] * factor
    if "se" in out.columns:
        out["se"] = out["se"] * factor
    return out


def _target_effects(
    frame: pd.DataFrame, outcome: str, targets: pd.DataFrame | None
) -> pd.Series:
    """The condition effects to aim at: supplied ones, or the frame's own."""
    if targets is None:
        return condition_effects(frame, outcome, CONTROL)
    picked = targets[targets["outcome"] == outcome]
    if picked.empty:
        return condition_effects(frame, outcome, CONTROL)
    wanted = picked.set_index("condition")["estimate"]
    if CONTROL not in wanted.index:
        wanted[CONTROL] = 0.0
    return wanted


def calibrate_binary(
    frame: pd.DataFrame,
    outcome: str,
    targets: pd.Series,
    seed: int = 0,
) -> np.ndarray:
    """Move a 0/1 outcome's per-arm rate onto target by flipping the fewest rows.

    The control arm's own rate is the baseline the targets are added to, so a
    target of zero leaves an arm alone.  Rates are clipped into ``[0, 1]``: an
    effect large enough to push a rate outside that is not representable, and the
    clip is reported by the caller's drift audit rather than silently applied.
    """
    rng = np.random.default_rng(seed)
    values = frame[outcome].to_numpy(float).copy()
    baseline = float(np.nanmean(values[frame["condition"].to_numpy() == CONTROL]))
    for arm in frame["condition"].unique():
        rows = np.flatnonzero(frame["condition"].to_numpy() == arm)
        if len(rows) == 0:
            continue
        wanted = float(np.clip(baseline + float(targets.get(arm, 0.0)), 0.0, 1.0))
        needed = int(round(wanted * len(rows)))
        ones = rows[values[rows] >= 0.5]
        zeros = rows[values[rows] < 0.5]
        surplus = len(ones) - needed
        if surplus > 0:  # too many signups: flip some to zero
            values[rng.choice(ones, size=min(surplus, len(ones)), replace=False)] = 0.0
        elif surplus < 0:  # too few: flip some zeros up
            values[rng.choice(zeros, size=min(-surplus, len(zeros)), replace=False)] = (
                1.0
            )
    return values


def calibrate(
    frame: pd.DataFrame,
    targets: pd.DataFrame | None = None,
    levels: dict[str, float] | None = None,
    offsets: dict[str, dict[str, pd.Series]] | None = None,
    moderators: tuple[str, ...] = tuple(pfander_outcomes.MODERATORS),
    outcomes: tuple[str, ...] = tuple(pfander_outcomes.OUTCOMES),
    seed: int = 20260823,
    targets_in_pp: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild a Tier-1 frame with calibrated effects, levels and demographic gaps.

    ``targets`` is an effect table (``outcome``, ``condition``, ``estimate``); any
    outcome it omits keeps its own effects, so a partial calibration is a valid
    input rather than an error.  ``levels`` overrides control-arm means per
    outcome — the route by which an external anchor such as CCAM or TISP enters,
    since Pfänder publishes no human data of its own.  ``offsets`` overrides the
    demographic offsets per outcome.

    Returns the rebuilt frame and the per-outcome drift audit.  Check the audit:
    a non-zero ``max_abs_effect_drift`` means a target was not reachable on that
    outcome's scale.
    """
    if targets is not None and targets_in_pp:
        targets = pp_to_raw(targets)

    continuous = [
        name
        for name in outcomes
        if name not in BINARY_OUTCOMES and name in frame.columns
    ]
    parts: dict[str, Decomposition] = {}
    for outcome in continuous:
        part = decompose(frame, outcome, moderators, CONTROL)
        parts[outcome] = Decomposition(
            outcome=outcome,
            level=(levels or {}).get(outcome, part.level),
            effects=_target_effects(frame, outcome, targets),
            offsets=(offsets or {}).get(outcome, part.offsets),
            residuals=part.residuals,
        )

    bounds = {name: (0.0, scale_of(name)) for name in continuous}
    rebuilt, drift = recompose_frame(frame, parts, CONTROL, bounds=bounds, seed=seed)

    for outcome in BINARY_OUTCOMES:
        if outcome not in frame.columns:
            continue
        rebuilt[outcome] = calibrate_binary(
            frame, outcome, _target_effects(frame, outcome, targets), seed=seed
        ).astype(int)

    rebuilt = align_trust_items(frame, rebuilt)
    return rebuilt, _audit(frame, rebuilt, parts, targets, drift)


def _audit(
    original: pd.DataFrame,
    rebuilt: pd.DataFrame,
    parts: dict[str, Decomposition],
    targets: pd.DataFrame | None,
    drift: pd.DataFrame,
) -> pd.DataFrame:
    """Re-measure drift on the *finished* frame, not on the intermediate one.

    Two steps run after :func:`recompose_frame` and both can move a condition
    mean: the binary flip approximates a rate at a 0/1 grain, and clipping the
    twelve trust items to 0-100 pulls the composite for respondents already at a
    boundary.  Auditing before those steps reported a clean 2e-14 while the
    primary outcome's realised shrinkage was 7% off its target — the audit has to
    describe the file that will actually be submitted, or it is worse than no
    audit at all.
    """
    rows = []
    for outcome in list(parts) + [
        name for name in BINARY_OUTCOMES if name in rebuilt.columns
    ]:
        wanted = _target_effects(original, outcome, targets)
        realised = condition_effects(rebuilt, outcome, CONTROL)
        shared = realised.index.intersection(wanted.index)
        gap = (realised.loc[shared] - wanted.loc[shared]).abs()
        level_target = (
            parts[outcome].level
            if outcome in parts
            else float(original.loc[original["condition"] == CONTROL, outcome].mean())
        )
        rows.append(
            {
                "outcome": outcome,
                "max_abs_effect_drift": float(gap.max()) if len(gap) else float("nan"),
                "mean_abs_effect_drift": (
                    float(gap.mean()) if len(gap) else float("nan")
                ),
                "level_drift": abs(
                    float(rebuilt.loc[rebuilt["condition"] == CONTROL, outcome].mean())
                    - level_target
                ),
            }
        )
    audited = pd.DataFrame(rows)
    if audited.empty:
        return audited
    if drift.empty or "outcome" not in drift.columns:
        # Nothing continuous was recomposed — a binary-only calibration — so there
        # are no pre-format figures to sit beside these.
        return audited
    # Keep the pre-clip figures alongside, so the cost of the format constraints
    # is legible rather than merely absorbed into a worse number.
    return audited.merge(
        drift.rename(
            columns={
                "max_abs_effect_drift": "max_abs_effect_drift_pre_format",
                "mean_abs_effect_drift": "mean_abs_effect_drift_pre_format",
                "level_drift": "level_drift_pre_format",
            }
        ),
        on="outcome",
        how="left",
    )


def align_trust_items(original: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Shift the twelve trust items by whatever moved the composite.

    A uniform shift is the right correction rather than a convenience: the
    composite is the items' plain mean, so adding the same delta to each
    reproduces the new composite exactly while leaving every inter-item
    difference — and therefore the subscale structure and the battery's internal
    reliability — untouched.  Clipping to 0-100 can pull the item mean off the
    composite for respondents already at a boundary, so the composite is refreshed
    from the clipped items afterwards, keeping the file internally consistent even
    where the requested shift was not fully representable.
    """
    present = [item for item in TRUST_ITEMS if item in rebuilt.columns]
    if "trust_multidimensional" not in rebuilt.columns or not present:
        return rebuilt
    out = rebuilt.copy()
    target = out["trust_multidimensional"].to_numpy(float)
    delta = target - original["trust_multidimensional"].to_numpy(float)
    items = original[present].to_numpy(float) + delta[:, None]
    moved = (
        _shift_to_row_mean(items, target) if len(present) == len(TRUST_ITEMS) else items
    )
    for index, item in enumerate(present):
        out[item] = np.clip(moved[:, index], 0.0, 100.0)
    if len(present) == len(TRUST_ITEMS):
        out["trust_multidimensional"] = out[present].mean(axis=1)
    return out


def _shift_to_row_mean(
    items: np.ndarray,
    target: np.ndarray,
    bounds: tuple[float, float] = (0.0, 100.0),
    rounds: int = 40,
) -> np.ndarray:
    """Make each row's mean equal ``target`` while keeping every item in range.

    A uniform shift plus a clip is not enough.  Respondents whose items already
    sit at 0 or 100 cannot absorb their share, so the row mean lands short and the
    composite drifts off the effect it was aimed at — measured at 0.273 raw points
    on the primary outcome, which is 36% of a shrunk effect of 0.75 pp.  So the
    shortfall is redistributed across the items that still have headroom, and
    iterated, exactly as the condition-mean clip does across rows.

    A row whose target is unreachable — every item pinned and the mean still short
    — stops at its closest achievable value rather than diverging, and the residue
    shows up in the caller's audit.
    """
    low, high = bounds
    values = np.clip(items, low, high)
    width = values.shape[1]
    for _ in range(rounds):
        gap = target - values.mean(axis=1)
        if np.abs(gap).max() < 1e-12:
            break
        movable = np.where(gap[:, None] > 0, values < high, values > low)
        room = movable.sum(axis=1)
        step = np.zeros_like(gap)
        np.divide(gap * width, room, out=step, where=room > 0)
        values = np.clip(values + np.where(movable, step[:, None], 0.0), low, high)
    return values


def composite_consistency(frame: pd.DataFrame) -> float:
    """Largest gap between the composite and its items' mean, the check's tolerance.

    The format gate warns above 0.5, so this is the number to assert on before
    writing a submission.
    """
    present = [item for item in TRUST_ITEMS if item in frame.columns]
    if "trust_multidimensional" not in frame.columns or len(present) != len(
        TRUST_ITEMS
    ):
        return float("nan")
    gap = (frame[present].mean(axis=1) - frame["trust_multidimensional"]).abs()
    return float(gap.max())
