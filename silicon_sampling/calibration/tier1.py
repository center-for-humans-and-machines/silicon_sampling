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

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Instrument:
    """The study-specific facts :func:`calibrate` needs, so it is not Pfänder-only.

    An earlier version hard-wired Pfänder's control label, scale ranges and trust
    battery, which meant a calibration could be *applied* through this module but
    had to be *validated* by calling the layer underneath it.  That is exactly the
    gap where a bug survives review: the path that produced the evidence was not
    the path that produced the submission.  Everything study-specific now arrives
    here instead.
    """

    #: outcome -> native scale range, e.g. 100.0 for a slider, 10.0 for the donation.
    scales: dict[str, float]
    control: str = CONTROL
    moderators: tuple[str, ...] = ()
    #: outcomes that are 0/1 and must be moved by flipping rows, not by adding.
    binary: tuple[str, ...] = ()
    #: composite -> the items whose plain mean it must equal.
    composites: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(self.scales)


def pfander_instrument() -> Instrument:
    """Pfänder's own configuration, read from the study package rather than copied."""
    return Instrument(
        scales=dict(pfander_outcomes.SCALE_RANGE),
        control=CONTROL,
        moderators=tuple(pfander_outcomes.MODERATORS),
        binary=BINARY_OUTCOMES,
        composites={"trust_multidimensional": TRUST_ITEMS},
    )


def scale_of(outcome: str, instrument: Instrument | None = None) -> float:
    """The outcome's native range, used to convert pp back to raw points."""
    scales = (instrument or pfander_instrument()).scales
    return float(scales[outcome])


def pp_to_raw(
    effects: pd.DataFrame,
    column: str = "estimate",
    instrument: Instrument | None = None,
) -> pd.DataFrame:
    """Convert an effect table from pp of scale range back to raw outcome points.

    The benchmark scores in pp, so calibrations are fitted and expressed there.
    The data is in raw points.  For the eleven 0-100 outcomes the conversion is
    the identity, which is exactly why forgetting it is easy: it is only wrong on
    ``donation_ams`` (by 10x) and ``newsletter_signup`` (by 100x).
    """
    out = effects.copy()
    factor = out["outcome"].map(lambda name: scale_of(name, instrument) / 100.0)
    out[column] = out[column] * factor
    if "se" in out.columns:
        out["se"] = out["se"] * factor
    return out


def _target_effects(
    frame: pd.DataFrame,
    outcome: str,
    targets: pd.DataFrame | None,
    control: str,
) -> pd.Series:
    """The condition effects to aim at: supplied ones, or the frame's own."""
    if targets is None:
        return condition_effects(frame, outcome, control)
    picked = targets[targets["outcome"] == outcome]
    if picked.empty:
        return condition_effects(frame, outcome, control)
    wanted = picked.set_index("condition")["estimate"]
    if control not in wanted.index:
        wanted[control] = 0.0
    return wanted


def calibrate_binary(
    frame: pd.DataFrame,
    outcome: str,
    targets: pd.Series,
    seed: int = 0,
    control: str = CONTROL,
) -> np.ndarray:
    """Move a 0/1 outcome's per-arm rate onto target by flipping the fewest rows.

    The control arm's own rate is the baseline the targets are added to, so a
    target of zero leaves an arm alone.  Rates are clipped into ``[0, 1]``: an
    effect large enough to push a rate outside that is not representable, and the
    clip is reported by the caller's drift audit rather than silently applied.

    **Missing values are missing, not zeros.**  An unparseable answer leaves the
    cell ``NaN``, and ``NaN`` satisfies neither ``>= 0.5`` nor ``< 0.5`` — so a
    naive version counts those rows in the arm's denominator while being unable
    to flip any of them, which drives the rate off target by the missingness
    rate, and then ``astype(int)`` on the surviving ``NaN`` turns it into
    ``-2**63``.  Pfänder's ``newsletter_signup`` has no missing values so the
    shipped entry never met this, but ICPC's ``sharing`` is missing on 24% of
    rows and produced exactly that integer.  The rate is therefore computed over
    the rows that *have* an answer, and missing rows are returned missing.
    """
    rng = np.random.default_rng(seed)
    values = frame[outcome].to_numpy(float).copy()
    conditions = frame["condition"].to_numpy()
    baseline = float(np.nanmean(values[conditions == control]))
    for arm in frame["condition"].unique():
        rows = np.flatnonzero((conditions == arm) & ~np.isnan(values))
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
    moderators: tuple[str, ...] | None = None,
    outcomes: tuple[str, ...] | None = None,
    seed: int = 20260823,
    targets_in_pp: bool = True,
    instrument: Instrument | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild a Tier-1 frame with calibrated effects, levels and demographic gaps.

    ``targets`` is an effect table (``outcome``, ``condition``, ``estimate``); any
    outcome it omits keeps its own effects, so a partial calibration is a valid
    input rather than an error.  ``levels`` overrides control-arm means per
    outcome — the route by which an external anchor such as CCAM or TISP enters,
    since Pfänder publishes no human data of its own.  ``offsets`` overrides the
    demographic offsets per outcome.

    ``instrument`` supplies the study's control label, scale ranges, binary
    outcomes and composites; it defaults to Pfänder's.  Passing another study's
    lets a calibration be validated through this exact code path rather than
    through the layer beneath it.

    Returns the rebuilt frame and the per-outcome drift audit.  Check the audit:
    a non-zero ``max_abs_effect_drift`` means a target was not reachable on that
    outcome's scale.
    """
    design = instrument or pfander_instrument()
    control = design.control
    moderators = moderators if moderators is not None else design.moderators
    outcomes = outcomes if outcomes is not None else design.outcomes
    if targets is not None and targets_in_pp:
        targets = pp_to_raw(targets, instrument=design)

    # A composite is rebuilt through its items, never beside them.  Recomposing
    # the composite directly draws it a fresh residual, and the twelve items then
    # have to absorb a shift of that size to stay consistent with it — which clips
    # against the scale and put 0.27 raw points of drift on the primary outcome,
    # 36% of a shrunk effect.  Giving each item the composite's target effect makes
    # their mean carry that effect exactly, while each item keeps its own level,
    # demographic offsets and residual.
    composites = {
        name: [item for item in items if item in frame.columns]
        for name, items in design.composites.items()
        if name in frame.columns
    }
    composites = {
        name: items
        for name, items in composites.items()
        if len(items) == len(design.composites[name])
    }
    via_items = {item for items in composites.values() for item in items}
    # Every item of a composite is given the *composite's* effect vector, which is
    # right when the items exist only to carry it (Pfänder's twelve trust items,
    # which the benchmark never scores) and destroys the calibration when the
    # items are themselves scored outcomes -- their own targets are silently
    # replaced.  Declaring such a composite is a study-description error, so it
    # stops here rather than printing a plausible number.
    clashing = sorted(via_items & set(outcomes))
    if clashing:
        raise ValueError(
            "composite items that are themselves scored outcomes: "
            f"{', '.join(clashing)} — each would be given the composite's effects "
            "instead of its own; drop the composite from the instrument, or the "
            "items from the scored outcome set"
        )

    continuous = [
        name
        for name in outcomes
        if name not in design.binary
        and name not in composites
        and name in frame.columns
    ]
    parts: dict[str, Decomposition] = {}
    for outcome in continuous:
        part = decompose(frame, outcome, moderators, control)
        parts[outcome] = Decomposition(
            outcome=outcome,
            level=(levels or {}).get(outcome, part.level),
            effects=_target_effects(frame, outcome, targets, control),
            offsets=(offsets or {}).get(outcome, part.offsets),
            residuals=part.residuals,
        )

    for name, items in composites.items():
        composite_part = decompose(frame, name, moderators, control)
        wanted_level = (levels or {}).get(name)
        shift = 0.0 if wanted_level is None else wanted_level - composite_part.level
        wanted_effects = _target_effects(frame, name, targets, control)
        for item in items:
            item_part = decompose(frame, item, moderators, control)
            parts[item] = Decomposition(
                outcome=item,
                level=item_part.level + shift,
                effects=wanted_effects,
                offsets=(offsets or {}).get(name, item_part.offsets),
                residuals=item_part.residuals,
            )

    bounds = {name: (0.0, scale_of(name, design)) for name in continuous}
    bounds.update({item: (0.0, 100.0) for item in via_items})
    rebuilt, drift = recompose_frame(
        frame, parts, control, bounds=bounds, seed=seed, resample_residuals=False
    )
    for name, items in composites.items():
        rebuilt[name] = rebuilt[items].mean(axis=1)

    for outcome in design.binary:
        if outcome not in frame.columns:
            continue
        moved = calibrate_binary(
            frame,
            outcome,
            _target_effects(frame, outcome, targets, control),
            seed=seed,
            control=control,
        )
        # ``astype(int)`` on a NaN is ``-2**63``, which then reads as a real
        # answer everywhere downstream.  A nullable integer keeps the hole a hole.
        rebuilt[outcome] = pd.array(moved, dtype="Float64").astype("Int64")

    return rebuilt, _audit(frame, rebuilt, parts, targets, drift, design)


def _composite_items(design: Instrument) -> set[str]:
    """Items that exist only to carry a composite, and are not scored themselves."""
    return {item for items in design.composites.values() for item in items}


def _audit(
    original: pd.DataFrame,
    rebuilt: pd.DataFrame,
    parts: dict[str, Decomposition],
    targets: pd.DataFrame | None,
    drift: pd.DataFrame,
    design: Instrument,
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
    audited = [name for name in parts if name not in _composite_items(design)]
    audited += [name for name in design.composites if name in rebuilt.columns]
    audited += [name for name in design.binary if name in rebuilt.columns]
    for outcome in audited:
        wanted = _target_effects(original, outcome, targets, design.control)
        realised = condition_effects(rebuilt, outcome, design.control)
        shared = realised.index.intersection(wanted.index)
        gap = (realised.loc[shared] - wanted.loc[shared]).abs()
        level_target = (
            parts[outcome].level
            if outcome in parts
            else float(
                original.loc[original["condition"] == design.control, outcome].mean()
            )
        )
        rows.append(
            {
                "outcome": outcome,
                "max_abs_effect_drift": float(gap.max()) if len(gap) else float("nan"),
                "mean_abs_effect_drift": (
                    float(gap.mean()) if len(gap) else float("nan")
                ),
                "level_drift": abs(
                    float(
                        rebuilt.loc[
                            rebuilt["condition"] == design.control, outcome
                        ].mean()
                    )
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


def align_composites(
    original: pd.DataFrame,
    rebuilt: pd.DataFrame,
    design: Instrument | None = None,
) -> pd.DataFrame:
    """Shift each composite's items by whatever moved the composite.

    A uniform shift is the right correction rather than a convenience: the
    composite is the items' plain mean, so adding the same delta to each
    reproduces the new composite exactly while leaving every inter-item
    difference — and therefore the subscale structure and the battery's internal
    reliability — untouched.  Clipping to 0-100 can pull the item mean off the
    composite for respondents already at a boundary, so the composite is refreshed
    from the clipped items afterwards, keeping the file internally consistent even
    where the requested shift was not fully representable.
    """
    design = design or pfander_instrument()
    out = rebuilt.copy()
    for composite, items in design.composites.items():
        if composite not in out.columns:
            continue
        present = [item for item in items if item in out.columns]
        if not present:
            continue
        target = out[composite].to_numpy(float)
        delta = target - original[composite].to_numpy(float)
        shifted = original[present].to_numpy(float) + delta[:, None]
        complete = len(present) == len(items)
        limit = scale_of(composite, design)
        moved = (
            _shift_to_row_mean(shifted, target, bounds=(0.0, limit))
            if complete
            else shifted
        )
        for index, item in enumerate(present):
            out[item] = np.clip(moved[:, index], 0.0, limit)
        if complete:
            out[composite] = out[present].mean(axis=1)
    return out


#: Kept as an alias: the Pfänder-only name this function used to have.
align_trust_items = align_composites


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


def composite_consistency(
    frame: pd.DataFrame, design: Instrument | None = None
) -> float:
    """Largest gap between any composite and its items' mean, across the whole file.

    The format gate warns above 0.5, so this is the number to assert on before
    writing a submission.  Returns NaN when the instrument declares no composite,
    or when a declared one is missing items from the frame — a missing check is
    reported as unknown rather than as a pass.
    """
    design = design or pfander_instrument()
    worst = float("nan")
    for composite, items in design.composites.items():
        present = [item for item in items if item in frame.columns]
        if composite not in frame.columns or len(present) != len(items):
            continue
        gap = float((frame[present].mean(axis=1) - frame[composite]).abs().max())
        worst = gap if np.isnan(worst) else max(worst, gap)
    return worst
