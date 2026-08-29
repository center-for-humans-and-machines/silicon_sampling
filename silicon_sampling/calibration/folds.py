"""The four reference studies, described the way the benchmark scorer needs them.

``scripts/nested_cv.py`` used to carry its own miniature of the benchmark: a
scales dict, a substring match for the control arm, and a hand-rolled level and
dispersion comparison.  That was enough to grade *effects*, which is one of the
benchmark's four sections, and it is why the cross-validation could say nothing
about the other three.

:class:`~silicon_sampling.benchmark.scored.ScoredDesign` already describes a study
completely enough for every scored analysis -- interactions, response
distributions, demographic baselines, parity, stereotyping.  What was missing was
one of those objects per reference study.  This module supplies them, built from
each study package's own constants rather than restated here, so a study that
renames an arm or adds a moderator does not silently desynchronise from its fold.

The moderator level order comes from the *human* frame wherever the declared
order and the data disagree, because every interaction coefficient is a gap to
the first level and a level the humans do not contain cannot be a reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from ..benchmark.scored import ScoredDesign
from .tier1 import Instrument

#: Outcomes that are an exact arithmetic function of other scored outcomes of the
#: same study, and so contribute pairs without contributing information.
#:
#: Voelkel's ``Composite`` is the mean of its other eight outcomes -- exactly, to
#: floating-point equality -- so scoring all nine puts six of the fold's
#: fifty-four pairs in twice, in their smoothest and most predictable form.  The
#: measured effect on the fold means is under 0.002 in every variant, so this is a
#: correctness fix rather than a consequential one; it is applied because Pfänder
#: has no such duplication (``trust_multidimensional`` is scored, its four
#: subscales are not), and a fold should not be graded on a grid the target study
#: does not have.
REDUNDANT_OUTCOMES: Mapping[str, tuple[str, ...]] = {
    "Voelkel": ("Composite",),
}

#: Outcomes that are 0/1 and so are estimated by logistic marginal effects rather
#: than OLS, exactly as the benchmark treats ``newsletter_signup``.
#:
#: On a saturated ``outcome ~ condition`` fit the two agree to floating point --
#: both reproduce the difference in arm proportions -- so this changes the
#: standard errors rather than the estimates.  It is declared because the
#: standard errors feed ``pearson_adj``, ``rmse_adj`` and ``beta_adj``.
BINARY_OUTCOMES: Mapping[str, tuple[str, ...]] = {
    "Goldwert": ("newsletter", "petition"),
    "ICPC": ("sharing",),
}

#: Composite outcomes and the columns whose plain mean they must equal, so that a
#: calibration that moves the parts moves the whole with them.
#:
#: **Empty on purpose for Voelkel.**  ``Composite`` is the mean of the other eight
#: outcomes, so it looks like Pfänder's ``trust_multidimensional``.  It is not the
#: same situation: Pfänder scores the composite and not its twelve items, while
#: Voelkel scores the eight components and (here) not the composite.  Declaring it
#: makes :func:`~silicon_sampling.calibration.tier1.calibrate` hand every
#: component the *composite's* effect vector -- which is correct when the items
#: are unscored carriers and catastrophic when they are the outcomes, dropping the
#: Voelkel fold from r = 0.576 to 0.132.  ``Composite`` is dropped from the scored
#: grid instead (see :data:`REDUNDANT_OUTCOMES`), which is what it deserves.
COMPOSITES: Mapping[str, Mapping[str, tuple[str, ...]]] = {}


@dataclass(frozen=True)
class FoldStudy:
    """One reference study: its design, its humans, and its silicon samples."""

    name: str
    design: ScoredDesign
    load_humans: Callable[[], pd.DataFrame]
    effects: Callable[[pd.DataFrame], pd.DataFrame]
    samples_dir: Callable[[str], object]
    #: Repair applied to every frame -- human and synthetic alike -- before it is
    #: scored.  Used where a study's instrument was rendered without a constraint
    #: the real questionnaire enforced; see :func:`silicon_sampling.ccc.score.prepare`.
    prepare: Callable[[pd.DataFrame], pd.DataFrame] = lambda frame: frame
    #: Outcomes dropped from the scored grid; see :data:`REDUNDANT_OUTCOMES`.
    dropped: tuple[str, ...] = ()
    #: Composite -> its component columns, for the calibration layer.
    composites: Mapping[str, tuple[str, ...]] = None  # type: ignore[assignment]

    @property
    def scales(self) -> dict[str, float]:
        return dict(self.design.outcomes)

    @property
    def instrument(self) -> Instrument:
        """The same study, in the shape :mod:`.recipes` asks for.

        The scored design and the calibration instrument have to describe one
        study or a recipe is *applied* under one set of facts and *graded* under
        another -- which is exactly the seam the shipped Pfänder path closed and
        the cross-validation had left open, because it never built a calibrated
        frame at all.
        """
        return Instrument(
            scales=dict(self.design.outcomes),
            control=self.design.control,
            moderators=tuple(self.design.moderators),
            binary=tuple(self.design.binary),
            composites=dict(self.composites or {}),
        )


def _moderators(
    human: pd.DataFrame, declared: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Each moderator's level order, taken from the human frame.

    Declared-but-absent levels are dropped and observed-but-undeclared levels are
    appended, so the reference level is always one the human data contains.
    """
    out: dict[str, tuple[str, ...]] = {}
    for name in declared:
        if name not in human.columns:
            continue
        observed = human[name].dropna().astype(str)
        levels = tuple(sorted(observed.unique()))
        if len(levels) < 2:
            continue
        out[name] = levels
    return out


def _design(
    human: pd.DataFrame,
    scales: Mapping[str, float],
    control: str,
    moderators: Sequence[str],
    condition_col: str,
    dropped: Sequence[str] = (),
    binary: Sequence[str] = (),
) -> ScoredDesign:
    labels = human[condition_col].dropna().astype(str)
    conditions = [control] + sorted(set(labels.unique()) - {control})
    outcomes = {
        name: scale
        for name, scale in scales.items()
        if name in human.columns and name not in set(dropped)
    }
    return ScoredDesign(
        outcomes=outcomes,
        control=control,
        moderators=_moderators(human, moderators),
        conditions=conditions,
        condition_col=condition_col,
        binary=tuple(name for name in binary if name in outcomes),
    )


def voelkel() -> FoldStudy:
    from ..voelkel import outcomes as oc
    from ..voelkel import paths, score

    human = score.load_humans()
    dropped = REDUNDANT_OUTCOMES.get("Voelkel", ())
    return FoldStudy(
        name="Voelkel",
        design=_design(
            human,
            dict(oc.OUTCOMES),
            score.CONTROL,
            (*score.VISIBLE_MODERATORS, *score.INVISIBLE_MODERATORS),
            "condition",
            dropped,
        ),
        load_humans=score.load_humans,
        effects=score.effects,
        samples_dir=paths.samples_dir,
        dropped=dropped,
        composites=COMPOSITES.get("Voelkel", {}),
    )


def icpc() -> FoldStudy:
    from ..icpc import outcomes as oc
    from ..icpc import paths, score

    human = score.load_humans()
    return FoldStudy(
        name="ICPC",
        design=_design(
            human,
            dict(oc.OUTCOMES),
            score.CONTROL,
            score.VISIBLE_MODERATORS,
            "condition",
            binary=BINARY_OUTCOMES.get("ICPC", ()),
        ),
        load_humans=score.load_humans,
        effects=score.effects,
        samples_dir=paths.samples_dir,
    )


def goldwert() -> FoldStudy:
    from ..goldwert import outcomes as oc
    from ..goldwert import paths, score

    human = score.load_humans()
    return FoldStudy(
        name="Goldwert",
        design=_design(
            human,
            dict(oc.SCORED),
            score.CONTROL,
            (*score.VISIBLE_MODERATORS, *score.INVISIBLE_MODERATORS),
            "condition",
            binary=BINARY_OUTCOMES.get("Goldwert", ()),
        ),
        load_humans=score.load_humans,
        effects=score.effects,
        samples_dir=paths.samples_dir,
    )


def ccc() -> FoldStudy:
    from ..ccc import outcomes as oc
    from ..ccc import paths, score

    human = score.load_humans()
    return FoldStudy(
        name="CCC",
        design=_design(
            human,
            dict(oc.SCORED),
            score.CONTROL,
            ("party", "gender", "race", "education", "age_band"),
            "condition",
        ),
        load_humans=score.load_humans,
        # The three placebo arms carry their own labels in a raw silicon sample
        # and the constant-sum donation budget is unenforced in one, so both
        # sides have to be prepared before anything is estimated.
        effects=score.effects,
        samples_dir=paths.samples_dir,
        prepare=score.prepare,
    )


#: The four folds, in the order the reports list them.
BUILDERS: Mapping[str, Callable[[], FoldStudy]] = {
    "Voelkel": voelkel,
    "ICPC": icpc,
    "Goldwert": goldwert,
    "CCC": ccc,
}


def load_folds(names: Sequence[str] | None = None) -> list[FoldStudy]:
    """Build the requested folds, or all four."""
    wanted = list(names or BUILDERS)
    unknown = [name for name in wanted if name not in BUILDERS]
    if unknown:
        raise ValueError(
            f"unknown study: {', '.join(unknown)} (have {sorted(BUILDERS)})"
        )
    return [BUILDERS[name]() for name in wanted]
