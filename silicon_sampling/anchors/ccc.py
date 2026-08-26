"""Pfänder anchors measured on Voelkel et al. (2026), the Climate Change Challenge.

This is the strongest anchor source in the project, because Pfänder took several of
its items from this study. Two of its outcomes are the **same questions on the same
scale in the same population** — verified verbatim against both questionnaires —
where TISP is ``near`` grade on a different survey and CCAM is a
nationally-representative panel whose gaps run larger than experimental samples'.

Everything here is read off CCC's pooled placebo-control arm (n = 3,183): a control
that read about neckties, baseball or dances, so nothing in it moves climate
attitudes.

**Provenance is recorded per anchor and it is not decoration.** CCC is also one of
the cross-validation folds. An anchor derived from it must be switched off when CCC
is the held-out study, or the fold is scored partly against numbers taken from its
own human data. :func:`for_study` exists so that is one argument rather than a
thing to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lazy import lazy_module

pd = lazy_module("pandas")

#: Which study each anchor came from, so a fold can exclude its own.
SOURCE = "CCC"


@dataclass(frozen=True)
class Anchor:
    """One measured quantity, with the grade of its item match."""

    pfander_outcome: str
    ccc_outcome: str
    #: ``identical`` — same question, same scale, verified verbatim.
    #: ``partial`` — same stem and scale, some items shared.
    #: ``construct-only`` — same construct, different items.
    grade: str
    level: float
    sd: float
    party_gap: float
    note: str
    #: Whether ``level`` and ``sd`` transfer to Pfänder's scale at all.
    #:
    #: A gap expressed in pp of each scale's own range is scale-free and transfers
    #: whatever the units; a mean and a standard deviation in raw points do not.
    #: CCC's donation is cents allocated out of 100 across five charities, and
    #: Pfänder's is an amount on 0-10 — so its gap is usable and its level is not.
    #: Anchoring the raw 61.54 onto a 0-10 outcome clipped every respondent to the
    #: ceiling, collapsed the column's variance to zero, and pushed the entry's
    #: effect drift from 4.9e-04 to 1.9e-01 before this field existed.
    scale_comparable: bool = True


#: Measured on ``CCC - Data - Recoded.csv``, pooled control arm, n = 3,183.
#:
#: Reproduce with :func:`measure`. The values are frozen here rather than computed
#: on import so that a submission is reproducible from the repository alone, and so
#: that a change in the source file shows up as a test failure rather than as a
#: silently different submission.
ANCHORS = (
    Anchor(
        "concern_mean",
        "Concern_Post",
        "identical",
        60.42,
        31.65,
        37.7,
        "all three concern items verbatim identical, same 101-point scale",
    ),
    Anchor(
        "policy_general",
        "Policies_Post",
        "identical",
        68.01,
        29.32,
        32.9,
        "'The U.S. government should do more to reduce global warming' verbatim",
    ),
    Anchor(
        "behavior_mean",
        "IntentNp_Post",
        "partial",
        54.55,
        24.26,
        16.2,
        "3 of 6 non-political intention items verbatim, same stem and scale",
    ),
    Anchor(
        "belief_post",
        "Belief_Post",
        "construct-only",
        65.44,
        22.52,
        22.8,
        "Pfänder condensed three belief items into one accuracy rating",
    ),
    Anchor(
        "policy_specific_mean",
        "PoliciesSp_Post",
        "construct-only",
        53.29,
        23.98,
        25.3,
        "both are four-item specific-policy batteries, different policies",
    ),
    Anchor(
        "donation_ams",
        "Donation",
        "construct-only",
        61.54,
        45.32,
        12.4,
        "a real-money donation either side, but cents-out-of-100 against 0-10",
        scale_comparable=False,
    ),
)

#: Grades trusted for level and dispersion anchoring, in descending order.
TRUSTED_GRADES = ("identical", "partial", "construct-only")


def by_outcome(grades: tuple[str, ...] = TRUSTED_GRADES) -> dict[str, Anchor]:
    """Anchors keyed by Pfänder outcome, restricted to the grades named."""
    return {a.pfander_outcome: a for a in ANCHORS if a.grade in grades}


def levels(grades: tuple[str, ...] = TRUSTED_GRADES) -> dict[str, float]:
    """Control-arm means, for the outcomes whose scale is comparable."""
    return {
        name: a.level for name, a in by_outcome(grades).items() if a.scale_comparable
    }


def dispersion(grades: tuple[str, ...] = TRUSTED_GRADES) -> dict[str, float]:
    """Control-arm standard deviations, for the outcomes whose scale is comparable."""
    return {name: a.sd for name, a in by_outcome(grades).items() if a.scale_comparable}


def party_gaps(grades: tuple[str, ...] = TRUSTED_GRADES) -> dict[str, float]:
    return {name: a.party_gap for name, a in by_outcome(grades).items()}


def for_study(held_out: str | None) -> tuple[str, ...]:
    """The grades usable when *held_out* is the study being scored.

    Returns an empty tuple when CCC itself is held out, which switches every
    CCC-derived anchor off. Anything else gets the full set.
    """
    return () if (held_out or "").upper() == SOURCE else TRUSTED_GRADES


def measure() -> "pd.DataFrame":
    """Re-measure everything in :data:`ANCHORS` from the source file.

    Used by the test that keeps the frozen numbers honest.
    """
    from ..ccc import outcomes as co
    from ..ccc import score as cs

    humans = cs.load_humans()
    control = humans[humans["condition"] == cs.CONTROL]
    rows = []
    for anchor in ANCHORS:
        column = anchor.ccc_outcome
        if column not in control.columns:
            continue
        values = pd.to_numeric(control[column], errors="coerce")
        scale = co.SCORED[column]
        democrat = values[control["party"] == "Democrat"]
        republican = values[control["party"] == "Republican"]
        rows.append(
            {
                "pfander_outcome": anchor.pfander_outcome,
                "grade": anchor.grade,
                "level": float(values.mean()),
                "sd": float(values.std()),
                "party_gap": float((democrat.mean() - republican.mean()) / scale * 100),
            }
        )
    return pd.DataFrame(rows)
