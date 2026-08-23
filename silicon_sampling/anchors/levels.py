"""The assembled anchor table: one control-arm level per Pfänder outcome, or none.

This is the module a calibration talks to.  Everything upstream of it is
measurement — read the file, filter to the US, weight it, convert the response
options — and everything here is *judgement*: which of those measurements is
allowed to stand in for a Pfänder level, what is subtracted from it first, and
which outcomes get nothing.

The judgement is deliberately conservative, for a reason the validation makes
quantitative.  On Voelkel, re-levelling a silicon sample to the true human levels
cuts mean Wasserstein-1 from 23.3 to 6.6 points for Qwen2.5-7B and from 10.0 to
4.2 for DeepSeek-V4-Flash — but an anchor carrying its own error of about 9
points erases the whole gain for the better model, and one carrying under 5 points
erases it on KS.  So an anchor is only worth offering if it is likely to be
accurate to within about five points, and a mapping that might be off by ten is
not a weak anchor, it is a harmful one.  Hence:

* only ``verbatim`` and ``near`` crosswalk grades are offered by default;
* the trust battery's measured referent shift is subtracted rather than ignored;
* every anchor also reports what the rival conversion assumption would have given,
  because that spread — ``(50 - mean) / k`` — is an error floor no sample size
  removes and it is the same size as the tolerance.

## Choosing between candidate groups

Two groups can offer the same outcome: TISP has both the battery and a rival
single item for ``trust_post``, and all three sources reach for one composite or
another.  The rule is grade, then coverage (how many of the outcome's Pfänder items
the group carries), then whether the source is post-stratified, then effective
sample size.

Grade first because a better-worded item beats a bigger sample by a wide margin
here — the sampling SEs are near 0.5 points and the wording differences are worth
several.  Post-stratification ahead of sample size because it is the difference
between estimating a population level and estimating a panel's, and a bigger
unweighted sample does not fix that; it is placed *after* grade and coverage
because it says nothing about whether the item asks Pfänder's question.

The rule does trade wording against representativeness, and ``behavior_mean`` is
where it shows: Goldwert's ``conversation`` is a 0-100 commitment-to-talk slider,
much the closer match to Pfänder's item, while CCAM's ``discuss_GW`` is a
four-point past-frequency question from a post-stratified panel.  The rule picks
CCAM.  It hardly matters, because the two disagree by 16.6 points — five times
break-even — so neither is offered and the outcome is simply not anchorable.

## What ends up anchored

Three of thirteen outcomes, all from TISP, all at grade ``near``:
``trust_multidimensional``, ``trust_post`` and ``policy_role_mean``.  The other
ten are either graded ``construct-only``/``unusable`` in the crosswalk or have no
candidate item at all.  That is a thin result and it is the correct one: the
climate-policy items that *look* anchorable disagree across two sources by 6.5 and
7.0 points, and the behaviour-intention items by 16.6, all above break-even.

Goldwert closed the two gaps where nothing had even been *found* — the donation and
the newsletter signup, both on Pfänder's own scales — without adding an offered
anchor, because instrument identity is not level transfer: see
:mod:`silicon_sampling.anchors.goldwert` for the $5 mode its group-contingent match
manufactures, and for why an embedded advocacy form is not an outbound link to a
scientist's newsletter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..calibration.tier1 import pfander_instrument
from ..pfander.outcomes import OUTCOMES, SCALE_RANGE
from . import ccam, crosswalk as cw, goldwert, tisp
from .scales import Weighted, sheppard_sd

#: Where each source's measurement comes from.  Adding a source needs nothing
#: else: selection is grade, then coverage, then effective sample size, and all
#: three are read off the crosswalk rather than off the module.
SOURCES = {tisp.SOURCE: tisp, ccam.SOURCE: ccam, goldwert.SOURCE: goldwert}

#: The only group the measured referent shift is applied to.  It was estimated
#: from a pair of *trust* items, so it generalises to trust items and no further:
#: applying it to the science-in-policy norm items would move an anchor that
#: already agrees with both samplers to within three points away from both.
REFERENT_ADJUSTED = ("TISP TRUST_SCI battery",)


@dataclass(frozen=True)
class Anchor:
    """One outcome's borrowed control level, with everything needed to doubt it."""

    outcome: str
    mean: float
    sd: float
    se: float
    n: int
    source: str
    group: str
    grade: str
    items: tuple[str, ...]
    mean_raw: float
    referent_adjustment: float
    mean_bin_midpoint: float
    sd_slider: float
    post_stratified: bool
    provenance: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def conversion_spread(self) -> float:
        """Distance to the rival conversion: the anchor's irreducible error."""
        return abs(self.mean_bin_midpoint - self.mean_raw)

    def as_row(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "mean": self.mean,
            "sd": self.sd,
            "sd_slider": self.sd_slider,
            "se": self.se,
            "n": self.n,
            "source": self.source,
            "group": self.group,
            "grade": self.grade,
            "post_stratified": self.post_stratified,
            "n_items": len(self.items),
            "mean_raw": self.mean_raw,
            "referent_adjustment": self.referent_adjustment,
            "mean_bin_midpoint": self.mean_bin_midpoint,
            "conversion_spread": self.conversion_spread,
        }


def _measure(entries: tuple[cw.Entry, ...], midpoint: bool = False) -> Weighted:
    module = SOURCES[entries[0].source]
    return module.measure(entries, midpoint=midpoint)


def _coverage(entries: tuple[cw.Entry, ...]) -> int:
    return len({entry.pfander_item for entry in entries})


def _slider_sd(sd: float, entry: cw.Entry) -> float:
    """The SD with the *conversion's* granularity removed, and only that.

    Sheppard's correction takes out variance that binning added, so it applies
    exactly when the source's granularity is coarser than the target's.  A native
    item shares the target's granularity — whole dollars for whole dollars, 0/1 for
    0/1 — so there is nothing to remove and the corrected SD is the SD.  Applying
    the five-option correction there by default would have subtracted 52 from a
    variance of 14 and reported a dispersion of zero.
    """
    if entry.conversion != "likert" or entry.source_options is None:
        return sd
    return sheppard_sd(sd, entry.source_options, scale=SCALE_RANGE[entry.pfander_outcome])


@lru_cache(maxsize=8)
def build(min_grade: str = cw.DEFAULT_MIN_GRADE, referent_adjust: bool = True):
    """Every anchor at or above ``min_grade``, one per outcome, best group wins."""
    candidates: dict[str, list[tuple[tuple[cw.Entry, ...], str]]] = {}
    for (outcome, group), entries in cw.groups().items():
        grade = cw.group_grade(entries)
        if cw.grade_rank(grade) > cw.grade_rank(min_grade):
            continue
        # A group whose source item has no ordered scale to convert cannot be an
        # anchor however it is graded, and a NaN level would sail into
        # ``calibrate`` and come back out as a NaN column.
        if not pd.notna(_measure(entries).mean):
            continue
        candidates.setdefault(outcome, []).append((entries, group))

    anchors: dict[str, Anchor] = {}
    shift = tisp.referent_gap().mean if referent_adjust else 0.0
    for outcome, options in candidates.items():
        entries, group = min(
            options,
            key=lambda pair: (
                cw.grade_rank(cw.group_grade(pair[0])),
                -_coverage(pair[0]),
                not SOURCES[pair[0][0].source].POST_STRATIFIED,
                -_measure(pair[0]).n_effective,
            ),
        )
        linear = _measure(entries)
        adjustment = shift if group in REFERENT_ADJUSTED else 0.0
        notes = tuple(dict.fromkeys(entry.note for entry in entries))
        if adjustment:
            notes = notes + (
                f"referent shift of {adjustment:.2f} points subtracted "
                "(TRUST_PEW minus CLIM_TRUST, measured within TISP)",
            )
        anchors[outcome] = Anchor(
            outcome=outcome,
            mean=linear.mean - adjustment,
            sd=linear.sd,
            se=linear.se,
            n=linear.n,
            source=entries[0].source,
            group=group,
            grade=cw.group_grade(entries),
            items=tuple(entry.source_item for entry in entries),
            mean_raw=linear.mean,
            referent_adjustment=adjustment,
            mean_bin_midpoint=_measure(entries, midpoint=True).mean,
            sd_slider=_slider_sd(linear.sd, entries[0]),
            post_stratified=SOURCES[entries[0].source].POST_STRATIFIED,
            provenance=SOURCES[entries[0].source].PROVENANCE,
            notes=notes,
        )
    return anchors


def levels(
    min_grade: str = cw.DEFAULT_MIN_GRADE,
    referent_adjust: bool = True,
    applicable_only: bool = True,
) -> dict[str, float]:
    """Just the numbers, keyed the way ``calibration.tier1.calibrate`` wants them.

    ``calibrate(frame, levels=anchors.levels())`` replaces the control-arm mean of
    each anchored outcome and holds every condition effect exactly fixed, so an
    anchor cannot move the leaderboard's ATE metrics in either direction — only the
    response distributions, the demographic baselines and the parity gap, which
    are the three scored analyses no effect calibration reaches.

    ``applicable_only`` drops the outcomes ``calibrate`` treats as binary.  It
    applies ``levels`` to the continuous outcomes only — a 0/1 outcome's arm rate is
    moved by flipping rows against a target *effect*, with the frame's own rate as
    the baseline — so a level handed in for ``newsletter_signup`` is silently
    ignored.  Silently is the problem: a caller would read the anchor in the table,
    pass it in, see no error and no change, and have no way to tell which.  Set this
    to ``False`` to get the raw mapping including levels that will not be applied.
    """
    binary = set(pfander_instrument().binary)
    return {
        outcome: anchor.mean
        for outcome, anchor in build(min_grade, referent_adjust).items()
        if not (applicable_only and outcome in binary)
    }


def facet_levels(referent_adjust: bool = True) -> dict[str, float]:
    """The four trust subscale levels, for a caller that wants the twelve items right.

    ``tier1.align_trust_items`` shifts all twelve items by whatever moved the
    composite, which reproduces the composite exactly but leaves the four facets in
    whatever relative position the sampler put them.  TISP says they are 12.6 points
    apart — competence highest, openness lowest — so a caller that cares about the
    item-level distributions has to place them, and these are the levels to use.
    """
    shift = tisp.referent_gap().mean if referent_adjust else 0.0
    return {facet: level - shift for facet, level in tisp.facet_levels().items()}


def to_frame(min_grade: str = "unusable", referent_adjust: bool = True) -> pd.DataFrame:
    """The anchor table, plus a row for every outcome that could not be anchored."""
    anchors = build(min_grade, referent_adjust)
    rows = [anchors[outcome].as_row() for outcome in OUTCOMES if outcome in anchors]
    for outcome in OUTCOMES:
        if outcome in anchors:
            continue
        rows.append(
            {
                "outcome": outcome,
                "mean": float("nan"),
                "grade": _why_not(outcome),
                "group": cw.UNMATCHED.get(outcome, ""),
            }
        )
    return pd.DataFrame(rows)


def _why_not(outcome: str) -> str:
    """Why an outcome has no anchor: nothing was found, or what was found is too weak."""
    found = [
        cw.group_grade(entries)
        for (name, _), entries in cw.groups().items()
        if name == outcome
    ]
    if not found:
        return "no candidate item"
    return f"best candidate graded {min(found, key=cw.grade_rank)}"
