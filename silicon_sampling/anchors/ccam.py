"""The Climate Change in the American Mind trend file, as climate-item levels.

CCAM is an Ipsos KnowledgePanel trend survey with a per-wave post-stratification
weight, and it is the obvious place to look for Pfänder's climate outcomes:
belief, concern, general and specific policy support.  What it actually delivers
is much less than that, and the honest result of this module is mostly negative,
so the reasoning is set out here rather than buried in the crosswalk notes.

**The wave.**  The file spans Nov 2008 to Dec 2024 across 31 waves.  Levels are
what we are after and climate attitudes trend, so pooling waves would blur a
sixteen-year drift into an average that describes no year.  The most recent wave,
**Dec 2024 (wave 31, n = 1,013)**, carries every item the crosswalk asks for —
which is not true of Apr 2024 or Oct 2023, both of which drop two or three — so
no pooling is needed and none is done.  At n = 1,013 with a Kish effective n of
about 880, an item's SE lands near 1.1 points on the converted 0-100 scale, an
order of magnitude smaller than the conversion ambiguity in
:mod:`silicon_sampling.anchors.scales`.

**Why almost nothing here is offered as an anchor.**  Pfänder measures belief as a
rated accuracy of "Human activities are causing climate change" on a slider; CCAM
measures cause attribution as unordered categories and "is it happening" as
yes/no/don't-know.  A proportion is not a slider mean and scoring the categories
0/50/100 would invent the number rather than measure it, so ``belief_post`` is
graded unusable.  ``concern_mean`` is a three-item composite of which CCAM has a
near counterpart to one item (``worry``) and nothing for the other two.
``policy_general`` asks for support for a statement about federal action; CCAM's
nearest item asks whether global warming should be a priority for the president
and Congress, which is a different question.  Not one CCAM item reaches ``near``.

**Where CCAM is still earning its place.**  Two of its policy items overlap TISP's,
which makes them the only cross-source check available on the whole scale-conversion
exercise: two nationally representative samples, different instruments, different
response granularities, same construct.  They disagree by 6.5 and 7.0 points, which
is above the break-even anchor error measured in
:mod:`silicon_sampling.anchors.validate`.  That disagreement is the most useful
number CCAM produces, and it argues against using either source for policy support.

**Missing codes.**  ``-1`` is Refused throughout and ``0`` is "Don't know" on the
``harm_*`` items; both are dropped rather than scored, per the crosswalk row.  The
codes are declared per entry instead of applied by a blanket rule because ``2`` is
a substantive "Don't know" inside ``happening`` and ``1`` is one inside
``cause_recoded`` — a blanket rule would be right for the items we use and wrong
for the two we reject, which is exactly the sort of thing that stops being
noticeable once written as a default.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .. import paths as _paths

import pandas as pd

from . import crosswalk as cw
from .scales import NOT_MEASURED, Weighted, composite, weighted_moments

ROOT = Path(__file__).resolve().parents[2]
SAV = _paths.resolve("calibration", "datasets") / "ccam.sav"
CODEBOOK = _paths.resolve("calibration", "codebooks") / "ccam_codebook.pdf"

SOURCE = "CCAM"
WEIGHT = "weight_wave"
#: Wave 31, Dec 2024: the most recent, and the most recent one that is complete.
DEFAULT_WAVE = 31.0
#: Post-stratified: ``weight_wave`` reweights each wave to population margins.
POST_STRATIFIED = True
PROVENANCE = "CCAM (Ipsos KnowledgePanel), wave 31, Dec 2024, n = 1,013, weight_wave"


@lru_cache(maxsize=4)
def load(wave: float | None = DEFAULT_WAVE, path: str | None = None) -> pd.DataFrame:
    """One wave with a numeric ``weight`` column; ``wave=None`` pools everything.

    Pooling is offered because refusing to offer it would only push a caller into
    doing it by hand, but it is not the default and nothing in this package uses
    it: an average over sixteen years of trending attitudes is a level that
    describes no year, which is the opposite of what an anchor is for.
    """
    import pyreadstat

    frame, _ = pyreadstat.read_sav(str(Path(path) if path else SAV))
    if wave is not None:
        frame = frame[frame["wave"] == wave].copy()
    frame["weight"] = pd.to_numeric(frame[WEIGHT], errors="coerce")
    return frame.reset_index(drop=True)


def wave_label(wave: float = DEFAULT_WAVE, path: str | None = None) -> str:
    """The wave's own label out of the SPSS value labels, for the report."""
    import pyreadstat

    _, meta = pyreadstat.read_sav(
        str(Path(path) if path else SAV), metadataonly=True  # type: ignore[call-arg]
    )
    return str(meta.variable_value_labels.get("wave", {}).get(wave, wave))


def measure(
    entries: tuple[cw.Entry, ...],
    frame: pd.DataFrame | None = None,
    midpoint: bool = False,
) -> Weighted:
    """The weighted level of one crosswalk group, converted to Pfänder's scale."""
    entries = tuple(entry for entry in entries if entry.source == SOURCE)
    if not entries or any(entry.source_options is None for entry in entries):
        return NOT_MEASURED
    options = {entry.source_options for entry in entries}
    if len(options) != 1:
        raise ValueError(f"mixed response granularity in group: {sorted(options)}")
    missing = tuple(sorted({code for entry in entries for code in entry.missing_codes}))
    data = load() if frame is None else frame
    values = composite(
        data,
        tuple(entry.source_item for entry in entries),
        options=options.pop(),
        missing_codes=missing,
        midpoint=midpoint,
    )
    return weighted_moments(values, data["weight"])


def item_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every CCAM crosswalk item on its own, both conversions side by side."""
    data = load() if frame is None else frame
    rows = []
    for entry in cw.CROSSWALK:
        if entry.source != SOURCE or entry.source_options is None:
            continue
        linear = measure((entry,), data)
        rows.append(
            {
                "pfander_outcome": entry.pfander_outcome,
                "pfander_item": entry.pfander_item,
                "source_item": entry.source_item,
                "options": entry.source_options,
                "grade": entry.grade,
                "mean": linear.mean,
                "sd": linear.sd,
                "se": linear.se,
                "mean_bin_midpoint": measure((entry,), data, midpoint=True).mean,
                "n": linear.n,
            }
        )
    return pd.DataFrame(rows)


def cross_source_check(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Where CCAM and TISP measure the same Pfänder item, how far apart they land.

    The only empirical handle this project has on how much a converted Likert
    anchor can be trusted: two nationally representative US samples, two
    instruments, two granularities, one construct.  The spread is the error a
    caller should expect from *any* single-source anchor on that construct, and it
    is reported next to the break-even error for exactly that comparison.
    """
    from . import tisp

    data = load() if frame is None else frame
    rows = []
    for entry in cw.CROSSWALK:
        if entry.source != SOURCE or entry.source_options is None:
            continue
        partners = [
            other
            for other in cw.CROSSWALK
            if other.source == tisp.SOURCE
            and other.pfander_item == entry.pfander_item
            and other.source_options is not None
        ]
        for partner in partners:
            ours = measure((entry,), data)
            theirs = tisp.measure((partner,))
            rows.append(
                {
                    "pfander_item": entry.pfander_item,
                    "ccam_item": entry.source_item,
                    "ccam_mean": ours.mean,
                    "tisp_item": partner.source_item,
                    "tisp_mean": theirs.mean,
                    "abs_gap": abs(ours.mean - theirs.mean),
                }
            )
    return pd.DataFrame(rows)
