"""Goldwert's US advocacy megastudy, as levels for the two behavioural outcomes.

TISP and CCAM between them left four Pfänder outcomes with no candidate item at
all, and two of those were the behavioural ones: a $10 allocation to a named
organisation and a newsletter signup.  No general-population attitude survey
carries either, because both are experiments rather than questions.  Goldwert's
megastudy (n = 31,324, control arm n = 1,739) carries both, on **exactly Pfänder's
scales** — whole dollars out of ten, and 0/1 — with the same real-money mechanism
of paying out 100 randomly chosen participants.  On instrument identity this is
the closest match anywhere in the crosswalk.

It is still graded ``construct-only``, and the reasoning is worth stating in full
because the temptation to promote it is real.

**The donation carries a mode that Pfänder cannot have.**  Goldwert doubles the
pool if at least half of participants give $5 or more.  In the control arm, 29.6%
of respondents give precisely $5 — against 1.5% to 3.6% at each of $1, $2, $3, $4,
$6, $7, $8 and $9 — alongside 29.0% at $0 and 23.8% at $10.  The match manufactures
a mode at its own threshold.  Pfänder has no match, so that mode cannot exist
there, and the mass sitting in it is worth roughly half a dollar of the mean, which
is about the entire error budget the validation allows.  The recipient differs in
kind as well: an unnamed environmental advocacy organisation against the American
Meteorological Society, named and framed as a scientific society, which in the US
changes who is willing to give and does so in the opposite direction from the
match.

**The newsletter differs in friction, which is what sets signup rates.**  Goldwert
embeds an advocacy group's own form in the survey page; Pfänder links out to a
scientist's newsletter in a new tab and asks whether the respondent subscribed.
Goldwert's two embedded forms already differ by 2.4 points from each other
(350.org 0.243, Citizens' Climate Lobby 0.218), which is a floor on how much an
otherwise-identical item can move; a link-out is a far larger change than one
organisation for another.  ``newsletter1`` is the column to use, not ``newsletter``:
the latter is the OR of the two forms and mechanically exceeds either.

**The sample is not post-stratified.**  This is the axis that separates Goldwert
from the other two sources as a *level* source rather than an effect source, and it
is the authors' own position, not an inference: they write that despite the sample
approximating the US population on age, race, gender and ethnicity, "it is not a
truly representative sample, and might embed biases associated with online panel
samples", and vouch for between-condition comparisons.  There is no weight column
in the file, so :func:`measure` reports unweighted moments and says so.

**Two things that looked like problems and are not.**  Display order is randomised
across nine outcome blocks and the study finds five-to-nine-point position swings
on other items — ``march`` falls from 48 to 39 — but not on these two: within the
control arm the donation's cell means run 4.44 to 5.17 with no trend (slope −0.005
a position, first 4.70, last 4.74) against a cell standard error of 0.33, and
``newsletter1`` drifts by about 4.6 points across eight positions.  And the gap
between the arm's 1,739 assignments and the donation's 1,212 answers is early
dropout, not selection at the donation page: non-reachers have a mean ``Progress``
of 29.5 against 99.1, and among the 35 who answered the petition but not the
donation the petition rate differs by 4.9 points on n = 35, which is noise.  So the
level is a completer-sample level, which is ordinary, rather than a
donation-specific self-selection artefact.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..pfander.outcomes import SCALE_RANGE
from . import crosswalk as cw
from .scales import NOT_MEASURED, Weighted, composite, weighted_moments

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "calibration" / "datasets" / "goldwert_etal2026.csv"

SOURCE = "Goldwert"
#: The control arm's own label in the published file.
CONTROL = "Control"
#: Not post-stratified.  No weight exists in the file and the authors disclaim
#: level representativeness, so a level from here is a sample's, not a population's.
POST_STRATIFIED = False
PROVENANCE = (
    "Goldwert et al. 2026 advocacy megastudy, US quota-matched CloudResearch "
    "Connect panel, control arm n = 1,739; unweighted (no weight in the file, and "
    "the authors disclaim level representativeness)"
)


@lru_cache(maxsize=2)
def load(condition: str | None = CONTROL, path: str | None = None) -> pd.DataFrame:
    """The control arm with a unit ``weight`` column, cached.

    The weight is a column of ones rather than an absent column, so that
    :func:`~silicon_sampling.anchors.scales.weighted_moments` takes the same code
    path as it does for TISP and CCAM and the effective *n* it reports is the real
    *n*.  Faking a weight would be wrong; declaring that there isn't one, in the
    shape the rest of the package expects, is not.

    Only the control arm is loaded.  An anchor is a *control-arm* level, and the
    treated arms of an intervention megastudy are the one place a level is
    guaranteed not to be one.
    """
    frame = pd.read_csv(
        Path(path) if path else CSV, encoding="utf-8-sig", low_memory=False
    )
    if condition is not None:
        frame = frame[frame["condName"] == condition].copy()
    frame["weight"] = 1.0
    return frame.reset_index(drop=True)


def measure(
    entries: tuple[cw.Entry, ...],
    frame: pd.DataFrame | None = None,
    midpoint: bool = False,
) -> Weighted:
    """The level of one crosswalk group, on Pfänder's own scale.

    Every Goldwert entry is ``native``: the source item already lives on the
    target scale, so the identity is the transform and ``midpoint`` has nothing to
    undo — it is accepted and ignored so that the three source modules present one
    interface to :mod:`silicon_sampling.anchors.levels`.
    """
    entries = tuple(entry for entry in entries if entry.source == SOURCE)
    if not entries or any(entry.conversion == "none" for entry in entries):
        return NOT_MEASURED
    if any(entry.conversion != "native" for entry in entries):
        raise ValueError("Goldwert items are all native; nothing here converts a Likert")
    scales = {SCALE_RANGE[entry.pfander_outcome] for entry in entries}
    if len(scales) != 1:
        raise ValueError(f"mixed target scales in group: {sorted(scales)}")
    missing = tuple(sorted({code for entry in entries for code in entry.missing_codes}))
    data = load() if frame is None else frame
    values = composite(
        data,
        tuple(entry.source_item for entry in entries),
        missing_codes=missing,
        scale=scales.pop(),
        conversion="native",
    )
    return weighted_moments(values, data["weight"])


def donation_shape(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The control arm's donation distribution, which is the case against the anchor.

    Printed rather than described because the $5 mode is the whole argument: a
    reader who sees 29.6% on one interior dollar and under 4% on each of its
    neighbours does not need to be told that the group-contingent match is doing
    it.
    """
    data = load() if frame is None else frame
    values = pd.to_numeric(data["donation"], errors="coerce").dropna()
    share = values.value_counts(normalize=True).sort_index()
    return pd.DataFrame({"dollars": share.index, "share": share.to_numpy()})


def newsletter_variants(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The three signup columns side by side, to justify picking ``newsletter1``.

    ``newsletter`` is the OR of the two forms *and* zero-filled to every assigned
    row, so it is wrong twice over — once by combining organisations and once by
    scoring people who never reached the page as refusals.
    """
    data = load() if frame is None else frame
    rows = []
    for column, note in (
        ("newsletter1", "350.org, embedded form"),
        ("newsletter2", "Citizens' Climate Lobby, embedded form"),
        ("newsletter", "OR of the two, zero-filled to every assigned row"),
    ):
        values = pd.to_numeric(data[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "what": note,
                "n_observed": int(values.notna().sum()),
                "rate": float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def item_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every Goldwert crosswalk item on its own."""
    data = load() if frame is None else frame
    rows = []
    for entry in cw.CROSSWALK:
        if entry.source != SOURCE or entry.conversion == "none":
            continue
        measured = measure((entry,), data)
        rows.append(
            {
                "pfander_outcome": entry.pfander_outcome,
                "pfander_item": entry.pfander_item,
                "source_item": entry.source_item,
                "scale": SCALE_RANGE[entry.pfander_outcome],
                "grade": entry.grade,
                "mean": measured.mean,
                "sd": measured.sd,
                "se": measured.se,
                "n": measured.n,
            }
        )
    return pd.DataFrame(rows)
