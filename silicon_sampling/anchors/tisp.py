"""The 2023 TISP survey's US sample, as ground-truth levels for Pfänder's items.

TISP (Trust in Science and Science-Related Populism, Cologna et al. 2025) fielded
the same four-facet trust battery Pfänder uses as its primary outcome — Besley's
competence / integrity / benevolence / openness triads — to a post-stratified US
sample of 2,559 respondents in February and March 2023.  Eleven of the twelve
stems match Pfänder's near-verbatim.  That makes this the single most valuable
external anchor available to the project, because Pfänder publishes no human data
and its primary outcome is otherwise unfalsifiable.

Three things about the file are worth knowing before trusting a number out of it.

**It is semicolon-separated with comma decimals.**  A default ``read_csv`` fails
on line 3 of 69,534, and reading ``WEIGHT_CNTRY`` without the decimal swap yields
a column of strings that silently becomes an unweighted mean.  Both are handled
here so no caller has to remember.

**The weight is per country.**  ``WEIGHT_CNTRY`` post-stratifies within country
and averages to 1 inside the US subsample, which is what we want; the global and
sample-size weights are for cross-country work and would be wrong here.  The US
rows are all ``UserLanguage == 'EN'`` and all complete (``Progress == 100``), so
the filter is the country code and nothing else.

**The referent is scientists, not climate scientists.**  Every trust item asks
about "most scientists".  Pfänder asks about "most climate scientists", and in the
US that is not a cosmetic difference.  TISP can size it from within: the same
respondents rated confidence in scientists (``TRUST_PEW``) and trust in climate
scientists (``CLIM_TRUST``), and :func:`referent_gap` returns the weighted paired
difference — 3.9 points on the converted 0-100 scale, SE 0.5, with the two items
correlated at 0.70.  That is a confounded estimate, because the two stems differ
in more than their referent, but it is the only one the data can give, and
ignoring a measured 3.9-point shift is a worse decision than correcting by it.
:mod:`silicon_sampling.anchors.levels` applies it to the trust battery only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .. import paths as _paths

import pandas as pd

from . import crosswalk as cw
from .scales import NOT_MEASURED, Weighted, composite, weighted_moments

ROOT = Path(__file__).resolve().parents[2]
CSV = _paths.resolve("calibration", "datasets") / "cologna_etal2025.csv"
CODEBOOK = _paths.resolve("calibration", "codebooks") / "cologna_etal2025_qs.docx"

SOURCE = "TISP"
COUNTRY = "USA"
WEIGHT = "WEIGHT_CNTRY"
#: Post-stratified: ``WEIGHT_CNTRY`` reweights the respondents to population
#: margins, so a level from here estimates a population level rather than a
#: sample's.  Read by :mod:`silicon_sampling.anchors.levels` as a tie-break.
POST_STRATIFIED = True
#: What the report has to state alongside any number from here.
PROVENANCE = "TISP (Cologna et al. 2025), US sample, fielded 2023-02-07 to 2023-03-08"

#: The four facets, in Pfänder's order, as source items.
FACETS = {
    "competence": ("TRUST_SCI_expert", "TRUST_SCI_intellig", "TRUST_SCI_qualified"),
    "integrity": ("TRUST_SCI_honest", "TRUST_SCI_ethical", "TRUST_SCI_sincere"),
    "benevolence": ("TRUST_SCI_concerned", "TRUST_SCI_improve", "TRUST_SCI_otherint"),
    "openness": ("TRUST_SCI_open", "TRUST_SCI_trans", "TRUST_SCI_otherviews"),
}


@lru_cache(maxsize=4)
def load(path: str | None = None, country: str = COUNTRY) -> pd.DataFrame:
    """The country's rows with a numeric ``weight`` column, cached.

    Cached because the full file is 69,534 rows by 141 columns and every anchor in
    the crosswalk reads it; the frame is treated as read-only by everything here.
    """
    frame = pd.read_csv(
        Path(path) if path else CSV,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame = frame[frame["COUNTRY_CODE"] == country].copy()
    frame["weight"] = pd.to_numeric(
        frame[WEIGHT].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    return frame.reset_index(drop=True)


def measure(
    entries: tuple[cw.Entry, ...],
    frame: pd.DataFrame | None = None,
    midpoint: bool = False,
) -> Weighted:
    """The weighted level of one crosswalk group, converted to Pfänder's scale.

    Every entry in a group must share the source's number of response options —
    they are components of one composite, and averaging items converted from
    different granularities would mix two different conversion assumptions into a
    single number without saying so.
    """
    entries = tuple(entry for entry in entries if entry.source == SOURCE)
    if not entries or any(entry.source_options is None for entry in entries):
        return NOT_MEASURED
    options = {entry.source_options for entry in entries}
    if len(options) != 1:
        raise ValueError(f"mixed response granularity in group: {sorted(options)}")
    missing = tuple(sorted({code for entry in entries for code in entry.missing_codes}))
    data = load() if frame is None else frame
    items = tuple(entry.source_item for entry in entries)
    # A battery carrying a "not applicable" option loses a quarter of the sample
    # to listwise deletion, so those groups score whoever answered at least half.
    min_answered = max(1, len(items) // 2) if missing else None
    values = composite(
        data,
        items,
        options=options.pop(),
        missing_codes=missing,
        min_answered=min_answered,
        midpoint=midpoint,
    )
    return weighted_moments(values, data["weight"])


def referent_gap(frame: pd.DataFrame | None = None) -> Weighted:
    """How much less the US trusts *climate* scientists than scientists, in points.

    The paired within-respondent difference ``TRUST_PEW - CLIM_TRUST`` on the
    converted scale.  Positive means the climate referent scores lower.  Both items
    are five-point, so the conversion is the same on each side and cancels out of
    the difference except through the endpoints.
    """
    data = load() if frame is None else frame
    general = composite(data, ("TRUST_PEW",), options=5)
    climate = composite(data, ("CLIM_TRUST",), options=5)
    return weighted_moments(general - climate, data["weight"])


def facet_levels(frame: pd.DataFrame | None = None) -> dict[str, float]:
    """The four subscale means, unadjusted.

    Pfänder's primary outcome is the mean of these four, so anchoring the
    composite alone leaves the facets wherever the sampler put them.  A caller
    that wants the facets right as well needs these; the spread across them is 12
    points, which is far too large to treat as noise.
    """
    data = load() if frame is None else frame
    return {
        facet: weighted_moments(composite(data, items, options=5), data["weight"]).mean
        for facet, items in FACETS.items()
    }


def item_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every TISP crosswalk item on its own, both conversions side by side."""
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
