"""Silicon sample of the International Climate Psychology Collaboration.

The second effect-calibration study, and the one that makes cross-study
validation possible: 63 countries, twelve arms, four outcomes, and — unlike the
Strengthening Democracy Challenge — a climate topic and an outcome battery that
overlap the Pfänder megastudy's directly.

The study was published twice from a single data collection.  Doell et al. (2024,
*Scientific Data*) is the raw 668-column export; Vlasceanu et al. (2024, *Science
Advances*) is the cleaned 28-column analysis extract over the same respondents.
This package treats the raw export as primary and the cleaned extract as a check
on the outcome construction, because only the raw file carries the intervention-
internal items and the WEPT page-by-page record.
"""

from __future__ import annotations

__all__ = [
    "convert",
    "export",
    "instrument",
    "outcomes",
    "paths",
    "profiles",
    "run",
    "score",
    "templates",
]
