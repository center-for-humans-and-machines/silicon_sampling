"""Climate Change in the American Mind, recoded onto a study's own answer options.

The benchmark scores six moderators — gender, age band, race, education, income
and party.  Pfänder's recruitment quotas cover the first three; the other three
are whatever the respondents happen to be.  A base model asked to invent them
does badly enough to break an analysis: Qwen2.5-7B put 139 of 18,000 synthetic
respondents (0.77%) in the ``Less than $30,000`` bracket, which is that
moderator's dummy-coding reference level.  That is 18 of the 2,000 control
respondents and about 8 of each 1,000-respondent intervention arm, against the
benchmark's 30-respondent floor, so every income interaction in that run was
measured against a cell the benchmark would not have scored.  This module
supplies a real distribution instead: the same draw puts 2,321 respondents there,
264 of them in the control arm.

CCAM is the source because it is a nationally representative Ipsos KnowledgePanel
probability sample, raked by its own producers to Census/CPS/ACS margins on age,
gender, race, region, metro status, education *and* income, and because it ships
respondent-level microdata with the covariance structure intact.  What it costs
is a crosswalk: CCAM's education and income categories were written for a
different questionnaire and line up with no study's exactly.  Every mapping
decision lives in ``codebook``, one ``Codebook`` per study, because the crosswalk
— not the arithmetic — is where this package can be wrong, and because two
studies asking different education items should not have to share one set of
level strings.

**Waves.**  ``STRUCTURE_WAVES`` (Apr 2022 - Dec 2024, n = 6,191) supplies the
associations and ``LEVEL_WAVES`` (the two 2024 waves, n = 2,044) supplies the
levels.  The split exists because the two quantities need different things.  The
most recent wave alone is n = 1,013, which puts a median of 16 respondents in the
32 gender x age-band x race cells — far too thin to say anything about a
120-category education x income x party distribution inside them.  Pooling three
years fixes that at a median of 87 and a minimum of 44.  But levels drift, and
one of the drifts is mechanical: Pfänder's income brackets are fixed dollar
amounts, so nominal income growth alone moved the top bracket from 19.6% (pooled
2022-2024) to 21.7% (2024) to 23.3% (Dec 2024 alone).  Party moved too, by 1.1
points toward the Republicans — 26.7% to 27.8% — over the same window.  Taking
levels from calendar 2024 and associations from 2022-2024 keeps the recency where
it matters without paying for it in cell size.  Nothing older is pooled in: waves
before Nov 2016 use a coarser top income bracket, and 2020-2021 carries the
pandemic income shock.

**What CCAM cannot supply.**  Asian respondents are not separable — CCAM folds
them into "Other, Non-Hispanic" — and there is no third gender code.  Neither is
fatal here, because gender, age and race come from the quotas and CCAM is only
ever conditioned *on* them; see ``joint.py``.  ``UNFILLABLE_SLOTS`` names the
Pfänder items that stay generated because CCAM has no variable for them at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .codebook import (
    PARTY_REFUSED,
    PFANDER,
    Codebook,
    age_band,
    GENDER_FROM_CCAM,
    INCOME_BRACKETS,
    RACE_FROM_CCAM,
)

#: Repository root (``.../silicon_sampling``), found by walking up from this file.
ROOT = Path(__file__).resolve().parents[2]

CCAM_SAV = ROOT / "data" / "calibration" / "datasets" / "ccam.sav"
CCAM_CODEBOOK = ROOT / "data" / "calibration" / "codebooks" / "ccam_codebook.pdf"

#: Waves the association structure is estimated from: Apr 2022 - Dec 2024.
STRUCTURE_WAVES = (26, 27, 28, 29, 30, 31)

#: Waves the marginal levels are calibrated to: Apr 2024 and Dec 2024.
LEVEL_WAVES = (30, 31)

#: ``weight_aggregate / weight_wave``, smallest and largest over
#: ``STRUCTURE_WAVES``.  Constant inside a wave, not across them; see
#: ``donor_table`` for why the choice between the two weights is immaterial
#: anyway.
WEIGHT_RATIO_RANGE = (1.0498, 1.1322)

#: Largest total-variation distance between a per-cell conditional fitted on
#: ``weight_aggregate`` and the same cell fitted on ``weight_wave``.
WEIGHT_CHOICE_MAX_TV = 0.0029

#: Republican share in percent, ``STRUCTURE_WAVES`` pooled then ``LEVEL_WAVES``.
#: Quoted in the module docstring, and pinned by a test, because a drift figure
#: stated as a difference is easy to get wrong by a rounding step.
PARTY_DRIFT = (26.7, 27.8)

#: Top income bracket share in percent under ``PFANDER``: pooled 2022-2024,
#: calendar 2024, then Dec 2024 alone.  The mechanical drift that motivates
#: taking levels from the most recent year.
INCOME_TOP_DRIFT = (19.6, 21.7, 23.3)

#: Pfänder items CCAM has no variable for, so they stay model-generated.  Named
#: rather than described in prose because a claim about which slots cannot be
#: pre-filled is only useful if the slot ids are real; ``test_demographics``
#: checks each one against ``pfander.instrument``.
UNFILLABLE_SLOTS = ("social_class", "rural", "zip_code")

#: Pfänder items CCAM *could* fill with the same machinery, and the CCAM variable
#: each would come from.  Not wired up: they are not benchmark moderators, so they
#: buy fewer rejected draws rather than a better analysis.
FILLABLE_SLOTS = {
    "household": "house_size",
    "religion": "religion",
    "religion_bornagain": "evangelical",
    "religiosity": "service_attendance",
}


@lru_cache(maxsize=2)
def load(path: Path | str | None = None) -> pd.DataFrame:
    """The raw CCAM trend file, with the columns this module reads.

    ``pyreadstat`` is imported lazily: nothing in the sampling path needs it,
    because ``joint.py`` ships the fitted table as a CSV.
    """
    import pyreadstat

    frame, _ = pyreadstat.read_sav(str(path or CCAM_SAV))
    return frame


@lru_cache(maxsize=8)
def donor_table(
    book: Codebook = PFANDER,
    waves: Sequence[int] = STRUCTURE_WAVES,
    path: Path | str | None = None,
    weight: str = "weight_aggregate",
) -> pd.DataFrame:
    """One weighted row per (respondent, education level, income level, party level).

    A respondent whose CCAM category spans two of the study's levels appears more
    than once, with the design weight divided between the pieces, so a bracket
    split is carried as fractional weight rather than as a coin flip.  Splitting
    in expectation keeps the table deterministic and keeps the split out of the
    random-number stream, where it would otherwise have to be reproduced exactly
    to reproduce a run.

    The weight is ``weight_aggregate``, which the codebook prescribes for pooled
    waves.  Inside any single wave it is ``weight_wave`` times a constant, but the
    constant is not the same in every wave: it runs 1.0498 to 1.1322 over the six
    waves used here (``WEIGHT_RATIO_RANGE``), so the choice does reweight the
    waves relative to each other, by 7.9% end to end.  What makes it immaterial
    is *where* the weights enter.  ``joint.fit`` pins the drawn axes' marginals in
    a second pass, so refitting on ``weight_wave`` moves them by exactly zero; all
    the weight choice can move is the association structure, and there it shifts a
    per-cell conditional by at most 0.0029 in total variation
    (``WEIGHT_CHOICE_MAX_TV``, mean 0.0018).  So the prescribed weight is used and
    the alternative is a rounding error — but not for the reason a smaller spread
    would have given.

    ``weight`` names the CCAM weight column, and exists only so that the previous
    paragraph is a measurement rather than a claim: ``test_demographics`` refits on
    ``weight_wave`` and compares.  Nothing in the sampling path passes it.
    """
    frame = load(path)
    rows = frame[frame["wave"].isin(list(waves))]
    kept = rows[
        rows["income"].between(1, max(INCOME_BRACKETS))
        & rows["party"].isin(list(book.party_codes))
    ]
    columns = ["gender", "age_band", "race", *book.drawn, "weight"]
    records: list[tuple] = []
    for row in kept.itertuples():
        cell = (
            GENDER_FROM_CCAM[int(row.gender)],
            age_band(row.age, book.age_bands),
            RACE_FROM_CCAM[int(row.race)],
        )
        source = {"education": row.educ, "income": row.income, "party": row.party}
        splits = [book.shares(axis, source[axis]) for axis in book.drawn]
        for combination in _product(splits):
            levels = tuple(level for level, _ in combination)
            mass = float(getattr(row, weight))
            for _, share in combination:
                mass *= share
            records.append(cell + levels + (mass,))
    return pd.DataFrame(records, columns=columns)


def _product(splits: Sequence[Mapping[str, float]]):
    """Cartesian product of per-axis ``{level: share}`` maps, as (level, share) tuples."""
    import itertools

    return itertools.product(*(tuple(split.items()) for split in splits))


def dropped(
    book: Codebook = PFANDER,
    waves: Sequence[int] = STRUCTURE_WAVES,
    path: Path | str | None = None,
) -> dict[str, int]:
    """How many rows the crosswalk discards, and why."""
    frame = load(path)
    rows = frame[frame["wave"].isin(list(waves))]
    return {
        "rows": int(len(rows)),
        "income_out_of_range": int(
            (~rows["income"].between(1, max(INCOME_BRACKETS))).sum()
        ),
        "party_refused": int((rows["party"] == PARTY_REFUSED).sum()),
        "party_unmapped": int((~rows["party"].isin(list(book.party_codes))).sum()),
    }


def marginal(donors: pd.DataFrame, column: str, levels: Sequence[str]) -> pd.Series:
    """Weighted proportions of one column, in the study's level order."""
    total = donors.groupby(column)["weight"].sum()
    return (total.reindex(levels).fillna(0.0) / total.sum()).rename(column)


def level_targets(
    book: Codebook = PFANDER,
    waves: Sequence[int] = LEVEL_WAVES,
    path: Path | str | None = None,
) -> dict[str, pd.Series]:
    """The drawn axes' marginals to calibrate levels to."""
    donors = donor_table(book, tuple(waves), path)
    return {axis: marginal(donors, axis, book.levels(axis)) for axis in book.drawn}


def cell_sizes(donors: pd.DataFrame) -> pd.Series:
    """Unweighted donor rows per gender x age band x race cell."""
    return donors.groupby(["gender", "age_band", "race"]).size()


def weight_ratios(
    waves: Sequence[int] = STRUCTURE_WAVES, path: Path | str | None = None
) -> pd.Series:
    """``weight_aggregate / weight_wave`` per wave, which is what pins the docstring.

    A separate function rather than a comment because the claim it supports —
    that the two weight variables differ by a per-wave constant — is checkable and
    was once stated wrongly.
    """
    frame = load(path)
    rows = frame[frame["wave"].isin(list(waves))]
    ratio = rows["weight_aggregate"] / rows["weight_wave"]
    return ratio.groupby(rows["wave"]).mean()


def crosswalk_notes(book: Codebook = PFANDER) -> Mapping[str, Iterable[str]]:
    """The crosswalk, as lines a report or a CLI can print verbatim."""
    out: dict[str, list[str]] = {}
    if "education" in book.drawn:
        codes = sorted(set(book.education_from_ccam) | set(book.education_split))
        out["education"] = [
            _share_line("educ", code, book, "education") for code in codes
        ]
    if "income" in book.drawn:
        out["income"] = []
        for code in sorted(INCOME_BRACKETS):
            low, high = INCOME_BRACKETS[code]
            detail = _shares_detail(book.income_shares(code), places=4)
            flag = "  <- split" if len(book.income_shares(code)) > 1 else ""
            out["income"].append(
                f"income {code:>2d} (${low:,}-${high:,}) -> {detail}{flag}"
            )
    if "party" in book.drawn:
        out["party"] = [
            _share_line("party", code, book, "party") for code in book.party_codes
        ]
        out["party"].append(
            f"party {PARTY_REFUSED} -> dropped (no refusal level in {book.name})"
        )
    out["race"] = [
        f"{level} -> {target}" for level, target in book.race_to_ccam.items()
    ]
    out["race"] += [
        f"{alias} (alias) -> {level}" for alias, level in book.race_aliases.items()
    ]
    return out


def _shares_detail(shares: Mapping[str, float], places: int = 2) -> str:
    return " + ".join(f"{share:.{places}f} {level}" for level, share in shares.items())


def _share_line(label: str, code: int, book: Codebook, axis: str) -> str:
    shares = book.shares(axis, code)
    flag = "  <- split" if len(shares) > 1 else ""
    return f"{label} {code:>2d} -> {_shares_detail(shares)}{flag}"
