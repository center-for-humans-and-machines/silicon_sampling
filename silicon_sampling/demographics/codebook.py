"""What one study's demographic items are called, and how CCAM maps onto them.

CCAM is the donor for every study in this repository, but no two studies code
their demographics the same way: Pfänder asks six moderators with a forced
four-option party stem and five fixed-dollar income brackets, Voelkel asks four
education levels, a three-option party item and no income at all.  The fitting
machinery in ``joint`` is identical for both — the only thing that differs is the
vocabulary.  So the vocabulary is what gets named here, once per study, and
``ccam`` and ``joint`` take a ``Codebook`` rather than hard-coding one study's
strings.

The tradeoff this buys is worth stating.  A single hard-coded codebook is shorter
and its crosswalk can be documented inline next to the mapping it describes,
which is how this module started life.  The cost is that a second study cannot
reuse any of it: its level strings would fall through ``.get()`` lookups and
silently produce national-average draws instead of conditional ones, which is a
failure that looks like a working sampler.  Naming the vocabulary explicitly
makes an unrecognised level an error instead — see ``Codebook.collapsible``, which
is the short, deliberate list of levels that *are* allowed to fall back.

**A codebook is a claim about coding, not about levels.**  Which CCAM code maps
to which of a study's answer options is a fact about two questionnaires and
belongs here.  How common each option is in the study's own sample is a fact
about that sample, drifts with the calendar, and is passed to ``joint.fit`` as a
target instead; that is why ``PFANDER`` carries no marginals even though its
income brackets are pinned to the dollar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

#: CCAM's own race categories, and the only ones its trend file distinguishes.
#: Asian, Native American and multiracial respondents share ``Other,
#: Non-Hispanic``; a study that separates them has to fold them back together.
CCAM_RACE = (
    "White, Non-Hispanic",
    "Black, Non-Hispanic",
    "Other, Non-Hispanic",
    "Hispanic",
)

#: CCAM ``gender`` has two codes.  A study with a third option declares it
#: collapsible rather than pretending CCAM can condition on it.
CCAM_GENDER = ("Male", "Female")

GENDER_FROM_CCAM = {1: "Male", 2: "Female"}

RACE_FROM_CCAM = {
    1: "White, Non-Hispanic",
    2: "Black, Non-Hispanic",
    3: "Other, Non-Hispanic",
    4: "Hispanic",
    # Labelled but never used in the trend file; the Fall 2024 report splits it
    # out and folds it into code 3, so we do the same.
    5: "Other, Non-Hispanic",
}

#: CCAM ``income`` code -> (lower, upper) dollar bound of the bracket.
#:
#: The two open-ended brackets are closed off arbitrarily.  Whether that is safe
#: depends on the study: it is safe exactly when no cut point falls inside them,
#: which ``Codebook.check`` asserts per codebook rather than once globally.
INCOME_BRACKETS = {
    1: (1_000, 5_000),
    2: (5_000, 7_500),
    3: (7_500, 10_000),
    4: (10_000, 12_500),
    5: (12_500, 15_000),
    6: (15_000, 20_000),
    7: (20_000, 25_000),
    8: (25_000, 30_000),
    9: (30_000, 35_000),
    10: (35_000, 40_000),
    11: (40_000, 50_000),
    12: (50_000, 60_000),
    13: (60_000, 75_000),
    14: (75_000, 85_000),
    15: (85_000, 100_000),
    16: (100_000, 125_000),
    17: (125_000, 150_000),
    18: (150_000, 175_000),
    19: (175_000, 200_000),
    20: (200_000, 250_000),
    21: (250_000, 400_000),
}

#: Party refusals (code -1) are dropped rather than mapped.  No study here has a
#: refusal level, and it is 0.7% of the pooled window.
PARTY_REFUSED = -1

#: CCAM's "No party/not interested in politics" code.  Every study here asks a
#: forced-choice party stem with no such option, so this code always has to be
#: divided across levels that do exist; see ``Codebook.party_split``.
PARTY_NO_PARTY = 5

#: The four age bands the benchmark scores.  CCAM records exact age, so these are
#: cut from ``age`` rather than crosswalked, and every study here uses the same
#: four.
AGE_BANDS = ("18-29", "30-44", "45-59", "60+")

#: Lower edges of the age bands, parallel to ``AGE_BANDS``.
AGE_BAND_EDGES = (18, 30, 45, 60)

_EMPTY: Mapping = MappingProxyType({})


def age_band(age: float, bands: tuple[str, ...] = AGE_BANDS) -> str:
    """The band an exact age falls in."""
    position = sum(1 for edge in AGE_BAND_EDGES[1:] if age >= edge)
    return bands[position]


@dataclass(frozen=True, eq=False)
class Codebook:
    """One study's demographic vocabulary, and CCAM's route into it.

    Compared and hashed by identity, not by value: the codebooks are module-level
    singletons, they carry unhashable mapping fields, and the fitting path caches
    on them.  Two codebooks with the same contents are two codebooks.

    ``gender``, ``age_bands`` and ``race`` are the axes the joint conditions *on*
    and are therefore given in CCAM's resolution, not the study's: a study level
    CCAM cannot represent goes in ``race_to_ccam`` if it folds onto a CCAM
    category and in ``collapsible`` if it does not.  ``education``, ``income`` and
    ``party`` are the axes the joint *draws*, in the study's own strings and in
    the study's own order, and an empty tuple means the study never asks.
    """

    name: str
    education: tuple[str, ...]
    education_from_ccam: Mapping[int, str]
    party: tuple[str, ...]
    party_from_ccam: Mapping[int, str]
    #: CCAM codes that straddle two of the study's levels, and how they divide.
    education_split: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    party_split: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    #: Income levels and the lower dollar edge of every level but the first.
    #: Splits are derived from ``INCOME_BRACKETS`` rather than declared.
    income: tuple[str, ...] = ()
    income_cuts: tuple[int, ...] = ()
    #: The study's race levels, and which CCAM category each folds onto.
    race_to_ccam: Mapping[str, str] = field(default_factory=dict)
    #: Extra spellings of a race level that the sampler should accept, e.g. the
    #: survey's on-screen hyphenation against the submission's.
    race_aliases: Mapping[str, str] = field(default_factory=dict)
    #: ``(axis, level)`` pairs the sampler may answer with the population average
    #: instead of a conditional draw, because CCAM has no such level.  Keep this
    #: list short and deliberate: everything not on it raises.
    collapsible: frozenset[tuple[str, str]] = frozenset()
    gender: tuple[str, ...] = CCAM_GENDER
    age_bands: tuple[str, ...] = AGE_BANDS
    #: Filename the fitted table ships under.  Defaults to ``<name>_joint.csv``;
    #: Pfänder's is pinned to the name it shipped under before this package
    #: served more than one study, so that file is rewritten rather than orphaned.
    table_file: str = ""

    def __post_init__(self) -> None:
        self.check()

    # -- shape ------------------------------------------------------------ #

    @property
    def given(self) -> tuple[str, ...]:
        """Axes the caller supplies: the ones a quota or a screener already fixed."""
        return ("gender", "age_band", "race")

    @property
    def drawn(self) -> tuple[str, ...]:
        """Axes CCAM supplies, in table order, skipping items the study never asks."""
        return tuple(
            axis for axis in ("education", "income", "party") if self.levels(axis)
        )

    @property
    def axes(self) -> tuple[str, ...]:
        return self.given + self.drawn

    def levels(self, axis: str) -> tuple[str, ...]:
        if axis == "gender":
            return self.gender
        if axis == "age_band":
            return self.age_bands
        if axis == "race":
            return CCAM_RACE
        return {
            "education": self.education,
            "income": self.income,
            "party": self.party,
        }[axis]

    @property
    def table_name(self) -> str:
        """Filename the fitted table for this study ships under."""
        return self.table_file or f"{self.name}_joint.csv"

    # -- the crosswalk ---------------------------------------------------- #

    def race(self, level: str) -> str:
        """The CCAM race category a study race level folds onto."""
        resolved = self.race_aliases.get(level, level)
        if resolved in CCAM_RACE:
            return resolved
        return self.race_to_ccam[resolved]

    def education_shares(self, code: int) -> Mapping[str, float]:
        """How one CCAM ``educ`` code divides across this study's levels."""
        code = int(code)
        if code in self.education_split:
            return self.education_split[code]
        return {self.education_from_ccam[code]: 1.0}

    def party_shares(self, code: int) -> Mapping[str, float]:
        """How one CCAM ``party`` code divides across this study's levels."""
        code = int(code)
        if code in self.party_split:
            return self.party_split[code]
        return {self.party_from_ccam[code]: 1.0}

    def income_shares(self, code: int) -> Mapping[str, float]:
        """How one CCAM income bracket divides across this study's levels.

        A bracket that sits wholly inside one level maps to it.  A bracket that
        straddles a cut point is divided in proportion to the log-width of each
        piece, i.e. assuming income is locally log-uniform within a bracket —
        the same assumption as a locally Pareto tail.  It is deliberately a
        *rule* rather than a judgement call per bracket: for Pfänder it gives
        62.2/37.8 on the $50-59,999 bracket and 73.5/26.5 on the $150-174,999
        one, against 60/40 and 72/28 for a flat within-bracket density, and the
        choice between the two moves the income marginal by at most 0.15 points.
        Having one rule matters more than which rule it is.
        """
        import math

        if not self.income:
            return _EMPTY
        low, high = INCOME_BRACKETS[int(code)]
        edges = [low] + [cut for cut in self.income_cuts if low < cut < high] + [high]
        span = math.log(high / low)
        shares: dict[str, float] = {}
        for lower, upper in zip(edges, edges[1:]):
            level = self.income[sum(1 for cut in self.income_cuts if cut <= lower)]
            shares[level] = shares.get(level, 0.0) + math.log(upper / lower) / span
        return shares

    def shares(self, axis: str, code: int) -> Mapping[str, float]:
        """Dispatch to the crosswalk for one drawn axis."""
        return {
            "education": self.education_shares,
            "income": self.income_shares,
            "party": self.party_shares,
        }[axis](code)

    @property
    def split_codes(self) -> tuple[tuple[str, int], ...]:
        """Every ``(axis, CCAM code)`` that spans two of this study's levels.

        Counted from the crosswalk rather than asserted in prose, because the
        count is easy to state wrongly and a reader uses it to decide how much of
        the result rests on a judgement call.
        """
        found: list[tuple[str, int]] = []
        for axis in self.drawn:
            codes = (
                sorted(INCOME_BRACKETS)
                if axis == "income"
                else sorted(
                    set(
                        self.education_from_ccam
                        if axis == "education"
                        else self.party_from_ccam
                    )
                    | set(
                        self.education_split
                        if axis == "education"
                        else self.party_split
                    )
                )
            )
            found += [
                (axis, code) for code in codes if len(self.shares(axis, code)) > 1
            ]
        return tuple(found)

    #: Every CCAM ``party`` code this codebook accepts.
    @property
    def party_codes(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.party_from_ccam) | set(self.party_split)))

    # -- self-consistency ------------------------------------------------- #

    def check(self) -> None:
        """Refuse a codebook whose crosswalk cannot be right.

        Cheap and total: every crosswalk target has to be a declared level, every
        split has to sum to one, and — the one that is genuinely study-specific —
        no income cut point may fall inside an open-ended CCAM bracket, because
        those brackets' outer bounds are invented and would then enter the
        arithmetic.
        """
        for axis in self.drawn:
            levels = set(self.levels(axis))
            codes = sorted(
                set(INCOME_BRACKETS)
                if axis == "income"
                else set(
                    self.education_from_ccam
                    if axis == "education"
                    else self.party_from_ccam
                )
                | set(self.education_split if axis == "education" else self.party_split)
            )
            for code in codes:
                shares = self.shares(axis, code)
                unknown = set(shares) - levels
                if unknown:
                    raise ValueError(
                        f"{self.name}/{axis}: code {code} maps to unknown level(s) {sorted(unknown)}"
                    )
                if abs(sum(shares.values()) - 1.0) > 1e-9:
                    raise ValueError(
                        f"{self.name}/{axis}: code {code} shares sum to {sum(shares.values())}"
                    )
        for code, (low, high) in INCOME_BRACKETS.items():
            if not 0 < low < high:
                raise ValueError(f"income bracket {code} is not a positive interval")
        for code in (min(INCOME_BRACKETS), max(INCOME_BRACKETS)):
            low, high = INCOME_BRACKETS[code]
            if any(low < cut < high for cut in self.income_cuts):
                raise ValueError(
                    f"{self.name}: open-ended income bracket {code} straddles a cut point"
                )
        for level, category in self.race_to_ccam.items():
            if category not in CCAM_RACE:
                raise ValueError(f"{self.name}: race {level!r} -> unknown {category!r}")
        for axis, _ in self.collapsible:
            if axis not in self.axes:
                raise ValueError(
                    f"{self.name}: collapsible axis {axis!r} is not an axis"
                )


# --------------------------------------------------------------------------- #
# Pfänder: the megastudy this repository is scored on
# --------------------------------------------------------------------------- #

#: CCAM ``educ`` codes 1-8 run "No formal education" through "12th grade, no
#: diploma": all of them are less than high school.  Code 9 is the GED/diploma
#: line, 10 and 11 are "some college, no degree" and "Associate's degree" —
#: Pfänder merges exactly those two — and 12 and 13 are the bachelor's and
#: master's lines.  Code 14, "Professional or Doctorate degree", is the only
#: education code that spans two Pfänder levels; see ``PROFESSIONAL_SHARE``.
#:
#: Note that ``educ_category``, CCAM's own four-level collapse, is unusable from
#: wave 24 on — its "High school" code is never emitted and the "<HS" code
#: absorbs both groups — so the mapping is rebuilt from raw ``educ`` instead.
_PFANDER_EDUCATION_FROM_CCAM = {
    **{code: "Less than high school" for code in range(1, 9)},
    9: "High school diploma / GED",
    10: "Some college or Associate's degree",
    11: "Some college or Associate's degree",
    12: "Bachelor's degree",
    13: "Master's degree / Professional degree",
}

#: CCAM code 14 merges professional and doctoral degrees; Pfänder separates them,
#: putting professional degrees with master's.  The Census Bureau's
#: educational-attainment tables put professional-degree holders at roughly 1.6%
#: of adults 25 and over against 1.5% for doctorates, so a little over half of
#: the merged category belongs on the master's side.
#:
#: This is the module's least defensible number: unlike the party and income
#: splits it is not read off any file in this repository, and it alone determines
#: the ``Doctorate degree / Ph.D.`` level.  What bounds the damage is that the
#: whole merged category is 4.2% of adults, so the plausible range 0.50-0.60
#: moves that level only between 2.3% and 1.9% of the sample — comfortably clear
#: of the benchmark's 30-respondent scoring floor either way.  ``test_demographics``
#: pins the value so a change to it cannot pass unnoticed.
PROFESSIONAL_SHARE = 0.52

#: Pfänder asks the ANES/Gallup party stem ("...a Republican, a Democrat, an
#: Independent, or what?") with four forced options and no way to decline, so
#: CCAM's "No party/not interested in politics" group (9.9%) has to go somewhere.
#: Most of it goes to Independent, and CCAM says so twice over.  Gallup, asking
#: that same stem without a no-party option, returns about 43% Independent
#: against CCAM's 29% — a gap larger than CCAM's no-party group.  And among CCAM
#: respondents who already declined both major parties, Independent beats Other
#: 94.1 to 5.9, which is the split used here: the same proportional division the
#: education and income crosswalks use, with the proportion read off the data
#: rather than chosen.
#:
#: The choice has teeth.  Sending the whole no-party group to Independent puts
#: "Other" at 1.6%, which is 32 respondents in the 2,000-person control arm —
#: essentially the benchmark's 30-respondent floor, so a Monte Carlo wobble would
#: decide whether that level is scored at all.  Sending them all to Other puts
#: Independent at 28% and "Other" at 11%, which no forced-choice survey of this
#: stem has ever reported.
NO_PARTY_SHARES = {"Independent": 0.94, "Other": 0.06}

PFANDER = Codebook(
    name="pfander",
    education=(
        "Less than high school",
        "High school diploma / GED",
        "Some college or Associate's degree",
        "Bachelor's degree",
        "Master's degree / Professional degree",
        "Doctorate degree / Ph.D.",
    ),
    education_from_ccam=_PFANDER_EDUCATION_FROM_CCAM,
    education_split={
        14: {
            "Master's degree / Professional degree": PROFESSIONAL_SHARE,
            "Doctorate degree / Ph.D.": 1.0 - PROFESSIONAL_SHARE,
        }
    },
    income=(
        "Less than $30,000",
        "$30,000 to $55,999",
        "$56,000 to $99,999",
        "$100,000 to $167,999",
        "$168,000 or more",
    ),
    income_cuts=(30_000, 56_000, 100_000, 168_000),
    party=("Republican", "Democrat", "Independent", "Other"),
    party_from_ccam={1: "Republican", 2: "Democrat", 3: "Independent", 4: "Other"},
    party_split={PARTY_NO_PARTY: NO_PARTY_SHARES},
    #: Asian respondents have no CCAM category of their own, so an Asian and an
    #: "Other" synthetic respondent draw from the same conditional.  That is a
    #: real loss — Asian Americans hold bachelor's degrees at roughly twice the
    #: rate of the "Other" residual — but the residual is Asian-dominated, so the
    #: error runs mostly the other way: "Other" respondents come out slightly
    #: better educated and better paid than they should be.  Together they are
    #: 9.4% of the quota, and the alternative (inventing an Asian cell from an
    #: outside source whose covariance with party we do not have) would be worse.
    race_to_ccam={
        "White / Caucasian": "White, Non-Hispanic",
        "Black / African American": "Black, Non-Hispanic",
        "Hispanic / Latino": "Hispanic",
        "Asian / Asian American": "Other, Non-Hispanic",
        "Other": "Other, Non-Hispanic",
    },
    #: The survey's on-screen race labels hyphenate differently from the
    #: submission's, and profiles carry the on-screen spelling.
    race_aliases={
        "Black / African-American": "Black / African American",
        "Latino / Hispanic": "Hispanic / Latino",
        "Asian / Asian-American": "Asian / Asian American",
    },
    #: The quotas never produce ``gender="Other"``, but the instrument allows it,
    #: so a transcript replayed from a hand-written profile has to work.
    collapsible=frozenset({("gender", "Other")}),
    table_file="us_joint.csv",
)


# --------------------------------------------------------------------------- #
# Voelkel: the validation study
# --------------------------------------------------------------------------- #

#: Voelkel collapses attainment to four levels, none of which splits a CCAM code:
#: everything up to a diploma is "HS or less", CCAM's two some-college codes are
#: "Some college", and master's, professional and doctoral degrees are all
#: "Postgraduate".  So this crosswalk carries no judgement call at all — which
#: also means ``PROFESSIONAL_SHARE`` never enters a Voelkel number.
_VOELKEL_EDUCATION_FROM_CCAM = {
    **{code: "HS or less" for code in range(1, 10)},
    10: "Some college",
    11: "Some college",
    12: "Bachelor",
    13: "Postgraduate",
    14: "Postgraduate",
}

VOELKEL = Codebook(
    name="voelkel",
    education=("HS or less", "Some college", "Bachelor", "Postgraduate"),
    education_from_ccam=_VOELKEL_EDUCATION_FROM_CCAM,
    #: Voelkel's ``Party_Gen`` offers three options and screened out
    #: non-leaning independents, so it has no "Other" level.  CCAM's "Other"
    #: partisans and its no-party group both land on Independent — the only
    #: level that can hold a respondent who named neither major party.  The
    #: collapse costs nothing in levels, which stage two pins to Voelkel's own
    #: published marginals; it only pools those respondents' association
    #: structure with the Independents', which is where they would sit in a
    #: three-option item anyway.
    party=("Republican", "Democrat", "Independent"),
    party_from_ccam={
        1: "Republican",
        2: "Democrat",
        3: "Independent",
        4: "Independent",
        PARTY_NO_PARTY: "Independent",
    },
    #: Voelkel reports no income item, so the joint has five axes rather than six.
    race_to_ccam={
        "White": "White, Non-Hispanic",
        "Black": "Black, Non-Hispanic",
        "Hispanic": "Hispanic",
        "Asian": "Other, Non-Hispanic",
        "Other": "Other, Non-Hispanic",
    },
    race_aliases={
        "White / Caucasian": "White",
        "Black / African American": "Black",
        "Hispanic / Latino": "Hispanic",
        "Asian / Asian American": "Asian",
    },
    #: 0.39% of Voelkel's sample answered the gender item with a third option
    #: CCAM has no counterpart for; those respondents draw the population
    #: average over gender rather than a conditional.
    collapsible=frozenset({("gender", "Other")}),
)

STUDIES = {book.name: book for book in (PFANDER, VOELKEL)}


def study(name: str) -> Codebook:
    """Look a codebook up by study name, with a legible error for a typo."""
    try:
        return STUDIES[name]
    except KeyError:
        raise ValueError(
            f"no codebook for {name!r}; known studies: {sorted(STUDIES)}"
        ) from None
