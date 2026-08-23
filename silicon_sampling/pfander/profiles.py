"""Build the 18,000 respondent profiles the sampler walks through.

The preregistration publishes two *two-way* margins — age x gender and
race x gender (2024 Census PEP counts, rescaled to N = 18,000) — not the three-way
joint.  Fitting a joint to those two margins by iterative proportional fitting
converges to the closed form

    P(age, gender, race) = P(gender) . P(age | gender) . P(race | gender)

i.e. age and race independent *given* gender, which is the maximum-entropy joint
consistent with what was published.  That form is computed directly here; running
IPF would only rediscover it.

The other three benchmark moderators — education, income and party — are not in
the quotas, and letting the model invent them went badly: Qwen2.5-7B put 139 of
18,000 respondents (0.77%) in ``Less than $30,000``, which is that moderator's
dummy-coding reference level, so every income interaction was estimated against
18 control-arm respondents against the benchmark's floor of 30.  They are now
pre-filled too, from ``demographics.joint``: CCAM's
``P(education, income, party | gender, age band, race)``, calibrated to 2024
national levels.  The quota axes are untouched by that — this module still builds
the cells the way it always has, and the draw only fills in the axes CCAM
supplies — so the published gender x age and gender x race margins come out
exactly as before, cell for cell.

**Prefill is a property of the profiles file, not of this module.**  Three
finished or running Pfänder runs read a ``profiles.csv`` written before any of
this existed, and one of them is sampling right now.  So the drawn moderators are
written as three extra *columns*: a file that has them was built with prefill on
and its respondents are handed their education, income and party; a file that
does not is read exactly as it always was and its respondents generate them.  No
existing file changes meaning, and ``build(prefill=False)`` still reproduces one
byte for byte, because the draw runs on its own seed stream rather than on the
per-profile ``local`` one — drawing from ``local`` would shift every age,
consensus order and per-profile seed downstream of it.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..demographics import joint as demographics
from . import instrument
from .conditions import CONSENSUS_ORDERS, CONTROL_TEXTS, INTERVENTIONS
from .outcomes import age_band

#: Preregistration Table 3, age x gender.  (band, total, male, female)
AGE_QUOTA = (
    ("18-29", 3629, 1848, 1781),
    ("30-44", 4688, 2365, 2323),
    ("45-59", 4122, 2048, 2074),
    ("60+", 5561, 2566, 2995),
)

#: Preregistration Table 3, race x gender, in the survey's on-screen labels.
RACE_QUOTA = (
    ("Asian / Asian-American", 1201, 568, 633),
    ("Black / African-American", 2212, 1042, 1170),
    ("Latino / Hispanic", 3263, 1646, 1617),
    ("Other", 492, 240, 252),
    ("White / Caucasian", 10832, 5332, 5500),
)

TOTAL_N = 18000
CONTROL_N = 2000
PER_INTERVENTION_N = 1000

#: Age range each band spans.  The open-ended top band is capped at 89.
BAND_RANGE = {"18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59), "60+": (60, 89)}

#: Seed stream for the CCAM-drawn moderators.  Deliberately separate from the
#: per-profile ``local`` stream: drawing from ``local`` would shift every value
#: after it and stop the finished runs from reproducing their own profiles.csv.
DEMOGRAPHIC_SEED = 20260823


def demographic_answers(profile_id: str, gender: str, band: str, race: str) -> dict:
    """Education, income and party for one quota cell, from ``demographics.joint``.

    Keyed on the profile id rather than on a loop position, so a profile keeps its
    moderators when the file is rebuilt and ``read_csv`` can check a stored value
    against a fresh draw.  ``joint.Sampler.draw`` consumes exactly one float, so
    the result is a pure function of the id and the cell.
    """
    rng = random.Random(DEMOGRAPHIC_SEED * 1_000_003 + int(profile_id[1:]))
    return demographics.draw(gender, band, race, rng)


@dataclass
class Profile:
    """One synthetic respondent, before any generation happens."""

    profile_id: str
    condition: str
    code_name: str
    gender: str
    race: str
    age: int
    year_birth: int
    age_band: str
    control_text: str = ""
    consensus_order: str = ""
    post_order: str = ""
    seed: int = 0
    #: The three moderators CCAM supplies.  Empty on a profile built before
    #: prefill, which is how such a profile stays readable.
    education: str = ""
    income: str = ""
    party: str = ""
    #: Answers supplied rather than sampled.
    prefilled: dict = field(default_factory=dict, repr=False)

    @property
    def drawn(self) -> dict[str, str]:
        """The pre-filled moderators, or nothing when this profile has none."""
        values = {
            "education": self.education,
            "income": self.income,
            "party": self.party,
        }
        return {name: value for name, value in values.items() if value}


def _largest_remainder(weights: dict, total: int) -> dict:
    """Apportion ``total`` integer units across ``weights``, minimising rounding drift."""
    scale = sum(weights.values())
    exact = {key: value * total / scale for key, value in weights.items()}
    counts = {key: int(value) for key, value in exact.items()}
    short = total - sum(counts.values())
    for key in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[:short]:
        counts[key] += 1
    return counts


def cell_counts(total: int = TOTAL_N) -> dict[tuple[str, str, str], int]:
    """Counts per (gender, age band, race) cell under the max-entropy joint."""
    genders = ("Male", "Female")
    gender_totals = {
        "Male": sum(row[2] for row in AGE_QUOTA),
        "Female": sum(row[3] for row in AGE_QUOTA),
    }
    age_by_gender = {
        gender: {row[0]: row[2 + index] for row in AGE_QUOTA}
        for index, gender in enumerate(genders)
    }
    race_by_gender = {
        gender: {row[0]: row[2 + index] for row in RACE_QUOTA}
        for index, gender in enumerate(genders)
    }

    weights: dict[tuple[str, str, str], float] = {}
    for gender in genders:
        p_gender = gender_totals[gender] / sum(gender_totals.values())
        age_sum = sum(age_by_gender[gender].values())
        race_sum = sum(race_by_gender[gender].values())
        for band, age_count in age_by_gender[gender].items():
            for race, race_count in race_by_gender[gender].items():
                weights[(gender, band, race)] = (
                    p_gender * (age_count / age_sum) * (race_count / race_sum)
                )
    return _largest_remainder(weights, total)


def _age_weights(band: str) -> dict[int, float]:
    """Ages within a band.

    Uniform inside the closed bands.  The open-ended 60+ band declines linearly
    from 60 to 89, which is a rough but honest stand-in for the real US age
    structure; only ``year_birth`` depends on it, and ``age_band`` — the quantity
    the benchmark actually scores — is exact either way.
    """
    lo, hi = BAND_RANGE[band]
    if band != "60+":
        return {age: 1.0 for age in range(lo, hi + 1)}
    span = hi - lo
    return {age: 1.0 - 0.85 * (age - lo) / span for age in range(lo, hi + 1)}


def condition_slots() -> list[str]:
    """The 18,000 condition assignments, before shuffling."""
    return ["control"] * CONTROL_N + [
        title for title in INTERVENTIONS for _ in range(PER_INTERVENTION_N)
    ]


def build(
    total: int = TOTAL_N, seed: int = 20260814, prefill: bool = True
) -> list[Profile]:
    """Construct every profile, deterministically.

    ``prefill=False`` leaves education, income and party for the model to
    generate, which is what the finished runs did; the rest of the profile is
    identical either way, so the two differ only in whether the three extra
    columns are present.
    """
    rng = random.Random(seed)

    cells: list[tuple[str, str, str]] = []
    for cell, count in cell_counts(total).items():
        cells.extend([cell] * count)
    rng.shuffle(cells)

    assignments = condition_slots()
    if len(assignments) != total:  # pragma: no cover - only if the constants change
        raise ValueError(f"condition slots ({len(assignments)}) do not fill N={total}")
    rng.shuffle(assignments)

    from .build import (
        CODE_NAMES,
    )  # local import: build imports conditions, not profiles

    post_blocks = [block.key for block in instrument.POST_RANDOMISED]
    control_names = list(CONTROL_TEXTS)

    profiles: list[Profile] = []
    for index, ((gender, band, race), condition) in enumerate(
        zip(cells, assignments), start=1
    ):
        profile_id = f"p{index:05d}"
        drawn = demographic_answers(profile_id, gender, band, race) if prefill else {}
        local = random.Random(seed * 1_000_003 + index)
        weights = _age_weights(band)
        age = local.choices(list(weights), weights=list(weights.values()))[0]
        year = instrument.SURVEY_YEAR - age
        control_text = local.choice(control_names) if condition == "control" else ""
        order = list(post_blocks)
        local.shuffle(order)
        profiles.append(
            Profile(
                profile_id=profile_id,
                condition=condition,
                code_name=(
                    control_text if condition == "control" else CODE_NAMES[condition]
                ),
                gender=gender,
                race=race,
                age=age,
                year_birth=year,
                age_band=age_band(year),
                control_text=control_text,
                consensus_order="".join(
                    str(item) for item in local.choice(CONSENSUS_ORDERS)
                ),
                post_order="|".join(order),
                seed=local.randrange(2**31),
                **drawn,
                prefilled=prefilled_answers(gender, year, race, drawn),
            )
        )
    return profiles


def prefilled_answers(gender: str, year_birth: int, race: str, drawn: dict) -> dict:
    """Answers supplied rather than sampled.

    Consent, the AI screener and both attention checks are pre-filled as passed:
    every respondent in the human sample passed them, so sampling them would
    create a selection effect with no counterpart in the target data.  The quota
    axes are pre-filled because the quotas are the sample definition, and the
    three in ``drawn`` because CCAM knows their joint distribution and a base
    model does not.
    """
    return {
        "filter": "Yes",
        "filter_ai": "Yes",
        "gender": gender,
        "year_birth": year_birth,
        "race": race,
        "attention1": 3,
        "attention2": "attention",
        **drawn,
    }


#: Columns every ``profiles.csv`` has carried since the first run.
BASE_FIELDS = (
    "profile_id",
    "condition",
    "code_name",
    "gender",
    "race",
    "age",
    "year_birth",
    "age_band",
    "control_text",
    "consensus_order",
    "post_order",
    "seed",
)

#: The CCAM-drawn moderators.  Written only when the profiles carry them, so a
#: file's header is what distinguishes a prefilled run from an older one.
DEMOGRAPHIC_FIELDS = ("education", "income", "party")

FIELDS = BASE_FIELDS + DEMOGRAPHIC_FIELDS


def fieldnames(profiles: Sequence[Profile]) -> tuple[str, ...]:
    """The columns a set of profiles writes: the three extras only if they exist."""
    if any(profile.drawn for profile in profiles):
        return FIELDS
    return BASE_FIELDS


def write_csv(profiles: Sequence[Profile], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames(profiles)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for profile in profiles:
            row = asdict(profile)
            row.pop("prefilled")
            writer.writerow(row)


def read_csv(path) -> list[Profile]:
    """Load profiles, pre-filling the CCAM moderators only if the file has them.

    This is the whole compatibility story.  ``education``/``income``/``party`` are
    read with ``.get`` and default to empty, so the three ``profiles.csv`` files
    written before prefill existed load with exactly the answers they always had
    and their respondents keep generating those three items.  A run is therefore
    identified by its own profiles file rather than by the version of this module
    that happens to be checked out.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    profiles = []
    for row in rows:
        profile = Profile(
            profile_id=row["profile_id"],
            condition=row["condition"],
            code_name=row["code_name"],
            gender=row["gender"],
            race=row["race"],
            age=int(row["age"]),
            year_birth=int(row["year_birth"]),
            age_band=row["age_band"],
            control_text=row["control_text"],
            consensus_order=row["consensus_order"],
            post_order=row["post_order"],
            seed=int(row["seed"]),
            **{name: row.get(name, "") for name in DEMOGRAPHIC_FIELDS},
        )
        profile.prefilled = prefilled_answers(
            profile.gender, profile.year_birth, profile.race, profile.drawn
        )
        profiles.append(profile)
    return profiles
