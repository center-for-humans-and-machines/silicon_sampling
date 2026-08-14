"""Build the 18,000 respondent profiles the sampler walks through.

The preregistration publishes two *two-way* margins — age x gender and
race x gender (2024 Census PEP counts, rescaled to N = 18,000) — not the three-way
joint.  Fitting a joint to those two margins by iterative proportional fitting
converges to the closed form

    P(age, gender, race) = P(gender) . P(age | gender) . P(race | gender)

i.e. age and race independent *given* gender, which is the maximum-entropy joint
consistent with what was published.  That form is computed directly here; running
IPF would only rediscover it.

Only gender, age and race are pre-filled.  Education, income and party are also
benchmark moderators, but the task fixes the pre-filled set to what the
preregistration quotas cover, so the model generates those three and their
marginals become a diagnostic rather than an input.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

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
    #: Answers supplied rather than sampled.
    prefilled: dict = field(default_factory=dict, repr=False)


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


def build(total: int = TOTAL_N, seed: int = 20260814) -> list[Profile]:
    """Construct every profile, deterministically."""
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
        local = random.Random(seed * 1_000_003 + index)
        weights = _age_weights(band)
        age = local.choices(list(weights), weights=list(weights.values()))[0]
        year = instrument.SURVEY_YEAR - age
        control_text = local.choice(control_names) if condition == "control" else ""
        order = list(post_blocks)
        local.shuffle(order)
        profiles.append(
            Profile(
                profile_id=f"p{index:05d}",
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
                prefilled={
                    "filter": "Yes",
                    "filter_ai": "Yes",
                    "gender": gender,
                    "year_birth": year,
                    "race": race,
                    # Every respondent in the human sample passed both checks.
                    "attention1": 3,
                    "attention2": "attention",
                },
            )
        )
    return profiles


FIELDS = (
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


def write_csv(profiles: Sequence[Profile], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for profile in profiles:
            row = asdict(profile)
            row.pop("prefilled")
            writer.writerow(row)


def read_csv(path) -> list[Profile]:
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
        )
        profile.prefilled = {
            "filter": "Yes",
            "filter_ai": "Yes",
            "gender": profile.gender,
            "year_birth": profile.year_birth,
            "race": profile.race,
            "attention1": 3,
            "attention2": "attention",
        }
        profiles.append(profile)
    return profiles
