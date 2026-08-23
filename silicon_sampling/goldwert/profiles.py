"""Synthetic respondent profiles for the Goldwert silicon sample.

Built from **marginals only**, not from the study's participant-level rows.  The
point of this exercise is to estimate how the pipeline will do on the Pfänder
megastudy, where no participant-level data exists; a profile drawn from real
respondent tuples would exploit information the real task will not have and would
flatter the result.  Marginals are what a paper reports, so marginals are what we
use, and the joint is left at maximum entropy — the same position the Pfänder
quotas put us in.

There is a wrinkle here that the Voelkel study did not have, and it is worth
being blunt about: **this instrument asks its demographics last**.  Gender, age,
party, education, income and the SES ladder all come *after* the outcome battery,
so the 7,784 respondents who completed the interventions and the outcomes but
dropped before the demographic block have no demographics at all.  Every marginal
below is therefore a marginal over completers, not over the recruited sample, and
the recruited sample is the one that was quota-matched to the Census.  Any
subgroup result computed over these variables inherits that: it describes the
people who finished, and the interventions themselves affected who finished.

The one demographic the instrument shows the respondent *before* anything else is
nothing at all — there is no pre-treatment demographic block — so unlike Voelkel
there is no distinction here between what a model could see and what it could not.
A profile's attributes shape the answers only because we put them in the prompt.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

from .instrument import ARMS, BY_NAME, CONDITIONS, PREFILLED_ANSWERS, dv_order

#: Observed marginals over the 23,444-23,541 respondents who reached the
#: demographic block (Goldwert et al. 2026, N recruited = 31,324).
GENDER = {"Male": 0.4001, "Female": 0.5999}
PARTY = {"Democrat": 0.4313, "Republican": 0.2760, "Other": 0.2928}
AGE_BANDS = {"18-29": 0.1949, "30-44": 0.3505, "45-59": 0.2344, "60+": 0.2202}
EDUCATION = {
    "Grade school": 0.0025,
    "High school": 0.2319,
    "College": 0.6287,
    "Postgraduate": 0.1369,
}
INCOME = {
    1: 0.0614,
    2: 0.0424,
    3: 0.0862,
    4: 0.2422,
    5: 0.3352,
    6: 0.1360,
    7: 0.0539,
    8: 0.0427,
}
SES_LADDER = {
    1: 0.0336,
    2: 0.0524,
    3: 0.1320,
    4: 0.1560,
    5: 0.2025,
    6: 0.1803,
    7: 0.1453,
    8: 0.0644,
    9: 0.0161,
    10: 0.0174,
}

BAND_RANGE = {"18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59), "60+": (60, 90)}

#: The on-screen wording for each drawn value, so a prefilled answer is legal.
GENDER_ONSCREEN = {"Male": "Male", "Female": "Female"}
PARTY_ONSCREEN = {"Democrat": "Democrat", "Republican": "Republican", "Other": "Other"}
EDUCATION_ONSCREEN = {
    "Grade school": "0-6 (up to grade school/elementary school)",
    "High school": "7-12 (up to high school)",
    "College": "13-16 (college/undergraduate university/certificate training)",
    "Postgraduate": "More than 17 years (doctorate degree, medical degree, etc.)",
}
INCOME_ONSCREEN = {
    1: "Less than $10,000",
    2: "$10,000 to $14,999",
    3: "$15,000 to $24,999",
    4: "$25,000 to $49,999",
    5: "$50,000 to $99,999",
    6: "$100,000 to $149,999",
    7: "$150,000 to $199,999",
    8: "$200,000 or more",
}
SES_ONSCREEN = {
    1: "Rung 1 (Bottom) People here are the worst off",
    10: "Rung 10 (Top) People here are the best off",
    **{rung: f"Rung {rung}" for rung in range(2, 10)},
}

#: Respondents per arm in the human half a sample is scored against (a 50/50
#: split of the 1,733-1,745 assigned to each arm, on the preregistered seed).
HUMAN_HALF_PER_ARM = 870


@dataclass
class Profile:
    """One synthetic respondent, before any generation happens."""

    profile_id: str
    condition: str
    cond: int
    gender: str
    party: str
    age: int
    age_band: str
    education: str
    income: int
    ses: int
    battery: str = ""
    seed: int = 0
    prefilled: dict = field(default_factory=dict, repr=False)


def _draw(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def condition_slots(per_arm: int | None = None) -> list[str]:
    """Arm assignments matched to the human half being scored against."""
    count = per_arm or HUMAN_HALF_PER_ARM
    return [condition for condition in CONDITIONS for _ in range(count)]


def build(seed: int = 20260823, per_arm: int | None = None) -> list[Profile]:
    """Construct every profile, deterministically."""
    rng = random.Random(seed)
    assignments = condition_slots(per_arm)
    rng.shuffle(assignments)

    profiles: list[Profile] = []
    for index, condition in enumerate(assignments, start=1):
        local = random.Random(seed * 1_000_003 + index)
        band = _draw(local, AGE_BANDS)
        low, high = BAND_RANGE[band]
        profile = Profile(
            profile_id=f"g{index:05d}",
            condition=condition,
            cond=BY_NAME[condition].cond,
            gender=_draw(local, GENDER),
            party=_draw(local, PARTY),
            age=local.randint(low, high),
            age_band=band,
            education=_draw(local, EDUCATION),
            income=int(_draw(local, INCOME)),
            ses=int(_draw(local, SES_LADDER)),
            battery="|".join(dv_order(local)),
            seed=local.randrange(2**31),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles


def prefilled_answers(profile: Profile) -> dict:
    """Answers supplied rather than sampled.

    Consent and both attention checks are filled in as passed, because the
    published file contains only respondents who passed them: sampling them would
    build a selection effect with no counterpart in the data being predicted.  The
    demographics are supplied rather than sampled for the opposite reason — they
    are the profile, and asking a model to invent them would make the quota
    meaningless.
    """
    answers = dict(PREFILLED_ANSWERS)
    answers.update(
        {
            "Gender": GENDER_ONSCREEN[profile.gender],
            "Age": str(profile.age),
            "Party": PARTY_ONSCREEN[profile.party],
            "Edu": EDUCATION_ONSCREEN[profile.education],
            "Income": INCOME_ONSCREEN[profile.income],
            "MacArthur_SES": SES_ONSCREEN[profile.ses],
        }
    )
    return answers


FIELDS = (
    "profile_id",
    "condition",
    "cond",
    "gender",
    "party",
    "age",
    "age_band",
    "education",
    "income",
    "ses",
    "battery",
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
            cond=int(row["cond"]),
            gender=row["gender"],
            party=row["party"],
            age=int(row["age"]),
            age_band=row["age_band"],
            education=row["education"],
            income=int(row["income"]),
            ses=int(row["ses"]),
            battery=row["battery"],
            seed=int(row["seed"]),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles


def arm_table() -> list[dict]:
    """Every arm with its published n, and whether a profile is drawn for it."""
    return [
        {
            "cond": arm.cond,
            "condName": arm.name,
            "sampled": arm.name in CONDITIONS,
            "modality": arm.modality,
        }
        for arm in ARMS
    ]
