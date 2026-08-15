"""Synthetic respondent profiles for the Voelkel silicon sample.

Built from the study's **published marginals**, not from its participant-level
rows.  The point of this exercise is to estimate how the pipeline will do on the
Pfänder megastudy, where no participant-level data exists; a profile drawn from
real respondent tuples would exploit information the real task will not have and
would flatter the result.  Marginals are what a paper reports, so marginals are
what we use, and the joint is left at maximum entropy — the same position the
Pfänder quotas put us in.

Only gender, race and party appear anywhere in the instrument.  Age, education
and ideology came from the panel supplier and were never shown, so a synthetic
respondent cannot condition on them.  They are still attached to the profile,
because the analysis needs them as moderators, but any subgroup result over those
three is measuring what the model could not see.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

from .instrument import CCT5_SCENARIOS, CONDITIONS, PARTY_PIPES, post_order

#: Published marginals (Voelkel et al. 2024, N = 35,252).
GENDER = {"Male": 0.4550, "Female": 0.5411, "Other": 0.0039}
RACE = {
    "White": 0.7634,
    "Black": 0.1043,
    "Hispanic": 0.0396,
    "Asian": 0.0368,
    "Other": 0.0559,
}
PARTY_GEN = {"Republican": 0.4258, "Democrat": 0.4404, "Independent": 0.1338}
EDUCATION = {
    "HS or less": 0.1944,
    "Some college": 0.3690,
    "Bachelor": 0.2792,
    "Postgraduate": 0.1574,
}
AGE_BANDS = {"18-29": 0.1212, "30-44": 0.2589, "45-59": 0.2703, "60+": 0.3497}
IDEOLOGY = {1: 0.0846, 2: 0.1694, 3: 0.0874, 4: 0.2484, 5: 0.1136, 6: 0.2027, 7: 0.0931}

#: Independents were screened out unless they leaned; the lean sets their inparty.
INDEPENDENT_LEAN = {"Republican": 0.5, "Democrat": 0.5}

BAND_RANGE = {"18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59), "60+": (60, 90)}

#: The survey's on-screen race labels, keyed by the cleaned label in the data.
RACE_ONSCREEN = {
    "White": "White / Caucasian",
    "Black": "Black / African American",
    "Hispanic": "Hispanic / Latino",
    "Asian": "Asian / Asian American",
    "Other": "Other",
}

#: Respondents per arm in the human half the sample is scored against
#: (50/50 split of 35,252 on the preregistered seed).
HUMAN_HALF = {"Null_Control": 2825, "_intervention": 563}


@dataclass
class Profile:
    """One synthetic respondent, before any generation happens."""

    profile_id: str
    condition: str
    party_gen: str
    inparty: str
    gender: str
    race: str
    age: int
    age_band: str
    education: str
    ideology: int
    scenario: str = ""
    battery: str = ""
    seed: int = 0
    prefilled: dict = field(default_factory=dict, repr=False)


def _draw(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def condition_slots(total_per_intervention: int | None = None) -> list[str]:
    """Condition assignments matched to the human half being scored against."""
    control = HUMAN_HALF["Null_Control"]
    per_arm = total_per_intervention or HUMAN_HALF["_intervention"]
    slots = ["Null_Control"] * control
    for condition in CONDITIONS:
        if condition != "Null_Control":
            slots += [condition] * per_arm
    return slots


def build(seed: int = 20260815, per_intervention: int | None = None) -> list[Profile]:
    """Construct every profile, deterministically."""
    rng = random.Random(seed)
    assignments = condition_slots(per_intervention)
    rng.shuffle(assignments)

    profiles: list[Profile] = []
    for index, condition in enumerate(assignments, start=1):
        local = random.Random(seed * 1_000_003 + index)
        party_gen = _draw(local, PARTY_GEN)
        inparty = (
            _draw(local, INDEPENDENT_LEAN) if party_gen == "Independent" else party_gen
        )
        band = _draw(local, AGE_BANDS)
        low, high = BAND_RANGE[band]
        race = _draw(local, RACE)
        gender = _draw(local, GENDER)
        profile = Profile(
            profile_id=f"v{index:05d}",
            condition=condition,
            party_gen=party_gen,
            inparty=inparty,
            gender=gender,
            race=race,
            age=local.randint(low, high),
            age_band=band,
            education=_draw(local, EDUCATION),
            ideology=int(_draw(local, IDEOLOGY)),
            scenario=(
                local.choice(CCT5_SCENARIOS)
                if condition == "Misperception_Competition"
                else ""
            ),
            battery="|".join(post_order(local)),
            seed=local.randrange(2**31),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles


def prefilled_answers(profile: Profile) -> dict:
    """Answers supplied rather than sampled, plus the party pipes.

    Consent and the attention checks are pre-filled as passed: the published
    sample contains only respondents who passed them, so sampling them would
    create a selection effect with no counterpart in the target data.
    """
    answers = dict(PARTY_PIPES[profile.inparty])
    answers.update(
        {
            "Gender": profile.gender,
            "Race": RACE_ONSCREEN[profile.race],
            "Party_Gen": profile.party_gen,
            "Filter": "Yes",
            "Attention_1": "Somewhat disagree",
            "Attention_2": "attention",
        }
    )
    if profile.party_gen == "Independent":
        answers["Party_Ind"] = (
            f"Closer to {'Republican' if profile.inparty == 'Republican' else 'Democratic'} Party"
        )
    if profile.scenario:
        answers["CCT5_ScenarioCondition"] = profile.scenario
    return answers


FIELDS = (
    "profile_id",
    "condition",
    "party_gen",
    "inparty",
    "gender",
    "race",
    "age",
    "age_band",
    "education",
    "ideology",
    "scenario",
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
            party_gen=row["party_gen"],
            inparty=row["inparty"],
            gender=row["gender"],
            race=row["race"],
            age=int(row["age"]),
            age_band=row["age_band"],
            education=row["education"],
            ideology=int(row["ideology"]),
            scenario=row["scenario"],
            battery=row["battery"],
            seed=int(row["seed"]),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles
