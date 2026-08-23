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

**Marginals are not the same as a joint, and the difference is measurable.**
Drawing every attribute independently reproduces each published marginal exactly
and gets every association between them wrong — it puts postgraduates and
"HS or less" at the same rate inside every race, and Republicans at the same rate
at every education level.  Real samples do not look like that, and a subgroup
analysis is exactly where the difference shows.  So education and party are now
drawn from ``demographics.joint``: CCAM supplies
``P(education, party | gender, age band, race)`` and the published marginals stay
the calibration target, which keeps the "marginals only" discipline above intact —
what CCAM contributes is covariance, never a level.  Gender, race and age band
have nothing to condition on, so they stay independent draws from their published
marginals, which is still the maximum-entropy joint given what the paper reports.

That draw is opt-out (``build(demographics=False)``) and recorded in the profiles
file, because two finished Voelkel runs read a ``profiles.csv`` written before it
existed: a file with a ``demographics`` column was built with the joint, one
without it was not, and either loads.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..demographics import codebook
from ..demographics import joint as demographics_joint
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

#: Seed stream for the CCAM-drawn moderators, separate from the per-profile one
#: so the draw is a pure function of the profile id and its quota cell.
DEMOGRAPHIC_SEED = 20260823

#: What a profile records about where its education and party came from.
CCAM_SOURCE = "ccam"


def given_margin() -> np.ndarray:
    """The gender x age band x race margin the joint is calibrated to.

    The product of the three published marginals, because independence is all
    the paper's marginals will support — the same maximum-entropy position the
    rest of this module takes.  Two foldings happen on the way: the 0.39% who
    answered the gender item with a third option are renormalised away, since
    CCAM has no such code and those respondents draw the population average over
    gender anyway; and Asian and Other fold together, because CCAM's trend file
    does not separate them.
    """
    book = codebook.VOELKEL
    space = demographics_joint.space(book)
    gender = np.array([GENDER[level] for level in book.gender], dtype=float)
    bands = np.array([AGE_BANDS[level] for level in book.age_bands], dtype=float)
    race = np.zeros(len(codebook.CCAM_RACE))
    for level, share in RACE.items():
        race[space.position("race", level)] += share
    margin = (
        gender[:, None, None]
        * (bands / bands.sum())[None, :, None]
        * (race / race.sum())[None, None, :]
    )
    return margin / margin.sum()


def level_targets() -> dict[str, np.ndarray]:
    """The published education and party marginals, in codebook order."""
    book = codebook.VOELKEL
    return {
        "education": np.array([EDUCATION[level] for level in book.education]),
        "party": np.array([PARTY_GEN[level] for level in book.party]),
    }


@lru_cache(maxsize=1)
def demographic_sampler() -> demographics_joint.Sampler:
    """The fitted joint for this study, from its shipped table or from CCAM.

    Built explicitly rather than through ``joint.shipped`` because this study's
    levels come from its own published marginals: a fallback that quietly used
    CCAM's national ones would produce a sampler that looks fine and answers a
    different question.
    """
    book = codebook.VOELKEL
    path = demographics_joint.table_path(book)
    if path.exists():
        return demographics_joint.Sampler(
            demographics_joint.read_table(path, book), book
        )
    model = demographics_joint.fit(
        book, targets=level_targets(), demographics=given_margin()
    )
    return demographics_joint.Sampler(model.table, book)


def demographic_answers(profile_id: str, gender: str, band: str, race: str) -> dict:
    """Education and party for one respondent, conditional on the other three."""
    rng = random.Random(DEMOGRAPHIC_SEED * 1_000_003 + int(profile_id[1:]))
    return demographic_sampler().draw(gender, band, race, rng)


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
    #: ``"ccam"`` when education and party came from the joint, empty when they
    #: came from their own published marginals.  Carried in the CSV so a run says
    #: for itself which it was.
    demographics: str = ""
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


def build(
    seed: int = 20260815,
    per_intervention: int | None = None,
    demographics: bool = True,
) -> list[Profile]:
    """Construct every profile, deterministically.

    The two branches below differ only in *order*: the joint has to condition on
    gender, age band and race, so those three move ahead of party, and the two
    marginal draws the joint replaces are then not made at all.  Keeping the old
    order intact under ``demographics=False`` is what lets the two finished runs'
    ``profiles.csv`` still be regenerated byte for byte.
    """
    rng = random.Random(seed)
    assignments = condition_slots(per_intervention)
    rng.shuffle(assignments)

    profiles: list[Profile] = []
    for index, condition in enumerate(assignments, start=1):
        profile_id = f"v{index:05d}"
        local = random.Random(seed * 1_000_003 + index)
        drawn: dict[str, str] = {}
        if demographics:
            band = _draw(local, AGE_BANDS)
            race = _draw(local, RACE)
            gender = _draw(local, GENDER)
            drawn = demographic_answers(profile_id, gender, band, race)
            party_gen = drawn["party"]
        else:
            party_gen = _draw(local, PARTY_GEN)
        inparty = (
            _draw(local, INDEPENDENT_LEAN) if party_gen == "Independent" else party_gen
        )
        if not demographics:
            band = _draw(local, AGE_BANDS)
            race = _draw(local, RACE)
            gender = _draw(local, GENDER)
        low, high = BAND_RANGE[band]
        profile = Profile(
            profile_id=profile_id,
            condition=condition,
            party_gen=party_gen,
            inparty=inparty,
            gender=gender,
            race=race,
            age=local.randint(low, high),
            age_band=band,
            education=drawn["education"] if drawn else _draw(local, EDUCATION),
            ideology=int(_draw(local, IDEOLOGY)),
            scenario=(
                local.choice(CCT5_SCENARIOS)
                if condition == "Misperception_Competition"
                else ""
            ),
            battery="|".join(post_order(local)),
            seed=local.randrange(2**31),
            demographics=CCAM_SOURCE if drawn else "",
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
            # The second check is a comprehension item about a short article,
            # not the Pfänder-style "type this word" check.
            "Attention_2": "Event licensing",
            "Transition_Int": "I understand I must complete the full study to collect my full earnings",
            # A video screener everyone saw. Our arms are all text, but the block
            # is part of the instrument, and the published sample contains only
            # respondents who passed it — so it is filled in as passed rather
            # than asked of a model that was shown no video.
            "VideoCheck": "Saw waves crashing, Heard ocean waves",
        }
    )
    if profile.party_gen == "Independent":
        answers["Party_Ind"] = (
            f"Closer to {'Republican' if profile.inparty == 'Republican' else 'Democratic'} Party"
        )
    if profile.scenario:
        answers["CCT5_ScenarioCondition"] = profile.scenario
    return answers


#: Columns every ``profiles.csv`` has carried since the first run.
BASE_FIELDS = (
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

#: Written only when the profiles were built with the joint, so a file's header
#: is what distinguishes the two.
FIELDS = BASE_FIELDS + ("demographics",)


def fieldnames(profiles: Sequence[Profile]) -> tuple[str, ...]:
    return FIELDS if any(profile.demographics for profile in profiles) else BASE_FIELDS


def write_csv(profiles: Sequence[Profile], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames(profiles), extrasaction="ignore"
        )
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
            demographics=row.get("demographics", ""),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles
