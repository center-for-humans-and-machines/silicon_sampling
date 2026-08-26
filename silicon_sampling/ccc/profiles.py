"""Synthetic respondents for the Climate Change Challenge.

Demographics are **resampled from the study's own participants** rather than drawn
from a fitted joint distribution. For a calibration study that is the right call
twice over: the composition then matches the human reference exactly, so a level or
subgroup comparison is like-for-like, and the joint distribution is real rather than
a max-entropy reconstruction of its margins.

The Pfänder run needs a fitted model because its target sample does not exist yet.
Here it does.

Arm sizes mirror the retained human sample, so each synthetic arm has the same n as
the arm it is compared against. ``System Preservation Framing`` is excluded on both
sides — see :mod:`~silicon_sampling.ccc.instrument`.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field

from ..lazy import lazy_module

from . import instrument as inst
from .paths import RECODED_CSV

#: Imported on first use.  ``read_csv`` below deliberately avoids it, because the
#: sampling path has to run in the Muse-Glimmer container, which ships no pandas;
#: ``build`` and ``write_csv`` only ever run locally.
pd = lazy_module("pandas")

#: Columns the profile carries over from a real respondent.
SOURCE_COLUMNS = ("Gender", "YOB", "Race", "Education", "ANES_Gen", "Age", "PartyC3")

RACE_LABELS = {
    1: "White / Caucasian",
    2: "Black / African-American",
    3: "Latino / Hispanic",
    4: "Asian / Asian-American",
    5: "Other",
}
EDUCATION_LABELS = {
    1: "Less than high school",
    2: "High school diploma / GED",
    3: "Some college or Associate's degree",
    4: "Bachelor's degree",
    5: "Postgraduate (Master's degree, Ph.D., Professional degree)",
}
PARTY_LABELS = {
    1: "Republican",
    2: "Independent",
    3: "Democrat",
    4: "Other (please specify)",
}

FIELDS = (
    "profile_id",
    "condition",
    "gender",
    "yob",
    "age",
    "age_band",
    "race",
    "education",
    "party",
    "seed",
)


@dataclass
class Profile:
    """One synthetic respondent."""

    profile_id: str
    condition: str
    gender: str
    yob: int
    age: int
    age_band: str
    race: str
    education: str
    party: str
    seed: int
    prefilled: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {name: getattr(self, name) for name in FIELDS}


def _band(age) -> str:
    try:
        value = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 29:
        return "18-29"
    if value <= 44:
        return "30-44"
    if value <= 59:
        return "45-59"
    return "60+"


def prefilled_answers(profile: "Profile") -> dict:
    """Answers supplied rather than sampled.

    Consent, the filter and the attention check are pre-filled as passed, because
    the released sample contains only respondents who passed them — every row has
    ``Attention1 == 3``. Demographics come from the profile so composition matches
    the human reference.
    """
    return {
        **inst.PREFILLED_PASSES,
        "Gender": profile.gender,
        "YOB": str(profile.yob),
        "Race": profile.race,
        "Education": profile.education,
        "ANES_Gen": profile.party,
    }


def build(total: int | None = None, seed: int = 20260826) -> list[Profile]:
    """Profiles mirroring the retained human sample, arm sizes included.

    ``total`` is normally left alone: the default reproduces the retained sample's
    own size and per-arm balance, which is what makes the comparison paired.
    """
    frame = pd.read_csv(RECODED_CSV, encoding="utf-8-sig", low_memory=False)
    frame = frame[frame["Condition"].isin(inst.ARM_BLOCKS)].copy()
    counts = frame["Condition"].value_counts().to_dict()
    if total is not None:
        scale = total / sum(counts.values())
        counts = {arm: max(1, round(n * scale)) for arm, n in counts.items()}

    donors = (
        frame[list(SOURCE_COLUMNS)]
        .dropna(subset=["Gender", "YOB"])
        .reset_index(drop=True)
    )
    rng = random.Random(seed)
    order = list(range(len(donors)))
    rng.shuffle(order)

    assignments: list[str] = []
    for arm, n in counts.items():
        assignments.extend([arm] * int(n))
    rng.shuffle(assignments)

    profiles: list[Profile] = []
    for index, condition in enumerate(assignments, start=1):
        row = donors.iloc[order[index % len(order)]]
        age = row.get("Age")
        profile = Profile(
            profile_id=f"c{index:05d}",
            condition=condition,
            gender=str(row["Gender"]),
            yob=int(row["YOB"]),
            age=int(age) if pd.notna(age) else 0,
            age_band=_band(age),
            race=(
                RACE_LABELS.get(int(row["Race"]), "Other")
                if pd.notna(row.get("Race"))
                else "Other"
            ),
            education=EDUCATION_LABELS.get(
                _education_code(row.get("Education")),
                "Some college or Associate's degree",
            ),
            party=(
                PARTY_LABELS.get(int(row["ANES_Gen"]), "Independent")
                if pd.notna(row.get("ANES_Gen"))
                else "Independent"
            ),
            seed=rng.randrange(2**31),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles


def _education_code(value) -> int:
    """The released file collapses education to three strings; map back to a level.

    The survey asked five levels and the released data kept three, so a synthetic
    respondent shown the original five-level question has to be given one of them.
    The midpoint of each released band is used, and the mapping is recorded here
    rather than hidden in a comprehension because it is a judgement call.
    """
    text = str(value)
    if text.startswith("HS"):
        return 2
    if text.startswith("Some"):
        return 3
    if text.startswith("Bachelor"):
        return 4
    return 3


def write_csv(profiles: list[Profile], path) -> None:
    frame = pd.DataFrame([p.as_row() for p in profiles])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_csv(path) -> list[Profile]:
    """Profiles from disk, using the stdlib csv module.

    Deliberately pandas-free: this is the only function in this module the
    sampler calls, and the sampler has to run in a container without pandas.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        profile = Profile(
            profile_id=row["profile_id"],
            condition=row["condition"],
            gender=row["gender"],
            yob=int(row["yob"]),
            age=int(row["age"]),
            age_band=row["age_band"],
            race=row["race"],
            education=row["education"],
            party=row["party"],
            seed=int(row["seed"]),
        )
        profile.prefilled = prefilled_answers(profile)
        out.append(profile)
    return out
