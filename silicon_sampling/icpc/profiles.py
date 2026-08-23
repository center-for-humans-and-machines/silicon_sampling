"""Synthetic respondents for the ICPC silicon sample.

Built from the **marginals** of the study's own US quota subsample, not from its
respondent tuples, and that is a deliberate handicap.  Participant-level data
does exist here — that is the entire reason this study is worth sampling — but the
point of the exercise is to estimate how the pipeline will do on the Pfänder
megastudy, where no such data will exist and only quota targets are known.  A
profile drawn from real joint tuples would exploit information the real task will
not have and would flatter the result.  So the joint is left at maximum entropy
except in the one place where independence would be badly wrong.

That place is political orientation.  The instrument asks it twice, on social and
on economic issues, and the two correlate at r = 0.82 in the US sample; drawing
them independently would manufacture a large population of social-liberal
economic-conservatives that barely exists.  The economic item is therefore drawn
as the social item plus a displacement with the observed mean and spread, which
reproduces the correlation to within a point without touching a single respondent
row.

Age is drawn inside its band rather than from the band's midpoint, because the
bands are wide and the instrument asks for a number.

Everything the panel supplier knew is attached to the profile and printed in the
panel record above the consent form; see
:data:`silicon_sampling.icpc.instrument.PANEL_FIELDS` for why that block has to
exist in this study when it did not in the other two.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..vlasceanu.country import UNITED_STATES
from .instrument import ARMS, BY_KEY, DV_BLOCKS, dv_order, extras_order
from .outcomes import SHARE_CODES

#: Marginals of the US quota subsample (n = 8,253; ``country == "usa"``),
#: renormalised over answered responses.  Keys are the published codes.
GENDER = {1: 0.4773, 2: 0.5068, 3: 0.0052, 4: 0.0107}
EDUCATION = {1: 0.0040, 2: 0.2557, 3: 0.5701, 4: 0.1611, 5: 0.0091}
INCOME = {
    1: 0.0644,
    2: 0.0486,
    3: 0.0941,
    4: 0.2165,
    5: 0.2874,
    6: 0.1416,
    7: 0.0569,
    8: 0.0420,
    9: 0.0486,
}
SES_LADDER = {
    1: 0.0275,
    2: 0.0384,
    3: 0.0944,
    4: 0.1272,
    5: 0.1869,
    6: 0.1727,
    7: 0.1676,
    8: 0.1007,
    9: 0.0414,
    10: 0.0431,
}
AGE_BANDS = {"18-29": 0.1806, "30-44": 0.3108, "45-59": 0.2515, "60+": 0.2571}
BAND_RANGE = {"18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59), "60+": (60, 90)}

#: Share of the US sample in each ten-point bin of the 0-100 social-issues
#: political-orientation slider.  Bins, not deciles: the slider piles up on round
#: numbers, so equal-width bins are what the data supports and the sixth of them
#: (50-59, the midpoint) holds 22% of respondents on its own.
IDEOLOGY_BINS = (
    0.1073,
    0.0604,
    0.0520,
    0.0560,
    0.0910,
    0.2238,
    0.0887,
    0.0958,
    0.0867,
    0.1382,
)
#: Observed within-person displacement from the social to the economic item.
IDEOLOGY_SHIFT_MEAN = 2.53
IDEOLOGY_SHIFT_SD = 17.20

#: On-screen labels for the coded demographics, from the survey itself.
GENDER_OPTIONS = {
    1: "Male",
    2: "Female",
    3: "Prefer not to say",
    4: "Non-binary/third gender/other",
}
EDUCATION_OPTIONS = dict(enumerate(UNITED_STATES.education_options, start=1))
INCOME_OPTIONS = dict(enumerate(UNITED_STATES.income_options, start=1))
SES_OPTIONS = {
    10: "Rung 10 (Top) People here are the best off",
    9: "Rung 9",
    8: "Rung 8",
    7: "Rung 7",
    6: "Rung 6",
    5: "Rung 5",
    4: "Rung 4",
    3: "Rung 3",
    2: "Rung 2",
    1: "Rung 1 (Bottom) People here are the worst off",
}

#: The two WEPT demonstration rows and their correct answers.  Prefilled, not
#: sampled: the published sample contains only respondents who got both right,
#: because the cleaning script removed the 354 who did not.
WEPT_DEMO_ANSWERS = {"WEPTdemo1_1": "67, 85", "WEPTdemo2_1": "23, 81"}

#: Household goods the modal US respondent reported owning.  Multi-select, so the
#: transcript can only record one; the modal single item is used.
INDIRECT_SES_ANSWER = "Washing machine"

#: Respondents per arm in the human half the sample is scored against — a 50/50
#: split of the 8,253 US respondents on the benchmark's seed, so roughly half of
#: each arm's count.
HUMAN_HALF_PER_ARM = 350


@dataclass
class Profile:
    """One synthetic respondent, before any generation happens."""

    profile_id: str
    condition: str
    cond: int
    gender: str
    age: int
    age_band: str
    education: str
    income: str
    ses_ladder: int
    politics_social: int
    politics_economic: int
    battery: str
    extras: str
    probe_index: int
    seed: int
    prefilled: dict = field(default_factory=dict, repr=False)


def _draw(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _draw_ideology(rng: random.Random) -> tuple[int, int]:
    """A social-issues position from the bins, and an economic one beside it."""
    chosen = rng.choices(range(10), weights=IDEOLOGY_BINS)[0]
    social = rng.uniform(chosen * 10, chosen * 10 + 10)
    economic = social + rng.gauss(IDEOLOGY_SHIFT_MEAN, IDEOLOGY_SHIFT_SD)
    clamp = lambda value: int(round(min(100.0, max(0.0, value))))  # noqa: E731
    return clamp(social), clamp(economic)


def condition_slots(per_arm: int | None = None) -> list[str]:
    """Arm assignments, balanced, matched to the human half being scored against."""
    count = per_arm or HUMAN_HALF_PER_ARM
    return [arm.key for arm in ARMS for _ in range(count)]


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
        social, economic = _draw_ideology(local)
        profile = Profile(
            profile_id=f"i{index:05d}",
            condition=condition,
            cond=BY_KEY[condition].code,
            gender=GENDER_OPTIONS[_draw(local, GENDER)],
            age=local.randint(low, high),
            age_band=band,
            education=EDUCATION_OPTIONS[_draw(local, EDUCATION)],
            income=INCOME_OPTIONS[_draw(local, INCOME)],
            ses_ladder=int(_draw(local, SES_LADDER)),
            politics_social=social,
            politics_economic=economic,
            battery="|".join(dv_order(local)),
            extras="|".join(extras_order(local)),
            probe_index=local.randrange(9),
            seed=local.randrange(2**31),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles


def prefilled_answers(profile: Profile) -> dict:
    """Answers supplied rather than sampled, plus the echoes the survey pipes.

    Consent, both attention checks and the WEPT demonstration are prefilled as
    passed, because the published file is the *cleaned* export: it contains only
    respondents who passed all four, so sampling them would create a selection
    effect with no counterpart in the target data.
    """
    answers = {
        "profile_id": profile.profile_id,
        "cond": profile.cond,
        "panel_gender": profile.gender,
        "panel_age": profile.age,
        "panel_education": profile.education,
        "panel_income": profile.income,
        "panel_ses": profile.ses_ladder,
        "panel_politics_social": profile.politics_social,
        "panel_politics_economic": profile.politics_economic,
        # consent and the two attention checks
        "Q5": "Yes, I am at least 18 years old and I want to participate",
        "AttentionCheck_purp": "Purple",
        "Attn_60": "sixty",
        # demographics, asked at the very end of this instrument
        "Gender": profile.gender,
        "Age": profile.age,
        "Education.2": profile.education,
        "Income": profile.income,
        "MacArthur_SES": SES_OPTIONS[profile.ses_ladder],
        "Politics2_1": profile.politics_social,
        "Politics2_9": profile.politics_economic,
        "Indirect_SES": INDIRECT_SES_ANSWER,
    }
    answers.update(WEPT_DEMO_ANSWERS)
    return answers


FIELDS = (
    "profile_id",
    "condition",
    "cond",
    "gender",
    "age",
    "age_band",
    "education",
    "income",
    "ses_ladder",
    "politics_social",
    "politics_economic",
    "battery",
    "extras",
    "probe_index",
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
            age=int(row["age"]),
            age_band=row["age_band"],
            education=row["education"],
            income=row["income"],
            ses_ladder=int(row["ses_ladder"]),
            politics_social=int(row["politics_social"]),
            politics_economic=int(row["politics_economic"]),
            battery=row["battery"],
            extras=row["extras"],
            probe_index=int(row["probe_index"]),
            seed=int(row["seed"]),
        )
        profile.prefilled = prefilled_answers(profile)
        profiles.append(profile)
    return profiles


def sanity(profiles: Sequence[Profile]) -> dict:
    """Marginals of the drawn sample, for a quick check against the targets."""
    total = len(profiles)
    out = {
        "n": total,
        "per_arm": {
            arm.key: sum(1 for p in profiles if p.condition == arm.key) for arm in ARMS
        },
        "gender": {},
        "age_band": {},
        "battery_orders": len({p.battery for p in profiles}),
        "dv_blocks": len(DV_BLOCKS),
        "share_codes": len(SHARE_CODES),
    }
    for key in ("gender", "age_band"):
        counts: dict[str, int] = {}
        for profile in profiles:
            counts[getattr(profile, key)] = counts.get(getattr(profile, key), 0) + 1
        out[key] = {k: round(v / total, 4) for k, v in sorted(counts.items())}
    return out
