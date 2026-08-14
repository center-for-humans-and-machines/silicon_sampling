"""From sampled answers to the benchmark's 13 outcomes.

Construction rules are the codebook's, not ours: ``trust_multidimensional`` is
the mean of the four *subscale means* (identical to the mean of the twelve items
only because the subscales are equally sized, but computed as specified),
``funding_perceptions`` is ``100 - funding_5`` so that higher means "supports more
funding", and ``newsletter_signup`` is the Yes/No item recoded to 1/0.
"""

from __future__ import annotations

from typing import Mapping

from .instrument import SURVEY_YEAR

#: Qualtrics item id -> submission column, for items that ship as-is.
DIRECT = {
    "trust_competent_1": "trust_competence_1",
    "trust_intelligent_1": "trust_competence_2",
    "trust_qualified_1": "trust_competence_3",
    "trust_honest_1": "trust_integrity_1",
    "trust_ethical_1": "trust_integrity_2",
    "trust_sincere_1": "trust_integrity_3",
    "trust_concerned_1": "trust_benevolence_1",
    "trust_improve_1": "trust_benevolence_2",
    "trust_considerate_1": "trust_benevolence_3",
    "trust_feedback_1": "trust_openness_1",
    "trust_transparent_1": "trust_openness_2",
    "trust_attention_1": "trust_openness_3",
    "trust_post_1": "trust_post",
    "distrust_1": "distrust_post",
    "belief_post_1": "belief_post",
    "policy_general_1": "policy_general",
    "donation": "donation_ams",
}

SUBSCALES = {
    "trust_competence": (
        "trust_competent_1",
        "trust_intelligent_1",
        "trust_qualified_1",
    ),
    "trust_integrity": ("trust_honest_1", "trust_ethical_1", "trust_sincere_1"),
    "trust_benevolence": (
        "trust_concerned_1",
        "trust_improve_1",
        "trust_considerate_1",
    ),
    "trust_openness": ("trust_feedback_1", "trust_transparent_1", "trust_attention_1"),
}

MEANS = {
    "inst_trust_mean": (
        "inst_trust_epa_1",
        "inst_trust_nasa_1",
        "inst_trust_noaa_1",
        "inst_trust_uni_1",
        "inst_trust_gov_1",
    ),
    "policy_role_mean": ("policy_1_1", "policy_2_1", "policy_3_1", "policy_4_1"),
    "concern_mean": ("concern_1_1", "concern_2_1", "concern_3_1"),
    "policy_specific_mean": tuple(
        f"policy_specific_{index}_1" for index in range(1, 8)
    ),
    "behavior_mean": (
        "individual_meat_1",
        "individual_transport_1",
        "individual_solar_1",
        "individual_fly_1",
        "individual_talk_1",
        "individual_donate_1",
    ),
}

#: The 13 scored outcomes, in the benchmark's order.
OUTCOMES = (
    "trust_multidimensional",
    "trust_post",
    "distrust_post",
    "funding_perceptions",
    "policy_role_mean",
    "inst_trust_mean",
    "belief_post",
    "concern_mean",
    "policy_general",
    "policy_specific_mean",
    "behavior_mean",
    "donation_ams",
    "newsletter_signup",
)

#: Scale range of each outcome, for converting effects to percentage points.
SCALE_RANGE = {name: 100.0 for name in OUTCOMES}
SCALE_RANGE["donation_ams"] = 10.0
SCALE_RANGE["newsletter_signup"] = 1.0

#: Every item a respondent must answer for the outcomes to be computable.
REQUIRED_ITEMS = (
    tuple(DIRECT)
    + tuple(item for items in MEANS.values() for item in items)
    + ("funding_5", "newsletter")
)

#: The six moderators, with the exact levels a submission must carry.
MODERATORS = {
    "gender": ("Male", "Female", "Other"),
    "age_band": ("18-29", "30-44", "45-59", "60+"),
    "race": (
        "White / Caucasian",
        "Black / African American",
        "Hispanic / Latino",
        "Asian / Asian American",
        "Other",
    ),
    "education": (
        "Less than high school",
        "High school diploma / GED",
        "Some college or Associate's degree",
        "Bachelor's degree",
        "Master's degree / Professional degree",
        "Doctorate degree / Ph.D.",
    ),
    "income": (
        "Less than $30,000",
        "$30,000 to $55,999",
        "$56,000 to $99,999",
        "$100,000 to $167,999",
        "$168,000 or more",
    ),
    "party": ("Republican", "Democrat", "Independent", "Other"),
}

#: The survey's on-screen race labels hyphenate differently from the submission's.
RACE_ONSCREEN_TO_SUBMISSION = {
    "White / Caucasian": "White / Caucasian",
    "Black / African-American": "Black / African American",
    "Latino / Hispanic": "Hispanic / Latino",
    "Asian / Asian-American": "Asian / Asian American",
    "Other": "Other",
}

#: Tier-1 column order, matching the benchmark's example submission.
TIER1_COLUMNS = (
    ("profile_id", "condition")
    + tuple(MODERATORS)
    + ("trust_multidimensional",)
    + tuple(
        f"trust_{scale}_{index}"
        for scale in ("competence", "integrity", "benevolence", "openness")
        for index in (1, 2, 3)
    )
    + (
        "trust_post",
        "distrust_post",
        "funding_perceptions",
        "policy_role_mean",
        "inst_trust_mean",
        "belief_post",
        "concern_mean",
        "policy_general",
        "policy_specific_mean",
        "behavior_mean",
        "donation_ams",
        "newsletter_signup",
    )
)


def age_band(year_birth: int, survey_year: int = SURVEY_YEAR) -> str:
    age = survey_year - int(year_birth)
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def _mean(answers: Mapping[str, object], items) -> float:
    return sum(float(answers[item]) for item in items) / len(items)


def compute(answers: Mapping[str, object]) -> dict[str, object]:
    """The scored outcomes and moderators for one respondent."""
    out: dict[str, object] = {
        target: answers[source] for source, target in DIRECT.items()
    }
    for name, items in SUBSCALES.items():
        out[name] = _mean(answers, items)
    out["trust_multidimensional"] = sum(out[name] for name in SUBSCALES) / len(
        SUBSCALES
    )
    for name, items in MEANS.items():
        out[name] = _mean(answers, items)
    out["funding_perceptions"] = 100 - float(answers["funding_5"])
    out["newsletter_signup"] = 1 if answers["newsletter"] == "Yes" else 0
    out["gender"] = answers["gender"]
    out["age_band"] = age_band(int(answers["year_birth"]))
    out["race"] = RACE_ONSCREEN_TO_SUBMISSION[str(answers["race"])]
    out["education"] = answers["education"]
    out["income"] = answers["income"]
    out["party"] = answers["party"]
    return out
