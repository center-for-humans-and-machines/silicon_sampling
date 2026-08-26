"""The Climate Change Challenge outcomes, and the reverse-coding trap in them.

Every outcome is a mean of 0-100 slider items except the donation, which is a sum
in cents.  The formulas below are transcribed from ``CCC - Script - Step 2 -
Preparation.R`` and were verified numerically against all 13,821 released rows.

**The trap.** Twelve item columns in the released data are ``100 - x`` of what the
respondent saw.  Step 2 applies the reverse-coding *in place, before* writing the
CSV, so the column named ``Belief_Post_3_1`` does not hold the number that was on
screen — it holds its complement.  Confirmed by diffing the deidentified export
against the recoded one: ``100 - raw`` holds exactly on these twelve and the other
85 shared numeric columns are untouched.

A silicon sample collects the *on-screen* answer, so it has to apply the same flip
before averaging, or belief, specific-policy support and the whole candidate
outcome come out inverted against the human reference.
"""

from __future__ import annotations

#: Items whose released column is ``100 - on-screen``.  Slot ids, not columns.
REVERSED_ITEMS = (
    "Belief_Pre_3",
    "Belief_Post_3",
    "PoliciesSp_Pre_3",
    "PoliciesSp_Post_3",
    "Candidate_Pre_1",
    "Candidate_Pre_2",
    "Candidate_Pre_3",
    "Candidate_Pre_4",
    "Candidate_Post_1",
    "Candidate_Post_2",
    "Candidate_Post_3",
    "Candidate_Post_4",
)

#: composite -> (items, scale range).  Means unless listed in :data:`SUMMED`.
COMPOSITES: dict[str, tuple[tuple[str, ...], float]] = {
    "Belief_Pre": (("Belief_Pre_1", "Belief_Pre_2", "Belief_Pre_3"), 100.0),
    "Belief_Post": (("Belief_Post_1", "Belief_Post_2", "Belief_Post_3"), 100.0),
    "Concern_Pre": (("Concern_Pre_1", "Concern_Pre_2", "Concern_Pre_3"), 100.0),
    "Concern_Post": (("Concern_Post_1", "Concern_Post_2", "Concern_Post_3"), 100.0),
    "Policies_Pre": (("Policies_Pre_1", "Policies_Pre_2", "Policies_Pre_3"), 100.0),
    "Policies_Post": (("Policies_Post_1", "Policies_Post_2", "Policies_Post_3"), 100.0),
    "Intent_Pre": (tuple(f"Intent_Pre_{i}" for i in range(1, 5)), 100.0),
    "Intent_Post": (tuple(f"Intent_Post_{i}" for i in range(1, 5)), 100.0),
    "PoliciesSp_Pre": (tuple(f"PoliciesSp_Pre_{i}" for i in range(1, 5)), 100.0),
    "PoliciesSp_Post": (tuple(f"PoliciesSp_Post_{i}" for i in range(1, 5)), 100.0),
    "Candidate_Pre": (tuple(f"Candidate_Pre_{i}" for i in range(1, 5)), 100.0),
    "Candidate_Post": (tuple(f"Candidate_Post_{i}" for i in range(1, 5)), 100.0),
    "Companies_Pre": (("Companies_Pre_1", "Companies_Pre_2", "Companies_Pre_3"), 100.0),
    "Companies_Post": (
        ("Companies_Post_1", "Companies_Post_2", "Companies_Post_3"),
        100.0,
    ),
    "IntentNp_Pre": (tuple(f"IntentNp_Pre_{i}" for i in range(1, 7)), 100.0),
    "IntentNp_Post": (tuple(f"IntentNp_Post_{i}" for i in range(1, 7)), 100.0),
    #: A sum in cents over the five organisations.  The sixth box, "keep for
    #: myself", is deliberately excluded, and Donation == 100 - keep.
    "Donation": (tuple(f"Donation_{i}" for i in range(1, 6)), 100.0),
}

#: Composites that sum rather than average.
SUMMED = ("Donation",)

#: The four primary outcomes, in the paper's order.
PRIMARY = ("Belief", "Concern", "Policies", "Intent")

#: The secondary outcomes.
SECONDARY = ("PoliciesSp", "Candidate", "Companies", "IntentNp")

#: What the cross-validation scores: the post-treatment composites plus donation.
#: Pre-treatment composites are built too, because the published estimand uses
#: them as covariates and because a faithful transcript asks them.
SCORED: dict[str, float] = {
    **{f"{name}_Post": 100.0 for name in PRIMARY + SECONDARY},
    "Donation": 100.0,
}

#: Which Pfänder outcome each CCC outcome corresponds to, and how close.
#:
#: ``identical`` means the same question on the same scale — verified verbatim
#: against both questionnaires.  This mapping is why the study is here.
PFANDER_CROSSWALK = {
    "Concern_Post": ("concern_mean", "identical", "all three items verbatim"),
    "Policies_Post": (
        "policy_general",
        "identical",
        "'The U.S. government should do more to reduce global warming'",
    ),
    "IntentNp_Post": (
        "behavior_mean",
        "partial",
        "3 of 6 items verbatim, same stem and scale",
    ),
    "Belief_Post": (
        "belief_post",
        "construct-only",
        "Pfänder condensed three items to one accuracy rating",
    ),
    "PoliciesSp_Post": (
        "policy_specific_mean",
        "construct-only",
        "both 4-item policy batteries, different policies",
    ),
    "Donation": ("donation_ams", "construct-only", "scale differs: cents vs 0-10"),
}


def composite(answers: dict, name: str) -> float | None:
    """One composite from a respondent's on-screen answers, or None if incomplete.

    Applies the reverse-coding the published pipeline applies, so the result is on
    the same footing as the human columns.
    """
    items, _ = COMPOSITES[name]
    values = []
    for item in items:
        raw = answers.get(item)
        if raw is None:
            return None
        value = float(raw)
        if item in REVERSED_ITEMS:
            value = 100.0 - value
        values.append(value)
    total = sum(values)
    return total if name in SUMMED else total / len(values)
