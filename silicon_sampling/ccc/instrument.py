"""The Climate Change Challenge instrument, assembled per respondent.

The ``.qsf`` says what each block contains; this module says which blocks a
respondent walks through and in what order.  Recovered from the survey flow rather
than assumed:

1. Consent, Filter, Demographics, Party Identification, Attention Check 2,
   Transition to Study, Issue Positions
2. a randomiser over the four **pre-treatment primary** batteries
3. a randomiser over the four **pre-treatment secondary** batteries
4. Transition from Pre-Treatment to Treatment
5. one arm, chosen by a single ``BlockRandomizer`` with ``EvenPresentation``
6. Transition from Treatment to Post-Treatment
7. a randomiser over the four **post-treatment primary** batteries
8. a randomiser over the four post-treatment secondary batteries **plus donation**
9. Manipulation Checks, Other Questions, End of Survey

Three things about this study shape the transcript.

**Every primary outcome is measured twice**, before and after the message.  That is
unlike the other four reference studies and it is not optional: the published
estimand regresses the post-measure on condition *and* the pre-measure, and a
respondent who never answered the pre-battery cannot be scored that way.

**One arm is dropped.**  ``System Preservation Framing`` is six pages of twelve
images with about 1,200 characters of prose that never refers to them; the images
are the concrete "American way of life" the last page asks the reader to preserve.
That is the same "core is not in the text" rule that dropped seven Goldwert arms.
Twelve of thirteen remain, including all three placebo controls.

**The Warmth arm contains a writing task**, the only response slot inside any arm.
Human median dwell was 130.8 s on that page against 17.0 s on the arm's first page,
making it the longest arm of the thirteen by 40%.  Its answer was collected but
never released, so nothing can be scored from it — and omitting it would remove
what the arm actually does.
"""

from __future__ import annotations

import random
from functools import lru_cache

from ..survey.elements import Text
from .convert import convert_block
from .paths import QSF

PRE_BLOCKS = (
    "Consent Form",
    "Filter",
    "Demographics",
    "Party Identification",
    "Attention Check 2",
    "Transition to Study",
    "Issue Positions",
)

PRE_PRIMARY = (
    "Belief in Climate Change - PRE",
    "Climate Change Concern - PRE",
    "Support for General Climate Change Mitigation Policies - PRE",
    "Political Behavioral Intentions - PRE",
)

PRE_SECONDARY = (
    "Support for Specific Climate Change Mitigation Policies - PRE",
    "Support for Pro-Environmental Candidates - PRE",
    "Support for Company-Led Climate Change Mitigation - PRE",
    "Non-Political Behavioral Intentions - PRE",
)

TO_TREATMENT = "Transition from Pre-Treatment to Treatment"
FROM_TREATMENT = "Transition from Treatment to Post-Treatment"

POST_PRIMARY = (
    "Belief in Climate Change - POST",
    "Climate Change Concern - POST",
    "Support for General Climate Change Mitigation Policies - POST",
    "Political Behavioral Intentions - POST",
)

#: The donation rides in this randomiser, which is why its SubSet is 5 not 4.
POST_SECONDARY = (
    "Support for Specific Climate Change Mitigation Policies - POST",
    "Support for Pro-Environmental Candidates - POST",
    "Support for Company-Led Climate Change Mitigation - POST",
    "Non-Political Behavioral Intentions - POST",
    "Donation Behavior - Only POST",
)

TAIL = ("Manipulation Checks", "Other Questions", "End of Survey")

#: Condition label in the released data -> the block that arm shows.
#:
#: The labels are the raw ``Condition`` strings, not the paper's renamings: the
#: R scripts relabel Free Market as "Compatible Solution" and Consensus 1/2 as
#: "Scientific Consensus 1/2" for publication, and the data does not.
ARM_BLOCKS = {
    "Control Neckties": "History of Neckties",
    "Control Baseball": "Rules of Baseball",
    "Control Dances": "Different Types of Dances",
    "Binding Framing": "Binding Framing",
    "Consensus Framing 1": "Consensus Framing I",
    "Consensus Framing 2": "Consensus Framing II",
    "Dire But Solvable Framing": "Dire But Solvable Framing",
    "Free Market Framing": "Free Market Framing",
    "Gains Framing": "Gains Framing",
    "High Social Distance Framing": "High Social Distance Framing",
    "Purity Framing": "Purity Framing",
    "Warmth Framing": "Warmth Framing",
}

#: Dropped, with the reason, so the exclusion travels with the data.
DROPPED_ARMS = {
    "System Preservation Framing": (
        "six pages, twelve images, all URLs dead, ~1,200 characters of prose that "
        "never refers to them; the images are the intervention"
    ),
}

#: The three placebo arms the published analysis pools into one ``Control``.
CONTROL_ARMS = ("Control Neckties", "Control Baseball", "Control Dances")

#: How much a text-only rendering loses, per retained arm, from the fidelity audit.
#: 0 nothing, 1 peripheral, 2 substantive, 3 the intervention itself (dropped).
MEDIA_LOSS = {
    "Consensus Framing 2": 0,
    "Gains Framing": 0,
    "Free Market Framing": 0,
    "Control Neckties": 1,
    "Control Baseball": 1,
    "Control Dances": 1,
    "Consensus Framing 1": 1,
    "Binding Framing": 1,
    "High Social Distance Framing": 2,
    "Purity Framing": 2,
    "Warmth Framing": 2,
    "Dire But Solvable Framing": 2,
}

#: Notes restoring content an image carried, where the record supports it.
#:
#: Only Purity qualifies.  Its prose points deictically at the picture ("On the
#: left is what visitors used to see … on the right is what visitors see now"),
#: its URL is one of the four still live, and measuring the fetched image confirmed
#: the halves differ by a 3.2x saturation collapse — the clear-day/hazy-day pair
#: the text claims.  So the content is recoverable in words.
#:
#: Nothing is written for the other arms.  Eighteen of twenty-two URLs are dead and
#: inventing a description would be worse than admitting the gap.
ARM_IMAGE_NOTES = {
    "Purity Framing": (
        "[The picture referred to is a single side-by-side photograph of the Great "
        "Smoky Mountains. The left half shows a clear, vivid, far-reaching view; the "
        "right half shows the same view washed out and hazy with polluted air.]"
    ),
}

#: Answers supplied by the profile rather than sampled.
#:
#: Consent, the filter and both attention checks are pre-filled as passed, because
#: the released sample contains only respondents who passed them — sampling them
#: would create a selection effect with no counterpart in the target data. Every
#: respondent in the released file has ``Attention1 == 3``.
PREFILLED_PASSES = {
    "Filter": "Yes",
    "Attention1": "Somewhat disagree",
}

#: Demographic slots taken from the profile, so composition matches the study's
#: own sample rather than whatever a base model invents.
PROFILE_SLOTS = ("Gender", "YOB", "Race", "Education", "ANES_Gen")


@lru_cache(maxsize=1)
def survey():
    """The parsed questionnaire.  ``voelkel.qsf`` reads it unmodified."""
    from ..voelkel.qsf import load_survey

    return load_survey(QSF, condition_field="Condition")


@lru_cache(maxsize=1)
def _blocks_by_name() -> dict:
    out = {}
    for block in survey().blocks.values():
        name = getattr(block, "title", None) or getattr(block, "description", "")
        out[name] = block
    return out


@lru_cache(maxsize=1)
def _payloads() -> dict:
    import json

    raw = json.loads(QSF.read_text(encoding="utf-8"))
    return {
        element["Payload"].get("QuestionID"): element["Payload"]
        for element in raw["SurveyElements"]
        if element.get("Element") == "SQ" and isinstance(element.get("Payload"), dict)
    }


def conditions() -> tuple[str, ...]:
    """The retained arms, controls first."""
    treatments = tuple(a for a in ARM_BLOCKS if a not in CONTROL_ARMS)
    return CONTROL_ARMS + treatments


def block_order(condition: str, rng: random.Random) -> list[str]:
    """Every block this respondent sees, in flow order with randomisers drawn."""
    primary_pre = list(PRE_PRIMARY)
    rng.shuffle(primary_pre)
    secondary_pre = list(PRE_SECONDARY)
    rng.shuffle(secondary_pre)
    primary_post = list(POST_PRIMARY)
    rng.shuffle(primary_post)
    secondary_post = list(POST_SECONDARY)
    rng.shuffle(secondary_post)
    return [
        *PRE_BLOCKS,
        *primary_pre,
        *secondary_pre,
        TO_TREATMENT,
        ARM_BLOCKS[condition],
        FROM_TREATMENT,
        *primary_post,
        *secondary_post,
        *TAIL,
    ]


def elements_for(condition: str, rng: random.Random | None = None) -> list:
    """The whole transcript for one respondent, as transcript elements."""
    rng = rng or random.Random(0)
    known = _blocks_by_name()
    payloads = _payloads()
    arm_block = ARM_BLOCKS[condition]
    note = ARM_IMAGE_NOTES.get(condition)
    elements: list = []
    for name in block_order(condition, rng):
        block = known.get(name)
        if block is None:
            continue
        converted = convert_block(survey(), block, payloads)
        elements.extend(converted.elements)
        if name == arm_block and note:
            elements.append(Text(note))
    return elements


def data_columns() -> dict:
    """Slot id -> released column, over every fielded block."""
    known = _blocks_by_name()
    payloads = _payloads()
    columns: dict = {}
    for name, block in known.items():
        if name in ("Trash / Unused Questions",):
            continue
        columns.update(convert_block(survey(), block, payloads).data_columns)
    return columns


def header(profile_id: str, condition: str) -> str:
    """The transcript preamble, matching the other studies' convention."""
    return (
        f"Respondent {profile_id} | study: Climate Change Challenge "
        f"(Voelkel et al. 2026) | condition: {condition}"
    )
