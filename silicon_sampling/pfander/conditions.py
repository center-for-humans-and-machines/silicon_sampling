"""The 17 conditions: one shared control (three filler texts) and 16 interventions.

Fifteen of the nineteen stimulus blocks are plain prose and are built straight
from ``questionnaire.txt``, split into screens at the source file's own page-break
markers.  Four carry response positions inside the treatment and are structured
by hand:

``Funding``
    four agreement sliders, then three belief-then-rebuttal cycles.
``High public trust``
    an estimate of American trust, then the corrective statistic.
``Consensus``
    three consensus estimates, each followed immediately by its feedback.  Item
    order is randomised with item 3 always in the middle, as the survey specifies.
``Extreme weather predictions``
    state-adaptive: the respondent names their state, which selects one of four
    case texts and fills the intro paragraph.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

from ..survey.elements import Block, Conditional, PageBreak, Text
from ..survey.slots import ChoiceSlot
from . import sources
from .instrument import SLIDER_BLURB, STATE_OPTIONS, slider

AGREE_100 = "0 = Strongly disagree … 100 = Strongly agree."

#: Canonical condition titles, in the benchmark's own order.
INTERVENTIONS = (
    "Corporate reliance",
    "Social justice",
    "Interview Prof. Maraun",
    "Funding",
    "Oil industry misinformation",
    "Measurement & modeling (1)",
    "Former skeptics",
    "High public trust",
    "Measurement & modeling (2)",
    "Peer-review",
    "Scientist community helpers",
    "Consensus",
    "Portrait Prof. Cherry",
    "Model accuracy",
    "Interview Prof. Sebille",
    "Extreme weather predictions",
)

CONDITIONS = ("control",) + INTERVENTIONS

#: The three control filler texts, keyed by the survey's own code names.
CONTROL_TEXTS = {
    "control neckties": "control — filler text 1 of 3: The History of Neckties",
    "control baseball": "control — filler text 2 of 3: The Rules of Baseball",
    "control dances": "control — filler text 3 of 3: Different Types of Dances",
}

CASE_LABELS = {
    1: "states with high or recurrent flood risk",
    2: "states with high or increasing wildfire risk",
    3: "states with severe cold, snow, ice, or blizzards",
}


def slug(title: str) -> str:
    """Filesystem-safe form of a condition title."""
    text = title.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _prose_pages(heading: str) -> list[object]:
    """Elements for a stimulus that is nothing but text and page breaks."""
    elements: list[object] = []
    for index, page in enumerate(sources.pages(heading)):
        if index:
            elements.append(PageBreak())
        elements += [Text(paragraph) for paragraph in page]
    return elements


# --------------------------------------------------------------------------- #
# Funding
# --------------------------------------------------------------------------- #

_FUNDING_AGREEMENT = (
    (
        "funding_intv_fairness",
        "Everyone should be held to the same standards of honesty and fairness.",
    ),
    (
        "funding_intv_taxpayer",
        "It is important that taxpayer-funded programs show exactly how the taxmoney is spent.",
    ),
    (
        "funding_intv_corporations",
        "Corporations have too much influence on what gets researched.",
    ),
    (
        "funding_intv_powerful",
        "Some people in powerful positions push certain ideas not because they're true, but because they fit their political or financial interests.",
    ),
)


def _funding_elements() -> list[object]:
    page_texts = sources.pages("Funding")
    # Screens 4, 6 and 8 are the three rebuttals, 9 the closing frame; indices
    # follow the source file's own page breaks.
    rebuttal_paid, rebuttal_federal, rebuttal_private, closing = (
        page_texts[4],
        page_texts[6],
        page_texts[8],
        page_texts[9],
    )
    elements: list[object] = [
        Text(
            "Please indicate how much you agree or disagree with the following statements."
        ),
        PageBreak(),
        Text("How much do you agree or disagree with the following statements?"),
        Text(SLIDER_BLURB),
    ]
    elements += [
        slider(slot_id, stem, AGREE_100) for slot_id, stem in _FUNDING_AGREEMENT
    ]
    elements += [
        PageBreak(),
        Text("Thank you for sharing your thoughts."),
        Text(
            "Many Americans care deeply about fairness, honesty, and transparency in public decision-making."
        ),
        Text(
            "Over the next few pages, we will ask you to answer questions about your beliefs and attitudes concerning climate scientists."
        ),
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "funding_intv_paid",
            '"Climate scientists are paid to support certain climate policies."',
            AGREE_100,
        ),
        PageBreak(),
        *[Text(paragraph) for paragraph in rebuttal_paid],
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "funding_intv_federal",
            '"The federal government allocates significant resources to climate change research."',
            AGREE_100,
        ),
        PageBreak(),
        *[Text(paragraph) for paragraph in rebuttal_federal],
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "funding_intv_private",
            '"Climate scientists receive large amounts of private research funding."',
            AGREE_100,
        ),
        PageBreak(),
        *[Text(paragraph) for paragraph in rebuttal_private],
        PageBreak(),
        *[Text(paragraph) for paragraph in closing],
    ]
    return elements


# --------------------------------------------------------------------------- #
# High public trust
# --------------------------------------------------------------------------- #


def _high_public_trust_elements() -> list[object]:
    corrective = sources.pages("High public trust")[1]
    return [
        Text(SLIDER_BLURB),
        slider(
            "hpt_estimate",
            "Please provide your best estimate: What percentage of Americans trust climate scientists to provide full and accurate information on climate change?",
            "0 = 0% of Americans … 100 = 100% of Americans.",
        ),
        PageBreak(),
        *[Text(paragraph) for paragraph in corrective],
    ]


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #

_CONSENSUS_ITEMS = {
    1: '"Human activities are the primary cause of global warming since the mid-20th century."',
    2: '"Increasing carbon dioxide in the atmosphere warms the planet."',
    3: '"The world will reach net-zero CO₂ emissions before 2085".',
}


@lru_cache(maxsize=1)
def _consensus_feedback() -> dict[int, list[str]]:
    """The three feedback texts, split out of the Consensus stimulus."""
    body = sources.stimulus("Consensus")
    start = body.index("Feedback: Given directly after each item")
    tail = body[start:]
    summary_at = tail.index("\nSummary:")
    feedback_region, summary = tail[:summary_at], tail[summary_at:]
    out: dict[int, list[str]] = {}
    for chunk in re.split(r"\n(?=[123]\) )", feedback_region):
        match = re.match(r"([123])\) ", chunk.strip())
        if match:
            out[int(match.group(1))] = _body_and_source(chunk.strip()[3:])
    out[0] = sources.paragraphs(summary.replace("Summary:", "", 1))
    return out


def _body_and_source(text: str) -> list[str]:
    """Paragraphs, with a trailing ``Source:`` citation kept on its own line."""
    parts = re.split(r"\n\s*Source:\s*\n", text, maxsplit=1)
    out = sources.paragraphs(parts[0])
    if len(parts) > 1:
        out.append("Source: " + " ".join(parts[1].split()))
    return out


#: Item 3 always sits in the middle; items 1 and 2 take the outer slots.
CONSENSUS_ORDERS = ((1, 3, 2), (2, 3, 1))


def _consensus_elements(order: Sequence[int]) -> list[object]:
    feedback = _consensus_feedback()
    intro = sources.paragraphs(sources.stimulus("Consensus"))[:2]
    elements: list[object] = [Text(paragraph) for paragraph in intro]
    for item in order:
        elements += [
            PageBreak(),
            Text(SLIDER_BLURB),
            slider(
                f"consensus_{item}",
                f"Please indicate the percentage of scientists you think agree with the following statement: {_CONSENSUS_ITEMS[item]}",
                "0 = 0% of scientists … 100 = 100% of scientists.",
            ),
            PageBreak(),
            *[Text(paragraph) for paragraph in feedback[item]],
        ]
    elements += [PageBreak(), *[Text(paragraph) for paragraph in feedback[0]]]
    return elements


# --------------------------------------------------------------------------- #
# Extreme weather predictions
# --------------------------------------------------------------------------- #

_CASE_1_STATES = {
    "Alabama",
    "Arkansas",
    "Delaware",
    "Florida",
    "Georgia",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maryland",
    "Mississippi",
    "Missouri",
    "Nebraska",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Pennsylvania",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Virginia",
    "West Virginia",
    "Washington, D.C.",
}
_CASE_2_STATES = {
    "Alaska",
    "Arizona",
    "California",
    "Colorado",
    "Idaho",
    "Montana",
    "Nevada",
    "New Mexico",
    "Oregon",
    "Utah",
    "Washington",
    "Wyoming",
    "Hawaii",
}
_CASE_3_STATES = {
    "Connecticut",
    "Maine",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "New Hampshire",
    "New Jersey",
    "New York",
    "Rhode Island",
    "Vermont",
    "Wisconsin",
}


def case_for_state(state: str | None) -> int:
    """Map a reported home state to one of the four stimulus cases."""
    if state in _CASE_2_STATES:
        return 2
    if state in _CASE_3_STATES:
        return 3
    if state in _CASE_1_STATES:
        return 1
    return 4


@lru_cache(maxsize=1)
def _weather_cases() -> dict[int, list[str]]:
    """The four case texts, split out of the state-adaptive stimulus."""
    body = sources.stimulus("Extreme weather predictions")
    region = body[
        body.index("Intervention page 3") : body.index(
            "References [not displayed to participants]"
        )
    ]
    out: dict[int, list[str]] = {}
    for chunk in re.split(r"\n(?=Case [1-4]\s*$)", region, flags=re.MULTILINE):
        lines = [line.strip() for line in chunk.strip().splitlines() if line.strip()]
        match = re.fullmatch(r"Case ([1-4])", lines[0]) if lines else None
        if match:
            # First line is the case's on-screen headline, the rest its paragraphs.
            out[int(match.group(1))] = lines[1:]
    return out


def _weather_elements() -> list[object]:
    cases = _weather_cases()
    state_slot = ChoiceSlot(
        id="state",
        prompt='Which U.S. state do you currently live in? (You may choose not to answer. If so, please select "Prefer not to say.")',
        options=STATE_OPTIONS,
        max_tokens=8,
    )
    elements: list[object] = [state_slot, PageBreak()]
    elements.append(
        Conditional(
            note='state = "Prefer not to say"',
            predicate=lambda a: a.get("state") == "Prefer not to say",
            elements=[
                Text(
                    "You are living in the United States, a country facing risks by more and more extreme weather "
                    "events. Please read the text on the following page carefully. It describes a real project in the "
                    "U.S., working particularly on reducing the risks from these hazards by helping communities "
                    "prepare for extreme weather."
                )
            ],
        )
    )
    elements.append(
        Conditional(
            note='state is any state or "Washington, D.C."',
            predicate=lambda a: a.get("state") not in (None, "Prefer not to say"),
            elements=[
                Text(
                    "You reported that you are currently living in <<=state>>, one of several <<=_case_label>>. Please "
                    "read the text on the following page carefully. It describes a real project in the U.S., working "
                    "particularly on reducing the risks from these hazards by helping communities prepare for extreme "
                    "weather."
                )
            ],
        )
    )
    elements.append(PageBreak())
    for case in (1, 2, 3, 4):
        elements.append(
            Conditional(
                note=f"the respondent's state maps to case {case}",
                predicate=(lambda c: lambda a: case_for_state(a.get("state")) == c)(
                    case
                ),
                elements=[Text(line) for line in cases[case]],
            )
        )
    return elements


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

#: Stimuli that are prose only, keyed by condition title.
_PLAIN = {
    "Corporate reliance": "Corporate reliance",
    "Social justice": "Social justice",
    "Interview Prof. Maraun": "Interview Prof. Maraun",
    "Oil industry misinformation": "Oil industry misinformation",
    "Measurement & modeling (1)": "Measurement & modeling (1)",
    "Former skeptics": "Former skeptics",
    "Measurement & modeling (2)": "Measurement & modeling (2)",
    "Peer-review": "Peer-review",
    "Scientist community helpers": "Scientist community helpers",
    "Portrait Prof. Cherry": "Portrait Prof. Cherry",
    "Model accuracy": "Model accuracy",
    "Interview Prof. Sebille": "Interview Prof. Sebille",
}


@lru_cache(maxsize=None)
def condition_block(
    title: str,
    *,
    control_text: str | None = None,
    consensus_order: Sequence[int] = (1, 3, 2),
) -> Block:
    """The stimulus block one respondent sees.

    ``control_text`` names which filler text a control respondent reads (one of
    the keys of :data:`CONTROL_TEXTS`); ``consensus_order`` fixes the item order
    for the Consensus arm.
    """
    if title == "control":
        code_name = control_text or "control neckties"
        return Block(
            key=slug(code_name),
            title="control",
            elements=_prose_pages(CONTROL_TEXTS[code_name]),
        )
    if title in _PLAIN:
        return Block(key=slug(title), title=title, elements=_prose_pages(_PLAIN[title]))
    if title == "Funding":
        return Block(key="funding", title=title, elements=_funding_elements())
    if title == "High public trust":
        return Block(
            key="high_public_trust", title=title, elements=_high_public_trust_elements()
        )
    if title == "Consensus":
        return Block(
            key="consensus", title=title, elements=_consensus_elements(consensus_order)
        )
    if title == "Extreme weather predictions":
        return Block(
            key="extreme_weather_predictions", title=title, elements=_weather_elements()
        )
    raise KeyError(f"unknown condition: {title!r}")
