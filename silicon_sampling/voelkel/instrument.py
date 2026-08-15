"""The Strengthening Democracy Challenge instrument, assembled per respondent.

The ``.qsf`` says what every block contains; this module says which blocks a
given respondent walks through, in what order.  Three things vary between
respondents and all three are reproduced:

**Party.** The instrument is party-adaptive throughout.  Most of it adapts by
piped text — the same sentence reads "Republicans" or "Democrats" — and two of
the retained stimuli exist as separate Republican and Democrat blocks, gated in
the survey flow on the respondent's own party.

**Outcome order.** The outcome battery is presented in randomised order, in the
nested groups the survey's own randomisers define, not as one free shuffle.

**Condition.** One stimulus, or none at all for the null control, which goes
straight from the pre-treatment transition to the outcome battery.

Only gender, race and party are asked on screen.  Age, education and ideology
came from the panel supplier and appear nowhere in the instrument, so a synthetic
respondent cannot condition on them — a fact that has to travel with any subgroup
result computed over those three.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ..survey.elements import Block as TranscriptBlock
from ..survey.elements import Conditional, Text
from .convert import build_slot_index, convert_block
from .paths import QSF
from .qsf import load_survey

#: Blocks every respondent sees before the stimulus, in flow order.
PRE_BLOCKS = (
    "Consent",
    "Filter",
    "Demographics",
    "Party",
    "Party Identity",
    "Attention Check 2",
    "Video Check",
    "Transition to Intervention",
)

#: The outcome battery.  Each tuple is one randomiser: its members are shown in
#: random order, and a nested tuple is itself shuffled in place.
POST_GROUPS: tuple[tuple, ...] = (
    (
        (
            "Partisan Animosity: Feeling Thermometers",
            "Partisan Animosity: Dictator Game",
        ),
        ("Support for Partisan Violence",),
        ("Anti-Democratic Attitudes",),
    ),
    (
        ("Support for Undemocratic Candidates",),
        ("Support for Democratic Reform",),
    ),
    (
        ("Biased Evaluation of Politicized Facts",),
        ("Attitudinal Polarization",),
        ("Support for Bipartisanship",),
        ("Partisan Animosity - Voters and Politicians",),
        ("Voting Intentions",),
        ("Social Distance",),
        ("Social Trust",),
    ),
)

#: Shown after the randomised battery, in fixed order.
POST_TAIL = ("Mediators", "Vaccine Intentions", "End")

#: The six retained interventions plus the shared control, from the modality audit.
CONDITIONS = (
    "Null_Control",
    "Party_Overlap",
    "Misperception_Suffering",
    "Partisan_Threat",
    "Misperception_Competition",
    "Harmful_Experiences",
    "System_Justification",
)

#: Embedded fields the instrument pipes into its own text, by respondent party.
PARTY_PIPES = {
    "Republican": {
        "Inparty_Person": "Republican",
        "Inparty_Party": "Republican",
        "Outparty_Person": "Democrat",
        "Outparty_Party": "Democratic",
        "Inparty_Elite": "Donald Trump",
        "Outparty_Friendship_ID": "a Republican",
    },
    "Democrat": {
        "Inparty_Person": "Democrat",
        "Inparty_Party": "Democratic",
        "Outparty_Person": "Republican",
        "Outparty_Party": "Republican",
        "Inparty_Elite": "Joe Biden",
        "Outparty_Friendship_ID": "a Democrat",
    },
}

#: Slots supplied by the profile rather than sampled.  The consent and check
#: items are pre-filled as passed because the published sample contains only
#: respondents who passed them, exactly as in the Pfänder run.
PREFILLED = {
    "Filter": "Yes",
    "Attention_1": "Somewhat disagree",
    "Attention_2": "attention",
    "VideoCheck": None,  # set from the question's own correct option at build time
    "Transition_Int": None,
    "Gender": None,
    "Race": None,
    "Party_Gen": None,
}

#: Stimulus blocks that exist once per party, gated in the flow on the
#: respondent's own party.  Matched on the trailing word, because the two
#: retained cases punctuate differently — "... Suffering - Democrat" against
#: "... Threat Democrat" — and requiring the hyphen silently showed a
#: `Partisan_Threat` respondent *both* versions.
PARTY_VARIANT_SUFFIX = (" Democrat", " Republican")

#: `Misperception_Competition` shows one of two scenarios, then corrects the
#: respondent's own estimates against the true figures.  The true figures are
#: constants in the survey flow, and they differ by party.
CCT5_SCENARIOS = ("2", "4")
CCT5_TOPIC = {"2": "campaign finance law", "4": "financial transparency in government"}
CCT5_TRUTH = {
    ("2", "Democrat"): {"Dis": 49, "Opp": 54, "Una": 50},
    ("2", "Republican"): {"Dis": 54, "Opp": 45, "Una": 44},
    ("4", "Democrat"): {"Dis": 46, "Opp": 48, "Una": 52},
    ("4", "Republican"): {"Dis": 31, "Opp": 31, "Una": 33},
}


@dataclass(frozen=True)
class Loaded:
    """The parsed survey plus everything derived from it once."""

    survey: object
    payloads: dict
    slot_index: dict
    by_description: dict


@lru_cache(maxsize=1)
def loaded() -> Loaded:
    survey = load_survey(QSF)
    payloads = {
        e["PrimaryAttribute"]: e["Payload"]
        for e in json.loads(QSF.read_text(encoding="utf-8"))["SurveyElements"]
        if e["Element"] == "SQ"
    }
    return Loaded(
        survey=survey,
        payloads=payloads,
        slot_index=build_slot_index(survey),
        by_description={block.description: block for block in survey.blocks.values()},
    )


def _elements_of(description: str, page_break: bool = True) -> list:
    state = loaded()
    block = state.by_description[description]
    return convert_block(
        state.survey, block, state.payloads, state.slot_index, page_break=page_break
    ).elements


def data_columns() -> dict[str, str]:
    """Slot id -> published data column, across every block we render."""
    state = loaded()
    columns: dict[str, str] = {}
    for block in state.survey.blocks.values():
        columns.update(
            convert_block(
                state.survey, block, state.payloads, state.slot_index
            ).data_columns
        )
    return columns


def stimulus_blocks(
    condition: str, party: str, scenario: str | None = None
) -> list[str]:
    """Block descriptions for one condition, resolved for this respondent's party."""
    if condition == "Null_Control":
        return []
    state = loaded()
    names = [
        block.description for block in state.survey.condition_only_blocks(condition)
    ]
    if condition == "Misperception_Competition":
        # One of the two scenarios, with its correction; never both.
        keep_scenario = scenario or CCT5_SCENARIOS[0]
        names = [
            name
            for name in names
            if "Scenario" not in name or name.rstrip().endswith(keep_scenario)
        ]
    variants = [name for name in names if name.endswith(PARTY_VARIANT_SUFFIX)]
    if not variants:
        return names
    keep = f" {party}"
    return [
        name
        for name in names
        if not name.endswith(PARTY_VARIANT_SUFFIX) or name.endswith(keep)
    ]


def derive(answers) -> None:
    """Fill embedded fields the instrument computes from earlier answers.

    `Misperception_Competition` quotes the respondent's own three estimates back
    at them on the correction screen, beside the true figures.  Those echoes have
    to exist before that screen renders, and the survey's fallback for an
    unanswered estimate is the literal string below.
    """
    scenario = str(answers.get("CCT5_ScenarioCondition", "") or "")
    if not scenario:
        return
    party = (
        "Republican" if answers.get("Inparty_Person") == "Republican" else "Democrat"
    )
    answers["Scenario_Topic"] = CCT5_TOPIC[scenario]
    for short, value in CCT5_TRUTH[(scenario, party)].items():
        answers[f"S{scenario}_TR_{short}"] = value
    for short in ("Dis", "Opp", "Una"):
        given = answers.get(f"CCT5_S{scenario}_{short}")
        answers[f"S{scenario}_MP_{short}"] = (
            "[not answered]" if given is None else given
        )


def post_order(rng: random.Random) -> list[str]:
    """One respondent's outcome-battery order."""
    order: list[str] = []
    for group in POST_GROUPS:
        members = list(group)
        rng.shuffle(members)
        for member in members:
            block = list(member)
            if len(block) > 1:
                rng.shuffle(block)
            order.extend(block)
    return order + list(POST_TAIL)


def party_conditionals() -> list:
    """The follow-up party question, asked only of the matching partisans."""
    state = loaded()
    block = state.by_description["Party"]
    converted = convert_block(state.survey, block, state.payloads, state.slot_index)
    wanted = {
        "Party_Rep": "Republican",
        "Party_Dem": "Democrat",
        "Party_Ind": "Independent",
    }
    out: list = [
        element for element in converted.elements if not hasattr(element, "id")
    ]
    for element in converted.elements:
        target = wanted.get(getattr(element, "id", ""))
        if target is None:
            continue
        out.append(
            Conditional(
                note=f'Party_Gen = "{target}"',
                predicate=(
                    lambda value: lambda answers: answers.get("Party_Gen") == value
                )(target),
                elements=[element],
            )
        )
    return out


def elements_for(
    condition: str,
    party: str,
    battery: Sequence[str] | None = None,
    scenario: str | None = None,
) -> list:
    """The full element sequence one respondent walks through."""
    elements: list = []
    for name in PRE_BLOCKS:
        if name == "Party":
            elements.append(
                TranscriptBlock(
                    key="party", title="Party", elements=party_conditionals()
                )
            )
            continue
        elements.append(
            TranscriptBlock(key=_key(name), title=name, elements=_elements_of(name))
        )

    for name in stimulus_blocks(condition, party, scenario):
        elements.append(
            TranscriptBlock(key=_key(name), title=name, elements=_elements_of(name))
        )

    elements.append(
        TranscriptBlock(
            key="transition_dvs",
            title="Transition to DVs",
            elements=_elements_of("Transition to DVs"),
        )
    )
    for name in battery if battery is not None else post_order(random.Random(0)):
        elements.append(
            TranscriptBlock(key=_key(name), title=name, elements=_elements_of(name))
        )
    return elements


def _key(description: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in description.lower()
    ).strip("_")


def header(profile_id: str, condition: str) -> str:
    """The transcript preamble, dated to the study's own fielding window."""
    index = CONDITIONS.index(condition)
    return "\n".join(
        [
            "=" * 78,
            " STRENGTHENING DEMOCRACY CHALLENGE",
            " A megastudy of interventions on partisan animosity and antidemocratic attitudes",
            " Response transcripts, one file per participant.",
            "-" * 78,
            f" File           : responses/{profile_id}.txt",
            f" Participant ID : {profile_id}",
            f" Condition      : {index:02d}",
            " Instrument     : US national sample, fielded April–May 2022",
            " Note           : Verbatim record of one session, screens in the order they were",
            '                  displayed. Lines beginning "Response:" hold what the participant',
            "                  entered.",
            "=" * 78,
        ]
    )


def transition_text() -> Text:  # pragma: no cover - convenience for the report
    return Text("")
