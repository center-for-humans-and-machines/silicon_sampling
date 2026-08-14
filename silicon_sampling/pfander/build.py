"""Assemble one respondent's instrument, or one condition's template.

The transcript is presented as a per-participant response file from a public data
deposit.  The header names the condition by its *internal code name* only — the
identifier the raw survey files actually use — and never by its descriptive title
("Oil industry misinformation"), which is a researcher's evaluative label that
would leak the experimenter's framing into what the respondent is supposed to be
reacting to.
"""

from __future__ import annotations

from typing import MutableMapping, Sequence

from ..survey.elements import Block
from ..survey.session import Session
from . import conditions, instrument
from .conditions import CASE_LABELS, CONDITIONS, case_for_state

STUDY_TITLE = (
    "16 information interventions to strengthen trust in climate scientists in the US"
)

#: Code name shown in the header for each condition; control varies per respondent.
CODE_NAMES = {
    "Corporate reliance": "giant gibbon; brick bobcat",
    "Social justice": "difficult dog",
    "Interview Prof. Maraun": "flimsy fish",
    "Funding": "phony parrotfish",
    "Oil industry misinformation": "worse wildfowl",
    "Measurement & modeling (1)": "perfect prawn",
    "Former skeptics": "limping llama; friendly frog",
    "High public trust": "crushing chicken; gross grasshopper; homely halibut",
    "Measurement & modeling (2)": "orchid orangutan; defiant dragonfly",
    "Peer-review": "honored haddock",
    "Scientist community helpers": "periwinkle partridge",
    "Consensus": "jealous jaguar",
    "Portrait Prof. Cherry": "complicated cockroach",
    "Model accuracy": "apple aardvark",
    "Interview Prof. Sebille": "heartfelt hummingbird",
    "Extreme weather predictions": "practical planarian",
}


def header(profile_id: str, condition: str, code_name: str) -> str:
    """The transcript preamble."""
    number = CONDITIONS.index(condition)
    return "\n".join(
        [
            "=" * 78,
            " THE SILICON SAMPLE BENCHMARK — PARENT MEGASTUDY",
            f' "{STUDY_TITLE}"',
            " Response transcripts, one file per participant.",
            "-" * 78,
            f" File           : responses/{profile_id}.txt",
            f" Participant ID : {profile_id}",
            f" Condition      : {number:02d}  [code name: {code_name}]",
            f" Instrument     : US master version, fielded {instrument.SURVEY_YEAR}",
            " Note           : Verbatim record of one session, screens in the order they were",
            '                  displayed. Lines beginning "Response:" hold what the participant',
            "                  entered.",
            "=" * 78,
        ]
    )


def elements_for(
    condition: str,
    *,
    control_text: str | None = None,
    consensus_order: Sequence[int] = (1, 3, 2),
    post_order: Sequence[Block] | None = None,
) -> list[object]:
    """The full element sequence one respondent walks through."""
    stimulus = conditions.condition_block(
        condition, control_text=control_text, consensus_order=consensus_order
    )
    tail = (
        list(post_order) if post_order is not None else list(instrument.POST_RANDOMISED)
    )
    return [
        *instrument.PRE_CONDITION,
        stimulus,
        instrument.TRANSITION_TO_OUTCOMES,
        instrument.POST_PRIMARY,
        *tail,
        instrument.END_OF_SURVEY,
    ]


def derive(answers: MutableMapping[str, object]) -> None:
    """Fill answer-dependent piped text.

    The state-adaptive arm echoes the respondent's state *and* the risk phrase
    that names its case, so the phrase has to exist before that screen renders.
    """
    state = answers.get("state")
    if state is not None:
        answers["_case_label"] = CASE_LABELS.get(case_for_state(state), "")


def make_session(
    profile_id: str,
    condition: str,
    *,
    code_name: str,
    answers: MutableMapping[str, object] | None = None,
    control_text: str | None = None,
    consensus_order: Sequence[int] = (1, 3, 2),
    post_order: Sequence[Block] | None = None,
) -> Session:
    """A session ready to be walked slot by slot."""
    return Session(
        header=header(profile_id, condition, code_name),
        elements=elements_for(
            condition,
            control_text=control_text,
            consensus_order=consensus_order,
            post_order=post_order,
        ),
        answers=answers,
        derive=derive,
    )


def template_code_name(condition: str) -> str:
    return "control neckties" if condition == "control" else CODE_NAMES[condition]
