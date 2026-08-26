"""Wiring for the Climate Change Challenge sampling run."""

from __future__ import annotations

import random

from ..sampling.driver import token_budget
from ..sampling.runner import Runner
from ..sampling.tokens import load_tokenizer
from ..survey.render import walk
from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from . import instrument as inst
from .profiles import Profile


def _click_throughs(elements) -> dict:
    """Single-option items: buttons the respondent pressed, not questions.

    Qualtrics records an "I understand" click as a one-option multiple choice.
    There is nothing to sample — the only legal answer is the button's own label —
    so asking for it spends draws and manufactures rejections.
    """
    filled = {}
    for event, payload in walk(elements):
        if (
            event == "slot"
            and isinstance(payload, ChoiceSlot)
            and len(payload.options) == 1
        ):
            filled[payload.id] = payload.options[0]
    return filled


def session_for(profile: Profile) -> Session:
    """The session one profile walks through.

    The battery order is drawn from the profile's own seed, so a respondent's
    randomised block order is reproducible and identical across models — which is
    what makes the models comparable respondent by respondent.
    """
    elements = inst.elements_for(profile.condition, random.Random(profile.seed))
    answers = {**_click_throughs(elements), **profile.prefilled}
    return Session(
        header=inst.header(profile.profile_id, profile.condition),
        elements=elements,
        answers=answers,
    )


def record_for(profile: Profile, session: Session) -> dict:
    answers = {
        key: value for key, value in session.answers.items() if not key.startswith("_")
    }
    return {
        "profile_id": profile.profile_id,
        "condition": profile.condition,
        "gender": profile.gender,
        "age": profile.age,
        "age_band": profile.age_band,
        "race": profile.race,
        "education": profile.education,
        "party": profile.party,
        "n_asked": len(session.asked),
        "answers": answers,
    }


def shard_for(profile: Profile) -> str:
    """Group by arm, so a group shares one prefix and one transcript length."""
    return profile.condition.lower().replace(" ", "_")


def _all_slots() -> dict[str, Slot]:
    found: dict[str, Slot] = {}
    for condition in inst.conditions():
        elements = inst.elements_for(condition, random.Random(0))
        for event, payload in walk(elements):
            if event == "slot":
                found.setdefault(payload.id, payload)
    return found


def fit_token_budgets(model: str) -> dict[str, int]:
    tokenizer = load_tokenizer(model)
    return {
        slot_id: token_budget(slot, tokenizer) for slot_id, slot in _all_slots().items()
    }


def _widest(slot: Slot):
    if isinstance(slot, ChoiceSlot):
        return max(slot.options, key=len)
    if isinstance(slot, IntSlot):
        return max((slot.lo, slot.hi), key=lambda value: len(str(value)))
    if isinstance(slot, FreeTextSlot):
        return "x " * min(slot.max_chars // 2, slot.max_tokens)
    return ""


def worst_case_sessions(shard: str | None = None):
    """One session per arm, for transcript sizing and the tokenisation test."""
    for condition in inst.conditions():
        if shard is not None and shard_for_condition(condition) != shard:
            continue
        yield Session(
            header=inst.header("c00000", condition),
            elements=inst.elements_for(condition, random.Random(0)),
            answers={
                "Filter": "Yes",
                "Attention1": "Somewhat disagree",
                "Gender": "Male",
                "Race": "White / Caucasian",
                "Education": "Bachelor's degree",
                "ANES_Gen": "Independent",
                "YOB": "1980",
            },
        )


def shard_for_condition(condition: str) -> str:
    return condition.lower().replace(" ", "_")


def max_transcript_tokens(model: str, shard: str | None = None) -> int:
    """Worst-case transcript length for one arm.

    Sized per arm because the arms differ: Gains Framing carries 4,693 characters
    of stimulus against Binding Framing's 1,538, and sizing every group to the
    longest would cut the concurrency of the short arms for no reason.
    """
    tokenizer = load_tokenizer(model)
    longest = 0
    for session in worst_case_sessions(shard):
        while (step := session.next_prompt()) is not None:
            session.submit(step[1], _widest(step[1]))
        longest = max(
            longest,
            len(tokenizer(session.transcript(), add_special_tokens=False)["input_ids"]),
        )
    return int(longest * 1.05)


def make_runner(out_dir, engine_config, sampler_config) -> Runner:
    return Runner(
        out_dir,
        engine_config,
        sampler_config,
        session_for=session_for,
        record_for=record_for,
        shard_for=shard_for,
        token_budgets=fit_token_budgets,
        worst_case_tokens=max_transcript_tokens,
    )
