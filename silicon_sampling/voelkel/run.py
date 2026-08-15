"""Wiring for the Voelkel sampling run."""

from __future__ import annotations

import random

from ..sampling.driver import token_budget
from ..sampling.runner import Runner
from ..survey.render import walk
from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from . import instrument as inst
from .profiles import Profile


def session_for(profile: Profile) -> Session:
    """The session one profile walks through."""
    elements = inst.elements_for(
        profile.condition,
        profile.inparty,
        battery=profile.battery.split("|") if profile.battery else None,
        scenario=profile.scenario or None,
    )
    return Session(
        header=inst.header(profile.profile_id, profile.condition),
        elements=elements,
        answers=dict(profile.prefilled),
        derive=inst.derive,
    )


def record_for(profile: Profile, session: Session) -> dict:
    answers = {
        key: value for key, value in session.answers.items() if not key.startswith("_")
    }
    return {
        "profile_id": profile.profile_id,
        "condition": profile.condition,
        "party_gen": profile.party_gen,
        "inparty": profile.inparty,
        "gender": profile.gender,
        "race": profile.race,
        "age": profile.age,
        "age_band": profile.age_band,
        "education": profile.education,
        "ideology": profile.ideology,
        "scenario": profile.scenario,
        "n_asked": len(session.asked),
        "answers": answers,
    }


def shard_for(profile: Profile) -> str:
    return profile.condition.lower()


def _all_slots() -> dict[str, Slot]:
    found: dict[str, Slot] = {}
    for condition in inst.CONDITIONS:
        for party in ("Republican", "Democrat"):
            scenarios = (
                inst.CCT5_SCENARIOS
                if condition == "Misperception_Competition"
                else (None,)
            )
            for scenario in scenarios:
                elements = inst.elements_for(
                    condition,
                    party,
                    battery=inst.post_order(random.Random(0)),
                    scenario=scenario,
                )
                for event, payload in walk(elements):
                    if event == "slot":
                        found.setdefault(payload.id, payload)
    return found


def fit_token_budgets(model: str) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
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


def max_transcript_tokens(model: str, shard: str | None = None) -> int:
    """Worst-case transcript length, for the condition named by ``shard``.

    Sized per condition because one arm (`Party_Overlap`) runs to twice the
    length of the rest; sizing every group to that worst case would halve the
    concurrency of the six short arms for no reason.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    conditions = [c for c in inst.CONDITIONS if shard is None or c.lower() == shard]
    longest = 0
    for condition in conditions:
        for party in ("Republican", "Democrat"):
            scenarios = (
                inst.CCT5_SCENARIOS
                if condition == "Misperception_Competition"
                else (None,)
            )
            for scenario in scenarios:
                elements = inst.elements_for(
                    condition,
                    party,
                    battery=inst.post_order(random.Random(0)),
                    scenario=scenario,
                )
                session = Session(
                    header=inst.header("v00000", condition),
                    elements=elements,
                    answers={
                        **inst.PARTY_PIPES[party],
                        "Party_Gen": party,
                        "Gender": "Male",
                        "Race": "White / Caucasian",
                        **({"CCT5_ScenarioCondition": scenario} if scenario else {}),
                    },
                    derive=inst.derive,
                )
                while (step := session.next_prompt()) is not None:
                    session.submit(step[1], _widest(step[1]))
                longest = max(
                    longest,
                    len(
                        tokenizer(session.transcript(), add_special_tokens=False)[
                            "input_ids"
                        ]
                    ),
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
