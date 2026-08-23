"""Wiring that turns the Goldwert instrument into a restartable sampling run.

The run itself is study-independent and lives in
:class:`silicon_sampling.sampling.runner.Runner`; everything here is the
study-specific parts it asks for.  Subclassing rather than re-implementing is
deliberate: resumability in this project is a property of the *write order* —
transcript before answer record, `fsync` per group, torn final line truncated
before any append — and a second copy of that ordering is a second thing that can
drift out of agreement with the first.  The constructor signature is re-declared
so the ``Runner(out_dir, EngineConfig(...), SamplerConfig(...))`` contract the
launcher scripts depend on still holds.

Two things about this study needed a decision here rather than in the instrument.

**One arm's page text was written by JavaScript, and a session cannot render
without standing in for it.**  ``MispCorrectionRisks`` pipes five embedded fields
onto its pages — the correct/incorrect feedback (``text``, ``col``), the chart the
respondent's own answer selects (``img``), and two echoes of the option they
picked (``choice_text``, ``option_text``).  All five are declared in the survey
flow as ``Recipient`` with no value, which means the live page script set them:
the export does not contain the strings, and nothing here has seen them.  The
template file leaves them as ``<<=...>>`` markers, which is honest for a template
and fatal for a session, because a marker reaching the model is exactly what
:mod:`~silicon_sampling.survey.render` exists to prevent.  So
:func:`derive_piped_fields` fills them: the one that *is* recoverable —
``option_text``, the respondent's own choice — is echoed from the answer they just
gave, and the four that are not become the same bracketed note the converter uses
for absent media.  Inventing feedback prose would be worse than recording that
some was there.

**The arm's own block order is drawn from the profile's seed, not from a fresh
RNG.**  Two arms (``MispCorrectionRisks``, ``HopeAngerNarratives``) have their
blocks permuted by the survey's own randomiser.  Seeding that draw from
``profile.seed`` is what makes a resumed run reproduce an uninterrupted one: the
profile row is the entire state of the respondent, so re-entering the sampler
rebuilds the identical session rather than a differently-shuffled one.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..sampling.driver import SamplerConfig, token_budget
from ..sampling.engine import EngineConfig
from ..sampling.runner import Runner as BaseRunner
from ..sampling.tokens import load_tokenizer
from ..survey.render import walk
from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from . import instrument as inst
from .profiles import Profile

#: The bracketed line that stands in for a page element the export does not hold.
#: Worded like the converter's own media notes, so a transcript has one vocabulary
#: for "something was here and it is not reproduced".
ABSENT = (
    "[ page element shown here — set by the survey's own page script, not reproduced ]"
)

#: Piped fields ``MispCorrectionRisks`` sets at runtime, and the stand-in each
#: gets.  ``option_text`` is overwritten from the respondent's own answer as soon
#: as they have given one; the rest never become knowable.
PIPED_STAND_INS = {
    "text": ABSENT,
    "col": ABSENT,
    "img": ABSENT,
    "choice_text": ABSENT,
    "option_text": ABSENT,
}

#: The slot whose answer ``option_text`` echoes: the issue the respondent named as
#: most disruptive, quoted back at them on the writing page.
OPTION_TEXT_SOURCE = "misperception_correction_risks__Q34"

#: Margin on the measured worst-case transcript.  Real prose tokenises denser than
#: the repeated placeholder the free-text slots are filled with below, so the
#: measurement understates the writing arms; the Pfänder run measured about 1% of
#: understatement and this keeps the same figure.
LENGTH_MARGIN = 1.05


def derive_piped_fields(answers: dict) -> None:
    """Fill the runtime-piped fields, so no marker can reach the model.

    Called by the session after every submitted answer, which is what lets
    ``option_text`` track the choice it echoes instead of freezing at the
    stand-in.
    """
    for field, stand_in in PIPED_STAND_INS.items():
        answers.setdefault(field, stand_in)
    chosen = answers.get(OPTION_TEXT_SOURCE)
    if chosen:
        answers["option_text"] = chosen


def session_for(profile: Profile) -> Session:
    """The session one profile walks through."""
    return Session(
        header=inst.header(profile.profile_id, profile.condition),
        elements=inst.elements_for(
            profile.condition,
            battery=profile.battery.split("|") if profile.battery else None,
            rng=random.Random(profile.seed),
        ),
        answers=dict(profile.prefilled),
        derive=derive_piped_fields,
    )


def record_for(profile: Profile, session: Session) -> dict:
    """One line of ``answers.jsonl``: the profile it came from, and its answers.

    The piped fields are dropped: they are page furniture the respondent read, not
    anything they produced, and leaving them in would put five columns of
    boilerplate into the analysis frame.
    """
    answers = {
        key: value
        for key, value in session.answers.items()
        if not key.startswith("_") and key not in PIPED_STAND_INS
    }
    return {
        "profile_id": profile.profile_id,
        "condition": profile.condition,
        "cond": profile.cond,
        "gender": profile.gender,
        "party": profile.party,
        "age": profile.age,
        "age_band": profile.age_band,
        "education": profile.education,
        "income": profile.income,
        "ses": profile.ses,
        "battery": profile.battery,
        "n_asked": len(session.asked),
        "answers": answers,
    }


def shard_for(profile: Profile) -> str:
    """Where this respondent's transcript is filed, and which group it sizes with."""
    return inst.BY_NAME[profile.condition].slug


def all_slots() -> dict[str, Slot]:
    """Every response position of every usable arm, keyed by id."""
    found: dict[str, Slot] = {}
    for name in inst.CONDITIONS:
        elements = inst.elements_for(
            name, battery=list(inst.DV_BLOCK_ORDER), rng=random.Random(0)
        )
        for event, payload in walk(elements):
            if event == "slot":
                found.setdefault(payload.id, payload)
    return found


def fit_token_budgets(model: str) -> dict[str, int]:
    """Per-slot token budgets, measured with the model's own tokenizer.

    Sizing these by hand is what turns a long option into a systematically
    rejected one; see :func:`silicon_sampling.sampling.driver.token_budget`.
    """
    tokenizer = load_tokenizer(model)
    return {
        slot_id: token_budget(slot, tokenizer) for slot_id, slot in all_slots().items()
    }


def widest_answer(slot: Slot):
    """The longest answer a slot will accept, for worst-case sizing."""
    if isinstance(slot, ChoiceSlot):
        return max(slot.options, key=len)
    if isinstance(slot, IntSlot):
        return max((slot.lo, slot.hi), key=lambda value: len(str(value)))
    if isinstance(slot, FreeTextSlot):
        return "x " * min(slot.max_chars // 2, slot.max_tokens)
    return ""  # pragma: no cover - every slot type is covered above


def worst_case_sessions(shard: str | None = None):
    """A fresh session per usable arm.

    Shared by the transcript sizing below and by the tokenisation test, so both
    cover the same arms.  The block permutation is left at seed 0: the randomised
    arms show every one of their blocks, so a permutation changes the order of the
    transcript and not its length.
    """
    for name in inst.CONDITIONS:
        arm = inst.BY_NAME[name]
        if shard is not None and arm.slug != shard:
            continue
        yield Session(
            header=inst.header("g00000", name),
            elements=inst.elements_for(
                name, battery=list(inst.DV_BLOCK_ORDER), rng=random.Random(0)
            ),
            answers={},
            derive=derive_piped_fields,
        )


def max_transcript_tokens(model: str, shard: str | None = None) -> int:
    """Length of the longest transcript an arm can produce, in tokens.

    This is what a session must hold in the KV cache for its whole life, so it is
    what the group size divides into.  Measured by walking the arm to the end with
    the widest legal answer everywhere, then tokenising — cheap, and no GPU.
    """
    tokenizer = load_tokenizer(model)
    longest = 0
    for session in worst_case_sessions(shard):
        while (step := session.next_prompt()) is not None:
            session.submit(step[1], widest_answer(step[1]))
        longest = max(
            longest,
            len(tokenizer(session.transcript(), add_special_tokens=False)["input_ids"]),
        )
    return int(longest * LENGTH_MARGIN)


class Runner(BaseRunner):
    """The study-independent runner, wired to this instrument."""

    def __init__(
        self,
        out_dir: Path,
        engine_config: EngineConfig,
        sampler_config: SamplerConfig,
    ) -> None:
        super().__init__(
            out_dir,
            engine_config,
            sampler_config,
            session_for=session_for,
            record_for=record_for,
            shard_for=shard_for,
            token_budgets=fit_token_budgets,
            worst_case_tokens=max_transcript_tokens,
        )
