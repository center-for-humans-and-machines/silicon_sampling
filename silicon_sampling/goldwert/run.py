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
onto its pages — the correct/incorrect feedback (``text``, ``col``), the photograph
the respondent's own answer selects (``img``), and two echoes of the option they
picked (``choice_text``, ``option_text``).  All five are declared in the survey flow
as ``Recipient`` with no value, so the first version of this module concluded that
the export does not contain them and filled four of the five with a bracketed
"something was here" note.  That was wrong about the export.  The *values* are all
there — ``correct`` and ``incorrect`` are literals in the flow, the six correction
paragraphs and the six photographs are literals in the flow — and the *rules* are
all in the questions' own ``QuestionJS``: recode 1 pastes ``correct``, and the
summary question's handler pastes ``<topic>_text`` and ``<topic>_img`` for the topic
the respondent picked.  So :func:`derive_piped_fields` now reproduces the script
rather than apologising for it, and only ``col`` stays empty, because it is a hex
colour in a ``style`` attribute.  The cost of getting this wrong was concentrated:
the writing screen asks the respondent to write about the issue whose correction
paragraph and photograph it is showing them, and it was showing them a placeholder.

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
#: gets before the answer it depends on has been given.  Four of the five are
#: overwritten from the respondent's own answers by :func:`derive_piped_fields`; the
#: exception is ``col``, which is a hex colour in a ``style`` attribute and has
#: nothing in it for a transcript to carry.
PIPED_STAND_INS = {
    "text": ABSENT,
    "col": "",
    "img": ABSENT,
    "choice_text": ABSENT,
    "option_text": ABSENT,
}

#: Margin on the measured worst-case transcript.  Real prose tokenises denser than
#: the repeated placeholder the free-text slots are filled with below, so the
#: measurement understates the writing arms; the Pfänder run measured about 1% of
#: understatement and this keeps the same figure.
LENGTH_MARGIN = 1.05


def is_piped_field(key: str) -> bool:
    """Whether this answer key is page furniture rather than something a respondent
    produced."""
    return key in PIPED_STAND_INS or key.endswith(inst.FEEDBACK_SUFFIX)


def derive_piped_fields(answers: dict) -> None:
    """Stand in for the arm's page script, so no marker and no blank reaches a model.

    Called by the session after every submitted answer, which is what lets these
    track the answers they are functions of instead of freezing at a stand-in.

    Each correction screen's "That's correct!" / "That's incorrect!" line gets its
    own field rather than sharing the survey's single ``text``, because a session
    re-renders the whole transcript on every step and a shared field would let a
    later answer rewrite an earlier screen; see
    :data:`~silicon_sampling.goldwert.instrument.FEEDBACK_SUFFIX`.  Written here in
    the same pass as the rest so that there is one place where a page the export does
    not hold gets filled in.
    """
    for field, stand_in in PIPED_STAND_INS.items():
        answers.setdefault(field, stand_in)

    for slot_id, table in inst.correction_feedback().items():
        given = answers.get(slot_id)
        answers[inst.feedback_field(slot_id)] = (
            table.get(str(given), ABSENT) if given is not None else ABSENT
        )

    source, pages = inst.summary_choice_pages()
    chosen = answers.get(source)
    if chosen:
        answers["option_text"] = chosen
        prose, note = pages.get(str(chosen), ("", ""))
        answers["choice_text"] = prose or ABSENT
        answers["img"] = note or ABSENT


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
    anything they produced, and leaving them in would put eleven columns of
    boilerplate into the analysis frame.
    """
    answers = {
        key: value
        for key, value in session.answers.items()
        if not key.startswith("_") and not is_piped_field(key)
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
