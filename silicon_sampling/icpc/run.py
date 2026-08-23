"""Wiring that turns the ICPC instrument into a restartable sampling run.

The run itself is study-independent and lives in
:class:`silicon_sampling.sampling.runner.Runner`; everything here is the five
study-specific things that runner asks for.  Subclassing rather than
re-implementing is deliberate: resumability in this project is a property of the
*write order* — transcript before answer record, `fsync` per group, torn final
line truncated before any append — and a second copy of that ordering is a second
thing that can drift out of agreement with the first.  The cost is that the
constructor signature has to be re-declared here to keep the
``Runner(out_dir, EngineConfig(...), SamplerConfig(...))`` contract the launcher
scripts depend on; that is a cheap price for one implementation of the ordering.

Two things about this study shape the wiring.

**Group size has to be decided per arm.**  A control transcript runs to 107
response positions against 86 for the leanest treated arm, and it reads two extra
blocks of covariates nobody else sees; sizing every group to the control's
worst case would throttle the eleven other arms for no reason.  So
:func:`shard_for` files a respondent under its arm and the runner sizes each
arm's groups from that arm's own worst case.

**The worst case is walked, not guessed.**  :func:`max_transcript_tokens` drives
a session to the end filling every slot with the longest answer it will accept,
which for the WEPT chain means accepting all eight pages and for the sharing item
means being willing to share — the branches that make a transcript longest.  That
is the length a session must keep resident in the KV cache for its whole life, so
it is the number the group size divides into.
"""

from __future__ import annotations

from pathlib import Path

from ..sampling.driver import SamplerConfig, token_budget
from ..sampling.engine import EngineConfig
from ..sampling.runner import Runner as BaseRunner
from ..sampling.tokens import load_tokenizer
from ..survey.render import walk
from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from . import instrument as inst
from . import profiles as prof
from .profiles import Profile

#: Margin on the measured worst-case transcript.  Real prose tokenises denser
#: than the repeated placeholder the free-text slots are filled with below, so the
#: measurement understates the length of the writing arms; the Pfänder run
#: measured about 1% of understatement and this keeps the same figure.
LENGTH_MARGIN = 1.05


def _panel_stand_in() -> dict[str, object]:
    """Panel-record values for a session with no profile behind it.

    The panel block is printed from seven ``<<=...>>`` echoes, so a session that
    supplies none of them cannot render at all — and the sizing walk below has no
    profile.  The longest legal value is used for each, so the stand-in header is
    the worst case rather than a typical one.
    """
    return {
        "panel_gender": max(prof.GENDER_OPTIONS.values(), key=len),
        "panel_age": 100,
        "panel_education": max(prof.EDUCATION_OPTIONS.values(), key=len),
        "panel_income": max(prof.INCOME_OPTIONS.values(), key=len),
        "panel_ses": 10,
        "panel_politics_social": 100,
        "panel_politics_economic": 100,
    }


def session_for(profile: Profile) -> Session:
    """The session one profile walks through.

    The randomised block orders come off the profile rather than being drawn
    here, so a resumed run reproduces an uninterrupted one respondent for
    respondent: the profile row is the whole state of the draw.
    """
    arm = inst.BY_KEY[profile.condition]
    elements = inst.elements_for(
        arm,
        battery=profile.battery.split("|") if profile.battery else None,
        extras=profile.extras.split("|") if profile.extras else None,
        probe_index=profile.probe_index,
    )
    return Session(
        header=inst.header(profile.profile_id, arm),
        elements=elements,
        answers=dict(profile.prefilled),
    )


def record_for(profile: Profile, session: Session) -> dict:
    """One line of ``answers.jsonl``: the profile it came from, and its answers.

    The moderators are carried alongside the answers rather than recovered from
    them later.  They are prefilled, so the two agree by construction today — but
    an analysis frame that reads the demographics off the *profile* keeps working
    if a future run ever samples them instead.
    """
    answers = {
        key: value for key, value in session.answers.items() if not key.startswith("_")
    }
    return {
        "profile_id": profile.profile_id,
        "condition": profile.condition,
        "cond": profile.cond,
        "gender": profile.gender,
        "age": profile.age,
        "age_band": profile.age_band,
        "education": profile.education,
        "income": profile.income,
        "ses_ladder": profile.ses_ladder,
        "politics_social": profile.politics_social,
        "politics_economic": profile.politics_economic,
        "battery": profile.battery,
        "extras": profile.extras,
        "probe_index": profile.probe_index,
        "n_asked": len(session.asked),
        "answers": answers,
    }


def shard_for(profile: Profile) -> str:
    """Where this respondent's transcript is filed, and which group it sizes with."""
    return inst.BY_KEY[profile.condition].slug


def _arms_for(shard: str | None):
    return [arm for arm in inst.ARMS if shard is None or arm.slug == shard]


def all_slots() -> dict[str, Slot]:
    """Every response position of every arm, keyed by id."""
    found: dict[str, Slot] = {}
    for arm in inst.ARMS:
        probes = range(inst.PROBE_WORDINGS) if arm.code == 1 else (0,)
        for probe in probes:
            elements = inst.elements_for(arm, probe_index=probe)
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
    """The longest answer a slot will accept, for worst-case sizing.

    For a gated chain this is also the answer that opens the gate — "yes" is
    longer than "no", and willingness to share is longer than either refusal — so
    walking with it reaches the pages a shorter answer would have skipped.  That
    coincidence is what makes one walk both the widest and the deepest.
    """
    if isinstance(slot, ChoiceSlot):
        return max(slot.options, key=len)
    if isinstance(slot, IntSlot):
        return max((slot.lo, slot.hi), key=lambda value: len(str(value)))
    if isinstance(slot, FreeTextSlot):
        return "x " * min(slot.max_chars // 2, slot.max_tokens)
    return ""  # pragma: no cover - every slot type is covered above


def worst_case_sessions(shard: str | None = None):
    """A fresh session per arm, with the control arm's longest probe wording.

    Shared by the transcript sizing below and by the tokenisation test, so both
    cover the same arms.
    """
    for arm in _arms_for(shard):
        probes = range(inst.PROBE_WORDINGS) if arm.code == 1 else (0,)
        for probe in probes:
            yield Session(
                header=inst.header("i00000", arm),
                elements=inst.elements_for(arm, probe_index=probe),
                answers=_panel_stand_in(),
            )


def max_transcript_tokens(model: str, shard: str | None = None) -> int:
    """Length of the longest transcript an arm can produce, in tokens."""
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
