"""Walk the instrument the way the sampler will, and complain if it cannot.

A rendered template proves the *text* is right.  It does not prove the thing is
drivable: that every echo resolves before it is displayed, that no conditional
depends on an answer that has not been given yet, that no prompt reaches a model
with a ``<<...>>`` marker still in it, and that every legal answer the slot
declares actually parses.  Those are session-time properties, so they are checked
by running a session with a deterministic stand-in for the model.

This is the cheapest test in the package and the one most likely to catch a real
break, because it exercises the same code path a GPU run would and needs no GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..survey.render import MARKER_RE
from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from . import instrument as inst
from . import profiles as prof

#: What the stand-in respondent types into a free-text box.  Long enough to be a
#: plausible answer to the letter-writing arms, short enough to read.
CANNED_TEXT = (
    "I worry about this and I try to do what I can, but it is hard to know "
    "whether any of it makes a difference."
)


@dataclass
class DryRun:
    """The result of driving one respondent to the end of one arm."""

    profile_id: str
    condition: str
    n_asked: int
    transcript: str
    answers: dict


def answer(slot: Slot, rng: random.Random):
    """A legal answer, chosen without a model."""
    if isinstance(slot, IntSlot):
        return rng.randint(slot.lo, slot.hi)
    if isinstance(slot, ChoiceSlot):
        return rng.choice(list(slot.options))
    if isinstance(slot, FreeTextSlot):
        if slot.id.startswith("WEPT") and "nums" in slot.id:
            return "none"
        if slot.id == "Attn_60":
            return "sixty"
        return CANNED_TEXT
    raise TypeError(f"no stand-in answer for {slot!r}")  # pragma: no cover


def dry_run(profile: prof.Profile, panel_header: bool = True) -> DryRun:
    """Drive one profile to the end of its arm, checking every prompt."""
    arm = inst.BY_KEY[profile.condition]
    elements = inst.elements_for(
        arm,
        battery=profile.battery.split("|"),
        extras=profile.extras.split("|"),
        probe_index=profile.probe_index,
        panel_header=panel_header,
    )
    session = Session(
        inst.header(profile.profile_id, arm), elements, answers=dict(profile.prefilled)
    )
    rng = random.Random(profile.seed)
    steps = 0
    while True:
        pending = session.next_prompt()
        if pending is None:
            break
        prompt, slot = pending
        marker = MARKER_RE.search(prompt)
        if marker is not None:
            raise AssertionError(
                f"{profile.profile_id}/{arm.key}: marker {marker.group()!r} "
                f"reached the model before slot {slot.id!r}"
            )
        if not prompt.endswith("Response: "):
            raise AssertionError(
                f"{profile.profile_id}/{arm.key}: prompt for {slot.id!r} does not "
                "end at 'Response: '"
            )
        value = answer(slot, rng)
        if slot.parse(str(value)) is None:
            raise AssertionError(
                f"{profile.profile_id}/{arm.key}: slot {slot.id!r} rejects its own "
                f"legal answer {value!r}"
            )
        session.submit(slot, value)
        steps += 1
        if steps > 400:  # pragma: no cover - a loop would mean a broken instrument
            raise AssertionError("session did not terminate")
    transcript = session.transcript()
    marker = MARKER_RE.search(transcript)
    if marker is not None:
        raise AssertionError(f"marker {marker.group()!r} survived into the transcript")
    return DryRun(
        profile_id=profile.profile_id,
        condition=arm.key,
        n_asked=steps,
        transcript=transcript,
        answers=dict(session.answers),
    )


def dry_run_all(seed: int = 20260823, panel_header: bool = True) -> list[DryRun]:
    """One respondent per arm, driven to the end."""
    built = prof.build(seed=seed, per_arm=1)
    return [dry_run(profile, panel_header=panel_header) for profile in built]
