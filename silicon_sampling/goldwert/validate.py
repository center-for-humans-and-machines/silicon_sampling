"""Walk the instrument the way the sampler will, and complain if it cannot.

A rendered template proves the *text* is right.  It does not prove the thing is
drivable, and on this study the difference was not academic: a template prints
``<<=slot>>`` where a session has to *resolve* it, so an echo naming a slot that
does not exist renders perfectly and then raises ``KeyError`` the first time a
respondent reaches the page.  That is exactly what
``IndStructuralChange``'s three "You guessed …%" screens did — see
:func:`~silicon_sampling.goldwert.instrument.slot_index` — and eleven rendered
templates plus a passing test suite said nothing about it.

So the properties checked here are the session-time ones: every echo resolves
before it is displayed, no conditional depends on an answer not yet given, no
prompt reaches a model with a ``<<...>>`` marker still in it, every prompt ends
exactly at ``"Response: "``, and every slot accepts its own legal answers.  It is
the cheapest test in the package and the one most likely to catch a real break,
because it exercises the same code path a GPU run would and needs no GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..survey.render import MARKER_RE
from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from . import instrument as inst
from . import profiles as prof
from .run import derive_piped_fields

#: What the stand-in respondent types into a free-text box.  Written to be a
#: plausible answer to the letter-writing arms *and* to the letter outcome's own
#: coding rule, so a dry run produces a frame that looks like a real one rather
#: than one where every open box is empty.
CANNED_TEXT = (
    "I want you to take climate change seriously and vote for policies that cut "
    "emissions, because the flooding and the heat are already costing people here."
)


@dataclass
class DryRun:
    """The result of driving one respondent to the end of one arm."""

    profile_id: str
    condition: str
    n_asked: int
    transcript: str
    answers: dict


#: How often the stand-in respondent takes a slider's opt-out instead of
#: answering on the scale.  Not a calibration target — it is here so that the
#: dry run actually walks the escape path.  Five live sliders print an opt-out,
#: and a stand-in that always answers with a number would leave every one of
#: them untested against exactly the failure they used to have: the option was
#: printed and the slot refused it.
ESCAPE_SHARE = 0.25


def answer(slot: Slot, rng: random.Random):
    """A legal answer, chosen without a model."""
    if isinstance(slot, IntSlot):
        if getattr(slot, "escape", "") and rng.random() < ESCAPE_SHARE:
            return slot.escape
        return rng.randint(slot.lo, slot.hi)
    if isinstance(slot, ChoiceSlot):
        return rng.choice(list(slot.options))
    if isinstance(slot, FreeTextSlot):
        return CANNED_TEXT
    raise TypeError(f"no stand-in answer for {slot!r}")  # pragma: no cover


def dry_run(profile: prof.Profile) -> DryRun:
    """Drive one profile to the end of its arm, checking every prompt."""
    session = Session(
        header=inst.header(profile.profile_id, profile.condition),
        elements=inst.elements_for(
            profile.condition,
            battery=profile.battery.split("|") if profile.battery else None,
            rng=random.Random(profile.seed),
        ),
        answers=dict(profile.prefilled),
        derive=derive_piped_fields,
    )
    rng = random.Random(profile.seed)
    steps = 0
    while (pending := session.next_prompt()) is not None:
        prompt, slot = pending
        marker = MARKER_RE.search(prompt)
        if marker is not None:
            raise AssertionError(
                f"{profile.profile_id}/{profile.condition}: marker "
                f"{marker.group()!r} reached the model before slot {slot.id!r}"
            )
        if not prompt.endswith("Response: "):
            raise AssertionError(
                f"{profile.profile_id}/{profile.condition}: prompt for {slot.id!r} "
                "does not end at 'Response: '"
            )
        value = answer(slot, rng)
        parsed = slot.parse(str(value))
        if parsed is None:
            raise AssertionError(
                f"{profile.profile_id}/{profile.condition}: slot {slot.id!r} rejects "
                f"its own legal answer {value!r}"
            )
        # What is submitted is what the *slot* made of the draw, not the draw, so
        # this walk stores what a real run would store. For every slot but the
        # five with an opt-out the two are the same object; for those five the
        # draw is the escape's own wording and the stored value is the sentinel
        # that becomes NaN in the frame, so submitting the draw would have hidden
        # the one path this stand-in was extended to exercise.
        session.submit(slot, parsed)
        steps += 1
        if steps > 400:  # pragma: no cover - a loop would mean a broken instrument
            raise AssertionError("session did not terminate")
    transcript = session.transcript()
    marker = MARKER_RE.search(transcript)
    if marker is not None:
        raise AssertionError(f"marker {marker.group()!r} survived into the transcript")
    return DryRun(
        profile_id=profile.profile_id,
        condition=profile.condition,
        n_asked=steps,
        transcript=transcript,
        answers=dict(session.answers),
    )


def dry_run_all(seed: int = 20260823) -> list[DryRun]:
    """One respondent per usable arm, driven to the end."""
    return [dry_run(profile) for profile in prof.build(seed=seed, per_arm=1)]
