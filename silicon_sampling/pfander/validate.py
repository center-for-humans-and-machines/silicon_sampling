"""Checks that must pass before a sampling run is worth starting.

These are cheap and catch the failure that costs the most: discovering after
18,000 GPU-hours' worth of generation that one condition never asked an item the
primary outcome needs.
"""

from __future__ import annotations

from ..survey.render import MARKER_RE, slot_manifest
from ..survey.slots import ChoiceSlot, IntSlot
from . import build, outcomes, templates
from .conditions import CONDITIONS
from .outcomes import MODERATORS


def _dummy(slot):
    if isinstance(slot, ChoiceSlot):
        return slot.options[0]
    if isinstance(slot, IntSlot):
        return (slot.lo + slot.hi) // 2
    if slot.id == "zip_code":
        return "12345"
    return "attention"


def check() -> list[str]:
    """Return a list of problems; empty means the instrument is ready."""
    problems: list[str] = []

    for condition in CONDITIONS:
        slots = slot_manifest(templates.template_elements(condition))
        ids = [slot["id"] for slot in slots]
        duplicates = {slot_id for slot_id in ids if ids.count(slot_id) > 1}
        if duplicates:
            problems.append(f"{condition}: duplicate slot ids {sorted(duplicates)}")

        missing = [item for item in outcomes.REQUIRED_ITEMS if item not in ids]
        if missing:
            problems.append(f"{condition}: missing scored items {missing}")

        # Walk a whole session: every prompt must be marker-free and end at the
        # response position, and the answers must yield all 13 outcomes.
        session = build.make_session(
            "p00001", condition, code_name=build.template_code_name(condition)
        )
        while (step := session.next_prompt()) is not None:
            text, slot = step
            if MARKER_RE.search(text):
                problems.append(
                    f"{condition}: marker leaked into the prompt at {slot.id}"
                )
                break
            if not text.endswith("Response: "):
                problems.append(
                    f"{condition}: prompt for {slot.id} does not end at the response position"
                )
                break
            value = _dummy(slot)
            if slot.source == "generated" and slot.parse(str(value)) is None:
                problems.append(
                    f"{condition}: {slot.id} rejects its own legal value {value!r}"
                )
            session.submit(slot, value)
        else:
            computed = outcomes.compute(session.answers)
            absent = [name for name in outcomes.OUTCOMES if name not in computed]
            if absent:
                problems.append(f"{condition}: outcomes not computable: {absent}")
            for name, levels in MODERATORS.items():
                if computed.get(name) not in levels:
                    problems.append(
                        f"{condition}: moderator {name} produced {computed.get(name)!r}, not a codebook level"
                    )

    return problems


def main() -> int:
    problems = check()
    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        return 1
    print(
        f"OK    {len(CONDITIONS)} conditions: slot ids unique, all 13 outcomes computable, prompts clean"
    )
    return 0
