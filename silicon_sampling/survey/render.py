"""Turn an instrument into transcript text.

One traversal serves both jobs.  :func:`render_template` walks it with every
conditional branch expanded and every response position left as a ``<<...>>``
marker, producing the template file a human reads.  :class:`Session
<silicon_sampling.survey.session.Session>` walks the same structure with one
respondent's answers, resolving conditionals and filling responses in.

Marker vocabulary (all markers are stripped before the model sees any text):

===========================  ==========================================
``<<id :: kind :: spec>>``   a response position and its legal values
``<<=id>>``                  piped text: an answer echoed back on screen
``<<?if note>> … <<?endif>>``  a branch shown only to some respondents
===========================  ==========================================
"""

from __future__ import annotations

import re
from typing import Iterator, Mapping, Sequence

from .elements import Block, Conditional, PageBreak, Text
from .slots import Slot

#: Matches any marker, so a prompt can be asserted marker-free before use.
MARKER_RE = re.compile(r"<<[^>]*>>")

_ECHO_RE = re.compile(r"<<=([A-Za-z0-9_]+)>>")

PAGE_RULE = "- - - [ page {n} ] - - -"


def walk(
    elements: Sequence[object], answers: Mapping[str, object] | None = None
) -> Iterator[tuple[str, object]]:
    """Yield ``(event, payload)`` over an element sequence.

    With ``answers`` given, conditionals are resolved against them and skipped
    branches are not visited.  Without, every branch is visited and the
    conditional's own boundaries are announced as ``open``/``close`` events.
    """
    for element in elements:
        if isinstance(element, Block):
            yield "block", element
            yield from walk(element.elements, answers)
        elif isinstance(element, Conditional):
            if answers is None:
                yield "cond_open", element
                yield from walk(element.elements, answers)
                yield "cond_close", element
            elif element.predicate(answers):
                yield from walk(element.elements, answers)
        elif isinstance(element, Slot):
            yield "slot", element
        elif isinstance(element, PageBreak):
            yield "page", element
        elif isinstance(element, Text):
            yield "text", element
        else:  # pragma: no cover - guards against a malformed instrument
            raise TypeError(f"not a transcript element: {element!r}")


def resolve_echoes(text: str, answers: Mapping[str, object]) -> str:
    """Replace ``<<=slot_id>>`` with the answer the respondent already gave."""
    return _ECHO_RE.sub(lambda m: str(answers[m.group(1)]), text)


def question_lines(slot: Slot, number: int, body: str) -> list[str]:
    """The three-line shape every response position takes in a transcript."""
    lines = []
    stem = slot.prompt.strip()
    if stem:
        lines.append(f"Q{number}. {stem}")
    description = slot.describe().strip()
    if description:
        lines.append(f"      {description}")
    lines.append(f"Response: {body}")
    return lines


def render_template(header: str, elements: Sequence[object]) -> str:
    """Render the fill-in-the-blank template for one condition."""
    out: list[str] = [header.rstrip(), ""]
    page = 1
    number = 0
    for event, payload in walk(elements):
        if event == "block":
            if payload.note:
                out += [f"[ block: {payload.title} — {payload.note} ]", ""]
        elif event == "page":
            page += 1
            out += ["", PAGE_RULE.format(n=page), ""]
        elif event == "text":
            out += [payload.text.strip(), ""]
        elif event == "cond_open":
            out += [f"<<?if {payload.note}>>", ""]
        elif event == "cond_close":
            out += ["<<?endif>>", ""]
        elif event == "slot":
            number += 1
            out += question_lines(payload, number, payload.marker()) + [""]
    return "\n".join(out).rstrip() + "\n"


def slot_manifest(elements: Sequence[object]) -> list[dict]:
    """Every response position of one condition, in display order."""
    manifest = []
    order = 0
    conditional_note: str | None = None
    for event, payload in walk(elements):
        if event == "cond_open":
            conditional_note = payload.note
        elif event == "cond_close":
            conditional_note = None
        elif event == "slot":
            entry = {
                "order": order,
                "id": payload.id,
                "kind": payload.kind,
                "source": payload.source,
                "legal": payload.legal_spec,
                "prompt": payload.prompt,
                "max_tokens": payload.max_tokens,
            }
            if conditional_note:
                entry["shown_if"] = conditional_note
            manifest.append(entry)
            order += 1
    return manifest
