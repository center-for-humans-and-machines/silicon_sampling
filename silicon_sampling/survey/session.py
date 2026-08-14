"""One respondent's walk through an instrument.

The sampler drives a session: ask for the next prompt, sample a value, submit it,
repeat.  The prompt is the transcript rendered up to and including ``"Response: "``
for the pending slot and nothing more — which is exactly what a base model needs
in order to continue as that respondent.

Conditional branches are resolved against the answers given so far, so a session
that has not yet answered the party question does not know whether it will be
asked about partisan importance.  That is why the transcript is re-rendered on
every step rather than accumulated blindly.
"""

from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Sequence

from .render import PAGE_RULE, question_lines, resolve_echoes, walk
from .slots import Slot


class Session:
    """Mutable state of one respondent filling out one condition's transcript."""

    def __init__(
        self,
        header: str,
        elements: Sequence[object],
        answers: Mapping[str, object] | None = None,
        derive: Callable[[MutableMapping[str, object]], None] | None = None,
    ) -> None:
        self.header = header
        self.elements = elements
        self.answers: MutableMapping[str, object] = dict(answers or {})
        #: Called after every submitted answer; may add derived keys that later
        #: conditionals or piped text depend on.
        self.derive = derive
        #: Slot ids in the order they were actually asked of this respondent.
        self.asked: list[str] = []
        if self.derive:
            self.derive(self.answers)

    # -- driving ---------------------------------------------------------- #

    def next_prompt(self) -> tuple[str, Slot] | None:
        """Text up to the next unanswered ``Response: ``, or ``None`` when done."""
        text, pending = self._render()
        if pending is None:
            return None
        return text, pending

    def submit(self, slot: Slot, value: object) -> None:
        self.answers[slot.id] = value
        self.asked.append(slot.id)
        if self.derive:
            self.derive(self.answers)

    @property
    def done(self) -> bool:
        return self._render()[1] is None

    def transcript(self) -> str:
        text, pending = self._render()
        if pending is not None:
            raise RuntimeError(
                f"transcript is not complete: {pending.id} is unanswered"
            )
        return text

    # -- rendering -------------------------------------------------------- #

    def _render(self) -> tuple[str, Slot | None]:
        out: list[str] = [self.header.rstrip(), ""]
        page = 1
        number = 0
        for event, payload in walk(self.elements, self.answers):
            if event == "block":
                continue
            if event == "page":
                page += 1
                out += ["", PAGE_RULE.format(n=page), ""]
            elif event == "text":
                out += [resolve_echoes(payload.text.strip(), self.answers), ""]
            elif event == "slot":
                number += 1
                if payload.id not in self.answers:
                    lines = question_lines(_echoed(payload, self.answers), number, "")
                    # The prompt must end exactly at "Response: ", with the space.
                    out += lines[:-1] + ["Response: "]
                    return "\n".join(out[:-1]) + "\n" + out[-1], payload
                body = payload.render(self.answers[payload.id])
                out += question_lines(_echoed(payload, self.answers), number, body) + [
                    ""
                ]
        return "\n".join(out).rstrip() + "\n", None


def _echoed(slot: Slot, answers: Mapping[str, object]) -> Slot:
    """A copy of ``slot`` with piped text in its stem resolved."""
    if "<<=" not in slot.prompt:
        return slot
    from dataclasses import replace

    return replace(slot, prompt=resolve_echoes(slot.prompt, answers))
