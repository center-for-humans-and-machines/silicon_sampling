"""Response positions and the grammars that say what a legal answer looks like.

Every slot knows three things:

* how to describe its legal answers **in prose**, for the line the model reads;
* how to describe them **machine-readably**, for the ``<<...>>`` marker in the
  template file and for ``manifest.json``;
* how to turn a raw continuation into a value, or reject it.

Parsing is *prefix-truncation then validation*: a base model does not stop on its
own, so the sampler asks for a handful of tokens, cuts at the first newline, and
then asks the slot whether what is left starts with something legal.  Selecting
the first legal draw out of *n* independent draws is exact rejection sampling
from the model's distribution restricted to the legal set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

#: Characters allowed to trail a legal answer before it stops being legal — just
#: punctuation and whitespace.
_TRAILING = ".,;:)!?’'\"—- \t"


def _clip(raw: str) -> str:
    """Cut a raw continuation down to the part that could be an answer."""
    return raw.split("\n", 1)[0].strip()


#: Cosmetic characters in money-shaped options that a respondent may drop.
_DECORATION = str.maketrans("", "", "$,")


def _decorate(text: str) -> str:
    """Strip currency symbols and thousands separators."""
    return text.translate(_DECORATION)


def _trailing_ok(rest: str) -> bool:
    """Is what follows the matched answer merely punctuation?

    Strict on purpose.  A response line in this transcript format contains the
    answer and nothing else, so a draw like ``"Yes | No"`` or
    ``"No | Not applicable? I"`` is the model echoing the option list rather than
    answering it, and must be rejected — accepting its first option would quietly
    record a hedge as a choice.
    """
    return rest.strip(_TRAILING) == ""


@dataclass(frozen=True)
class Slot:
    """Base class: a position where a respondent produced a value.

    ``id`` is the Qualtrics variable name wherever the study has one, so sampled
    answers line up with the published data without a translation table.
    """

    id: str
    #: Prose shown to the model on the line above ``Response:``.
    prompt: str = ""
    #: ``generated`` (sampled) or ``prefilled`` (supplied by the profile).
    source: str = "generated"
    #: Tokens to request per draw.  Deliberately small: the model will not stop.
    max_tokens: int = 8

    @property
    def kind(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def legal_spec(self) -> str:  # pragma: no cover - overridden
        """The machine-readable legal-value spec that goes inside the marker."""
        raise NotImplementedError

    def describe(self) -> str:
        """The human-readable statement of the legal answers."""
        return self.prompt

    def parse(self, raw: str):
        """Return the parsed value, or ``None`` if the draw is illegal."""
        raise NotImplementedError

    def render(self, value) -> str:
        """Render a parsed value back into the transcript."""
        return str(value)

    def marker(self) -> str:
        return f"<<{self.id} :: {self.kind} :: {self.legal_spec}>>"


@dataclass(frozen=True)
class ChoiceSlot(Slot):
    """Exactly one option from a closed list."""

    options: Sequence[str] = ()
    #: Optional raw Qualtrics codes, parallel to ``options``.
    codes: Sequence[object] = ()
    #: Rendered on the prose line instead of the full option list when the list
    #: is long enough that repeating it would swamp the question.
    describe_as: str | None = None

    @property
    def kind(self) -> str:
        return "choice"

    @property
    def legal_spec(self) -> str:
        return " | ".join(self.options)

    def describe(self) -> str:
        if self.describe_as:
            return self.describe_as
        return "Options: " + " | ".join(self.options)

    @staticmethod
    def _match(text: str, pairs: Sequence[tuple[str, str]]):
        """Longest-prefix match of ``text`` against ``(spelling, option)`` pairs.

        Longest first, so "Master's degree / Professional degree" wins over any
        option that happens to be a prefix of it.
        """
        low = text.lower()
        for spelling, option in sorted(
            pairs, key=lambda pair: len(pair[0]), reverse=True
        ):
            if (
                spelling
                and low.startswith(spelling.lower())
                and _trailing_ok(text[len(spelling) :])
            ):
                return option
        return None

    def parse(self, raw: str):
        text = _clip(raw)
        if not text:
            return None

        matched = self._match(text, [(option, option) for option in self.options])
        if matched is not None:
            return matched

        # The model reliably reformats money — "100,000 to $167,999" or
        # "30000 to 55999" for "$30,000 to $55,999" — dropping currency symbols
        # and thousands separators. That is not a hedge or a different answer, it
        # is the same answer typed the way people type money, and rejecting it
        # does real damage: the failure rate is *option-dependent*. On the income
        # item it ran from 3% for the one option beginning with a word to 58% for
        # the ones beginning with "$", and rejection sampling turns any such
        # asymmetry straight into a skewed distribution. Comparing with the
        # cosmetics removed on both sides fixes it; the guard below refuses to do
        # so if it would make two options indistinguishable.
        spellings = [(_decorate(option), option) for option in self.options]
        if len({spelling for spelling, _ in spellings}) < len(self.options):
            return None
        if any(spelling != option for spelling, option in spellings):
            return self._match(_decorate(text), spellings)
        return None


@dataclass(frozen=True)
class IntSlot(Slot):
    """An integer in a closed range: sliders, Likert points, year of birth, dollars."""

    lo: int = 0
    hi: int = 100
    #: Endpoint labels, e.g. ``"0 = not at all ... 100 = very strongly"``.
    anchors: str | None = None
    #: Accept (and discard) a trailing ``%`` — several items ask for percentages.
    allow_percent: bool = True
    #: Accept (and discard) a leading ``$`` — the donation item.
    allow_dollar: bool = False

    _NUM = re.compile(r"\d+")

    @property
    def kind(self) -> str:
        return "int"

    @property
    def legal_spec(self) -> str:
        return f"{self.lo}..{self.hi}"

    def describe(self) -> str:
        if self.anchors:
            return self.anchors
        return f"Whole number from {self.lo} to {self.hi}."

    def parse(self, raw: str):
        text = _clip(raw)
        if self.allow_dollar:
            text = text.lstrip("$").lstrip()
        match = self._NUM.match(text)
        if not match:
            return None
        rest = text[match.end() :]
        whole = match.group()

        # A slider is a continuous control that the survey records as an integer,
        # so "92.36" is a real position, not a malformed answer: round it the way
        # the instrument would. Refusing decimals instead would reject an
        # answer the model *did* give, and do so more often at some scale
        # positions than others.
        value = float(whole)
        if rest[:1] == "." and rest[1:2].isdigit():
            fraction = re.match(r"\.\d+", rest)
            value = float(whole + fraction.group())
            rest = rest[fraction.end() :]
        elif rest[:1] == "," and rest[1:2].isdigit():
            # "1,200" is a thousands separator, not a decimal: read it as 1200,
            # which then fails the range check on its own merits.
            group = re.match(r"(?:,\d{3})+", rest)
            if group:
                value = float(whole + group.group().replace(",", ""))
                rest = rest[group.end() :]
            else:
                return None

        if self.allow_percent and rest[:1] == "%":
            rest = rest[1:]
        if not _trailing_ok(rest):
            return None
        rounded = int(value + 0.5)
        if not self.lo <= rounded <= self.hi:
            return None
        return rounded


@dataclass(frozen=True)
class FreeTextSlot(Slot):
    """An open text box."""

    #: Optional full-match regex the answer has to satisfy.
    pattern: str | None = None
    max_chars: int = 400
    hint: str = "Free text."
    max_tokens: int = 60

    _compiled: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def kind(self) -> str:
        return "pattern" if self.pattern else "text"

    @property
    def legal_spec(self) -> str:
        return self.pattern if self.pattern else f"free text, <= {self.max_chars} chars"

    def describe(self) -> str:
        return self.hint

    def parse(self, raw: str):
        text = _clip(raw)
        if not text or len(text) > self.max_chars:
            return None
        if self.pattern:
            if "re" not in self._compiled:
                self._compiled["re"] = re.compile(self.pattern)
            match = self._compiled["re"].match(text)
            if not match or not _trailing_ok(text[match.end() :]):
                return None
            return match.group()
        return text
