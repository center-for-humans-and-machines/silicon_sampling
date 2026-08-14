"""Building blocks for a screen-by-screen description of a Qualtrics survey.

A survey is a sequence of :class:`Screen` objects (one Qualtrics page each).  A
screen holds *elements*: things the participant read (:class:`Text`,
:class:`Image`), things the survey echoed back at them (:class:`Echo`) and
positions at which they produced a response (all the response elements).

Every response element carries a ``slot`` id.  Wherever the Qualtrics survey
recorded the answer in a named variable, the slot id *is* that variable name, so
sampled answers can be compared against the published response data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# --------------------------------------------------------------------------- #
# things the participant read
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Text:
    """A run of on-screen text.

    ``style`` distinguishes a page heading rendered in bold by Qualtrics
    ("head"), ordinary body copy ("body"), a source/attribution line ("cite")
    and an image caption ("caption").
    """

    text: str
    style: str = "body"


@dataclass(frozen=True)
class Bullets:
    """A bulleted list."""

    items: Sequence[str]
    marker: str = "*"


@dataclass(frozen=True)
class Image:
    """A picture, described in words.

    Text that is *inside* the image and that carries the manipulation (a flyer,
    a pie-chart label, an infographic) is transcribed verbatim in ``alt``.
    """

    alt: str
    caption: str | None = None


@dataclass(frozen=True)
class Echo:
    """Piped text: the survey re-displayed an answer the participant gave earlier.

    Not a response position.
    """

    column: str
    description: str


# --------------------------------------------------------------------------- #
# response positions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Slider:
    """A single 0-100 Qualtrics slider."""

    slot: str
    left: str
    right: str
    mid: str | None = None
    extra: str | None = None
    lo: int = 0
    hi: int = 100
    stem: str | None = None
    ticks: bool = True


@dataclass(frozen=True)
class Matrix:
    """A stack of sliders sharing one header row.

    ``items`` pairs a slot id with the row label.  ``randomised`` records that
    Qualtrics shuffled the row order per participant.
    """

    left: str
    right: str
    items: Sequence[tuple[str, str]]
    mid: str | None = None
    extra: str | None = None
    lo: int = 0
    hi: int = 100
    randomised: bool = False


@dataclass(frozen=True)
class Choice:
    """Radio buttons: exactly one option."""

    slot: str
    options: Sequence[str]
    other_slot: str | None = None
    randomised: bool = False


@dataclass(frozen=True)
class MultiChoice:
    """Check boxes: any number of options."""

    slot: str
    options: Sequence[str]
    other_slot: str | None = None
    randomised: bool = False
    allow_none: bool = True


@dataclass(frozen=True)
class Number:
    """A free numeric entry box."""

    slot: str
    hint: str = "Enter a number"


@dataclass(frozen=True)
class FreeText:
    """A free-text box."""

    slot: str
    hint: str = "Write your answer in the box below"


@dataclass(frozen=True)
class NumberGrid:
    """A WEPT screening grid: rows of two-digit numbers, each row click-able.

    One slot per displayed row, matching ``WEPT<n>nums_1`` .. ``_6``.
    """

    slots: Sequence[str]
    rows: Sequence[Sequence[int]]
    randomised: bool = True


RESPONSE_TYPES = (Slider, Matrix, Choice, MultiChoice, Number, FreeText, NumberGrid)


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Screen:
    """One Qualtrics page.

    ``timer`` is the name of the page-timer variable in the published data, which
    is how the page boundaries were recovered; ``None`` means the page carried no
    timer.  ``condition`` records display logic ("shown only if ...").
    """

    elements: Sequence[object]
    timer: str | None = None
    condition: str | None = None


@dataclass(frozen=True)
class Block:
    """A named run of screens, e.g. one intervention."""

    key: str
    title: str
    screens: Sequence[Screen]
    note: str | None = None


@dataclass(frozen=True)
class Condition:
    """One of the twelve experimental conditions."""

    code: int
    key: str
    title: str
    cond_name: str
    block: Block | None
    country_adapted: Sequence[str] = field(default_factory=tuple)
