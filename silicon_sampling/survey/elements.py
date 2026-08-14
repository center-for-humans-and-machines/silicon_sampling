"""The things a transcript is made of.

An instrument is a flat sequence of these.  :class:`Slot` objects (from
:mod:`~silicon_sampling.survey.slots`) sit in the same sequence and are the only
elements that produce a value.

Inline piping — the survey re-displaying an answer the respondent already gave,
as in "How important is being a [Republican/Democrat] to you?" — is written as
``<<=party>>`` inside any text or question stem, so it needs no element of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Text:
    """A run of on-screen text.

    ``style`` is ``"body"`` for ordinary copy, ``"head"`` for a screen heading and
    ``"cite"`` for a source/attribution line.
    """

    text: str
    style: str = "body"


@dataclass(frozen=True)
class PageBreak:
    """A screen boundary: the respondent clicked on."""

    label: str | None = None


@dataclass(frozen=True)
class Conditional:
    """Elements shown only to some respondents.

    ``note`` states the display logic in words and appears in the *template*
    file; a rendered per-respondent transcript simply contains the branch, or
    does not.  ``predicate`` decides, given the answers so far.
    """

    note: str
    predicate: Callable[[Mapping[str, object]], bool]
    elements: Sequence[object]


@dataclass(frozen=True)
class Block:
    """A named run of elements, e.g. the demographics section or one intervention."""

    key: str
    title: str
    elements: Sequence[object]
    note: str | None = None
