"""Study-independent machinery for turning a questionnaire into fill-in-the-blank text.

A survey is described as a sequence of :mod:`~silicon_sampling.survey.elements`
(things the respondent read) and :mod:`~silicon_sampling.survey.slots` (positions
at which they produced a response).  :mod:`~silicon_sampling.survey.render` emits
a plain-text transcript template plus a machine-readable manifest, and
:mod:`~silicon_sampling.survey.session` walks one respondent through it.
"""

from __future__ import annotations

from .elements import Block, Conditional, PageBreak, Text
from .render import MARKER_RE, render_template, slot_manifest
from .session import Session
from .slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot

__all__ = [
    "Block",
    "ChoiceSlot",
    "Conditional",
    "FreeTextSlot",
    "IntSlot",
    "MARKER_RE",
    "PageBreak",
    "Session",
    "Slot",
    "Text",
    "render_template",
    "slot_manifest",
]
