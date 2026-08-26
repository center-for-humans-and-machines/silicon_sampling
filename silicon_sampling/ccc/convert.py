"""Qualtrics questions -> transcript elements, for the Climate Change Challenge.

Structurally this is the easiest of the five studies to convert: one block per
arm, one randomiser, and inside the arms zero display logic, zero piped text, zero
embedded data, zero JavaScript and no video, audio or iframes anywhere.  The
fidelity work is therefore concentrated in three places, each of which was a real
defect somewhere in this project before.

**Every slider states its numeric range.**  The convention inherited from
``voelkel.convert`` prints endpoint labels *instead of* the range when labels
exist — ``anchors or f"Whole number from {lo} to {hi}."`` — and that is exactly
the bug that cost an entire sampling round on ICPC and Goldwert: models asked for
a number with no range given answered on an implicit 0-10 scale, putting 80-94% of
0-100 slider answers at 10 or below against 8-31% for real participants.  Here the
range is not optional, and we know precisely what the respondent saw: verification
of the Qualtrics print export confirmed the numerals "0 50 100" were rendered
under every slider track (``GridLines = 2``).  So the transcript prints both.

**Images are described, not dropped.**  Of 22 images across the arms none is in
the archive and 18 of 22 URLs are dead.  The inherited SDC policy — drop any arm
containing an image — would delete nine of thirteen arms here including all three
controls, and with them the entire control baseline.  So an image becomes a
bracketed note saying one was shown, and the modality audit records per arm what
that costs.

**The donation is one constant-sum question, not six sliders.**  ``QID232`` is
``QuestionType CS`` with ``ChoiceTotal 100``: six boxes that must add to 100 cents,
five charities plus "keep for myself".  Rendering it as six independent 0-100
sliders would let a respondent allocate 600 cents.

**Sliders come in two shapes and the difference is load-bearing.**  ``Concern`` and
``Belief`` are *separate one-bar questions* — ``Concern_Post_1``, ``_2``, ``_3`` —
whose single choice has an empty display and whose released column takes a further
``_1`` suffix.  ``Policies``, ``Intent``, ``Companies`` and ``IntentNp`` are
*single questions carrying several bars*, one per statement, exporting to
``Policies_Post_1`` … ``_3``.  Treating the second kind as one slider collapses
sixteen items into four and silently corrupts every composite built from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..survey.elements import PageBreak, Text
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot
from .paths import QSF  # noqa: F401  (re-exported for callers)

#: Question types that record nothing.
DISPLAY_ONLY = {"DB", "Timing", "Meta", "HotSpot"}

#: Blocks in the QSF that were never fielded.  The trash block matters: it holds
#: sliders on a 0-7 range with SnapToGrid, so a converter that walks every
#: question rather than the flow would mix two different scales into one study.
UNFIELDED_BLOCKS = ("Trash / Unused Questions",)

_IMG = re.compile(r"<img\b[^>]*>", re.I)


def state_range(anchors: str, lo: int, hi: int) -> str:
    """Endpoint labels *and* the numeric range.

    Never ``anchors or range`` — always both.  See the module docstring: the
    either/or form is the defect this function exists to make impossible.
    """
    stated = f"Whole number from {lo} to {hi}."
    return f"{anchors}  {stated}" if anchors else stated


def slider_bounds(payload: dict) -> tuple[int, int]:
    config = payload.get("Configuration", {}) or {}
    return int(config.get("CSSliderMin", 0)), int(config.get("CSSliderMax", 100))


def anchors_from_labels(payload: dict) -> str:
    """The scale's endpoint labels, as the respondent saw them."""
    labels = payload.get("Labels")
    if isinstance(labels, dict):
        shown = [str(v.get("Display", "")) for v in labels.values()]
    elif isinstance(labels, list):
        shown = [str(v.get("Display", "")) for v in labels if isinstance(v, dict)]
    else:
        shown = []
    cleaned = []
    for item in shown:
        text = re.sub(r"<[^>]+>", " ", item)
        text = re.sub(r"\s+", " ", text.replace("|", "")).strip()
        if text:
            cleaned.append(text)
    return " … ".join(cleaned)


def describe_images(raw_html: str) -> str:
    """A bracketed note for each ``<img>``, since no asset is in the archive.

    Deliberately says only that an image was shown and how large it was, because
    18 of 22 URLs are dead and inventing a description would be worse than
    admitting the gap.  Where the surrounding prose makes the content recoverable
    — Purity Framing's "on the left … on the right" — the arm-specific note in
    :mod:`~silicon_sampling.ccc.instrument` supplies it.
    """
    notes = []
    for tag in _IMG.findall(raw_html or ""):
        width = re.search(r'width="?(\d+)', tag)
        height = re.search(r'height="?(\d+)', tag)
        size = f", {width.group(1)}x{height.group(1)}" if width and height else ""
        notes.append(
            f"[An image was shown here{size}. Its content is not in the record.]"
        )
    return "\n".join(notes)


def _slider_bars(question, payload: dict) -> list[tuple[str, str]]:
    """``(recode, statement)`` per bar, or a single empty bar.

    A one-bar slider's only choice has an empty display (Qualtrics stores
    ``&nbsp;``), which is how the two shapes are told apart.
    """
    choices = payload.get("Choices") or {}
    order = payload.get("ChoiceOrder") or list(choices)
    recodes = payload.get("RecodeValues") or {}
    bars: list[tuple[str, str]] = []
    for key in order:
        entry = choices.get(str(key)) or choices.get(key) or {}
        display = re.sub(r"<[^>]+>", " ", str(entry.get("Display", "")))
        display = display.replace("\xa0", " ").replace("&nbsp;", " ")
        display = re.sub(r"\s+", " ", display).strip()
        bars.append((str(recodes.get(str(key), key)), display))
    if len(bars) <= 1 or not any(statement for _, statement in bars):
        return bars[:1]
    return bars


def constant_sum_slot(question, payload: dict) -> list:
    """The donation: one allocation task over a fixed total, not N sliders."""
    validation = (payload.get("Validation") or {}).get("Settings") or {}
    total = int(validation.get("ChoiceTotal", 100) or 100)
    options = tuple(o for o in question.choices if o)
    tag = question.export_tag or question.qid
    parts: list = [Text(question.text)]
    parts.append(
        Text(
            f"[These {len(options)} amounts must add up to exactly {total}.]"
            if options
            else f"[The amounts must add up to exactly {total}.]"
        )
    )
    for index, option in enumerate(options, start=1):
        parts.append(
            IntSlot(
                id=f"{tag}_{index}",
                prompt=option,
                anchors=state_range("", 0, total),
                lo=0,
                hi=total,
                max_tokens=6,
            )
        )
    return parts


def convert_question(question, payload: dict) -> list:
    """One question -> zero or more transcript elements."""
    if question.kind in {"Timing", "Meta"}:
        return []

    text = question.text
    images = describe_images(question.raw_text)
    prefix: list = []
    if images:
        prefix.append(Text(images))

    if question.kind in DISPLAY_ONLY:
        out: list = []
        if text.strip():
            out.append(Text(text))
        return out + prefix

    if question.kind == "CS":
        return prefix + constant_sum_slot(question, payload)

    slot_id = question.export_tag or question.qid

    if question.kind == "Slider":
        lo, hi = slider_bounds(payload)
        anchors = state_range(anchors_from_labels(payload), lo, hi)
        bars = _slider_bars(question, payload)
        if len(bars) <= 1:
            return prefix + [
                IntSlot(
                    id=slot_id,
                    prompt=text,
                    anchors=anchors,
                    lo=lo,
                    hi=hi,
                    max_tokens=6,
                )
            ]
        # One question, several bars: the stem is shown once, then one response
        # position per statement, each carrying its own released column.
        out: list = list(prefix)
        if text.strip():
            out.append(Text(text))
        for code, statement in bars:
            out.append(
                IntSlot(
                    id=f"{slot_id}_{code}",
                    prompt=statement,
                    anchors=anchors,
                    lo=lo,
                    hi=hi,
                    max_tokens=6,
                )
            )
        return out

    if question.kind == "MC":
        options = tuple(o for o in question.choices if o)
        if not options:
            return prefix + ([Text(text)] if text.strip() else [])
        multi = question.selector.startswith("MA")
        return prefix + [
            ChoiceSlot(
                id=slot_id,
                prompt=text,
                options=options,
                codes=question.codes,
                max_tokens=max(6, max(len(o.split()) for o in options) * 3 + 4),
                describe_as=("Select all that apply: " if multi else "Options: ")
                + " | ".join(options),
            )
        ]

    if question.kind == "TE":
        return prefix + [
            FreeTextSlot(
                id=slot_id,
                prompt=text,
                hint="Free text.",
                max_tokens=120,
                max_chars=900,
            )
        ]

    return prefix + ([Text(text)] if text.strip() else [])


@dataclass(frozen=True)
class Converted:
    """What one block became."""

    elements: list
    data_columns: dict


def data_column(question) -> str:
    """The released column this answer occupies.

    Qualtrics suffixes a slider's export tag with its single bar's recode value,
    so ``Belief_Pre_1`` is stored as ``Belief_Pre_1_1``.  The statement-per-question
    sliders in this study all carry that suffix; the ones authored as a single
    multi-statement question (``Policies``, ``Intent``, ``Companies``, ``IntentNp``)
    do not.
    """
    tag = re.sub(r"\s", "_", question.export_tag or question.qid)
    if question.kind == "Slider" and question.codes:
        return f"{tag}_{question.codes[0]}"
    return tag


def slot_columns(question, payload: dict) -> dict:
    """Slot id -> released column, for one question.

    The two slider shapes map differently: a one-bar slider's slot keeps the
    export tag and its column gains the bar's recode suffix, while a multi-bar
    slider's slots already carry the suffix and the column matches them exactly.
    """
    tag = re.sub(r"\s", "_", question.export_tag or question.qid)
    if question.kind == "CS":
        # The constant-sum boxes export one column per box, numbered from one, so
        # the slots this module creates already match them.  Falling through to
        # data_column() instead would map all six boxes onto the single column
        # named by the export tag.
        options = tuple(o for o in question.choices if o)
        return {f"{tag}_{i}": f"{tag}_{i}" for i in range(1, len(options) + 1)}
    if question.kind != "Slider":
        return {}
    bars = _slider_bars(question, payload)
    if len(bars) <= 1:
        code = bars[0][0] if bars else "1"
        return {tag: f"{tag}_{code}"}
    return {f"{tag}_{code}": f"{tag}_{code}" for code, _ in bars}


def convert_block(survey, block, payloads: dict, page_break: bool = True) -> Converted:
    """A block's questions, in builder order, as transcript elements."""
    elements: list = []
    columns: dict = {}
    if page_break:
        elements.append(PageBreak())
    for qid in block.question_ids:
        question = survey.questions.get(qid)
        if question is None:
            continue
        payload = payloads.get(qid, {})
        slider_map = slot_columns(question, payload)
        for element in convert_question(question, payload):
            elements.append(element)
            if isinstance(element, (Text, PageBreak)):
                continue
            columns[element.id] = slider_map.get(element.id) or data_column(question)
    return Converted(elements=elements, data_columns=columns)
