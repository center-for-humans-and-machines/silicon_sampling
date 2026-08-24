"""Two sources of screens, one transcript vocabulary.

The ICPC instrument reaches us twice over, and each copy is authoritative about
something the other is not.

The **``.qsf``** carries every intervention verbatim — question text, slider
bounds, endpoint labels, choice sets, the survey's own page breaks and its piped
text.  Nothing about the eleven stimuli needs to be transcribed by hand, which
matters: transcription is where a manipulation quietly acquires a word the
respondent never read.  The one thing the ``.qsf`` does *not* carry is what the
pictures showed; :mod:`~silicon_sampling.icpc.images` supplies that.

The **hand transcription in** :mod:`silicon_sampling.vlasceanu.content_shared`
carries the page boundaries recovered from the ``*_Page.Submit`` timers and the
described images, for every screen that is shared across arms.  It also carries
the WEPT number grids, which look hand-made and are not: a Profile matrix stores
its rows as empty ``Choices`` and the sixty numbers as nested ``Answers``, so a
reader who checks ``choices`` sees a blank grid and concludes the numbers were
generated client-side.  They are in the file, and
``tests/test_icpc.py::test_the_wept_grids_are_the_numbers_the_qsf_holds`` holds
the transcription to them — as *sets* of rows, because Qualtrics randomises the
row order per respondent.

The transcription also uses the *published data* column names as slot ids
(``Belief.in.CC_1``), which is what makes a sampled answer comparable without a
translation table.

That last property is also where the division of labour has a sharp edge, so the
``.qsf`` is made to police it.  A slot id *is* a column name here, so a matrix
transcribed in the wrong order does not look wrong — it looks like a survey whose
items were asked in a different sequence, and every battery-level check passes,
because the mean of four sliders does not care which slider was which.
:func:`qsf_item_wording` and :func:`codebook_item_wording` re-derive the binding
from the two authorities that do know it, and ``tests/test_icpc.py`` holds the
transcription to both on every run.

So the shared screens come from the transcription and the stimuli come from the
``.qsf``, and both are converted into the one element vocabulary
:mod:`silicon_sampling.survey.render` knows how to emit.  Two adapters, one
renderer, no third description of the survey.

One deliberate loss is recorded here rather than hidden.  Three items are
check-all-that-apply — the sharing platform, the household-goods SES index and
the psychological-distance impact list — and the transcript vocabulary has no
multi-select slot.  They become single-select.  None of the three is a scored
outcome (the sharing outcome is the *willingness* item before it), so the cost
falls entirely on one echo screen inside arm 7, which will quote back one
impact where a human might have named five.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache

from ..survey.elements import Conditional, PageBreak, Text
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot
from ..vlasceanu import elements as V
from ..voelkel.convert import anchors_from_labels, resolve_pipes, slider_bounds
from ..voelkel.qsf import Question, Survey, load_survey, strip_html
from .images import describe as describe_image
from .paths import CODEBOOK, QSF

#: Qualtrics question types that record nothing a respondent produced.
DISPLAY_ONLY = {"DB", "Timing", "Meta", "HotSpot"}

#: Suffixes Qualtrics adds to a codebook label in place of an item statement.
#: ``"What is your gender? - Selected Choice"`` names no item; it says the column
#: holds the radio button rather than its free-text companion.
QUALTRICS_LABEL_ANNOTATIONS = frozenset({"Selected Choice", "Text"})

#: Text boxes the survey validated as numbers, and the range a legal answer falls
#: in here.
#:
#: A Qualtrics text box carries its type in ``Validation``, not in its question
#: type: both of these are ``TE``/``SL``, indistinguishable from a comment box
#: until you read ``ContentType == "ValidNumber"``.  Rendered as free text they
#: admit an answer of a *kind* no participant could give — the two columns hold
#: 726 and 692 human answers and not one of them is non-numeric, because
#: Qualtrics refused to advance the page — and a sampled essay would then sit in
#: a column every downstream read treats as a number.
#:
#: The bound is ours and the tradeoff is real: Qualtrics set no ``Min`` or ``Max``,
#: so the screen accepted any number, and a handful of participants answered in
#: the billions.  A range has to be stated all the same, because a slider or a
#: number box with no range stated is the exact defect that cost this study its
#: first run.  1000 years is chosen to cover every answer that is an answer —
#: the medians are 30 and 10 — at the cost of the joke replies, which are noise
#: in a covariate nothing is scored against.
QSF_NUMBER_RANGE: dict[str, tuple[int, int]] = {
    "negEmo_cliThreshTime": (0, 1000),
    "1.5 Threshold": (0, 1000),
}

_IMG_RE = re.compile(r"<img\b[^>]*>", re.I)
_SRC_RE = re.compile(r'src="([^"]+)"', re.I)
_IM_ID_RE = re.compile(r"IM=(IM_\w+)")
_NON_NAME = re.compile(r"[^A-Za-z0-9._]")


@dataclass(frozen=True)
class Converted:
    """What one block became."""

    elements: list
    #: Slot id -> the column that answer occupies in the published data.
    data_columns: dict


# --------------------------------------------------------------------------- #
# the .qsf side
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Loaded:
    """The parsed survey plus the things derived from it once."""

    survey: Survey
    payloads: dict
    block_elements: dict
    by_description: dict


@lru_cache(maxsize=2)
def loaded(path=QSF) -> Loaded:
    """Parse the ``.qsf`` once, keeping the raw payloads and block element lists.

    ``voelkel.qsf`` drops a block's page breaks because that study's stimuli were
    one screen each.  Here they are load-bearing — the collective-action arm is
    thirteen screens — so the raw ``BlockElements`` list is kept alongside.
    """
    survey = load_survey(path, condition_field="cond")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payloads = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in raw["SurveyElements"]
        if element["Element"] == "SQ"
    }
    block_payload = next(e for e in raw["SurveyElements"] if e["Element"] == "BL")[
        "Payload"
    ]
    entries = (
        block_payload.values() if isinstance(block_payload, dict) else block_payload
    )
    block_elements = {
        entry["ID"]: list(entry.get("BlockElements", []) or []) for entry in entries
    }
    return Loaded(
        survey=survey,
        payloads=payloads,
        block_elements=block_elements,
        by_description={block.description: block for block in survey.blocks.values()},
    )


def make_names(tag: str) -> str:
    """R's ``make.names``, which is what produced the published column names.

    The export ran through R, so ``"1.5 Threshold"`` is stored as
    ``X1.5.Threshold`` and ``"Q8: Geo"`` as ``Q8..Geo``.  Reproducing the mangling
    is the only way a slot id can be pointed at the column holding the answers
    respondents actually gave.
    """
    name = _NON_NAME.sub(".", tag.strip())
    if not name:
        return "X"
    if name[0].isdigit() or (name[0] == "." and name[1:2].isdigit()):
        name = "X" + name
    return name


def echo_safe(tag: str) -> str:
    """A slot id the transcript's echo marker can name.

    ``<<=id>>`` is matched by ``[A-Za-z0-9_]+``, so an id carrying R's dots would
    survive rendering and then fail to resolve — leaving a literal marker in the
    prompt of the one screen that quotes the respondent back at themselves.  Slot
    ids on the ``.qsf`` side are therefore word characters only, and the published
    column they map to is recorded separately.
    """
    name = re.sub(r"[^A-Za-z0-9_]", "_", tag.strip()) or "X"
    return "X" + name if name[0].isdigit() else name


def slot_id(question: Question, code: str | None = None) -> str:
    """The transcript's name for this response position."""
    stem = echo_safe(question.export_tag or question.qid)
    return stem if code is None else f"{stem}_{code}"


def data_column(question: Question, code: str | None = None) -> str:
    """Where this question's answer sits in the published data."""
    stem = make_names(question.export_tag or question.qid)
    if code is None:
        return stem
    return f"{stem}_{code}"


def qsf_item_wording(path=QSF) -> dict[str, str]:
    """Published column -> the statement printed beside it, for every battery row.

    This is the authority on the item-to-column binding.  None of the batteries
    defines ``RecodeValues``, ``ChoiceDataExportTags`` or ``VariableNaming``, so
    Qualtrics' export suffix is the choice code verbatim and the mapping needs no
    interpretation.  Single unlabelled bars are left out: their wording is the
    question text rather than a choice display, and a transcript folds the two
    into one stem.
    """
    state = loaded(path)
    wording: dict[str, str] = {}
    for question in state.survey.questions.values():
        if question.kind != "Slider":
            continue
        rows = list(zip(question.choices, question.codes or question.choices))
        if len(rows) == 1 and not rows[0][0].strip():
            continue
        for label, code in rows:
            wording[data_column(question, code)] = label
    return wording


def codebook_item_wording(path=CODEBOOK) -> dict[str, str]:
    """The same binding as the study published it, out of ``codebook.xlsx``.

    An independent authority, and worth the Excel read: it was produced from the
    live survey rather than from the ``.qsf``, so agreement between the two rules
    out a mistake in how this module reads a ``.qsf`` as easily as it rules out a
    mistake in the transcription.  Labels are ``"<question stem> - <item>"``, and
    only the part after the last dash is the item — except where Qualtrics
    appended one of its own annotations instead of an item, which is not wording
    anybody read and is dropped.
    """
    import pandas as pd

    frame = pd.read_excel(path)
    wording: dict[str, str] = {}
    for name, label in zip(frame["Variable"], frame["Label"]):
        if not isinstance(name, str) or not isinstance(label, str):
            continue
        _, separator, item = label.rpartition(" - ")
        item = " ".join(item.split())
        if separator and item not in QUALTRICS_LABEL_ANNOTATIONS:
            wording[name] = item
    return wording


def qsf_gated_stems(path=QSF) -> frozenset[str]:
    """Column stems of every question the survey showed only conditionally.

    Read off ``DisplayLogic`` rather than out of a sentence, and returned as
    *stems* because the one gated matrix — a WEPT number grid — ships empty in
    this ``.qsf`` and so cannot say how many columns it will occupy.  A stem
    covers the column itself and every ``<stem>_<row>`` beneath it, which is
    enough to assert that a transcript gates exactly what Qualtrics gated.
    """
    state = loaded(path)
    return frozenset(
        make_names(state.survey.questions[qid].export_tag or qid)
        for qid, payload in state.payloads.items()
        if payload.get("DisplayLogic") and qid in state.survey.questions
    )


def is_gated(column: str, stems=None) -> bool:
    """Was the question behind this published column shown conditionally?"""
    for stem in qsf_gated_stems() if stems is None else stems:
        if column == stem or column.startswith(f"{stem}_"):
            return True
    return False


def numeric_bounds(question: Question, payload: dict) -> tuple[int, int] | None:
    """The range this text box accepted, or ``None`` if it accepted prose."""
    settings = (payload.get("Validation") or {}).get("Settings") or {}
    if settings.get("ContentType") != "ValidNumber":
        return None
    return QSF_NUMBER_RANGE.get(question.export_tag or question.qid, (0, 1000))


def with_images(raw_html: str) -> str:
    """Replace every ``<img>`` with a described ``[IMAGE: ...]`` line.

    Done before the HTML is stripped, because stripping first would leave a blank
    screen where the manipulation was.
    """

    def replace(match: re.Match) -> str:
        src = _SRC_RE.search(match.group())
        url = src.group(1) if src else ""
        found = _IM_ID_RE.search(url)
        key = found.group(1) if found else url
        return f"\n[IMAGE: {describe_image(key)}]\n"

    return _IMG_RE.sub(replace, raw_html or "")


def visible_text(question: Question, qid_to_slot: dict[str, str]) -> str:
    """What was on screen for this question: its text, its pictures, its pipes."""
    return resolve_pipes(strip_html(with_images(question.raw_text)), qid_to_slot)


def qsf_escape_note(payload: dict) -> tuple[str, str]:
    """A ``.qsf`` slider's endpoint labels, and the escape box it showed.

    Two things have to be read separately here and reading them as one is a
    defect.  Qualtrics files the "Not Applicable" checkbox's *wording* under the
    ``"NA"`` key of ``Labels``, alongside the scale's endpoint labels, and files
    whether the checkbox was ever *shown* under ``Configuration.NotApplicable``.
    The two disagree in this instrument: nine ``.qsf`` sliders carry an ``"NA"``
    label and only five had the box turned on, so ``Enviro_motiv``, ``Enviro_ID``,
    ``ID_hum`` and ``ID_GC`` have wording for a control the respondent never saw.

    Handing ``Labels`` straight to ``anchors_from_labels`` gets both cases wrong at
    once.  On those four it prints "Strongly disagree … Strongly agree … Not
    Applicable", inventing a third scale point out of dead JSON; on the five where
    the box was real it buries a separate control in the endpoint line instead of
    naming it as one.  So the flag decides, the label only supplies the words, and
    the note is worded exactly as the hand-transcribed side words it -- see
    :func:`_anchor_line` -- because the same screen is reached both ways and the
    two descriptions of it should not differ.

    None of the four currently reaches a template: every block that shows them is
    hand-transcribed today.  This is the ``.qsf`` path being made correct before
    something is routed through it, which is how the same read went wrong in the
    Goldwert converter.
    """
    labels = payload.get("Labels")
    scale = payload
    escape = ""
    if isinstance(labels, dict):
        scale = {
            **payload,
            "Labels": {k: v for k, v in labels.items() if str(k) != "NA"},
        }
        entry = labels.get("NA")
        if isinstance(entry, dict):
            escape = strip_html(str(entry.get("Display", ""))).strip()
    if not (payload.get("Configuration") or {}).get("NotApplicable"):
        escape = ""
    elif not escape:
        escape = "Not Applicable"
    return anchors_from_labels(scale), escape


def _slider_slots(
    question: Question, payload: dict, text: str
) -> tuple[list, dict[str, str]]:
    """One 0-100 slider, or a stack of them sharing a header.

    A single unlabelled bar takes the question text as its own stem; a stack
    prints the header once and gives each row its statement.
    """
    low, high = slider_bounds(payload)
    labelled, escape = qsf_escape_note(payload)
    anchors = _anchor_line(labelled, "", None, escape or None, low, high)
    rows = [
        (label, code)
        for label, code in zip(question.choices, question.codes or question.choices)
    ]
    elements: list = []
    columns: dict[str, str] = {}
    single = len(rows) == 1 and not rows[0][0].strip()
    if text.strip() and not single:
        elements.append(Text(text))
    for label, code in rows:
        slot = IntSlot(
            id=slot_id(question, code),
            prompt=text if single else label,
            anchors=anchors,
            lo=low,
            hi=high,
            max_tokens=6,
        )
        elements.append(slot)
        columns[slot.id] = data_column(question, code)
    return elements, columns


def _choice_slot(question: Question, text: str) -> tuple[list, dict[str, str]]:
    options = tuple(option for option in question.choices if option)
    if not options:
        return ([Text(text)] if text.strip() else []), {}
    multi = question.selector.startswith("MA")
    slot = ChoiceSlot(
        id=slot_id(question),
        prompt=text,
        options=options,
        codes=question.codes,
        max_tokens=max(8, max(len(o.split()) for o in options) * 3 + 4),
        describe_as=(
            "Select all that apply (one option per line; this transcript records one): "
            if multi
            else "Options: "
        )
        + " | ".join(options),
    )
    return [slot], {slot.id: data_column(question)}


def convert_question(
    question: Question, payload: dict, qid_to_slot: dict[str, str]
) -> tuple[list, dict[str, str]]:
    """One Qualtrics question -> transcript elements and their data columns."""
    if question.kind in {"Timing", "Meta"}:
        return [], {}
    text = visible_text(question, qid_to_slot)
    if question.kind in DISPLAY_ONLY:
        return ([Text(text)] if text.strip() else []), {}
    if question.kind == "Slider":
        return _slider_slots(question, payload, text)
    if question.kind == "MC":
        return _choice_slot(question, text)
    if question.kind == "TE":
        bounds = numeric_bounds(question, payload)
        if bounds is not None:
            low, high = bounds
            number = IntSlot(
                id=slot_id(question),
                prompt=text,
                anchors=state_range("", low, high),
                lo=low,
                hi=high,
                max_tokens=6,
            )
            return [number], {number.id: data_column(question)}
        slot = FreeTextSlot(
            id=slot_id(question),
            prompt=text,
            hint="Free text.",
            max_tokens=160 if question.selector == "ESTB" else 60,
            max_chars=2000 if question.selector == "ESTB" else 200,
        )
        return [slot], {slot.id: data_column(question)}
    return ([Text(text)] if text.strip() else []), {}


def slot_index(path=QSF) -> dict[str, str]:
    """QID -> slot id, so a piped answer can name the slot it echoes."""
    state = loaded(path)
    index: dict[str, str] = {}
    for qid, question in state.survey.questions.items():
        payload = state.payloads.get(qid, {})
        if question.kind == "Slider" and question.codes:
            index[qid] = slot_id(question, question.codes[0])
        else:
            index[qid] = slot_id(question)
        del payload
    return index


def convert_qsf_block(description: str, path=QSF) -> Converted:
    """A ``.qsf`` block as transcript elements, honouring its own page breaks."""
    state = loaded(path)
    block = state.by_description[description]
    index = slot_index(path)
    elements: list = [PageBreak()]
    columns: dict[str, str] = {}
    for entry in state.block_elements[block.bid]:
        if entry.get("Type") == "Page Break":
            # Two page breaks in a row render as a page with nothing on it. This
            # block is entered with one already emitted, so a block whose own first
            # element is a Page Break -- arm 3's is -- produced an empty page 4,
            # the only empty page in the twelve templates. Qualtrics does not show
            # a blank screen for a doubled break either, so collapsing them is
            # closer to the instrument, not a cosmetic tidy-up.
            if not (elements and isinstance(elements[-1], PageBreak)):
                elements.append(PageBreak())
            continue
        question = state.survey.questions.get(entry.get("QuestionID"))
        if question is None:
            continue
        produced, mapped = convert_question(
            question, state.payloads.get(question.qid, {}), index
        )
        elements.extend(produced)
        columns.update(mapped)
    return Converted(elements=elements, data_columns=columns)


# --------------------------------------------------------------------------- #
# the hand-transcription side
# --------------------------------------------------------------------------- #

#: Free numeric entries on the *transcription* side, and the range a legal answer
#: falls in.  Only ``Age`` reaches here, and 18-100 is the range the cleaning
#: script kept; the ``.qsf`` side has its own map, :data:`QSF_NUMBER_RANGE`,
#: because there the numeric type has to be read out of ``Validation`` first.
NUMBER_RANGE = {"Age": (18, 100)}


def state_range(anchors: str, lo: int, hi: int) -> str:
    """Endpoint labels *and* the numeric range, because labels alone are not enough.

    The project's slider convention prints the endpoint labels the respondent saw
    and nothing else — "Not at all accurate … Extremely accurate" — which is
    faithful to the screen and useless to a model, because the screen also showed
    a 0-100 track and the transcript does not.  Asked for a number with no range
    given, the models answered on a small scale: 80% to 94% of every 0-100 slider
    answer in this study came back as an integer of 10 or less, against 8% to 31%
    for real participants, and mean control-arm level error ran 20 to 47 points on
    a 0-100 scale.

    That is not a modelling failure, it is a missing sentence.  Voelkel and Pfänder
    do not have the problem because their source questionnaires happen to state the
    range in prose ("Below is a range from 0 to 100 …", "0 = Not important at all,
    50 = Moderately important, 100 = Extremely important") — so the convention held
    up by luck on the two studies built first and broke on the two built later.
    Saying the range on every integer slider makes it hold on purpose.
    """
    stated = f"Whole number from {lo} to {hi}."
    return f"{anchors}  {stated}" if anchors else stated


def _anchor_line(
    left: str,
    right: str,
    mid: str | None,
    extra: str | None,
    lo: int | None = None,
    hi: int | None = None,
) -> str:
    """The scale's endpoints as the respondent saw them, plus any escape option."""
    parts = [part for part in (left, mid, right) if part]
    line = " … ".join(parts)
    if extra:
        line += f" (the screen also offered a '{extra}' box)"
    if lo is not None and hi is not None:
        return state_range(line, lo, hi)
    return line or "Whole number from 0 to 100."


def convert_element(element) -> tuple[list, dict[str, str]]:
    """One hand-transcribed element -> transcript elements and data columns."""
    if isinstance(element, V.Text):
        style = element.style if element.style in {"body", "head", "cite"} else "body"
        return ([Text(element.text, style=style)] if element.text.strip() else []), {}
    if isinstance(element, V.Bullets):
        body = "\n".join(f"{element.marker} {item}" for item in element.items)
        return [Text(body)], {}
    if isinstance(element, V.Image):
        body = f"[IMAGE: {' '.join(element.alt.split())}]"
        if element.caption:
            body += f"\n{element.caption}"
        return [Text(body)], {}
    if isinstance(element, V.Echo):
        return [Text(f"<<={element.column}>>")], {}
    if isinstance(element, V.Slider):
        slot = IntSlot(
            id=element.slot,
            prompt=element.stem or "",
            anchors=_anchor_line(
                element.left,
                element.right,
                element.mid,
                element.extra,
                element.lo,
                element.hi,
            ),
            lo=element.lo,
            hi=element.hi,
            max_tokens=6,
        )
        return [slot], {slot.id: slot.id}
    if isinstance(element, V.Matrix):
        anchors = _anchor_line(
            element.left,
            element.right,
            element.mid,
            element.extra,
            element.lo,
            element.hi,
        )
        out: list = []
        columns: dict[str, str] = {}
        for slot_id, label in element.items:
            slot = IntSlot(
                id=slot_id,
                prompt=label,
                anchors=anchors,
                lo=element.lo,
                hi=element.hi,
                max_tokens=6,
            )
            out.append(slot)
            columns[slot_id] = slot_id
        return out, columns
    if isinstance(element, (V.Choice, V.MultiChoice)):
        options = tuple(element.options)
        multi = isinstance(element, V.MultiChoice)
        slot = ChoiceSlot(
            id=element.slot,
            prompt="",
            options=options,
            max_tokens=max(8, max(len(o.split()) for o in options) * 3 + 4),
            describe_as=(
                "Select all that apply (one option per line; this transcript records one): "
                if multi
                else "Options: "
            )
            + " | ".join(options),
        )
        return [slot], {slot.id: slot.id}
    if isinstance(element, V.Number):
        low, high = NUMBER_RANGE.get(element.slot, (0, 1000))
        slot = IntSlot(
            id=element.slot,
            prompt=element.hint,
            anchors=f"Whole number from {low} to {high}.",
            lo=low,
            hi=high,
            max_tokens=6,
        )
        return [slot], {slot.id: slot.id}
    if isinstance(element, V.FreeText):
        slot = FreeTextSlot(
            id=element.slot, prompt=element.hint, max_tokens=160, max_chars=2000
        )
        return [slot], {slot.id: slot.id}
    if isinstance(element, V.NumberGrid):
        return _number_grid(element)
    raise TypeError(f"not an ICPC survey element: {element!r}")  # pragma: no cover


def _number_grid(grid: V.NumberGrid) -> tuple[list, dict[str, str]]:
    """A WEPT row of ten two-digit numbers, and the row's answer.

    The published data records *which boxes were ticked*, not the numbers, and
    the effort outcome is the count of pages accepted rather than anything inside
    a page — so the row answer is a free-text list and is never scored.  It is
    rendered anyway because a respondent who skipped straight from "yes" to the
    next screen without seeing sixty numbers did not do this task.
    """
    out: list = []
    columns: dict[str, str] = {}
    for slot_id, row in zip(grid.slots, grid.rows):
        out.append(Text("   ".join(f"{number:02d}" for number in row)))
        slot = FreeTextSlot(
            id=slot_id,
            prompt="Which numbers in this row are target numbers?",
            hint=(
                "The target numbers from this row, comma separated, "
                "or 'none' if there are none."
            ),
            max_tokens=40,
            max_chars=120,
        )
        out.append(slot)
        columns[slot_id] = slot_id
    return out, columns


#: Widgets the transcription describes without a stem of their own, because on
#: the real screen the question text sat above them as its own display element.
#: Qualtrics stored both as *one* question, and the Voelkel templates render it
#: that way — the whole on-screen text is the stem — so the preceding copy is
#: folded in rather than left as an orphaned paragraph above a stemless
#: ``Response:`` line.
STEMLESS = (V.Choice, V.MultiChoice, V.Number, V.FreeText)


def _takes_preceding_text(element) -> bool:
    if isinstance(element, STEMLESS):
        return True
    return isinstance(element, V.Slider) and not (element.stem or "").strip()


def _screen_elements(screen: V.Screen) -> tuple[list, dict[str, str]]:
    """One screen's elements, with its copy folded into the item it introduces."""
    produced: list = []
    columns: dict[str, str] = {}
    pending: list[str] = []
    for element in screen.elements:
        if isinstance(element, (V.Text, V.Bullets, V.Image, V.Echo)):
            made, _ = convert_element(element)
            pending.extend(text.text for text in made)
            continue
        made, mapped = convert_element(element)
        columns.update(mapped)
        if pending and _takes_preceding_text(element) and made:
            head = made[0]
            stem = "\n\n".join(pending + ([head.prompt] if head.prompt else []))
            made = [replace(head, prompt=stem)] + list(made[1:])
        else:
            produced.extend(Text(text) for text in pending)
        pending = []
        produced.extend(made)
    produced.extend(Text(text) for text in pending)
    return produced, columns


def convert_screens(block: V.Block, page_break: bool = True) -> Converted:
    """A hand-transcribed block as transcript elements, one page break per screen.

    A screen carrying a :class:`~silicon_sampling.vlasceanu.elements.Gate` becomes
    a :class:`~silicon_sampling.survey.elements.Conditional`; its response
    positions are still reported in ``data_columns``, because the column exists
    whether or not this respondent reached it.
    """
    elements: list = []
    columns: dict[str, str] = {}
    for index, screen in enumerate(block.screens):
        produced: list = []
        if page_break or index:
            produced.append(PageBreak())
        made, mapped = _screen_elements(screen)
        produced.extend(made)
        columns.update(mapped)
        if screen.condition and screen.gate is None:
            # A `condition` on a gated screen becomes the Conditional's note, which
            # a template prints and a session evaluates. On an *ungated* screen it
            # used to become nothing at all, and exactly one screen in this
            # instrument is in that position: the WEPT practice grid, whose
            # CustomValidation pinned all nine cells and would not let the page
            # advance until the respondent had ticked {67, 85} and {23, 81}. That
            # is the only reason the transcript shows a correct answer there rather
            # than the respondent's own -- the demo answers are prefilled -- so
            # dropping the sentence left two answers appearing out of nowhere.
            produced.append(Text(f"[ {screen.condition} ]"))
        if screen.gate is not None:
            elements.append(
                Conditional(
                    note=screen.condition or screen.gate.describe(),
                    predicate=screen.gate.matches,
                    elements=produced,
                )
            )
        else:
            elements.extend(produced)
    return Converted(elements=elements, data_columns=columns)
