"""Turn parsed Qualtrics questions into transcript elements and response slots.

The Pfänder instrument was small enough to transcribe by hand.  This one is not —
1,168 questions across 120 blocks — and it does not need to be: the ``.qsf``
carries the question text, the answer options, the recode values and the slider
ranges, so the conversion can be mechanical and therefore checkable.  Anything
the file does not determine (which blocks a respondent sees, in what order) is
resolved in :mod:`~silicon_sampling.voelkel.instrument`.

Two Qualtrics mechanisms matter here and both map onto machinery the transcript
renderer already has:

``${e://Field/Inparty_Person}``
    Piped embedded data.  This is how the instrument adapts to the respondent's
    party — the same sentence reads "Republicans" or "Democrats" depending on who
    is answering — and it becomes an ``<<=field>>`` echo.

``${q://QID355/ChoiceNumericEntryValue/1}``
    A piped *answer*: the correction screens in `Misperception_Competition` quote
    the respondent's own estimate back at them before giving the true figure.
    It becomes an echo of that question's slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..survey.elements import PageBreak, Text
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot
from .qsf import Question, Survey, strip_html

_PIPE_FIELD = re.compile(r"\$\{e://Field/([A-Za-z0-9_]+)\}")
_PIPE_QUESTION = re.compile(r"\$\{q://(QID\d+)/[^}]*\}")
_PIPE_OTHER = re.compile(r"\$\{[^}]*\}")

#: Qualtrics question types that record nothing.
DISPLAY_ONLY = {"DB", "Timing", "Meta", "HotSpot"}


@dataclass(frozen=True)
class Converted:
    """What one block became."""

    elements: list
    #: Slot id -> the column that answer occupies in the published data.
    data_columns: dict


def resolve_pipes(text: str, qid_to_slot: dict[str, str]) -> str:
    """Rewrite Qualtrics pipes as transcript echoes.

    A pipe the transcript cannot resolve — a timestamp, a panel id, a scoring
    variable — is dropped rather than left in, because a literal ``${...}`` in
    the prompt is a token sequence no respondent ever saw.
    """
    text = _PIPE_FIELD.sub(lambda m: f"<<={m.group(1)}>>", text)
    text = _PIPE_QUESTION.sub(
        lambda m: f"<<={qid_to_slot.get(m.group(1), m.group(1))}>>", text
    )
    return _PIPE_OTHER.sub("", text)


def slider_bounds(question_payload: dict) -> tuple[int, int]:
    config = question_payload.get("Configuration", {}) or {}
    return int(config.get("CSSliderMin", 0)), int(config.get("CSSliderMax", 100))


def anchors_from_labels(payload: dict) -> str:
    """The scale's endpoint labels, as the respondent saw them."""
    labels = payload.get("Labels")
    if isinstance(labels, dict):
        shown = [strip_html(str(v.get("Display", ""))) for v in labels.values()]
    elif isinstance(labels, list):
        shown = [
            strip_html(str(v.get("Display", ""))) for v in labels if isinstance(v, dict)
        ]
    else:
        shown = []
    shown = [re.sub(r"\s+", " ", s.replace("|", "")).strip() for s in shown]
    shown = [s for s in shown if s]
    return " … ".join(shown)


def data_column(question: Question) -> str:
    """The column this question's answer occupies in the published data.

    Qualtrics suffixes a slider's export tag with the recode value of its single
    bar, so ``SPV_1`` is stored as ``SPV_1_2``.  Recovering that lets every slot
    be checked against the values respondents actually gave.
    """
    # Qualtrics turns whitespace in an export tag into an underscore on export,
    # so a tag of "SUC_4 " is stored as "SUC_4_". Mirror that or the column is
    # unfindable.
    tag = re.sub(r"\s", "_", question.export_tag)
    if question.kind == "Slider" and question.codes:
        return f"{tag}_{question.codes[0]}"
    return tag


def convert_question(question: Question, payload: dict, qid_to_slot: dict[str, str]):
    """One question -> a transcript element, or ``None`` if it shows nothing."""
    if question.kind in {"Timing", "Meta"}:
        # Page timers and browser metadata are recorded, never displayed.
        return None
    text = resolve_pipes(question.text, qid_to_slot)
    if question.kind in DISPLAY_ONLY:
        return Text(text) if text.strip() else None

    slot_id = question.export_tag or question.qid

    if question.kind == "Slider":
        lo, hi = slider_bounds(payload)
        anchors = anchors_from_labels(payload)
        return IntSlot(
            id=slot_id,
            prompt=text,
            anchors=anchors or f"Whole number from {lo} to {hi}.",
            lo=lo,
            hi=hi,
            max_tokens=6,
        )

    if question.kind == "MC":
        multi = question.selector.startswith("MA")
        options = tuple(o for o in question.choices if o)
        if not options:
            return Text(text) if text.strip() else None
        return ChoiceSlot(
            id=slot_id,
            prompt=text,
            options=options,
            codes=question.codes,
            max_tokens=max(6, max(len(o.split()) for o in options) * 3 + 4),
            describe_as=("Select all that apply: " if multi else "Options: ")
            + " | ".join(options),
        )

    if question.kind == "TE":
        return FreeTextSlot(
            id=slot_id, prompt=text, hint="Free text.", max_tokens=80, max_chars=600
        )

    return Text(text) if text.strip() else None


def convert_block(
    survey: Survey,
    block,
    payloads: dict,
    qid_to_slot: dict[str, str],
    page_break: bool = True,
) -> Converted:
    """A block's questions, in builder order, as transcript elements."""
    elements: list = []
    columns: dict = {}
    if page_break:
        elements.append(PageBreak())
    for qid in block.question_ids:
        question = survey.questions.get(qid)
        if question is None:
            continue
        element = convert_question(question, payloads.get(qid, {}), qid_to_slot)
        if element is None:
            continue
        elements.append(element)
        if not isinstance(element, (Text, PageBreak)):
            columns[element.id] = data_column(question)
    return Converted(elements=elements, data_columns=columns)


def build_slot_index(survey: Survey) -> dict[str, str]:
    """QID -> slot id, so a piped answer can name the slot it echoes."""
    return {
        qid: (question.export_tag or qid) for qid, question in survey.questions.items()
    }
