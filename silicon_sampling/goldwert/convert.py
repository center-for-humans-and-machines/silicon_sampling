"""Turn Goldwert's Qualtrics questions into transcript elements and response slots.

The Voelkel converter handles four question types because the Strengthening
Democracy Challenge only used four.  This instrument uses eight, and three of the
extra ones carry outcomes we care about: the donation is a *constant-sum* item
(allocate ten dollars across two boxes, and the boxes must total ten), the
efficacy mediator is a Likert *matrix*, and half the sliders are multi-row, so one
question is several response positions.  Handling them here rather than letting
them fall through as descriptive text is the difference between a transcript that
asks for the donation and one that merely mentions it.

Two further Goldwert-specific mechanisms live here.

**Slot ids are the names in the published file, not the names in the survey.**
Qualtrics exported the donation as ``donation_1``/``donation_2`` and the ten
emotion sliders as ``Q65_1``…``Q65_10``; the authors' cleaning script renamed
them to ``donation``/``donation_keep`` and ``Anger``…``Disgust`` before
publishing.  :data:`PUBLISHED_COLUMN` is that rename table, transcribed from
``Advocacy_Cleaning_main.ipynb``, and it is applied at slot-construction time so
that a sampled answer needs no translation to sit beside a real one.

**Media becomes a bracketed note, not silence.**  Eleven of the eighteen arms
survive the modality audit, and several of those still display a photograph or a
chart beside prose that already states everything the picture states.  Deleting
the ``<img>`` silently would leave a transcript claiming the respondent read a
page that, as displayed, had something else on it.  A single bracketed line
saying an image was there — carrying its caption or alt text when Qualtrics
supplies one, and nothing when it does not — records the gap instead of hiding
it, and never invents a description of a picture nobody here has seen.

The same line has to distinguish three things an ``<iframe>`` can be in this
survey, because four of the nine outcome pages are built on one: the petition is
an Environmental Defense Fund action page, the two newsletter signups are the
organisations' own subscribe forms, and the bank score is a lookup on
``bank.green``.  Those are live third-party panels the respondent acted inside,
not videos, and calling them videos would misdescribe the outcome itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from ..survey.elements import Conditional, PageBreak, Text
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from ..voelkel.convert import anchors_from_labels, resolve_pipes, slider_bounds
from ..voelkel.qsf import Question, Survey, strip_html

#: Qualtrics question types that record nothing a respondent typed.
DISPLAY_ONLY = {"DB", "Timing", "Meta", "HotSpot", "GRB"}

#: Survey export name -> column in ``goldwert_etal2026.csv``.  Transcribed from
#: the rename table in the authors' ``Advocacy_Cleaning_main.ipynb``.
PUBLISHED_COLUMN = {
    "pol_candidate_2": "pol_candidate",
    "pol_campagin_2": "pol_campaign",
    "letter": "letter_content",
    "bank_1": "bank_raw",
    "donation_1": "donation",
    "donation_2": "donation_keep",
    "march_1": "march",
    "newsletter": "newsletter1",
    "cclobby": "newsletter2",
    "conversation_2": "conversation",
    "flyless_2": "flyless",
    "lessbeef_2": "lessbeef",
    "PEfficacy_1": "Pefficacy",
    "CEfficacy_1": "Cefficacy",
    "Q65_1": "Anger",
    "Q65_2": "Sadness",
    "Q65_3": "Fear",
    "Q65_4": "Guilt",
    "Q65_5": "Hope",
    "Q65_6": "Pride",
    "Q65_7": "Disappointment",
    "Q65_8": "Anxiety",
    "Q65_9": "Joy",
    "Q65_10": "Disgust",
    # Qualtrics turns the space in the tag "Education 2" into an underscore on
    # export, and the authors renamed it from the spaced spelling, so both
    # spellings have to land on the same column.
    "Education 2": "Edu",
    "Education_2": "Edu",
    "Politics2_1": "Politics_Soc",
    "Politics2_9": "Politics_Econ",
    "politics": "Party",
    # De-identified out of the published file. The letter *text* is gone; what
    # survives is `letter`, a 0/1 code of whether it expressed clear thoughts
    # about climate change, so this slot has no column to be scored against.
    "letter": "letter_content",
}

_MEDIA_TAG = re.compile(r"<(img|iframe|video|audio)\b[^>]*>", re.I)
_ALT = re.compile(r'alt="([^"]{2,160})"', re.I)
_SRC = re.compile(r'src="([^"]{0,400})"', re.I)
_VIDEO_HOST = re.compile(r"youtube|youtu\.be|vimeo|wistia|\.mp4\b|/File\.php", re.I)
_YOUTUBE = re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{6,})", re.I)
_PIPE_FIELD = re.compile(r"\$\{e://Field/([A-Za-z0-9_]+)\}")


@dataclass(frozen=True)
class Converted:
    """What one block became."""

    elements: list
    #: Slot id -> the column that answer occupies in the published data.
    data_columns: dict = field(default_factory=dict)
    #: Slot id -> the display condition gating it, where the survey sets one.
    display_logic: dict = field(default_factory=dict)


def published_column(name: str) -> str:
    """The name this Qualtrics export column carries in the published file."""
    return PUBLISHED_COLUMN.get(name, name)


def export_column(question: Question, row_key: str = "") -> str:
    """The column Qualtrics writes this question — or this row of it — into.

    Qualtrics suffixes a multi-row question's export tag with the row's *key* for
    sliders and matrices, but with the row's *position* for constant-sum items.
    Both spellings show up in the published file (``Politics2_9`` alongside
    ``donation_2``), so both rules are reproduced rather than guessed at.
    """
    tag = re.sub(r"\s", "_", question.export_tag) or question.qid
    return f"{tag}_{row_key}" if row_key else tag


def media_notes(question: Question, payload: dict) -> list[Text]:
    """Bracketed placeholders for whatever was on the page besides words.

    Deliberately content-free.  The note says what kind of thing was displayed and
    repeats a caption, an alt text or a host only when Qualtrics stored one; it
    never describes a picture, because nothing in this pipeline has looked at one.
    """
    notes: list[Text] = []
    for found in _MEDIA_TAG.finditer(question.raw_text):
        notes.append(Text(f"[ {_describe_media(found.group(0))} ]", style="cite"))
    if payload.get("Graphics"):
        described = str(payload.get("GraphicsDescription") or "").strip()
        detail = f": {described}" if described else ""
        notes.append(
            Text(f"[ image shown here{detail} — not reproduced ]", style="cite")
        )
    return notes


def _describe_media(tag: str) -> str:
    """What one media tag was, said without inventing its content."""
    name = re.match(r"<(\w+)", tag).group(1).lower()
    src = _SRC.search(tag)
    href = src.group(1) if src else ""
    if name == "audio":
        return "audio clip played here — not reproduced"
    if name == "video" or (name == "iframe" and _VIDEO_HOST.search(href)):
        found = _YOUTUBE.search(href)
        where = f" (YouTube {found.group(1)})" if found else ""
        return f"video shown here{where} — not reproduced"
    if name == "iframe":
        host = re.sub(r"^https?://(www\.)?", "", href).split("/")[0]
        where = f" from {host}" if host else ""
        return (
            f"live web panel embedded here{where}, which the participant acted inside"
        )
    alt = _ALT.search(tag)
    detail = f": {strip_html(alt.group(1))}" if alt else ""
    return f"image shown here{detail} — not reproduced"


def resolve_embedded(question: Question, embedded: dict) -> Question:
    """Paste the survey flow's own embedded values into a question's text.

    One arm is built this way and would otherwise be empty.  ``Misperception
    Correction`` holds its six correction paragraphs — roughly 700 words, the
    whole content of the intervention — as ``EmbeddedData`` values in the survey
    flow, and its feedback pages display them as ``${e://Field/employment_text}``.
    Left unresolved those pages read as blank screens; resolved, the arm is what
    the respondent actually read.  Fields the flow declares without a value
    (``text``, ``col``, ``img`` — set at runtime from the respondent's own answer)
    are left as echoes, which is what they are.
    """
    if not embedded:
        return question

    def swap(match: re.Match) -> str:
        return embedded.get(match.group(1), match.group(0))

    raw = _PIPE_FIELD.sub(swap, question.raw_text)
    if raw == question.raw_text:
        return question
    return replace(question, raw_text=raw, text=strip_html(raw))


def display_condition(payload: dict, survey: Survey, prefix: str = "") -> str | None:
    """The question's own display logic, stated in words, or ``None``.

    Only the one shape this instrument uses is read: "show this page if choice N
    of an earlier question was selected", which is how the two feedback arms give
    a respondent a "Wrong!" or a "You are right!" screen.  Anything else returns
    ``None`` and the element is rendered unconditionally, which is the safe
    failure: an extra screen in a template is visible, a missing one is not.
    """
    logic = payload.get("DisplayLogic")
    if not isinstance(logic, dict):
        return None
    expression = ((logic.get("0") or {}).get("0")) or {}
    if expression.get("LogicType") != "Question":
        return None
    source = survey.questions.get(str(expression.get("QuestionID") or ""))
    locator = str(expression.get("ChoiceLocator") or "")
    match = re.search(r"/SelectableChoice/(\d+)", locator)
    if source is None or match is None:
        return None
    keys = list((source.choices or ()))
    index = int(match.group(1)) - 1
    answer = keys[index] if 0 <= index < len(keys) else f"choice {match.group(1)}"
    return f'{prefix}{export_column(source)} = "{answer}"'


def _slider_slots(question: Question, payload: dict, prompt: str) -> list[Slot]:
    lo, hi = slider_bounds(payload)
    config = payload.get("Configuration") or {}
    not_applicable = str(config.get("NotApplicableText") or "").strip()
    # Qualtrics files the "Not Applicable" escape among the scale labels, so
    # anchors_from_labels picks it up and the endpoint line reads "Definitely not
    # ... Definitely yes ... Not Applicable". It is a separate control, not a
    # scale point, so it comes out of the anchors and is stated once, its own way.
    anchors = " … ".join(
        part
        for part in anchors_from_labels(payload).split(" … ")
        if part and part != not_applicable
    )
    rows = _rows(payload)
    slots: list[Slot] = []
    for key, label in rows:
        described = anchors or f"Whole number from {lo} to {hi}."
        if not_applicable:
            # The source text for two of these items has an unbalanced quote of
            # its own; quoting it again would compound the mess, so it is
            # normalised to a bare parenthetical.
            described += "  Or answer: Not Applicable."
        slots.append(
            IntSlot(
                id=published_column(export_column(question, key)),
                prompt=prompt if len(rows) == 1 else f"{prompt} — {label}",
                anchors=described,
                lo=lo,
                hi=hi,
                max_tokens=6,
            )
        )
    return slots


def _rows(payload: dict) -> list[tuple[str, str]]:
    """``(key, display)`` for every row of a multi-row question, in shown order."""
    choices = payload.get("Choices")
    if not isinstance(choices, dict):
        return [("", "")]
    order = payload.get("ChoiceOrder") or list(choices)
    keys = [str(k) for k in order if str(k) in {str(x) for x in choices}]
    out = []
    for key in keys:
        label = strip_html(str(choices[key].get("Display", "")))
        out.append((key, "" if label in {"", "\xa0"} else label))
    return out or [("", "")]


#: The only numeric free-text item in the instrument is age, and the survey
#: validates it as a number without stating bounds; the authors' cleaning script
#: keeps 18 to 100 and discards the rest, so those are the legal answers.
NUMERIC_TEXT_RANGE = (18, 100)


def _numeric_range(payload: dict) -> tuple[int, int] | None:
    """Bounds for a text entry the survey validates as a number, if it is one."""
    settings = (payload.get("Validation") or {}).get("Settings") or {}
    if settings.get("ContentType") != "ValidNumber":
        return None
    stated = settings.get("ValidNumber") or {}
    lo = stated.get("Min")
    hi = stated.get("Max")
    return (
        int(lo) if str(lo).strip() not in {"", "None"} else NUMERIC_TEXT_RANGE[0],
        int(hi) if str(hi).strip() not in {"", "None"} else NUMERIC_TEXT_RANGE[1],
    )


def convert_question(
    question: Question,
    payload: dict,
    survey: Survey,
    embedded: dict | None = None,
    qid_to_slot: dict | None = None,
) -> list:
    """One question -> zero or more transcript elements, in display order."""
    if question.kind in {"Timing", "Meta"}:
        return []
    question = resolve_embedded(question, embedded or {})
    prompt = resolve_pipes(question.text, qid_to_slot or {})
    notes = media_notes(question, payload)

    if question.kind in DISPLAY_ONLY:
        head = [Text(prompt)] if prompt.strip() else []
        return head + notes

    if question.kind == "Slider":
        return notes + list(_slider_slots(question, payload, prompt))

    if question.kind == "CS":
        # Constant sum: the donation. Each box is its own integer, and the survey
        # refuses a submission whose boxes do not total the required amount.
        lo, hi = slider_bounds(payload)
        total = ((payload.get("Validation") or {}).get("Settings") or {}).get(
            "ChoiceTotal"
        )
        rows = _rows(payload)
        tail = f" The two amounts must total {total}." if total else ""
        return notes + [
            IntSlot(
                id=published_column(export_column(question, str(position))),
                prompt=f"{prompt} — {label}" if label else prompt,
                anchors=f"Whole number of dollars from {lo} to {hi}.{tail}",
                lo=lo,
                hi=hi,
                allow_dollar=True,
                max_tokens=6,
            )
            for position, (_, label) in enumerate(rows, start=1)
        ]

    if question.kind == "Matrix":
        # Rows are the statements, the shared answer scale is the option list.
        options = tuple(s for s in question.statements if s)
        if not options:
            return notes + ([Text(prompt)] if prompt.strip() else [])
        return (
            notes
            + [Text(prompt)]
            + [
                ChoiceSlot(
                    id=published_column(export_column(question, key)),
                    prompt=label,
                    options=options,
                    max_tokens=max(6, max(len(o.split()) for o in options) * 3 + 4),
                )
                for key, label in _rows(payload)
            ]
        )

    if question.kind == "MC":
        options = tuple(o for o in question.choices if o)
        if not options:
            return notes + ([Text(prompt)] if prompt.strip() else [])
        multi = question.selector.startswith("MA")
        return notes + [
            ChoiceSlot(
                id=published_column(export_column(question)),
                prompt=prompt,
                options=options,
                codes=question.codes,
                max_tokens=max(6, max(len(o.split()) for o in options) * 3 + 4),
                describe_as=("Select all that apply: " if multi else "Options: ")
                + " | ".join(options),
            )
        ]

    if question.kind == "TE":
        numeric = _numeric_range(payload)
        if numeric is not None and question.selector != "FORM":
            lo, hi = numeric
            return notes + [
                IntSlot(
                    id=published_column(export_column(question)),
                    prompt=prompt,
                    anchors=f"Whole number from {lo} to {hi}.",
                    lo=lo,
                    hi=hi,
                    max_tokens=6,
                )
            ]
        rows = _rows(payload) if question.selector == "FORM" else [("", "")]
        if len(rows) == 1 and not rows[0][1]:
            return notes + [
                FreeTextSlot(
                    id=published_column(export_column(question)),
                    prompt=prompt,
                    hint="Free text.",
                    max_tokens=120,
                    max_chars=1200,
                )
            ]
        return (
            notes
            + ([Text(prompt)] if prompt.strip() else [])
            + [
                FreeTextSlot(
                    id=published_column(export_column(question, key)),
                    prompt=label,
                    hint="Free text.",
                    max_tokens=80,
                    max_chars=600,
                )
                for key, label in rows
            ]
        )

    return notes + ([Text(prompt)] if prompt.strip() else [])


def convert_run(
    survey: Survey,
    payloads: dict,
    items,
    page_from_timers: bool = False,
    prefix: str = "",
    embedded: dict | None = None,
    qid_to_slot: dict | None = None,
) -> Converted:
    """A run of block elements, in builder order, as transcript elements.

    ``items`` is the block's own ``BlockElements`` sequence — ``("Question", qid)``
    and ``("Page Break", None)`` pairs — because where the screens break is a
    property the survey states outright for every live block, and guessing at it
    would put questions on the wrong page in a document whose whole purpose is to
    be what the respondent saw.

    ``page_from_timers`` is for the outcome battery only.  Its blocks survive
    solely inside the master export's trash block, which Qualtrics flattens: 630
    questions, no page breaks.  What the trash block does keep is the battery's
    page timers, and a page timer belongs to exactly one page, so a timer marks
    the end of a screen.  That is the available evidence, and it is weaker than a
    stated page break, which is why the two rules are kept visibly apart.

    ``prefix`` namespaces every slot id this run produces.  The eighteen arm
    exports were built separately and reuse each other's export tags — three
    different arms call a writing prompt ``Q5``, and so does the consent item,
    which *is* a column in the published file — so an arm's items are pushed
    behind the arm's own name.  Nothing an intervention asks is scored, because
    those answers exist only in the unpublished raw export, so the rename costs
    nothing and stops an arm from overwriting a real outcome.

    A question carrying display logic has *all* of its elements wrapped in a
    conditional, not just its response positions.  The two feedback arms react to
    a yes/no answer with a whole screen headed "Wrong!" or "You are right!", and
    that screen has no response position on it at all: gating only the slots
    would show every respondent both.
    """
    elements: list = []
    columns: dict = {}
    gates: dict = {}
    pending_break = True
    for kind, qid in items:
        if kind == "Page Break":
            pending_break = True
            continue
        question = survey.questions.get(qid)
        if question is None:
            continue
        if page_from_timers and question.kind == "Timing":
            pending_break = True
            continue
        payload = payloads.get(qid, {})
        produced = convert_question(
            question, payload, survey, embedded=embedded, qid_to_slot=qid_to_slot
        )
        if not produced:
            continue
        if prefix:
            produced = [
                (
                    replace(element, id=f"{prefix}{element.id}")
                    if isinstance(element, Slot)
                    else element
                )
                for element in produced
            ]
        for element in produced:
            if isinstance(element, Slot):
                columns[element.id] = element.id
        gate = display_condition(payload, survey, prefix=prefix)
        if pending_break:
            elements.append(PageBreak())
            pending_break = False
        if gate is None:
            elements.extend(produced)
            continue
        column, _, wanted = gate.partition(" = ")
        target = wanted.strip('"')
        for element in produced:
            if isinstance(element, Slot):
                gates[element.id] = gate
        elements.append(
            Conditional(
                note=gate,
                predicate=(
                    lambda col, value: lambda answers: answers.get(col) == value
                )(column, target),
                elements=produced,
            )
        )
    return Converted(elements=elements, data_columns=columns, display_logic=gates)


def block_items(payload: dict, block_id: str) -> list[tuple[str, str]]:
    """One block's ``BlockElements``, keeping the page breaks the reader drops.

    :class:`~silicon_sampling.voelkel.qsf.Block` filters to questions, which is
    all the Voelkel instrument needed.  Here the page breaks are the page
    structure, so they have to survive.
    """
    entries = payload.values() if isinstance(payload, dict) else payload
    for entry in entries:
        if entry.get("ID") != block_id:
            continue
        return [
            (
                str(element.get("Type")),
                str(element.get("QuestionID") or ""),
            )
            for element in entry.get("BlockElements", [])
            if element.get("Type") in {"Question", "Page Break"}
        ]
    return []
