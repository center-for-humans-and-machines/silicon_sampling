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

**Media becomes a bracketed description, not a bracketed silence.**  Eleven of the
eighteen arms survive the modality audit and nine of those eleven display something
that is not text.  The first version of this converter emitted a content-free note
— "[ image shown here — not reproduced ]" — on the honest ground that nothing in
the pipeline had looked at a picture.  That was the right note for the wrong
situation: two whole screens of ``ThreatInjustEfficacy`` have no content but
photographs, so they rendered as nothing but brackets, and ``BindingMorals`` asks
how impure the Great Smoky Mountains look "in the picture on the right above" of a
screen that had no picture on it.  So the pictures were fetched from the hosts the
exports hot-link and looked at, and :mod:`~silicon_sampling.goldwert.images` holds
what each one shows.  Six files are gone from every host that ever served them, and
those say exactly that instead of being invented.

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
from . import images

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
    """A bracketed line for whatever was on the page besides words.

    The line carries the picture's *content* whenever
    :mod:`~silicon_sampling.goldwert.images` has a description written from the file
    — which is every picture the eleven kept arms display.  It used to be
    deliberately content-free, on the ground that nothing in the pipeline had looked
    at a picture; that was true and it was also the reason two whole screens of
    ``ThreatInjustEfficacy`` rendered as nothing but brackets and
    ``BindingMorals`` asked how impure the mountains looked "in the picture on the
    right above" with no picture anywhere.  So the pictures were fetched and looked
    at, and what is left content-free is only what could not be recovered, which
    says so in as many words.

    A live third-party panel keeps its old treatment, because there is nothing to
    describe: the petition, the two newsletter forms and the bank lookup are pages
    the respondent acted *inside*.
    """
    notes: list[Text] = []
    for found in _MEDIA_TAG.finditer(question.raw_text):
        # A <video> keeps its address in a nested <source>, which the tag match
        # stops short of, so the note is resolved against a window of the raw text
        # rather than against the opening tag alone.
        window = question.raw_text[found.start() : found.start() + 800]
        notes.append(
            Text(f"[ {_describe_media(found.group(0), window)} ]", style="cite")
        )
    graphic = str(payload.get("Graphics") or "")
    if graphic:
        described = images.describe(graphic)
        if described is None:
            described = str(payload.get("GraphicsDescription") or "").strip()
            detail = f": {described}" if described else ""
            notes.append(
                Text(f"[ image shown here{detail} — not described ]", style="cite")
            )
        else:
            notes.append(Text(f"[ image shown here: {described} ]", style="cite"))
    return notes


def describe_media_html(html: str) -> str:
    """Bracketed note(s) for a snippet of HTML that is nothing but media.

    Exists for the fields ``MispCorrectionRisks`` pipes onto a page at runtime: the
    survey stores the chart for each of its six topics as an ``<img>`` tag in an
    embedded value, so the thing that has to be described is a string rather than a
    question.  Same vocabulary as :func:`media_notes`, deliberately.
    """
    found = [
        f"[ {_describe_media(match.group(0), html[match.start():match.start() + 800])} ]"
        for match in _MEDIA_TAG.finditer(html)
    ]
    return "\n".join(found)


def _asset_key(tag: str, href: str) -> str:
    """The string this table is keyed by: a Qualtrics asset id, else the whole URL.

    Qualtrics serves the same picture from nine different brand hosts across the
    eighteen exports, so keying on the URL alone would need the host to be right as
    well as the asset; keying on ``IM=``/``F=``/the YouTube id is keying on the thing
    itself.  Everything hot-linked from outside Qualtrics has no such id, and there
    the URL *is* the identity.
    """
    for pattern in (r"[?&]IM=([A-Za-z0-9_]+)", r"[?&]F=([A-Za-z0-9_]+)"):
        found = re.search(pattern, href)
        if found:
            return found.group(1)
    found = _YOUTUBE.search(href)
    if found:
        return found.group(1)
    return href


def _describe_media(tag: str, window: str = "") -> str:
    """What one media tag showed, from the file, or a loud note that it is gone."""
    name = re.match(r"<(\w+)", tag).group(1).lower()
    src = _SRC.search(tag) or (_SRC.search(window) if window else None)
    href = src.group(1) if src else ""
    if name == "iframe" and not _VIDEO_HOST.search(href):
        host = re.sub(r"^https?://(www\.)?", "", href).split("/")[0]
        where = f" from {host}" if host else ""
        return (
            f"live web panel embedded here{where}, which the participant acted inside"
        )
    key = _asset_key(tag, href)
    if name in {"video", "audio"} or name == "iframe":
        described = images.MEDIA_ALT.get(key)
        kind = "audio clip played here" if name == "audio" else "video shown here"
        if described:
            return f"{kind}: {' '.join(described.split())}"
        where = f" ({key})" if key and key != href else ""
        return f"{kind}{where} — not reproduced"
    described = images.describe(key)
    if described is not None:
        return f"image shown here: {described}"
    alt = _ALT.search(tag)
    detail = f": {strip_html(alt.group(1))}" if alt else ""
    return f"image shown here{detail} — not described"


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


#: Minus signs a respondent might type.  Qualtrics renders the scale with an
#: ASCII hyphen, but a model writing prose reaches for the typographic ones.
_MINUS = ("-", "\u2212", "\u2013", "\u2014")

#: The value recorded when a respondent takes a slider's opt-out instead of
#: answering on the scale.  A short string on purpose: ``pd.to_numeric(...,
#: errors="coerce")`` turns it into ``NaN``, which is exactly what the escape
#: became in the published file, so the sampled column arrives in the analysis
#: frame already shaped like the human one and needs no special case anywhere
#: downstream.  It is never what the transcript shows — see
#: :meth:`EscapableIntSlot.render`.
NOT_APPLICABLE = "N/A"

#: Spellings of the opt-out that are accepted on top of the item's own wording.
#: A model that has read a forty-six-character parenthetical escape will often
#: type its first two words and stop, and refusing that would recreate the very
#: failure this slot exists to fix.
_ESCAPE_ALIASES = ("Not Applicable", "Not applicable", "N/A", "NA")

#: Punctuation allowed to trail the opt-out before the draw stops being the
#: opt-out.  One character wider than the shared slot's set, and deliberately:
#: the ``flyless`` wording ends mid-parenthesis in the survey's own text, so a
#: model that closes the bracket the survey never closed is agreeing with the
#: option rather than hedging between two.
_ESCAPE_TRAILING = ".,;:()!?\u2019'\"\u2014- \t"


def escape_text(payload: dict) -> str:
    """The wording of a slider's opt-out checkbox, or ``""`` if it has none.

    Read from three places in that order because Qualtrics writes it in three
    places and no single one of them is reliable.  ``Configuration.NotApplicable``
    is the flag that says the checkbox was *shown*; ``NotApplicableText`` usually
    carries the wording but is absent on ``CC_policy``, which files it under the
    ``"NA"`` key of ``Labels`` instead; and when both are missing Qualtrics shows
    its own default.  Keying on the flag rather than on the presence of the text
    is what stops a shown escape from going unnoticed, and reading the ``"NA"``
    label is what stops it from leaking into the endpoint-label line as if it were
    a scale point.
    """
    config = payload.get("Configuration") or {}
    if not config.get("NotApplicable"):
        return ""
    stated = str(config.get("NotApplicableText") or "").strip()
    if stated:
        return stated
    labels = payload.get("Labels")
    if isinstance(labels, dict):
        entry = labels.get("NA")
        if isinstance(entry, dict):
            labelled = strip_html(str(entry.get("Display", ""))).strip()
            if labelled:
                return labelled
    return "Not Applicable"


def _sentence(text: str) -> str:
    """``text`` with a full stop, unless it already ends in punctuation."""
    return text if text[-1:] in ".!?" else f"{text}."


@dataclass(frozen=True)
class EscapableIntSlot(IntSlot):
    """A slider that also accepts the opt-out checkbox printed beside it.

    Five live sliders in this instrument carry a Qualtrics "Not Applicable"
    control, and until this class existed the transcript printed it and the slot
    refused it.  That combination is worse than either half alone.  A model that
    followed the printed instruction produced a draw no ``IntSlot`` could parse,
    so the sampler spent its four rounds and its constrained-decoding fallback on
    it and then recorded ``(lo + hi) // 2`` — a forced 50 on a 0-100 scale,
    written into the column as if the respondent had chosen the exact midpoint.

    The escape is not a nuisance option, it is where this study's missingness
    comes from.  ``flyless`` has 14,069 usable values out of 31,324 against
    23,706 for ``conversation`` on the same screen, so roughly 9,637 people —
    41% of everyone who reached the page — ticked "I already don't fly" rather
    than answering; ``lessbeef`` about 2,529 and ``pol_candidate`` about 1,431.
    ``flylessN`` and ``lessbeefN`` *are* ``lifestyle_changes``, both of its
    members, and ``pol_candidateN`` is one of four in ``political_advocacy``, so a
    sample where nobody can opt out is not measuring the same composite: it fills
    the arm mean with numbers from people the study recorded as having no answer
    to give.  Every existing run shows 100.0% non-null on all three, which is the
    signature of an escape nobody could take.

    So the value recorded is :data:`NOT_APPLICABLE`, a non-numeric string that
    ``outcomes.compute`` coerces to ``NaN`` exactly where the published file has
    ``NaN``, and the transcript shows the item's own wording rather than the
    sentinel.  Two residual gaps are worth naming.  The constrained-decoding
    fallback in :mod:`silicon_sampling.sampling.driver` builds its grammar from
    ``range(lo, hi + 1)`` for anything that is an ``IntSlot``, so on the rare path
    where four rounds of free draws all fail the model is handed a number-only
    grammar and cannot opt out; and the forced default is still the midpoint.
    Both are last-resort paths that this slot makes far harder to reach rather
    than closing, and both live in a shared module three other studies use.
    """

    #: The opt-out's on-screen wording, verbatim from the survey.
    escape: str = ""

    @property
    def legal_spec(self) -> str:
        base = f"{self.lo}..{self.hi}"
        return f"{base} | {NOT_APPLICABLE}" if self.escape else base

    def stated_range(self) -> str:
        """The sentence that says what numbers are legal."""
        return f"Whole number from {self.lo} to {self.hi}."

    def describe(self) -> str:
        parts = [self.anchors] if self.anchors else []
        parts.append(self.stated_range())
        if self.escape:
            parts.append(f"Or answer: {_sentence(self.escape)}")
        return "  ".join(parts)

    def escape_value(self, raw: str):
        """:data:`NOT_APPLICABLE` if ``raw`` takes the opt-out, else ``None``."""
        if not self.escape:
            return None
        text = raw.split("\n", 1)[0].strip()
        if not text:
            return None
        low = text.lower()
        for spelling in sorted((self.escape, *_ESCAPE_ALIASES), key=len, reverse=True):
            if low.startswith(spelling.lower()):
                rest = text[len(spelling) :]
                if rest.strip(_ESCAPE_TRAILING) == "":
                    return NOT_APPLICABLE
        return None

    def parse(self, raw: str):
        taken = self.escape_value(raw)
        return taken if taken is not None else super().parse(raw)

    def render(self, value) -> str:
        return self.escape if value == NOT_APPLICABLE else super().render(value)


@dataclass(frozen=True)
class SignedIntSlot(EscapableIntSlot):
    """A slider whose scale runs through zero, so half its range is negative.

    The shared :class:`~silicon_sampling.survey.slots.IntSlot` matches an answer
    with ``\\d+`` and therefore cannot parse ``"-40"`` at all.  On a 0-100 slider
    that is correct — a minus sign there is not an answer.  On the one bipolar
    item in this instrument it silently made the entire negative half of the scale
    illegal: every draw below the midpoint was rejected, the sampler burned its
    four rounds and its grammar fallback on each one, and the forced default
    landed on the exact midpoint.  A respondent who felt negative about the
    stories could not say so.

    Fixed here rather than in the shared slot because the shared module serves
    three other studies and none of them has a scale like this; the sign handling
    is this instrument's problem, so it lives with this instrument's converter.
    The prose line has to change too: the project's slider convention states the
    endpoint *labels* and no numbers, which leaves a model with no way to know a
    minus sign is permitted, so this slot says the range out loud — once.  It used
    to say it twice, because the caller pre-baked ``state_range`` into ``anchors``
    and then :meth:`describe` appended its own sentence to it, which is why
    :meth:`EscapableIntSlot.describe` now composes the whole line from parts and
    the only thing a caller passes is the endpoint labels.
    """

    def stated_range(self) -> str:
        return (
            f"Whole number from {self.lo} to {self.hi}; negative answers are allowed."
        )

    def parse(self, raw: str):
        taken = self.escape_value(raw)
        if taken is not None:
            return taken
        text = raw.split("\n", 1)[0].strip()
        sign = 1
        for mark in _MINUS:
            if text.startswith(mark):
                sign, text = -1, text[len(mark) :].lstrip()
                break
        magnitude = IntSlot.parse(self, text)
        if magnitude is None:
            return None
        value = sign * magnitude
        return value if self.lo <= value <= self.hi else None


def slider_anchors(payload: dict) -> str:
    """The endpoint labels a slider showed, with the opt-out taken back out.

    Qualtrics files the opt-out's wording among the scale ``Labels``, under the
    key ``"NA"``, so reading the labels in order gives "Definitely not …
    Definitely yes … Not Applicable / Not Eligible to Vote" and presents a
    separate checkbox as a third point on the scale.  Dropping it by *key* rather
    than by comparing it against ``NotApplicableText`` is what makes this hold on
    ``CC_policy``, whose escape is in the labels and nowhere else.
    """
    labels = payload.get("Labels")
    if not isinstance(labels, dict):
        return anchors_from_labels(payload)
    scale = {key: value for key, value in labels.items() if str(key) != "NA"}
    return anchors_from_labels({**payload, "Labels": scale})


def _slider_slots(question: Question, payload: dict, prompt: str) -> list[Slot]:
    """One integer slot per bar, with the range stated and the opt-out honoured.

    ``anchors`` here is the endpoint labels and nothing else: the range sentence
    and the opt-out sentence are composed by
    :meth:`EscapableIntSlot.describe`, because an earlier version baked the range
    into ``anchors`` and the bipolar slider's own ``describe`` then appended a
    second copy of it.  The opt-out is printed in the survey's own words, which on
    two items includes an unbalanced quotation mark the survey itself left open;
    reproducing it is closer to what the respondent read than paraphrasing it
    away, and the earlier paraphrase — a flat "Not Applicable." — had thrown out
    the part that told a respondent what the escape was *for*.
    """
    lo, hi = slider_bounds(payload)
    escape = escape_text(payload)
    anchors = slider_anchors(payload)
    rows = _rows(payload)
    kind = SignedIntSlot if lo < 0 else EscapableIntSlot
    return [
        kind(
            id=published_column(export_column(question, key)),
            prompt=prompt if len(rows) == 1 else f"{prompt} — {label}",
            anchors=anchors,
            lo=lo,
            hi=hi,
            escape=escape,
            max_tokens=max(6, len(escape.split()) * 3 + 4) if escape else 6,
        )
        for key, label in rows
    ]


#: A tag whose closing ``>`` the survey's author never typed.  Anchored to the end
#: of the string and required to start at a real ``<``, so it cannot bite text
#: that merely contains a less-than sign.
_UNCLOSED_TAG = re.compile(r"</?[A-Za-z][^<>]*$")


def clean_label(text: object) -> str:
    """``strip_html``, then the one tag ``strip_html`` structurally cannot remove.

    The shared stripper matches ``<[^>]+>`` — a tag with both of its angle
    brackets — which is the right rule and misses this instrument's one broken
    label.  The emotion battery's third row is stored as
    ``<span …>Inspired</span></span`` with the final bracket missing, so the
    stripper took the two well-formed tags, left the third, and ``Inspired</span``
    went into the transcript of every arm that shows the battery as the text of a
    live matrix row.  Fixed here rather than in the shared stripper because the
    shared one serves four studies and a rule that deletes trailing ``<…`` is a
    rule that can eat real text; scoped to this instrument's labels, it cannot.
    """
    return _UNCLOSED_TAG.sub("", strip_html(str(text))).strip()


def one_line(text: str) -> str:
    """Collapse a choice label onto a single line.

    Not cosmetic.  A response line in this transcript format holds the answer and
    nothing else, so :class:`~silicon_sampling.survey.slots.ChoiceSlot` truncates
    a draw at the first newline before matching it — which means an option whose
    own label *contains* a newline can never be matched, and rejection sampling
    then rejects that option one hundred per cent of the time.  Seven options of
    ``ThreatInjustEfficacy``'s fairness item arrived that way, each stored in
    Qualtrics as ``"Completely unfair <br> 1"``, and the whole seven-point scale
    was unanswerable: the sampler would have exhausted its rounds on every one of
    them and recorded the forced default instead.  The template file shows the
    damage plainly — its ``Options:`` line runs down eight rows — which is why this
    is fixed where the label is built rather than papered over at parse time.
    """
    return " ".join(clean_label(text).split())


def _rows(payload: dict) -> list[tuple[str, str]]:
    """``(key, display)`` for every row of a multi-row question, in shown order."""
    choices = payload.get("Choices")
    if not isinstance(choices, dict):
        return [("", "")]
    order = payload.get("ChoiceOrder") or list(choices)
    keys = [str(k) for k in order if str(k) in {str(x) for x in choices}]
    out = []
    for key in keys:
        label = clean_label(choices[key].get("Display", ""))
        out.append((key, "" if label in {"", "\xa0"} else label))
    return out or [("", "")]


#: The only numeric free-text item in the instrument is age, and the survey
#: validates it as a number without stating bounds; the authors' cleaning script
#: keeps 18 to 100 and discards the rest, so those are the legal answers.
NUMERIC_TEXT_RANGE = (18, 100)


def custom_validation_bounds(payload: dict) -> tuple[int, int] | None:
    """Numeric bounds a question states through Qualtrics ``CustomValidation``.

    Read rather than assumed.  Only a question that declares *both* a
    ``GreaterThanOrEqual`` and a ``LessThanOrEqual`` clause gets bounds out of
    here, so a question the survey did not bound stays free text instead of being
    silently given a range nobody wrote down.
    """
    logic = ((payload.get("Validation") or {}).get("Settings") or {}).get(
        "CustomValidation"
    )
    if not isinstance(logic, dict):
        return None
    found: dict[str, str] = {}

    def walk(node) -> None:
        if isinstance(node, dict):
            operator, right = node.get("Operator"), node.get("RightOperand")
            if operator in {"GreaterThanOrEqual", "LessThanOrEqual"} and right not in (
                None,
                "",
            ):
                found[operator] = str(right)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(logic)
    try:
        return int(found["GreaterThanOrEqual"]), int(found["LessThanOrEqual"])
    except (KeyError, ValueError):
        return None


def row_numeric_range(payload: dict, row_key: str) -> tuple[int, int] | None:
    """Bounds for one row of a text-entry *form* that asks for a number.

    A single-box text entry declares its validation once, in
    ``Validation.Settings.ContentType``, which is what :func:`_numeric_range`
    reads.  A *form* declares it per row, in
    ``Choices[key]["TextEntryValidation"]`` — and missing that turned the three
    "What percentage of Americans …?" boxes of ``IndStructuralChange`` into
    free-text slots.  The consequence was not cosmetic: each of those three
    answers is echoed back on the *next* screen as "You guessed <answer>% of
    people believe …", so a prose answer produced a sentence reading "You guessed
    I want you to take climate change seriously and vote for policies…% of people
    believe…".  A template render cannot show this, because a template prints the
    echo marker instead of resolving it; only driving a session does.
    """
    choices = payload.get("Choices")
    if not isinstance(choices, dict):
        return None
    row = choices.get(str(row_key)) or {}
    if not str(row.get("TextEntryValidation") or "").startswith("ValidNumber"):
        return None
    return custom_validation_bounds(payload)


def _form_row_slot(question: Question, payload: dict, key: str, label: str) -> Slot:
    """One row of a text-entry form: a number if the survey validated it as one."""
    bounds = row_numeric_range(payload, key)
    if bounds is None:
        return FreeTextSlot(
            id=published_column(export_column(question, key)),
            prompt=label,
            hint="Free text.",
            max_tokens=80,
            max_chars=600,
        )
    lo, hi = bounds
    return IntSlot(
        id=published_column(export_column(question, key)),
        prompt=label,
        anchors=f"Whole number from {lo} to {hi}.",
        lo=lo,
        hi=hi,
        max_tokens=6,
    )


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


#: Prompts that point at another field on the *same page*, and the note that says
#: which. Serialising a Qualtrics page destroys simultaneity: a participant saw
#: every box on the page at once, so "include your zipcode below" plainly meant the
#: separate box further down. Read one question at a time, the same sentence is an
#: instruction about the box being answered — and the models took it that way. The
#: letter box came back holding a ZIP code as its median answer: median length 5
#: characters, and 79-84% of answers under 40, with V4-Flash's first five answers
#: being 08512, 90001, 11804, 88011, 97070. `letter` is a member of the
#: `political_advocacy` composite, so this was scored.
#:
#: The fix is to restore the missing context rather than to instruct the model: the
#: note describes the page the participant saw, and nothing more.
#: Notes restoring a same-page reference that serialising the page destroyed.
#:
#: A Qualtrics page shows its questions at once; a transcript shows them in
#: sequence, and a prompt that says "below" then points at nothing.  The letter
#: box is the case that bit: its prompt ends "please include your zipcode below
#: and we will look up the name for you", and the zipcode box is the *next*
#: question.  Read in sequence that is an instruction to type a zipcode, and the
#: models obeyed it -- median answer length 5 characters, and 46-59% of answers a
#: bare zipcode, against a question asking what the respondent would say to their
#: representative about climate change.
#:
#: The note alone did not fix it: the stimulus paragraph's own last sentence still
#: ends on the zipcode, and it is the last thing read before answering.  So the
#: **slot annotation** says what the box collects.  That annotation is our own
#: scaffolding rather than anything a participant saw, so putting the box's
#: purpose there changes no stimulus text.
SAME_PAGE_NOTES = {
    "letter_content": (
        "(The zipcode box referred to is a separate question on this same page, "
        "shown below this one.)"
    ),
}

#: What a free-text box collects, where its prompt alone is misleading in prose.
SLOT_PURPOSE = {
    "letter_content": "the message itself, not a zipcode",
}


def same_page_note(slot_id: str) -> str:
    """Any note restoring a same-page reference this slot's prompt makes."""
    return SAME_PAGE_NOTES.get(slot_id, "")


def slot_purpose(slot_id: str) -> str:
    """What this box collects, for the slot annotation, or an empty string."""
    return SLOT_PURPOSE.get(slot_id, "")


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
        options = tuple(one_line(s) for s in question.statements if s)
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
        options = tuple(one_line(o) for o in question.choices if o)
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
            slot_id = published_column(export_column(question))
            note = same_page_note(slot_id)
            return notes + [
                FreeTextSlot(
                    id=slot_id,
                    prompt=f"{prompt}\n{note}".rstrip() if note else prompt,
                    hint="Free text.",
                    max_tokens=120,
                    max_chars=1200,
                    purpose=slot_purpose(slot_id),
                )
            ]
        return (
            notes
            + ([Text(prompt)] if prompt.strip() else [])
            + [_form_row_slot(question, payload, key, label) for key, label in rows]
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
