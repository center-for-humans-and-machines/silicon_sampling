"""Read a Qualtrics ``.qsf`` export into something a transcript can be built from.

A ``.qsf`` is a JSON dump of the survey builder's internal state: a flat list of
``SurveyElements``, of which one holds every block, one holds the survey flow, and
the rest are individual questions.  Nothing in it is in display order, and the
flow is a tree of branches, randomisers and embedded-data assignments that has to
be interpreted to find out what any given respondent actually saw.

This module does that interpretation and nothing else — no study-specific
knowledge lives here beyond the name of the embedded field carrying the
experimental condition.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

#: Tags whose *presence* means the stimulus is not purely textual.
MEDIA_PATTERNS = {
    "video": re.compile(r"<video|\.mp4|youtube|vimeo|wistia|\.mov\b", re.I),
    "audio": re.compile(r"<audio|\.mp3|\.wav\b|soundcloud", re.I),
    "image": re.compile(r"<img\b", re.I),
    "iframe": re.compile(r"<iframe", re.I),
    "script": re.compile(r"<script", re.I),
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")


def strip_html(raw: str) -> str:
    """Visible text of a Qualtrics rich-text field."""
    text = re.sub(r"<br\s*/?>", "\n", raw or "", flags=re.I)
    text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", text, flags=re.I)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


@dataclass(frozen=True)
class Question:
    """One Qualtrics question, as displayed."""

    qid: str
    kind: str
    selector: str
    text: str
    raw_text: str
    #: Display text of each choice, in the order Qualtrics stores them.
    choices: tuple[str, ...] = ()
    #: Recode values parallel to ``choices``, where the survey defines them.
    codes: tuple[str, ...] = ()
    #: Matrix row statements, if this is a matrix question.
    statements: tuple[str, ...] = ()
    #: Exported variable name(s), where the survey names them.
    export_tag: str = ""

    @property
    def is_display_only(self) -> bool:
        """Descriptive text and page timers record nothing."""
        return self.kind in {"DB", "Timing", "Meta"}

    def media(self) -> dict[str, int]:
        return {
            name: len(pattern.findall(self.raw_text))
            for name, pattern in MEDIA_PATTERNS.items()
        }


@dataclass(frozen=True)
class Block:
    """A run of questions Qualtrics shows together, in builder order."""

    bid: str
    description: str
    question_ids: tuple[str, ...]
    #: ``True`` where the block inserts a page break after every question.
    page_break_after_each: bool = False


@dataclass
class Survey:
    """A parsed ``.qsf``."""

    blocks: dict[str, Block]
    questions: dict[str, Question]
    flow: dict
    condition_field: str = "Condition"
    _order: list[str] = field(default_factory=list)

    # -- construction ----------------------------------------------------- #

    @classmethod
    def load(cls, path: Path, condition_field: str = "Condition") -> "Survey":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        elements = payload["SurveyElements"]

        questions: dict[str, Question] = {}
        for element in elements:
            if element["Element"] != "SQ":
                continue
            question = _parse_question(element)
            questions[question.qid] = question

        blocks: dict[str, Block] = {}
        block_payload = next(e for e in elements if e["Element"] == "BL")["Payload"]
        for entry in (
            block_payload.values() if isinstance(block_payload, dict) else block_payload
        ):
            options = entry.get("Options") or {}
            blocks[entry["ID"]] = Block(
                bid=entry["ID"],
                description=entry.get("Description", ""),
                question_ids=tuple(
                    el["QuestionID"]
                    for el in entry.get("BlockElements", [])
                    if el.get("Type") == "Question"
                ),
                page_break_after_each=str(options.get("BlockLocking", "")).lower()
                == "true"
                or options.get("QuestionsPerPage") == "1",
            )

        flow = next(e for e in elements if e["Element"] == "FL")["Payload"]
        return cls(
            blocks=blocks,
            questions=questions,
            flow=flow,
            condition_field=condition_field,
        )

    # -- flow ------------------------------------------------------------- #

    def conditions(self) -> tuple[str, ...]:
        """Every value the condition field is randomised to, in flow order."""
        if self._order:
            return tuple(self._order)
        seen: list[str] = []
        for node in _walk(self.flow):
            if node.get("Type") != "BlockRandomizer":
                continue
            for child in node.get("Flow", []) or []:
                for entry in child.get("EmbeddedData", []) or []:
                    if entry.get("Field") == self.condition_field:
                        value = entry.get("Value")
                        if value and value not in seen:
                            seen.append(value)
        self._order = seen
        return tuple(seen)

    def blocks_for(self, condition: str) -> list[Block]:
        """Blocks gated on this condition, in flow order.

        A branch is taken when its logic mentions the condition field with this
        value; branches that test anything else (attention checks, screen-outs,
        party) are traversed, so their blocks are collected too and filtered
        later by whoever knows what party the respondent is.
        """
        found: list[Block] = []
        for node, gates in _walk_gated(self.flow, self.condition_field):
            if node.get("Type") not in {"Standard", "Block"}:
                continue
            if gates and condition not in gates:
                continue
            block = self.blocks.get(node.get("ID"))
            if block is not None and block not in found:
                found.append(block)
        return found

    def condition_only_blocks(self, condition: str) -> list[Block]:
        """Blocks shown *only* under this condition — the stimulus itself."""
        exclusive: list[Block] = []
        for node, gates in _walk_gated(self.flow, self.condition_field):
            if node.get("Type") not in {"Standard", "Block"} or not gates:
                continue
            if gates == {condition}:
                block = self.blocks.get(node.get("ID"))
                if block is not None and block not in exclusive:
                    exclusive.append(block)
        return exclusive

    # -- convenience ------------------------------------------------------ #

    def block_text(self, block: Block) -> str:
        return "\n\n".join(
            self.questions[qid].text
            for qid in block.question_ids
            if qid in self.questions
        )

    def block_raw(self, block: Block) -> str:
        return "\n".join(
            self.questions[qid].raw_text
            for qid in block.question_ids
            if qid in self.questions
        )

    def block_media(self, block: Block) -> dict[str, int]:
        raw = self.block_raw(block)
        return {
            name: len(pattern.findall(raw)) for name, pattern in MEDIA_PATTERNS.items()
        }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _parse_question(element: dict) -> Question:
    payload = element["Payload"]
    raw = str(payload.get("QuestionText", ""))
    choices_raw = payload.get("Choices")
    choices: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    if isinstance(choices_raw, dict):
        order = payload.get("ChoiceOrder") or list(choices_raw)
        keys = [str(k) for k in order if str(k) in {str(x) for x in choices_raw}]
        choices = tuple(
            strip_html(str(choices_raw[k].get("Display", ""))) for k in keys
        )
        recode = payload.get("RecodeValues") or {}
        codes = tuple(str(recode.get(k, k)) for k in keys)
    answers_raw = payload.get("Answers")
    statements: tuple[str, ...] = ()
    if isinstance(answers_raw, dict):
        statements = tuple(
            strip_html(str(v.get("Display", ""))) for v in answers_raw.values()
        )
    return Question(
        qid=element["PrimaryAttribute"],
        kind=str(payload.get("QuestionType", "")),
        selector=str(payload.get("Selector", "")),
        text=strip_html(raw),
        raw_text=raw,
        choices=choices,
        codes=codes,
        statements=statements,
        export_tag=str(payload.get("DataExportTag", "")),
    )


def _walk(node) -> Iterator[dict]:
    """Every node of the flow tree."""
    if isinstance(node, list):
        for child in node:
            yield from _walk(child)
        return
    if not isinstance(node, dict):
        return
    if "Type" in node:
        yield node
    for child in node.get("Flow", []) or []:
        yield from _walk(child)


def _condition_values(logic) -> set[str]:
    """Condition values a branch's logic tests for."""
    values: set[str] = set()
    if isinstance(logic, dict):
        if logic.get("LogicType") == "EmbeddedField" and logic.get("Operator") in {
            "EqualTo",
            "Selected",
        }:
            right = logic.get("RightOperand")
            if right:
                values.add(str(right))
        for value in logic.values():
            values |= _condition_values(value)
    elif isinstance(logic, list):
        for item in logic:
            values |= _condition_values(item)
    return values


def _mentions_field(logic, field_name: str) -> bool:
    if isinstance(logic, dict):
        if (
            logic.get("LeftOperand") == field_name
            or logic.get("LogicType") == "EmbeddedField"
            and logic.get("LeftOperand") == field_name
        ):
            return True
        return any(_mentions_field(value, field_name) for value in logic.values())
    if isinstance(logic, list):
        return any(_mentions_field(item, field_name) for item in logic)
    return False


def _walk_gated(
    node, condition_field: str, gates: frozenset = frozenset()
) -> Iterator[tuple[dict, set[str]]]:
    """Flow nodes, each with the set of condition values gating it.

    An empty set means "shown to everyone who reaches it".
    """
    if isinstance(node, list):
        for child in node:
            yield from _walk_gated(child, condition_field, gates)
        return
    if not isinstance(node, dict):
        return

    current = gates
    if node.get("Type") == "Branch":
        logic = node.get("BranchLogic")
        if _mentions_field(logic, condition_field):
            values = _condition_values(logic)
            current = (
                frozenset(values)
                if not gates
                else frozenset(gates & values) or frozenset(values)
            )

    if "Type" in node:
        yield node, set(current)
    for child in node.get("Flow", []) or []:
        yield from _walk_gated(child, condition_field, current)


def load_survey(path: Path, condition_field: str = "Condition") -> Survey:
    return Survey.load(Path(path), condition_field=condition_field)


def describe(survey: Survey, conditions: Sequence[str] | None = None) -> list[dict]:
    """One row per condition: its exclusive blocks and their media content."""
    rows = []
    for condition in conditions or survey.conditions():
        blocks = survey.condition_only_blocks(condition)
        media: dict[str, int] = {name: 0 for name in MEDIA_PATTERNS}
        chars = 0
        for block in blocks:
            for name, count in survey.block_media(block).items():
                media[name] += count
            chars += len(survey.block_text(block))
        rows.append(
            {
                "condition": condition,
                "n_blocks": len(blocks),
                "chars": chars,
                **media,
                "blocks": "; ".join(block.description for block in blocks),
            }
        )
    return rows
