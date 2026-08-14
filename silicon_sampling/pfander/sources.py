"""Readers for the benchmark's shipped materials.

Long verbatim prose — the 19 stimulus texts — is read out of
``survey/questionnaire.txt`` rather than retyped, so a transcription slip cannot
change what a respondent reads.  Structure that the text file only implies (which
paragraphs sit on which screen, where the response positions are inside the four
interactive interventions) lives in :mod:`~silicon_sampling.pfander.conditions`.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache

from .paths import CODEBOOK_CSV, CODENAMES_CSV, QUESTIONNAIRE_TXT

_SECTION_RE = re.compile(r"^### (.+?)\s*$", re.MULTILINE)

#: Both spellings of the page-break marker used in the source file.
PAGE_BREAK_RE = re.compile(
    r"^\s*[—–-]{1,2}\s*[Pp]age [Bb]reak\s*[—–-]{1,3}\s*$", re.MULTILINE
)


@lru_cache(maxsize=1)
def questionnaire_text() -> str:
    return QUESTIONNAIRE_TXT.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def sections() -> dict[str, str]:
    """``### heading`` → body, for the 19 stimulus texts."""
    text = questionnaire_text()
    found: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found[match.group(1)] = text[start:end]
    return found


def stimulus(heading: str) -> str:
    """The verbatim body of one stimulus section."""
    try:
        return sections()[heading]
    except KeyError:  # pragma: no cover - a source-file change should be loud
        raise KeyError(
            f"no '### {heading}' section in questionnaire.txt; have: {sorted(sections())}"
        ) from None


def paragraphs(text: str) -> list[str]:
    """Blank-line-separated paragraphs, whitespace-normalised."""
    return [
        re.sub(r"\s*\n\s*", " ", block).strip()
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
    ]


def pages(heading: str) -> list[list[str]]:
    """One stimulus split into screens at its page-break markers."""
    body = stimulus(heading)
    return [
        paragraphs(chunk) for chunk in PAGE_BREAK_RE.split(body) if paragraphs(chunk)
    ]


@lru_cache(maxsize=1)
def codebook() -> list[dict[str, str]]:
    with CODEBOOK_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def codename_map() -> dict[str, str]:
    """Internal survey code name → canonical condition title."""
    with CODENAMES_CSV.open(encoding="utf-8", newline="") as handle:
        return {row["code_name"]: row["title"] for row in csv.DictReader(handle)}


@lru_cache(maxsize=1)
def target_labels() -> dict[str, str]:
    """Qualtrics variable name → the column name a submission must carry."""
    return {
        row["qualtrics_label"]: row["target_label"]
        for row in codebook()
        if row["qualtrics_label"] not in ("", "NA")
    }
