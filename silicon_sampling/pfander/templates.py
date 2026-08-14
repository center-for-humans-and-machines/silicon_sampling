"""Render the 17 fill-in-the-blank templates and their manifest.

A template is the canonical form of one condition: every conditional branch
expanded, every response position left as a ``<<...>>`` marker.  It is what a
human reads to check the instrument.  The sampler does not read it — it walks the
same element structure directly — so the two can never drift, and a template is
regenerated rather than edited.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from ..survey.elements import Block, Conditional, Text
from ..survey.render import render_template, slot_manifest
from . import build, conditions, instrument, sources
from .paths import CODEBOOK_CSV, CODENAMES_CSV, QUESTIONNAIRE_TXT, TEMPLATES

FORMAT_DOC = """# Transcript and slot format

The 17 `.txt` files in this directory are **templates**: one per experimental
condition, holding everything a respondent in that condition read, with every
position at which they produced an answer left blank and annotated.

Filling one in for one respondent yields a transcript like the ones under
`data/pfander/silicon_sampling/*/raw/`.

## Where the content comes from

Every word is verbatim from the benchmark's own instrument,
`survey/questionnaire.txt` in the submission-template repository (snapshotted
under `data/pfander/submission_template/`). Nothing is paraphrased. Authoring
scaffolding that respondents never saw — intervention author lists, `Tag:` and
`Summary:` lines, the state-to-case assignment tables, sections marked
`[not displayed to participants]` — is excluded.

## Layout

```
==============================================================================
 THE SILICON SAMPLE BENCHMARK — PARENT MEGASTUDY
 ...
 Condition      : 05  [code name: worse wildfowl]
==============================================================================
```

The header identifies the condition by its **internal survey code name**, the
identifier `survey.qsf` itself uses, and never by its descriptive title. Titles
like "Oil industry misinformation" are the researchers' evaluative labels; a
respondent never saw one, and putting it at the top of the document would leak
the experimenter's framing into the very thing the manipulation is supposed to
produce.

`- - - [ page N ] - - -` marks a screen break. Pages are reconstructed from the
source file's own page-break markers plus one screen per question block; they
are not recovered from response timers, which do not exist for this study yet.

Every response position takes three lines:

```
Q40. How incompetent or competent are most climate scientists?
      0 = Very incompetent … 100 = Very competent.
Response: <<trust_competent_1 :: int :: 0..100>>
```

the question stem, an indented statement of what a legal answer is, and the
answer itself. A model is shown the transcript up to and including
`"Response: "` — never past it, and never with a marker in view.

## Markers

All markers share one sigil, `<<...>>`, and are stripped before any text reaches
a model.

| Marker | Meaning |
| --- | --- |
| `<<id :: int :: 0..100>>` | integer in range, inclusive |
| `<<id :: choice :: A \\| B \\| C>>` | exactly one of the listed options |
| `<<id :: pattern :: \\d{5}>>` | free text matching the regex |
| `<<id :: text :: free text, <= N chars>>` | open text box |
| `<<=id>>` | piped text: an answer echoed back on screen, not a response position |
| `<<?if note>> … <<?endif>>` | a branch shown only to respondents matching `note` |

`id` is the Qualtrics variable name from `codebook.csv` wherever the study has
one, so an answer needs no translation to be compared against the human data.
Ids that exist only inside an intervention (`funding_intv_*`, `hpt_estimate`,
`consensus_*`, `state`) are ours: those items are part of the manipulation and
are not scored.

## What varies between respondents within a condition

- **control** shows exactly one of three filler texts, drawn uniformly. All
  three appear in `00_control.txt`, each inside its own `<<?if>>` branch.
- **Consensus** randomises its three estimate items with item 3 always in the
  middle, per the survey's own instruction.
- **Extreme weather predictions** is state-adaptive: the respondent's state
  selects one of four case texts and fills the intro paragraph.
- The **secondary and tertiary outcome blocks** are presented in random order;
  the primary trust battery is always first. Templates show the canonical order.
- The conditional demographics (partisan importance, born-again, religiosity)
  appear only for the respondents the survey would have asked.

## Question numbering

`Qnn` numbers count response positions as displayed to that respondent. A
respondent who skips a conditional branch has fewer questions, so the same item
can carry different numbers in different transcripts — as it would in any survey
with display logic. In the templates, every branch is expanded, so the numbering
there is the maximal one.

## manifest.json

The same information, machine-readable: for each condition, every slot in display
order with its id, kind, legal values, whether it is pre-filled or generated, and
the display condition if it sits inside a branch. Also carries the codebook's
composite-construction rules and SHA-256 digests of the source files the
templates were built from.
"""


def _control_template_elements() -> list[object]:
    """The control stimulus with all three filler texts shown as branches."""
    branches: list[object] = []
    for code_name, heading in conditions.CONTROL_TEXTS.items():
        branches.append(
            Conditional(
                note=f'the drawn filler text is "{code_name}"',
                predicate=lambda answers: True,
                elements=[
                    Text(paragraph)
                    for page in sources.pages(heading)
                    for paragraph in page
                ],
            )
        )
    return [Block(key="control", title="control", elements=branches)]


def template_elements(condition: str) -> list[object]:
    """Canonical element sequence for one condition's template file."""
    if condition == "control":
        stimulus: Sequence[object] = _control_template_elements()
    else:
        stimulus = [conditions.condition_block(condition)]
    return [
        *instrument.PRE_CONDITION,
        *stimulus,
        instrument.TRANSITION_TO_OUTCOMES,
        instrument.POST_PRIMARY,
        *instrument.POST_RANDOMISED,
        instrument.END_OF_SURVEY,
    ]


def file_name(index: int, condition: str) -> str:
    return f"{index:02d}_{conditions.slug(condition)}.txt"


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def composites() -> dict[str, str]:
    """Composite-construction rules, straight out of the codebook."""
    return {
        row["target_label"]: row["question_text"]
        for row in sources.codebook()
        if row["section"].startswith("B.") and row["target_label"]
    }


def render_all(out_dir=TEMPLATES) -> dict:
    """Write every template, `manifest.json` and `00_FORMAT.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    codenames = sources.codename_map()
    manifest = {
        "study": build.STUDY_TITLE,
        "survey_year": instrument.SURVEY_YEAR,
        "n_conditions": len(conditions.CONDITIONS),
        "sources": {
            "questionnaire.txt": _digest(QUESTIONNAIRE_TXT),
            "codebook.csv": _digest(CODEBOOK_CSV),
            "condition_codenames.csv": _digest(CODENAMES_CSV),
        },
        "marker_syntax": {
            "response": "<<id :: kind :: legal-values>>",
            "echo": "<<=id>>",
            "branch": "<<?if note>> ... <<?endif>>",
            "kinds": ["int", "choice", "pattern", "text"],
        },
        "composites": composites(),
        "conditions": {},
    }

    for index, condition in enumerate(conditions.CONDITIONS):
        elements = template_elements(condition)
        code_name = build.template_code_name(condition)
        text = render_template(
            build.header("<<=profile_id>>", condition, code_name), elements
        )
        name = file_name(index, condition)
        (out_dir / name).write_text(text, encoding="utf-8")
        slots = slot_manifest(elements)
        manifest["conditions"][condition] = {
            "index": index,
            "file": name,
            "code_names": [
                key for key, title in codenames.items() if title == condition
            ],
            "n_slots": len(slots),
            "n_generated": sum(1 for slot in slots if slot["source"] == "generated"),
            "slots": slots,
        }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "00_FORMAT.md").write_text(FORMAT_DOC, encoding="utf-8")
    return manifest
