"""Render the retained arms as fill-in-the-blank transcript templates."""

from __future__ import annotations

import json
import random

from ..survey.render import render_template, slot_manifest
from . import instrument as inst
from .paths import TEMPLATES

FORMAT_DOC = """# Transcript and slot format — Voelkel et al. (2026), Climate Change Challenge

Same convention as the other four studies: every response position is three lines
(question stem, an indented statement of the legal answers, the `Response:` line),
and every marker uses one sigil, `<<...>>`, stripped before any text reaches the
model. See `data/pfander/text_templates/00_FORMAT.md` for the marker vocabulary.

## What is specific to this study

**Every slider states its numeric range.** All 48 live sliders run 0-100, and the
Qualtrics print export confirms the respondent saw the numerals "0 50 100" under
each track. The transcript therefore prints the endpoint labels *and* the range —
never one instead of the other, which is the defect that cost an entire sampling
round on ICPC and Goldwert.

**Outcomes are measured twice**, before and after the message, because the study
is designed that way and its published estimand uses the pre-measure as a
covariate.

**Images are described, not dropped.** Twenty-two images appear across the arms;
none is in the archive and 18 of 22 URLs are dead. Each becomes a bracketed note
saying an image was shown and how large it was. Only Purity Framing gets its
content restored in words, because its prose points at the picture deictically and
its image is one of the four still fetchable, so the content is recoverable.

**One arm is dropped.** `System Preservation Framing` is twelve images with ~1,200
characters of prose that never refers to them. `modality_audit.csv` records the
media-loss rating for every retained arm.

**The donation is one constant-sum task**, six amounts that must total 100 cents,
not six independent sliders.

**Slot ids are Qualtrics export tags**, so each maps to a column in the released
data — which is how the legal answer sets were checked before sampling. Twelve of
those columns are stored as `100 - x` of what the respondent saw; the conversion
back is in `outcomes.py`, not here.
"""


def render_all(out_dir=TEMPLATES) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.txt"):
        stale.unlink()
    columns = inst.data_columns()
    manifest = {
        "study": "Voelkel et al. (2026), Climate Change Challenge",
        "source": "Nature Climate Change 16(2), 214-225",
        "n_conditions": len(inst.conditions()),
        "dropped": inst.DROPPED_ARMS,
        "conditions": {},
    }
    for index, condition in enumerate(inst.conditions()):
        elements = inst.elements_for(condition, random.Random(0))
        slug = condition.lower().replace(" ", "_")
        name = f"{index:02d}_{slug}.txt"
        text = render_template(inst.header("<<=profile_id>>", condition), elements)
        (out_dir / name).write_text(text, encoding="utf-8")
        slots = slot_manifest(elements)
        for slot in slots:
            slot["data_column"] = columns.get(slot["id"], "")
        manifest["conditions"][condition] = {
            "index": index,
            "file": name,
            "media_loss": inst.MEDIA_LOSS.get(condition),
            "n_slots": len(slots),
            "chars": len(text),
            "slots": slots,
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "00_FORMAT.md").write_text(FORMAT_DOC, encoding="utf-8")
    return manifest
