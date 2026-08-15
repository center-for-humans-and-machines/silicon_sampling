"""Render the retained conditions as fill-in-the-blank transcript templates."""

from __future__ import annotations

import json
import random

from ..survey.render import render_template, slot_manifest
from . import instrument as inst
from .paths import TEMPLATES

FORMAT_DOC = """# Transcript and slot format — Voelkel (Strengthening Democracy Challenge)

Same convention as the Pfänder templates: every response position is three lines
(question stem, an indented statement of the legal answers, the `Response:` line),
and every marker uses one sigil, `<<...>>`, stripped before any text reaches the
model. See `data/pfander/text_templates/00_FORMAT.md` for the marker vocabulary.

## What is specific to this study

**Only the pure-text arms are here.** Of 27 conditions, 17 use video, audio, an
iframe or images and 3 more are interactive or participant-authored. The keep or
drop call for every arm, with its media counts, is in `modality_audit.csv`.

**The instrument is party-adaptive.** Most of it adapts by piped text — the same
sentence reads "Republicans" or "Democrats" — which appears here as `<<=field>>`
echoes resolved from the respondent's party. Two stimuli exist as separate
Republican and Democrat blocks, so those conditions render one template per party.

**The outcome battery order is randomised** in the nested groups the survey's own
randomisers define. Templates show one drawn order; each respondent gets their own.

**`Misperception_Competition` echoes the respondent back at themselves**: the
correction screen quotes their own three estimates beside the true figures, which
appear as `<<=S2_MP_Dis>>`-style echoes.

**Slot ids are Qualtrics variable names**, so every one maps to a column in the
published data — which is how the legal answer sets were validated before any
sampling ran.

Only gender, race and party are asked on screen. Age, education and ideology came
from the panel supplier and appear nowhere in the instrument, so a synthetic
respondent cannot condition on them.
"""


def render_all(out_dir=TEMPLATES) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale renders: the file set depends on which arms are party-split,
    # so a changed rule must not leave an orphan behind.
    for stale in out_dir.glob("*.txt"):
        stale.unlink()
    columns = inst.data_columns()
    manifest = {
        "study": "Voelkel et al. (2024), Strengthening Democracy Challenge",
        "fielded": "2022-04-27 to 2022-05-26",
        "n_conditions": len(inst.CONDITIONS),
        "conditions": {},
    }
    for index, condition in enumerate(inst.CONDITIONS):
        parties = ("Republican", "Democrat")
        variant_specific = len(
            set(inst.stimulus_blocks(condition, "Republican"))
        ) != len(
            set(inst.stimulus_blocks(condition, "Republican"))
            & set(inst.stimulus_blocks(condition, "Democrat"))
        )
        for party in parties if variant_specific else ("Republican",):
            scenario = (
                inst.CCT5_SCENARIOS[0]
                if condition == "Misperception_Competition"
                else None
            )
            elements = inst.elements_for(
                condition,
                party,
                battery=inst.post_order(random.Random(0)),
                scenario=scenario,
            )
            suffix = f"_{party.lower()}" if variant_specific else ""
            name = f"{index:02d}_{condition.lower()}{suffix}.txt"
            text = render_template(inst.header("<<=profile_id>>", condition), elements)
            (out_dir / name).write_text(text, encoding="utf-8")
            slots = slot_manifest(elements)
            for slot in slots:
                slot["data_column"] = columns.get(slot["id"], "")
            manifest["conditions"][f"{condition}{suffix}"] = {
                "index": index,
                "file": name,
                "party": party if variant_specific else "either (piped)",
                "n_slots": len(slots),
                "slots": slots,
            }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "00_FORMAT.md").write_text(FORMAT_DOC, encoding="utf-8")
    return manifest
