"""Render the twelve arms as fill-in-the-blank transcript templates.

The output directory is a drop-in twin of ``data/Voelkel/text_templates`` and
``data/pfander/text_templates``: one ``.txt`` per arm, a ``manifest.json`` giving
every response position in display order, ``00_FORMAT.md`` stating the contract,
and ``modality_audit.csv`` recording the keep-or-drop call for every arm.  The
sampler reads the format, not this module, so the three studies stay
interchangeable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random

from ..survey.render import render_template, slot_manifest
from . import instrument as inst
from .paths import MODALITY_AUDIT, QSF, TEMPLATES

FORMAT_DOC = """# Transcript and slot format — ICPC (International Climate Psychology Collaboration)

Same convention as the Pfänder and Voelkel templates: every response position is
three lines (question stem, an indented statement of the legal answers, the
`Response:` line), and every marker uses one sigil, `<<...>>`, stripped before any
text reaches the model. See `data/pfander/text_templates/00_FORMAT.md` for the
marker vocabulary.

## What is specific to this study

**All twelve arms are here.** No arm uses video, audio, an iframe or a script, and
none is interactive beyond typing — so unlike the Voelkel tournament, nothing had
to be dropped. Ten arms do carry static images, and those are the whole point of
two of them, so every picture is rendered as an `[IMAGE: ...]` line describing
what was on screen, with text *inside* a picture transcribed verbatim. The counts
per arm are in `modality_audit.csv`; the descriptions are in
`silicon_sampling/icpc/images.py` and the pictures themselves under
`data/ICPC/Materials/stimuli`.

**One picture is gone.** The third screen of *A Letter to Future Generations*
showed a graphic that has since been deleted from the survey's Qualtrics media
library and does not appear in `master_survey.pdf`. Its line says so rather than
guessing.

**The control arm reads its filler text before the shared definition**, and it
alone answers two extra blocks of covariates at the end — which is why
`01_control.txt` is the longest file here and carries the most response
positions.

**Three block orders are randomised.** Belief, policy support and the sharing item
are shuffled as a set of three (all six orders occur; the survey records which in
`FL_17_DO`), and the control arm's two extra blocks are shuffled as a pair
(`FL_32_DO`). Inside the second of those, Qualtrics draws one of nine wordings of
the same willingness item. Templates show one drawn order and the first wording;
each respondent gets their own.

**Two arms echo the respondent back at themselves.** *Correcting Pluralistic
Ignorance* quotes the participant's own estimate before giving the true figure,
and *Decreasing Psychological Distance* quotes the impacts they selected. Both
appear as `<<=slot>>` echoes.

**Slot ids.** For the shared screens they are the published data column names
(`Belief.in.CC_1`, `CC_policy_7`, `WEPT3confirm`), taken from the hand
transcription in `silicon_sampling/vlasceanu/content_shared.py`. For the
intervention screens they are the Qualtrics export tags with non-word characters
replaced, because an echo marker can only name word characters; `manifest.json`
carries the published column for every slot, and all of them were checked to exist
in the 668-column export before anything was rendered.

**Which item is bound to which column comes from the `.qsf`, not from the survey
PDF.** Existing is not the same as being the right column: an earlier render read
the batteries off `master_survey.pdf` visually, got the *order* wrong for four of
them, and so pointed 24 items — including both 0-100 scored outcomes — at a
neighbouring column. Nothing downstream could see it, because a battery mean is
invariant to permuting the items inside it. The binding is now re-derived from the
choice codes in `usa_3.qsf` (none of these questions defines `RecodeValues`,
`ChoiceDataExportTags` or `VariableNaming`, so the export suffix *is* the choice
code), cross-checked against `codebook.xlsx` and against the published per-item
means, and re-checked by `tests/test_icpc.py` on every run.

**Display logic is real, not decorative.** Sixteen questions carry Qualtrics
`DisplayLogic`, and each renders inside a `<<?if ...>> ... <<?endif>>` branch: the
eight WEPT number grids, the seven later WEPT decision pages (page *n* is only
offered to a respondent who accepted page *n*-1), and the sharing-platform
question (asked only of respondents who said they were willing to share). A
respondent's transcript therefore contains only the pages they would have reached.
That matters most for the effort outcome, which is the *count* of accepted pages:
ungated, the chain manufactures accept-after-decline patterns that occur zero
times in the 8,253 real US respondents.

## Known fidelity losses, stated once

- **Check-all-that-apply becomes select-one.** Three items are multi-select — the
  sharing platform, the household-goods SES index, and the psychological-distance
  impact list — and the transcript vocabulary has no multi-select slot. None is a
  scored outcome; the visible cost is that the arm 7 echo screen quotes back one
  impact where a human might have named several.
- **"Not applicable" and "No opinion" boxes cannot be selected.** The nine policy
  items and the control-only trust items offered one; a sampled answer is always a
  number. The prose line says the box was there. In the human data those answers
  are missing and drop out of the composite, so the effect is on which respondents
  contribute an item, not on the scale.
- **The WEPT grids are rendered but not scored.** The sixty numbers of each effort
  page are shown and asked about, because a respondent who never saw them did not
  do the task, but the published data records which boxes were ticked rather than
  which numbers, and the outcome is the count of pages accepted.
- **The debriefing form is the last element of the `.qsf`'s DEMOGRAPHICS block**,
  not a block of its own, and it is rendered here from the hand transcription
  rather than from the `.qsf`. Its page timer is in the data for 7,836 of the
  8,253 US respondents.
- **Every 0-100 slider states its range** on the prose line, as `Whole number from
  0 to 100.` after the endpoint labels. The endpoint labels alone are what the
  respondent saw, and they are also what left the models answering on a 0-10
  scale; the range is stated on all 257 sliders and
  `tests/test_icpc.py::test_every_slider_states_the_range_a_legal_answer_falls_in`
  keeps it that way.
- **The escape box is stated only where Qualtrics displayed one.** A Qualtrics
  slider can carry an `NA` *label* without the box being switched on
  (`Configuration.NotApplicable`), so the two are not the same question. The nine
  policy items, the political-orientation pair and the three control-only trust
  items had the box and say so; the sixteen `ID_*`, `Enviro_ID` and `Enviro_motiv`
  items carry the label but not the box, and correctly say nothing.
"""


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_modality_audit(path=MODALITY_AUDIT) -> list[dict]:
    """The keep-or-drop call for every arm, with its media counts."""
    rows = inst.modality_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_all(out_dir=TEMPLATES, seed: int = 20260823) -> dict:
    """Write one template per arm, plus the manifest, the format doc and the audit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale renders: the file set is derived from ARMS, so a changed arm
    # list must not leave an orphan behind.
    for stale in out_dir.glob("*.txt"):
        stale.unlink()

    manifest = {
        "study": (
            "Doell et al. (2024) / Vlasceanu et al. (2024), "
            "International Climate Psychology Collaboration"
        ),
        "fielded": "2022-07 to 2022-10",
        "sample": "United States quota subsample, n = 8253 (country == 'usa')",
        "n_arms": len(inst.ARMS),
        "source_qsf": QSF.name,
        "source_qsf_sha256": _digest(QSF),
        "instrument_note": (
            "Shared screens from silicon_sampling.vlasceanu.content_shared; "
            "the twelve stimuli straight from the .qsf."
        ),
        "item_binding": (
            "Every battery item's data_column is the .qsf choice code, "
            "cross-checked against codebook.xlsx and the published item means."
        ),
        "display_logic": (
            "Slots carrying 'shown_if' are the questions the .qsf gates; the "
            "set matches its DisplayLogic exactly."
        ),
        "arms": {},
    }
    rng = random.Random(seed)
    battery = inst.dv_order(rng)
    extras = inst.extras_order(rng)
    for arm in inst.ARMS:
        converted = inst.assemble(arm, battery=battery, extras=extras)
        name = f"{arm.code:02d}_{arm.slug}.txt"
        text = render_template(inst.header("<<=profile_id>>", arm), converted.elements)
        (out_dir / name).write_text(text, encoding="utf-8")
        slots = slot_manifest(converted.elements)
        for slot in slots:
            slot["data_column"] = converted.data_columns.get(slot["id"], "")
        manifest["arms"][arm.key] = {
            "cond": arm.code,
            "alias": arm.alias,
            "title": arm.title,
            "file": name,
            "qsf_block": arm.block,
            "battery_order": battery,
            "extras_order": extras if arm.code == 1 else [],
            "chars": len(text),
            "n_slots": len(slots),
            "slots": slots,
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "00_FORMAT.md").write_text(FORMAT_DOC, encoding="utf-8")
    write_modality_audit(out_dir / "modality_audit.csv")
    return manifest
