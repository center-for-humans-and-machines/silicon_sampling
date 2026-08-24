"""Render the usable arms as fill-in-the-blank transcript templates.

Also writes ``modality_audit.csv`` beside them, because the audit is the reason
the file set has eleven templates rather than eighteen and the two should never
drift apart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random

from ..survey.render import render_template, slot_manifest
from . import instrument as inst
from . import outcomes as oc
from .paths import ARM_QSF_DIR, MASTER_QSF, RESPONSES_CSV, TEMPLATES, arm_qsf

FORMAT_DOC = """# Transcript and slot format — Goldwert (Climate Advocacy Megastudy)

Same convention as the Pfänder templates: every response position is three lines
(question stem, an indented statement of the legal answers, the `Response:` line),
and every marker uses one sigil, `<<...>>`, stripped before any text reaches the
model. See `data/pfander/text_templates/00_FORMAT.md` for the marker vocabulary.

## What is specific to this study

**Eleven of eighteen arms are here.** Five are built around a video and two around
a screenshot of a newspaper article, and neither can go into a text transcript
without inventing content. The keep-or-drop call for every arm, its reason, its
media counts and a 0-3 `media_loss` grade are in `modality_audit.csv`; the grade
exists so that "do models do worse on media-heavy arms?" can be asked as a
regression rather than as an impression.

**The control is kept, and its screen is not blank.** Its only content is a
five-minute knot-tying video, and that video's *content* is null — which is why the
arm is usable. But real control participants spent five minutes being asked to
concentrate on something before they reached the outcome battery, and a synthetic
respondent handed the battery straight away is not an untreated respondent, they
are a differently-treated one. Every effect in this study is a contrast against
this arm, so the screen says what it was: the video's title, its length, its
subject, and that the page would not advance until it had finished.

**Media that remains is described, not marked.** Nine of the eleven kept arms show
photographs, charts or diagrams, and the transcript carries what each one showed —
`[ image shown here: A photograph of ... ]` — written from the file.
`data/Goldwert/Materials/stimuli` holds the pictures themselves, fetched from the
hosts the exports hot-link, with `index.json` recording provenance per file, and
`silicon_sampling/goldwert/images.py` holds the descriptions. This replaced a
content-free `[ image shown here — not reproduced ]` note, which was honest about
having looked at nothing and wrong about the consequence: two screens of
`ThreatInjustEfficacy` have no content except photographs and rendered as nothing
but brackets, and `BindingMorals` asks how impure the Great Smoky Mountains look
"in the picture on the right above". Six files have been deleted from their media
libraries and 404 on every host; those say so, and name what the surrounding copy
says they showed.

The same line distinguishes the live third-party panels four of the nine outcome
pages are built on: the petition is an Environmental Defense Fund action page, the
two newsletter signups are the organisations' own subscribe forms, and the bank
score is a lookup on `bank.green`. Those the respondent acted *inside*, which is
why they are described as panels and not as images.

**The outcome battery was reconstructed.** The authors published one Qualtrics
export per arm rather than one survey. Two of those exports are exports of the
whole master survey with everything but their own arm swept into a single
"Trash / Unused Questions" block: 630 questions, flattened, no page breaks, no
flow, no block names. That block is the only surviving copy of the nine outcome
pages, the mediators and the demographics. Which questions belong to which
outcome page is named by hand in `instrument.BATTERY`; the *order* of the pages
comes from the published file's own column order, which is the Qualtrics export
order and therefore builder order.

**Page breaks inside an arm are the survey's own; page breaks inside the battery
are inferred.** An arm's live blocks carry explicit `Page Break` elements and
those are used verbatim. The trash block carries none, so the battery's screens
are cut at its page timers instead — a page timer belongs to exactly one page.
That is weaker evidence, and the two rules are kept apart in the code so it stays
visible.

**The nine outcome blocks were shown in a random order per respondent.** Each
template shows one drawn order; `DV_order` in the published file gives each real
respondent's. This is not cosmetic: the study's own analysis finds five-point
swings between first and last position. The video-sharing outcome, the attention
check, the two efficacy items, the ten emotion ratings and the demographics
follow the nine blocks in fixed order.

**Two arms randomise their own blocks and show all of them.** `HopeAngerNarratives`
shows both the hope and the anger narrative, and `MispCorrectionRisks` shows all
six correction pages; in both cases the survey randomises the order, not the
selection. Templates show one drawn order.

Which blocks move is read from the `BlockRandomizer` node of the arm's own flow,
not from a count. Counting them and shuffling that many blocks off the end of the
list is what the first version did, and for `MispCorrectionRisks` — nine live
blocks, a randomiser over blocks two to seven — that shuffled the last four
corrections together with the writing prompt and the closing debrief, so the page
summarising all six corrections could appear after only two of them. `FL_34_DO`
and `FL_62_DO` in the published file record each real respondent's own draw and
confirm that only the six corrections, and only the two narratives, ever moved.

**`MispCorrectionRisks` holds its content in the survey flow, and its page script
holds the rest.** Its six correction paragraphs — about 700 words, the whole
substance of the intervention — are `EmbeddedData` values in the flow, displayed
through `${e://Field/employment_text}` pipes, and are resolved here. Five more
fields the flow declares without a value were written by the live page script, and
those are reproduced rather than stubbed, because the export holds both the values
and the rules:

* `Question N out of 6` counts up with the drawn order. The flow sets `n` to 1 once
  and the script increments it per screen, so resolving it from the flow alone
  headed all six screens "Question 1 out of 6".
* `text` is the "That's correct!" / "That's incorrect!" line. Both strings are flow
  literals and the recode table on each question says which answer earns which.
  A template shows it as `<<=..._feedback>>` — one field per screen, not one
  shared by six, because a session re-renders the whole transcript at every step
  and a shared field lets a later answer rewrite an earlier screen.
* `choice_text` and `img` are the correction paragraph and photograph for whichever
  issue the respondent named as most disruptive, on the page that then asks them to
  write about it. Both are flow literals selected by the recode value of that
  choice.
* `col` stays empty: it is a hex colour in a `style` attribute.

**`IndStructuralChange` echoes the respondent back at themselves**: "You guessed
32% … the actual number is 18%" appears as an `<<=...>>` echo of the slot holding
that guess.

**Slot ids are the column names in the published file**, not the survey's export
tags — the authors renamed a lot of them before publishing (`donation_1` became
`donation`, `Q65_5` became `Hope`) and the rename table is transcribed in
`convert.PUBLISHED_COLUMN`. So every scored slot maps straight onto a column in
`goldwert_etal2026.csv` with no translation. Two exceptions have no column at all:
`letter_content` and `zipcode_1` were de-identified out of the published file, and
what survives of the letter is `letter`, a 0/1 code of whether it expressed clear
thoughts about climate change.

**Intervention-internal slots are prefixed with the arm's own name.** The eighteen
exports were built separately and reuse each other's tags — three arms call a
writing prompt `Q5`, and so does the consent item, which *is* a published column.
Nothing an intervention asks is scored (those answers exist only in the
unpublished raw export), so the prefix costs nothing and stops an arm from
overwriting a real outcome.

**Consent and both attention checks are pre-filled as passed.** The published file
contains only respondents who passed them.

**Five sliders carry an opt-out, and it is answerable.** `pol_candidate`,
`flyless`, `lessbeef` and the two political-orientation rows printed a Qualtrics
"Not Applicable" checkbox, worded differently on each — "Not Applicable / Not
Eligible to Vote", "Not Applicable (e.g., \"I already don't fly\"", "Prefer not to
respond" — and the transcript prints the survey's own words and accepts them. An
answer that takes the escape records `N/A`, which the analysis frame coerces to the
same `NaN` the published file has; that is where this study's missingness comes
from, and on `flyless` it is 41% of everyone who reached the page. What is not
reproducible is the *rate*: a synthetic respondent decides whether the item applies
to it from a one-line profile.

**The donation is a constant-sum item and the transcript can only say so.** The
survey refused a page whose two boxes did not total ten. The driver samples one
slot at a time and cannot hold a constraint across two, so the rule is stated in
prose, the recorded pair is reconciled to `donation_keep = 10 - donation`, and
`samples.csv` carries `donation_keep_drawn` and a per-row `donation_sums_to_ten`
flag. Real models honour it 93% to 98% of the time unaided.

**Known remaining gaps.** Four questions randomise their *options* rather than
their order of appearance — the gender item, the two political-orientation slider
rows, `MispCorrectionRisks`'s eight-way EPA multi-select and
`EcologicalDisruptions`'s six-row affect matrix — and the transcript shows one fixed
order for each. All four are either pre-filled from the profile or
intervention-internal, so nothing scored moves. The debrief screen that followed the
demographics is not rendered; it comes after the last response position, so no
answer can depend on it. `Gender_4_TEXT`, the free-text box behind the gender item's
"other", has no slot, because gender is supplied from the profile.

## Two scales to distrust

`belief_1` and `policy_1` put "Very much so" at the *left* of the slider and "Not
at all" at the right — the opposite of every other slider in the instrument. In
the data they behave like agreement scales, but barely: the Democrat-Republican
gap on `belief_1` is 5.8 points against 21 points on `conversation`, and on
`policy_1` there is no party gap at all. The paper reports neither item. They are
rendered here because the respondent saw them, and excluded from
`outcomes.SCORED`.

Eleven sliders start at zero with a custom start position, so a respondent who
never touched the control records an exact 0 rather than a missing value. Their
distributions have a non-response spike at the bottom that cannot be separated
from a real "definitely not".
"""


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_modality_audit(out_dir=TEMPLATES) -> list[dict]:
    """Per-arm media counts and the keep-or-drop call, as a CSV."""
    rows = inst.modality_audit()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "modality_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_all(out_dir=TEMPLATES, seed: int = 0) -> dict:
    """Write one template per usable arm, plus the manifest, format doc and audit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{arm.cond:02d}_{arm.slug}.txt" for arm in inst.ARMS if arm.usable}
    # The Voelkel renderer deletes stale renders at this point. This one does not:
    # everything under `data/` is gitignored, so a delete here is unrecoverable,
    # and an orphan that is *named* in the manifest is a smaller problem than a
    # file that quietly disappeared. If the modality audit ever changes its call,
    # `orphaned_templates` is where that shows up.
    orphans = sorted(
        path.name for path in out_dir.glob("*.txt") if path.name not in expected
    )

    audit = write_modality_audit(out_dir)
    manifest = {
        "study": "Goldwert et al. (2026), A megastudy of behavioral interventions to "
        "catalyze public, political and financial climate advocacy",
        "journal": "PNAS Nexus 5(1): pgaf400",
        "osf_project": "wv7c3",
        "fielded": "2024-06",
        "n_participants": 31324,
        "n_arms": len(inst.ARMS),
        "n_arms_usable": len(inst.CONDITIONS),
        "orphaned_templates": orphans,
        "dropped_for_video": [
            a.name for a in inst.ARMS if not a.usable and a.modality == "video"
        ],
        "dropped_for_image_of_text": [
            a.name for a in inst.ARMS if not a.usable and a.modality == "image_of_text"
        ],
        "dv_block_order_canonical": list(inst.DV_BLOCK_ORDER),
        "post_battery_order": list(inst.POST_BLOCKS),
        "scored_outcomes": {
            name: {"scale_max": scale, "label": oc.LABELS.get(name, name)}
            for name, scale in oc.SCORED.items()
        },
        "media_loss_scale": inst.MEDIA_LOSS,
        "unscored_by_design": {
            "belief_1": "reverse-labelled slider, no party gap, unreported by the paper",
            "policy_1": "reverse-labelled slider, no party gap, unreported by the paper",
            "bankscore": "requires looking up the respondent's own real bank on a live site",
        },
        "composites": {name: list(parts) for name, parts in oc.COMPOSITES.items()},
        "political_advocacy_no_letter": list(oc.LETTER_FREE),
        "sources": {
            "master_qsf": {"path": MASTER_QSF.name, "sha256": _digest(MASTER_QSF)},
            "responses_csv": {
                "path": RESPONSES_CSV.name,
                "sha256": _digest(RESPONSES_CSV),
            },
        },
        "arms": {},
    }
    for arm in inst.ARMS:
        row = next(r for r in audit if r["condName"] == arm.name)
        entry = {
            "cond": arm.cond,
            "modality": arm.modality,
            "usable": arm.usable,
            "media_loss": arm.media_loss,
            "media_loss_meaning": inst.MEDIA_LOSS[arm.media_loss],
            "reason": arm.reason,
            "qsf": arm.qsf,
            "words": row["words"],
            "media": {
                key: row[key]
                for key in (
                    "image",
                    "piped_image",
                    "graphic",
                    "video",
                    "iframe",
                    "audio",
                    "n_assets",
                    "described",
                    "labelled_from_export",
                    "undescribed",
                    "unrecoverable",
                )
            },
        }
        if arm.usable:
            rng = random.Random(seed)
            elements = inst.elements_for(
                arm.name, battery=list(inst.DV_BLOCK_ORDER), rng=rng
            )
            name = f"{arm.cond:02d}_{arm.slug}.txt"
            text = render_template(inst.header("<<=profile_id>>", arm.name), elements)
            (out_dir / name).write_text(text, encoding="utf-8")
            slots = slot_manifest(elements)
            published = {
                slot["id"] for slot in slots if not slot["id"].startswith(arm.slug)
            }
            for slot in slots:
                slot["intervention_internal"] = slot["id"].startswith(f"{arm.slug}__")
                slot["scored"] = slot["id"] in oc.SCORED or slot["id"] in oc.BY_COLUMN
            entry.update(
                {
                    "file": name,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "chars": len(text),
                    "n_slots": len(slots),
                    "n_published_slots": len(published),
                    "slots": slots,
                }
            )
        manifest["arms"][arm.name] = entry

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "00_FORMAT.md").write_text(FORMAT_DOC, encoding="utf-8")
    return manifest


def arm_qsf_digests() -> dict[str, str]:
    """SHA-256 of every arm export, so a template can be traced to its source."""
    return {path.name: _digest(path) for path in sorted(ARM_QSF_DIR.glob("*.qsf"))}


def arm_source(arm_name: str):
    """The export a given arm's text was read from."""
    return arm_qsf(inst.BY_NAME[arm_name].qsf)
