# [ACTIVE] Vlasceanu et al. (2024) — Qualtrics survey → text-based fill-out templates

Task: [docs/tasks/translating_qualtrics_survey_to_text_based.md](../tasks/translating_qualtrics_survey_to_text_based.md)

## Goal

Turn the Vlasceanu et al. (2024) *ManyLabs Climate* Qualtrics master survey
(`data/Vlasceanu/master_survey.pdf`, 80 pages, corrupted text layer → read
visually) into one plain-text transcript per experimental condition, containing
**only what a participant saw**, with every response position machine-identifiable
and its legal answers declared.

Downstream use: give a base LLM the document truncated at a response position and
score/sample the continuation.

## Sources used

| Source | What it settled |
| --- | --- |
| `master_survey.pdf` (visual read of all 80 pages) | verbatim on-screen wording, images, answer widgets |
| `paper.pdf` | flow narrative, intervention rationale, item wording cross-check |
| `intervention_adaptation_manual.pdf` | which strings/images are country-adapted; attention-check placement |
| `data_raw.csv` header + missingness | true screen order, page boundaries, randomisations, response codings |

## Key findings that shape the output

1. **The print preview is not the display order.** The Qualtrics print preview
   lists *blocks* in builder order. Two blocks are displaced relative to what
   participants saw. Missingness analysis on `data_raw.csv` (n = 83,927) settles it:
   - `AttentionCheck_purp` ("select the color purple") is answered by 2,630
     respondents who never reached the climate-change definition page, and by 279
     control respondents who never reached the Dickens text → it is the **second
     screen, directly after consent**. Confirmed by the adaptation manual
     ("The first one is directly after the consent form").
   - `Attn_60` ("type the word sixty") is missing whenever `Gender` is missing →
     it is the **first screen of the demographics section**, not before the DVs.
2. **Page boundaries are recoverable from the timer variables.** Every screen
   carries a `*_Page.Submit` timer; there are 118. This gives an exact screen count
   per block (e.g. working-together norms = 16 screens, collective action = 13,
   psychological distance = 9, WEPT = 3 + 8×2). Used to place `[ page N ]` breaks.
3. **Control condition ordering is unusual**: control participants read the Dickens
   excerpt *before* the shared climate-change definition (73 have the Dickens timer
   but no definition timer, only 2 the reverse). Experimental participants get the
   definition *before* their intervention (93 vs 0).
4. **Randomisations** (from `_DO` columns): the three DV blocks appear in fully
   random order (`FL_17_DO`, all 6 permutations ≈ equally frequent); items within
   the belief, policy, emotion, environmental-identity/motivation matrices and
   within the gender and SES-item lists are randomised; the control-only
   "terms probing" block shows **exactly one randomly chosen probe out of nine**
   (verified: every control participant has 0 or 1 probe answered).
5. **Piped text** appears in three places (echo of the participant's own earlier
   answer, or of the condition code) — needs a distinct marker, it is not a
   response slot.

## Deliverables

1. `data/Vlasceanu/text_survey/`
   - `00_FORMAT.md` — the transcript/slot convention.
   - `01_control.txt` … `12_binding_moral_foundations.txt` — 12 transcripts,
     numbered by the dataset's `cond` code.
   - `manifest.json` — every slot, in display order, per condition: id, dataset
     column, kind, legal values, prompt anchor, screen index.
2. `silicon_sampling/vlasceanu/` — the survey content as structured Python data
   plus the renderer, so the text files are reproducible and other country
   variants can be rendered later (`CountryProfile`).
3. `docs/reports/vlasceanu_text_survey_translation.md` — report on the elements
   that resisted faithful textual transformation.

## Format decisions

- `- - - [ page N ] - - -` screen break (screens = Qualtrics pages, from timers).
- `[IMAGE: …]` alt text; text *inside* an image that carries the manipulation
  (flyer, pie charts, infographics) is transcribed in full inside the alt text.
- `( )` radio option, `[ ]` checkbox option — plain-text questionnaire idiom.
- `>> <how to answer> - {{slot_id}}` — the response line. `>>` marks the one kind
  of line that is *not* on-screen text (it renders a widget in words), so the
  transcript stays honest about what the participant actually read.
- `{{slot_id}}` = generate here. `{{echo:col}}` = the survey echoed an earlier
  answer here. Slot ids are the Qualtrics/dataset column names wherever one
  exists, so generated answers can be validated against the real data.
- 0–100 sliders → `>> Answer from 0 (Not at all) to 100 (Extremely) - {{id}}`,
  preceded by the tick labels as displayed.
- Country-variable strings are `{people}` / `{country}` / … templates in the spec
  and rendered with the US master profile (the master survey *is* the US version).

## Open questions / assumptions recorded in the report

- Exact intra-block page splits for a few blocks (negative emotions screen 11,
  terms probing) are not fully recoverable; best effort, documented.
- "Thank you for your responses, we will now ask you some questions about your
  climate change perceptions." is printed at the end of the future-self block;
  it may have been a shared transition. Kept where printed, flagged.
- The survey mislabels Yosemite Valley as being in Nevada; kept verbatim.

## Steps

1. [x] Read all 80 survey pages visually; extract embedded images for alt text.
2. [x] Recover flow, page boundaries, randomisations, codings from the dataset.
3. [x] Read the adaptation manual for country-variable strings.
4. [ ] Write `silicon_sampling/vlasceanu/survey_spec.py` (content) and
   `render.py` (emitters for `.txt` + `manifest.json`).
5. [ ] Generate the 12 transcripts + manifest + `00_FORMAT.md`.
6. [ ] Validate: every slot id that names a dataset column exists in
   `data_raw.csv`; slot counts per condition match expectations.
7. [ ] `black .`, `flake8 . --max-line-length=200 --extend-ignore=E203,W503`.
8. [ ] Write the report; move task to `docs/tasks/archive`; commit and push.
