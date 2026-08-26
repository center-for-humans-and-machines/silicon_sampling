# CCC (Voelkel 2026) template fidelity: verification before sampling

Independent verification of the Voelkel-2026 Climate Change Challenge materials,
run before any template was built. Nine agents: six extracting the questionnaire
spec section by section, three adversarial verifiers with one concern each —
numeric ranges, coverage and composite formulas, and the thirteen treatment arms.

## The bug we were hunting is not present

The defect that cost this project an entire sampling round on ICPC and Goldwert was
a slider rendered with endpoint labels but no numeric range, so models answered on
an implicit 0–10 scale. On CCC:

* All **48 live slider questions** are `CSSliderMin 0` / `CSSliderMax 100`,
  `NumDecimals 0`, `GridLines 2`, no `SnapToGrid`.
* Every released slider column shows exactly **101 distinct integer values, 0–100**.
* The questionnaire PDF states "101-point scale from 0 (…) to 100 (…)" for every one.
* The Qualtrics-rendered PDF **proves the human saw the numerals "0 50 100"
  printed under each slider.**

So the template must print the range, and we know the exact form the human saw.
Note that Parts E/F/J/K account for only 36 of the 48; the other **12 are the Part L
manipulation checks**, which are sliders too and easy to miss.

## Digit preference: better here, and not because of the range

For the record, since it is easy to attribute to the wrong cause. Excess
round-number preference over the human reference, Qwen2.5-7B, like-for-like on
single-item 0-100 sliders:

| study | ours | human | excess |
| --- | --- | --- | --- |
| CCC | 35.1% | 30.0% | **+5.1 pp** |
| Goldwert (range stated) | 55.8% | 33.2% | **+22.6 pp** |

CCC is much closer to human, but **not because it states the range**. Goldwert
states the range too, since the audit, and still carries a 22.6 pp excess — the
range fix repaired *levels*, by 21 points, and left digit preference alone.
Pfänder, which also states its range, sits at 29.4% with no human data to compare
against.

So the CCC difference is real and unexplained. One untested hypothesis: CCC asks
every battery twice, so a model answering the post-treatment version has its own
pre-treatment numbers in context and may adjust from those rather than reaching
for a round one. Goldwert and Pfänder ask once.

## Six traps that would have corrupted results silently

**1. Twelve columns are sign-flipped in the released data.** `Step 2 -
Preparation.R` applies `100 - x` in place *before* writing the CSV, so the released
column is the inverse of what the respondent saw:

`Belief_Pre_3_1`, `Belief_Post_3_1`, `PoliciesSp_Pre_3_1`, `PoliciesSp_Post_3_1`,
and all eight `Candidate_Pre_1_1…4_1` / `Candidate_Post_1_1…4_1`.

Verified by diffing `Deidentified.csv` against `Recoded.csv`: `100 - raw` holds
exactly on those twelve and the other 85 shared numeric columns are unchanged.
Collecting a model's on-screen answer and comparing it to the released column would
have inverted belief, specific-policy and the entire candidate outcome.

**2. Donation is a constant-sum question, not six sliders.** `QID232` is
`QuestionType CS`, `ChoiceTotal 100`, stem "You have 100 cents total to donate".
Six boxes: five organisations plus "I would like to keep the following amount for
myself". The composite is a **sum in cents** over the five organisations, excluding
the sixth, and equals `100 - keep_for_self`.

**3. The QSF's trash block contains sliders on different ranges** — `0–7` with
`SnapToGrid`. A naive scrape that walks all questions would mix scales.

**4. `Education` in the recoded data is not the survey's five-level scale.** It is
collapsed to three strings: "HS or less", "Some college", "Bachelor or Postgraduate".

**5. `Income_B` is not a 1–11 scale** — `99` means "prefer not to say" (n = 429).

**6. `Candidate_Post_AT` is NA for all 13,821 rows**, hard-coded by the authors as
"Error in Survey Flow", which makes `Candidate_Post_RC` bit-identical to
`Candidate_Post`.

Also: an entire administered block (`Other Questions` — willingness and media-usage
items) was collected but never released, and appears in neither questionnaire PDF.
A faithful transcript should include it even though nothing can be scored from it.

## The thirteen arms

Text is complete and **verbatim-matched against the SI's "Text of Revised
Treatments"** for all thirteen — no piped fields, no embedded data, no JS, no
video, audio or iframes anywhere. Structure is uniformly simple: one block per arm,
one `BlockRandomizer`, zero display logic inside any arm.

**Images are the failure surface**: 22 across the arms, **none in the archive**, and
**18 of 22 URLs are dead**. The SI says they are available only on request.

| arm | verdict | media loss |
| --- | --- | --- |
| Consensus Framing 2 | fully faithful, pure text | 0 |
| Gains Framing | fully faithful, longest text at 4,693 chars | 0 |
| Free Market Framing | faithful; loses only the pacing of 11 one-sentence pages | 0 |
| Control Neckties / Baseball / Dances | faithful; images fetched and verified as decorative photos with no prose reference | 1 |
| Consensus Framing 1 | near-faithful; the 97% figure the image conveys is stated verbatim both before and after it | 1 |
| Binding Framing | probably faithful; an 800×182 masthead above the slogan, which is separately present as text | 1 |
| High Social Distance | mostly faithful; an 800×65 strip under a 16:9 photo is probably a caption rendered as an image, so one line of text is likely lost | 2 |
| Purity Framing | load-bearing but recoverable — the prose says "Look at the Great Smoky Mountains in the picture below. On the left … on the right …", and the live image measures a 3.2× saturation collapse across halves, exactly the clear/hazy pair the text claims, so a written description substitutes | 2 |
| Warmth Framing | faithful **only if the writing task is kept** — the arm's one response slot, and its active ingredient: median dwell 130.8 s on that page against 17.0 s on page 1, arm total the longest of thirteen by 40% | 2 |
| Dire But Solvable | **unresolved** — two image-only questions at the top of pages 3 and 4, no captions, no prose reference, absent from the SI, URLs dead, yet humans dwelled 20.8 s and 15.0 s on them | 2 |
| **System Preservation** | **not faithfully renderable** — six pages, twelve dead images, only ~1,200 characters of prose, and the prose never refers to the images: they *are* the concrete instantiation of the "American way of life" the final page asks the reader to preserve | **3** |

### Decisions

* **Drop `System Preservation Framing`** from the evaluated set, on the same rule
  that dropped seven Goldwert arms: the intervention's core is not in the text.
  Twelve of thirteen arms remain, including all three controls.
* **Describe images in brackets rather than dropping arms.** The inherited SDC
  convention — drop any arm containing an image — would delete nine of thirteen
  here, including every control, and with them the whole baseline.
* **Keep the Warmth writing task** as a free-text slot. Its answer was never
  released, so it cannot be scored, but omitting it removes what the arm does.
* **Reproduce the stimulus verbatim, errors included.** `System Preservation`
  captions a Yosemite photograph "Yosemite Valley, Nevada"; Yosemite is in
  California. That is an error in the original and is not ours to correct. The
  stray backslash Qualtrics rendered above "Chicago, Illinois" is a rendering
  artefact and is dropped deliberately rather than by accident.
* **Mark `Dire But Solvable` as unresolved** in the modality audit rather than
  guessing, and note the two images as shown-but-unrecoverable.

## Repo hazards the verifiers flagged

* `silicon_sampling/voelkel/` targets the *other* Voelkel study — its `paths.py`
  expects `SDC - Questionnaire - Qualtrics.qsf`. CCC must not point at it. Only
  `voelkel/qsf.py` is study-agnostic, and it parses the CCC QSF cleanly.
* The R scripts rename arms for publication — `Free Market Framing` becomes
  "Compatible Solution Framing", `Consensus Framing 1/2` become "Scientific
  Consensus 1/2" — and pool the three controls into `ConditionR == "Control"`
  (n = 3,183). The raw `Condition` strings are what the QSF and the data use.
* `voelkel_etal2026.csv` in `data/calibration/datasets/` is a byte-equivalent copy
  of `CCC - Data Attriter - Recoded.csv`, not of the main recoded file — same rows,
  49 extra derived columns.
