# [DONE] Voelkel SDC — text templates, silicon sampling, and validation of the Pfänder pipeline

Task: [voelkel_pipleline.md](../tasks/voelkel_pipleline.md)
Prior art: [Pfänder plan](archive/2026-08-14-pfander-silicon-sampling.md) ·
[Pfänder report](../reports/pfander_silicon_sample/README.md)

## Goal

The Pfänder megastudy publishes no human data, by design, so nothing in that
submission can be checked. Voelkel et al. (2024), the *Strengthening Democracy
Challenge*, is the same shape — a US megastudy, one shared control, many
short-message arms, a fixed outcome battery — and it publishes **35,252
participant-level responses**. Running our pipeline over it end to end and
scoring it with the benchmark's own metrics is the closest available estimate of
how the Pfänder submission will actually do.

That purpose sets a hard constraint on the whole plan: **wherever a choice
arises, do what the Pfänder pipeline does**, even where Voelkel's richer data
would allow better. A score obtained by exploiting participant-level data we will
not have for Pfänder measures nothing we can use.

Three deliverable blocks, mirroring the Pfänder plan:

1. **Templates** — one text transcript per retained condition (`data/Voelkel/text_templates`).
2. **Sampling** — one filled transcript per synthetic respondent + an answer CSV
   (`data/Voelkel/silicon_sampling/qwen25_7b/`).
3. **Evaluation** — the benchmark's scoring metrics, ported from R to Python,
   applied to our synthetic sample against the real responses
   (`docs/reports/voelkel_validation/`).

## Sources, and what they establish

| Source | What it gives |
| --- | --- |
| `Materials/SDC - Questionnaire - Qualtrics.qsf` (2.0 MB) | The instrument as structured JSON: 120 blocks, 1,168 questions, the survey flow, and the verbatim intervention stimuli. **Primary content source.** |
| `Materials/SDC - Questionnaire.pdf` (374 pp) | The rendered instrument — the cross-check on screen order and on what an image actually showed. |
| `Data/SDC - Data - Recoded.csv` (35,252 × 113) | Participant-level responses: condition, demographics, the nine scored outcomes already rescaled to 0–100, per-outcome survey weights and attrition flags. **The ground truth.** |
| `Data/SDC - Data - Intervention Names.csv`, `Outcome Names.csv` | Canonical condition and outcome labels. |
| `Code/SDC - Script - *.R` | The paper's own estimation, including how the composite is built and how attrition is handled. |
| `/opt/llm_predictions_megastudy` (`R/functions/statistics.R`, `preregistration_benchmark.qmd`) | The benchmark's scoring functions and the half-split reference design. To be **translated**, not run. |

Measured while scoping:

- **27 conditions**: 25 interventions + `Null_Control` (n = 5,691) +
  `Alternative_Control`. Interventions run ~1,130 each.
- **Fielded 27 April – 26 May 2022.** The transcripts must be dated to then; the
  political context (Biden presidency, pre-midterms) is part of the stimulus.
- **All nine primary outcomes are already on 0–100** (`PA`, `ADA`, `SPV`, `SUC`,
  `OppBip`, `SocDistrust`, `SocDis`, `BEPF`, `Composite`), which makes the
  benchmark's percentage-points-of-scale-range conversion a no-op and puts every
  effect on one footing for free.
- **The instrument is party-adaptive.** Most blocks exist in Republican and
  Democrat versions and the outcome items are phrased about the *outparty*, so a
  respondent's party determines what they read. Party is therefore structurally
  pre-filled, not generated — unlike Pfänder. See *What this tests* below.
- **No media files ship** with the deposit: 13 videos, 65 audio tags, 241 images
  and 10 iframes are referenced but absent.
- **Half-split (seed 42) gives Human 1 = 17,550**, with 2,825 in `Null_Control`
  and a median of 563 per intervention.

## Decisions taken with the user

- **Pure text only.** An arm is dropped if its stimulus involves video, audio, an
  iframe, an image, or interaction (a trivia partner, a chatbot, a writing task).
  **Result: 6 interventions + the null control** (see below). This came in under
  the 8-arm checkpoint; re-confirmed with the user, who chose to stay strict
  rather than relax the rule.
- **Resumable under an unexpected kill.** The run must survive being killed at
  any moment, not merely exiting cleanly.
- **Profiles from published marginals**, mirroring the Pfänder quota procedure —
  not resampled from the real respondents, which would flatter a pipeline that
  will not have that luxury on Pfänder.
- **Unweighted human estimates as the headline**, with the paper's weighted,
  attrition-adjusted estimates reported alongside as a robustness check.
- **Sample size matched to the human half**: ~2,825 control + ~563 per retained
  intervention. At ~1,500 respondents/hour that is roughly **8,000 respondents,
  ~5–6 hours**.

### The modality audit, and the arms it retains

Produced by walking the survey flow, so the condition → block mapping is the
survey's own, not a guess from block names. Media counts are over each
condition's *exclusive* blocks — the stimulus, not the shared battery.

| Retained (6 + control) | chars | blocks |
| --- | --- | --- |
| `Null_Control` | 0 | — (straight to the outcome battery) |
| `Party_Overlap` | 25,622 | 10 |
| `Misperception_Suffering` | 10,641 | 2 |
| `Partisan_Threat` | 5,083 | 2 |
| `Misperception_Competition` | 3,584 | 5 |
| `Harmful_Experiences` | 3,178 | 1 |
| `System_Justification` | 1,260 | 1 |

Dropped for media: `Civity_Storytelling`, `Contact_Project`, `Democratic_Fear`,
`Economic_Interests`, `Utah_Cues`, `Misperception_Film` (video);
`Befriending_Meditation`, `Media_Trust` (audio); `Chatbot_Quiz` (iframe);
`Learning_Goals`, `Moral_Differences`, `Common_Identity`, `Misperception_Democratic`,
`Empathy_Beliefs`, `Violence_Efficacy`, `Inparty_Elites`, `Alternative_Control`
(images). The borderline single- and double-image arms were checked individually:
every image is 420–958 px with no alt text, i.e. a substantive graphic rather than
a logo or spacer, so none survives the rule.

Dropped for interaction despite carrying no media: `Epistemic_Rescue` (scripted
trivia with a simulated cross-party partner), `Counterfactual_Selves` and
`Outparty_Friendship` (writing tasks).

**Consequence, stated up front.** Six intervention clusters is thin. Pooled
correlations, the calibration slope and the cluster bootstrap (which resamples
*interventions*, because the nine outcomes within an arm share that arm's draw)
will all carry wide intervals — only large departures from the human replication
reference will be readable. The reference row is computed on the same six arms,
so the comparison stays fair; it is the absolute numbers that go soft. This is
reported as a limitation, not worked around.

---

## Part 1 — Text templates

### Output

```
data/Voelkel/text_templates/
  00_FORMAT.md          the transcript and slot convention (as Pfänder)
  manifest.json         every slot per condition, in display order
  00_null_control.txt
  01_<arm>.txt … NN_<arm>.txt
  modality_audit.csv    every one of the 27 arms, its media counts, and the keep/drop call
```

### Method

The `silicon_sampling/survey/` machinery is study-independent and is reused
unchanged: element and slot types, legal-value grammars, the renderer, the
session walker. Only the content module is new (`silicon_sampling/voelkel/`).

1. **Parse the qsf.** Blocks, questions, choices, sliders and text entries come
   out as structured JSON. The survey flow (`FL`) gives the condition →
   block mapping, the screen order, and the randomisations; the 381 page timers
   independently confirm the screen count, the same trick the Vlasceanu plan used.
2. **Audit modality per arm** and apply the pure-text rule. The audit table is a
   deliverable, not a footnote: an arm dropped for "one image" needs the call to
   be inspectable. A layout-only image (logo, spacer) does not count as a
   stimulus; a manipulation-carrying one does, and the questionnaire PDF is the
   arbiter where the qsf alone cannot say.
3. **Resolve the party branch.** Render each retained condition for both the
   Republican and the Democrat version. These are separate template files
   (`..._republican.txt` / `..._democrat.txt`) because the participant-visible
   text genuinely differs; they map to one condition label in the data.
4. **Validate against the real data.** Every slot id is a Qualtrics variable, so
   for each one the legal set can be checked against the values that actually
   occur in `Recoded.csv`. This catches a mis-parsed scale or a missed option
   before any GPU time is spent — a check the Pfänder work could not run.
5. **Date the header to 2022** and keep the condition identified by its internal
   code (`XOVS`, `8Z5I`, …) rather than its evaluative title, as in Pfänder.

---

## Part 2 — Silicon sampling

Reuses `silicon_sampling/sampling/` unchanged, with the configuration the Pfänder
re-tuning settled on: CUDA graphs, `gpu_memory_utilization=0.96`, bf16 KV cache
(fp8 is ruled out — it drove the illegal-draw rate from 1.8% to 37.5%), group
size auto-fitted to the measured cache from the worst-case transcript length, and
rejection sampling at temperature 1.0 with no top-p/top-k truncation.

**Profiles.** Built from the demographic marginals the paper and supplement
report (gender, age band, race, education, party, ideology), apportioned by the
same maximum-entropy joint and largest-remainder rounding used for Pfänder.
Party is pre-filled because the instrument branches on it; the remaining
demographics follow the Pfänder split between pre-filled and generated as closely
as the instrument allows.

**Runtime.** ~2,825 control + 6 × ~563 ≈ **6,200 respondents**, ~4–5 h.

**Resumability under an unexpected kill.** The Pfänder run was restartable, but
only tested against a clean exit. Hardening it, in the shared `sampling` module
so both studies get it:

- **Torn-line recovery.** `answers.jsonl` is appended to; a kill mid-write leaves
  a partial final line, and the *next* append then concatenates onto it, silently
  corrupting two records. On startup the file is truncated back to its last
  newline-terminated line before anything is appended.
- **Durable checkpoints.** `flush()` survives a killed process but not a killed
  machine; each group's records are now `fsync`ed. That is one sync per ~15
  respondents — free at this cadence.
- **Idempotent group completion.** Transcripts are written before the answer
  record, so a kill between the two costs a redo, never a half-recorded
  respondent. The answer log stays the single source of truth for what is done.
- **Deterministic retry.** Seeds derive from `profile_id`, so a respondent
  interrupted and re-run reproduces exactly — a resumed run and an uninterrupted
  one give the same dataset.
- **Stale-engine cleanup** in the wrapper, so a killed run does not leave a
  process holding the GPU and block the restart.

A kill test is part of the calibration step: interrupt mid-group, restart, and
confirm the answer count and the per-respondent values are what an uninterrupted
run produces.

**The parsing-bias policing carries over.** The near-miss diagnostic runs on this
run too: every rejected draw is classified into "meant a legal answer, rejected on
spelling" (a parser bug, which biases the distribution whenever the failure rate
depends on which answer was meant) versus "answered a different question"
(correctly rejected). Voelkel's answer options are mostly short Likert labels, so
the money-formatting class of bug should not recur — but the dictator game is
dollars, and that is exactly where it bit last time.

---

## Part 3 — Evaluation (the substantial new work)

This is where the task differs from Pfänder: not "analyse the sample" but "score
it against ground truth with the benchmark's own metrics".

### The benchmark's design, reproduced

Split the human sample 50/50 on a fixed seed. **Human 1** is the reference every
prediction is scored against. **Human 2** predicts Human 1 exactly as a
submission would, and its scores are the **human replication reference** — what a
fresh human sample of that size achieves. That row is the yardstick: an absolute
Pearson r means little, but "our r versus what a real replication gets" is
interpretable. Two further baselines, **no effect** and **all positive**, anchor
the metrics that have no natural null.

### To port from R to Python (`silicon_sampling/benchmark/`)

Written as a study-independent module, because it is equally the tool for
self-scoring the Pfänder submission before it is filed.

| Function | What it computes |
| --- | --- |
| `run_main_treatment_model` | ATE per intervention vs control, OLS with HC2, BH-adjusted p. (Our `analysis/ols.py` has HC1; HC2 is a small addition.) |
| `pooled_metrics` | Directional agreement (half credit for exact zeros), Spearman ρ, Pearson r, the within-outcome companion, and the noise-corrected `pearson_adj`. |
| `adjusted_metrics` | Noise correction: subtract the reference's mean sampling variance from the observed spread before correlating. |
| `run_calibration_pooled` | α and β of `ATE_human ~ ATE_predicted`, plus `beta_adj` corrected for noise in the predictions. |
| `compute_ovl`, `compute_w1`, KS, variance ratio | Distribution-shape metrics per condition × outcome, on a fixed 0–100 grid. |
| `signed_metrics`, subgroup pairs | Subgroup heterogeneity: condition × moderator interactions, scored the same way. |
| `compare_demographic_baselines`, `demographic_parity_gap` | Control-condition group means and the worst-vs-best-served gap. |
| `cluster_boot` | 95% intervals by bootstrapping over *interventions*, since the outcomes within an arm share that arm's draw. |

Each ported function gets a test against a hand-computed case, as `ols.py` did —
that check caught nothing but proved the estimator, and these are the numbers the
whole exercise reports.

### Report

```
docs/reports/voelkel_validation/
  README.md            how well the approach does, against the human replication reference
  01_effects.md        our ATEs vs the real ATEs, arm by arm and outcome by outcome
  02_distributions.md  variance ratio, OVL, KS, W1 — does it reproduce the spread, not just the mean
  03_subgroups.md      subgroup heterogeneity and demographic baselines
  04_diagnostics.md    sampling diagnostics and the near-miss audit
  plots/
```

The headline figure is the benchmark's own: predicted against human ATEs, one
point per arm × outcome, with the identity line, our calibration line, and the
human replication reference overlaid.

---

## What this actually tests, and what it cannot

Worth stating plainly, because the number this produces will be quoted.

**The informative part.** The Pfänder sample's defining weakness was that its
respondents were demographically flat: synthetic Republicans and Democrats
differed by 1.1 points on climate belief where the real gap is tens of points,
and no moderator explained more than R² = 0.002. Voelkel is close to an ideal
test of whether that is fixable, for two reasons. Its outcomes *are* partisan
animosity — if the model cannot separate partisans here it cannot do anything
useful on this instrument. And its instrument is party-adaptive, so party is
pre-filled and the questions themselves name the respondent's in- and out-party.
That is precisely the intervention the Pfänder write-up proposed as the obvious
next experiment, arriving for free.

**The limits.** Three, and the report will carry all of them:

1. **Different topic, different year.** Voelkel is democratic norms in 2022;
   Pfänder is climate scientists in 2026. A model can be good at one and bad at
   the other, and a 2022 instrument is inside the model's training window in a way
   a 2026 one is not — the model may know how this study came out.
2. **A different, smaller arm set.** Pure-text-only leaves ~8–10 arms against
   Pfänder's 16. Pooled correlation metrics over few arms are noisy, so intervals
   will be wide, and the human-replication row — which suffers the same thinness —
   is the only fair comparison.
3. **We score a subset the paper never reports.** Dropping the non-textual arms
   means our human reference is not the paper's headline result. Internally
   consistent, but not a replication of Voelkel et al.

## Risks, and what is done about them

| Risk | Response |
| --- | --- |
| Too few arms survive the pure-text rule for the pooled metrics to say much | Arm count is fixed in step 1, before any GPU time. If it falls below 8, report it and check with the user before sampling. |
| The composite outcome's construction is non-obvious | Reproduce it from the paper's own R, and verify our recomputation matches the published `Composite` column on the real data to floating-point tolerance. |
| Attrition and exclusions | The reference must use the paper's own exclusions; the `Attrited_*` flags make this checkable, and the recomputed unweighted ATEs are validated against the paper's reported effects before scoring anything against them. |
| Sign conventions | Six of the nine outcomes are "bad is high". Directional agreement is meaningless if a sign is flipped, so the recomputed ATEs are checked against the paper's reported direction per outcome. |
| The model has seen this study | Cannot be prevented. Flagged as a limit; the free-text and near-miss logs are inspected for signs of the model recognising the instrument. |
| Party-adaptive branching doubles the templates | Two files per condition, one manifest; the session walker already handles conditionals. |

## Steps

**Phase 1 — templates**
1. [x] qsf parser: blocks, questions, choices, flow, timers.
2. [x] Modality audit over all 27 arms. **Came in at 6 interventions + control,
   under the 8-arm checkpoint; confirmed with the user, who chose to stay strict.**
3. [x] Content module, party branch, the nested outcome-battery randomisation, and
   the correction screen that quotes the respondent's own estimates back at them.
4. [x] 9 templates + manifest + `00_FORMAT.md` + `modality_audit.csv`.
5. [x] Every slot's legal set validated against the values respondents actually
   gave: 55 slots checked, no violations. 57 of 63 backbone slots map to a
   published column; the 6 that do not are exactly the items never published.

**Phase 2 — sampling**
6. [x] 6,203 profiles from the published marginals, all reproduced within ~1%.
7. [x] Calibration + kill test.
8. [x] Full run: 6,203 respondents, ~1,850/h, **6.3% of draws rejected**.
9. [x] `samples.csv`; the nine outcomes recomputed and verified.

**Phase 3 — evaluation**
10. [x] `silicon_sampling/benchmark/` — the benchmark's scoring in Python, 13 tests.
11. [x] Outcome construction verified against the published columns: **max absolute
    difference 0.0 across 31,000 respondents**, all nine.
12. [x] Half-split, human replication reference, both null baselines.
13. [x] Scored, with cluster-bootstrap intervals over the six interventions.
14. [x] [`docs/reports/voelkel_validation/`](../reports/voelkel_validation/README.md).

**Wrap-up**
15. [x] `black`, `flake8`, 27 tests pass.
16. [x] Task and plan archived; committed and pushed.

## Result

| | silicon | human replication | no-effect baseline |
| --- | --- | --- | --- |
| directional agreement | 61% | 67% | 50% |
| Pearson r | 0.41 [0.11, 0.55] | 0.51 [0.33, 0.68] | — |
| RMSE (pp) | 3.62 | 1.68 | 1.54 |
| calibration slope β | 0.16 | 0.44 | 0.00 |

Three findings, in descending order of how much they should change what we do:

1. **Rank order is real, magnitudes are not.** r = 0.41 is ~79% of what a fresh
   human sample of the same size achieves — genuinely useful. But the effects are
   2.6× too spread out and β = 0.16, so the RMSE is *worse than predicting no
   effect at all*. Correcting for noise in the predictions barely moves the slope
   (β_adj = 0.17 against the replication's 0.66), so this is not a small-sample
   artefact: the model believes these messages do several times more than they do.
2. **The levels are wrong even where the spread is right.** Mean absolute level
   error is 23 points on a 0-100 scale, and two outcomes are effectively inverted
   (opposition to bipartisan cooperation: 21 real against 83 synthetic). Yet the
   mean variance ratio is 1.13 — within a condition the synthetic responses are
   about as dispersed as the real ones. This is not the degenerate
   everyone-answers-50 failure; it is people who disagree by about the right
   amount about the wrong thing.
3. **The Pfänder flatness result reproduces, under the strongest possible test.**
   The moderators the model could see predict its subgroup effects no better than
   the ones it could not (pooled r 0.26 against 0.24) — even though this
   instrument writes the respondent's party into the wording of nearly every
   question. Pre-filling party does *not* fix the flatness, which answers the
   follow-up the Pfänder write-up proposed.

## What this means for the Pfänder submission

Take the ordering seriously and the levels not at all. On a leaderboard scoring
correlation this approach looks respectable; on one scoring RMSE or calibration it
loses to a constant zero. If effort goes anywhere next, it should go at the
magnitude problem — the effects are big because the model treats a persuasive
message as far more persuasive than it is — rather than at demographic realism,
which pre-filling party has now been shown not to solve.

The three caveats from the plan all stand: different topic and year, six
intervention clusters so every interval is wide, and a human reference computed on
a subset the paper never reports.
