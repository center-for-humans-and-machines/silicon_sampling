# [DONE] Pfänder megastudy — text templates, silicon sampling with Qwen2.5-7B, and analysis

Tasks:
[jan_pfanders_silicon_sampling.md](../tasks/jan_pfanders_silicon_sampling.md) →
[questionnaire_to_templates.md](../tasks/questionnaire_to_templates.md) ·
[sampling_template_responses_with_local_model.md](../tasks/sampling_template_responses_with_local_model.md) ·
[data_analyze_silicon_samples.md](../tasks/data_analyze_silicon_samples.md)

## Goal

Produce a Tier-1 (individual-level) synthetic sample of the Silicon Sample
Benchmark megastudy — 17 conditions × 13 outcomes, N ≈ 18,000 — by feeding
Qwen2.5-7B (base) a plain-text transcript of the questionnaire truncated at each
response position and sampling the continuation. Then analyse the result as if
it were the real study.

Three deliverable blocks, one plan:

1. **Templates** — 17 text transcripts of the questionnaire (`data/pfander/text_templates`).
2. **Sampling** — 18,000 filled transcripts + one answer CSV
   (`data/pfander/silicon_sampling/qwen25_7b/{raw,samples.csv}`).
3. **Analysis** — effect sizes, demographic moderation, response distributions
   (`docs/reports/pfander_silicon_sample/`).

## Sources and what they settled

| Source | What it settled |
| --- | --- |
| `/opt/silicon-sample-submission/survey/questionnaire.txt` | **Primary content source.** The instrument in chronological display order, with the Qualtrics variable name and legal codes annotated on every item, all 17 condition texts verbatim, and the state→case logic for #16. |
| `/opt/silicon-sample-submission/codebook.csv` | Exact submission levels for the six moderators, `qualtrics_label` → `target_label` mapping, and every composite's construction rule. |
| `/opt/silicon-sample-submission/survey/condition_codenames.csv`, `scripts/lib/submission_spec.R` | Canonical condition labels and the code-name map; the Tier-1 schema the checker enforces. |
| `/opt/silicon-sample-submission/predictions/example_T1_primary_v1.csv` | Exact Tier-1 column order. |
| `data/pfander/questionnaire.html` | Cross-check on wording; section groupings. Superseded by `questionnaire.txt` where they differ. |
| `data/pfander/preregistration.html` | N and per-arm allocation, the census quota table, the 13 scored outcomes, the six moderators, the pp-of-scale-range scoring unit. |
| Local probe | RTX 4090 (24 GB), vLLM 0.11.0 / torch 2.8.0, `Qwen/Qwen2.5-7B` present in the HF cache. `statsmodels` and `seaborn` are **not** installed → analysis uses numpy/scipy/pandas/matplotlib only. First vLLM start pays a long one-time `torch.compile`; a killed driver leaves an orphan `VLLM::EngineCore` holding 15 GB. |

### Facts that shape the design

- **17 conditions**: 16 text interventions + one shared control. Allocation
  1,000 per intervention, 2,000 control = 18,000.
- **13 outcomes**, all 0–100 except `donation_ams` ($0–10) and
  `newsletter_signup` (0/1). Multi-item outcomes are the mean of their items,
  so every item must be sampled individually (12 + 5 + 4 + 3 + 6 + 7 = 37 items
  feed the composites).
- **Quotas are two 2-way margins**, not a 3-way joint: age × gender and
  race × gender (Table 3 of the prereg, 2024 Census PEP, rescaled to 18,000).
  The two margins agree on the gender total (8,827 vs 8,828 male — rounding),
  and gender is Male/Female only in the table ("Other" is explicitly not
  quota-constrained). Building profiles therefore needs iterative proportional
  fitting, not a lookup.
- **Four interventions carry response slots inside the treatment**: #4 Funding
  (7 sliders), #8 High public trust (1 estimate), #12 Consensus (3 estimates),
  #16 Extreme weather (a US-state choice that selects which of four texts is
  shown next). Templates must support conditional branches for this and for the
  pre-treatment conditionals (party importance, born-again, religiosity).
- **Per participant ≈ 85–92 response slots**, of which 7 are pre-filled →
  **≈ 1.4–1.5 M generation calls** for the full run. This is the dominant
  engineering constraint (see *Throughput*).

### Decisions taken with the user

- **Control arm**: each control participant reads exactly one of the three
  filler texts (Neckties / Baseball / Dances), drawn uniformly. Recorded as
  `control_text` in the CSV.
- **Attention and consent items**: pre-filled as passed (consent = Yes, no-AI =
  Yes, attention check 1 = "Somewhat disagree", attention check 2 = "attention").
  The human reference contains only respondents who passed these, so sampling
  them would create a selection effect with no counterpart in the target data.
- **Missingness**: none. Every non-pre-filled slot receives a legal value.
  Documented as a known deviation (the real instrument forces only consent,
  gender, age and race).

---

## Part 1 — Text templates

### Output

```
data/pfander/text_templates/
  00_FORMAT.md            the transcript + slot convention, in full
  manifest.json           every slot, in display order, per condition
  00_control.txt
  01_corporate_reliance.txt … 16_extreme_weather_predictions.txt
```

Numbering follows the questionnaire's own intervention numbering; each file
carries the intervention title and its code name (`giant gibbon; brick bobcat` —
semicolons are part of the identifier, never split).

### Transcript style

The templates are written to read like a per-participant response file from an
OSF data deposit, because that is the document class we want the base model to
believe it is completing:

```
================================================================================
 THE SILICON SAMPLE BENCHMARK — PARENT MEGASTUDY
 "16 information interventions to strengthen trust in climate scientists in the US"
 Full response transcripts, one file per participant.
--------------------------------------------------------------------------------
 File           : responses/participant_00417.txt
 Participant ID : 00417
 Condition      : 05 — Oil industry misinformation   [code name: worse wildfowl]
 Instrument     : US master version, fielded 2026
 Note           : Verbatim record of one session, screens in display order.
                  Lines starting "Response:" hold what the participant entered.
================================================================================

- - - [ page 9 ] - - -

Q23. How much do you trust climate scientists?
     Slider, 0 = not at all … 100 = very strongly.
Response: 72
```

Rules:

- Every item is `Qnn.` + verbatim stem + an indented line stating the legal
  answers in words + a `Response:` line. This is the format the task asked for.
- `- - - [ page N ] - - -` marks a screen break. Pages come from the explicit
  `—page break—` markers in the questionnaire plus one page per question block;
  the reconstruction is documented in `00_FORMAT.md` as reconstructed, not
  recovered from timers (unlike the Vlasceanu transcripts, no response data
  exists yet to recover them from).
- Matrix questions render as a shared header + one `Qnn.x` line per item, so
  each item is its own response position.

### Slot markers

One sigil, `<<…>>`, for everything the renderer touches. Markers are always
stripped before the model sees the text — a prompt ends at `Response: `.

| Marker | Meaning |
| --- | --- |
| `<<trust_post :: int :: 0..100>>` | generate an integer in range |
| `<<education :: choice :: Less than high school \| High school diploma / GED \| …>>` | generate exactly one listed option |
| `<<race :: multi :: White / Caucasian \| Black / African American \| …>>` | generate one or more listed options, comma-separated |
| `<<donation_ams :: money :: 0..10>>` | dollars, integer or two decimals |
| `<<zip :: pattern :: \d{5}>>` | regex-constrained free text |
| `<<comments :: text :: 60>>` | free text, token budget |
| `<<=party>>` | echo an earlier answer (piped text, not a response position) |
| `<<?if party in {Republican, Democrat}>> … <<?endif>>` | conditional branch |

So the legal set is stated twice: once in prose the model reads, once machine-
readably inside the marker. `manifest.json` mirrors the markers as structured
JSON (slot id, order index, type, legal values, `source: generated|prefilled`,
the outcome variable it feeds) so the sampler and the analysis never re-parse
the text files.

Slot ids are the prereg/codebook variable names wherever one exists
(`trust_post`, `distrust_post`, `belief_post`, `donation_ams`,
`newsletter_signup`, …) and the 37 composite items are named
`<outcome>_<item>` so composites are rebuildable by prefix.

### Content sources for the template body

Everything is transcribed verbatim from the questionnaire, including the
scale-instruction blurb ("Below is a range from 0 to 100. Click on any space
within this range…"), the section transitions, and the intervention texts.
Editorial scaffolding that participants never saw (author lists, "Tag:",
"Summary:", "Code name:", the `[not displayed to participants]` reference lists
in #16, the tabsets' section headings) is excluded from the transcripts and
recorded in `manifest.json` metadata instead.

---

## Part 2 — Silicon sampling

### Output

```
data/pfander/silicon_sampling/qwen25_7b/
  raw/<condition_slug>/<profile_id>.txt     18,000 filled transcripts
  samples.csv                               18,000 rows × every answer + composites
  tier1_submission.csv                      the benchmark's Tier-1 column set
  draws.jsonl                               per-slot audit log (attempts, rejects, raw text)
  run_meta.json                             model revision, sampling params, seeds, timings
```

`raw/` is sharded by condition — 18,000 files in one flat directory is
needlessly painful to work with; the task's path is preserved as the root.

### Profiles (pre-filled demographics)

1. Read the prereg quota table into an age × gender table and a race × gender
   table (counts, not percentages — the percentages round).
2. **IPF** the two margins into a 4 × 2 × 5 joint (age band × gender × race),
   then allocate integer counts to 18,000 with largest-remainder rounding.
3. Assign each profile to a condition by shuffling within the whole sample, so
   demographics are balanced across arms by construction (the real study
   randomises, which is the same thing in expectation and less noisy here).
4. Expand age band → exact age (uniform within band, 60+ drawn from a
   Census-shaped tail truncated at 90) → year of birth for the questionnaire's
   "What is your year of birth?" item.
5. Race quota category → the questionnaire's select-all-that-apply wording as a
   single selection. Documented deviation: real respondents can tick several.
6. Everything is seeded from a single run seed; `profile_id` → per-participant
   seed, so any participant is reproducible in isolation.

`education`, `income` and `party` are also benchmark moderators but are **not**
pre-filled — the task specifies age/gender/race only, so those three are
generated by the model and their marginals become an analysis diagnostic
(comparable to published US distributions).

### Engine and sampling parameters

```python
LLM(model="Qwen/Qwen2.5-7B", dtype="bfloat16", max_model_len=8192,
    enable_prefix_caching=True, gpu_memory_utilization=0.90)

SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,        # untruncated
               repetition_penalty=1.0, presence_penalty=0.0,
               frequency_penalty=0.0,
               max_tokens=<per-slot, 4–12 (60 for free text)>,
               stop=["\n"], n=<attempts>, seed=<slot seed>)
```

`top_p=1.0, top_k=-1, temperature=1.0` and no penalties is the "faithful to the
learned distribution" setting the task asks for. `stop=["\n"]` does the
truncation: the model cannot run past the response line, and the tokens before
the newline are still drawn from the untruncated distribution.

### Rejection sampling

Per slot: request `n = 4` independent continuations in one call (they share the
cached prefix, so four attempts cost barely more than one), parse each in order,
take the first legal one. If all four fail, retry the call with a fresh seed up
to `max_rounds = 4` (→ 16 attempts). If it still fails, fall back to vLLM's
grammar-constrained decoding for that one slot and log it; if *that* fails, log
the slot as forced and record the fallback in `run_meta.json`. The fallback rate
is a reported diagnostic, not a silent patch.

Parsing is *prefix-truncation then validation*: strip leading whitespace, take
the longest prefix that matches the slot's legal grammar (e.g. `0..100` takes
the leading integer run and checks the range; `choice` matches the longest
option, case-insensitively, and rejects if the remainder is not empty or
punctuation). Selecting the first legal draw out of *n* i.i.d. draws is exact
rejection sampling from the model's distribution restricted to the legal set.

> Note for the record: this is distributionally equivalent to renormalising the
> model's probability over the legal set, and cheaper to do that way for the
> 0–100 sliders. The task specifies rejection sampling, so that is the default
> path; the renormalisation path is not implemented unless the pilot shows the
> rejection loop is the bottleneck.

### Re-tuning after the environment rebuild (vLLM 0.23, same GPU)

The container was rebuilt mid-project with vLLM 0.23 / torch 2.11. Ported the
engine (`guided_decoding` → `structured_outputs`, several `LLM()` keywords became
engine-arg passthroughs) and re-measured everything. Four configurations,
85–102 respondents each, all 17 arms:

| config | KV cache | group | steady rate | illegal draws |
| --- | --- | --- | --- | --- |
| CUDA graphs, util 0.92 | 102,000 | 12 | ~1,416/h | 1.85% |
| CUDA graphs, util 0.96 | 119,600 | 14 | ~1,491/h | 1.81% |
| CUDA graphs, util 0.96, **auto group** | 119,600 | 15 | ~1,440/h | 1.89% |
| CUDA graphs, util 0.96, **fp8 KV** | 192,048 | 22 | ~1,113/h | **37.5%** |

What that settles:

1. **CUDA graphs are worth ~17% per group slot**, not the 1.5–2.5× I predicted
   from the eager-mode decode timings. The workload is serialised by 78
   sequential dependencies per respondent, so *concurrency* — not kernel-launch
   overhead — is the binding constraint, and concurrency is capped by KV cache.
2. **fp8 KV cache is ruled out on evidence, not principle.** It does double the
   cache and does allow a 57% bigger group, and it is still *slower*, because it
   drives the illegal-draw rate from 1.8% to 37.5% and the retries eat the gain.
   Quantised attention visibly corrupts this model's short-answer generation.
   That is a much better reason to reject it than the fidelity worry I started
   with.
3. **Group size is now fitted to the cache** the engine actually reports, from
   the worst-case transcript length (7,499 tokens incl. margin — the measured max
   is 7,185). Worst-case working set at group 15 is 107,775 of 119,600 tokens, so
   no session is ever evicted.

Net: ~1,450–1,490 respondents/hour, **≈12.2 h** for 18,000, against ~1,280/h
measured on the old stack. Two environment breakages also had to be fixed:
`HF_HOME` points at a container-local path holding only metadata while the
weights live in the mounted cache (and the image's `hf_xet` is broken, so the
fallback download raises), and the compile cache had to move out of the repo
tree.

### What the first calibration measured (vLLM 0.11)

Three calibration runs, 34-68 respondents each, all 17 arms:

| | value |
| --- | --- |
| GPU KV cache | **101,232 tokens** at `gpu_memory_utilization=0.90`; **114,432** at 0.93 |
| Longest transcript | 7,064 tokens (Funding arm) → ~16 concurrent sessions fit |
| Throughput | **~1,300 respondents/hour** at group 16 → **≈ 14 h** for 18,000 |
| Rejection rate | **2.4 %** of draws |
| Constrained-decoding fallbacks | 2 in 51 respondents (~4,300 slots) |
| Python overhead | 1 ms per lockstep round — the loop is entirely GPU-bound |

Three things had to be fixed before those numbers were trustworthy:

1. **The V1 engine's separate `EngineCore` process hangs** on this machine, every
   time, right after the weights reach the GPU — reproduced on a bare vLLM script
   with none of this package involved. `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs the
   engine in-process and fixes it.
2. **`/home/claude/.cache` is root-owned**, so vLLM could not cache compiled
   graphs and paid a full recompile on every start. Caches now live under
   `data/.cache`.
3. **Hand-set `max_tokens` truncated long answer options.** The income item's
   budget fitted `"Less than $30,000"` but not `"$100,000 to $167,999"`, so the
   long brackets were rejected *systematically*: the first calibration produced a
   barbell of the two shortest brackets, with the three middle ones never once
   selected. Budgets are now fitted per slot with the model's own tokenizer.
   Income immediately became unimodal and centred, and the overall rejection rate
   halved. This is the single most dangerous class of bug in this design — a
   parsing rule that fails differently across options silently reshapes the
   distribution it is supposed to be measuring.

### The option-dependent rejection problem, and how it is now policed

The `max_tokens` bug was the first instance of a general failure mode, and two
more turned up in the live run. Rejection sampling is unbiased **only if the
rejection probability does not depend on which answer the model meant**. Where it
does, the retained distribution is skewed by exactly that asymmetry.

Found and fixed:

1. **Truncated long options** (above) — income collapsed onto its two shortest
   brackets.
2. **Dropped currency symbols and separators.** The model writes
   `100,000 to $167,999` or `30000 to 55999`. Rejecting these gave a failure rate
   of 3% for the one income option starting with a word and 58% for the ones
   starting with `$`, inflating the `$30,000–$55,999` bracket from ~18% to ~28%.
   Money options now compare with the cosmetics stripped, guarded so the laxity
   is refused if it would make two options ambiguous.
3. **Slider decimals.** `92.36` was refused as a non-integer. But a slider is a
   continuous control that the survey records as an integer, so the answer is
   real and is now rounded the way the instrument would round it.

Policed by a **near-miss diagnostic** (`stats.near_misses`) that splits rejected
draws into those that mean a legal answer under a loose comparison — a parser
bug, bias-prone — and those that answer a different question — correctly
rejected. After the fixes the remaining rejections are overwhelmingly the second
kind: bare numerals on `newsletter` and `education_climate_*` (slider-style
values bleeding in from neighbouring items, 2% near-miss share), and
out-of-frame refusals like "Not applicable" on `social_class`.

One residual is reported rather than fixed: on `newsletter` — a *scored* binary
outcome — the in-format answers run 65/35 No/Yes while the rejected numerals, if
read as codes, would imply 56/44. The numerals include values like 30 and 7, so
they are not a coherent coding and cannot be mapped without guessing; the
discrepancy is recorded in the diagnostics sub-report instead.

### Throughput — the real risk

≈ 1.45 M generation calls. Each call is cheap **only if the participant's
transcript prefix is still in the KV cache**; a cache miss costs a full ~5 k
token prefill, and 1.45 M full prefills is not feasible on one 4090.

Qwen2.5-7B in bf16 leaves roughly 7–8 GB for KV cache after weights. With GQA
(4 KV heads × 128 dim, 28 layers) that is ~56 KB/token → order 130 k tokens of
cache → only ~20–25 concurrent 6 k-token transcripts. So the driver processes
participants in **groups sized to the measured cache capacity**, walking a whole
group slot-by-slot to completion before starting the next group:

```
for group in chunks(profiles, G):          # G from the measured cache size
    for step in range(max_slots):
        prompts = [transcript_so_far(p) for p in group if p.has_slot(step)]
        outs = llm.generate(prompts, params_for(step))
        ...parse, retry illegal, append...
```

Mitigations if the measured capacity is too small: `kv_cache_dtype="fp8"`
(≈ 2× the cache, weights untouched, negligible effect on sampled text), and
capping `max_model_len` to the longest actual transcript.

**A calibration run is step 0 of this phase**: 64 participants, end to end,
measured — cache blocks, tokens/s, legality rate per slot type, projected
wall-clock. If the projection exceeds ~24 h the plan pauses and comes back with
options (block-mode generation for matrix questions, fp8 cache, reduced N to the
benchmark minimum of 500/intervention + 1,000 control = 9,000).

### Robustness

- **Checkpointing**: append-only `draws.jsonl` + a per-condition done-set; the
  driver is restartable and skips completed participants. Required — the shell
  caps a single command at 6 h and the run will not fit in one.
- **Orphan engines**: a killed driver leaves a `VLLM::EngineCore` holding 15 GB
  (observed during environment probing). The runner traps exit and tears the
  engine down; the launch script refuses to start if the GPU is not clean.
- **Validation before the big run**: every template renders; every manifest slot
  is reachable; slot ids are unique per condition; the 13 outcomes are
  reconstructible from item slots; conditional branches produce the expected
  slot counts.

### samples.csv

One row per participant: `profile_id`, `condition`, `condition_code`,
`control_text`, the pre-filled demographics, every generated slot answer
(~85 columns, raw as sampled), the 13 derived outcomes, and per-participant
sampling diagnostics (total rejected draws, fallbacks used). `tier1_submission.csv`
is the same data reduced to the benchmark's Tier-1 columns.

Reverse-codings and recodes applied when building composites (documented in the
analysis report): `funding_perceptions` reversed so higher = supports more
funding; the epistemic-autonomy reverse item; the behaviour items' "I already
do this" options coded to the scale maximum, with a sensitivity check that
treats them as missing instead.

---

## Part 3 — Data analysis

Written as if reporting the real study, on `samples.csv`. Everything with
numpy/scipy/pandas/matplotlib; OLS with HC1 robust standard errors implemented
in-package (`silicon_sampling/analysis/ols.py`) rather than adding statsmodels.
Plots follow the `dataviz` skill.

### Reports

```
docs/reports/pfander_silicon_sample/
  README.md          main report — core results, links out
  01_effects.md      intervention effects
  02_demographics.md age / gender / race and the other moderators
  03_distributions.md response distributions and scale properties
  04_diagnostics.md  what the sampler did (rejection rates, degenerate responding)
  plots/
```

### 01 — Effects (the main result)

- ATE of each of the 16 interventions vs control, for each of the 13 outcomes:
  OLS `outcome ~ condition`, HC1 SEs, 95% CI; Cohen's *d*; and the benchmark's
  own unit, **percentage points of the outcome's scale range**.
- 208 effects → report raw and Holm/BH-adjusted significance.
- Plots: forest plot of the primary outcome (`trust_multidimensional`, 16 arms
  ordered by effect); 16 × 13 heatmap in pp of scale range; per-subscale
  breakdown of the primary outcome (competence / integrity / benevolence /
  openness).
- Descriptive companion: which outcomes move at all, and the spread of effects
  across messages — the two quantities the benchmark's scoring separates.

### 02 — Do age / gender / race matter?

- Control-condition demographic baselines: cell means per outcome for each of
  the six moderators, with CIs.
- Moderation: saturated `outcome ~ condition * moderator` for each moderator ×
  outcome; report the interaction estimates and subgroup ATEs for the primary
  outcome; test whether the moderator adds anything (joint F on the interaction
  block).
- **Stereotyping diagnostic** (the benchmark scores it): R² of
  `outcome ~ moderator + condition` per moderator — how much of the synthetic
  variance is demographics alone — plus the demographic parity gap
  (worst-vs-best cell) per outcome.
- Marginal check on the three *generated* demographics (education, income,
  party) against published US distributions — a direct read on whether the base
  model produces a plausible population when it is not told what to be.
- Plots: subgroup effect plot; baseline means by moderator; R² bar chart.

### 03 — General statistics and variance

- Per outcome and per item: mean, SD, median, IQR, skew, min/max.
- Degeneracy diagnostics: share of responses at 0 / 50 / 100, share at
  multiples of 5 and 10, modal-response share, per-participant response SD
  (straight-lining), and the SD of each item — a synthetic sample that collapses
  to a point is the characteristic failure mode here.
- Scale properties: Cronbach's α and inter-item correlations for the six
  multi-item scales; correlation matrix across the 13 outcomes;
  pre- vs post-treatment correlation for the two items measured twice
  (`belief`, single-item trust) as an internal consistency check.
- Position effects: does answer distribution drift with slot index within a
  matrix (an artefact the transcript format could induce)?
- Plots: histogram grid over the 13 outcomes; item-level violin/strip;
  correlation heatmap.

### 04 — Sampling diagnostics

Rejection rate and mean attempts by slot type and by position; constrained-
decoding fallbacks; wall-clock and tokens/s; the free-text answers (attention
check 2 is pre-filled, but the final comment box is generated — worth reading a
sample of them).

---

## Package layout

```
silicon_sampling/
  survey/            reusable: slot types, legal-value grammars, parse+validate,
                     template renderer, manifest I/O
  sampling/          reusable: vLLM driver, group scheduler, rejection sampler,
                     checkpointing, run metadata
  analysis/          reusable: ols.py (HC1), effects.py, moderators.py,
                     distributions.py, plotting helpers
  pfander/           study-specific: questionnaire content, the 17 conditions,
                     profile construction (IPF), CLI
```

`silicon_sampling/vlasceanu/` is left untouched — its plan
([2026-08-13](2026-08-13-vlasceanu-text-survey.md)) is still ACTIVE and its
renderer unwritten; folding it into the new shared modules is a separate job.
The new `survey/` element types overlap with `vlasceanu/elements.py` by design
and generalise it (7-point Likert matrices, per-item escape options, embedded
in-treatment slots, conditional branches).

Entry points, all resumable:

```
python -m silicon_sampling.pfander.cli render-templates
python -m silicon_sampling.pfander.cli build-profiles --n 18000 --seed 20260814
python -m silicon_sampling.pfander.cli sample --group-size auto [--limit 64]
python -m silicon_sampling.pfander.cli build-csv
python -m silicon_sampling.pfander.cli analyse
```

## Open questions and assumptions

Resolved once `/opt/silicon-sample-submission` was available:

- ~~No codebook~~ — present. Slot ids are the codebook's `qualtrics_label`;
  output columns are `target_label`; conditions and moderator levels come from
  `submission_spec.R`. Composites use the codebook's own construction rules
  (note `trust_multidimensional` is the mean of the four *subscale means*, and
  `funding_perceptions = 100 − funding_5`).
- ~~Race is select-all-that-apply~~ — it is single-select ("which race /
  ethnicity you **most** identify as"), so the quota categories map 1:1.
- ~~Behaviour-item escape options~~ — the scored instrument has none; the six
  behaviour items are plain 0–100 sliders. No recode needed.
- ~~Control assignment rule~~ — confirmed by three separate code names
  (`control neckties` / `control baseball` / `control dances`) all mapping to
  the single label `control`.

Still assumptions:

1. **Page boundaries are reconstructed** from the `—page break—` markers and
   block structure; `survey.qsf` is available as a cross-check if a boundary
   turns out to matter.
2. **Gender is Male/Female**, matching the quota table; the submission schema
   allows "Other" but the quotas do not constrain it.
3. **Post-treatment block order is randomised per participant** (the instrument
   randomises the secondary/tertiary blocks; the primary trust battery is always
   first). Reproduced, with the drawn order recorded per participant.
4. **Zip code and the #16 state question are both generated**, in that order, so
   the model sees its own zip before naming a state. They may still disagree;
   the state answer drives #16's case assignment, and the disagreement rate is
   reported in 04.
5. **Sampling faithfulness cannot be validated against ground truth** — no human
   data is available to any entrant by design. The template format is therefore
   chosen on measurable proxies (legality rate, non-degeneracy, plausibility of
   generated demographic marginals vs. published US distributions), on a small
   pilot, before the full run. The comparison is reported in 04.

## Steps

**Phase 1 — templates**
1. [x] Instrument as structured Python, verbatim from the benchmark's own
   `questionnaire.txt`; 15 prose stimuli parsed from it, the 4 with embedded
   response slots structured by hand.
2. [x] `silicon_sampling/survey/` — element types, slot grammars, renderer,
   session walker, manifest emitter.
3. [x] `data/pfander/text_templates/` — 17 `.txt`, `manifest.json`, `00_FORMAT.md`.
4. [x] Validation: slot ids unique, all 13 outcomes reconstructible from every
   condition, no marker ever reaches a prompt, moderator levels match the codebook.

**Phase 2 — sampling**
5. [x] 18,000 profiles from the quota margins; both margins reproduce to ±2.
6. [x] vLLM driver: group scheduler, rejection sampler, checkpointing, teardown.
7. [x] **Calibration** — twice, once per vLLM version; see the tables above.
8. [~] **Format pilot — dropped, deliberately.** Its purpose was to *choose*
   between transcript formats on legality, non-degeneracy and demographic
   plausibility. Two things killed it: the single format already meets those
   criteria, and selecting a format on the plausibility of *unscored*
   demographics means tuning the instrument against my own priors about what US
   party ID should look like, with no ground truth to adjudicate. The format is
   chosen on stated design reasoning (`00_FORMAT.md`) and the diagnostics are
   **reported rather than selected on**.
9. [x] Full run: **18,000 respondents in 803.8 min (13.4 h)**, 1,344/h average,
   1.74% of draws rejected, 350 constrained-decoding fallbacks, 0 forced defaults.
10. [x] `samples.csv` (18,000 × 124), `tier1_submission.csv` (18,000 × 33, column
    order identical to the benchmark's example), `run_meta.json`. Validated:
    schema, value ranges, codebook moderator levels, composite reconstruction and
    quota reproduction all check out.

**Phase 3 — analysis**
11. [x] `silicon_sampling/analysis/` — OLS with HC1 (checked against a
    hand-computed unequal-variance SE), effects, moderators, distributions,
    plotting on a validated palette.
12. [x] Computed; 12 plots generated.
13. [x] `docs/reports/pfander_silicon_sample/` — main report plus four
    sub-reports.

**Wrap-up**
14. [x] `black .`; `flake8` clean; 14 checks in `tests/test_pfander.py` pass.
15. [x] Tasks moved to `docs/tasks/archive/`; this plan moved to
    `docs/plans/archive/`; committed and pushed.

## Result

Headline numbers are in [the report](../reports/pfander_silicon_sample/README.md).
The three that matter:

- **Effects are plausible and well-powered.** Control mean 53.4 on the primary
  outcome; the 16 interventions run from −1.95 to +8.47 scale points, 15 of 16
  positive, 57 of 208 effects surviving Holm correction. Real between-message
  variation on the primary outcome is 2.41 pp after removing sampling noise.
- **The item-level psychometrics are genuinely survey-like.** Cronbach's α from
  0.72 to 0.92 across the nine multi-item batteries, sensible inter-item
  correlations, 3–13% flat profiles, no degenerate piling on 0/50/100.
- **The respondents have no demographics.** Synthetic Republicans and Democrats
  differ by 1.1 points on climate belief, where the real partisan gap is tens of
  points; the largest variance any of the six moderators explains beyond
  condition is R² = 0.002. The model writes a coherent individual and then writes
  nearly the same individual every time. This is the opposite of the stereotyping
  failure the benchmark screens for, and for a submission scored on subgroup
  heterogeneity it is the more damaging one — the ATEs stay usable, every
  subgroup and demographic-baseline estimate does not.

Worth a follow-up if this is taken further: the flatness is the thing to attack,
and the obvious lever is the one deliberately left alone here — the pre-filled
set. Age, gender and race were pre-filled because the task fixed that scope;
pre-filling party, income and education from a joint distribution, so the model
reads a *committed* identity rather than one it invented a page earlier, is the
natural next experiment.
