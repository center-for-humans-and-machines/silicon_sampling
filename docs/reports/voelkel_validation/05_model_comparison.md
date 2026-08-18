# Four questions about DeepSeek-V4-Flash against Qwen2.5-7B

[← main report](README.md)

The same 6,203 respondent profiles, sampled twice — once with Qwen2.5-7B and once with DeepSeek-V4-Flash, about a 40x jump in
parameters — and both scored against the same real responses. Same seeds, so
the two samples are paired respondent by respondent, which is what lets the
difference between them carry an interval rather than just a sign.

Respondents: Qwen2.5-7B: 6,203 · DeepSeek-V4-Flash: 6,203.

**Everything here is read against the human replication**, not against a
perfect score. Human 2 predicting Human 1 is what a fresh sample of this size
achieves, and it is the ceiling the pipeline is chasing — not 1.0.

## The answers, in short

| question | answer |
| --- | --- |
| **1.** Over- or underestimating effect sizes? | **Both overestimate.** Qwen2.5-7B spreads its effects 3.5x too wide, DeepSeek-V4-Flash 2.3x. Scaling up helped here. |
| **2.** Right rank order and direction? | **Qwen2.5-7B partly; DeepSeek-V4-Flash worse.** Rank correlation 0.31 against 0.19, with a real replication at 0.40. |
| **3.** Do predictions change with demographics? | **Yes, both.** Real partisans differ by 3.0 points; Qwen2.5-7B produces 3.9 and DeepSeek-V4-Flash 9.8. |
| **4.** Using demographics to their advantage? | **No, neither.** The moderators a model could read predict its subgroup effects no better than the ones it could not. |

And one precondition before any of them: the bigger model's answers sit far
closer to where real answers sit — mean absolute level error 22.9 -> 8.0 points on a 0-100 scale.

In one line: **scaling the base model about 40x bought a much more realistic
sample and no better prediction of which interventions work.**

---

## First, a precondition: are the answers even in the right place?

Treatment effects are differences, so a constant bias cancels out of them and
none of the four questions below would notice if every synthetic respondent
sat 40 points off. Raw response distributions have no such mercy.

**Mean absolute level error fell from 22.9 to 8.0 points on a 0-100 scale.**
This is the largest single change between the two models, and it is not
marginal: on opposition to bipartisan cooperation the smaller model answered
83 where real Americans answered 21, and the bigger one answers 37. On support
for undemocratic candidates the smaller model said 18 against a real 53; the
bigger one says 56.

| model | outcome | mean_human | mean_synthetic | level_error | variance_ratio | ovl | w1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | ADA | 27.48 | 23.10 | 4.37 | 1.27 | 0.71 | 5.99 |
| Qwen2.5-7B | ADA | 27.48 | 17.29 | 10.19 | 1.09 | 0.56 | 10.88 |
| DeepSeek-V4-Flash | BEPF | 52.05 | 45.24 | 6.81 | 1.18 | 0.85 | 6.97 |
| Qwen2.5-7B | BEPF | 52.05 | 33.97 | 18.09 | 1.38 | 0.65 | 18.14 |
| DeepSeek-V4-Flash | Composite | 39.22 | 44.77 | 5.55 | 0.88 | 0.80 | 5.61 |
| Qwen2.5-7B | Composite | 39.22 | 47.26 | 8.04 | 0.49 | 0.62 | 8.49 |
| DeepSeek-V4-Flash | OppBip | 21.13 | 36.71 | 15.57 | 2.31 | 0.68 | 15.55 |
| Qwen2.5-7B | OppBip | 21.13 | 82.98 | 61.85 | 1.12 | 0.21 | 61.72 |
| DeepSeek-V4-Flash | PA | 65.81 | 66.34 | 0.54 | 1.41 | 0.81 | 4.18 |
| Qwen2.5-7B | PA | 65.81 | 62.46 | 3.35 | 1.44 | 0.80 | 4.82 |
| DeepSeek-V4-Flash | SPV | 10.79 | 15.10 | 4.31 | 1.31 | 0.61 | 4.36 |
| Qwen2.5-7B | SPV | 10.79 | 14.02 | 3.22 | 1.15 | 0.64 | 3.82 |
| DeepSeek-V4-Flash | SUC | 53.28 | 55.64 | 2.36 | 2.12 | 0.63 | 12.82 |
| Qwen2.5-7B | SUC | 53.28 | 18.06 | 35.22 | 1.22 | 0.37 | 35.16 |
| DeepSeek-V4-Flash | SocDis | 30.61 | 60.80 | 30.19 | 1.29 | 0.56 | 30.14 |
| Qwen2.5-7B | SocDis | 30.61 | 75.39 | 44.78 | 1.05 | 0.42 | 44.69 |
| DeepSeek-V4-Flash | SocDistrust | 52.61 | 55.20 | 2.59 | 1.22 | 0.83 | 4.56 |
| Qwen2.5-7B | SocDistrust | 52.61 | 73.96 | 21.35 | 1.19 | 0.57 | 21.64 |

`ovl` is the overlap between the two distributions (1 is identical) and `w1`
the Wasserstein distance in scale points. Both improve on eight of nine
outcomes.

![Level error by outcome](plots/05_level_error.png)

---

## 1. Are the models over- or underestimating effect sizes?

**Both overestimate, and the bigger model overestimates less.**

| submission | mean_abs_effect | sd_estimate | ratio_vs_human1 | ratio_vs_true |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 2.787 | 3.978 | 2.566 | 3.537 |
| DeepSeek-V4-Flash | 1.930 | 2.635 | 1.700 | 2.343 |
| Human replication (Human 2) | 1.366 | 1.826 | 1.178 | 1.624 |
| Human 1 (the reference itself) | 1.125 | 1.550 | 1.000 | 1.379 |

The spread of the real between-arm effects, once sampling noise is removed,
is **1.12 percentage points** — these interventions genuinely do very
little. Against that:

- Qwen2.5-7B spreads its effects **3.5x** too wide
- DeepSeek-V4-Flash spreads them **2.3x** too wide
- even a real human replication looks **1.6x** too wide, because its own estimates carry sampling noise

So the direction of the error is the same for both models — they make these
messages look far more consequential than they are — and scaling up cut the
exaggeration by roughly a third.

**Why the calibration slope does not show this.** `beta` in the leaderboard is
0.159 for Qwen2.5-7B and 0.112 for DeepSeek-V4-Flash — it got *worse*, which looks
like a contradiction. It is not: beta is `cov(human, predicted) / var(predicted)`,
so it moves with alignment as well as with magnitude. The bigger model's effects
shrank toward the right size *and* became less aligned with the real ones, and
the second effect is the larger. Which is question 2.

---

## 2. Do they predict the right rank order and direction?

**Qwen2.5-7B gets part of the ordering right. DeepSeek-V4-Flash does worse.**

| submission | directional_pct | spearman_rho | pearson_r | pearson_adj |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 61.111 | 0.311 | 0.408 | 0.563 |
| DeepSeek-V4-Flash | 55.556 | 0.186 | 0.190 | 0.262 |
| Human replication (Human 2) | 66.667 | 0.395 | 0.514 | 0.709 |
| Baseline: no effect | 50.000 |  |  |  |
| Baseline: all positive | 55.556 |  |  |  |

`spearman_rho` is the rank-order question in its purest form and
`directional_pct` the sign question, with 50% as the no-information floor.

- rank order: 0.31 for Qwen2.5-7B, 0.19 for DeepSeek-V4-Flash, against 0.40 for a real replication
- direction: 61% and 56%, against 67% and a floor of 50%

DeepSeek-V4-Flash's directional agreement, 55.6%, is the same as
the "predict every effect positive" baseline. That is the sharpest way to put
it: on which interventions help, the bigger model carries about as much
information as a constant guess.

![Both models against human effects](plots/05_models_vs_human.png)

The dashed lines remove each predictor's own sampling noise from the slope,
which matters because attenuation depends only on the x-axis and each
submission has a different amount of it. Read those, not the solid ones, when
comparing submissions by eye.

---

## 3. Do the models change their predictions based on demographics?

**Yes, both do — and the bigger model varies far more than real people.**

This question is about whether the demographics move the answers at all,
regardless of whether they move them correctly. The cleanest measure is the
Republican-minus-Democrat gap in the control arm, before any intervention:
party is named in every question of this instrument, so a model that ignores
its assigned identity has no excuse for a flat gap.

| outcome | human | Qwen2.5-7B | DeepSeek-V4-Flash |
| --- | --- | --- | --- |
| PA | 0.6 | -8.3 | 3.5 |
| ADA | 4.2 | -0.6 | 18.8 |
| SPV | -1.2 | 0.7 | 9.3 |
| SUC | 2.0 | -0.0 | 0.5 |
| OppBip | 6.3 | 1.3 | 17.0 |
| SocDistrust | 1.8 | 1.3 | 7.3 |
| SocDis | -9.1 | -2.4 | 9.5 |
| BEPF | 1.2 | -17.2 | 12.5 |
| Composite | 0.7 | -3.1 | 9.8 |

| model | mean_abs_gap | sd_gap | corr_with_human_gaps | n_outcomes |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 3.90 | 6.08 | 0.10 | 9 |
| DeepSeek-V4-Flash | 9.80 | 5.85 | 0.31 | 9 |
| human (Human 1) | 3.02 | 4.29 | 1.00 | 9 |

- real partisans differ by **3.0 points** on average
- Qwen2.5-7B produces **3.9** — about the right size
- DeepSeek-V4-Flash produces **9.8** — roughly three times too large

So neither model is demographically inert on this instrument. The failure is
in *how* they vary, not whether they vary — which is question 4.

---

## 4. Are they using demographics to their advantage?

**No. Neither model turns the demographics it was given into a better
prediction.** Two independent tests, and both come out the same way.

### Test one: do the partisan gaps point the right way?

Correlation between each model's nine party gaps and the real ones:
**0.10** for Qwen2.5-7B, **0.31** for DeepSeek-V4-Flash.

DeepSeek-V4-Flash is the better of the two here, but look at the signs in the table
above rather than the summary. Real respondents are *less* socially distant
from the out-party if they are Republican (-9.1); both models say the
opposite, Qwen2.5-7B mildly (-2.4 is at least the right sign) and DeepSeek-V4-Flash
confidently wrong (+9.5). Getting a gap of the right rough size pointed the
wrong way is not an advantage.

### Test two: do the moderators the model could *see* beat the ones it could not?

This is the decisive test, and it needs no assumption about what the right
answer is. Three moderators appear in the transcript — gender, race, party —
and two never do: age and education came from the panel supplier and were never
shown. A model that uses what it is told should predict subgroup effects better
for the first group than the second. Neither does:

| model | moderators | n_moderators | pearson_r | directional_pct |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | invisible | 2 | 0.237 | 53.472 |
| Qwen2.5-7B | visible | 3 | 0.236 | 56.852 |
| DeepSeek-V4-Flash | invisible | 2 | 0.079 | 50.694 |
| DeepSeek-V4-Flash | visible | 3 | 0.040 | 48.704 |

- Qwen2.5-7B: visible 0.236 against invisible 0.237 — no gap at all
- DeepSeek-V4-Flash: visible 0.040 against invisible 0.079 — visible does *worse*

If either model were reading its assigned identity to any useful effect, the
visible row would beat the invisible one. For the smaller model the two are
indistinguishable; for the bigger one the ordering is backwards. The
demographic variation in question 3 is real, and it is noise.

**The two models fail in opposite directions**, which is worth naming because
the benchmark's diagnostics are built to catch only one of them. A model
answering from a stereotype produces subgroup differences that are too large
and too clean; a model ignoring its assigned identity produces almost none.
DeepSeek-V4-Flash is the first kind and Qwen2.5-7B closer to the second, so moving
from one to the other is not simply progress.

---

## So did the bigger model win? The paired verdict

Both models answered the same instrument about the same interventions and are
scored against the same reference, so their errors move together. Each
bootstrap draw resamples one set of intervention clusters and rescores *both*,
which estimates the difference far more precisely than either score — and is
why this table, not the leaderboard, is the verdict.

| metric | Qwen2.5-7B | DeepSeek-V4-Flash | delta | delta_lo | delta_hi | p_contender_better |
| --- | --- | --- | --- | --- | --- | --- |
| directional_pct | 61.111 | 55.556 | -5.556 | -31.481 | 22.222 | 0.318 |
| pearson_r | 0.408 | 0.190 | -0.218 | -0.428 | 0.137 | 0.102 |
| pearson_adj | 0.563 | 0.262 | -0.301 | -0.569 | 0.276 | 0.101 |
| rmse | 3.620 | 2.808 | -0.812 | -1.711 | 0.220 | 0.944 |
| beta | 0.159 | 0.112 | -0.047 | -0.164 | 0.096 | 0.258 |

Clusters resampled: 6. `delta` is signed raw, so on `rmse` — the
one row where lower is better — a negative delta is the improvement.

Every row is unsettled — no metric's resamples agree on a direction — so there are no per-metric conclusions to list.

**Nothing clears its interval.** With six intervention clusters to resample,
this study cannot certify a difference of the size at stake, so the honest
reading of the effect metrics is "no improvement, possibly a regression",
not a demonstrated regression.

The one row that comes close is `rmse`, and it is the row most likely to be
misread. It improved because the exaggeration shrank (question 1), not
because the ordering improved (question 2) — squared error rewards a
predictor for moving toward the right scale even as its ranking degrades.
Note also that predicting **no effect at all** scores 1.537 on this metric,
better than either model: when the true effects are barely larger than the
noise, shrinking everything to zero is close to optimal.

![Paired change per metric](plots/05_paired_change.png)

---

## What this implies for the pipeline

The three things that improved — level accuracy, exaggeration, demographic
responsiveness — are all properties of *how a respondent answers in isolation*,
and those are exactly what a better language model should be expected to fix.
The thing that did not improve is the one the megastudy actually scores:
whether the sample can tell the interventions apart. That is a claim about
counterfactual sensitivity to a paragraph of text, and 40x more parameters
bought none of it.

So the next thing worth trying is probably not a bigger model. It is a change
to what the model is asked to do — conditioning it more strongly on the
stimulus, or abandoning single-pass simulation for something that reasons about
the message before answering.

---

## What the samplers did

| model | hf_id | respondents_per_hour | rejection_rate | structured_fallbacks | forced_defaults | gpus |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Qwen/Qwen2.5-7B | 1678.1000 | 0.0633 | 2689 | 0 | 1 |
| DeepSeek-V4-Flash | deepseek-ai/DeepSeek-V4-Flash-Base | 1205.1000 | 0.0508 | 195 | 0 | 4 |

## Caveats

1. **The two runs differ in KV-cache precision.** DeepSeek-V4-Flash requires
   `fp8_ds_mla` — on an H200 vLLM selects its FlashMLA attention, whose paged
   layout *is* the fp8 format, and it will not start with anything else. The
   checkpoint ships the UE8M0 scales, so this is the precision the model was
   built to run at rather than a compromise, but the Qwen run used bf16 KV and
   that asymmetry cannot be ruled out as a contributor.
2. **Six intervention clusters.** The pure-text rule left 6 of 27 arms, so every
   interval here is wide and the paired bootstrap resamples six things. That is
   why the verdict table settles nothing.
3. **A subset the paper never reports.** Dropping the non-textual arms means the
   human reference is not the study's headline result.
4. **2022 sits inside both models' training windows.** Neither result should be
   read as a clean out-of-sample prediction.

