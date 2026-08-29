# Building a synthetic sample of the Pfänder megastudy

**Method summary.** How we construct 18,000 synthetic survey respondents whose
treatment effects, response distributions and demographic structure are meant to
match a real megastudy that has not yet been unblinded.

---

## At a glance

| | |
| --- | --- |
| **Task** | Predict the results of a 16-intervention climate-trust megastudy before the human data are released |
| **Approach** | Behavioural cloning: base LLMs answer the real questionnaire as sampled respondents |
| **Submission** | 18,000 individual-level synthetic respondents × 33 columns |
| **Key design choice** | One respondent's answer is assembled from **four different models**, each supplying the one term it is best at |
| **Validated on** | 4 published megastudies, 52,652 real respondents, held out one at a time — **all four terms, including the structural donor** |
| **Expected score** | Pooled *r* ≈ **0.34** on effect recovery (95% CI ≈ ±0.13); a fresh half-sample of real humans scores 0.54 on the same task |

---

## 1. The prediction task

The megastudy fields **16 message interventions** intended to increase trust in
climate scientists, against a shared control, and measures **13 preregistered
outcomes** on a US sample of roughly 18,000 people. The outcome data are locked.

Submissions are scored by a preregistered pipeline. The human sample is split in
half on a fixed seed; every submission is scored against **Human 1**, and
**Human 2** is run through the identical pipeline to give a *human replication
reference* — how well a fresh sample of real people predicts real people. That
reference, not 1.0, is the yardstick.

Four families of metrics are scored, and they are close to independent of one
another:

1. **Treatment effects** — the 16 × 13 = 208 intervention × outcome effects. The
   leaderboard sorts on the pooled Pearson correlation between our effects and
   the humans'.
2. **Subgroup effects** — the same, split by six demographic moderators.
3. **Response distributions** — how the synthetic control arm's answers are
   *distributed*, not just where they average.
4. **Demographic calibration** — group means, parity across groups, and whether
   the model exaggerates demographic differences.

A submission can be excellent at (1) and terrible at (3). That fact is the whole
basis of the design.

## 2. The central design decision

A synthetic respondent's answer to one item is overwhelmingly *not* about which
intervention they saw. Measured on our own samples:

| outcome | total SD | between-arm SD | share of variance |
| --- | --- | --- | --- |
| trust (12-item battery) | 22.5 | 0.55 | 0.06% |
| trust (single item) | 28.3 | 0.50 | 0.03% |
| distrust | 31.1 | 0.46 | 0.02% |
| **mean** | | | **0.03%** |

**The condition effect is about one-fiftieth of one within-arm standard
deviation.** Where the distribution sits and how wide it is accounts for 99.97%
of the response — and the benchmark grades those two things *separately*, against
different references.

So we do not ask one model to do both. Each respondent's answer is assembled as

```
response  =  level  +  condition effect  +  demographic offset  +  residual
```

with each term taken from whichever model predicts *that term* best, and the
terms are separable in practice as well as in principle: changing the demographic
term moves 21 outcome columns and leaves the effect vector bit-identical
(*r* = 1.000000).

This matters because the models are bad in complementary ways. DeepSeek-V4-Flash
ranks interventions worse than chance (mean *r* = −0.04 across four validation
studies) yet places response levels closer to external benchmarks than anything
else (2.6 points of error against 6.1–7.7 for the alternatives). Qwen2.5-7B is
its mirror image: the best ranker, and the least plausible people (its trust
battery has a mean inter-item correlation of 0.47 where real respondents have
0.61). Neither model is good. Splitting the terms lets each contribute only what
it is good at.

## 3. The components

**Respondents.** Rows come from a run in which each synthetic respondent is
handed census-quota demographics rather than inventing their own. Left to
themselves the models produce 0.8% of respondents under \$30,000 where the real
population has 13.5%, which starves the demographic analyses of cases.

**Condition effects** — the term the leaderboard sorts on. We average the effect
vectors of **ten sampling runs across three models** (Qwen2.5-7B, Qwen2.5-72B,
Muse-Glimmer-30B), averaging *within* a model first and then across models, so
each model carries equal weight regardless of how many runs of it exist.
DeepSeek-V4-Flash is excluded from this term only.

Averaging is the single largest free improvement in the method, and it works
because the models' errors are close to independent: two runs of the same model
agree at *r* ≈ 0.84, two different models at *r* ≈ 0.55. Averaging decorrelated
estimates of the same signal raises the correlation with that signal. It pays for
*decorrelation*, not for the added model being good — Muse-Glimmer is the weakest
single model of the three and still improves the average.

Two shrinkage steps follow. Each intervention's effect is pulled halfway toward
its own outcome's mean effect (**within-outcome shrinkage, 0.5**), because we
predict *which outcomes move* far better than *which message moves them*. Then
every effect is multiplied by a global factor (**0.400**), which is provably
neutral on the correlation — a positive scalar cannot move a correlation — but is
most of the story for error magnitude and calibration slope.

**Levels**, i.e. where each outcome's control-arm distribution sits, come from
DeepSeek-V4-Flash. Eight of the thirteen outcomes are additionally **pinned
exactly to external anchors**: three from a public science-trust survey that asks
the same battery, five from a climate megastudy from which this study borrowed
items — two of them *verbatim identical*. On the anchored outcomes the residual
error is the anchor's transfer error rather than the model's.

**Demographic structure** comes from DeepSeek-V4-Flash, except **party**, which is
the one moderator this questionnaire does not tell the respondent — it is
*elicited* early in the survey, so the party structure in a sample is the model's
own consistency rather than a demographic it was asked to perform. Party offsets
therefore come from a different run, and are blended **70% toward externally
estimated Democrat–Republican gaps**. Unanchored, the models produce party gaps
roughly half the real size.

**Dispersion.** These models answer with more within-cell spread than real
participants, so residuals are rescaled by a fitted factor to match human
dispersion. This touches only within-cell spread; level, effects and demographic
offsets are separate terms and pass through untouched, so it can reach the
distributional metrics without moving the sort key.

### Summary

| term | source | graded against |
| --- | --- | --- |
| rows and demographics | Qwen2.5-7B, quota demographics | — |
| condition effect | 10 runs / 3 models, averaged, shrunk twice | human treatment effects |
| level | DeepSeek-V4-Flash; 8 of 13 pinned to external anchors | human control-arm means |
| demographic offset | DeepSeek-V4-Flash; party blended 70% to anchors | human group means, parity |
| residual | DeepSeek-V4-Flash, rescaled to human dispersion | distribution shape |

## 4. How the method was validated

Nothing about arm contrasts can be measured on the target study, which publishes
no participant responses. So every free choice is validated on **four published
megastudies** that do — 52,652 real respondents, 36 interventions, 283
intervention × outcome effects between them — using the same sampling pipeline
and the same scoring code.

**Leave-one-study-out, with the fitting nested.** For each held-out study, every
free choice — which runs to average, both shrinkage factors, which run supplies
the level and the dispersion — is fitted on the other three alone, and the
assembled recipe is scored once on the study it never saw. Earlier versions of
this project fitted parameters out-of-fold but chose the *structure* by looking at
all studies at once; nesting the fitting is what makes that selection visible.

**Averaged over half-splits.** Every variant is scored against the same half of
the human sample, so that half's sampling noise is common to all of them and one
draw can reorder variants that differ by less than it. Results are therefore
reported both for a single preregistered-style split — which is what the target
study will actually do — and averaged over eight. The split-to-split standard
deviation is 0.04–0.08, larger than most differences between design variants, and
an earlier conclusion of this project reversed once it was averaged.

**With the benchmark's own uncertainty interval**, a cluster bootstrap over
interventions, which is what every leaderboard row will carry.

### What the validation says

Effect recovery, averaged over four held-out studies and eight splits:

| | pooled *r* |
| --- | --- |
| **the method above** | **0.35** |
| membership decided by a rule per fold instead of fixed | 0.32 |
| the same without the third model | 0.31 |
| every free choice refitted per fold | 0.29 |
| best single model, uncalibrated | 0.26 |
| the structural donor, scored on effects | −0.00 |
| *human replication reference* | *0.54* |

The calibrated method beats every raw single model in 8 of 8 splits. The last two
rows are the design in miniature: the model that supplies three of the four terms
is worthless at the fourth, and the method works by never asking it.

**The structural half is now validated too**, which it was not before — the
donor had no sample on one of the four studies, so the fold search could not
consider it. It has one now, the search picks it on every fold, and the
distributional metrics roughly halve their distance to the human reference:

| control-arm distribution | fitted donor before | **now** | *human reference* |
| --- | --- | --- | --- |
| OVL | 0.663 | **0.784** | *0.859* |
| KS | 0.277 | **0.150** | *0.041* |
| W1 | 12.97 | **6.21** | *1.45* |
| variance ratio | 0.968 | **1.023** | *1.008* |
| demographic baseline RMSE | 11.36 | **6.04** | *2.03* |
| demographic parity gap | 7.35 | **3.43** | *1.66* |

Those earlier figures were not the method scoring badly; they were a different
donor being scored in its place.

## 5. Expected performance, and what dominates the uncertainty

**Pooled *r* ≈ 0.34**, with a 95% cluster-bootstrap interval of about **±0.13**.

One measured adjustment separates the validation number from the prediction. The
validation studies give one sampling run per model; the submission averages ten,
which makes its effect vector less noisy and its correlation correspondingly
higher. Both reliabilities are measured from the fits' own standard errors —
0.923 across the folds, 0.964 on the submission — and correlations scale as the
square root of the ratio, so the bridge is **×1.022**. It carries a basis of 0.32
to 0.33 and the fixed three-model row of 0.35 to 0.36.

**What dominates is neither.** The same method scores 0.01 on one validation study
and 0.46 on another. That eightfold spread is larger than every design choice in
this document put together, and it tracks one thing: how large a study's true
intervention effects are relative to the precision with which a half-sample can
measure them. On the hardest study, 79% of the observed variance in the human
reference is sampling noise, and even the humans only predict themselves at
*r* = 0.37.

Whether the target study looks like the easy end or the hard end of that range is
not knowable from its published protocol. It is the single most important
unknown, it applies to every entry on the leaderboard rather than only to this
one, and it is why the interval above is wide.

## 6. Known limitations

**Subgroup effects remain the weakest prediction.** Across four held-out studies
the method recovers condition × moderator interactions at *r* ≈ −0.01 — better
than the −0.08 measured before the structural donor could be validated, but still
no better than predicting no heterogeneity at all. The consolation is that the
human replication reference on the same task is only +0.07: at these cell sizes a
fresh human sample barely predicts the other half either.

**Two validation studies carry known instrument-rendering defects** that would
require re-sampling to fix — one where slider ranges were never shown to the
model, one where a fixed-budget allocation question was rendered without its
constraint. The second was repaired in analysis by renormalising to the budget
the real instrument enforced; the first cannot be repaired and is a reason to
distrust that study's level metrics.

**The dispersion rescale is the one constant the new evidence argues against.**
It is 1.12, fitted so the built entry's control-arm spread matches the external
dispersion targets, which it does almost exactly (ratio 1.008 against 0.941 at
1.00). But the donor's spread can now be graded directly against a held-out
study's humans, and there it is already right untouched (ratio 1.028) — applying
1.12 pushes it to 1.103 and costs OVL, KS and W1. Two external references
disagree because they are different instruments; the Pfänder-specific one is
kept, and this is the least secure constant in the recipe.

---

## Appendix — reproducibility

The submission is built by `scripts/build_entries.py`, which writes three
Tier-1 entries (the benchmark scores all three at no penalty): the method above;
the same without global shrinkage, the one axis where that step could hurt and
cannot help the sort key; and the uncalibrated best single ranker, as insurance
against every calibration being wrong at once.

Validation is reproduced by `scripts/nested_cv.py` (effect recovery, both split
readings) and `scripts/nested_benchmark.py` (all four metric families, running
the real recipe rather than a reimplementation of it). Full results in
[the verified cross-validation](four_study_cross_validation_verified.md);
defects found and fixed in [the audit](audit_findings.md).

**Changes from the previous version of this method**, all supported by the
re-derivation above: a third model joins the effect average (+0.03 expected on
the sort key); a noise-flooring step is removed, its two target outcomes having
been selected by a diagnostic run with a placeholder standard error; one external
anchor is corrected from a three-item composite to the single item the target
study actually reuses (level 68.0 → 65.9, dispersion 29.3 → 32.9, party gap
32.9 → 37.3); the party blend weight rises 0.5 → 0.7 on an out-of-sample
measurement; and the global shrinkage constant is corrected 0.413 → 0.383, having
been rescaled against the wrong reference quantity.
