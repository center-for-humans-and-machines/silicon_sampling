# What to expect on Pfänder

[← back to the final report](README.md) · basis:
[four-study cross-validation](four_study_cross_validation.md) ·
[three-study version](cross_validation.md) · [the recipe](the_recipe.md)

**Revised 2026-08-28** against four folds rather than three. The additions are
Voelkel et al. 2026 (CCC) and Muse-Glimmer-30B; the headline `pearson_r`
prediction moves **0.41 → 0.35**, entirely because CCC is a much harder study
than the original three and not because anything got worse.

## Predictions

Point prediction per scored metric for the **primary** entry, with a plausible
range. Target column says which direction is good.

### Section 1 — intervention effect recovery

| metric | target | **prediction** | range | human replication, for scale |
| --- | --- | --- | --- | --- |
| `pearson_r` (leaderboard sort key) | high | **0.35** | 0.15 – 0.55 | 0.542 |
| `spearman_rho` | high | **0.37** | 0.18 – 0.55 | 0.478 |
| `directional_pct` | high | **70** | 58 – 80 | 81.1 |
| `pearson_within` | high | **0.32** | 0.20 – 0.45 | 0.278 |
| `pearson_adj` | high | **0.54** | 0.30 – 0.75 | 0.855 |
| `rmse` | low | **2.8** | 1.4 – 4.6 | 2.667 |
| `rmse_adj` | low | **2.0** | 0.8 – 3.5 | 1.815 |
| `alpha` | 0 | **1.3** | 0.0 – 2.6 | 0.656 |
| `beta` | 1 | **1.08** | 0.55 – 1.60 | 0.486 |

### Section 2 — subgroup (condition × moderator) effects

| metric | target | **prediction** | range | human replication |
| --- | --- | --- | --- | --- |
| subgroup `pearson_r` | high | **0.02** | −0.06 – 0.10 | 0.146 |
| subgroup `spearman_rho` | high | **0.02** | −0.06 – 0.10 | — |
| subgroup `directional_pct` | high | **51** | 47 – 56 | — |

### Section 3 — response distributions (control arm)

| metric | target | **prediction** | range | human replication |
| --- | --- | --- | --- | --- |
| `variance_ratio` | 1 | **1.02** | 0.85 – 1.20 | 0.993 |
| `ovl` | 1 | **0.71** | 0.62 – 0.82 | 0.925 |
| `ks` | 0 | **0.23** | 0.13 – 0.34 | 0.018 |
| `w1` | 0 | **5.0** | 2.0 – 10.0 | 0.539 |

### Sections 10–12 — demographic baselines, parity, stereotyping

| metric | target | **prediction** | range | human replication |
| --- | --- | --- | --- | --- |
| `baseline_rmse` | low | **3.0** | 1.5 – 6.0 | 0.371 |
| `parity_dpd` | low | **3.5** | 2.0 – 6.5 | 0.486 |
| `parity_worst` | low | **6.5** | 4.0 – 11.0 | 4.372 |
| `stereo_coef_rmse` | low | **3.3** | 2.0 – 5.5 | 0.845 |
| `stereo_r2_gap` | 0 | **−0.005** | −0.03 – 0.02 | 0.003 |

### The eight anchored outcomes

Eight of thirteen outcomes now have an external control-arm level: three from
TISP at `near` grade, five from CCC, two of those on **verbatim identical items**.
The pinning is arithmetic, so the residual error on those eight is the anchor's
own transfer error rather than the model's — call it 2–5 pp. The remaining five
(`distrust_post`, `funding_perceptions`, `inst_trust_mean`, `donation_ams`,
`newsletter_signup`) carry the unanchored error, which the CCC hold-out measures
at **10–14 pp**. Weighted over thirteen outcomes that is a mean level error near
**6.4 pp**, against 11.8 pp when only three were anchored.

---

## How each number was obtained

**Section 1** comes from the four-study nested cross-validation, and **not** from
its best-looking row.

The tempting basis is the variant with membership and the within factor both
fixed, which scores mean held-out r of 0.344. That is not admissible as an
out-of-sample number for Pfänder: `within = 0.5` was chosen by measuring on
Voelkel, and both "average the two Qwens" and "exclude V4-Flash" were decided by
looking at the reference studies. Those choices are pre-committed relative to the
*folds* but not relative to *the data as a whole*, which is what matters when the
target is a further study.

Decomposing which informed choice is load-bearing, now over four folds:

| variant | 4 folds | *(3 folds)* |
| --- | --- | --- |
| both membership and within fitted per fold | **0.247** | *0.325* |
| **membership fixed by prior, within fitted** | **0.330** | *0.385* |
| membership fitted, within fixed at 0.5 | 0.263 | *0.390* |
| both fixed | 0.344 | *0.426* |
| *single Qwen2.5-7B, no calibration* | *0.303* | *0.349* |

The picture changed with the fourth fold. On three studies the two choices
recovered about +0.06 each and looked additive; on four, **the membership prior is
worth +0.083 and the within factor only +0.014**. Fixing the within factor while
still fitting membership (0.263) barely beats fitting both. What the recipe gets
from being told its structure is almost entirely "average these models", not
"shrink by this much".

The basis used here is the **second row, 0.330**, on the same reading as before:
the *form* of each choice is a genuine prior while its *specifics* are not.
Averaging comparable estimators is standard variance reduction and shrinking
toward a group mean is standard empirical Bayes, but which models and what factor
both needed data. That grants the forms and charges the magnitudes.

One adjustment follows, and it is not leakage: the reference studies have one run
per model where the submission averages seven, raising measured ensemble
reliability from 0.870 to 0.957. That was measured on Pfänder's own seed
replicates with no reference to the four studies. Correlations scale as the square
root of reliability, so `pearson_r`, `spearman_rho`, `pearson_within` and
`pearson_adj` are multiplied by **√(0.957/0.870) = 1.049**, giving 0.346, 0.372,
0.321 and 0.544.

*(That multiplier was 1.053 in the previous revision, from a claimed reliability
of 0.964 over eight runs. One of those eight,* `qwen25_72b_seed3`*, turned out to
be the same draw as* `qwen25_72b_seed2` *— it was sampled against a copy of its
profile file — so it is no longer counted as a seed. See the*
[four-study report](four_study_cross_validation.md#4-the-eighth-run-was-not-a-run).)

`rmse`, `alpha` and `beta` are taken from the **both-fixed** row rather than the
basis row, because unlike the correlations they are set by the shrinkage
arithmetic, and the shipped entry really does use `within = 0.5` rather than a
per-fold fitted value. That matters most for β: 1.079 fixed against 1.242 fitted.
All three still depend on the size and mean of the study's true effects, which for
Pfänder is unknown — the folds ran 1.33 to 4.70 on RMSE and 0.04 to 2.16 on α
purely because real effect magnitudes differ several-fold between these studies.
Those ranges are wide for that reason, not because of anything about the model.

**The ranges are set by the fold spread, and it is enormous.** The basis recipe
scored 0.084, 0.290, 0.343 and 0.669 on CCC, Goldwert, ICPC and Voelkel after the
reliability adjustment. A formal prediction interval on four points spans
essentially the whole unit interval and would be useless; the ranges quoted above
are a judgment that trims the tails, and the honest summary is that **which study
Pfänder resembles matters far more than anything in the recipe.**

**Section 3** now rests on a direct measurement rather than an adjusted fold mean,
because CCC supplies real human distributions for outcomes whose items Pfänder
reuses. Grading the built entry's control arm against CCC's real control arm on
those five:

| | level err (pp) | sd ratio | OVL | KS | W1 |
| --- | --- | --- | --- | --- | --- |
| built entry, five anchored outcomes | 0.00 | 1.027 | **0.838** | **0.066** | **3.10** |
| *CCC human half against half* | *0.60* | *0.998* | *0.852* | *0.027* | *0.99* |
| built entry, **un**anchored (CCC hold-out) | 10–14 | 0.93–1.25 | 0.66–0.68 | 0.20–0.28 | 11–15 |

The level and sd rows are circular — those are the two moments the anchor sets —
but OVL, KS and W1 are not: matching a mean and a variance does not force the rest
of a distribution to line up, and here it very nearly does. **OVL 0.838 against a
human ceiling of 0.852** is the best distributional result in the project.

It is still an optimistic bound, for a reason the quantile-mapping experiment
measured. Here the anchor and the grader are the same distribution. On Pfänder
the anchor is *borrowed* from CCC and TISP and graded against Pfänder's own
respondents, and moment-matching to a borrowed reference reached OVL 0.725 rather
than the 0.883 an own-study reference gives. So the anchored outcomes are
predicted near **0.73–0.75**, the five unanchored ones near **0.67**, and the
thirteen-outcome mean at **0.71**. `variance_ratio` stays at 1.02, from the built
entry's measured dispersion ratio of 1.011.

**Section 2** does not come from the cross-validation, which did not compute
subgroup interactions. It comes from a single earlier measurement on Voelkel over
756 condition × moderator pairs: Qwen scored −0.041, V4-Flash +0.001, a human
replication +0.146. Both models are at or below zero, and nothing in the recipe
addresses subgroup interactions, so the prediction is "near zero" with a range
that includes negative values. **This is the weakest row in the report.**

**Sections 10–12** come from the Voelkel bake-off, which is in-sample and
single-study, adjusted for two Pfänder-specific improvements the bake-off predates
— quota demographics, and party offsets blended to external gaps, which cut
party-gap error against external estimates from 19.1 pp to 3.8. The direction is
sound; the magnitudes are the least trustworthy numbers here after Section 2.

## What would make these wrong

**Between-study variation dominates everything else, and the fourth fold made
that worse, not better.** The basis recipe scored 0.084, 0.290, 0.343 and 0.669
on CCC, Goldwert, ICPC and Voelkel — an eightfold spread on the same recipe. That
is larger than every adjustment applied above put together, and it is the whole
uncertainty. Pfänder is a different topic (trust in climate science rather than
democratic norms, climate advocacy or climate framing), a different arm count and
a different year from all four.

The single most useful diagnostic is **how big Pfänder's true arm effects are**,
because that is what set the fold ordering. CCC's effects average 1.46 pp against
a median standard error of 1.00, a signal-to-noise ratio of 0.61, and even its own
participants can only predict themselves at r = 0.372. ICPC's effects average 4.93
pp and its ceiling is 0.640. Pfänder's design — 1,000 per treatment arm against a
2,000 control — gives standard errors about 9% *larger* than CCC's, so if its
interventions move trust by 1–2 pp, expect the bottom of every range here.

**The selection question sets the floor, and it is a real floor.** If none of the
design transfers — if the right simulation is the fully automated one —
`pearson_r` lands near **0.26** rather than 0.35. That is *below* an uncalibrated
Qwen2.5-7B at 0.303, which is a change from the three-study picture where every
variant still beat it. On four folds the fully-fitted recipe is actively worse
than submitting a single raw model, and only the membership prior rescues it.

What argues against the floor is that the design is insensitive inside a wide
band: with membership fixed, within 0.3, 0.5 and 1.0 score 0.341, 0.344 and 0.331.
What damaged the automated variant was fitting sharply per fold — on the Goldwert
fold it chose a membership and factor that generalised at 0.136.

**Four metrics rest on anchors covering eight of thirteen outcomes.** The level
anchors, the dispersion targets and hence the residual scale come from TISP's
three `near`-grade outcomes and CCC's five. If Pfänder's remaining five —
`distrust_post`, `funding_perceptions`, `inst_trust_mean`, `donation_ams`,
`newsletter_signup` — behave unlike the anchored eight, `variance_ratio`, `ovl`,
`ks` and `w1` all move together and in the same direction. The CCC hold-out puts a
number on what unanchored looks like: level error 10–14 pp and OVL 0.66–0.68,
against 0.00 and 0.838 anchored.

**`alpha` is the metric most likely to embarrass this table.** It is an intercept,
and no component of the recipe supplies one. On the two folds where nearly every
arm pushed the same direction it came out above 2.0 against a human replication's
0.02–1.9. If Pfänder's interventions are similarly one-directional — plausible,
since they are all pro-trust messages — expect the upper end.

**Nothing here is a prediction about being right.** Every number is a prediction
about a *score*, and several of the scores reward smoothness rather than accuracy
at this sample size. RMSE beating the human replication in cross-validation
(2.893 against 2.956) is the clearest example: the recipe is not better than real
people at predicting real people, it is less noisy than a half sample.

**Confidence, stated plainly.** Section 1 rests on a nested cross-validation and
is the part I would defend. Section 3 rests on external anchors for three
outcomes. Section 2 rests on one measurement in one study. Sections 10–12 rest on
in-sample numbers with a directional correction. Treat the four blocks as
descending in trustworthiness in exactly that order.
