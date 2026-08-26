# What to expect on Pfänder

[← back to the final report](README.md) · basis:
[nested cross-validation](cross_validation.md) · [the recipe](the_recipe.md)

## Predictions

Point prediction per scored metric for the **primary** entry, with a plausible
range. Target column says which direction is good.

### Section 1 — intervention effect recovery

| metric | target | **prediction** | range | human replication, for scale |
| --- | --- | --- | --- | --- |
| `pearson_r` (leaderboard sort key) | high | **0.44** | 0.30 – 0.55 | 0.599 |
| `spearman_rho` | high | **0.43** | 0.28 – 0.55 | 0.520 |
| `directional_pct` | high | **71** | 62 – 79 | 82.7 |
| `pearson_within` | high | **0.37** | 0.24 – 0.46 | 0.354 |
| `pearson_adj` | high | **0.70** | 0.48 – 0.85 | 0.872 |
| `rmse` | low | **2.9** | 1.5 – 4.5 | 2.956 |
| `rmse_adj` | low | **2.0** | 0.8 – 3.4 | 2.081 |
| `alpha` | 0 | **1.2** | 0.0 – 2.4 | 0.664 |
| `beta` | 1 | **0.95** | 0.55 – 1.40 | 0.514 |

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
| `ovl` | 1 | **0.72** | 0.60 – 0.82 | 0.925 |
| `ks` | 0 | **0.25** | 0.15 – 0.38 | 0.018 |
| `w1` | 0 | **4.0** | 1.5 – 8.0 | 0.539 |

### Sections 10–12 — demographic baselines, parity, stereotyping

| metric | target | **prediction** | range | human replication |
| --- | --- | --- | --- | --- |
| `baseline_rmse` | low | **3.0** | 1.5 – 6.0 | 0.371 |
| `parity_dpd` | low | **3.5** | 2.0 – 6.5 | 0.486 |
| `parity_worst` | low | **6.5** | 4.0 – 11.0 | 4.372 |
| `stereo_coef_rmse` | low | **3.3** | 2.0 – 5.5 | 0.845 |
| `stereo_r2_gap` | 0 | **−0.005** | −0.03 – 0.02 | 0.003 |

### The three anchored outcomes

Level error on `trust_multidimensional`, `trust_post` and `policy_role_mean` is
predicted at **2–5 pp**, not the 11.8 pp the cross-validation shows, because
those three are pinned to TISP exactly and the pinning is arithmetic rather than
estimated. The other ten carry the cross-validated error.

---

## How each number was obtained

**Section 1** comes directly from the nested cross-validation's fold means for
the shipped design, with one adjustment: the reference studies have one run per
model where the submission averages eight, raising measured ensemble reliability
from 0.870 to 0.964. Effect-recovery correlations scale as the square root of
reliability, so `pearson_r`, `spearman_rho`, `pearson_within` and `pearson_adj`
are multiplied by **√(0.964/0.870) = 1.053**. Nothing else is adjusted.

`rmse`, `alpha` and `beta` are carried across unadjusted. All three depend on the
size and the mean of the study's true effects, which for Pfänder is unknown — the
folds ran 1.28 to 4.51 on RMSE and 0.04 to 2.16 on α purely because real effect
magnitudes differ 4.5-fold between these studies. Those ranges are wide for that
reason and not because of anything about the model.

**Section 3** comes from the cross-validated donor and residual scale, then
corrected for the one thing Pfänder has that the folds do not: the residual scale
is fitted against TISP's dispersion on Pfänder's own questions, and the built
entry measures at a dispersion ratio of 1.011, so `variance_ratio` is predicted
at 1.011² ≈ 1.02 rather than the folds' 1.015 with their much larger spread. OVL,
KS and W1 are taken from the fold means and nudged toward the good end to reflect
three anchored outcomes.

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

**Between-study variation dominates everything else.** The same recipe scored
`pearson_r` of 0.342, 0.358 and 0.576 on the three held-out studies. That ±0.12
spread is larger than every adjustment applied above, and Pfänder is not drawn
from the same population as the three reference studies — it is a different topic
(trust in climate science rather than democratic norms or climate advocacy), a
different arm count, and a different year. If Pfänder is more like Voelkel the
result lands near the top of the range; if more like ICPC, the bottom.

**The selection question puts a floor under the prediction, not a ceiling.** A
fully automated version of this recipe, refitting every choice per fold, scores
0.325 rather than 0.426 — so if the pre-committed structure turns out not to
transfer, `pearson_r` lands near **0.34** instead of 0.44. The reason for
centring on the higher number is that within 0.3, 0.5 and 1.0 all score
0.398–0.428, so the design is insensitive to the choice inside a wide band; what
was harmful in cross-validation was fitting it sharply, not choosing it roughly.

**Three metrics rest on an extrapolation from three of thirteen outcomes.** The
level anchors, the dispersion target and hence the residual scale are all fitted
on the outcomes TISP grades `near`. If Pfänder's trust battery is unrepresentative
of its other outcomes, `variance_ratio`, `ovl`, `ks` and `w1` all move together
and in the same direction.

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
