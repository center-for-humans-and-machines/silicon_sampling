# Nested leave-one-study-out cross-validation

[← back to the final report](README.md) · reproduce with
[`scripts/nested_cv.py`](../../scripts/nested_cv.py)

Every earlier evaluation here held out *parameters* while choosing other things
on all three studies at once. `loso.py` fits a shrinkage factor on two studies and
scores it on the third — honest as far as it goes — but the ensemble membership,
the within-outcome factor and the structural donor were picked by looking at the
mean across all three, so the resulting numbers were contaminated by selection
even though each individual score was out-of-fold.

This closes that gap, and it changed one conclusion.

## Design

For each held-out study, **every free choice is fitted on the other two only**:
which runs' effects to average, the within-outcome factor, the global factor,
which run supplies level and residual spread, and the residual scale.

The selection rule was fixed before any held-out score was looked at, and it is
lexicographic rather than a weighted composite, because the leaderboard sorts on
pooled Pearson r:

1. membership and within-outcome factor maximise **training** pooled `pearson_r`
2. the global factor is set so the **training** calibration slope is exactly 1
3. donor and residual scale minimise **training** level and dispersion error

Three folds cannot support an interval, and none is reported.

## What the training folds chose

| held out | membership | within | κ | donor | residual scale |
| --- | --- | --- | --- | --- | --- |
| Voelkel | 7B + 72B | 0.90 | 0.291 | `v4_flash` | 1.050 |
| ICPC | 7B + 72B | 0.40 | 0.622 | `v4_flash` | 0.938 |
| Goldwert | **72B alone** | 0.20 | 0.437 | `v4_flash` | 0.936 |

The structure side **transfers**: the same donor is chosen in all three folds and
the residual scale is stable within 0.94–1.05. The effect side does **not**: the
within factor ranges 0.20 to 0.90 and one fold drops a model from the ensemble
entirely.

## The headline result, and the trap in it

| variant | Voelkel | ICPC | Goldwert | **mean r** |
| --- | --- | --- | --- | --- |
| recipe with **everything** fitted per fold | 0.478 | 0.339 | **0.159** | **0.325** |
| **shipped design** — structure pre-committed, only κ fitted | 0.576 | 0.342 | 0.358 | **0.426** |
| Qwen pair, within 0.3 | 0.638 | 0.334 | 0.312 | 0.428 |
| Qwen pair, no within shrink | 0.460 | 0.341 | 0.394 | 0.398 |
| 7B alone, within 0.5 | 0.439 | 0.330 | 0.320 | 0.363 |
| single: Qwen2.5-7B, uncalibrated | 0.408 | 0.320 | 0.319 | 0.349 |
| single: Qwen2.5-72B, uncalibrated | 0.340 | 0.323 | 0.325 | 0.329 |
| single: DeepSeek-V4-Flash, uncalibrated | 0.190 | −0.246 | −0.064 | −0.040 |
| *human replication* | *0.514* | *0.640* | *0.642* | *0.599* |

**Fitting the membership and the within factor is worse than pre-committing
them.** Letting the training folds choose scores **0.325**, below plain
Qwen2.5-7B at 0.349. Fixing the structure a priori — average every Qwen run, take
0.5 as the default — scores **0.426**. The entire difference is the Goldwert fold,
where training on Voelkel and ICPC chose 72B alone at within 0.20, and that
generalised at 0.159.

This is the strongest argument in the project for pre-committed defaults over
fitted ones, and it was only visible once the fitting was nested.

## Full metric set, shipped design, per held-out study

| held out | r | ρ | dir % | within | adj | RMSE | RMSE adj | α | β |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Voelkel | 0.576 | 0.507 | 61.1 | 0.389 | 0.795 | 1.277 | 0.701 | 0.042 | 1.359 |
| ICPC | 0.342 | 0.378 | 72.7 | 0.247 | 0.485 | 4.513 | 3.220 | 2.156 | 0.545 |
| Goldwert | 0.358 | 0.372 | 77.3 | 0.417 | 0.725 | 2.890 | 2.066 | 2.065 | 0.833 |
| **mean** | **0.426** | **0.419** | **70.4** | **0.351** | **0.668** | **2.893** | **1.996** | **1.421** | **0.912** |
| *human replication* | *0.599* | *0.520* | *82.7* | *0.354* | *0.872* | *2.956* | *2.081* | *0.664* | *0.514* |

Three things worth pulling out.

**RMSE beats the human replication** — 2.893 against 2.956 — and β is closer to 1
(0.912 against 0.514). That is not a claim to be better than real people at
predicting real people; it is what happens when a smoother predictor is graded
against a noisy one at this sample size.

**`pearson_within` matches the replication almost exactly**, 0.351 against 0.354.
Within-outcome ranking is where the recipe is genuinely at human level.

**α is much worse than the replication's**, 1.421 against 0.664, and it is
study-driven: 0.042 on Voelkel where the mean human effect is −0.06 pp, but 2.1
on ICPC and Goldwert where nearly every arm pushes the same way. A single global
factor cannot supply an intercept it was never given.

## Distributional metrics, same folds

From the fitted donor and residual scale, on the control arm:

| held out | level err (pp) | variance ratio | OVL | KS | W1 |
| --- | --- | --- | --- | --- | --- |
| Voelkel | 8.29 | 1.485 | 0.723 | 0.199 | 9.99 |
| ICPC | 18.89 | 0.695 | 0.597 | 0.363 | 3.62 |
| Goldwert | 8.11 | 0.864 | 0.730 | 0.292 | 1.82 |
| **mean** | **11.76** | **1.015** | **0.683** | **0.285** | **5.15** |

The level error here is **much worse than the shipped entry's**, and for a
structural reason rather than a modelling one: the cross-validation has no level
anchors, because TISP and CCAM measure Pfänder's questions and not these studies'.
The shipped entry pins three of thirteen outcomes to external survey levels
exactly. That advantage is real and this design cannot measure it.

## Honest bounds

Two numbers bracket the effect side, and neither alone is the answer:

* **0.325** is what a fully automated version of this recipe achieves when every
  choice is refitted per fold. It is pessimistic as a prediction for the shipped
  entry, because the shipped entry does not refit those choices.
* **0.426** is what the shipped structure achieves with those two choices held
  fixed. It is **not** an out-of-sample number for a fourth study, because both
  were motivated by these same three studies. It belongs in this table as an upper
  bound, not as the estimate.
* **0.385** is the defensible middle: the membership prior granted, the within
  factor charged as fitted. This is what the Pfänder prediction is built on.

The gap between them is mostly closed by one observation: within 0.3, 0.5 and 1.0
score 0.428, 0.426 and 0.398, so the design is insensitive to the choice inside a
wide band. What was harmful was fitting it sharply, not choosing it roughly.

The fold spread is larger than either adjustment — 0.342 to 0.576 on the same
recipe — and that between-study variation, not the selection question, is the
dominant uncertainty in predicting Pfänder.
