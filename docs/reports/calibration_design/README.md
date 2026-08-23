# Designing the calibrations — what moves which metric

Intermediate report, 2026-08-23. Everything here is measured on data already on
disk: the Voelkel silicon samples scored against real participant responses, and
the Pfänder samples scored against themselves where no human data exists.
Reproduce with [`scripts/verify_calibration_levers.py`](../../../scripts/verify_calibration_levers.py)
and [`scripts/select_calibration.py`](../../../scripts/select_calibration.py).

Sub-reports: [1 · the metric map](01_metric_map.md) ·
[2 · what the models are each good at](02_model_components.md) ·
[3 · methodological traps](03_traps.md)

## The one-paragraph version

Scoring is multi-objective, and the scored analyses read **different terms** of the
same decomposition — which means the calibrations are largely independent and can
be combined rather than chosen between. Our two samplers turn out to be good at
different terms, so the best submission is not either of them: it is Qwen2.5-7B's
condition effects carrying DeepSeek-V4-Flash's levels, demographic structure and
respondent coherence. That combination is free, needs no new sampling, and beats
both models on their own metrics.

## The decomposition everything hangs on

    y_ijc = level_j + effect_jc + offset_j(m_i) + residual_i

| term | what reads it | who is better at it |
| --- | --- | --- |
| `effect` | ATE recovery, calibration regression, RMSE — the leaderboard sort key | **Qwen2.5-7B** (pooled r 0.408 vs 0.190) |
| `level` | response distributions (OVL, KS, W1), demographic baselines | **V4-Flash** (level error 8.0 vs 22.9 pp) |
| `offset` | stereotyping coefficients, parity gap, subgroup effects, baselines | **V4-Flash** (offset r 0.190 vs 0.027) |
| `residual` | variance ratio, respondent coherence, the SEs behind `beta_adj` | **V4-Flash** (cross-outcome r 0.341 vs 0.138) |

Three of the four favour the big model. The one that decides the leaderboard's
sort key does not. This is the third time this project has found that pattern,
and it is the central fact about the submission.

## What each calibration actually buys

Measured on the Voelkel pairs, 54 (outcome × condition) cells, Qwen2.5-7B.

| calibration | r | ρ | dir % | RMSE | β | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.408 | 0.311 | 61.1 | 3.620 | 0.159 | — |
| global shrink, k = 0.159 | 0.408 | 0.311 | 61.1 | **1.402** | **1.003** | mandatory hygiene, zero leaderboard value |
| within-outcome shrink, 0.5 | **0.439** | **0.402** | 63.0 | 2.468 | 0.252 | held, not adopted — see below |
| cross-model average | 0.428 | 0.410 | 57.4 | 2.318 | 0.267 | free, positive, not established |
| both | **0.484** | **0.484** | 61.1 | 1.671 | 0.455 | — |
| outcome profile replaced (oracle) | **0.491** | 0.420 | 61.1 | 3.080 | 0.214 | needs an anchor at ρ ≥ 0.7 |
| *human replication, for scale* | *0.514* | *0.395* | *66.7* | *1.682* | *0.436* | |

**A single global rescale provably cannot move six of the ten Section-1 numbers.**
`directional_pct`, `spearman_rho`, `pearson_r`, `pearson_within`, `pearson_adj`
and `alpha` come out bit-identical at k = 0.5, 0.159 and 0.05. It moves β
(0.159 → 1.003), `beta_adj`, RMSE (3.620 → 1.402) and `rmse_adj`. So shrinkage is
worth a large slice of RMSE and provably nothing on the sort key. Note
`pearson_adj` survives it, because that metric is corrected with the *reference's*
standard errors rather than ours — getting this wrong makes shrinkage look far
more dangerous than it is, and inflated an earlier estimate of how much extra
sampling we needed.

**Two shrinkage factors, not one.** The factor minimising RMSE regresses through
the origin, `Σhl/Σl²`; the factor driving the benchmark's β to exactly 1 regresses
*with* an intercept, `cov(h,l)/var(l)`, because that is what the benchmark fits.
On Voelkel α is 0.014 and the two agree to 0.25% (0.15878 against 0.15918) — so
the difference is invisible on real data and would have sat there as a silent bug.

## The largest lever, and why it may not be usable

**56.6% of the variance in the human effect vector is between-outcome**, and an
oracle that predicts only the per-outcome mean human effect — with no information
whatever about which message works — scores a pooled r of **0.752**, against a
fresh human replication's 0.514. Most of what the pooled correlation rewards is
knowing which outcomes move, not ranking the sixteen messages.

But the profile has to come from somewhere, and a borrowed one is not the truth:

| ρ(anchor, truth) | 0.2 | 0.4 | 0.6 | **0.8** | 1.0 |
| --- | --- | --- | --- | --- | --- |
| resulting pooled r | 0.293 | 0.329 | 0.392 | **0.438** | 0.491 |

Our own profile already reaches ρ = 0.473 and r = 0.408, so **a transferred anchor
must reach ρ ≈ 0.7 before it beats leaving our own profile alone**, and one at
ρ = 0.4 would cost us 0.08 of r. Measuring that transfer across studies is the
single most decision-relevant number still outstanding.

Two facts sharpen it. Qwen's own outcome profile correlates 0.473 with the human
one and V4-Flash's only 0.240 — and the two models' profiles correlate **−0.174
with each other**. They disagree about which outcomes move, and the better of the
two is Qwen's. So averaging the profile component would pull toward mush even
though averaging the whole effect vector helped slightly; the ensemble wants to be
asymmetric.

## Where V4-Flash wins, and why it can be taken without its costs

Because the terms are separable, V4-Flash's advantages can be transplanted onto
Qwen's effects. Measured on Voelkel:

| submission | r | ρ | dir % | offset r | level err | variance ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 0.408 | 0.311 | 61.1 | 0.027 | 23.7 | 1.052 |
| V4-Flash | 0.190 | 0.186 | 55.6 | 0.190 | 8.3 | 1.379 |
| **Qwen effects + V4 level** | **0.408** | **0.311** | **61.1** | 0.081 | **8.3** | **0.933** |
| Qwen effects + V4 level & demographics | 0.408 | 0.311 | 61.1 | 0.135 | 8.3 | 1.310 |
| *human replication* | *0.514* | *0.395* | *66.7* | *0.897* | *0.3* | *0.990* |

Taking V4-Flash's **level** is a strict improvement: identical effect metrics,
level error cut from 23.7 to 8.3 pp, and the best variance ratio of the three at
0.933 against the human replication's 0.990. Taking its **demographics** as well
is a genuine trade — offset correlation more than doubles, but the variance ratio
degrades to 1.310 because V4-Flash over-disperses. That term wants rescaling
rather than copying, which is what the offset calibration below does.

## Demographic responsiveness

The failure is real and structured: Qwen puts synthetic Republicans and Democrats
1.1 points apart on climate belief where reality is tens; V4-Flash puts them 12.4
apart. Over 162 control-arm cells on Voelkel:

| moderator | cells | r, Qwen | r, V4-Flash | sd synth Q / V4 | sd human |
| --- | --- | --- | --- | --- | --- |
| party | 27 | +0.117 | **+0.425** | 2.79 / 5.17 | 3.04 |
| race | 45 | −0.011 | +0.229 | 1.93 / 2.02 | 3.62 |
| gender | 18 | +0.325 | −0.233 | 0.36 / 0.47 | 1.25 |
| education | 36 | +0.002 | +0.053 | 0.96 / 1.09 | 2.40 |
| age | 36 | −0.169 | −0.023 | 0.56 / 0.88 | 3.88 |
| **pooled** | 162 | **+0.027** | **+0.190** | 1.60 / 2.44 | 3.21 |

Every synthetic moderator is *under*-dispersed against the humans, so the natural
move is to inflate everything. **That is measurably worse** — it takes Qwen's
offset RMSE from 3.563 to 4.229. The deficit is not too little variation, it is
variation pointing the wrong way. Fitting the error-minimising scale per moderator
instead, `r · sd_human / sd_synth`, shrinks the uninformative moderators toward
zero and improves Qwen on both counts (offset r 0.027 → 0.105, RMSE → 3.301).
V4-Flash trades correlation for error, because the fit shrinks the two moderators
that were genuinely carrying its signal. So this is a per-model decision.

This reaches four *reported* analyses that no effect-level calibration touches —
stereotyping coefficients, parity gap, demographic baselines, within-subgroup
distributions. It cannot move the sort key: all four scored subgroup metrics are
scale-invariant.

**And there is little subgroup signal to rescue.** Scored subgroup recovery over
756 condition × moderator interaction pairs on Voelkel: Qwen `pearson_r` = −0.041,
V4-Flash +0.001, human replication +0.146. Both models are at or below zero.

## Respondent coherence, and a response-style general factor

A finding that separates the models sharply and was not anticipated.

**Qwen's synthetic respondents are not people.** On the Pfänder control arm its
twelve-item trust composite correlates **+0.000** with distrust and only +0.307
with its own other trust measure. A respondent reporting high trust *and* high
distrust at once is incoherent. V4-Flash gets −0.270 and +0.637.

Voelkel shows the mechanism. Qwen's mean cross-outcome correlation is −0.013 as
the study scores its constructs, but **+0.288 in the raw item direction** — higher
than the human +0.088. Humans run the other way: +0.180 scored, +0.088 raw. Six of
the nine Voelkel outcomes are `100 − x` reversals, so a respondent that is merely
consistent about *where it puts the slider* looks inconsistent about what it
believes. **Qwen has a response-style general factor; humans have an attitudinal
one.** That is also the cleanest available explanation of the earlier finding that
undoing the reversals leaves a mean signed raw-scale error of −18.6 pp.

## Two things not worth building

Recorded because both looked promising and the measurements closed them.

**Cross-construct dispersion repair.** Voelkel's Composite is under-dispersed at a
variance ratio of 0.452, and the mechanism is entirely the missing correlation:
predicted composite SD from the correlation alone is 8.28 against an observed
8.35, and from the human correlation 12.36 against 12.41. But Pfänder scores no
cross-construct composite, and its own multi-item outcomes have healthy internal
correlations of 0.51–0.66 with α 0.82–0.91. The deficit reaches nothing Pfänder
scores.

**Per-outcome shrinkage factors.** Per-outcome β on Voelkel ranges −0.043 to
0.500, which looks like ample structure to exploit. The *oracle* upper bound on
pooled r from using them is **+0.003**. Thirteen parameters for nothing; the
apparent value is the outcome-profile effect, already captured above.

## Compute note

Qwen2.5-72B on 4×H200 measured at **2,368 respondents/hour** at group size 128 —
about twice DeepSeek-V4-Flash's 1,022–1,245 on identical hardware, and matching
the derived estimate of ~2,400. Dense 72B streams half V4-Flash's bytes per step
and its GQA-8 cache costs 320 KB/token against ~1 MB, which buys a four-times
larger resident group. It is both the cheaper model to schedule and the faster one
per respondent.

## What is still outstanding

1. **The cross-study profile transfer test** — does one study's per-outcome effect
   profile predict another's at ρ ≥ 0.7? Needs ICPC and Goldwert. This decides
   whether the largest lever on the board is usable at all.
2. **External level anchors** for Pfänder, which publishes no human data. TISP
   carries the identical twelve-item Besley trust battery; CCAM carries climate
   belief, worry and policy support.
3. **Leave-one-study-out selection** under the pre-committed rules, replacing the
   within-study stand-in used so far.
