# Four studies, four models: what the expansion changed

[← back to the final report](README.md) · reproduce with
[`scripts/nested_cv.py`](../../scripts/nested_cv.py),
[`scripts/score_ccc_holdout.py`](../../scripts/score_ccc_holdout.py),
[`scripts/level_compression.py`](../../scripts/level_compression.py),
[`scripts/quantile_mapping.py`](../../scripts/quantile_mapping.py)

The [three-study cross-validation](cross_validation.md) ended on a number it
could not defend: 0.426 for the shipped design, flagged there as an upper bound
because the design was motivated by the same three studies it was scored on.
This adds a fourth study (Voelkel et al. 2026, the Climate Change Challenge) and
a fourth model (Muse-Glimmer-30B), and asks the same questions again.

Four things came out of it. Two are results, one is a defect this found in the
shipped ensemble, and one is a set of calibrations that were tried and do not
work. The defect is the most consequential and it is in
[its own section](#the-eighth-run-was-not-a-run).

## 1. The fourth study

CCC is a 13-arm climate-framing megastudy, n = 13,821, with pre- and
post-treatment measures on nine outcomes and a pooled placebo control of 3,183.
It matters here for a reason no other reference study has: **Pfänder took items
from it.** The three-item concern battery and the general-policy item are
verbatim identical, and three of six behavioural-intention items are.

It is also, by a wide margin, the **hardest fold**. Splitting CCC's own
participants in half and predicting one half's arm effects from the other's
reaches only r = 0.372, against 0.514–0.642 on the other three studies. Whatever
a model scores there is measured against a much lower ceiling.

## 2. Effect recovery, four folds

Every free choice is fitted on the other three studies; the assembled recipe is
scored once on the held-out one. Run availability is reported rather than assumed
— DeepSeek-V4-Flash has no CCC sample yet, CCC is in the training set of every
fold, so **no ensemble containing V4 was a candidate anywhere**. That is the
deliberate pre-V4 state, not a search that considered and rejected it.

| variant | CCC | Goldwert | ICPC | Voelkel | **mean r** |
| --- | --- | --- | --- | --- | --- |
| **rule: average all available, within 0.5** | 0.134 | 0.292 | 0.449 | 0.542 | **0.354** |
| Qwen pair, within 0.5 (shipped design) | 0.101 | 0.358 | 0.342 | 0.576 | **0.344** |
| Qwen pair, within 0.3 | 0.080 | 0.312 | 0.334 | 0.638 | 0.341 |
| 7B alone, within 0.5 | 0.248 | 0.320 | 0.330 | 0.439 | 0.334 |
| Qwen pair, no within shrink | 0.128 | 0.394 | 0.341 | 0.460 | 0.331 |
| Muse alone, within 0.5 | 0.191 | 0.046 | 0.602 | 0.275 | 0.279 |
| recipe with **everything** fitted per fold | 0.103 | 0.136 | 0.315 | 0.432 | 0.247 |
| single: Qwen2.5-7B, uncalibrated | 0.163 | 0.319 | 0.320 | 0.408 | 0.303 |
| single: Qwen2.5-72B, uncalibrated | 0.069 | 0.325 | 0.323 | 0.340 | 0.264 |
| single: Muse-Glimmer-30B, uncalibrated | 0.234 | 0.041 | 0.444 | 0.256 | 0.244 |
| *human replication* | *0.372* | *0.642* | *0.640* | *0.514* | *0.542* |

**The three-study finding survives.** Refitting membership and the within-outcome
factor per fold scores 0.247, below plain Qwen2.5-7B at 0.303. Pre-committing the
structure scores 0.344. Nesting the fitting is what makes that visible.

**Adding a fourth study lowered every number.** The shipped design went 0.426 →
0.344 and the human ceiling 0.599 → 0.542, almost entirely because CCC is hard.
Nothing got worse; a harder question got asked.

### Muse-Glimmer is a genuinely different model, not a better one

Its fold-to-fold spread is the widest of any model here — best on ICPC by a
distance (0.602 against 0.342 for the Qwen pair), near-zero on Goldwert (0.046).
Averaged over four folds it is the *weakest* single model at 0.244.

That combination is what makes it useful. Averaging pays when errors are
decorrelated, not when the added model is good, and Muse's errors sit differently
from the Qwens'.

### The membership question, decided by a rule rather than by reading the table

Declaring "average the Qwens and Muse" after looking at four fold means would be
a selection made on held-out data — the same contamination the three-study report
was pulled up for. So membership is decided by **rules**, each instantiated on the
training folds and never on the fold it is scored against:

| rule | 4 studies | 3 studies (V4 available) |
| --- | --- | --- |
| average all available runs | 0.354 | 0.388 |
| average every run with positive training r | 0.354 | **0.407** |
| *(shipped: Qwen pair, structure pre-committed)* | *0.344* | *0.426* |

On four studies the two rules coincide, because all three available runs clear
the bar on every fold. On three studies they separate, and that separation is the
evidence that the bar does real work: the positive-r rule **drops V4-Flash on the
Voelkel and Goldwert folds** and keeps it on ICPC, and gains 0.019 over averaging
everything. It will decide the V4 question on its own once the CCC run lands.

### What Muse changes, metric by metric

Pearson r is a wash. Everything else is not, and it moves the same way on both
fold sets:

| | 4 studies | | 3 studies | |
| --- | --- | --- | --- | --- |
| | Qwen pair | **+ Muse** | Qwen pair | **+ Muse** |
| pearson_r | 0.344 | **0.354** | 0.426 | 0.428 |
| spearman_rho | 0.347 | **0.372** | 0.419 | **0.434** |
| directional % | 70.7 | **76.3** | 70.4 | **76.6** |
| RMSE | 2.775 | **2.559** | 2.893 | **2.654** |
| RMSE adj | 1.980 | **1.661** | 1.996 | **1.618** |
| α | 1.291 | **1.001** | 1.421 | **1.032** |
| β | **1.079** | 0.869 | 0.912 | 0.899 |

Six of seven improve on both fold sets; β is the exception and it is close either
way. **This understates nothing and overstates one thing:** the cross-validation
compares one-run-per-model ensembles, while the shipped Qwen side averages
several seeds per model and is correspondingly less noisy. Muse would enter with
one Pfänder run and undiminished sampling noise, so the real gain is smaller than
the table shows.

That is why the decision is not taken here. Two Muse replicate seeds are sampling
now (DAIS 425081); with three runs its variance components can be measured the
same way the Qwens' were, and `ensemble_reliability` returns `None` rather than a
guess until they exist.

## 3. CCC as a held-out structural test

The cross-validation grades effects. It says nothing about the other three
quarters of the benchmark — `response_distributions`, `subgroup_distributions`,
`compare_demographic_baselines` and `demographic_parity_gap` all read the
**control arm only**, and none involves an arm contrast. Until now that half was
graded only on Pfänder itself against TISP anchors: one study, three outcomes.

Six of the nine shipped party-gap anchors and five of the eight dispersion
anchors were measured *on CCC*, so grading a CCC prediction that used them
against CCC humans would score a number against itself. They are switched off
here via `anchors.ccc.for_study("CCC")`, which returns nothing when CCC is held
out.

| | level err (pp) | sd ratio | OVL | KS | W1 | party gap RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 14.25 | 0.934 | 0.655 | 0.279 | 15.10 | 25.43 |
| Qwen2.5-72B | **10.04** | 1.218 | **0.680** | **0.203** | **11.23** | 13.90 |
| Muse-Glimmer-30B | 11.13 | 1.245 | 0.663 | 0.213 | 12.80 | **10.46** |
| *human half against half* | *0.60* | *0.998* | *0.852* | *0.027* | *0.99* | *0.91* |

Two findings, both about how much the anchors are worth.

**Unanchored levels are off by 10–14 pp**, and systematically: our means span
43–57 where the humans span 33–71. The models understate every high outcome and
overstate every low one.

**Unanchored party gaps are roughly half their human size.** Humans show
Democrat-minus-Republican gaps of 12–38 pp on these outcomes; our raw gaps run
6–18. This is the clearest evidence yet that the party anchoring is load-bearing
rather than decorative, and it is only visible because CCC lets the anchors be
taken away.

## 4. The eighth run was not a run

Checking replicate independence for the Muse decision turned up a defect in the
**shipped** ensemble.

`qwen25_72b_seed3` was sampled against a copy of `qwen25_72b_seed2`'s
`profiles.csv`. Replicates are supposed to differ in exactly one column — the
per-respondent RNG seed — and these two shared it, so they are one draw sampled
twice, not two draws.

| pair | identical scored cells | per-respondent r | effect-vector r |
| --- | --- | --- | --- |
| every genuine replicate pair (10 of them) | 9.4 – 11.5% | 0.01 – 0.08 | 0.745 – 0.885 |
| `qwen25_72b_seed2` × `qwen25_72b_seed3` | **77.1%** | **0.853** | **0.959** |

Nothing failed. The entry built, every check passed, and
`ensemble_reliability` quietly divided Qwen2.5-72B's sampling noise by four when
it should have divided by three.

**The consequence for the submission is small.** The eight-run and seven-run
averaged effect vectors correlate 0.998 and differ by 0.049 pp against a vector
spread of 1.53 pp; the shrinkage constant moves 0.4156 → 0.4127 and the claimed
reliability 0.964 → 0.957. But a duplicate counted as a seed is a false statement
about how much noise has been averaged away, so it is out of `BEST_RANKERS`,
which now has seven runs.

`tests/test_calibration_recipes.py` now compares the **draws** rather than the
profile files, because sharing a profile set is not itself the defect — every
model is meant to sample the same respondents, and the `_demo` runs share a seed
column with their parent while still differing, since the quota demographics
change the prompt. Only the answers can say whether two runs are one draw. The
guard was checked against the known duplicate before being relied on: it reports
`qwen25_72b_seed2 and qwen25_72b_seed3 agree on 77.1% of cells`.

To restore an eighth run, re-sample it against `qwen25_7b_seed2`'s profile set —
the one replicate set Qwen2.5-72B has never used.

## 5. Three calibrations that do not work

CCC opened three new routes. All three were tested and none survives; they are
recorded because the measurements are the useful part.

**Level expansion.** If the models compress control-arm levels toward the
midpoint, regressing human means on ours across outcomes recovers an expansion
factor. The compression is real on CCC — the Qwen2.5-72B slope is 1.32 — but the
slope is not stable: it runs 0.05 to 1.32 across studies and models, i.e. on most
folds our outcome means vary *more* than the humans', not less. Fitted on three
studies and applied to the fourth it helps in 7 of 12 cells and moves the mean
error 15.94 → 14.11 pp, with a per-cell range from −7.6 to +4.9. Not a
transferable correction.

**Quantile mapping.** The benchmark scores OVL, KS and W1, which are properties
of the whole distribution, and nothing shipped touches them except as a side
effect. Mapping our answers onto a human reference distribution targets them
directly — but only if the reference transfers. Tested on the two constructs ICPC
and CCC both measure on the same 0–100 range, with the map always fitted on the
*other* study's humans:

| | level err (pp) | OVL | KS | W1 |
| --- | --- | --- | --- | --- |
| raw | 12.22 | 0.695 | 0.237 | 12.59 |
| moment matching (mean + sd) | **3.05** | **0.725** | 0.225 | **7.87** |
| quantile map, borrowed reference | 3.14 | 0.721 | **0.213** | 8.70 |
| *quantile map, own reference (the ceiling)* | *0.13* | *0.883* | *0.067* | *0.68* |

Quantile mapping beat moment matching in 5 of 12 cells and loses on average.
Only ~14% of the available OVL gain survives borrowing: the shape difference
between two studies' human distributions on the same construct is about as large
as the shape error being corrected. **Moment matching — what the recipe already
does — is the right amount of machinery.**

**A donation anchor from Goldwert.** Pfänder's `donation_ams` has no level
anchor: CCC's donation is cents allocated across five charities against
Pfänder's dollars out of a $10 bonus, so its scale does not transfer. Goldwert's
*does* — same $10 bonus, same 100-participants-paid mechanism, same 0–10 whole
dollars — and its control mean of 4.77 sits 15.6 pp above our donor's 3.22, which
would be a large correction.

It is still wrong, and `anchors/goldwert.py` already said so before this was
re-derived. Goldwert doubles the donation pool if at least half of participants
give $5 or more, and the control arm duly puts **29.6% of respondents on exactly
$5**, against 1.5–3.5% on each of $1–$4 and $6–$9. The match manufactures a mode
at its own threshold; Pfänder has no match, so that mode cannot exist there, and
the mass sitting in it is worth about half a dollar of the mean. The recipient
differs in kind too — an unnamed advocacy organisation against the American
Meteorological Society — and the authors disclaim level representativeness for
their sample. The anchor stays `construct-only` and unused.

## What this changes for the Pfänder prediction

The three-study report built its prediction on **0.385** — the membership prior
granted, the within factor charged as fitted, which sat 59% of the way from the
fully-fitted 0.325 to the pre-committed 0.426. The four-study bracket is 0.247 to
0.344, and the same position inside it gives **0.305**.

Whether that is the right basis depends on whether Pfänder resembles CCC or the
other three, and the thing to compare is not arm size but how large the true arm
effects are against their standard errors:

| study | arms | mean \|effect\| (pp) | median SE (pp) | effect signal/noise | replication r |
| --- | --- | --- | --- | --- | --- |
| Voelkel | 6 | 1.04 | 0.75 | 2.83 | 0.514 |
| ICPC | 11 | 4.93 | 2.04 | 3.62 | 0.640 |
| Goldwert | 10 | 3.66 | 1.34 | 1.25 | 0.642 |
| **CCC** | 9 | **1.46** | **1.00** | **0.61** | **0.372** |

CCC's ceiling is low because climate-framing effects are genuinely tiny relative
to the precision with which they can be measured — not because the study is small.
It is the largest of the four.

**Pfänder is not protected by its design.** It runs 1,000 per treatment arm
against a 2,000 control; CCC runs about 1,065 against a pooled 3,183. That makes
Pfänder's arm contrasts roughly **9% noisier** than CCC's per unit of outcome
spread, so on the noise side it is slightly worse off, not better. An earlier
draft of this section had that backwards.

So the question is entirely about the numerator: if Pfänder's science-trust
interventions move trust by 1–2 pp like CCC's framings, its achievable
correlation is capped near 0.37 and a recipe scoring 0.30 is close to the ceiling;
if they move it by 3–5 pp like ICPC's and Goldwert's, both the ceiling and our
score should be higher. Nothing in the published protocol settles it.

The revised point prediction and its intervals are in
[prediction.md](prediction.md).
