# The four-study cross-validation, re-derived

[← back to the final report](README.md) · reproduce with
[`scripts/nested_cv.py`](../../scripts/nested_cv.py),
[`scripts/nested_benchmark.py`](../../scripts/nested_benchmark.py),
[`scripts/score_ccc_holdout.py`](../../scripts/score_ccc_holdout.py) ·
defects and fixes in [audit_findings.md](audit_findings.md)

This re-derives everything behind
[four_study_cross_validation.md](four_study_cross_validation.md) from the data
up: every script re-read, every quoted number recomputed, every method checked
against the benchmark's own preregistration. Then it re-runs the
cross-validation with the defects fixed and re-states the Pfänder prediction.

No new LLM inference was run and none is needed for anything here.

---

## 1. How much of the old report can you still trust?

**The numbers are mostly valid; two of the conclusions drawn from them are not.**
Every effect-recovery figure is about 6% too generous, the qualitative reading of
§2 reverses once you stop reading one arbitrary half-split, and §5's party-gap
argument does not survive at all. In more detail, by section:

| section of the old report | verdict |
| --- | --- |
| §1 The fourth study | **Valid.** CCC's size, arm count and item overlap with Pfänder all check out. |
| §2 Effect recovery, four folds | **Every number ~0.02 too high, and its headline conclusion is a seed artefact.** The variant *ordering* is stable; the claim that per-fold fitting costs you against a raw single model is not — it reverses on the split average. |
| §3 CCC as a held-out structural test | **Numbers valid, mechanism misdescribed.** The table reproduces to three decimals. The "leakage guard" it credits was a `print` statement that filtered nothing — the numbers are unanchored because the script never anchors, not because a guard removed anchors. Two narrative sentences describe Qwen2.5-72B only and are false for the other two models. |
| §4 The eighth run was not a run | **Valid, and it was the right call.** I re-derived the duplicate three independent ways and re-measured the reliability three ways; 0.957 stands. |
| §5 Four calibrations tested | **Three of four valid; the party-gap argument does not survive.** Level expansion, quantile mapping and the Goldwert donation rejection all reproduce. The arithmetic that concludes "0.5 sits inside the defensible range" rests on a distance measured against the wrong anchor set, and contradicts a direct measurement printed two sections earlier in the same report. |
| §6 Why CCC's ceiling is low | **Half valid.** The precision story is right; the specific comparison 0.372 → 0.624 is confounded, because the two estimands were scored on different pair sets (81 against 72). On the same 72 pairs the simple ATE scores 0.293, so the ANCOVA gain is larger than reported, not smaller. |
| "What this changes for the Pfänder prediction" | **Superseded, and it already disagreed with `prediction.md`.** It says the basis is 0.305; `prediction.md` uses 0.330. Neither is now right. |

**The claim that does not survive.** The old report's bolded conclusion —
"Refitting membership and the within factor per fold scores 0.247, below plain
Qwen2.5-7B at 0.303" — comes from a single half-split of the humans. Averaged
over eight splits the fully-fitted recipe scores **0.310 against 0.254**, i.e.
above the single model rather than below it, and the split-to-split standard
deviation (0.04–0.08) is larger than nearly every difference the old table asks
you to read. This is a correction in the recipe's favour.

**The one number that really moves.** Every effect-recovery figure in the old
report was computed with CCC's donation outcome on the wrong scale. CCC's
donation is a *constant-sum* item — Qualtrics held six boxes to a total of
exactly 100, and all 13,173 real respondents obey it — but the transcript the
silicon samples were drawn against rendered six unconstrained sliders. 42% of
Qwen2.5-7B's synthetic respondents write some other total, and the scored sum
reaches 500 on a scale the code declares to be 0–100. Repairing that (rescaling
each respondent's allocation back to the budget the instrument enforced, which is
the identity on the human side) costs **0.015 to 0.024** on every four-fold mean.

That is the whole of the *numerical* movement: every other correction I made is
worth 0.003 or less on a fold mean, and the ordering of the variants within the
table is stable. What is not stable is the comparison the old report drew across
that ordering, which is the split-average point above.

**What was checked and found fine.** The metric implementations are a faithful
translation of the benchmark's `R/functions/statistics.R` — I compared them
function by function against the preregistration. The half-sample reference is
correct (the benchmark really does score against Human 1). The percentage-point
conversion is correct and is what the benchmark specifies, explicitly, over the
alternative of pooling raw mixed units. The control-arm identification is right
in all four studies. No anchor derived from a cross-validation fold reaches the
effect cross-validation. There is no leakage of held-out data into any fitted
parameter.

---

## 2. Effect recovery, four folds, corrected

Every free choice is fitted on the other three studies; the assembled recipe is
scored once on the held-out one. Run availability is reported rather than
assumed — DeepSeek-V4-Flash has no CCC sample, CCC is in the training set of
every fold, so **no ensemble containing V4 was a candidate anywhere**.

| variant | CCC | Goldwert | ICPC | Voelkel | **mean r** | *(old)* |
| --- | --- | --- | --- | --- | --- | --- |
| **rule: average all available, within 0.5** | 0.073 | 0.292 | 0.449 | 0.541 | **0.339** | *0.354* |
| Qwen pair, within 0.5 (shipped design) | 0.009 | 0.358 | 0.342 | 0.576 | **0.321** | *0.344* |
| Qwen pair, within 0.3 | −0.018 | 0.312 | 0.334 | 0.639 | 0.317 | *0.341* |
| Qwen pair + Muse, no within shrink | 0.124 | 0.286 | 0.427 | 0.429 | 0.316 | *0.331* |
| 7B alone, within 0.5 | 0.170 | 0.320 | 0.330 | 0.439 | 0.315 | *0.334* |
| Qwen pair, no within shrink | 0.054 | 0.394 | 0.341 | 0.460 | 0.312 | *0.331* |
| 2x2: membership by prior, within fitted | 0.009 | 0.339 | 0.344 | 0.460 | 0.288 | *0.330* |
| Muse alone, within 0.5 | 0.206 | 0.046 | 0.602 | 0.272 | 0.282 | *0.279* |
| recipe with **everything** fitted per fold | 0.039 | 0.136 | 0.315 | 0.432 | **0.231** | *0.247* |
| single: Qwen2.5-7B, uncalibrated | 0.113 | 0.319 | 0.320 | 0.408 | 0.290 | *0.303* |
| single: Qwen2.5-72B, uncalibrated | −0.002 | 0.325 | 0.323 | 0.341 | 0.247 | *0.264* |
| single: Muse-Glimmer-30B, uncalibrated | 0.240 | 0.041 | 0.444 | 0.253 | 0.245 | *0.244* |
| *human replication* | *0.372* | *0.642* | *0.640* | *0.527* | *0.545* | *0.542* |

**The three-study finding does *not* survive, and this is the correction I
expected least.** On this split, refitting membership and the within-outcome
factor per fold scores 0.231, below plain Qwen2.5-7B at 0.290 — which is what
the old report reported and read as "fitting the structure per fold costs you".
Averaged over eight half-splits it is 0.310 against 0.254, i.e. **above** plain
7B by +0.057, and it wins in 5 of 8 splits. The old conclusion is an artefact of
one draw of the reference half. See the next section.

**CCC got harder, not easier.** The shipped design's CCC fold went 0.101 → 0.009
and the fully-fitted recipe 0.103 → 0.039. Both were resting on a donation
outcome measured on a scale five times too wide. The human ceiling on CCC is
unchanged at 0.372, because the humans were always on the right scale.

**Muse-Glimmer is still a genuinely different model rather than a better one.**
Best on ICPC by a distance (0.602 against 0.342 for the Qwen pair), near-zero on
Goldwert (0.046), weakest single model averaged over four folds. That
combination is what makes averaging pay.

### Averaged over eight half-splits

Every variant in the table above is scored against the same reference half, so
that half's sampling noise is *common* to all of them: one draw of it moves the
whole column together and can reorder any pair separated by less than it. A
single split is exactly right for predicting a Pfänder **score**, because the
benchmark fixes one preregistered split. It is the wrong instrument for deciding
which recipe is better.

| variant | mean over 8 splits | se | sd | min | max | *(at seed 42)* |
| --- | --- | --- | --- | --- | --- | --- |
| rule: positive training r, within 0.5 | **0.353** | 0.014 | 0.039 | 0.278 | 0.392 | *0.339* |
| rule: average all available, within 0.5 | **0.350** | 0.014 | 0.040 | 0.278 | 0.392 | *0.339* |
| Qwen pair, within 0.3 | 0.315 | 0.024 | 0.068 | 0.212 | 0.391 | *0.317* |
| rule: positive training r, no within shrink | 0.314 | 0.016 | 0.044 | 0.231 | 0.362 | *0.316* |
| Qwen pair, within 0.5 (shipped design) | 0.312 | 0.024 | 0.067 | 0.205 | 0.388 | *0.321* |
| 2x2: membership fitted, within fixed 0.5 | 0.311 | 0.014 | 0.039 | 0.248 | 0.360 | *0.248* |
| **recipe, everything fitted per fold** | **0.310** | 0.016 | 0.045 | 0.231 | 0.383 | ***0.231*** |
| 7B alone, within 0.5 | 0.306 | 0.028 | 0.079 | 0.161 | 0.411 | *0.315* |
| Muse alone, within 0.5 | 0.293 | 0.009 | 0.025 | 0.255 | 0.318 | *0.282* |
| Qwen pair, no within shrink | 0.292 | 0.023 | 0.066 | 0.183 | 0.363 | *0.312* |
| **2x2: membership by prior, within fitted** | **0.282** | 0.028 | 0.079 | 0.160 | 0.365 | ***0.288*** |
| *single: Qwen2.5-72B, uncalibrated* | *0.257* | *0.018* | *0.050* | *0.187* | *0.336* | *0.247* |
| *single: Qwen2.5-7B, uncalibrated* | *0.254* | *0.028* | *0.078* | *0.110* | *0.337* | *0.290* |
| *single: Muse-Glimmer-30B, uncalibrated* | *0.244* | *0.004* | *0.012* | *0.223* | *0.256* | *0.245* |
| *human replication* | *0.540* | *0.012* | *0.034* | *0.484* | *0.588* | *0.545* |

**The split-to-split standard deviation is 0.04 to 0.08 — larger than almost
every difference between variants.** That is the whole lesson. Reading an
ordering off one split is reading noise, and the old report did exactly that.

Against an uncalibrated Qwen2.5-7B, counting how often each variant wins:

| variant | mean delta | splits won | *at seed 42* |
| --- | --- | --- | --- |
| rule: positive training r, within 0.5 | **+0.099** | **8/8** | *+0.049* |
| rule: average all available, within 0.5 | **+0.096** | **8/8** | *+0.049* |
| Qwen pair, within 0.3 | +0.061 | 8/8 | *+0.027* |
| Qwen pair, within 0.5 (shipped design) | +0.059 | 8/8 | *+0.032* |
| recipe, everything fitted per fold | +0.057 | 5/8 | ***−0.059*** |
| 2x2: membership by prior, within fitted | +0.029 | 5/8 | *−0.002* |

**Every calibrated variant beats a raw single model on the split average**, and
the ones with pre-committed structure beat it in 8 of 8 splits. The old report
concluded the opposite about the fully-fitted variant, and that conclusion came
entirely from seed 42's draw. This is the largest of the three corrections here
that run *in the project's favour*, the others being the party-gap weight (which
was too low, not too high) and the ANCOVA gain in §6 (which is bigger than
reported, not smaller).

Two things do survive unchanged. The membership **rules** are still the best
variants and the most stable (sd 0.039–0.040, the smallest of any calibrated
variant, and 8/8 wins). And the prediction basis barely moves: 0.288 at seed 42
against 0.282 averaged, so the Pfänder point prediction below does not depend on
which of the two readings you take.

### The interval the benchmark will actually print

The old report said four folds cannot support an interval and reported none.
That is true of the *fold mean* and beside the point: the benchmark's own
uncertainty interval is a cluster bootstrap over interventions **within a
study**, 2,000 draws, and every leaderboard row will carry one. Each fold can
report it, and now does.

| variant | CCC | Goldwert | ICPC | Voelkel |
| --- | --- | --- | --- | --- |
| shipped design | 0.009 [−0.16, 0.18] | 0.358 [0.18, 0.52] | 0.342 [0.16, 0.50] | 0.576 [0.36, 0.78] |
| rule: average all | 0.073 [−0.11, 0.26] | 0.292 [0.04, 0.52] | 0.449 [0.25, 0.63] | 0.541 [0.39, 0.77] |
| *human replication* | *0.372 [0.18, 0.58]* | *0.642 [0.56, 0.73]* | *0.640 [0.41, 0.81]* | *0.527 [0.33, 0.69]* |

Mean half-width is **±0.18** at these arm counts (6 to 11). Pfänder has 16
interventions, so scaling by √(arms) puts its half-width near **±0.13**. That is
the single most useful number in this report: it is larger than the entire
spread between the recipe variants above, and it is what separates two
leaderboard rows.

### Measured reliability, and the only honest bridge to Pfänder

A fold's score is attenuated by noise in *our* effect vector as well as by noise
in the human reference. The reference studies give us one run per model; Pfänder
gets seven. Both sides are now measured rather than assumed, from the standard
errors the OLS fits already produce:

| | CCC | Goldwert | ICPC | Voelkel |
| --- | --- | --- | --- | --- |
| one-run-each Qwen pair (what the folds score) | 0.957 | 0.844 | 0.873 | 0.940 |
| human reference half | 0.213 | 0.245 | 0.497 | 0.540 |

Against Pfänder's shipped seven-run ensemble at **0.957**, measured three
independent ways that agree — the variance-components model in `recipes.py`
(0.9574), the standard errors of the averaged vector (0.9573), and a split-half
over the seven runs stepped up by Spearman-Brown (0.9610). The third makes no
independence assumption at all, which settles a doubt raised about counting the
two `_demo` runs as seeds: they correlate 0.856 with their parent against 0.834
for two elicited seeds, close enough that the reliability does not move.

The per-fold lift on correlations is therefore 1.000 (CCC), 1.065 (Goldwert),
1.047 (ICPC), 1.009 (Voelkel) — a mean of **1.030**, against the 1.049 the old
prediction used.

The second row is the uncomfortable one. **The human reference half on CCC and
Goldwert is 79% and 76% sampling noise.** That is not a defect in this analysis;
it is what the benchmark's estimand does to a study whose arm effects are small
relative to a half sample. Pfänder will have its own value of that number and
nobody knows it.

---

## 3. All four benchmark sections, cross-validated

The old cross-validation graded effects. That is one of the benchmark's four
scored sections, which is why the shipped predictions for response
distributions, demographic baselines and subgroup recovery rested on a single
in-sample study.

[`scripts/nested_benchmark.py`](../../scripts/nested_benchmark.py) closes that.
For each held-out study it assembles a **real** `Recipe`, applies it through the
same `recipes.apply` the submission is built with, and scores the resulting
respondent-level frame with `leaderboard_row` — every scored analysis, on the
held-out study's humans. It agrees with the fast effect-vector grid to within
0.002 on Section 1 (0.320 against 0.321 for the shipped design, 0.339 against
0.339 for the rule, 0.545 for the human replication), which is what makes the
fast grid trustworthy.

**Every recipe row below is the *unanchored* recipe.** The level and party-gap
anchors the submission uses exist for Pfänder and not for these studies, so
Sections 3 and 10–12 here are pessimistic for the entry by the amount the
anchoring is worth. That amount is measured separately in §4.

| | shipped design | rule: avg all | best single | *human replication* |
| --- | --- | --- | --- | --- |
| **§1** pearson_r | 0.320 | 0.339 | 0.290 | *0.545* |
| **§2** subgroup pearson_r | −0.081 | −0.099 | −0.044 | *0.071* |
| **§2** subgroup directional % | 51.7 | 51.5 | 51.7 | *52.3* |
| **§3** variance_ratio | 0.968 | 0.968 | 1.043 | *1.008* |
| **§3** OVL | 0.663 | 0.663 | 0.656 | *0.859* |
| **§3** KS | 0.277 | 0.277 | 0.260 | *0.041* |
| **§3** W1 | 12.97 | 12.97 | 13.42 | *1.45* |
| **§10** baseline RMSE | 11.36 | 11.36 | 10.22 | *2.03* |
| **§11** parity DPD | 7.35 | 7.35 | 6.09 | *1.66* |
| **§11** parity worst | 18.27 | 18.27 | 16.81 | *5.27* |
| **§12** stereotyping coef RMSE | 4.58 | 4.58 | 5.19 | *1.26* |
| **§12** stereotyping R² gap | −0.008 | −0.008 | −0.009 | *0.002* |

Three things this settles that nothing previously could.

**Subgroup recovery is negative out of fold, on four studies rather than one.**
The old prediction rested on a single Voelkel measurement (Qwen −0.041, human
+0.146) and called itself "the weakest row in the report". Four folds put the
recipe at **−0.08** and, more importantly, put the *human replication ceiling* at
**+0.071**, half the single-study figure. Condition × moderator interactions
come from small cells; at these sample sizes a fresh half sample barely predicts
the other half either.

**The recipe's calibration helps Section 1 and hurts Sections 10–12.** Baseline
RMSE goes 10.22 (raw Qwen2.5-7B) → 11.36 (calibrated), parity DPD 6.09 → 7.35.
That is not a contradiction: the structural donor here is chosen from
{7B, 72B, Muse} because V4-Flash has no CCC sample, and V4-Flash is the donor
the submission actually uses precisely because it is much the best at levels and
demographics. **So the structural half of the shipped recipe is still not
cross-validated** — the fold search cannot even consider its donor. This is the
largest remaining gap and no amount of re-analysis closes it; it needs a CCC run
of V4-Flash.

**Section 3 is dominated by the anchoring.** All four recipe variants print
identical Section 3 numbers, because effects do not touch the control arm and the
variants differ only in effects. Whatever the recipe scores on OVL, KS and W1 is
set entirely by the donor, the residual scale, and the anchors.

---

## 4. What the anchoring is worth, held out on CCC

CCC is the one study that can grade the structural half out of sample, because
six of Pfänder's outcomes reach it through the crosswalk and five carry level
anchors measured on it. The hold-out numbers reproduce the old report's §3 table
to three decimals, and are unchanged by the donation repair (the repair moves
effects, not control-arm levels):

| | level err (pp) | sd ratio | OVL | KS | W1 | party gap RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B, unanchored | 14.22 | 0.932 | 0.654 | 0.288 | 15.28 | 25.40 |
| Qwen2.5-72B, unanchored | **10.05** | 1.216 | **0.680** | **0.208** | **11.24** | 13.90 |
| Muse-Glimmer-30B, unanchored | 11.10 | 1.245 | 0.663 | 0.213 | 12.77 | **10.45** |
| *human half against half* | *0.60* | *0.998* | *0.852* | *0.027* | *0.99* | *0.91* |

Against the *anchored* half of the built entry, graded on the five Pfänder
outcomes whose items CCC shares:

| | level err (pp) | sd ratio | OVL | KS | W1 |
| --- | --- | --- | --- | --- | --- |
| built entry, five anchored outcomes | 0.13 | 1.047 | **0.841** | **0.067** | **3.06** |
| *CCC human half against half* | *0.35* | *1.034* | *0.839* | *0.032* | *1.19* |

The level and sd rows are circular — those are the two moments the anchor sets —
but OVL, KS and W1 are not, and OVL 0.841 against a human ceiling of 0.839 is
the best distributional result in the project. It remains an optimistic bound,
because here the anchor and the grader are the same distribution; the
quantile-mapping experiment measures the penalty for a *borrowed* reference at
OVL 0.725 rather than 0.883.

**Two corrections to how the old report described this section.** First, its
"leakage guard" was a `print` whose return value was discarded — the table is
unanchored because the grading path never anchors, not because a filter removed
anything. That is now an assertion which stops the run. Second, the two
narrative claims ("levels off by 10–14 pp", "party gaps roughly half their human
size") are true of Qwen2.5-72B and not of the other two models: Qwen2.5-7B is off
by 14.2 pp with gaps at a *quarter* of human size, Muse by 11.1 with gaps much
closer. The conclusion — that party anchoring is load-bearing — survives and is
in fact understated.

---

## 5. The Pfänder prediction

### Basis

The old report's basis reasoning stands and I keep it: grant the *form* of each
choice as a prior — averaging comparable estimators is standard variance
reduction, shrinking toward a group mean is standard empirical Bayes — and charge
the *magnitude* as fitted, because which models and what factor both needed data.
That is the "membership by prior, within fitted" row, **0.288** on the
preregistered-style split and **0.282** averaged over eight. The two agree, which
is the main reason to use it: it is the one basis that does not depend on which
reading of the cross-validation you take.

That figure has moved from the old report's 0.330 for two separate reasons, both
corrections: the donation repair (−0.023) and a fix to the row itself, which had
been reusing a within-factor fitted jointly with a *different* membership rather
than refitting it for the prior membership (−0.018).

Stepping the correlations up per fold to the shipped ensemble's reliability gives
**0.298**.

**The bracket is now narrower than the old report's and sits higher at its
bottom.** On one split it runs 0.231 (every choice charged as fitted) to 0.321
(structure pre-committed); on the split average it runs 0.282 to 0.353, and the
fully-fitted variant is 0.310 rather than the bottom of the range. The old
report's floor — "if none of the design transfers, `pearson_r` lands near 0.26,
*below* an uncalibrated Qwen2.5-7B" — was the seed artefact. On the split average
every calibrated variant beats every raw single model, so that floor should be
read as roughly **0.28**, not below the single-model baseline.

### Section 1 — intervention effect recovery

| metric | target | **prediction** | 95% interval the benchmark will print | human replication |
| --- | --- | --- | --- | --- |
| `pearson_r` (sort key) | high | **0.30** | ±0.13 → **0.17 – 0.43** | 0.545 |
| `spearman_rho` | high | **0.30** | 0.17 – 0.43 | 0.487 |
| `directional_pct` | high | **71** | 62 – 79 | 81.7 |
| `pearson_within` | high | **0.31** | 0.18 – 0.44 | 0.282 |
| `pearson_adj` | high | **0.47** | 0.28 – 0.66 | 0.858 |
| `rmse` | low | **2.8** | 1.4 – 4.8 | 2.68 |
| `rmse_adj` | low | **2.1** | 0.9 – 3.6 | 1.83 |
| `alpha` | 0 | **1.3** | 0.0 – 2.6 | 0.656 |
| `beta` | 1 | **1.15** | 0.55 – 1.65 | 0.490 |

`rmse`, `alpha` and `beta` are taken from the structure-pre-committed row rather
than the basis row, because unlike the correlations they are set by the shrinkage
arithmetic and the shipped entry really does use `within = 0.5`.

The interval is now the benchmark's own cluster bootstrap over interventions,
scaled from the folds' 6–11 arms to Pfänder's 16, rather than a judgment about
where to trim. The RMSE and α ranges stay judgment-wide for the reason the old
report gave and which still holds: they depend on the size of Pfänder's true
effects, which is unknown, and the folds ran 1.41 to 4.76 on RMSE purely because
real effect magnitudes differ several-fold between these studies.

### Section 2 — subgroup effects

| metric | target | **prediction** | range | human replication |
| --- | --- | --- | --- | --- |
| subgroup `pearson_r` | high | **−0.08** | −0.16 – 0.02 | 0.071 |
| subgroup `spearman_rho` | high | **0.04** | −0.04 – 0.12 | 0.073 |
| subgroup `directional_pct` | high | **51.7** | 49 – 55 | 52.3 |
| subgroup `pearson_adj` | high | **0.09** | −0.10 – 0.28 | 0.204 |

Now from four out-of-fold studies rather than one in-sample one, and the
prediction has moved from "near zero" to **slightly negative**. The consolation
is that the ceiling moved down with it: a fresh human half sample scores 0.071
here, so the gap to the reference is small in absolute terms even though the
sign is wrong.

### Section 3 — response distributions (control arm)

Eight of thirteen outcomes carry an external level anchor and five do not. The
two halves are predicted separately and combined 8:5.

| metric | anchored (8) | unanchored (5) | **13-outcome prediction** | range | human replication |
| --- | --- | --- | --- | --- | --- |
| `variance_ratio` | 1.05 | 0.97 | **1.02** | 0.85 – 1.20 | 0.99 |
| `ovl` | 0.73 | 0.66 | **0.70** | 0.62 – 0.80 | 0.93 |
| `ks` | 0.10 | 0.28 | **0.17** | 0.10 – 0.30 | 0.02 |
| `w1` | 3.5 | 13.0 | **7.2** | 3 – 12 | 0.54 |

The anchored column is the CCC-graded built entry (OVL 0.841) discounted to the
borrowed-reference level the quantile-mapping experiment measured (0.725), then
rounded down. The unanchored column is the four-fold cross-validated recipe,
which agrees with the CCC hold-out's 0.66–0.68 — two independent routes to the
same number, which is the main reason to believe this row at all.

`w1` is the row I would bet against: the old report predicted 5.0 and the
four-fold measurement says 13.0 unanchored, so this is a substantial upward
revision on a metric nothing in the recipe targets directly.

### Sections 10–12 — demographic baselines, parity, stereotyping

| metric | target | **prediction** | range | human replication |
| --- | --- | --- | --- | --- |
| `baseline_rmse` | low | **6.0** | 3 – 11 | 2.03 |
| `parity_dpd` | low | **5.0** | 2.5 – 9 | 1.66 |
| `parity_worst` | low | **12.0** | 7 – 19 | 5.27 |
| `stereo_coef_rmse` | low | **4.5** | 2.5 – 7 | 1.26 |
| `stereo_r2_gap` | 0 | **−0.008** | −0.03 – 0.02 | 0.002 |

These are the four-fold cross-validated numbers (11.36 / 7.35 / 18.27 / 4.58)
**halved toward the shipped configuration**, and that adjustment is the weakest
step in this report. Two things justify some of it and neither is measured
end-to-end: the folds' structural donor is picked from {7B, 72B, Muse} while the
submission uses V4-Flash, which beats all three on levels and demographics by a
wide margin on Pfänder's own anchors (level error 2.59 pp against 6.06–7.74);
and the submission blends its party offsets toward external anchors, which cut
party-gap error from 19.1 pp to 3.8 in the one place it could be measured.

I have made these predictions optimistic relative to the raw fold means and I
would not defend the exact magnitudes. Treat the fold means (11.36, 7.35, 18.27,
4.58) as the pessimistic bound and these as the optimistic one.

### Confidence, stated plainly

Section 1 rests on a nested cross-validation over four studies with a measured
reliability bridge and the benchmark's own interval; it is the part I would
defend. Section 2 now rests on four out-of-fold studies rather than one, which
is a real upgrade even though the answer got worse. Section 3's unanchored half
is cross-validated twice over and its anchored half once, circularly. Sections
10–12 are cross-validated for a *donor the submission does not use*, and the
step from there to the shipped configuration is a judgment. Descending
trustworthiness in exactly that order.

### What would still make all of this wrong

**Between-study variation dominates everything else.** The basis recipe scores
0.009, 0.339, 0.344 and 0.460 on CCC, Goldwert, ICPC and Voelkel. That spread is
larger than every adjustment in this report put together, and larger than the
bootstrap interval.

**And the half-split adds a second layer of noise on top of it.** The same basis
row ranges 0.160 to 0.365 across eight splits of the *same* four studies, sd
0.079. Pfänder will be scored on one preregistered split, so this is not a
correction to apply — it is a reason the point prediction cannot be tighter than
the interval quoted above, and a warning against reading small differences
between leaderboard rows.

**The diagnostic that sets the ordering is how big Pfänder's true arm effects are
relative to their standard errors**, and the corrected numbers make that sharper
than before:

| study | arms | mean \|effect\| (pp) | human reference reliability | replication r | recipe r |
| --- | --- | --- | --- | --- | --- |
| Voelkel | 6 | 1.04 | 0.540 | 0.527 | 0.460 |
| ICPC | 11 | 4.93 | 0.497 | 0.640 | 0.344 |
| Goldwert | 10 | 3.66 | 0.245 | 0.642 | 0.339 |
| **CCC** | 9 | **1.46** | **0.213** | **0.372** | **0.009** |

Pfänder runs 1,000 per treatment arm against a 2,000 control, and its human
reference half is 500 and 1,000. If its interventions move trust by 1–2 pp like
CCC's framings, its reference half will be mostly noise and the whole
leaderboard will compress toward zero — for every entry, not only this one. If
they move it by 3–5 pp like ICPC's and Goldwert's, both the ceiling and this
score should land near the top of the intervals above.

**The structural half is validated for the wrong donor.** Every Section 3 and
10–12 number here describes a recipe whose level, offset and residual donor is
one of the three models with a CCC sample. The submission's donor is V4-Flash. A
CCC run of V4-Flash would close the single largest remaining gap in this
validation, and it is the one piece of new inference that would be worth buying.
