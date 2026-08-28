# Every defect found behind the four-study cross-validation

[← back to the verified report](four_study_cross_validation_verified.md)

Six auditors read the scripts and the package behind
[four_study_cross_validation.md](four_study_cross_validation.md), each on a
different surface, and raised 63 findings. Every CRITICAL and MAJOR one then
went to an independent adversarial verifier told to refute it and to reproduce
the defect itself before upholding it: **19 of 24 verdicts uphold the finding**,
almost all of them with a corrected magnitude, and five refute it outright.
Findings below are ordered by how much they move a reported number.

Two categories throughout:

* **POST-INFERENCE** — in analysis, scoring or calibration code. Fixed.
* **PRE-INFERENCE** — in prompt construction, instrument rendering or sampling.
  Only logged and, where possible, worked around: no new LLM inference was run
  and none is available.

The verified numbers are in
[four_study_cross_validation_verified.md](four_study_cross_validation_verified.md).

---

## The three that move a headline

### 1. CCC's donation was scored on a budget the models never obeyed

**POST-INFERENCE · fixed · the only defect that moves the four-fold means.**

CCC's donation question is a *constant-sum* item. Qualtrics holds the six boxes
to a total of exactly 100, and all 13,173 real respondents obey it because the
instrument would not let them do otherwise. The transcript the silicon samples
were drawn against rendered six independent sliders with no such constraint.

The result: 42% of Qwen2.5-7B's synthetic respondents, 13% of Qwen2.5-72B's and
1% of Muse's write a total other than 100; totals reach 600; and the scored
`Donation` composite — the sum of five of those boxes — reaches **500 on a scale
`ccc/outcomes.py` declares to be 0–100**. Nine of CCC's 81 scored pairs were
measured against a human column that is literally `100 − Donation_6`.

*Fix.* `ccc/score.prepare` rescales each respondent's six boxes to total 100,
which preserves the relative allocation — the only thing a constant-sum item
elicits — and discards an absolute magnitude the instrument never asked for. It
is the identity on the human side, and it moves the synthetic control means
*toward* the humans (65.5 → 63.2 and 66.9 → 61.6 against a human 60.6), so it is
a correction rather than a rescaling that happens to flatter.

*Impact,* isolated by re-running with the repair disabled:

| variant | CCC fold | four-fold mean |
| --- | --- | --- |
| rule: average all available | 0.134 → **0.073** | 0.354 → **0.339** |
| Qwen pair, within 0.5 (shipped) | 0.101 → **0.009** | 0.344 → **0.321** |
| 2x2: membership by prior | 0.080 → **0.009** | 0.330 → **0.311** |
| recipe, everything fitted | 0.103 → **0.039** | 0.247 → **0.231** |
| single Qwen2.5-7B | 0.163 → **0.113** | 0.303 → **0.290** |

The human ceiling does not move at 0.372: the humans were always on the right
scale. The underlying rendering defect is PRE-INFERENCE (finding 11) and cannot
be fixed; this is the best available work-around.

### 2. The whole report rests on one arbitrary half-split, and its headline claim flips across splits

**POST-INFERENCE · fixed by reporting both · the largest methodological issue.**

`load()` splits the humans with `seed = 42` and every cell of the old report —
the variant table, the fold means, the ceiling row — comes from that one draw.
The benchmark does fix one preregistered split, so a single seed is the right
model of *what Pfänder will score*. It is the wrong tool for *ranking variants*:
every variant is scored against the same reference half, so that half's sampling
noise is common to all of them, and one draw of it can reorder any pair
separated by less than it.

The verifier's independent re-run found the report's bolded conclusion —
"Refitting membership and the within factor per fold scores 0.247, below plain
Qwen2.5-7B at 0.303" — **holds in 6 of 60 alternative splits**. My own eight-split
run agrees: the fully-fitted recipe averages **0.310 against the single model's
0.254**, winning 5 of 8 splits, where at seed 42 it loses by 0.059.

The split-to-split standard deviation is **0.04 to 0.08**, larger than nearly
every difference the old table asks the reader to interpret. The variant
*ordering* is stable enough; the cross-comparison against a raw single model is
not.

*Fix.* `scripts/nested_cv.py --splits N` averages over N half-splits and reports
the split-to-split spread alongside the single-split numbers, with the
distinction stated in the output. The single split stays the default because it
is what the benchmark does.

*Impact.* The single-split numbers are unchanged and remain the right basis for
predicting a Pfänder *score*. What changes is the old report's central
qualitative claim, and it changes **in the recipe's favour**: on the split
average every calibrated variant beats every raw single model, and the
pre-committed ones win 8 of 8 splits. The old report's stated floor — "if none of
the design transfers, `pearson_r` lands near 0.26, below an uncalibrated
Qwen2.5-7B" — was the artefact. The multi-split table is in the verified report.

### 3. The party-gap blend weight was defended by arithmetic that is infeasible on its own data

**POST-INFERENCE · fixed · changes a shipped constant.**

§5 concludes that `PARTY_GAP_WEIGHT = 0.5` "sits inside" a defensible range of
0.24 to 0.6. The chain is: the pre-CCC proxy and CCC's measurement of the same
five gaps disagree by an RMS of 9.07 pp, so a single anchor carries
9.07/√2 = 6.41 pp of error; the donor sits 7.36 pp from the anchors, so its own
error is √(7.36² − 6.41²) = 3.6 pp; inverse-variance weighting therefore puts
0.24 on the anchor.

The arithmetic is internally consistent and the input is wrong. **7.36 is the
donor's distance from the *pre-CCC* eight-anchor set**, recorded one commit
before the CCC anchors existed. I re-measured it: against the shipped anchors
the distance is 8.02, and against the six CCC-backed ones — the like-for-like
comparison the argument needs — it is **6.03**, which is *below* the 6.41 the
same chain assigns to a single anchor. The implied donor variance is negative.
The model is infeasible on its own data.

Worse, the report had already measured the quantity directly two sections
earlier: the CCC hold-out puts the donor's party-gap error at **13.9 pp**, four
times the 3.6 the subtraction inferred. Both cannot be true.

*Fix.* Measured instead of argued. CCC can grade the blend with no circularity
as long as the anchor does not come from CCC: blend the model's own gaps toward
the **pre-CCC** values and grade against CCC's real humans. The least-squares
optimum over the five outcomes:

| model | optimal weight |
| --- | --- |
| Qwen2.5-7B | 1.09 |
| **Qwen2.5-72B** (the donor's model) | **0.83** |
| Muse-Glimmer-30B | 0.46 |

`PARTY_GAP_WEIGHT` raised **0.5 → 0.7**, shading 0.83 down for two things this
cannot measure: the donor is the quota-demographics run rather than the elicited
one CCC graded, and CCC's human party variable folds leaners into the two
parties while ours does not — both inflate the measured model error.

*Impact.* On the built entry, party-gap RMSE against the anchors **4.45 → 3.12
pp**, 21 outcome columns move by up to 2 pp, and the effect vector is
bit-identical (r = 1.000000) — party offsets are a demographic term and provably
cannot reach Section 1. **The entries under `data/pfander/submission` are not
regenerated; re-run `scripts/build_entries.py` to ship this.**

---

## Defects that break a computation

### 4. A composite of scored outcomes silently overwrites their targets

**POST-INFERENCE · fixed · found by building the thing rather than reading it.**

`tier1.calibrate` gives every item of a composite the *composite's* effect
vector. That is right for Pfänder, whose twelve trust items exist only to carry
`trust_multidimensional` and are never scored on their own. It is catastrophic
when the items *are* the outcomes: each one's own target is replaced and nothing
says so.

Found by declaring Voelkel's `Composite` (the exact mean of its other eight
outcomes) as a composite while building the full-recipe cross-validation. The
Voelkel fold dropped from **0.576 to 0.132** and printed a clean audit with zero
drift.

*Fix.* `calibrate` now raises when a composite's items intersect the scored
outcome set, and Voelkel's composite is dropped from the scored grid instead
(see finding 9). Test in `tests/test_calibration_folds.py`.

*Impact.* Nothing published moves — this defect was reachable only by the new
code path. It is a live trap for any future study whose composite is scored
alongside its parts.

### 5. `astype(int)` on a missing binary answer produces −2⁶³

**POST-INFERENCE · fixed · latent for Pfänder, live for ICPC.**

`tier1.calibrate_binary` moves a 0/1 outcome's per-arm rate onto target by
flipping rows. A `NaN` satisfies neither `>= 0.5` nor `< 0.5`, so those rows
were counted in the arm's denominator while being impossible to flip — driving
the rate off target by the missingness rate — and the surviving `NaN` then went
through `.astype(int)`, which yields **−9223372036854775808**.

Pfänder's `newsletter_signup` has no missing values, so the shipped entry never
met this. ICPC's `sharing` is missing on 24% of rows and produced exactly that
integer, with a drift audit reading 5.5 × 10¹⁷.

*Fix.* The rate is computed over rows that have an answer, and missing rows come
back missing in a nullable `Int64`. I verified the shipped Pfänder entry is
**bit-identical** before and after: rebuilding with the old and new function and
diffing every column gives zero differing columns.

### 6. A stringified hole became a demographic group

**POST-INFERENCE · fixed.**

ICPC's `age_band` and `ideology_band` are built with `pd.cut(...).astype(str)`,
which turns an unbanded respondent — missing age, missing ideology, or a value
outside the outermost break — into the *string* `"nan"`. That string is then a
moderator level like any other: it earns a dummy, an interaction estimate for
every arm, and a row in every demographic table. **44 of ICPC's 836 human
interaction estimates belonged to a group called `"nan"`.**

The silicon samples have no such respondents, so the subgroup grid could not be
joined at all (748 pairs against 836 expected) — which is how it surfaced, and
only because `build_subgroup_pairs` asserts the grid instead of joining loosely.

*Fix.* Fixed at the source in `icpc/score.py`, and defended in `scored._labels`,
which now treats `"nan"`, `"None"`, `"<NA>"` and `""` as holes.

### 7. CCC's synthetic and human demographic labels were different vocabularies

**POST-INFERENCE · fixed.**

CCC's human `education` comes from the released file's three collapsed levels
(`HS or less` / `Some college` / `Bachelor or Postgraduate`); the silicon
samples carry the survey's on-screen wording (`High school diploma / GED`, …).
The counts match exactly — 4765 / 4262 / 3730 on both sides — so the clone
carries the real respondent's education and this is pure relabelling. Left
undone, no subgroup pair joins and every demographic comparison is against a
reference level the other side does not have.

`party` is worse: the human variable is `PartyC3`, which folds leaners into the
party they lean toward, while the synthetic variable is the model's own
self-identification with no leaner follow-up. Mapping `Independent` and `Other`
onto `Neither` puts both sides on the same three groups, which is the most that
can be done after the fact — our `Neither` stays at 31% against the humans' 13%.

*Fix.* `ccc/score.harmonise_moderators`. Democrat and Republican are untouched,
so no party *gap* moves; the CCC hold-out's party-gap RMSEs change by under
0.03 pp.

---

## Defects in what the report claims about itself

### 8. §6 compares 81 pairs against 72

**POST-INFERENCE · logged · two auditors and two verifiers, independently.**

`effects_ancova` resolves each outcome's pre-measure by substituting `_Post` →
`_Pre`. `Donation` has no `_Post` in its name, so the substitution is a no-op and
the guard drops it: the ANCOVA grid is 8 outcomes × 9 arms = 72 while `effects`
returns 9 × 9 = 81. §6 puts the two side by side and reads the difference as the
precision the pre-measure buys.

On the same 72 pairs the simple ATE scores **0.293, not 0.372**. The correct
like-for-like statement is that the pre-measure lifts the ceiling from 0.293 to
0.624 — a *larger* gain than reported, so §6's conclusion survives and is
understated. The "cuts the standard errors by 58%" becomes 55.0% like-for-like.

### 9. Voelkel's `Composite` was scored alongside the eight outcomes it averages

**POST-INFERENCE · fixed · numerically negligible.**

`Composite` is the mean of the other eight Voelkel outcomes, exactly, to
floating-point equality. Scoring all nine puts six of the fold's 54 pairs in
twice, in their smoothest and most predictable form.

*Impact.* Measured by dropping it: **+0.003 on the human ceiling and 0.000 on
every recipe variant.** Fixed anyway, because Pfänder has no such duplication —
`trust_multidimensional` is scored and its four subscales are not — and a fold
should not be graded on a grid the target study does not have.

### 10. The CCC hold-out's "leakage guard" was a print statement

**POST-INFERENCE · fixed · three auditors raised it.**

§3 says the CCC-derived anchors are "switched off here via
`anchors.ccc.for_study('CCC')`, which returns nothing when CCC is held out". The
call's return value was printed and discarded. The grading path never applies an
anchor at all, so there was nothing for a filter to remove.

**The section's conclusion is true** — the numbers really are unanchored — but
for a different reason than the one given, and the protection was by convention
rather than by code.

*Fix.* It is now an assertion that stops the run, and the docstring says what
actually happens.

### 11. Two more §3 statements are about one model, not three

**POST-INFERENCE · logged.**

"Unanchored levels are off by 10–14 pp" and "unanchored party gaps are roughly
half their human size" both describe Qwen2.5-72B. Qwen2.5-7B is off by 14.2 pp
with gaps at a *quarter* of human size; Muse is off by 11.1 with gaps much
closer. And the stated range "our raw gaps run 6–18" is over absolute values and
hides a sign reversal on `Candidate_Post`, where the model's gap runs the wrong
way (−6.5 against a human +11.3). The correct range is −7 to +18.

The conclusion — that party anchoring is load-bearing — survives and is
understated by the sign flip.

### 12. The prediction basis disagreed with `prediction.md`

**POST-INFERENCE · superseded.**

The old report's closing section says the Pfänder basis is 0.305, reconstructed
by taking "the same position inside the bracket" as a superseded three-study
ratio. `nested_cv.py` computes the analogous quantity directly and printed
0.330, which is what `prediction.md` actually used. Two reports linked to each
other disagreed by 8%.

The verified report uses the directly computed row, which is now 0.288 after
findings 1 and 13.

### 13. The 2x2's "membership by prior" corner used a within factor fitted for a different membership

**POST-INFERENCE · fixed.**

The corner is meant to grant the membership prior and charge the within factor
as fitted. It reused `fit["within"]` — the argmax of a joint search over
*membership × within*, which belongs to whatever ensemble that search picked —
rather than refitting the factor for the prior membership. That charges the
prior membership for a factor nothing chose for it.

*Impact.* The basis row moves **0.306 → 0.288**, independently of finding 1.

### 14. `ensemble_reliability` counts the two `_demo` runs as independent seeds

**POST-INFERENCE · logged, not fixed, because the number survives.**

Two auditors flagged that `_demo` runs are quota-demographics variants rather
than seed replicates, so treating them as independent draws could overstate the
shipped ensemble's reliability of 0.957 — which feeds the shrinkage constant and
the Pfänder prediction's multiplier.

I measured it three ways instead of arguing:

| method | reliability | assumes independence? |
| --- | --- | --- |
| variance-components model in `recipes.py` | 0.9574 | yes |
| standard errors of the averaged vector | 0.9573 | yes |
| split-half over the seven runs + Spearman-Brown | **0.9610** | **no** |

The third makes no independence assumption at all — if two runs were
near-duplicates their agreement would show up as agreement and pull the estimate
down. It comes out *higher*. The pairwise evidence agrees: a `_demo` run
correlates 0.856 with its parent against 0.834 for two elicited seeds, a
difference far too small to matter.

Two verifiers then attacked the same doubt from opposite directions and landed in
the same place. One upheld the code-semantics point but measured its worth at
**0.9574 → 0.9563** — a 0.11% change in the shrinkage constant, not the 1.9% the
finding claimed. The other refuted the stated mechanism outright (the `_demo`
runs' sampling noise is not measurably shared with their parent) and put the
fully corrected figure at **0.950**. So the defensible range is **0.950–0.961**,
the Pfänder multiplier moves by at most 0.004 across it, and **0.957 stands as a
working figure.**

### 15. Several §5 and §4 figures are hand-typed and do not reproduce

**POST-INFERENCE · logged.**

Neither 9.07 nor 7.36 is computed anywhere in the repository. The §5 level-
expansion slope of 1.32 is computed on unclipped, out-of-range CCC donation
values and becomes 1.77 once the same clipping §3 applies is applied. "The
eight-run and seven-run averaged effect vectors differ by 0.049 pp against a
vector spread of 1.53 pp" does not reproduce from the runs on disk, though the
qualitative conclusion (r = 0.998, difference small against the spread) does.
The Goldwert donation rejection compares against the *party* donor rather than
the run that supplies the level, which puts the correction at 0.5 pp rather than
15.6 — strengthening the rejection.

None of these moves a headline. All of them are numbers in a report that no
script will regenerate.

---

## Pre-inference defects: logged, not fixable

These need new LLM sampling. They are recorded because they bear on how the
folds should be read, not because anything can be done about them now.

### 16. Voelkel's sliders never state their range, and four carry raw Qualtrics pipes

**PRE-INFERENCE · CRITICAL as raised.**

52 of Voelkel's 53 integer sliders reach the model with no numeric range — the
exact defect the `_v3` fidelity audit fixed for ICPC and Goldwert, where it had
compressed every effect roughly five-fold. Voelkel has no `validate` module, so
it was never checked; `models.REVISED_STUDIES` records Voelkel as not needing
the revision.

Separately, the four SUC items reach the model with **unresolved
`${e://Field/Outparty_Party}` pipes**. I confirmed this directly: the control
template contains 8 literal `${e://Field/...}` strings, on the items that ask
which candidate the respondent would vote for. The synthetic respondent cannot
tell which end of the scale is their own party.

*How to read the Voelkel fold.* A rendering defect common to every arm shifts
*levels* and largely cancels in *arm contrasts*, so the fold's Section-1 numbers
are far more trustworthy than its Section-3 ones. Voelkel's control-arm level
error is 25.6 pp in the corrected cross-validation, second worst of four. Its
Section-1 score is the best of four. Both readings are consistent with a
level-shifting rendering defect.

### 17. CCC's branch logic was never rendered

**PRE-INFERENCE.** All 16 `DisplayLogic` rules in CCC's questionnaire are
ignored, so every synthetic respondent answers all three mutually exclusive
party-specific blocks. This is an alternative explanation for CCC being the
hardest fold, and it cannot be separated from the "CCC's effects are genuinely
tiny" explanation without a re-run.

### 18. CCC's constant-sum donation was rendered as six free sliders

**PRE-INFERENCE.** The cause of finding 1. Worked around post hoc as described
there.

### 19. Goldwert's video opt-out is differential across arms

**PRE-INFERENCE.** `public_awareness` is a listwise composite over `video`, and
the model's rate of the option that maps to `NaN` swings more than 20 pp across
arms — differential attrition, which biases that outcome's ATE. It is 1 of 11
Goldwert outcomes, so 10 of the fold's 110 pairs.

### 20. Every run straight-lines multi-item batteries at 2–8× the human rate

**PRE-INFERENCE.** Roughly symmetric across arms, so it does not move
effect-recovery correlations, but it bears directly on the dispersion table and
on the fitted residual scale — the synthetic within-cell spread is not the same
kind of object as the human one.

### 21. Three cross-model run pairs share a profile set

**PRE-INFERENCE.** 19.8–25.4% identical scored cells against a 9.4–11.4%
baseline for genuinely independent pairs. §4's "every genuine replicate pair
(10 of them) | 9.4–11.5%" should read "every genuine *within-model* replicate
pair". This does not affect §4's conclusion about `qwen25_72b_seed3`, which sits
at 77% and is in a different regime entirely.

---

## Raised and refuted

Three findings went to a verifier and did not survive. Recording them because
the reasoning is the useful part:

* **A negative fitted `kappa` would flip the sign of the held-out r.** The
  algebra is right and the conclusion is wrong: `nested_cv` scores the vector the
  recipe *emits*, and if `kappa < 0` that vector really is anti-correlated. The
  printed number is the honest score of the thing that would be submitted. The
  most negative `kappa` anywhere in the 660-point search is −0.0025 and sits at a
  grid point no printed row addresses.
* **The §3 table should have used the shipped `RESIDUAL_SCALE = 1.12` rows.**
  The observation reproduces — the ×1.12 rows are worse on most shape metrics —
  but the preferred fix would publish a false claim, because those rows are not
  the shipped structural calibration either.
* **Stale `se_l` makes `beta_adj` wrong.** The code fact is real and I fixed it
  anyway, but the claimed failure scenario is false: `beta_adj` is never printed,
  and the folds the finding names have finite values.
* **`WITHIN_SHRINK = 0.5` is scored as pre-committed but was fitted.** The
  provenance claim is a misdiagnosis — git history shows the constant did not come
  from where the finding says — and the selection optimism it buys is **0.005**,
  two grades below the severity claimed. The docstring in `nested_cv.py` now
  states the measured figure rather than the insinuation.
* **The `_demo` runs are not independent seeds.** Refuted on the mechanism: their
  noise sharing with the parent is essentially absent, measured three ways. See
  finding 14.

## What was checked and found sound

Worth stating, because a defect list reads as if everything is broken:

* The metric implementations in `benchmark/metrics.py` are a faithful,
  function-by-function translation of the benchmark's `R/functions/statistics.R`.
* Scoring against a **half** human sample is correct — the benchmark really does
  score every submission against Human 1.
* Converting every estimate to **percentage points of scale range** before
  pooling is correct, and is what the benchmark's preregistration specifies,
  explicitly over the alternative of pooling raw mixed units.
* The `contains("ontrol")` control-arm mask picks exactly the right rows in all
  four studies, including CCC's three pooled placebo arms, and agrees with each
  study's own `effects()` reference level.
* No anchor derived from a cross-validation fold reaches the effect
  cross-validation.
* No run in any of the four studies is truncated, missing an arm, or a duplicate
  draw of another.
* The model rows and the human-replication row are scored on the same grid.
* §4's duplicate-run finding is correct, and dropping `qwen25_72b_seed3` was the
  right call.
* The quantile-mapping table, the Goldwert `$5`-mode numbers, and the
  "9% noisier than CCC" arithmetic all reproduce exactly.

## Reproducing this

```
python scripts/nested_cv.py --bootstrap 2000        # Section 1, one split, with intervals
python scripts/nested_cv.py --splits 8              # the same, averaged over eight splits
python scripts/nested_benchmark.py                  # all four benchmark sections
python scripts/score_ccc_holdout.py                 # the structural hold-out
python -m pytest tests/test_calibration_folds.py    # the guards on the fixes
```

`openpyxl` is required for `tests/test_icpc.py` and is not in the container
image; three ICPC tests fail without it for that reason alone.
