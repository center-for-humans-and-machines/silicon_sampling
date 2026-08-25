# [ACTIVE] Optimize the Pfänder megastudy prediction

Task: [docs/tasks/macro_task.md](../tasks/macro_task.md)

Deliver calibrated Tier-1 Pfänder predictions by **2026-08-29 15:00** (156 h from
now), chosen by cross-validation on studies that publish real participant data,
plus a report saying which model should win and why.

Everything below rests on numbers that are either measured on disk or verified in
this session. Where a recon agent's claim did not survive checking, the corrected
version is used and the error is named.

---

## 0. The finding that reorganises the whole task

**Scoring is multi-objective.** The benchmark scores ATE recovery (directional %,
Spearman, Pearson, `pearson_within`, `pearson_adj`, RMSE, `rmse_adj`), the
calibration regression (α, β, `beta_adj`), response distributions (variance ratio,
OVL, KS, W1), subgroup effects, demographic baselines, parity gaps and
stereotyping coefficients. The *leaderboard sort key* is pooled Pearson r —
`arrange(desc(submission == "Human replication"), desc(pearson_r))` — and the
preregistration names r the "key metric" for Sections 1–2 and the variance ratio
the key metric for Section 3, but it explicitly declines to collapse them into a
composite. **We optimise all of them.**

That decision is what makes calibration central rather than decorative. A global
effect rescale cannot move the sort key, but it is close to the whole story for β
and RMSE; level and dispersion repair are the whole story for Section 3; and only
a per-outcome re-profiling moves r. These are largely *independent* levers on
*different* metrics, so the right posture is a portfolio, not a single bet.

Decomposing the human effect vector on Voelkel (the one study where we have both
a silicon sample and real responses), verified this session:

| quantity | value |
| --- | --- |
| share of `var(estimate_h)` that is **between-outcome** | **0.566** |
| pooled r of an oracle predicting **only the per-outcome mean human ATE**, zero message information | **0.752** |
| pooled r, our Qwen2.5-7B sample | 0.408 |
| pooled r, a **fresh human replication** of the same size | 0.514 |
| correlation of our 9-outcome profile with the true one | 0.473 |

**A predictor that knows nothing about which message works, but knows the
13-number profile of which outcomes move and by how much, beats a real human
replication by a wide margin.** Ranking the 16 messages is the minority
shareholder in the metric we are scored on.

So the largest single lever on the sort key is **a calibrated per-outcome effect
profile for Pfänder**, and the calibration studies exist to supply it. Message
ranking is a secondary concern that we improve where it is free. But the profile is
one lever among several, and the metrics it cannot touch — RMSE, β, and every
distributional and demographic diagnostic — are reached by the other calibrations
in §2, which is why the plan runs all of them rather than picking one.

### The contradiction in the recon, resolved

Two recon agents disagreed on whether calibration can move `pearson_r` at all.
Settled empirically — reproduce with
[`scripts/verify_calibration_levers.py`](../../scripts/verify_calibration_levers.py),
which is where every number in this section comes from:

- A **single global** multiplicative rescale `l → k·l` leaves `pearson_r` at
  0.4084 for every k ∈ {0.5, 0.159, 0.05} — bit-identical, as the algebra
  requires. Also invariant: `spearman_rho`, `directional_pct`, `alpha`,
  **and `pearson_adj`**.
- A **per-outcome** re-profiling is not a global rescale and moves r a lot:
  Qwen 0.408 → **0.491**, V4-Flash 0.190 → **0.465** (w = 1, true anchor).

So "calibration cannot move r" is true only of the one calibration everybody
thinks of first. Correction to the recon: `pearson_adj` is computed from `se_h`
only (`benchmark/metrics.py:adjusted_metrics`), so shrinkage cannot damage it —
the claimed "λ cliff destroys 2 of 10 numbers" costs **only `beta_adj`**, and on
Voelkel `beta_adj` stays finite and well-behaved (1.087) at the optimal k. This
materially cheapens the plan: **we do not need to quadruple N to 72k rows.**

---

## 1. What the calibration data actually is

Six candidates in `data/calibration/`. Three hard gates: questionnaire
reconstructible, textual intervention, participant-level responses.

| study | gate 1 | gate 2 | gate 3 | verdict | role |
| --- | --- | --- | --- | --- | --- |
| **Voelkel** (SDC) | ✓ | ✓ | ✓ 35,252 | **effect study** — already built | 6 arms × 9 outcomes = 54 pairs; expandable |
| **Vlasceanu / ICPC** | ✓ (qsf) | ✓ 12 arms, text+static image | ✓ US 8,253 | **effect study** — build it | 11 arms × ~6 outcomes; the transfer test |
| **Goldwert** 2026 | ✓ (qsf, OSF) | ✓ 18 arms, 5 video / 2 image-of-text | ✓ 31,324 | **effect study — third, if time** | closest sibling to Pfänder; has donation + newsletter items |
| **Doell** 2024 | — | — | — | **duplicate of Vlasceanu** | see below |
| **Cologna / TISP** | ✓ (full Qualtrics master) | ✗ observational | ✓ US 2,559 | **level anchor only** | same 12-item Besley trust battery as Pfänder's *primary* |
| **CCAM** Fall 2024 | ✓ | ✗ observational | ✓ 35,309 | **level + demographic anchor** | nationally representative; the demographic joint |

**Doell = Vlasceanu, verified this session.** 99.1% of `ResponseId` values overlap
(58,928 of 59,440) and the 12 arms are identical bar two renames. Doell et al.
2024 (*Scientific Data*) is the data descriptor for the same ICPC collection that
Vlasceanu et al. 2024 (*Sci. Adv.*) analyses. Doell's 668-column raw export is the
better file to use (more outcomes, full randomiser record); Vlasceanu's 28-column
file is the cleaned analysis extract. **This is one study, not two** — a
correction that matters, because it halves the number of independent transfer
tests available.

So: **two effect-calibration studies are comfortably affordable (Voelkel,
Vlasceanu), a third (Goldwert) is the stretch goal**, and two observational
studies serve as level/demographic anchors at zero sampling cost.

### Why the two "unusable" studies are load-bearing anyway

They fail gate 2 and cannot calibrate effects, but they are the *only* ground
truth we have on Pfänder's own constructs:

- **TISP** carries the identical four-facet Besley trust battery (competence,
  integrity, benevolence, openness — 11 of 12 stems near-verbatim against
  Pfänder's), the NORMPERC items that became Pfänder's `policy_role`, and two
  verbatim `policy_specific` items. It is the only local ground truth on the
  **primary outcome's level**.
- **CCAM** gives a nationally representative weighted joint over
  age × gender × race × education × income × party, and ground-truth US levels and
  subgroup gaps on climate belief, worry and policy support.

---

## 2. The calibration menu, with verified payoffs

All numbers from `scripts/verify_calibration_levers.py`, Voelkel, 54 pairs,
scored with the benchmark's own `pooled_metrics`.

### Tier A — free, no external data, works on samples already on disk

| # | calibration | r | ρ | dir | RMSE | β |
| --- | --- | --- | --- | --- | --- | --- |
| — | Qwen2.5-7B raw (incumbent) | 0.408 | 0.311 | 61.1 | 3.620 | 0.159 |
| A1 | + global shrink k = 0.16 | 0.408 | 0.311 | 61.1 | **1.402** | **1.001** |
| A2 | + **within-outcome shrink κ_w = 0.5** | **0.439** | **0.402** | 63.0 | 2.468 | 0.252 |
| A3 | + cross-model average (Q, D) | **0.428** | **0.410** | 57.4 | 2.318 | 0.267 |
| A2+A3 | both | **0.484** | **0.484** | 61.1 | 1.671 | 0.455 |

Two of these were not anticipated by the prior reports and are the plan's cheapest
real wins:

**A2 — down-weight the message-ranking component relative to the outcome
profile.** Shrinking only the within-outcome deviations, keeping our own profile,
*raises* r from 0.408 to 0.439 and ρ from 0.311 to 0.402. Mechanism: it
re-weights the composite toward the component we predict better. Requires no
external information at all. The design agent's LOCO check found the *direction*
holds 6/6 folds but the *magnitude* only 3/6, so we adopt κ_w ≈ 0.5 and refuse to
tune it further.

**A2 is model-dependent, and that is the argument for validating rather than
assuming.** On V4-Flash the same κ_w = 0.5 nudges r up (0.190 → 0.203) but pulls
ρ *down* (0.186 → 0.135) and directional % down (55.6 → 51.9). The lever works by
re-weighting toward whichever component a given sampler predicts better, so it has
to be re-fit per model rather than adopted as a constant. Qwen has a usable
outcome profile to lean on; V4-Flash has less of one.

**A3 — average the two models' effect vectors.** r 0.408 → 0.428, ρ 0.311 →
0.410, and the average beats the better model at every (w, κ_w) combination
tested. Paired cluster bootstrap: Δr = +0.020 [−0.050, +0.152], p(better) = 0.73;
Δρ = +0.099 [−0.088, +0.315], p = 0.80. Not established, but free, positive on
both correlation metrics, and the cross-model effect correlation is only +0.091,
so the two samples are close to independent draws on the same signal. Caveat:
`directional_pct` *falls* (61.1 → 57.4).

**A1 is hygiene, not an edge.** It provably cannot move a leaderboard place
(r-invariant) and its margin over the trivial "predict the mean human effect"
entry is 0.135 pp of RMSE. Do it once, correctly, then stop.

### Tier B — needs the calibration studies, and has a break-even

**B1 — anchored outcome-profile substitution.** Replace our per-outcome mean ATE
with one transferred from the calibration studies:
`m̂_j = w·â_j + (1−w)·m̄_j`, keeping our within-outcome deviations.

With a *perfect* anchor this is the single largest lever on the board. But the
anchor will be noisy, and **there is a break-even**:

| ρ(anchor, truth) | 0.2 | 0.4 | 0.6 | **0.8** | 1.0 |
| --- | --- | --- | --- | --- | --- |
| resulting pooled r | 0.293 | 0.329 | 0.392 | **0.438** | 0.491 |

Our own profile already achieves ρ = 0.473 and r = 0.408. **A transferred anchor
must reach ρ ≈ 0.7 with the truth before it beats doing nothing**, and an anchor
at ρ = 0.4 would cost us 0.08 of r. This is the plan's single biggest open
empirical question, and §5 is built around measuring it rather than assuming it.

**B2 — flatten structurally unidentified outcomes.** `belief_post` and
`donation_ams` have `true_sd_effect_pp = 0.000` in the Pfänder sample: their
observed between-message spread is entirely sampling noise. Shrinking them hard
removes covariance-free variance from `sd(l)`. Needs no ground truth — our own
diagnostics suffice — worth ×1.04–1.08 on r.

### Tier C — off the leaderboard, but scored and reported

| # | calibration | fixes | anchor |
| --- | --- | --- | --- |
| C1 | per-outcome level recentring | W1 22.9 → ~8 pp, OVL, KS, demographic baselines | TISP (primary), CCAM (belief/concern/policy) |
| C2 | per-outcome dispersion match | `variance_ratio` — Section 3's headline (Qwen 1.13, V4 1.44, composite 0.49) | human SDs from Voelkel/CCAM |
| C3 | demographic prefill | the income reference cell (§4) | CCAM joint |
| C4 | heaping / midpoint / straight-lining jitter | OVL, KS, W1 | human response distributions |

C1 and C2 buy **nothing** on the sort key. They are worth doing because the
benchmark reports them, `variance_ratio` is explicitly the key metric of its own
section, and a submission that is right on levels is more defensible. They are
also the only place V4-Flash's advantages show up.

### Rejected outright

- **Per-outcome free shrinkage factors (empirical Bayes / James–Stein).** The
  *oracle* upper bound is +0.003 on r. Thirteen parameters for nothing.
- **Constant-effect submission.** Zero within-outcome variance makes `pearson_r`
  undefined and sorts last. Keep it only as the RMSE floor a real entry must beat.
- **Subgroup-interaction shrinkage.** All four scored subgroup metrics are
  scale-invariant, so shrinking buys exactly zero — and shrinking to *exact* zero
  makes `pearson_r` NaN. Voelkel shows visible moderators (r = 0.236) predict no
  better than invisible ones (0.237), so there is no signal to rescue either.
- **Reliability-weighted trust composite.** `make check`'s 0.5-point
  composite-consistency tolerance caps the achievable change at ~1 pp.
- **Quantile-mapping response distributions** except on outcomes with a real
  anchor — it perturbs every ATE non-linearly and invalidates the fitted κ.

---

## 3. Compute and the schedule

Measured throughput, from `run_meta.json` files on disk:

| model | hardware | group | resp/h | Pfänder 18k | Voelkel 6.2k |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 1×RTX 4090 (local, free) | 15 | 1,344 | 13.4 h | 3.7 h |
| DeepSeek-V4-Flash | 4×H200 | 30 | 1,022–1,245 | 14–16 h | 2.6 h |
| Qwen2.5-72B | 4×H200 | ~128 (est.) | **~2,400 (1,900–3,600)** | **5–9.5 h** | 1.5–3 h |

**Qwen2.5-72B base is already on DAIS** — verified live:
`/opt/huggingface/hub/models--Qwen--Qwen2.5-72B`, 37 shards, `total_size`
145.4 GB bf16, `Qwen2ForCausalLM`, 80 layers, hidden 8192, 64 q / 8 kv heads. It
should be **~2× faster per respondent than V4-Flash**, not because its step is
faster (at group 128 it is ~1.5× slower) but because 145 GB of dense weights and
320 KB/token of GQA-8 KV buy a 4× larger group than V4-Flash's ~1 MB/token hybrid
cache. That estimate is *derived arithmetic*, and the last derived config in this
project missed by 13× (81 resp/h against ~1,100 predicted), so it is gated on a
pilot.

**DAIS GPU-hours are not the constraint.** Demand is ~35–55 runtime-h; supply is
~104–208. The constraints are (a) **job round-trips** — observed 4-GPU queue waits
0.01, 0.13, 1.04, 2.54, 7.04, 8.08 h, median ~1.8 h, and 8-GPU jobs have sat 10.75 h
and been abandoned — and (b) **agent-days to build new study packages**, empirically
1.5–2 days each. Local 4090 time is free and currently 100% idle.

Rules: never request 8 GPUs; 4–8 h walls, never 24 h; two jobs in flight; resume
by resubmitting the identical body (`scripts/dais_auto_resume.sh`).

### Day by day

| day | DAIS A | DAIS B | local 4090 | agent |
| --- | --- | --- | --- | --- |
| **Sat 23** | 72B pilot, 2×H200, 1.5 h wall, 300 resp | — | prefill A/B pilot (3.6 h) | **Phase 0: ship the MVD.** Python format checker, insurance CSV, Tier-A calibrations |
| **Sun 24** | 72B × Pfänder (8 h wall) | 72B × Voelkel | Qwen-7B × Pfänder prefill (11 h) | Vlasceanu build |
| **Mon 25** | resume 72B Pfänder | V4-Flash × Voelkel prefill | Qwen-7B × Voelkel prefill | Vlasceanu build; **transfer test as soon as one arm lands** |
| **Tue 26** | V4-Flash × Pfänder job 1 | 72B × Vlasceanu | Qwen-7B × Vlasceanu | fit calibrations on partials; intermediate report |
| **Wed 27** | V4-Flash × Pfänder job 2 | Goldwert *if green* | spare | **LOSO + LOO-O; freeze the calibration choice** |
| **Thu 28** | resumes only. **No new full Pfänder job after 12:00** | resumes only | — | apply calibrations, build 3 entries, pick the model |
| **Fri 29** | **freeze 03:00** | — | — | `make check` equivalent, report, deliver by 15:00 |

**Point of no return:** a fresh 18,000-respondent Pfänder DAIS job needs p75 queue
(7 h) + runtime (5–16 h) ≈ 19 h worst case, so the last launch is **Aug 28 12:00**.
The 9,000-row precision floor (2.5–4.5 h) can launch as late as **Aug 28 20:00**; a
resume of a ≥90%-complete run as late as **Aug 29 03:00**. After that, ship what
exists.

---

## 4. Demographic prefill

Today gender, age and race are pre-filled from the preregistration's census
quotas (`pfander/profiles.py`); education, income and party are *generated by the
model*, and they come out badly wrong:

| | Qwen-7B | V4-Flash | CCAM truth |
| --- | --- | --- | --- |
| Republican | 13.1% | 18.5% | 29.1% |
| income < $30k | **0.8%** | 5.7% | 13.1% |
| income ≥ $168k | 13.9% | 31.6% | 20.7% |

**The sharpest consequence is structural, not cosmetic.** `income`'s dummy-coding
reference level is `Less than $30,000`, and Qwen produces **144 such rows in
18,000 — 8.5 per arm**. Every one of the 80 scored income interaction estimates is
measured against an 8-respondent reference cell, and the `min_n = 30` threshold
**silently skips** that group in two further reported sections. Prefill takes it to
~1,170 rows (69/arm). Prefill also cuts ~20 of ~98 transcript slots (−17% runtime)
and removes the two largest rejection sources (party 2,602, religion 2,241 draws in
the V4 run).

**But it is unlikely to move the leaderboard.** Qwen is demographically inert
(party gap on the primary outcome +0.48 pt; moderator R² beyond condition 0.0020),
so reweighting its composition to truth moves the primary level by < 0.1 pt. For
V4-Flash the same reweighting moves the level by ~−2.1 pt and ATEs — second
differences, interaction R² ≈ 0.01 — by ≲ 0.5 pt against an ATE SD of 3–4 pt.
Expected Δ`pearson_r` ≈ 0.02, a tenth of the bootstrap half-width.

**Resolution: prefill everything from here on, but do not spend DAIS hours
re-running finished samples to get it.** Build the module, use it for every new
run, resample Pfänder with prefill *locally* on the free 4090 and on DAIS only for
the model chosen as primary.

### The pilot that decides it (Sat 23, local, ~3.6 h, free)

5,000 control-arm sessions with Qwen-7B, reusing the existing `profiles.csv` seeds
and the already-prefilled gender/age/race so the only contrast is prefill:

1. 1,000 with (education, income, party) prefilled from the CCAM joint;
2. 1,000 forced to (Republican, < $30k);
3. 1,000 forced to (Democrat, ≥ $168k);
4. 1,000 control + 1,000 of the largest-|ATE| arm under representative prefill.

Pre-committed gates: (i) the forced Rep/low vs Dem/high gap on
`trust_multidimensional` — SE 0.98 pt, MDE 2.7 pt — **if < 5 pt the model is inert
and no DAIS Pfänder resample is justified for prefill alone**; (ii) ΔATE against
the existing run on the matched arm (SE 1.39, MDE 3.9 pt); (iii) rounds/respondent
falls ≥ 15% with `forced == 0`; (iv) the representative third reproduces the CCAM
cell table exactly — a deterministic code test, not a sampling test.

---

## 5. Cross-validation, and the discipline that keeps it honest

The task asks for item-level and study-level leave-one-out CV. With two or three
usable studies, **study-level LOO is a 2–3-point transfer check, not
cross-validation, and the plan will say so in those words.**

**Folds.**
- **LOSO** — {Voelkel, Vlasceanu, (Goldwert)}. Fit free parameters on K−1, score
  the held-out study with `pooled_metrics` + `run_calibration_pooled`. 2–3 folds:
  the *sign* of a parameter and whether one calibration dominates in K of K folds
  is all that is available. No mean-across-folds standard error will be computed or
  reported.
- **LOO-outcome** — 9 folds on Voelkel, ~6 on Vlasceanu. Measured: global-κ beats
  raw in 8/9, out-of-fold κ 0.132–0.183.
- **LOO-condition** — 6 folds on Voelkel. Global-κ beats raw 6/6, κ 0.109–0.183.

**The pre-commitment, written before any fitting:**

1. **Two free parameters, total.** `κ_w` and the anchor weight `w`. Anything with
   a per-outcome parameter is rejected a priori — the oracle gain is +0.003.
2. **K-of-K rule.** Adopt a calibration only if it beats the simpler nested
   alternative on `pearson_r` in *every* LOSO fold, and on RMSE in ≥ K−1.
3. **Parameter stability as a substitute for a CI.** Accept a parameter whose
   out-of-fold estimates all lie within a factor of 2 (κ: 0.109–0.183 ✓; the
   between/within ratio: 1.43–3.12 ✗, so take its sign only and cap κ_w at 0.5).
4. **No metric shopping.** `pearson_r` is the selection criterion, RMSE a
   constraint (must beat the constant-anchor floor). Ties break toward fewer
   parameters.
5. **Never tune on Pfänder's own diagnostics**, except the two knobs that need no
   ground truth: flattening `true_sd ≈ 0` outcomes, and cross-model agreement.

**The one measurement that decides Tier B.** Compute the per-outcome-family human
ATE profile for Voelkel and Vlasceanu independently, map both onto a shared family
taxonomy (belief, concern, policy-general, policy-specific, trust, behaviour,
donation/costly), and correlate them. **If cross-study profile ρ < 0.7, the anchor
does not clear the break-even in §2 and we set w = 0** — keeping only Tier A, which
is already +0.08 on r. If ρ ≥ 0.7, w is chosen by LOSO within the cap. This single
number is the plan's main scientific result regardless of which way it lands.

---

## 6. How a calibration becomes a Tier-1 CSV

The benchmark refits effects from respondent-level data
(`lm(outcome ~ condition)`, HC2, no covariates), so a calibration is expressed by
transforming respondent values:

```
y'_ijc = μ*_j + λ_j · (y_ijc − ȳ_jc) + ê_jc · scale_range_j / 100
ê_jc   = κ_b · m̂_j + κ_w · (d_jc − m̄_j),     ê_j,control ≡ 0
m̂_j    = w · â_j + (1 − w) · m̄_j
```

Subtracting the *observed* condition mean `ȳ_jc` and adding the target effect
makes the refit ATE exactly `ê_jc`, while `λ_j` controls within-condition
dispersion independently — so `variance_ratio`, OVL, KS and W1 are untouched by
the effect calibration, and vice versa. The two levers are orthogonal, which is
what makes it safe to optimise the leaderboard metric and the reported
distribution metrics at the same time.

**This is post-processing of simulated responses, and it must be disclosed as
such** — see the open question in §9. The raw, untransformed export goes into
`raw_data_deposit/` exactly as the submission template requires, so the
transformation is visible to anyone who looks.

Mandatory guards before any submission:
- `assert_full_grid`: exactly 208 pairs, no NA. One mislabeled condition **halts
  the benchmark's pipeline**.
- The 17 condition strings and all 6 moderator level strings must match
  `codebook.csv` byte for byte — including the reference levels **Male / 18-29 /
  White / Caucasian / Less than high school / Less than $30,000 / Republican**.
- Composite consistency: transform the 12 trust items alongside
  `trust_multidimensional`, staying inside the 0.5-point tolerance.
- `λ = 1 − mean(se_l²)/var(estimate_l) ≥ 0.5` so `beta_adj` stays finite.
- If the response-distribution repairs (C1/C2/C4) run, recompute all 208 effects
  afterwards and require |Δ effect| < 0.1 pp before applying the effect map.

---

## 7. Work plan

### Phase 0 — ship a submittable prediction on day 1 (Sat 23, ~8 h, zero GPU)

Nothing else starts until this is green. Every later phase then *replaces files at
the same path* rather than being the only copy of the deliverable.

1. **Port the format gate to Python.** `Rscript` is not installed in this
   container and there is no `r-base-core` in apt, so `make check` — the entire
   format validation, 757 LOC of R across `check_lib.R` and `clean.R` — is
   currently unrunnable. Port `check_repo` + `submission_spec.R` to
   `silicon_sampling/submission/check.py`. Green it against the existing
   `v4_flash/tier1_submission.csv` by midday.
2. **Insurance submission** from the Qwen-7B Pfänder sample already on disk
   (18,000 rows, 2× the precision floor): `metadata.json`, SHA-256 manifest,
   `predictions/<team>_T1_primary_v1.csv`.
3. **Tier-A calibrations** as a CSV→CSV transform (§6), with A1/A2/A3 measured on
   Voelkel and applied to Pfänder.
4. **Fix the two benchmark bugs** that corrupt scored quantities:
   `analysis/ols.py:design_matrix` defaults to the alphabetically-first level,
   which is wrong for 5 of 6 moderators and corrupts every interaction and
   stereotyping estimate; `moderators.py` uses HC1 where the spec says HC2.
   One line each plus a unit test asserting the six reference levels.

### Phase 1 — demographics (Sat 23 – Sun 24)

5. CCAM weighted joint over the six moderators at Pfänder's codebook levels, with
   the covariance structure; conditional sampler that respects the preregistered
   gender × age and gender × race margins exactly (those are published quotas and
   override CCAM).
6. Prefill module in `survey/slots.py` / `session.py`; all six demographics
   injected. Run the §4 pilot and record the verdict.

### Phase 2 — the second study (Sun 24 – Mon 25)

7. **Vlasceanu / ICPC package**: vendor the three US `.qsf` files, build
   `silicon_sampling/vlasceanu/` on the `voelkel/` template (`qsf.py` is already
   study-agnostic), 12 arms, filter to the US quota subsample (n = 8,253), modality
   audit, human reference + half-split scoring.
8. Local Qwen-7B sample (6.5 h, free) → the **transfer test of §5** as soon as it
   lands. This is the gate on Tier B.

### Phase 3 — sampling campaign (Sun 24 – Wed 27)

9. 72B pilot → gate: ≥ 1,500 resp/h at group ≥ 64, else drop 72B and give its
   hours to V4-Flash. Pre-submit ritual for **every** DAIS job: 20-session local
   smoke run, eyeball 3 transcripts, assert no marker leak and that the prompt ends
   at `"Response: "`. Three prefill bugs were caught this way on Voelkel; the
   ritual is not optional.
10. Prefilled Pfänder on the primary model; prefilled Voelkel and Vlasceanu on
    the DAIS models as slots free.

### Phase 4 — calibration selection (Wed 27)

11. LOSO + LOO-O + LOCO under the §5 pre-commitment. Freeze the choice Wed
    evening; nothing after that changes the estimator.

### Phase 5 — deliver (Thu 28 – Fri 29)

12. Three Tier-1 entries (§8), format-checked, with the raw deposit.
13. Report in `docs/reports/pfander_calibration/`: summary plus sub-reports on
    dataset assessment, the outcome-profile finding, the bias catalogue, the CV
    result, and the model prediction. Intermediate reports land as partial results
    arrive, per the task's instruction.

### Cut from scope, deliberately

- **Goldwert** unless Phase 2 finishes a day early. It is the closest sibling to
  Pfänder and would give a second transfer test, so it is the *first* thing added
  back if there is slack — but it needs an OSF fetch, 18 arms, 5 video arms and 2
  image-of-text arms, at 2–3 agent-days.
- **TISP as a sampled study.** Used as a level anchor only.
- **Voelkel arm recovery** (6 → 17 arms). Tempting — it is the only way to shrink
  the ±0.17–0.28 CI on Δr between models — but it reshuffles every profile and
  `instrument.header()` indexes off `CONDITIONS`.
- **The four missing *reported* benchmark analyses** (demographic baselines,
  parity gap, predictability, within-subgroup distributions). None is on the
  leaderboard.
- **Buying N to 72k rows.** Dropped once `pearson_adj` was shown to be
  shrinkage-invariant; the only casualty was `beta_adj`, which survives at the
  relevant κ.

---

## 8. Which model, and the three entries

Measured on Voelkel, the only ground truth we have:

| | dir % | ρ | **r** | RMSE | levels |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 61.1 | 0.311 | **0.408** | 3.620 | 22.9 pp error |
| DeepSeek-V4-Flash | 55.6 | 0.186 | 0.190 | **2.808** | **8.0 pp** |
| Human replication | 66.7 | 0.395 | 0.514 | 1.682 | — |

**The leaderboard sorts on r, so the prediction is Qwen2.5-7B — the free, local,
7-billion-parameter model — or an ensemble containing it.** V4-Flash wins exactly
the things calibration can synthesise (levels, dispersion, non-flat demographics)
and loses the one thing it cannot (ordering). Note honestly that Δr = −0.218
[−0.428, +0.137]: with six clusters the two models are not distinguishable, and
adding a third model does not fix that — only adding clusters would. Also
V4-Flash's directional score (55.6) equals the "all positive" constant baseline.

The cap is 3 entries per tier, 9 per team, exactly one `primary`; every entry
enters the leaderboard and the field statistics, and the benchmark preregisters no
test between approaches. So hedging is free. The three entries should differ along
**the one axis CV cannot resolve** — the anchor weight — not be near-duplicates:

| entry | content | hedges against |
| --- | --- | --- |
| `primary` | best CV configuration: ensemble or Qwen, κ_w ≈ 0.5, w from LOSO, B2 flattening, A1 shrink, C1/C2 level+dispersion repair | — |
| `secondary-1` | same pipeline, **w = 1** (profile fully external) | our own profile being worthless |
| `secondary-2` | same pipeline, **w = 0**, global κ only — one free parameter | the anchor being worse than nothing |

`secondary-2` doubles as the diagnostic reference showing what calibration bought.

---

## 8.5 Progress and corrections (2026-08-23)

Three of the plan's own defaults have been overturned by measurement. They are
recorded here rather than edited away, because the reasoning that produced them
was sound on the evidence available and the corrections are the useful part.

**Corrected: global effect shrinkage is not a transferable default.** §2 A1 called
κ = 0.159 "mandatory hygiene" on the strength of its stability across Voelkel
folds (0.137–0.185). That stability is *within* one study, and the quantity is not
a property of the sampler — it is the ratio of real effects to ours. Mean signed
human intervention effect, measured from each study's own participants: **Voelkel
1.125 pp, Goldwert 2.967, ICPC 5.035** — a 4.5-fold range. Our averaged Pfänder
effects run 2.467 pp, so the implied factor is 0.46 / 1.20 / 2.04 against the
three. **The range spans 1.0, so the direction of the correction is undetermined**,
and Pfänder is a climate study, making Voelkel the worst of the three to fit on.
Applying 0.159 would have shrunk our effects to 0.39 pp against a climate
reference of 3–5 pp. So "base models exaggerate effects 3–6×" is substantially a
Voelkel artefact. The primary entry applies no global shrinkage.

**Corrected: scale does not help effect recovery, now shown within a family.**
Qwen2.5-72B finished both studies at 2,465 resp/h on 4×H200 — twice V4-Flash, as
derived. It is *worse* than Qwen2.5-7B on Voelkel: r 0.340 against 0.408, ρ 0.189
against 0.311, directional 51.9% against 61.1%. Its level error stayed at 22.4
against 7B's 23.7, so V4-Flash's much better 8.3 is not a size effect.

**Added: averaging models is the largest free gain.** The samplers' errors are
close to independent (7B↔72B effect correlation +0.315, 7B↔V4 +0.091), so
`avg(7B, 72B)` reaches r = 0.460 — better than any single model. Combined with
within-outcome shrinkage and V4-Flash's context it reaches **r 0.576, ρ 0.507,
RMSE 1.256, β 0.987**, beating a fresh human half sample on five of eight
Section-1 metrics. That wins by being *smoother*, not wiser: the noise-corrected
between-arm effect SD has a **median of 0.000 in all three calibration studies**,
so at this sample size most of the truth about which message works is unmeasurable
and a predictor that declines to bet on it beats a noisy half sample.

**The separability §6 assumed is confirmed.** Recipes differing only in the effect
transform have bit-identical distribution and demographic numbers, and vice versa,
so the two families combine and the combination inherits the best of each.

### Delivered

- **Three format-valid Tier-1 entries** at `data/pfander/submission/{primary,
  secondary-1,secondary-2}`, all PASS WITH WARNINGS (41 checks, 0 failures; the
  two warnings are the human registration checklist and Zenodo metadata).
- Five silicon samples on Pfänder (7B, 7B-replicate, 72B, 72B-replicate, V4-Flash)
  and three on Voelkel; **ICPC and Goldwert sampling in flight**.
- All six demographics pre-filled: the `Less than $30,000` reference bracket goes
  from 139 of 18,000 (18 in the control arm) to 2,321, against a CCAM target of
  13.54%.
- Level anchors from TISP and Goldwert: 3 outcomes at grade `near`, break-even
  anchor error measured at 4.7 points. V4-Flash's Pfänder levels sit 2.0–3.1
  points from the TISP anchor where Qwen is 14.2 and 9.4 off — corroborating the
  hybrid's level source from data that had nothing to do with choosing it.

### Still outstanding

1. Genuine vLLM decode on ICPC and Goldwert was unexercised at build time (the
   4090 was full); the DAIS runs were the first real test, and they passed.
2. The response-shape gap: OVL 0.781 against a human replication's 0.925.
3. Team metadata for `metadata.json`, which the user fills at the end.

## 8.6 Progress (2026-08-25)

### The fidelity audit, which reset most of the calibration

The ICPC and Goldwert templates printed slider endpoint labels without stating
the 0-100 range, so models answered on an implicit 0-10 scale. Fixing it cut
mean control-arm level error from 30.7 to 9.7 pp on Goldwert and 38.8 to 18.3 on
ICPC, and — because it compressed the *effects* by the same factor — invalidated
every constant fitted before it. Four further defects were fixed alongside it.
All four `_v3` re-samples are in (7B locally, 72B and V4-Flash on DAIS).

### Delivered since

- **Leave-one-study-out restored to three folds.** Only ICPC and Goldwert were
  re-sampled, because only their templates changed; a study with no `_v3`
  directory was being read as unsampled, which silently cut the fold set to two.
  `models.resolve_run` now falls back to the pre-audit run.
- **Shrinkage re-derived, twice.** Refitting on audited samples roughly doubled
  it; then pairing it correctly with the within-outcome factor moved it again.
  The two interact — the best global k falls 0.475 to 0.250 as within rises 0.2
  to 1.0 — and they now ship as a matched pair, (within 0.5, global 0.402).
- **Within-outcome shrinkage adopted** on out-of-fold evidence: per-study mean r
  0.398 -> 0.426, pooled 0.446 -> 0.470, RMSE improving at the same time.
- **The seed replicates are now used.** 13% of 7B's and 26% of 72B's Pfander
  effect variance is sampling noise; averaging all four runs is worth about
  +0.015 r by three independent estimates. A third seed of each is sampling now.
- **The party gap calibrated.** Party offsets come from Qwen2.5-72B, because
  Pfander elicits party rather than printing it, and the per-outcome profile is
  blended half-way to anchors built from TISP, CCAM and ICPC. Error against
  external truth: 19.09 pp with no gap, 9.81 as it shipped, 3.93 now.
- **The media-loss question answered**: Pfander has no non-textual stimuli at
  all, Goldwert's seven core-media arms were excluded before any accuracy number
  existed, and severity does not predict recovery among the rest.
- **Final report** at `docs/reports/README.md` with five sub-reports.

### The demographic composition gap, found 2026-08-25

`scripts/verify_submission.py` runs every scored analysis over a built entry and
reports anything undefined -- which the format checker cannot see. It found that
the three moderators Pfander does *not* print in the profile are all badly skewed,
because the model invents them: 0.8% of respondents under $30,000 against a real
13.5%, 3.5% without a high-school diploma against 9%, 13% Republican against 29%.

That is not cosmetic. `Less than $30,000` is the dummy-coding reference level for
every income interaction and holds 18 respondents in the control arm, below the
benchmark's `min_group_n` of 30, so it is skipped and income effects are
estimated against a level that is not there.

`profiles_prefilled.csv` draws all three from the CCAM joint distribution
instead, giving 12.9% / 9.4% / 28.2%. Re-sampling on it is in flight for all
three models under the `_demo` run keys. Composition can only be fixed at
sampling time, unlike the *size* of a demographic gap, which the party anchors
correct after the fact.

### A DAIS outage, and what it cost

The cluster link died at 01:47 UTC on 2026-08-25 and came back around 07:10 --
every operation, including plain `squeue`, returning
`exit 255: Connection closed by UNKNOWN port 65535`. Notified the user per the
standing rule and continued on the local GPU.

It cost almost nothing, for two reasons worth keeping. Slurm does not care about
the submitting host, so job 421979 ran on through the outage and took
`qwen25_72b_seed3` from 896 to 8,192 respondents unattended; and `answers.jsonl`
is the source of truth, so resubmitting asked only for the remainder. The one
real hazard was the auto-resume loop: its `queued()` check greps `squeue` output,
which during the outage returned an error string that matched nothing, so it
concluded no job was queued and tried to submit six times. The submissions all
failed at the same precheck, so no duplicates were created -- but the loop should
distinguish "no job queued" from "cannot tell", and it does not.

### Negative results worth keeping

- **Per-moderator offset rescaling does not transfer.** Out-of-fold it takes
  offset r from 0.165 to 0.078 (V4-Flash) and 0.176 to 0.003 (Qwen), hurting
  both metrics in nearly every fold. Rescaling multiplies signal that is not
  there; substituting external values is what works.
- **Overshooting an anchor is punished as hard as undershooting.** On Goldwert,
  anchoring party to CCAM's raw 37.9 pp scores worse than submitting no gap at
  all.

## 9. Decisions taken (2026-08-23)

1. **Post-hoc transformation is approved.** Anything built on public datasets is
   allowed under the megastudy's rules and will be documented properly. So the
   §6 respondent-level reconstruction is the sanctioned mechanism, and the raw
   untransformed export still goes into `raw_data_deposit/`.
2. **Team metadata comes later.** Placeholders (`team_id: mpib`) go in now so the
   format gate can run; the real values are filled in at the end.
3. **All three effect studies are in: Voelkel, ICPC (Doell/Vlasceanu), Goldwert** —
   conditional only on producing good questionnaire templates for them. Nothing is
   cut on time grounds *now*; scope decisions get revisited near the deadline. The
   standing rule instead is: **always keep a calibrated Pfänder result ready**, and
   keep working the unfinished calibration studies behind it. Qwen2.5-7B runs
   locally and is effectively free, so breadth is not budget-limited.
4. **Disk has been freed.**

### Working rules for the rest of the task

- **Autonomous until done.** Ping Slack only for things a human must fix — a dead
  DAIS connection, a cluster misconfiguration that blocks the task — and then stop
  and wait rather than working around it.
- **Queue patience is the policy, not a fallback.** Per
  [handling_cluster_queues](../../.claude/skills/handling_cluster_queues/SKILL.md),
  a 15-hour wait is not a reason to cancel, resubmit, or escalate. Jobs are
  designed to minimise queue exposure instead: **never more than 4 GPUs**, walls
  sized near the estimate rather than at the 24 h cap, resume by resubmitting the
  identical body. **Qwen2.5-72B goes first** — V4-Flash needs the most GPUs at once
  and so queues worst, and 72B is both faster per respondent and cheaper to
  schedule.
- **The calibration search is open-ended.** Effect-size rescaling is the start, not
  the scope. Any transformation that can reasonably be expected to improve a
  Pfänder metric is a candidate, and candidates are generated *from* the data as it
  arrives. §2 lists what is already identified; §2.5 below is the standing agenda
  for what to look for next.

## 9.5 The open calibration agenda

Beyond the effect-level work in §2, these are the families to build and test as
each new sample lands. Each is testable on any study where we have both a silicon
sample and real responses.

**Demographic responsiveness rescaling.** The measured failure is that base models
barely condition on who they are told they are: Qwen's party gap on climate belief
is 1.1 points where reality is tens, and its largest moderator R² beyond condition
is 0.002. V4-Flash overshoots instead (12.4 points, R² 0.037). Neither is right, and
both are *fixable in the respondent-level data* because the direction and rough size
of real US demographic gaps are known from CCAM and measurable in every calibration
study. The estimator: decompose each respondent's answer into a condition mean, a
demographic-cell offset, and an individual residual; then rescale the offset toward
the human gap, `offset' = γ_m · offset` with `γ_m` fit per moderator on the
calibration studies, or replace it outright with the human cell offset where a
ground-truth anchor exists. This targets the stereotyping coefficients, the parity
gap, demographic baselines and the within-subgroup distributions — four scored
analyses that no effect-level calibration can reach. It also plausibly improves
subgroup-effect recovery, though the Voelkel finding that visible and invisible
moderators predict equally well (r 0.236 vs 0.237) says not to expect much there.

**Response-shape repair.** Round-number heaping (42–60% of item answers are
multiples of 10), the midpoint attractor (26.6% of `funding_perceptions` answers are
exactly 50), ceiling pile-up (36.5% of `belief_post` at 100), straight-lining, and
the within-battery position decay (−2.66 points per item across the 12-item trust
battery, r = −0.906). Each is a candidate for a rank-preserving repair that improves
OVL/KS/W1 without touching the condition means.

**Missingness injection.** Every synthetic respondent answers every item; the real
instrument lets participants skip most of them. The human data will carry
missingness ours does not. Whether that costs anything scored is unknown and worth
measuring.

**Per-outcome reliability weighting.** Use the calibration studies to learn which
*kinds* of outcome are predictable at all, and shrink the unpredictable ones harder
— a principled generalisation of the `true_sd ≈ 0` flattening in §2 (B2).

**Cross-model and cross-seed ensembling** beyond the pairwise average already
measured, including precision weighting rather than 50/50.

Each candidate is held to the §5 pre-commitment: it must earn its place in
held-out scoring, on a named metric, with its degrees of freedom counted.

## 10. Risks

| # | risk | mitigation | tripwire |
| --- | --- | --- | --- |
| 1 | `make check` unrunnable (no R) | port to Python, Phase 0 item 1 | green against an existing CSV by Aug 23 12:00 |
| 2 | **The anchor fails the ρ ≥ 0.7 break-even** and Tier B is worthless | Tier A alone is already +0.08 r; `secondary-2` is the w = 0 hedge | the transfer test, Aug 25 |
| 3 | No calibration beats raw in CV | pre-commit: primary = highest-r model uncalibrated unless a ≤2-parameter calibration wins on both folds by more than the bootstrap half-width | β already measured at 0.159/0.112 — shrinkage is directionally certain |
| 4 | LOSO has 2 folds | cap parameters at 2; call it "held-out transfer on one study", never "cross-validated selection" | if Vlasceanu is not template-green by Aug 25 12:00, K = 1 — announce it then, not on Aug 28 |
| 5 | Template bug found after sampling (3 such bugs on Voelkel) | 20-session smoke run + 3 eyeballed transcripts before every DAIS job | 2 minutes, non-negotiable |
| 6 | 72B config mis-derived (precedent: 81 resp/h against ~1,100 predicted) | 1.5 h pilot; abort below 1,000 resp/h | same day, before any 8 h ask |
| 7 | DAIS queue or outage | never 8 GPUs, 4–8 h walls, 2 jobs in flight, auto-resume; emergency path is fp8 72B on 1×H200 (~900–1,100 resp/h, ~zero queue) | any job PENDING > 4 h on `Resources` → cancel, resubmit smaller |
| 8 | `assert_full_grid` halts on one bad label | assert 17 conditions + 6 moderator level strings against `codebook.csv` in the ported checker | build the 208-grid against a synthetic human side on day 1 |
| 9 | Prefill turns out non-inert, forcing a resample that does not fit | the §4 pilot is the gate | Aug 23 evening |
| 10 | 4090 serialisation — `run_full_sample.sh` pkills the engine; 21 h of local work is on the critical path | one process at a time, explicit schedule; after Aug 27 nothing preempts the local Pfänder run | if local Pfänder prefill has not started by Aug 27 08:00, drop it |

**Minimum viable deliverable**, achievable in ~8 agent-hours with zero new GPU
time and finished first: the Phase 0 items plus a report skeleton carrying four
honest caveats (one transfer fold; Δr between models not distinguishable at six
clusters; education/income/party model-generated in the submitted file; level bias
uncalibrated). Everything above that is an upgrade path over a shipped
deliverable.
