# The recipe I would ship today

[← back to the final report](README.md) · basis:
[the verified cross-validation](four_study_cross_validation_verified.md) ·
[defects found](audit_findings.md) · supersedes
[the recipe, part by part](the_recipe.md)

What I would submit for Pfänder **right now**, using only the LLM inference
already on disk. No new sampling is required for any of it — one export step is,
and it is a column rename.

Four things change from the shipped entry. Three are corrections the audit
forced; one is an addition the corrected cross-validation now supports.

---

## The recipe

```python
Recipe(
    name="combined-v2",
    # --- effects: what the leaderboard sorts on -------------------------
    effects_from=(
        "qwen25_7b_demo", "qwen25_7b", "qwen25_7b_seed2", "qwen25_7b_seed3",
        "qwen25_72b_demo", "qwen25_72b", "qwen25_72b_seed2",
        "muse_glimmer_30b",                       # NEW — third model
    ),
    within_shrink=0.5,
    shrink=0.383,                                 # was 0.4127
    flatten_noise=False,                          # was True
    # --- structure: level, demographics, dispersion ---------------------
    level_from="v4_flash",
    offsets_from="v4_flash",
    residuals_from="v4_flash",
    residual_scale=1.12,
    level_anchors=LEVEL_ANCHORS,                  # policy_general 68.01 -> 65.88
    # --- party, the one moderator the model is not told ------------------
    party_offsets_from="qwen25_72b_demo",
    party_gap_anchors=PARTY_GAP_ANCHORS,          # policy_general 32.9 -> 37.3
    party_gap_weight=0.7,                         # was 0.5
)
```

Every row of the submission is still

    response = level + condition effect + demographic offset + residual

with each term taken from whichever run predicts *that term* best. The whole
design rests on those terms being separable, which they measurably are: the party
weight change below moves 21 outcome columns and leaves the effect vector
bit-identical at r = 1.000000.

| term | from | why |
| --- | --- | --- |
| rows and demographics | `qwen25_7b_demo` | quota-drawn income, education and party |
| condition effect | 8 runs / 3 models, model-balanced | best effect ranking; averaging strips sampling noise |
| level | `v4_flash`, 8 of 13 outcomes pinned to anchors | closest to external level anchors by a wide margin |
| demographic offset | `v4_flash`, party from `qwen25_72b_demo` | best pooled offsets; best party structure |
| residual | `v4_flash`, scaled 1.12 | dispersion closest to human, then fitted to it |

---

## The four changes, and the evidence for each

### 1. Add Muse-Glimmer-30B to the effect average

**Expected gain: about +0.03 on `pearson_r`.** This is the only change that
touches the sort key.

The corrected cross-validation says a three-model average beats the Qwen pair —
**0.350 against 0.312** averaged over eight half-splits of four studies, and the
three-model rule wins in 8 of 8 splits against any raw single model. Muse's
errors sit differently from the Qwens': cross-model effect vectors correlate
about 0.55 where two runs of the same model correlate about 0.84. Averaging pays
for decorrelation, not for the added model being good — and Muse is *not* good.
It is the weakest single model over four folds (0.244) and the worst of the three
on two folds outright.

**The catch, and why it does not sink the change.** The cross-validation compares
one run per model. On Pfänder the Qwen side is seven runs and Muse would be one,
and model-balanced averaging hands each model a third of the weight regardless.
Muse's single Pfänder run is the noisiest in the set — measured reliability
**0.580**, against 0.857 for `qwen25_7b` and 0.777 for `qwen25_72b`. So adding it
buys decorrelated signal and pays in noise. Both sides are measurable:

| | value |
| --- | --- |
| observed fold gain from adding Muse | ×1.122 (0.312 → 0.350) |
| of that, real signal rather than a reliability difference | **×1.110** |
| reliability cost on Pfänder (0.957 → 0.923 for the ensemble) | **×0.982** |
| **net** | **×1.090** |

The gain survives the cost with room to spare. If the two Muse replicate seeds
ever land, its noise divides by three, the ×0.982 penalty mostly disappears, and
the gain rises toward the full ×1.110.

**What I am not doing:** down-weighting Muse to reflect its noisier run.
Inverse-variance weighting across models would give it less than a third, and
might well be better — but equal model-balanced weighting is what the
cross-validation actually validated, and fitting a weight here is the exact move
the audit spent its time catching.

### 2. Turn `flatten_noise` off

`NOISE_FLOOR_OUTCOMES = ('belief_post', 'donation_ams')` shrinks those two
outcomes' within-outcome spread by 0.2, on the grounds that their effect variance
is pure sampling noise. That diagnostic was run with a **hardcoded `se = 1.0`**.
Redone with the ensemble's measured standard errors, neither outcome floors, and
the pair is not the pair the measurement picks out:

| outcome | sd of effects | rms se | **true sd** | signal/noise |
| --- | --- | --- | --- | --- |
| `donation_ams` * | 0.733 | 0.651 | **0.337** | 0.52 |
| `newsletter_signup` | 0.762 | 0.480 | 0.592 | 1.23 |
| `belief_post` * | 0.824 | 0.536 | **0.625** | 1.17 |
| `inst_trust_mean` | 0.791 | 0.455 | 0.647 | 1.42 |
| … | | | | |
| `policy_role_mean` | 1.900 | 0.512 | 1.830 | 3.57 |

\* currently flattened.

`belief_post` sits mid-pack and is indistinguishable from `newsletter_signup` and
`inst_trust_mean`, neither of which is flattened. Only `donation_ams` looks like
a genuine noise floor. Rather than keep a hand-picked pair that the measurement
contradicts, or invent a threshold that happens to select the one survivor, the
component comes out. If it goes back in it should be as a rule on measured
signal-to-noise, applied to whatever it catches.

### 3. Fix the `policy_general` anchor to the item Pfänder actually reuses

This is the sharpest single defect in the anchor set, and it moves three shipped
constants at once.

Pfänder asks: *"The U.S. government should do more to reduce global warming"*.
The anchor's own note says `'...verbatim'` — but it is measured on
`Policies_Post`, CCC's **three-item composite**, whose other two items ask about
*greenhouse gas emissions* and *energy efficiency*. Those are different
questions and they behave differently. CCC's Q19 is the verbatim match, and it is
`Policies_Post_3`:

| CCC item | level | sd | D−R party gap |
| --- | --- | --- | --- |
| `Policies_Post` (composite, **as shipped**) | 68.01 | 29.32 | 32.91 |
| `Policies_Post_1` — greenhouse gas emissions | 66.20 | 31.94 | 35.62 |
| `Policies_Post_2` — energy efficiency | 71.96 | 27.79 | 25.75 |
| **`Policies_Post_3` — reduce global warming (Q19)** | **65.88** | **32.89** | **37.34** |

The pattern corroborates itself: energy efficiency is the least polarised item
(gap 25.8) and explicit global-warming policy the most (37.3), which is what you
would expect and is why averaging them is the wrong summary for this outcome.

Three constants move: level anchor **68.01 → 65.88**, dispersion **29.32 →
32.89**, party gap **32.9 → 37.34**. The level anchor is arithmetic, so this
directly changes the submitted control-arm mean on one of thirteen outcomes.

### 4. Raise the party-gap blend weight to 0.7, and correct the shrink constant

**Party weight 0.5 → 0.7.** The argument for 0.5 does not survive: it needs the
donor's distance from the anchors, and the 7.36 pp it used is the distance from
the *pre-CCC* anchor set. Like-for-like it is 6.03, which makes the implied donor
variance negative. Measured out of sample instead — blend toward the pre-CCC
values, grade against CCC's real humans, so the anchor comes from neither the
model nor the grader — the least-squares optimum is 1.09 / **0.83** / 0.46 for
7B / 72B / Muse, and the party donor is a 72B run. 0.7 shades that down for two
things the measurement cannot see. Party-gap RMSE against the anchors falls
**4.45 → 3.12 pp**; Section 1 is untouched by construction.

**Global shrink 0.4127 → 0.383.** `shrink_for_runs` rescales the fitted factor by
`reliability_here / SHRINK_FITTED_RELIABILITY`, and that denominator is set to
0.870 — a **Pfänder** one-run-each figure. But `GLOBAL_SHRINK = 0.375` was fitted
on the *reference studies*, where the one-run-each Qwen pair's measured
reliability is **0.903**. Using the Pfänder number as the reference over-corrects.
With the right denominator and the three-model ensemble's 0.923, the factor is
0.375 × 0.923 / 0.903 = **0.383**.

Shrinkage is provably neutral on `pearson_r` — a positive scalar cannot move a
correlation — so this reaches RMSE and β only. It is worth getting right because
β is a reported metric and the shipped value overshoots it.

---

## What I would deliberately not change

**DeepSeek-V4-Flash stays out of the effect average and stays as the structural
donor.** It has a mean effect-recovery correlation of **−0.040** across the
reference studies — worse than submitting nothing — and adding it to the effect
average costs 0.075. It also places control-arm levels better than any other run
by a wide margin (2.59 pp against 6.06–7.74 for the Qwen donors). The component
split exists precisely so a model can be excluded from the term it is bad at and
kept for the three it is good at. This is the single strongest result in the
project and nothing in the audit touched it.

**`within_shrink` stays at 0.5.** Over eight splits, 0.3 scores 0.315 and 0.5
scores 0.312 — indistinguishable, and the split-to-split sd is 0.067. The curve
is flat and fitting inside a flat region buys noise. Its selection optimism is
measured at 0.005.

**`residual_scale` stays at 1.12, flagged.** The CCC hold-out found it makes two
of three models *worse* on every shape metric. But it was fitted on TISP for
V4-Flash, and V4-Flash has no CCC sample — so the negative result is about donors
we do not use. This is the largest untested constant in the recipe and I would
not defend it; I keep it because there is no evidence against it *for this donor*
and removing it is equally unsupported.

---

## What to expect

| | shipped entry | **this recipe** |
| --- | --- | --- |
| `pearson_r` (sort key) | 0.30 | **0.33** |
| 95% interval the benchmark prints | 0.17 – 0.43 | **0.20 – 0.46** |
| party-gap RMSE vs anchors | 4.45 pp | **3.12 pp** |
| `beta` | 1.15 | closer to 1 |

The point estimate depends on how strictly you charge the within-outcome factor.
Charging it as fitted per fold — the conservative reading my prediction report
uses — gives 0.325. Taking the pre-committed-rule row, which is what this recipe
actually instantiates, gives 0.35. I would state **0.33 ± the same ±0.13
cluster-bootstrap half-width**, and note that the difference between the two
readings (0.025) is a fifth of the interval width.

**The honest summary is that this is a ~10% improvement on one metric inside an
interval four times its size.** Which study Pfänder resembles still matters far
more than anything in this recipe: the same basis scores 0.009 on CCC and 0.460
on Voelkel.

---

## Building it

Everything below is post-inference. No sampling.

1. **Export Muse's Tier-1 frame.** `data/pfander/silicon_sampling/muse_glimmer_30b/`
   has `samples.csv` but no `tier1_submission.csv`, and no `answers.jsonl` to
   rebuild one from. It does not need one: the samples file already carries all
   13 scored outcomes, `condition`, all six moderators, and the twelve trust
   items under their raw Qualtrics names (`trust_competent_1` …). The Tier-1
   frame is a column rename through `pfander.outcomes.DIRECT`. Muse supplies only
   the effect term, so only the 13 outcomes are load-bearing.
2. **Update the two `policy_general` anchors** in `anchors/ccc.py` to
   `Policies_Post_3`, and let `PARTY_GAP_ANCHORS` / `HUMAN_DISPERSION` follow.
3. **Set `flatten_noise=False`** and **`party_gap_weight=0.7`** (already done) in
   `scripts/build_entries.py`'s `common` dict.
4. **Fix `SHRINK_FITTED_RELIABILITY`** to the reference-study figure (0.903), and
   give Muse measured variance components so `ensemble_reliability` stops
   returning `None` for any ensemble containing it — otherwise `shrink_for_runs`
   silently falls back to the unadjusted 0.375.
5. `python scripts/build_entries.py` and check the drift audit is at floating
   point, as it is today (4.3 × 10⁻⁴ pp).

The three-entry structure still makes sense, since the benchmark scores all
three with no penalty: **primary** as above; **secondary-1** identical without
global shrinkage, which is the one axis where shrinkage could hurt and cannot
help the sort key; **secondary-2** the uncalibrated best single ranker, as
insurance against every calibration being wrong at once.

## What would change this

**Two more Muse seeds** would take its reliability from 0.580 toward the Qwens'
and turn a ×1.090 net gain into something near ×1.110. This is the cheapest
available improvement and it is pure sampling — no new design work.

**A CCC run of DeepSeek-V4-Flash** would let the cross-validation grade the
structural donor the entry actually uses. Right now every Section 3 and 10–12
number in the cross-validation describes a donor chosen from {7B, 72B, Muse},
because V4 has no CCC sample. Three-quarters of the benchmark rests on a
component that has never been cross-validated, and this is the one run that would
fix it.

**Voelkel re-sampled with its slider ranges stated.** 52 of its 53 integer
sliders reach the model with no numeric range — the exact defect the `_v3` audit
fixed for ICPC and Goldwert, where it had compressed every effect roughly
five-fold — and four of its items carry unresolved `${e://Field/...}` pipes.
Voelkel is currently the fold this recipe scores highest on, which is a reason to
distrust that number rather than enjoy it.
