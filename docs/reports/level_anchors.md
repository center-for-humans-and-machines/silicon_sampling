# External level anchors: ground truth for Pfänder's control-arm levels

Pfänder publishes no human data, so nothing in our submission can be checked
against its own truth. Our synthetic respondents' **effects** are calibrated
against Voelkel; their **levels** are not calibrated against anything. That
matters because level error drives three scored analyses no effect calibration
touches: the control-condition response distributions (W1, OVL, KS), the
demographic baselines over control cell means (paired r and RMSE, in raw outcome
points), and the parity gap.

This report builds level anchors out of two nationally representative US surveys,
grades every item mapping, and — the part that decides whether any of it is used —
measures on Voelkel how accurate an anchor has to be before it stops helping.

**Verdict in three lines.**

1. Anchoring to *true* levels is worth a great deal: mean W1 falls from 23.3 to
   6.6 (Qwen2.5-7B) and 10.0 to 4.2 (DeepSeek-V4-Flash); the demographic-baseline
   correlation rises from 0.18 to 0.98 and from 0.80 to 0.98; baseline RMSE falls
   from 30.5 to 3.8 and 12.1 to 3.7 raw points. Every condition effect is held
   fixed to 1e-14, so none of this is bought from the leaderboard metrics.
2. The gain survives an anchor error of about **13 points** for Qwen and about
   **4.7 points** for DeepSeek-V4-Flash, on the strictest of the metrics (KS).
   That is the specification an anchor has to meet.
3. Three of Pfänder's thirteen outcomes can be anchored to that standard, all
   from TISP, all at grade `near`: **`trust_multidimensional` 67.6**,
   **`trust_post` 67.0**, **`policy_role_mean` 65.0**. The other ten cannot, and
   the climate-policy items that look anchorable are the ones the two sources
   disagree about by 6.5 and 7.0 points — above break-even, so they are excluded.

Code: `silicon_sampling/anchors/` (`scales`, `crosswalk`, `tisp`, `ccam`,
`levels`, `validate`). Tests: `tests/test_anchors.py`.

---

## 1. The two sources

### TISP — Cologna et al. 2025

`data/calibration/datasets/cologna_etal2025.csv`, 69,534 x 141, semicolon
separated with comma decimals. Filtering `COUNTRY_CODE == 'USA'` gives
**n = 2,559**, all `UserLanguage == 'EN'`, all `Progress == 100`, fielded
**2023-02-07 to 2023-03-08**. Post-stratification weight `WEIGHT_CNTRY`, which
averages to 1.000 within the US subsample (Kish effective n = 2,505, design
effect 1.02 — the weights are gentle).

Its twelve `TRUST_SCI_*` items are the same Besley four-facet battery Pfänder uses
as its primary outcome, in the same four triads. It also carries `CLIM_TRUST`,
`TRUST_PEW`, the six `NORMPERC_*` items that became Pfänder's `policy_role`, and
five `CLIM_POLSUPPORT_*` items whose stems match Pfänder's `policy_specific`
items 1-5.

### CCAM — Climate Change in the American Mind

`data/calibration/datasets/ccam.sav`, 35,309 x 58, Ipsos KnowledgePanel trend
file, read with `pyreadstat` so the value labels survive. It spans **Nov 2008 to
Dec 2024 across 31 waves**. Levels are what we are after and climate attitudes
trend, so pooling would produce an average describing no year.

**Wave used: 31, Dec 2024, n = 1,013**, weight `weight_wave` (Kish effective
n ≈ 880, so an item's SE is about 1.1 points on the converted scale). It is the
most recent wave *and* the most recent complete one — Apr 2024 is missing
`fund_research` and `reduce_tax`, Oct 2023 is missing three items. **No pooling
was needed and none was done.**

---

## 2. Scale conversion, and the assumption it rests on

TISP's items are 5- and 3-point fully labelled Likerts; CCAM's are 4-point;
Pfänder's are 0-100 sliders. There is no assumption-free conversion.

**The default is linear on the response options**: option 1 to 0, option *k* to
100, equal spacing between. A 1-5 Likert becomes 0 / 25 / 50 / 75 / 100. **This
assumes respondents treat the labelled categories as equally spaced and as
centred on the slider's endpoints — that "Somewhat expert" out of five options
means the same thing as 75 on a 0-100 bar. That is an assumption about how people
read scales, not a fact, and nothing in either dataset can test it.**

**The competing conversion is the bin midpoint**: each option covers an
equal-width band of the latent scale and scores at its centre, giving
10 / 30 / 50 / 70 / 90. This is what one would use if the categories were a
coarsening of an underlying continuous response. It is equally defensible a
priori.

The two disagree by exactly **`(50 - mean) / k`**. So an anchor sitting on the
scale midpoint is conversion-proof and one near an end is not. For the TISP trust
battery (k = 5, linear mean 71.5) the disagreement is **4.3 points** — the same
order of magnitude as the break-even error in §5. Both figures are reported for
every anchor; this spread, not the sampling SE, is the anchor's real uncertainty,
and no sample size removes it.

**Dispersion is reported twice.** A five-option answer converted to sliders puts
all its mass on five spikes 25 points apart, which inflates the variance by about
`h^2/12` (Sheppard's correction). The corrected `sd_slider` is the figure a
dispersion calibration would want.

**Standard errors** use Kish's effective sample size. Neither source publishes
the clustering or stratification variables a design-based interval would need, so
these SEs charge for the weights only. They come out at 0.4-1.3 points, four to
eight times smaller than the conversion ambiguity, so the omission does not
change any decision here.

**The referent shift.** Every TISP trust item asks about *scientists*; Pfänder
asks about *climate scientists*. TISP can size that from within: the same
respondents answered `TRUST_PEW` (confidence in scientists) and `CLIM_TRUST`
(trust in climate scientists, correlated 0.70), and the weighted paired
difference is **3.92 points, SE 0.48, n = 2,557** — the climate referent scores
lower. It is a confounded estimate, because the two stems differ in more than
their referent, but it is the only one available, and ignoring a measured
3.9-point bias is a worse decision than correcting for it. It is subtracted from
the trust battery **and from nothing else**, because it was measured on a pair of
trust items and generalises no further. (Applying it to `policy_role_mean` would
move an anchor that currently agrees with both our samplers to within 3 points
away from both.)

Two independent corrections happen to converge here: the bin-midpoint conversion
gives 67.2 for the trust battery and the referent-adjusted linear conversion
gives 67.6. The uncorrected linear figure is 71.5.

---

## 3. The graded crosswalk

One row per (Pfänder item, source item). `converted` is the weighted mean on
Pfänder's 0-100 scale under the linear conversion; `se` is its Kish standard
error. Grades: **`verbatim`** = same stem, same referent, endpoints meaning the
same thing; **`near`** = small named differences that plausibly leave the level
within a few points; **`construct-only`** = same construct but a difference that
can plausibly move the level by more than break-even; **`unusable`** = no
defensible conversion. **Only `verbatim` and `near` are offered by default.**

**Nothing in either source is `verbatim`.** Every trust row loses it on the
referent, and every policy-support row loses it on the response scale.

| Pfänder outcome | Pfänder item | source | source item | source scale | converted | se | grade | difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trust_multidimensional | trust_competence_1 | TISP | TRUST_SCI_expert | 5-pt, Very inexpert / Very expert | 73.44 | 0.50 | near | adjective differs (expert for competent); referent |
| trust_multidimensional | trust_competence_2 | TISP | TRUST_SCI_intellig | 5-pt, Very unintelligent / Very intelligent | 82.78 | 0.45 | near | adjective order; referent |
| trust_multidimensional | trust_competence_3 | TISP | TRUST_SCI_qualified | 5-pt, Very unqualified / Very qualified | 79.59 | 0.47 | near | source adds "conducting high-quality research"; referent |
| trust_multidimensional | trust_integrity_1 | TISP | TRUST_SCI_honest | 5-pt, Very dishonest / Very honest | 71.00 | 0.50 | near | adjective order; referent |
| trust_multidimensional | trust_integrity_2 | TISP | TRUST_SCI_ethical | 5-pt, Very unethical / Very ethical | 69.79 | 0.51 | near | adjective order; referent |
| trust_multidimensional | trust_integrity_3 | TISP | TRUST_SCI_sincere | 5-pt, Very insincere / Very sincere | 72.40 | 0.50 | near | adjective order; referent |
| trust_multidimensional | trust_benevolence_1 | TISP | TRUST_SCI_concerned | 5-pt, Not concerned / Very concerned | 70.75 | 0.54 | near | low pole "Not concerned" not "Very unconcerned"; referent |
| trust_multidimensional | trust_benevolence_2 | TISP | TRUST_SCI_improve | 5-pt, Very uneager / Very eager | 72.50 | 0.50 | near | adjective order; referent |
| trust_multidimensional | trust_benevolence_3 | TISP | TRUST_SCI_otherint | 5-pt, Very inconsiderate / Very considerate | 67.79 | 0.53 | near | adjective order; referent |
| trust_multidimensional | trust_openness_1 | TISP | TRUST_SCI_open | 5-pt, Not open / Very open | 65.40 | 0.56 | near | "if at all" dropped; referent |
| trust_multidimensional | trust_openness_2 | TISP | TRUST_SCI_trans | 5-pt, Very unwilling / Very willing | 67.13 | 0.55 | near | adjective order; referent |
| trust_multidimensional | trust_openness_3 | TISP | TRUST_SCI_otherviews | 5-pt, Very little / Very much attention | 65.63 | 0.57 | near | near-identical stem; referent |
| trust_post | trust_post_1 | TISP | CLIM_TRUST | 5-pt, **Not at all / Very strongly** | 67.01 | 0.65 | near | endpoints identical, referent correct; stem paraphrased, "in your country" |
| trust_post | trust_post_1 | TISP | TRUST_PEW | 5-pt, No confidence / A great deal | 70.95 | 0.56 | construct-only | confidence to act in the public interest, scientists in general — kept because it sizes the referent shift |
| policy_role_mean | policy_1_1 | TISP | NORMPERC_integrate | 5-pt, Strongly disagree / Strongly agree | 62.06 | 0.65 | near | "politicians" for "policy makers"; referent |
| policy_role_mean | policy_2_1 | TISP | NORMPERC_advocate | 5-pt, Strongly disagree / Strongly agree | 67.48 | 0.60 | near | referent only |
| policy_role_mean | policy_3_1 | TISP | NORMPERC_communicate | 5-pt, Strongly disagree / Strongly agree | 66.88 | 0.61 | near | "politicians" for "policy makers"; referent |
| policy_role_mean | policy_4_1 | TISP | NORMPERC_involved | 5-pt, Strongly disagree / Strongly agree | 63.55 | 0.63 | near | referent only |
| policy_specific_mean | policy_specific_1_1 | TISP | CLIM_POLSUPPORT_fueltax | 3-pt unipolar + N/A | 53.94 | 0.85 | construct-only | unipolar support intensity against a bipolar oppose-support slider; 5 of 7 items |
| policy_specific_mean | policy_specific_2_1 | TISP | CLIM_POLSUPPORT_publictransport | 3-pt unipolar + N/A | 71.08 | 0.69 | construct-only | stem **verbatim**, scale is not |
| policy_specific_mean | policy_specific_3_1 | TISP | CLIM_POLSUPPORT_sustenergy | 3-pt unipolar + N/A | 73.25 | 0.72 | construct-only | stem **verbatim**, scale is not |
| policy_specific_mean | policy_specific_4_1 | TISP | CLIM_POLSUPPORT_protection | 3-pt unipolar + N/A | 82.39 | 0.61 | construct-only | stem **verbatim**, scale is not |
| policy_specific_mean | policy_specific_5_1 | TISP | CLIM_POLSUPPORT_foodtax | 3-pt unipolar + N/A | 48.38 | 0.86 | construct-only | "carbon intense" for "carbon-intensive"; scale |
| policy_specific_mean | policy_specific_1_1 | CCAM | reduce_tax | 4-pt Strongly oppose / Strongly support | 60.44 | 1.11 | construct-only | revenue-neutral tax on companies, not on fuels — the cross-source check |
| policy_specific_mean | policy_specific_3_1 | CCAM | generate_renewable | 4-pt Strongly oppose / Strongly support | 66.23 | 1.04 | construct-only | restricted to public land — the cross-source check |
| concern_mean | concern_1_1 | CCAM | worry | 4-pt Not at all / Very worried | 58.83 | 1.15 | construct-only | item-level match is near, but one item for a three-item composite; CCAM's `priority` sits 5.6 points below `worry`, so worry-only overstates it |
| policy_general | policy_general_1 | CCAM | priority | 4-pt Low / Very high | 53.25 | 1.28 | construct-only | priority for federal action, not support for a statement about it (`transition_economy` gives 58.3 and is no closer) |
| behavior_mean | individual_talk_1 | CCAM | discuss_GW | 4-pt Never / Often | 36.72 | 1.03 | construct-only | past frequency against next-twelve-months likelihood; one item for a six-item composite |
| belief_post | belief_post_1 | CCAM | cause_recoded | unordered categories | — | — | unusable | a conditional cause attribution, not a rated accuracy; scoring categories 0/50/100 would invent the anchor. `happening` is a yes/no proportion, which is not a slider mean either |
| funding_perceptions | funding_5 reversed | CCAM | fund_research | 4-pt Strongly oppose / Strongly support | 68.10 | 1.05 | unusable | support for funding renewables, not a judgement of current federal spending; the Pfänder item is centred on an adequacy midpoint the source scale has no point for |

Outcomes with **no candidate item at all** in either source:

| outcome | why |
| --- | --- |
| distrust_post | no distrust item anywhere; distrust is not 100 minus trust, which is why Pfänder measures both |
| inst_trust_mean | nothing about the EPA, NASA, NOAA, universities or the federal government; TISP's `CLIM_GOV` battery is about government conduct, not trust in named institutions |
| donation_ams | a behavioural allocation of real money, 0-10 |
| newsletter_signup | a recorded click on a specific newsletter |

The full table, including every field and the untruncated notes, is
`silicon_sampling.anchors.crosswalk.to_frame()`.

---

## 4. The anchors

Offered by default (grade `near` or better). `mean` is what
`levels.levels()` returns; `sd` is the converted dispersion and `sd_slider` the
Sheppard-corrected one; `mean_bin_midpoint` is the rival conversion.

| outcome | mean | sd | sd_slider | se | n | source | items | grade | mean_raw | referent adj. | bin-midpoint | conversion spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trust_multidimensional | **67.58** | 20.62 | 19.31 | 0.41 | 2,555 | TISP | 12 | near | 71.50 | −3.92 | 67.20 | 4.30 |
| trust_post | **67.01** | 32.53 | 31.72 | 0.65 | 2,557 | TISP | 1 | near | 67.01 | 0 | 63.61 | 3.40 |
| policy_role_mean | **64.99** | 26.02 | 25.00 | 0.52 | 2,558 | TISP | 4 | near | 64.99 | 0 | 61.99 | 3.00 |

**The four trust facets are 12.6 points apart** and `tier1.align_trust_items`
shifts all twelve items by the same amount, so anchoring the composite leaves the
facets wherever the sampler put them. `levels.facet_levels()` gives the
referent-adjusted facet levels for a caller that wants the item-level
distributions right too:

| facet | anchor (adjusted) | raw |
| --- | --- | --- |
| competence | 74.68 | 78.60 |
| integrity | 67.15 | 71.07 |
| benevolence | 66.42 | 70.34 |
| openness | 62.13 | 66.05 |

### The cross-source check, and why policy support is excluded

Two of CCAM's items overlap TISP's, which is the only empirical handle available
on how much a converted Likert anchor can be trusted — two nationally
representative US samples, different instruments, different granularities, one
construct:

| Pfänder item | TISP | CCAM | gap |
| --- | --- | --- | --- |
| policy_specific_1_1 (fossil-fuel taxes) | 53.94 (`CLIM_POLSUPPORT_fueltax`) | 60.44 (`reduce_tax`) | **6.50** |
| policy_specific_3_1 (sustainable energy) | 73.25 (`CLIM_POLSUPPORT_sustenergy`) | 66.23 (`generate_renewable`) | **7.02** |

Both gaps exceed the 4.7-point break-even for our better sampler and one exceeds
the 6.5-point OVL break-even. On climate-policy support the two best available
sources cannot agree to within the tolerance, which is a direct argument against
using either — and it is why `policy_specific_mean` stays at `construct-only`
even though four of its five TISP stems are word-for-word identical.

---

## 5. The validation: does anchoring help, and up to what error?

Voelkel is the only study here with real participant data, so the rehearsal
happens there and the conclusion is carried across. Human 1 (n = 6,259, control
n = 2,832; `half_split(seed=42)`) supplies the true control levels; the two
silicon samples are re-levelled onto them with condition effects held exactly
fixed.

> **A note on the machinery.** `calibration.tier1.calibrate` is the production
> route and takes exactly the `levels` dict this package produces — verified in
> `tests/test_anchors.py`, and run against both real Pfänder Tier-1 submissions in
> §6 with drift 2.1e-14 and composite consistency 0.0. It cannot run on Voelkel:
> it hard-wires the control label to `"control"` (Voelkel's is `"Null_Control"`)
> and looks scales up in Pfänder's `SCALE_RANGE`, which raises `KeyError` on `PA`.
> Monkeypatching two module constants would test the patch, so the rehearsal calls
> the same components `tier1` delegates to (`decompose`, `Decomposition`,
> `recompose_frame`) with Voelkel's control label. The steps `tier1` adds on top
> are Pfänder format constraints with no Voelkel counterpart.
>
> All figures below were computed with `recompose_frame`'s current default
> (`couple_residuals=True`, one donor per rebuilt respondent across outcomes).
> Rerunning the driver reproduces them; small shifts in the third digit follow that
> setting and change no conclusion.

### 5.1 The ceiling — anchoring to the truth

Averages over the 63 condition x outcome cells (distribution metrics) and the 162
control cell means (baselines). `level_error` is the mean absolute distance
between the sample's control levels and Human 1's, over the nine outcomes.

| run | step | level error | effect drift | W1 ↓ | OVL ↑ | KS ↓ | var ratio →1 | baseline r ↑ | baseline RMSE ↓ | DPD ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen25_7b | raw sample | 23.72 | 0 | 23.26 | 0.537 | 0.413 | 1.126 | 0.182 | 30.54 | 2.81 |
| qwen25_7b | self-recomposed | 23.72 | 0 | 23.30 | 0.527 | 0.448 | 1.059 | 0.181 | 30.41 | 2.75 |
| qwen25_7b | **anchored to truth** | 0 | 0 | **6.56** | **0.656** | **0.226** | 0.853 | **0.976** | **3.80** | 2.45 |
| v4_flash | raw sample | 8.32 | 0 | 10.02 | 0.721 | 0.193 | 1.443 | 0.799 | 12.13 | 2.03 |
| v4_flash | self-recomposed | 8.32 | 0 | 9.80 | 0.729 | 0.207 | 1.351 | 0.799 | 12.06 | 1.77 |
| v4_flash | **anchored to truth** | 0 | 0 | **4.19** | **0.782** | **0.138** | 1.236 | **0.978** | **3.72** | 2.16 |

Reading it:

* **This is the ceiling.** It uses the truth, not a proxy, so no anchor can beat
  it.
* **Effect drift is 0.0** at every step (1.4e-14 in the raw audit). The gains are
  not bought from the leaderboard metrics.
* **The self-recomposed row is the control.** It rebuilds the sample from its own
  parts, so it isolates what the reconstruction costs from what the anchoring
  buys. Its distribution metrics are within 0.02 of the raw sample's except for
  KS on Qwen (0.413 → 0.448) and the variance ratio, which the reconstruction
  pulls toward 1 by 0.07-0.09 on its own.
* **The demographic baselines move most.** Paired r 0.18 → 0.98 and RMSE
  30.5 → 3.8 for Qwen; 0.80 → 0.98 and 12.1 → 3.7 for DeepSeek. This is the
  single largest improvement available anywhere in the anchoring exercise, and it
  is unreachable by any effect calibration.
* **The variance ratio is the exception.** For Qwen it goes 1.126 → 0.853, i.e.
  from 0.13 above target to 0.15 below it. Roughly a quarter of that is the
  reconstruction (1.126 → 1.059) and the rest is the level move interacting with
  the 0-100 clip. For DeepSeek it improves (1.443 → 1.236). Anchoring is not a
  variance-ratio fix and should not be sold as one.
* **The parity gap barely moves and is not reliably improved** (Qwen 2.81 → 2.45,
  DeepSeek 2.03 → 2.16, and DeepSeek's self-recomposed row is lower than either at
  1.77). Levelling shrinks every group's absolute error but not equally, and DPD
  reads the *spread* of those errors, not their size.

Per-outcome, the gain tracks the level error almost exactly — the outcomes where
the sampler was wildly off are the ones that improve:

| outcome | human | qwen level | error | W1 before → after | v4 level | error | W1 before → after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA | 68.05 | 62.55 | −5.5 | 4.82 → 4.72 | 69.24 | +1.2 | 4.18 → 3.87 |
| ADA | 26.66 | 15.93 | −10.7 | 10.88 → 7.18 | 21.91 | −4.7 | 5.99 → 2.99 |
| SPV | 10.77 | 12.75 | +2.0 | 3.82 → 3.57 | 14.37 | +3.6 | 4.37 → 2.52 |
| SUC | 51.89 | 15.64 | −36.3 | 35.16 → 8.43 | 58.42 | +6.5 | 12.82 → 11.41 |
| OppBip | 20.95 | 83.85 | +62.9 | 61.72 → 9.12 | 35.38 | +14.4 | 15.55 → 5.78 |
| SocDistrust | 53.58 | 77.48 | +23.9 | 21.64 → 9.75 | 54.81 | +1.2 | 4.56 → 3.85 |
| SocDis | 30.72 | 77.99 | +47.3 | 44.69 → 9.13 | 62.01 | +31.3 | 30.14 → 4.02 |
| BEPF | 51.57 | 35.07 | −16.5 | 18.14 → 3.50 | 45.72 | −5.8 | 6.98 → 2.15 |
| Composite | 39.24 | 47.66 | +8.4 | 8.49 → 3.64 | 45.23 | +6.0 | 5.61 → 1.12 |

### 5.2 The break-even — how wrong may an anchor be?

The same experiment with a deliberate error added to the true level, at eight
magnitudes under four sign patterns (all +, all −, and two seeded mixed
patterns), averaged. Mixed patterns are included because a uniform shift
preserves the *ordering* of the outcomes, and the pooled baseline correlation
mostly measures that ordering — degraded uniformly it stays at 0.98 even 30
points from the truth, which would read as an anchor that cannot be bad enough to
hurt. Realised error is reported because an outcome whose true level is 10.8
cannot be moved 30 points down.

**qwen25_7b** (raw sample in the first row):

| nominal error | realised | W1 ↓ | OVL ↑ | KS ↓ | var ratio | baseline r ↑ | baseline RMSE ↓ | effect drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *raw* | 23.72 | *23.26* | *0.537* | *0.413* | *1.126* | *0.182* | *30.54* | 0 |
| 0 | 0.00 | 6.56 | 0.656 | 0.226 | 0.853 | 0.976 | 3.80 | 0 |
| 2 | 2.00 | 6.73 | 0.655 | 0.234 | 0.847 | 0.974 | 4.33 | 0 |
| 5 | 5.00 | 7.81 | 0.641 | 0.267 | 0.829 | 0.963 | 6.33 | 0 |
| 10 | 10.00 | 11.14 | 0.578 | 0.360 | 0.781 | 0.928 | 10.76 | 1.08 |
| 15 | 14.77 | 15.29 | 0.508 | 0.453 | 0.737 | 0.879 | 15.31 | 1.46 |
| 20 | 19.49 | 19.70 | 0.430 | 0.541 | 0.679 | 0.829 | 19.97 | 2.24 |
| 25 | 23.87 | 23.87 | 0.354 | 0.619 | 0.611 | 0.782 | 24.38 | 2.96 |
| 30 | 27.99 | 27.77 | 0.293 | 0.682 | 0.544 | 0.738 | 28.60 | 4.67 |

**v4_flash**:

| nominal error | realised | W1 ↓ | OVL ↑ | KS ↓ | var ratio | baseline r ↑ | baseline RMSE ↓ | effect drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *raw* | 8.32 | *10.02* | *0.721* | *0.193* | *1.443* | *0.799* | *12.13* | 0 |
| 0 | 0.00 | 4.19 | 0.782 | 0.138 | 1.236 | 0.978 | 3.72 | 0 |
| 2 | 2.00 | 4.65 | 0.768 | 0.151 | 1.224 | 0.976 | 4.26 | 0 |
| 5 | 5.00 | 6.37 | 0.744 | 0.198 | 1.189 | 0.965 | 6.29 | 0 |
| 10 | 10.00 | 10.48 | 0.669 | 0.296 | 1.096 | 0.928 | 10.75 | 0.11 |
| 15 | 14.77 | 14.91 | 0.592 | 0.381 | 0.996 | 0.879 | 15.33 | 0.49 |
| 20 | 19.49 | 19.48 | 0.510 | 0.468 | 0.880 | 0.828 | 20.01 | 0.52 |
| 25 | 23.87 | 23.78 | 0.432 | 0.551 | 0.770 | 0.780 | 24.43 | 0.79 |
| 30 | 27.99 | 27.82 | 0.353 | 0.628 | 0.660 | 0.737 | 28.66 | 2.26 |

Effect drift stays exactly 0 up to a 5-point anchor error and only becomes
material past 15, where a target near the end of the scale stops being
representable. In the range that matters, anchoring is effect-neutral.

### **The break-even anchor error**

Linear interpolation to where each metric stops preferring the anchor. `inf`
means no crossing up to 30 points.

| run | metric | raw | at truth | **break-even error** |
| --- | --- | --- | --- | --- |
| qwen25_7b | W1 | 23.262 | 6.560 | **24.3** |
| qwen25_7b | OVL | 0.537 | 0.656 | **12.9** |
| qwen25_7b | KS | 0.413 | 0.226 | **12.8** |
| qwen25_7b | baseline r | 0.182 | 0.976 | inf (> 30) |
| qwen25_7b | baseline RMSE | 30.536 | 3.803 | inf (> 30) |
| v4_flash | W1 | 10.020 | 4.189 | **9.4** |
| v4_flash | OVL | 0.721 | 0.782 | **6.5** |
| v4_flash | KS | 0.193 | 0.138 | **4.7** |
| v4_flash | baseline r | 0.799 | 0.978 | **23.1** |
| v4_flash | baseline RMSE | 12.134 | 3.717 | **11.5** |

**The headline number is 4.7 points** — the tightest constraint, KS on the better
sampler. Read across the table:

* Break-even sits close to each sampler's own mean absolute level error (23.7 for
  Qwen, 8.3 for DeepSeek). The intuitive rule holds: *use the anchor when it is
  likely to be closer to truth than the sampler already is.* It is worth having
  measured rather than assumed, because the metrics disagree by a factor of five
  about where the line is.
* KS is the strictest, then OVL, then W1, then the baselines. KS is the maximum
  single CDF gap, so it punishes a shifted-but-otherwise-right distribution
  hardest.
* The demographic baselines are the most forgiving by far: an anchor 20 points
  wrong still beats Qwen's raw r of 0.18. If the baselines were the only scored
  analysis, almost any anchor would be worth applying.
* The variance ratio is excluded from break-even because its target is two-sided,
  so "worse" has no direction; it is reported instead.

Against this, the TISP anchor's own uncertainty is **3.0-4.3 points from the
conversion assumption**, plus 0.4-0.65 sampling SE, plus whatever residual bias
the referent correction leaves. So the trust and policy-role anchors sit *just
inside* the tolerance for the better sampler and comfortably inside it for the
weaker one — and the excluded policy-support anchors, whose cross-source spread
is 6.5-7.0, sit outside it.

---

## 6. What this means for the Pfänder submission

The anchors drop into `tier1.calibrate(frame, levels=...)` on both real Tier-1
submissions with **max effect drift 2.1e-14, level drift 1.4e-14, and composite
consistency 0.0** (the format gate warns above 0.5).

| outcome | anchor | qwen25_7b control | gap | v4_flash control | gap |
| --- | --- | --- | --- | --- | --- |
| trust_multidimensional | 67.58 | 53.38 | **−14.20** | 65.62 | −1.96 |
| trust_post | 67.01 | 57.57 | **−9.45** | 64.32 | −2.70 |
| policy_role_mean | 64.99 | 65.74 | +0.75 | 68.12 | +3.13 |

* **For Qwen2.5-7B the anchor is clearly worth applying.** It is 14 and 9 points
  away on the two trust outcomes, against a break-even of 12.8 and an anchor
  uncertainty of 4. Qwen's Voelkel level error was 23.7, so the prior that it is
  the one that is wrong is strong.
* **For DeepSeek-V4-Flash the anchor is roughly a coin flip.** All three gaps
  (2.0, 2.7, 3.1) are inside the anchor's own conversion uncertainty, so applying
  it neither clearly helps nor clearly hurts. It is defensible either way; what
  is *not* defensible is claiming an improvement.
* **`policy_role_mean` is convergent evidence that the conversion is not wild.**
  The TISP anchor lands within 0.8 and 3.1 points of two independently sampled
  models on the outcome whose items are the closest match in the crosswalk. That
  is not proof, but it is the only external agreement available, and it points the
  right way.

---

## 7. Honest verdict on coverage

**Anchorable (3 of 13), grade `near`, TISP:**

| outcome | anchor | why it qualifies |
| --- | --- | --- |
| trust_multidimensional | 67.58 | the same 12-item Besley battery, 11 stems near-verbatim, one adjective substituted; only the referent differs, and that shift is measured and removed |
| trust_post | 67.01 | endpoint labels identical, referent correct, stem paraphrased |
| policy_role_mean | 64.99 | four items, 5-point agree scale with identical endpoints, only the referent and "politicians"/"policy makers" differ |

**Not anchorable (10 of 13):**

| outcome | best candidate | why not |
| --- | --- | --- |
| policy_specific_mean | TISP `CLIM_POLSUPPORT` (5 items) | stems match — four word-for-word — but the source scale is unipolar support intensity plus a Not-applicable option against a bipolar oppose-support slider, it covers 5 of 7 items, and the two sources disagree by 6.5-7.0 points on the items they share |
| concern_mean | CCAM `worry` | one item for a three-item composite; item-level match is near, composite is not, and CCAM's own priority item sits 5.6 points lower |
| policy_general | CCAM `priority` | priority for federal action, not support for a statement about it |
| behavior_mean | CCAM `discuss_GW` | past frequency against next-twelve-months likelihood; one item of six |
| belief_post | CCAM `cause_recoded` | a conditional cause attribution in unordered categories; there is no ordered latent scale to convert, and `happening` is a proportion |
| funding_perceptions | CCAM `fund_research` | support for funding renewables, not a spending-adequacy judgement centred on a midpoint |
| distrust_post | — | no distrust item exists; distrust is not 100 minus trust |
| inst_trust_mean | — | no items about the EPA, NASA, NOAA, universities or the federal government |
| donation_ams | — | a behavioural money allocation |
| newsletter_signup | — | a recorded click |

Four of the ten are genuinely unanchorable by any survey — two behavioural
outcomes, one institution-specific battery, one construct (distrust) nobody else
measures. The other six fail on wording or scale, and the excluded climate-policy
group is the one where we can *prove* the failure would matter, because two
national samples disagree about it by more than break-even.

This is a thin result and it is the correct one. Three anchors that are probably
right within 4 points beat eleven that might be off by ten, because at ten points
of error the anchoring makes the distribution metrics **worse than doing nothing**.

---

## 8. Reproduction

```
python -c "from silicon_sampling.anchors import levels; print(levels.levels())"
python -c "from silicon_sampling.anchors import crosswalk; print(crosswalk.to_frame().to_csv(index=False))"
python -m silicon_sampling.anchors.validate     # about 100 s, prints every table in section 5
python -m pytest tests/test_anchors.py -q
```

```
$ python -m pytest tests/test_anchors.py -q
................................                                         [100%]
32 passed in 3.38s
```

`black --target-version py312` and
`flake8 --max-line-length=200 --extend-ignore=E203,W503` both pass clean on
`silicon_sampling/anchors` and `tests/test_anchors.py`. The rest of the suite was
green before this package existed (220 tests) and nothing here touches it; a later
full run shows one failure in `tests/test_icpc.py`
(`test_every_arm_can_be_driven_to_the_end`, an ICPC survey dry-run assertion added
by concurrent work on a different study) which is unrelated to anything in
`silicon_sampling/anchors`.

## 9. Limitations

* **The conversion assumption is untestable here.** The linear-on-options mapping
  is a choice; the bin-midpoint alternative moves the trust anchor by 4.3 points,
  which is comparable to the tolerance. Both are reported; neither can be checked
  against a slider version of the same battery, because no such data exists in
  this project.
* **The referent correction is confounded.** `TRUST_PEW` and `CLIM_TRUST` differ
  in stem as well as referent, so 3.92 points is an estimate of the referent shift
  plus whatever the stem difference contributes. It is applied to the trust
  battery only.
* **Standard errors ignore the survey design** beyond the weights. This is
  immaterial at the precision that decides anything here.
* **The break-even is measured on Voelkel, not Pfänder.** Voelkel's outcomes are
  all 0-100 with broadly comparable dispersions to Pfänder's, and the mechanism
  (a per-cell distribution metric degrading with a level shift) is
  instrument-independent, but the exact crossing points are Voelkel's.
* **TISP is 2023, CCAM's wave is Dec 2024, Pfänder is fielded 2026.** No temporal
  adjustment is made, and none is defensible without a trend model for the trust
  battery, which TISP's single US wave cannot support. CCAM's own trend suggests
  climate attitudes move a few points a year, so this is a real but unquantified
  bias.
* **Nothing here anchors dispersion.** The SDs are reported, Sheppard-corrected,
  because a later dispersion calibration will want them; no such calibration is
  implemented, and the variance ratio is the one metric level anchoring does not
  reliably improve.
