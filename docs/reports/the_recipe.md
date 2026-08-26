# The recipe, part by part, with ablations

[← back to the final report](README.md) · reproduce with
[`scripts/ablate_recipe.py`](../../scripts/ablate_recipe.py)

## Why one model can be indispensable and useless at once

DeepSeek-V4-Flash has a mean effect-recovery correlation of **−0.040** across the
three reference studies — worse than submitting nothing — and it supplies three of
the four terms in every submitted row. That looks contradictory until you measure
how much of a response the condition effect actually is.

| outcome | total sd | between-arm sd | share of variance |
| --- | --- | --- | --- |
| `trust_multidimensional` | 22.53 | 0.547 | 0.06% |
| `trust_post` | 28.26 | 0.497 | 0.03% |
| `distrust_post` | 31.05 | 0.463 | 0.02% |
| `policy_role_mean` | 24.13 | 0.519 | 0.05% |
| `belief_post` | 27.95 | 0.410 | 0.02% |
| **mean** | | | **0.03%** |

**The condition effect is about one fifty-ninth of one within-arm standard
deviation.** Everything else — where the distribution sits and how wide it is —
is 99.97% of the response.

So the two things are graded on quantities of wildly different size. V4-Flash
places the control level on the trust battery at 65.6 against an anchor of 67.6,
and its within-arm spread at 23.2 against a human 20.6. Its mean condition effect
is **1.58 pp**. Being unable to rank 208 contrasts averaging 1.6 pp is no evidence
at all about whether it can place a 65–70 level and a 20-point spread — those are
different measurements, and each is graded against its own external reference.

The asymmetry runs the other way too, and it is the reason this recipe exists at
all: the leaderboard sorts on Pearson r over exactly those 1.6 pp contrasts, so a
model can be excellent at 99.97% of the response and still be the wrong choice
for the sort key. Qwen2.5-7B is the mirror image of V4-Flash — it ranks
interventions best and produces the least plausible people (its trust battery has
a mean inter-item correlation of 0.474 where humans have 0.613).

Neither model is good. They are bad in complementary ways, and the terms are
separable, so each contributes only what it is good at.

## What is submitted

Every row is one synthetic respondent, built as

    response = level + condition effect + demographic offset + residual

with each term taken from whichever run predicts it best. The row's demographics
come from a CCAM quota profile; its residual donor is drawn **once** and reused
across all thirteen outcomes, so the respondent is one person rather than
thirteen unrelated draws.

| term | from | why |
| --- | --- | --- |
| rows and demographics | `qwen25_7b_demo` | quota-drawn education, income and party |
| condition effect | eight Qwen runs, averaged within model then across | best effect ranking; averaging removes sampling noise |
| level | `v4_flash`, 3 outcomes overridden by TISP | closest to the external level anchors |
| demographic offset | `v4_flash`, party from `qwen25_72b_demo` | best pooled offsets; best party structure |
| residual | `v4_flash`, scaled 1.12 | dispersion closest to human, then fitted to it |

## Effect side, ablated

Out-of-fold on the three reference studies. Pfänder publishes no participant
responses, so nothing about arm contrasts can be measured on Pfänder itself.

| variant | Voelkel | ICPC | Goldwert | mean r |
| --- | --- | --- | --- | --- |
| single: Qwen2.5-7B | 0.408 | 0.320 | 0.319 | 0.349 |
| single: Qwen2.5-72B | 0.340 | 0.323 | 0.325 | 0.329 |
| single: DeepSeek-V4-Flash | 0.190 | −0.246 | −0.064 | **−0.040** |
| average of the two Qwens | 0.460 | 0.341 | 0.394 | 0.398 |
| average of all three — **V4 included** | 0.440 | 0.243 | 0.286 | **0.323** |
| average + within-outcome shrink 0.3 | 0.638 | 0.334 | 0.312 | 0.428 |
| **average + within-outcome shrink 0.5 (shipped)** | 0.576 | 0.342 | 0.358 | **0.426** |
| average + within-outcome shrink 0.7 | 0.520 | 0.345 | 0.381 | 0.415 |
| shipped + global shrink | 0.576 | 0.342 | 0.358 | **0.426** |
| *human replication* | *0.514* | *0.640* | *0.642* | *0.599* |

Four things this settles.

**Averaging the two Qwens is the single largest free gain**, 0.349 → 0.398, and it
now replicates in all three studies rather than the one where it was found.

**Adding V4-Flash to the effect average costs 0.075**, 0.398 → 0.323. This is the
ablation that justifies the whole component split: V4-Flash must be excluded from
the term it is bad at and kept for the three it is good at.

**Within-outcome shrinkage is a real gain with a flat optimum** — 0.428 / 0.426 /
0.415 at 0.3 / 0.5 / 0.7 — so the pre-committed 0.5 is kept rather than fitting a
magnitude inside the flat region.

**The global factor moves r by exactly nothing**, which is not a null result but a
proof: a positive scalar cannot change a correlation. It is carried for RMSE and
β, where it takes β from 0.256 to about 1.0.

Not in the table because it cannot be measured on these studies: the shipped
entry averages **eight** runs where they have one per model. Measured reliability
rises from 0.870 to 0.964, projecting r ≈ **0.448**, or about 75% of a fresh human
replication.

## Structure side, ablated

Graded on Pfänder against external anchors — TISP for levels and dispersion on
its three `near`-graded outcomes, and `PARTY_GAP_ANCHORS` for the party gap. This
is the half that carries every distributional and demographic metric.

| structural donor | level error (pp) | sd ratio | \|sd ratio − 1\| | party RMSE |
| --- | --- | --- | --- | --- |
| **`v4_flash`** (shipped) | **2.59** | 1.020 | **0.063** | 4.01 |
| `v4_flash_demo` | 3.50 | 1.082 | 0.084 | 3.91 |
| `qwen25_72b_demo` | 6.06 | 1.145 | 0.145 | 3.88 |
| `qwen25_7b_demo` | 7.74 | 1.056 | 0.091 | 4.75 |

V4-Flash wins the two columns that matter by a wide margin. Level error is
measured with anchoring **off** so the donors actually differ; in the shipped
entry three of thirteen outcomes are pinned to TISP exactly, and this table is the
evidence for trusting the same donor on the other ten.

Party RMSE barely moves across donors because party offsets come from a separate
run in every variant — that is the point of `party_offsets_from`.

## Order of operations, and why it is fixed

1. **Component swap.** Terms are taken from their donors and the frame rebuilt.
   First, because everything after is fitted in percentage points on whatever
   effects will actually be submitted — reversing 1 and 2 would fit a shrinkage
   factor to one model's effects and apply it to another's.
2. **Effect transforms**: within-outcome shrink 0.5, then noise flattening on the
   two outcomes whose between-arm signal is indistinguishable from zero, then the
   global factor.
3. **Level and offset overrides**: TISP anchors on three outcomes, party offsets
   blended half-way to the external gaps. These are the terms the effect
   transforms do not touch, so they come last and cannot be undone.
4. **Format repair**: the twelve trust items are rebuilt and the composite taken
   as their mean, and the binary outcome is moved by flipping rows rather than by
   adding — recomposing a 0/1 column additively once collapsed a signup rate from
   0.311 to 0.003.

The audit after every build reports effect drift; the shipped entry is 4.5e-04 pp,
and composite drift 4.3e-14.

## What is extrapolated, and what is measured

Measured against real people: everything in the effect table, the pooled
demographic offsets on Voelkel and Goldwert, and the party gaps.

Measured against external survey anchors on **three of thirteen outcomes**: the
levels and the dispersion. The shipped residual factor of 1.12 is fitted on those
three and applied to all thirteen, exactly as the level anchors are trusted on
three and the donor choice extrapolated to the other ten. If TISP's trust battery
is unrepresentative of Pfänder's other outcomes, both extrapolations are wrong
together.

Not measured at all: whether any of this transfers to Pfänder's actual effects.
No one can measure that until the human data is released.
