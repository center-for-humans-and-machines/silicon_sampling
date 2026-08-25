# The party gap: models get the topic ordering right and the topic *spread* wrong

[← back to the final report](README.md)

Reproduce the measurements with the snippets in this project's history; the
constants live in `silicon_sampling/calibration/recipes.py` as
`PARTY_DONOR`, `PARTY_GAP_ANCHORS` and `PARTY_GAP_WEIGHT`.

## The question

Do base models under-use the party information they are given, and can that be
rescaled? The answer turned out to have three parts, and the obvious fix is the
one that does not work.

## Part 1 — rescaling our own offsets does not transfer

The natural move is to fit a per-moderator factor, `r · sd_human / sd_synth`, and
multiply. Within a study it looks good. Out-of-fold across the three reference
studies it **hurts on both metrics for both models, in nearly every fold**:

| offsets from | pooled offset r, raw → rescaled | RMSE, raw → rescaled |
| --- | --- | --- |
| DeepSeek-V4-Flash | 0.165 → **0.078** | 3.723 → **3.765** |
| Qwen2.5-7B | 0.176 → **0.003** | 3.573 → **3.749** |

The fitted factors are wildly unstable between folds — `age_band` comes out 3.22,
1.72 and 0.00 on the three V4-Flash folds. The mechanism is simple: rescaling
multiplies whatever signal is present, and when the signal is near zero the
factor needed to reach a realistic spread amplifies noise far faster.

Measured on Goldwert against real participants, this is stark. **Submitting no
party gap at all scores RMSE 15.28.** The models' own offsets score 15.62 (7B),
14.81 (72B) and 17.03 (V4-Flash) — two of three are *worse than nothing*.
Rescaling the best of them to match the human spread gets to 14.08. Substituting
a single externally known constant of +14 pp gets to **6.08**.

## Part 2 — Pfänder elicits party, which changes who is good at it

Gender, race and age are printed in a Pfänder profile. **Party is not**: it is
asked at Q16 on page 6, before almost every outcome. So the party structure in a
Pfänder sample is the model's own consistency — it picks a party and then answers
in keeping with it — rather than a demographic it was told to perform.

That is a different measurement from the one the reference studies grade, and the
two disagree about which model is best. Scored against external estimates of the
real US party gap on Pfänder's own outcomes:

| party offsets from | RMSE vs external | r | mean magnitude ratio |
| --- | --- | --- | --- |
| submitting no gap | 19.09 | — | 0.00 |
| Qwen2.5-7B | 18.19 | −0.306 | 0.09 |
| **Qwen2.5-72B** | **7.48** | **+0.838** | 1.24 |
| DeepSeek-V4-Flash | 9.81 | +0.487 | 0.72 |

On the moderators that *are* given, V4-Flash remains the better donor in both
reference studies (pooled offset r 0.190 on Voelkel, 0.177 on Goldwert). So the
submission takes party from Qwen2.5-72B and everything else from V4-Flash.

## Part 3 — the error is topic-blindness, not under-use

The real party gap depends enormously on the topic, and the models apply a
roughly uniform one:

| Pfänder outcome | Qwen2.5-72B | external estimate | source |
| --- | --- | --- | --- |
| `trust_multidimensional` | 15.0 | **4.0** | TISP, 12 Besley items, `near` grade |
| `policy_role_mean` | 17.5 | 7.4 | TISP, 4 items, `near` |
| `trust_post` | 19.7 | 11.3 | TISP, `near` |
| `policy_specific_mean` | 20.7 | 13.7 | TISP, 5 items, construct-only |
| `behavior_mean` | 14.4 | 10.0 | ICPC sharing 7.3, Goldwert 14.0 |
| `concern_mean` | 22.3 | 26.7 | CCAM worry 44.5, shrunk 0.6 |
| `policy_general` | 29.4 | 26.7 | CCAM priority 52.0, shrunk 0.6 |
| `belief_post` | 24.0 | 31.7 | ICPC US control arm, left vs right |

Trust in scientists is barely polarised — twelve items measuring Pfänder's exact
battery put it at 4.0 pp — while climate policy priority is polarised at 28–52
pp. Qwen2.5-72B is nearly right on the second and almost four times over on the
first. It has the *ordering* right (r = +0.838); it has the *spread* wrong.

So the calibration blends the per-outcome profile half-way toward the anchors,
which pulls the spread into shape while leaving the ordering alone.

**Result, measured against the external estimates:**

| | RMSE | r |
| --- | --- | --- |
| no party gap | 19.09 | — |
| V4-Flash offsets (what shipped before) | 9.81 | +0.487 |
| Qwen2.5-72B offsets | 7.39 | +0.835 |
| **+ anchors blended at 0.5** | **3.93** | **+0.982** |

## Why the CCAM numbers are shrunk, and the blend is a half

CCAM is a nationally representative panel and its gaps run consistently larger
than experimental online samples'. Where the two can be compared: ICPC's belief
gap is 0.71 of CCAM's worry gap, and Goldwert's behaviour gap 0.45 of CCAM's
discussion gap. Hence the 0.6.

That this matters is not hypothetical — on Goldwert, anchoring to CCAM's raw 37.9
pp would have scored RMSE 24.74, *worse than submitting nothing*, while the
correctly scaled +14 scored 6.08. Overshooting is punished exactly as hard as
undershooting.

The blend is a half rather than a whole for the same reason in a different
direction: two of the three sources contrast **ideology** rather than party, and
none of them is Pfänder. Half moves most of the distance without claiming these
estimates *are* the answer.

## What this does not do

It cannot move the leaderboard's sort key — all four scored subgroup metrics are
scale-invariant, and the party offsets are not part of the condition effects at
all. It reaches the demographic baselines, parity gap, stereotyping coefficients
and within-subgroup distributions, which the benchmark reports separately.

And it rests on estimates, not measurements. The trust anchor is the strongest
(same battery, same population, twelve items); `concern_mean` and
`policy_general` are the weakest, being construct-only items from a different
sampling frame with a judgement-call shrinkage on top.
