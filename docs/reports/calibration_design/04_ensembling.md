# What "ensembling" means when the unit is a synthetic person

[← back to the summary](README.md)

Averaging models is the largest free improvement this project found, so it is worth
being exact about what is being averaged. At the level of a Tier-1 submission —
one row per synthetic respondent — there are three different operations that all
get called ensembling, and they behave very differently.

All three are constructible here because **every model answered the same
respondents**: the profile set is byte-identical across runs, so `profile_id`
`v00001` is the same person with the same demographics and the same seed in all
three samples. That is a deliberate property of the sampler, and it is what makes
the comparison below possible at all.

## The three operations

**(a) Effect-level averaging.** Fit each model's ATEs, average them, then rebuild
*one* model's respondents so their condition means land on the averaged targets.
The aggregate is the ensemble; every submitted individual still belongs to one
model.

**(b) Pooling respondents.** Submit a mixture: some rows are model A's
respondents, some are model B's. Every row is a real synthetic respondent that
some model actually produced. No post-processing at all.

**(c) Per-respondent averaging.** For each `profile_id`, average that person's
answers across models. Every row is one person, whose answers are the mean of
what three models said about them.

## Measured on Voelkel, against real participants

| scheme | what one row is | n | r | ρ | dir % | RMSE | var ratio | OVL | KS | W1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single: Qwen2.5-7B | a Qwen respondent | 6,203 | 0.408 | 0.311 | 61.1 | 3.620 | 1.057 | 0.553 | 0.398 | 16.5 |
| single: V4-Flash | a V4 respondent | 6,203 | 0.190 | 0.186 | 55.6 | 2.808 | 1.240 | 0.710 | 0.201 | 5.9 |
| **(a)** effect-level average | a Qwen respondent, means moved | 6,203 | **0.440** | **0.367** | 63.0 | **2.442** | 1.057 | 0.553 | 0.398 | 16.5 |
| **(b)** pooled, all three | a respondent from one model | 18,609 | 0.440 | 0.367 | 63.0 | 2.442 | 1.284 | 0.639 | 0.326 | 13.2 |
| **(b)** pooled, n held fixed | as above, thinned | 6,203 | 0.420 ± 0.056 | 0.351 | — | 2.613 | 1.248 | 0.641 | 0.323 | 13.5 |
| **(c)** per-respondent average | the mean of three people | 6,203 | 0.440 | 0.367 | 63.0 | 2.442 | **0.428** | 0.564 | 0.370 | 13.3 |
| *human replication* | *a real person* | *6,242* | *0.514* | *0.395* | *66.7* | *1.682* | *0.993* | *0.925* | *0.018* | *0.5* |

## The two things this shows

**At the aggregate level, all three are the same operation.** The ATE of a mixture
is the mixture of the ATEs, so with balanced n every scheme produces *identical*
effect estimates — r 0.440, ρ 0.367, RMSE 2.442, three times over. Ensembling buys
nothing at the effect level beyond averaging the effects, and it cannot: there is
no extra information in rearranging which rows carry the average.

So the honest answer to "how does ensembling work for individual participants" is
that **it does not.** The gain is entirely aggregate. Averaging works because the
models' *errors* on the 54 effects are close to independent — the 7B and 72B effect
vectors correlate only +0.315 with each other — so the mean of two noisy estimates
of the same signal is a better estimate. That argument is about 54 numbers, not
about 6,203 people.

**Where the schemes differ is what they do to the individuals**, and that is what
the benchmark's Section 3 scores:

- **(c) per-respondent averaging is actively harmful.** Averaging three independent
  draws for the same person divides the idiosyncratic part of their variance by
  roughly three, and the variance ratio collapses to **0.428**. That is precisely
  the under-dispersion pathology Section 3 exists to catch — synthetic respondents
  answering too much alike — and it is manufactured here by the ensembling method
  rather than by the models. It buys nothing in return, since its effects are
  identical to (a)'s.
- **(b) pooling is legitimate and has a real trade.** Every row is something a
  model actually produced, so nothing is fabricated. It *improves* OVL (0.553 →
  0.639), KS and W1 over Qwen alone, because the mixture inherits some of
  V4-Flash's much better response levels. But it inflates the variance ratio to
  1.284, because a mixture of populations with different means is over-dispersed by
  construction.
- **(a) preserves one model's individual-level texture exactly.** Its distribution
  columns are bit-identical to the donor's, which is the property that makes the
  component hybrid work: the aggregate can come from the ensemble while the
  individual-level structure comes from whichever model is best at it, chosen term
  by term.

## A near-miss worth recording

Pooling at fixed n first scored **r = 0.509** — better than every other scheme and
almost level with the human replication. Over 25 random assignments the mean is
**0.420 ± 0.056** and 0.509 is the maximum: the 100th percentile of the draws.
Thinning each model to a third of its respondents adds sampling noise, so fixed-n
pooling is *worse* than effect-level averaging (0.440), not better.

One draw, reported without the spread, would have looked like the best result in
the project.
