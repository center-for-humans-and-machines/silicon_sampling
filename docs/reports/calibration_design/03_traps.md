# Three ways this analysis nearly lied to us

[← back to the summary](README.md)

Each of these produced a plausible, publishable-looking number that was wrong, and
each was caught only by insisting a claim be reproduced rather than read. They are
recorded because all three are easy to repeat and none announces itself.

## 1. A cross-validation design that could not see any calibration

**The symptom.** Five calibration candidates, scored out-of-fold, returned
*bit-identical* held-out Pearson r on every fold. Read naively that is the
project's second-biggest risk realised — "no calibration beats the uncalibrated
sample" — and it would have been reported as a finding.

**The cause.** Every candidate on the menu moves pooled Pearson r *only* by
re-weighting outcomes against each other. So the fold has to contain several
outcomes for the transform to be visible inside it at all:

- **Leave-one-condition-out** gives each outcome a single observation. The
  per-outcome mean equals that observation, so within-outcome shrinkage is the
  **identity**.
- **Leave-one-outcome-out** gives the fold one outcome. Within-outcome shrinkage
  is then an **affine** map of that outcome's values — and Pearson r is
  affine-invariant.

Both were tried, in that order, before the pattern was obvious. Neither is a
silly choice: leave-one-condition-out is the *natural* one, because conditions are
the clusters the benchmark itself bootstraps over.

**The fix.** Folds are groups of outcomes, and leave-one-study-out once more
studies land. The general rule: **a fold must preserve whatever structure the
transform operates on.** A pooling transform cannot be cross-validated on folds
that destroy the pooling.

**The lesson that generalises.** "Every candidate scored the same" is not a result
about the candidates. It is a result about the experiment.

## 2. An audit that described the wrong file

**The symptom.** The Tier-1 apply path reported effect drift of 2e-14 — exact to
floating point — while the primary outcome's realised shrinkage was 7% off its
target (0.170 against a requested 0.159).

**The cause.** The audit ran on the intermediate frame, before two steps that both
move a condition mean: the binary outcome's flip, which approximates a rate at a
0/1 grain, and the clipping of the twelve trust items to 0–100. Respondents whose
items already sat at a boundary could not absorb their share of the shift, so the
composite landed short by 0.273 raw points — 36% of a shrunk effect of 0.75 pp.

**The fix.** The audit re-measures on the finished frame and carries the pre-format
figures alongside, so the cost of the format constraints stays legible instead of
being absorbed into a worse number. And the shift itself now redistributes across
items that still have headroom, iteratively, the same way the condition-mean clip
works across rows. Drift on the primary outcome fell to 1.6e-14 and the realised
ratio to exactly 0.1590.

**The lesson.** An audit must describe the artefact that will actually be
submitted. One that describes an intermediate is worse than no audit, because it
is trusted.

## 3. A hybrid that inherited none of what it was built to inherit

**The symptom.** A component hybrid designed to take DeepSeek-V4-Flash's
respondent coherence produced respondents *less* coherent than either input model:
mean cross-outcome correlation +0.030, against +0.138 for Qwen2.5-7B and +0.341
for V4-Flash.

**The cause.** Residuals were drawn independently per outcome, so every rebuilt
row got a fresh personality for each outcome. Coherence is a property of the
residual *vector*, not of any single residual's distribution — and a per-outcome
draw preserves each margin perfectly while destroying the joint structure entirely.

**The fix.** Donor rows are chosen once per respondent, and every outcome reads the
same donor. Cross-outcome correlation went to +0.334, essentially V4-Flash's own
figure; trust against distrust from −0.033 to −0.365 against V4-Flash's −0.270.

**Why it was nearly missed.** Every per-outcome metric was correct throughout. The
marginal distributions, the variance ratios, the effects, the levels — all exactly
as intended. Only a quantity nobody had thought to check was destroyed.

## A pattern across all three

In each case the *scalar summaries* looked right and something structural was
broken: the fold design preserved every margin and destroyed the contrast; the
audit reported a true number about a frame nobody would submit; the hybrid matched
every marginal and lost the joint. Marginals are easy to check and easy to satisfy
by accident.

Two habits caught all three, and both are cheap:

- **Compute the invariance you are relying on, as a test.** "A global rescale
  cannot change Pearson r" became an assertion over three factors and six metrics,
  which is also what revealed that a global rescale's per-fold differences are
  floating-point dust and must be scored as *ties* rather than as wins and losses.
- **Recompose a run from itself and require nothing to change.** That single check
  caught the residual-dispersion bug (a self-recomposition under-dispersing to a
  variance ratio of 0.758) and would have caught the coherence bug immediately had
  it been extended across outcomes rather than run one outcome at a time.

A related instance from the parallel work on the ICPC study package is worth
naming here because it is the same shape. Twenty-four questionnaire items were
bound to the wrong published column — a permutation within each battery. The check
that was supposed to catch it compared battery **means**, which are
permutation-invariant, and returned a genuine `max_abs_diff` of 0.0. The test
alongside it asserted that each column *name existed*, which tests the
implementation rather than the requirement. Detecting a permutation needs
**per-item** comparison, and it needs running on every battery rather than the one
that was spot-checked.
