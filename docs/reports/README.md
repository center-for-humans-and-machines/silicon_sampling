# Predicting the Pfänder megastudy with base-model silicon sampling

Final report. The submission is three Tier-1 entries in
`data/pfander/submission/{primary,secondary-1,secondary-2}`, each 18,000
synthetic respondents × 33 columns, all passing the benchmark's validator.

Sub-reports: [**what to expect on Pfänder**](prediction.md) ·
[**nested cross-validation**](cross_validation.md) ·
[the recipe, part by part](the_recipe.md) ·
[two Voelkel studies](voelkel_2026.md) ·
[calibration design](calibration_design/README.md) ·
[level anchors](level_anchors.md) · [media loss](media_loss.md) ·
[Pfänder sample](pfander_silicon_sample/README.md) ·
[Voelkel validation](voelkel_validation/README.md)

## What we submit

| entry | recipe | why it exists |
| --- | --- | --- |
| **primary** | eight Qwen runs' averaged effects (four of each model, averaged within model first), shrunk within-outcome by 0.5 then globally by 0.416; rows from the quota-demographics 7B run; levels, offsets and residual structure from DeepSeek-V4-Flash with residuals scaled 1.12; party offsets from the quota-demographics 72B run, blended half-way to external gaps; three outcomes anchored to external survey levels | best on every metric we can measure out-of-fold |
| secondary-1 | the same, without the global shrinkage | hedges the one calibration constant that would be badly wrong if Pfänder's real effects are much larger than the three reference studies' |
| secondary-2 | raw Qwen2.5-7B, uncalibrated | the no-assumptions baseline; makes every calibration's contribution measurable after the fact |

## How well it works, measured against real people

Pfänder publishes no human data, so every number here comes from three studies
that do — Voelkel's democratic-norms megastudy, the ICPC climate tournament and
Goldwert's climate-advocacy megastudy. Effects are fitted on one half sample of
real participants; the yardstick is what the *other* half achieves predicting
the first.

**Pearson r against real human intervention effects:**

| | Goldwert | ICPC | Voelkel | mean |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 0.319 | 0.320 | 0.408 | 0.349 |
| Qwen2.5-72B | 0.325 | 0.323 | 0.340 | 0.329 |
| DeepSeek-V4-Flash | −0.064 | −0.246 | 0.190 | **−0.040** |
| average of the two Qwens | 0.394 | 0.341 | 0.460 | 0.398 |
| **the shipped recipe** | 0.358 | 0.342 | 0.576 | **0.426** |
| *a fresh human replication* | *0.642* | *0.640* | *0.514* | *0.599* |

That row is a **proxy**: the reference studies have one run per model, so it
measures a two-run ensemble. The submitted entry averages eight, whose measured
reliability is 0.964 against the proxy's 0.870, which projects **r ≈ 0.448 —
about 75% of what a second sample of real humans achieves**. Directional
agreement is 77.3 / 72.7 / 61.1% against the replication's 88.2 / 93.2 / 66.7%.

## The five things that actually mattered

### 1. The questionnaire was wrong, and it dominated everything else

The largest single gain in the project came from fixing the *stimulus*, not the
statistics. The ICPC and Goldwert conversions printed slider endpoint labels
("Not at all accurate … Extremely accurate") without ever stating the 0–100
range. Models answered on an implicit 0–10 scale: 80–94% of slider answers were
≤ 10, against 8–31% for humans.

| mean control-arm level error | before | after |
| --- | --- | --- |
| Goldwert, Qwen2.5-7B | 30.7 pp | **9.7 pp** |
| Goldwert, Qwen2.5-72B | 32.1 pp | **9.8 pp** |
| Goldwert, V4-Flash | 19.8 pp | **8.1 pp** |
| ICPC, Qwen2.5-7B | 38.8 pp | **18.3 pp** |
| ICPC, Qwen2.5-72B | 47.0 pp | **21.9 pp** |
| ICPC, V4-Flash | 23.1 pp | **18.9 pp** |

Goldwert's `march` outcome went from −28.2 pp to −0.09 pp (39.40 human against
39.32 ours). It also compressed the *effects* by the same factor, which is why
every calibration constant fitted before the fix was wrong — the shrinkage
factor roughly doubled once refitted on corrected samples.

A full fidelity audit found five further defects, each fixed: a block randomiser
permuting the wrong blocks so a summary could precede the corrections it
summarised; piped correction text and photographs rendering as placeholders
though the export held them; media-only screens rendered as bare brackets; a
scored outcome asking about a video nobody was shown; and a blank control arm
where participants had actually watched a five-minute knot-tying video, verified
against the source (`lengthSeconds=297`, published median duration 307.6 s over
1,739 participants).

### 2. Different models are good at different terms, and the terms are separable

Writing a synthetic response as

    y = level + effect + demographic offset + residual

the scored analyses read those terms almost disjointly, so they can be taken
from different models. Qwen ranks interventions; V4-Flash gets levels,
demographic structure and respondent coherence right. **V4-Flash's effects are
worse than useless** — mean r −0.040, and 22.7% directional agreement on ICPC,
which is worse than a coin flip — so it contributes everything *except* the
effects. That combination beats both parents.

### 3. Averaging models, and averaging seeds

Averaging the two Qwens' effect vectors lifts mean r from 0.349/0.329 to 0.398,
and it now replicates in all three studies rather than only the one where it was
found. It works because the models' errors are close to independent: their
effect vectors correlate only 0.32–0.78.

Seeds matter for the same reason. About 13% of Qwen2.5-7B's Pfänder effect
variance and 26% of Qwen2.5-72B's is nothing but which respondents the run drew.
Three independent estimates agree that averaging both seeds of both models is
worth about **+0.015 r**: halving the calibration samples costs 0.014 r;
Spearman-Brown on the seed reliabilities predicts ×1.03–1.07; and the two-run
ensemble correlates 0.866 with the ensemble built from the second seeds, which
extends to 0.928 at four runs.

Note what ensembling does *not* do. At the aggregate level all three ensembling
schemes are identical, because the ATE of a mixture is the mixture of the ATEs —
[the details are here](calibration_design/04_ensembling.md). Per-respondent
averaging is actively harmful: it divides idiosyncratic variance by three and
collapses the variance ratio to 0.428, manufacturing exactly the
under-dispersion pathology the benchmark exists to catch.

### 4. The two shrinkages are one decision, not two

Global shrinkage cannot move Pearson r at all — a positive scalar cannot change
a correlation — but it is worth a large slice of RMSE and all of β. Within-outcome
shrinkage *does* move r, from 0.398 to 0.426 out-of-fold.

They interact, and this is where the project's last real error was: the best
global factor falls from 0.475 to 0.250 as the within factor rises from 0.2 to
1.0. Pairing the no-within value with within-shrinkage 0.5 overshoots to β 1.53.
The shipped pair is (within 0.5, global 0.402 after the four-run noise
adjustment), giving β 1.018 out-of-fold.

### 5. The party gap is topic-blind, not merely too small

Asked whether the models under-use party, the answer is yes — but the shape of
the error is what makes it fixable. The real Democrat–Republican gap depends
enormously on topic: TISP's twelve Besley items, Pfänder's exact trust battery,
put it at **4.0 pp** for trust in scientists, while CCAM puts climate policy
priority at **52 pp**. The models apply a roughly uniform gap, so Qwen2.5-72B
gives 14.6 pp for the trust battery and 27.9 for policy — nearly right on the
second, almost four times over on the first.

It has the ordering right (r = +0.838 with the external estimates) and the
spread wrong, so the fix is to reshape the profile rather than to inflate it.
Party offsets now come from Qwen2.5-72B, because Pfänder *elicits* party rather
than printing it in the profile, and the per-outcome profile is blended half-way
to external anchors. Party-gap error against external truth falls **19.09 →
9.81 (what shipped) → 7.39 → 3.93 pp**.

The obvious version of this — fitting per-moderator rescaling factors — does not
work, and the negative result is worth as much as the positive one: out-of-fold
it takes offset r from 0.165 to 0.078 and from 0.176 to 0.003. Rescaling
multiplies signal that is not there. [Details.](party_calibration.md)

### 6. The response-shape gap is spread, not digit preference

These models answer with more within-cell spread than real participants — pooled
over the sliders of all three reference studies the synthetic standard deviation
is about 1.07× the human one, and larger in seven of nine (study, model) cells.
Scaling the **residual term** by 0.90 raises the KDE overlap in all nine, by
+0.026 on average. It reaches only the within-cell spread: level, effects and
demographic offsets are separate terms of the decomposition, so this cannot move
the sort key.

The obvious competing explanation is wrong, and worth recording. The models are
far spikier than humans on round numbers — Qwen2.5-72B puts **78%** of 0–100
slider answers on a multiple of ten against a human **34%**. But OVL is a
kernel-density overlap and W1 and KS are ECDF-based, so jittering answers off the
spikes moved OVL by 0.000 in eleven of twelve cells. A real and visible artefact
that none of the scored metrics can see.

### 7. Letting the model invent its demographics is a mistake

Pfänder prints gender, age band and race in the profile but leaves education,
income and party to the model. Asked to invent them, all three models produce an
affluent, educated, Democratic United States:

| | invented | quota-drawn | US population |
| --- | --- | --- | --- |
| income `< $30,000` | 0.8% | **12.9%** | ~13.5% |
| — in the control arm | **18 respondents** | **264** | (benchmark minimum: 30) |
| education `< high school` | 3.5% | 9.3% | ~9% |
| party Republican | 13.1% | 28.2% | ~29% |

The control-arm count is the part that matters: `Less than $30,000` is the
dummy-coding reference level for every income interaction, and at 18 respondents
it falls below the benchmark's minimum group size, so it is skipped and income
effects are estimated against a level that is not there. Re-sampling on
CCAM-drawn profiles fixes it, and **costs nothing on the effects** — the quota
run correlates 0.864 with the three elicited seeds where those seeds correlate
0.846 with each other, so it sits inside the seed-noise band.

**The predicted risk went the other way, which is worth recording.** Because
Pfänder *elicits* party rather than printing it, and because models
under-express *given* demographics in all three reference studies, handing the
model its party looked likely to flatten the gaps. It improved them for every
model: Qwen2.5-7B from r = −0.306 to +0.559 against the external gaps,
Qwen2.5-72B from +0.838 to **+0.912**, V4-Flash from +0.487 to +0.838.

### 8. The response-shape correction had to be reversed on the target study

Worth recording as the clearest case in this project of a calibration that
transferred badly. Measured on the reference studies' raw samples, these models
are over-dispersed by about 1.07×, and shrinking residuals by 0.90 raises the KDE
overlap in **all nine** (study, model) cells. That shipped.

On Pfänder it is wrong twice over. V4-Flash's raw dispersion here is already
right — sd ratio 1.014 against human, because Pfänder's headline outcomes are
multi-item composites whose averaging removes the noise the single sliders
carried. And the reconstruction shrinks spread further on its own, so the entry
came out **13% under-dispersed**.

What made the reversal possible is that TISP measures Pfänder's own questions
closely enough to grade *dispersion*, not just levels: standard deviations of
20.62, 27.99 and 26.02 on the three `near`-graded outcomes. Fitting the factor
end-to-end against those, and iteratively because clipping makes the response
sublinear, gives 1.12 and a mean dispersion ratio of **1.011**.

| residual factor | mean sd ratio | mean abs. error |
| --- | --- | --- |
| 0.90 (transferred) | 0.870 | 0.130 |
| 1.035 | 0.962 | 0.071 |
| **1.12 (shipped)** | **1.011** | **0.057** |

### 9. Text-only sampling, and where it cannot reach

Pfänder's instrument is **pure text** — no images, video, audio or embeds — so
the handicap does not apply to the target study at all. In Goldwert the seven
arms whose intervention is genuinely non-textual were excluded before any
accuracy number existed, and among the arms that remain, severity does not
predict recovery in any consistent direction. [Details and the full arm
listing.](media_loss.md)

## The largest theoretical lever, already captured

More than half the variance the pooled correlation rewards is *between* outcomes
rather than between arms — 51.7% on Voelkel — so knowing which outcomes move at
all is worth more than ranking the interventions. An earlier estimate put an
oracle that predicts only the per-outcome mean effect at r = 0.752 against a
human replication's 0.514, which made borrowing a profile from another study look
like the biggest available gain.

It is not available, and it is not needed. The profile is a real, reliable thing
within a study — split-half reliability 0.726 on Voelkel and 0.872 on Goldwert —
but the three studies share almost no outcomes, so nothing can be transferred
item to item. What does transfer is a regularity: **the construct a study's
interventions target ranks first every time.** Voelkel's partisan animosity is
first of nine, ICPC's climate belief first of four, Goldwert's conversation first
of eleven.

Pfänder's interventions target trust in climate scientists, and our models
already put `trust_multidimensional`, `trust_post` and `distrust_post` at ranks
**1, 2 and 3 of 13**, with the most distal outcome, `behavior_mean`, last. The
profile already agrees with the only cross-study regularity we can establish, so
substituting a borrowed one would risk more than it could gain — consistent with
the measured break-even, which requires a transferred anchor to reach ρ ≈ 0.7
before it beats leaving our own profile alone.

## What we would not claim

- **Three studies is a three-point transfer check, not cross-validation.** No
  mean across three folds has a usable standard error, and none is reported. What
  three folds can show is whether a calibration dominates in every fold and
  whether its parameter is stable; that is what the selection rules use.
- **The media-loss null is weak.** It rests on 10–11 arms per test with the worst
  cases already removed, and would miss a moderate effect easily.
- **Subgroup recovery is at or below zero** for both models — scored subgroup
  `pearson_r` is −0.041 for Qwen and +0.001 for V4-Flash against a human
  replication's +0.146. The demographic calibration improves the *reported*
  subgroup analyses without any claim that the underlying signal is real.
- **The 72B/V4-Flash split on respondent coherence is unresolved.** Against human
  correlations from TISP, Qwen2.5-72B on quota demographics is much the best on
  *cross-outcome* coherence (mean error 0.072 against V4-Flash's 0.115) while
  V4-Flash is much the best on *within-battery* consistency (0.026 against
  0.162 — 72B is over-consistent at α 0.978 where humans are 0.950). V4-Flash
  keeps the structural role because it also wins on the two things the benchmark
  actually scores here, levels and dispersion; the cross-outcome advantage
  reaches nothing Pfänder scores. If it scored a cross-construct composite, the
  choice would go the other way.
- **`letter_content` in the Goldwert samples on disk is still degenerate.** The
  template is fixed; the samples predate the fix. It is not a scored outcome and
  was not worth re-sampling for alone, and that judgement is recorded rather than
  hidden.
- **The remaining level failures are structural.** ICPC's `wept` effort task sits
  ~48 pp below humans in every model, and binary opt-ins (`newsletter`,
  `petition`) run 13–38 pp high everywhere. Text-only base models do not
  reproduce effortful behaviour or the friction of a real signup.

## Reproducing

```bash
python scripts/loso.py --model qwen25_72b_v3   # leave-one-study-out selection
python scripts/bakeoff.py                      # every recipe on every analysis
python scripts/media_loss.py                   # the non-textual-arm check
python scripts/build_entries.py                # writes all three entries
```
