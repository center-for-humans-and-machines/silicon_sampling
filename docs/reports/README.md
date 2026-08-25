# Predicting the Pfänder megastudy with base-model silicon sampling

Final report. The submission is three Tier-1 entries in
`data/pfander/submission/{primary,secondary-1,secondary-2}`, each 18,000
synthetic respondents × 33 columns, all passing the benchmark's validator.

Sub-reports: [calibration design](calibration_design/README.md) ·
[level anchors](level_anchors.md) · [media loss](media_loss.md) ·
[Pfänder sample](pfander_silicon_sample/README.md) ·
[Voelkel validation](voelkel_validation/README.md)

## What we submit

| entry | recipe | why it exists |
| --- | --- | --- |
| **primary** | four Qwen runs' averaged effects, shrunk within-outcome by 0.5 then globally by 0.402; levels, demographic offsets and residual structure from DeepSeek-V4-Flash; three outcomes anchored to external survey levels | best on every metric we can measure out-of-fold |
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

The submission recovers **71% of what a second sample of real humans achieves**.
Directional agreement is 77.3 / 72.7 / 61.1% against the replication's 88.2 /
93.2 / 66.7%.

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

### 5. Text-only sampling, and where it cannot reach

Pfänder's instrument is **pure text** — no images, video, audio or embeds — so
the handicap does not apply to the target study at all. In Goldwert the seven
arms whose intervention is genuinely non-textual were excluded before any
accuracy number existed, and among the arms that remain, severity does not
predict recovery in any consistent direction. [Details and the full arm
listing.](media_loss.md)

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
