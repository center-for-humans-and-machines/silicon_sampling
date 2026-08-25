# Which arms lose the most to a text-only model, and whether it shows

Reproduce with [`scripts/media_loss.py`](../../scripts/media_loss.py). Severity
ratings come from each study's `modality_audit.csv`, written during the fidelity
audit by opening every stimulus, and **before** any of the accuracy numbers here
existed.

## The short answer, in three parts

1. **Pfänder — the study we actually submit to — has no non-textual stimuli at
   all.** Zero `<img>`, `<video>`, `<audio>`, `<iframe>`, `<embed>` or `<object>`
   tags in its questionnaire, and no media files beside it (the 19 files in
   `questionnaire_files/` are the Quarto site's own JS and CSS). The one string
   "YouTube" in the instrument is inside a media-consumption item — *"Online news
   (e.g., news websites, podcasts, YouTube)"* — not a stimulus. So the text-only
   handicap does not apply to the target at all.
2. **The arms where it would have applied worst were already excluded**, in
   Goldwert, by the audit rather than by us.
3. **Among the arms that remain, severity does not predict accuracy.** Across
   three models and two studies the six rank correlations between severity and
   per-arm recovery are −0.41, −0.23, +0.10, +0.17, −0.18 and +0.23 — scattered
   either side of zero with only 10–11 arms per test.

## Goldwert: severity 3 is exactly the excluded set

`media_loss == 3` ("the intervention's core is not in the text") coincides
perfectly with `usable == no`. Seven of seventeen treatment arms:

| arm | what carries the intervention |
| --- | --- |
| `ClimatePolicyLiteracy` | one hosted video; three comprehension items ask what it said |
| `CoBenefits` | one Qualtrics graphic, unfetchable — an image of text nobody has seen |
| `GlobalHealthThreat` | a Lancet video, with a true/false matrix grading it |
| `GuiltCollResponsibility` | three screenshots of an NYT article; items quote figures only inside them |
| `ShiftFocusIndColl` | a recorded talk; all three follow-ups grade it |
| `BipartisanEliteCues` | two White House speech clips |
| `ActivistPerspective` | a documentary trailer, and nothing else |

We sample the other ten plus control. **The ten we evaluate on, ordered by what
a text-only model misses:**

| severity | arms | what is lost |
| --- | --- | --- |
| **2 — substantive** | `BindingMorals`, `CollEfficacyEmoBenefit`, `ThreatInjustEfficacy` | something the argument leans on is only in the media: four national-park photographs the respondent is asked to *rate*; five deleted graphics plus a climate-march clip; two screens whose entire content is photographs of children's protest placards |
| **1 — peripheral** | `Control`, `SystemJustification`, `EcologicalDisruptions`, `IndStructuralChange`, `DynamicAngerNorm` | affect and atmosphere, or a quantity the prose states in words anyway: a full-frame US flag and a Macy's parade as patriotic prime; a row of dead songbirds; one line chart whose starting level is only in the picture |
| **0 — nothing** | `MispCorrectionRisks`, `HopeAngerNarratives`, `LetterFuture` | the text carries everything the screen carried |

`Control` is rated 1 rather than 0 deliberately: its content is null — a
five-minute knot-tying video — but the five minutes of attention it consumed are
not, and every effect in the study is a contrast against it, so the transcript
describes the screen rather than leaving it blank.

## ICPC: images throughout, none of them load-bearing

ICPC's audit kept all twelve arms. Its stimuli are static images inside
self-contained argumentative prose, so the split below is image-heavy against
image-light, not a loss rating:

| bucket | arms (asset count) |
| --- | --- |
| 5+ images | `CollectAction` (20), `Identity-Social-Norms-Intervention` (16), `SystemJust` (12), `PsychDistance` (5) |
| 1–4 images | `NegativeEmotions` (4), `SciConsens`, `PluralIgnorance`, `Letter2Future`, `DynamicNorm`, `BindingMoral` (1 each) |
| none | `Control`, `FutureSelfCont` |

Two of these are stored in the samples under their raw Qualtrics name rather than
the audit's alias — `Identity-Social-Norms-Intervention` for `WorkTogetherNorm`,
`Letter2Future` for `LetterFutureGen`. Merging on the alias alone silently
dropped both, which took the study's second image-heaviest arm out of the
comparison; the script now matches on either spelling.

## The measurement

Per-arm accuracy is measured on **within-outcome demeaned, standardised**
effects. Raw effects are dominated by *which outcome* moves rather than *which
arm* moves it, and that component is common to every arm, so it would wash out
the contrast being tested.

Goldwert, Qwen2.5-72B (audited sample):

| severity | arms | directional % | mean r per arm | MAE |
| --- | --- | --- | --- | --- |
| 0 | 3 | 60.6 | +0.37 | 0.85 |
| 1 | 4 | 36.4 | −0.32 | 1.09 |
| 2 | 3 | 72.7 | +0.11 | **0.75** |

The severity-2 arms score *best* on two of three columns. ICPC runs the same way
and harder: its two image-heaviest arms, `CollectAction` (20 images) and
`Identity-Social-Norms-Intervention` (16), are the two best-recovered arms in the
study — 100% directional agreement each, per-arm r of +0.88 and +0.70.

That is not evidence that losing media helps. It is evidence that with three or
four arms per bucket this design cannot resolve anything. All six
severity/accuracy rank correlations, across three models and both studies, sit
between −0.41 and +0.23 with no consistent sign:

| model | Goldwert ρ(sev, r_arm) | ICPC ρ(sev, r_arm) |
| --- | --- | --- |
| Qwen2.5-7B | −0.405 | +0.102 |
| Qwen2.5-72B | −0.225 | +0.168 |
| DeepSeek-V4-Flash | +0.225 | −0.178 |

The two studies disagree in sign for every model, and the models disagree in sign
within each study.

## What this does and does not license

**It does not license the claim that media loss is harmless.** Seven Goldwert
arms were dropped precisely because it is not, and the test below is restricted
to the survivors — a range-restricted comparison with 9–10 points, which would
miss a moderate effect easily.

**It does license using these studies to calibrate Pfänder.** The concern would
be that κ, fitted on arms partly handicapped by missing media, is then applied to
a study with none — which would make the calibration pessimistic, not optimistic.
Since the handicap is undetectable among the arms actually used, and Pfänder is
pure text, the transfer is if anything conservative in the safe direction.

**One genuine caveat for the write-up.** Effect *recovery* is what was tested
here. Response *levels* are a separate matter, and the arms whose stimulus is a
described photograph rather than the photograph itself may still differ in level
from what a human who saw it reported. That is absorbed by the level-anchoring
step rather than measured here.
