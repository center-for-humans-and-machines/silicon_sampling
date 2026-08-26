# There are two Voelkel studies, and we used the less relevant one

[← back to the final report](README.md)

## Confirmed: two different studies

| | `data/Voelkel/` | `data/calibration/datasets/voelkel_etal2026.csv` |
| --- | --- | --- |
| study | Strengthening Democracy Challenge | Voelkel et al. (2026), *Nature Climate Change* 16(2), 214–225, "A registered report megastudy on the persuasiveness of the most-cited climate messages" |
| topic | antidemocratic attitudes, partisan animosity | climate message framings |
| outcomes | `PA`, `ADA`, `SPV`, `SocDistrust`, `BEPF`, … | belief, concern, policy support, behavioural intentions, donation |
| arms | 25 interventions | 10 framings + 3 placebo controls |
| n | 32,059 | 13,821 |
| used as a calibration study | **yes** | **no** |

Only 7 columns overlap between the two files, all of them generic (dates, gender,
age, education, race, `Condition`). They are unrelated datasets that share an
author.

**Voelkel 2026 is not Pfänder's target study.** Its arms are message framings —
Binding, Consensus I and II, Dire but Solvable, Free-Market, Gain, High Social
Distance, Purity, System Preservation, Warmth — against placebo controls about
neckties, baseball and dances. Pfänder's arms are scientist portraits, interviews,
corporate reliance, funding, peer review and misinformation. Different
interventions, different research question.

## Pfänder took its items from it

This is the part that matters, and `data/calibration/item_matching.csv` already
said so — "concern and seriousness items **adapted by the benchmark from this
study**" — with status `CONFIRM_IN_CODEBOOK`. It was never wired in. Confirming it
against the questionnaires:

| Pfänder outcome | Voelkel 2026 counterpart | match |
| --- | --- | --- |
| `concern_mean` | Climate Change Concern | **identical** — all three items verbatim, same 101-point 0–100 scale, same order randomisation |
| `policy_general` | General Mitigation Policies, item (iii) | **identical item** — "The U.S. government should do more to reduce global warming" |
| `behavior_mean` | Non-Political Behavioural Intentions | **partial** — 3 of 6 items verbatim (eat less meat; walk/bicycle/carpool/public transport; less air travel), same stem and scale; Pfänder swapped reusable bags, local food and plastic for solar panel, talking to friends, and donating |
| `policy_specific_mean` | Specific Mitigation Policies | construct-only — both are 4-item policy batteries, different policies |
| `belief_post` | Belief in Climate Change | construct-only — Pfänder condensed three items to one accuracy rating |
| `donation_ams` | Donation Behaviour | construct-only; scale differs (0–10 against 0–100) and needs checking |

Two outcomes are **the same questions on the same scale in the same population**.
That is a closer instrument match than anything else in this project, including
TISP, which is `near` grade but a different survey.

## What it gives us with no new sampling

Three things, all from human data alone.

**1. Level and dispersion anchors on six climate outcomes that currently have
none.** The submission anchors three outcomes, all from the trust battery via
TISP. These would add:

| Pfänder outcome | human control-arm mean | sd |
| --- | --- | --- |
| `belief_post` | 65.44 | 22.52 |
| `concern_mean` | 60.42 | 31.65 |
| `policy_general` | 68.01 | 29.32 |
| `policy_specific_mean` | 53.29 | 23.98 |
| `behavior_mean` | 33.83 | 28.89 |
| `donation_ams` | 61.54 | 45.32 |

This attacks the weakness the prediction report names explicitly — that levels,
dispersion and the residual scale all rest on three of thirteen outcomes.

**2. Directly measured party gaps, which correct the ones we shipped.** Measured
on party (not ideology) in a US online experimental sample, on matched items:

| outcome | shipped anchor | Voelkel 2026 | error |
| --- | --- | --- | --- |
| `concern_mean` | 26.7 | **37.7** | −11.0 |
| `policy_general` | 26.7 | **32.9** | −6.2 |
| `policy_specific_mean` | 13.7 | **25.3** | −11.6 |
| `behavior_mean` | 10.0 | **24.0** | −14.0 |
| `belief_post` | 31.7 | **22.8** | +8.9 |

The shipped anchors are wrong in both directions, and `behavior_mean` is off by a
factor of 2.4. They were built from CCAM with a judgement-call shrinkage of 0.6
and from ICPC's ideology split; these are the real thing.

**3. The best available prior on Pfänder's effect magnitudes.** Mean absolute
human effect across the ten framings is **1.70 pp** — against 1.125 for
Voelkel-SDC, 2.967 for Goldwert and 5.035 for ICPC. Pfänder is also a
climate-message megastudy, so this is the closest analogue, and it directly
narrows the `rmse`, `alpha` and `beta` ranges, which are currently the widest in
the prediction report precisely because effect magnitudes differ 4.5-fold across
the three studies we used.

## A fourth fold: yes, and the materials are unusually good

The replication package arrived in `data/calibration/SI/Voelkel/`, and it is the
best-conditioned study materials in this project.

**The intervention texts are in a Qualtrics export, not a PDF.** `Materials/CCC -
Questionnaire - Qualtrics.qsf`. Same study team as the Strengthening Democracy
Challenge, same Qualtrics conventions — so `silicon_sampling.voelkel.qsf` reads it
**unmodified**, recovering all thirteen arms from the survey's own display logic
with names matching the data file's `Condition` column exactly. No crosswalk
needed.

(`Interventions.csv` is not the stimuli — it is the 157-paper literature search
used to pick the most-cited messages.)

**No video and no audio in any arm**, which is the decisive difference from
Goldwert, where seven of seventeen arms were video- or graphic-core and had to be
dropped:

| arm | chars | images | video | audio |
| --- | --- | --- | --- | --- |
| Control Neckties / Baseball / Dances | 1,946 / 1,991 / 2,248 | 1 / 1 / 1 | 0 | 0 |
| Binding | 1,544 | 1 | 0 | 0 |
| Consensus I / II | 1,892 / 2,316 | 1 / 0 | 0 | 0 |
| Dire But Solvable | 1,998 | 2 | 0 | 0 |
| Free Market | 1,743 | 0 | 0 | 0 |
| Gains | 4,701 | 0 | 0 | 0 |
| High Social Distance | 1,804 | 2 | 0 | 0 |
| Purity | 1,828 | 1 | 0 | 0 |
| **System Preservation** | 2,720 | **12** | 0 | 0 |
| Warmth | 1,884 | 0 | 0 | 0 |

Every arm carries 1.5k–4.7k characters of prose. Only System Preservation needs a
real modality audit; the rest have at most two images against a substantial text.

**Also in the package**: `Data/CCC - Data - Recoded.csv` (13,821 rows, same
convention as the SDC file), the full R analysis scripts, and both questionnaire
PDFs. `voelkel_etal2026.csv` is the same 13,821 respondents with 49 extra derived
columns — weights, attrition flags and imputations.

## What is left to do

Not the risky part. What remains is a study package —
`paths`, `outcomes`, `convert`, `templates`, `profiles`, `run`, `export`, `score`,
`validate`, `cli` — mirroring the four that exist, plus a modality audit and a
stub-check pass before sampling. The QSF parser and the outcome composites are
already available.

One structural difference to handle: **CCC measures every primary outcome before
*and* after the message**, where the SDC measured only after. A faithful transcript
has to include the pre-measures, which lengthens it and raises the sampling cost.

Sampling 13,821 respondents costs roughly 9 h on the local 4090 for Qwen2.5-7B,
5 h on four H200s for Qwen2.5-72B and 13 h for DeepSeek-V4-Flash, plus queue.

## What it would and would not settle

It would make the cross-validation **four folds instead of three**, and add the
fold that most resembles the target: matched instrument on two outcomes, matched
population, same year, same genre of intervention.

It is worth being clear that this could make the recipe look **worse**. The
Goldwert fold is what dropped the fully-fitted variant to 0.325, and a fourth fold
drawn from a different corner of the space may do the same. The reason to run it is
to find out, not to improve the number.
