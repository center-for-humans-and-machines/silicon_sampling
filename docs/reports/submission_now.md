# Pfänder submission — status, prediction, and how to submit

Rebuilt on the three runs that just landed: two more Muse-Glimmer seeds on
Pfänder, and DeepSeek-V4-Flash on the CCC validation study.

---

## 1 · Cross-validation, in short

Four published megastudies, held out one at a time, every free choice fitted on
the other three, averaged over eight half-splits of the human sample.

| effect recovery (the leaderboard's sort key) | pooled *r* |
| --- | --- |
| **the shipped method** | **0.35** |
| membership by a per-fold rule instead of fixed | 0.32 |
| without the third model | 0.31 |
| every free choice refitted per fold | 0.29 |
| best single model, uncalibrated | 0.26 |
| *human replication reference* | *0.54* |

**The new data changed two things.**

*The third model got cheaper.* Muse-Glimmer now has three independent Pfänder
draws instead of one. Its measured reliability rose 0.580 → 0.769 — the
single-draw estimate had overstated its noise about twofold, because Muse answers
far more consistently per respondent than the Qwens and the estimator read that
consistency as variance. The submission's effect ensemble is now ten runs across
three models, reliability 0.964.

*The structural half is validated for the first time.* Levels, demographics and
dispersion all come from DeepSeek-V4-Flash, which previously had no run on one of
the four studies — so the fold search could not consider it, and every
distributional result described a donor the submission does not use. It has one
now, the search picks it on **every fold**, and the numbers roughly halve their
distance to the human reference:

| control-arm distribution | before | **now** | *human reference* |
| --- | --- | --- | --- |
| OVL | 0.663 | **0.784** | *0.859* |
| KS | 0.277 | **0.150** | *0.041* |
| W1 | 12.97 | **6.21** | *1.45* |
| variance ratio | 0.968 | **1.023** | *1.008* |
| demographic baseline RMSE | 11.36 | **6.04** | *2.03* |
| demographic parity gap | 7.35 | **3.43** | *1.66* |

Those earlier figures were not the method scoring badly — they were a substitute
donor being scored in its place. Nothing about the recipe changed to produce
them.

---

## 2 · What to expect on Pfänder

Point predictions for the **primary** entry. The interval on the sort key is the
benchmark's own cluster bootstrap over interventions, scaled to 16.

### Section 1 — intervention effect recovery

| metric | target | **prediction** | range | human reference |
| --- | --- | --- | --- | --- |
| `pearson_r` (sort key) | high | **0.34** | 0.21 – 0.47 | 0.545 |
| `spearman_rho` | high | **0.34** | 0.20 – 0.48 | 0.487 |
| `directional_pct` | high | **74** | 65 – 82 | 81.7 |
| `pearson_within` | high | **0.29** | 0.17 – 0.41 | 0.282 |
| `pearson_adj` | high | **0.51** | 0.32 – 0.70 | 0.858 |
| `rmse` | low | **2.8** | 1.4 – 4.8 | 2.68 |
| `rmse_adj` | low | **2.1** | 0.9 – 3.6 | 1.83 |
| `alpha` | 0 | **1.3** | 0.0 – 2.6 | 0.656 |
| `beta` | 1 | **1.14** | 0.55 – 1.65 | 0.490 |

### Section 2 — subgroup (condition × moderator) effects

| metric | target | **prediction** | range | human reference |
| --- | --- | --- | --- | --- |
| subgroup `pearson_r` | high | **−0.01** | −0.08 – 0.06 | 0.071 |
| subgroup `spearman_rho` | high | **0.03** | −0.04 – 0.10 | 0.073 |
| subgroup `directional_pct` | high | **51** | 48 – 54 | 52.3 |
| subgroup `pearson_adj` | high | **0.02** | −0.12 – 0.16 | 0.204 |

Still the weakest row, but better than the −0.08 predicted before the donor could
be validated. The human reference is only 0.07: at these cell sizes a fresh human
sample barely predicts the other half either.

### Section 3 — response distributions (control arm)

| metric | target | **prediction** | range | human reference |
| --- | --- | --- | --- | --- |
| `variance_ratio` | 1 | **1.02** | 0.90 – 1.15 | 0.993 |
| `ovl` | 1 | **0.79** | 0.72 – 0.85 | 0.925 |
| `ks` | 0 | **0.14** | 0.08 – 0.22 | 0.018 |
| `w1` | 0 | **5.5** | 3 – 9 | 0.539 |

Cross-validated directly on the shipped donor. Eight of thirteen outcomes are
additionally pinned to external level anchors, which the validation cannot use —
so if anything these are pessimistic.

### Sections 10–12 — demographic baselines, parity, stereotyping

| metric | target | **prediction** | range | human reference |
| --- | --- | --- | --- | --- |
| `baseline_rmse` | low | **5.5** | 3.5 – 8.5 | 0.371 |
| `parity_dpd` | low | **3.2** | 2.0 – 5.0 | 0.486 |
| `parity_worst` | low | **9.5** | 6 – 14 | 4.372 |
| `stereo_coef_rmse` | low | **4.4** | 3.0 – 6.0 | 0.845 |
| `stereo_r2_gap` | 0 | **−0.013** | −0.04 – 0.01 | 0.003 |

Shaded slightly better than the fold means because the submission blends party
offsets toward external gaps and the validation cannot.

### The dominant uncertainty

The same method scores **0.01 on one validation study and 0.46 on another**. That
spread is larger than every design choice in the recipe put together, and it
tracks how large a study's true intervention effects are relative to the
precision a half-sample can measure them with. Whether Pfänder sits at the easy
or hard end is not knowable from its protocol, and it applies to every entry on
the leaderboard, not only this one.

---

## 3 · Where the submission files are

```
/opt/silicon_sampling/data/pfander/submission/
├── primary/          ← the team's best effort
├── secondary-1/      ← identical, without global shrinkage
└── secondary-2/      ← uncalibrated single model, the baseline
```

Each is a complete copy of the benchmark's template with our content:
`predictions/`, `raw_data_deposit/`, `metadata.json`, `registration.md`,
`.zenodo.json`, `codebook.csv`, `survey/`.

**All three: PASS, 51 checks, 0 failures, 0 warnings.**

| entry | prediction file | SHA-256 |
| --- | --- | --- |
| primary | `mpib_T1_primary_v1.csv` | `80a58dd60cb7824d7f9b0330259b9953348091c6d321a54427f868b74f7a2e01` |
| secondary-1 | `mpib_T1_secondary-1_v1.csv` | `f6fe5b3ca08c8a10c3321991c6a1d6499200c4be43204ad20269359266138754` |
| secondary-2 | `mpib_T1_secondary-2_v1.csv` | `e6a1d5d38308d5707b0b5e75f1daa0169ff783f1020477ca59b7039122807f12` |

The first two fingerprints are new — those files were rebuilt on the added Muse
seeds. `secondary-2` is unchanged, as it must be: it uses one Qwen run.

---

## 4 · What you have to do to submit

**Deadline: 31 August 2026.**

**0 — Read `registration.md` section A.1 before anything else.** It now records that
this entry's pipeline — the code, the prompts, the calibration, the
cross-validation and the first draft of that form — was written by an **LLM
coding agent** (Claude Code / Claude Opus 5) rather than by people, with the
human decisions being: base models, which models, which validation studies, and
that a cross-validation was required. An earlier draft of the form asserted the
opposite and was wrong. If you disagree with how that is characterised, this is
the item to change before depositing — it is a ★ item, so it must be public and
accurate.

**1 — Fill five items in `registration.md`** (identical in all three entries; edit
once and copy). Each is marked `TODO-BEFORE-DEPOSIT` inline:

- **0.1** team member name(s), affiliation, corresponding contact
- **I.1** funding source (the factual half — no in-kind compute, no API, open
  weights, institutional hardware — is already written)
- **I.3** the blinding attestation — **mandatory, must be signed by a named
  person**; draft text is in place and needs a name and date
- **K.1** `code_doi`, only if you want to mint one
- **K.3** a monetary cost figure, only if one is required

**2 — Fill three fields in each `metadata.json`**: `team_name`, `contact`,
`creators[0].name`. Add an ORCID only if it is real — an invalid checksum makes
Zenodo reject the deposit with an opaque HTTP 500. Then regenerate the deposit
record:

```
python -c "from silicon_sampling.submission import zenodo as Z; \
           Z.write_zenodo('data/pfander/submission/primary')"
```

**3 — Decide about the raw logs.** Each entry ships the raw export of the run its
rows came from (14 MB). The **complete** generation logs — 234,000 session
transcripts across all 13 runs, plus parsed answers and every rejected draw — are
now recovered from the cluster, verified by checksum, and packed into 13 gzipped
tarballs totalling **242 MB** in `data/pfander/generation_logs/`. That is small
enough to mint as a separate Zenodo upload and link in K.2, which is what I would
do; offering them on request is the other permitted option.

**4 — One repository and one Zenodo deposit per entry.** For each of the three:
clone the template at `/opt/silicon-sample-submission`, copy the entry's contents
in, delete the shipped `example_*` files, connect the repo to Zenodo, publish a
**GitHub release**, and take the DOI.

**5 — Email all three DOIs and all three fingerprints together** to
`janlukas.pfaender@gmail.com`.

**6 — Optional but worth it: run `make check` once on a machine with R.** This
container has no R, so validation ran through a line-by-line Python port of the
benchmark's own `check.R`. Its schema agrees with the shipped `codebook.csv` on
every run, but the port is good evidence rather than proof.

### One thing to weigh before you send it

Our sample has **no `gender = "Other"` respondents**, which exactly matches the
published quota — Male + Female sum to the cell total in all nine cells and to
18,000 overall, and our draw reproduces that to the person. But the benchmark's
submission spec, codebook and its own placeholder data generator all carry three
gender levels, and `build_subgroup_pairs` ends with
`stopifnot(nrow(joined) == nrow(human_mod_side))` — so *if* the human sample
contains enough Other-gender respondents to identify the interactions, Section 2
stops for our submission rather than losing cells quietly.

The quota arithmetic makes that unlikely. The cheapest resolution is an email to
the organizers asking whether the human data contain gender-*Other* respondents —
a question they can answer without unblinding any outcome.
