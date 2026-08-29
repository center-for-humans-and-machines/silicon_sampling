> **Superseded** by [submission_now.md](submission_now.md), rebuilt after two more
> Muse-Glimmer seeds and a CCC run of DeepSeek-V4-Flash landed. The prediction
> files and their fingerprints changed; the structural half of the recipe is now
> cross-validated, which it was not when this was written.

# Pfänder submission — what is built, and what you still have to do

[← back to the final report](README.md) · method:
[the recipe](the_recipe_now.md) · validation:
[the verified cross-validation](four_study_cross_validation_verified.md)

Three complete Tier-1 entries are built and pass the format gate with **zero
warnings**. Nothing has been submitted, uploaded or emailed — everything below is
local.

---

## Where everything is

```
/opt/silicon_sampling/data/pfander/submission/
├── primary/          ← the team's best effort
├── secondary-1/      ← same, without global shrinkage
└── secondary-2/      ← uncalibrated single model, the baseline
```

Each directory is a complete copy of the benchmark's submission template with our
content in it:

| file | state |
| --- | --- |
| `predictions/mpib_T1_<entry>_v1.csv` | **built** — 18,000 respondents × 33 columns |
| `raw_data_deposit/mpib_T1_<entry>_v1_raw_export.csv` | **built** — the raw run the rows came from |
| `metadata.json` | **built** — fingerprints, models, coverage; 3 fields need you |
| `registration.md` | **built** — all 37 checklist items answered; 5 need you |
| `.zenodo.json` | **built** — generated from `metadata.json` |
| `codebook.csv`, `survey/` | copied verbatim from the benchmark template |

**The three prediction files and their fingerprints:**

| entry | file | SHA-256 |
| --- | --- | --- |
| primary | `mpib_T1_primary_v1.csv` | `d89c556a057652a920eac642b46c466afb45c79811689319914c6c8db1187ebe` |
| secondary-1 | `mpib_T1_secondary-1_v1.csv` | `8245a9be347b5eef15d677c79fac02f7eafb521aef00bf6acc4c67e11acd1a42` |
| secondary-2 | `mpib_T1_secondary-2_v1.csv` | `e6a1d5d38308d5707b0b5e75f1daa0169ff783f1020477ca59b7039122807f12` |

Fingerprints will change if anything in a prediction file changes. They are
already recorded in each `metadata.json`; re-derive with
`shasum -a 256 <file>` if you rebuild.

## Validation status

All three entries: **PASS, 51 checks, 0 failures, 0 warnings.**

The benchmark's own validator is `make check`, which needs R. **This container has
no R and cannot install it**, so the gate that ran is
`silicon_sampling/submission/check.py` — a line-by-line port of the benchmark's
`scripts/check.R` and `lib/check_lib.R`, written to reproduce the FAIL/WARN split
exactly rather than to improve on it. Its schema is checked against the shipped
`codebook.csv` and `condition_codenames.csv` on every run, and today that
comparison reports no disagreements.

**You should still run the real `make check` once on a machine with R.** The port
is good evidence, not proof.

A second, different check also ran — `scripts/verify_submission.py`, which
executes every scored analysis over the built entry to confirm each produces a
number rather than an error. Verdict: **clean**, 208 ATE pairs, 4,160 subgroup
pairs, one metric undefined for a reason that is an artefact of using a second
entry as a stand-in reference (`beta_adj` needs the *human* standard errors).

---

## What you still have to do

### 1 · Fill five items only a person can answer

In `registration.md` (identical in all three entries — edit once, copy across),
each marked `TODO-BEFORE-DEPOSIT` inline:

- **0.1 Team ★** — member name(s), affiliation, corresponding contact. Currently
  says MPIB / Center for Humans and Machines with names missing.
- **I.1 Competing interests ★** — funding source. I have stated the factual part
  (no in-kind compute, no API, open weights, institutional hardware); the funding
  line is yours.
- **I.3 Blinding attestation ★** — **mandatory, and must be signed by a named
  person.** Draft text is in place; it needs a name and a date.
- **K.1** — `code_doi`, if you want to mint one by releasing the code repo to
  Zenodo. Optional; the repository link is already there.
- **K.3** — a monetary cost figure, if you want one. All compute was local, so
  there is no bill; I have reported call counts, draw counts and GPU-hours.

### 2 · Fill three fields in each `metadata.json`

`team_name`, `contact`, and `creators[0].name` still read `TODO-BEFORE-DEPOSIT`.
Add an `orcid` if you have one — **and only if it is real**: an ORCID that fails
its checksum makes Zenodo reject the deposit with an opaque HTTP 500. The
generator omits an invalid one and says so rather than writing it, and the
all-zero dummy is correctly rejected.

After editing, regenerate `.zenodo.json` so the deposit record matches:

```
python -c "from silicon_sampling.submission import zenodo as Z; \
           Z.write_zenodo('data/pfander/submission/primary')"
```

(or `make zenodo_citation` if you are on a machine with R).

### 3 · Decide on the raw generation logs

`registration.md` K.2 asks for complete unprocessed model responses. Each entry
ships the raw export of the run its rows came from (14 MB). The **full** logs for
all eleven runs — `answers.jsonl` and `draws.jsonl`, including rejected draws —
are about **2.4 GB** and are not in the deposit. Either mint a separate Zenodo
upload and link it, or offer them on request. K.2 flags the choice.

### 4 · Turn each entry into its own repository and release it

The benchmark's rule is **one entry = one repository = one Zenodo deposit**. Our
three entries are three directories, not three repos. For each:

1. `git clone` the template at `/opt/silicon-sample-submission` (or use the
   GitHub "Use this template" button), re-init, and copy the entry's contents in.
2. Delete the shipped `example_*` files from `predictions/` and
   `raw_data_deposit/` — the build does not put them there, but a fresh clone
   does.
3. Connect the repo to Zenodo, publish a **GitHub release**, and take the DOI.
4. Email the three DOIs and the three fingerprints **together** to
   `janlukas.pfaender@gmail.com`.

**Deadline: 31 August 2026.** That is three days from this build.

---

## Two things I could not fix, which you should know before depositing

### `gender = "Other"` has zero respondents — a conditional risk, not a defect

The submission declares three gender levels and contains two. **This matches the
published quota exactly**, and an earlier draft of this section was wrong to call
it a gap in our sample.

The benchmark's quota table (preregistration, Table 2, N = 18,000) has only
*Male* and *Female* columns, and the arithmetic leaves no room for a third
category: Male + Female equals the cell total in **every one of the nine cells**,
and the cells sum to exactly 18,000 on both the age and the race margin. Our draw
reproduces it to the person — Male 8,827 against a quota of 8,827, Female 9,173
against 9,173, race *Other* 492 against 492. Recruiting to that quota produces a
sample with no gender-*Other* respondents at all.

Against that sit four places where the benchmark carries the level anyway: the
preregistration's remark that such participants "are not quota-constrained"
(which reads as *may participate without a quota limit* rather than *are
excluded*), the submission spec's required levels
`c("Male", "Female", "Other")`, the codebook's "Exact submission levels: Male |
Female | Other", and — most tellingly — the organizers' own placeholder data
generator, which simulates gender from three categories.

**So the question is whether the realised human sample contains any, and that is
locked data.** The consequence if it does is worse than a few lost cells:
`build_subgroup_pairs` ends with

```r
stopifnot(nrow(joined) == nrow(human_mod_side))
```

so a human-side interaction row with no counterpart in our refit **stops
Section 2 for this submission** rather than silently shrinking it. Three branches:

- humans have **none** → both sides carry two levels, the join matches, nothing
  happens. The quota arithmetic makes this the most likely branch.
- humans have **a handful** → most condition × *Other* cells are empty, those
  coefficients are aliased and dropped on the human side too, and the join
  probably still matches.
- humans have **enough to identify the interactions** (~1% of a US panel would be
  ~180 people, ~10 per condition) → the human side carries the rows, ours does
  not, and the assertion fails.

**Insurance, if you want it.** Sampling ~200 additional respondents with
gender = *Other* — about 1% of the sample, ~11 per condition — into the template
run and the structural donor run would remove the third branch entirely. At the
observed throughput that is well under an hour of GPU time against the ~78
GPU-hours already spent. It would put our sample slightly *off* the published
quota, which is the trade: exact quota conformance against immunity to a join
that may or may not fail.

I would not treat this as urgent. It is worth knowing before depositing, and it
is worth one email to the organizers asking whether the human data contain
gender-*Other* respondents — a question they can answer without unblinding any
outcome.

### The structural half of the recipe is validated on three studies, not four

Levels, demographic offsets and dispersion all come from DeepSeek-V4-Flash, which
has no run on one of the four validation studies. The cross-validation therefore
cannot consider it, and every distributional and demographic result in the
validation describes a donor the submission does not use. This is stated plainly
in `registration.md` J.1 rather than glossed.

---

## What changed in this build versus the previously built entries

The predictions are **not** the ones that were on disk before. Five changes, all
from the [audit](audit_findings.md) and the [recipe report](the_recipe_now.md):

1. **Muse-Glimmer-30B joins the effect average** (three models, eight runs,
   model-balanced) — expected +0.03 on the leaderboard's sort key.
2. **`flatten_noise` off** — its two target outcomes had been selected by a
   diagnostic run with a placeholder standard error.
3. **The `policy_general` anchor** moves from CCC's three-item composite to the
   single item Pfänder reuses: level 68.01 → 65.88, dispersion 29.32 → 32.89,
   party gap 32.9 → 37.34.
4. **Party-gap blend weight 0.5 → 0.7**, on an out-of-sample measurement.
5. **Global shrinkage 0.4127 → 0.3832**, having been rescaled against a Pfänder
   reliability where the constant was fitted on the reference studies.

Two further corrections were made to the build itself while preparing this:

- The raw deposit was `qwen25_7b`'s export while the submitted rows came from
  `qwen25_7b_demo` — a transparency record of a different run than the one on
  trial. It now follows the recipe's template run and byte-matches it.
- `metadata.json` preserved every field across rebuilds, including `models` and
  `approach_family`, which are derived from the recipe. A three-model hybrid was
  describing itself as `single model` with the wrong model list. Those two fields
  are now refreshed on every build; a hand-written `abstract` is still preserved.

## Reproducing this build

```
python scripts/build_entries.py          # all three entries, checked as it goes
python scripts/verify_submission.py      # every scored analysis over the entry
python -m pytest tests/                  # includes the format-gate port's own tests
```

The build is deterministic: the same runs and the same constants give the same
files and the same fingerprints.
