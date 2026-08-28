# [ACTIVE] Expand by one study and one model

Task: from the user, 2026-08-26. Hard deadline **2026-08-31**.

Two expansions, plus the evaluation and calibration work they unlock.

* **Study space +1**: Voelkel et al. 2026, the Climate Change Challenge (CCC) —
  13 arms, 13,821 respondents, materials in `data/calibration/SI/Voelkel/`.
* **Model space +1**: `meta-models/Muse-Glimmer-30B`, 1–2 GPUs on DAIS.

Two constraints given with the task:

* **Muse-Glimmer may not work with the DAIS vLLM container.** Check early. If it
  does not, ping the user on Slack and carry on with the study expansion.
* **DeepSeek-V4-Flash on CCC runs last**, because it is by far the slowest. The
  next cross-validation and report target the state with *everything except* that
  run. A second cross-validation follows when it lands.

## Todo

### A. Muse-Glimmer feasibility — **BLOCKED**, user pinged 2026-08-26 19:09
- [x] A1. Checkpoint is on DAIS and complete:
      `/opt/huggingface/hub/models--meta-models--Muse-Glimmer-30B`, two safetensors
      shards, bf16, `HF_HOME=/opt/huggingface`
- [x] A2. **The container cannot load it.** Four findings, each independently fatal:
      * vLLM 0.23.0 does not register `MuseGlimmerForConditionalGeneration` — 365
        architectures, zero matches on "muse" or "glimmer"
      * transformers 5.12.1 does not know `muse_glimmer`; `AutoConfig.from_pretrained`
        raises "Transformers does not recognize this architecture"
      * `config.json` has no `auto_map` and the snapshot ships no modelling files, so
        there is no `trust_remote_code` path
      * vLLM's Transformers backend (`TransformersMultiModalForCausalLM`) exists but
        routes through transformers' AutoConfig, which is where the previous point fails
      It is a genuinely new multimodal architecture — `image_token_id`, a projector,
      `processor_config.json`, `text_config.model_type = muse_glimmer_text` — not a
      renamed Qwen or Llama.
- [x] A3. Throughput measured: **2,887 respondents/hour on 2 GPUs**, 18,000
      Pfänder respondents in 6.2 h. The user supplied
      `vllm-openai-muse-glimmer.sif` (vLLM 0.26.1rc1, transformers 5.15.0), which
      resolves A2. It ships no pandas, hence `silicon_sampling.lazy`.
- [x] A4. Slacked the user with the diagnosis and three unblock routes: a newer
      vLLM/transformers in the container, a checkpoint variant shipping remote code,
      or a different image. Not attempting an in-job `pip install -U` unilaterally,
      because it risks the four working samplers.
- [x] A5. Registered `muse_glimmer_30b`, plus `_seed2`/`_seed3` replicates

**Consequence for the plan.** D4 and the model-space half of E2 are parked. The
study expansion is unaffected and is now the critical path.

### B. CCC study package
- [x] B1. `silicon_sampling/ccc/`: paths, outcomes, instrument, convert, templates,
      profiles, run, export, score, validate, cli — mirroring the four that exist
- [x] B2. Reuse `voelkel.qsf`, which already parses this QSF unmodified and
      recovers all 13 arms with names matching the data's `Condition` column
- [x] B3. Handle the structural difference: CCC measures every primary outcome
      **pre and post**, where the SDC measured only post
- [x] B4. Modality audit — `System Preservation Framing` carries 12 images; every
      other arm has 0–2 against 1.5k–4.7k characters of prose, and no arm has
      video or audio
- [x] B5a. **Estimand decided.** The published script fits
      `Post ~ 1 + ConditionR + Pre` — ANCOVA with the pre-measure as a covariate,
      HC3 robust SEs, `Control` as the reference. Pfänder's benchmark instead refits
      `outcome ~ condition` with HC2 and **no covariate**.
      For the cross-validation to predict Pfänder, `score.effects()` uses the
      **simple ATE**, matching how Pfänder is actually scored. The ANCOVA is
      computed alongside it, because the gap between them measures how much power
      the pre-measure adds — and therefore how much of Pfänder's achievable
      `pearson_r` is capped by noise in its own human effect estimates rather than
      by our model. That is a prediction-relevant number nothing else supplies.
- [x] B5b. `score.load_humans()` over `CCC - Data - Recoded.csv`, with the
      `ConditionR` pooling of the three placebo controls into one `Control`
- [x] B6. Render all 13 arm templates

### C. Independent verification of the templates — the step the audit exists for
- [x] C1. Fan out independent verifiers (one per concern, not one per author) over
      the rendered templates against the questionnaire PDF and the QSF
- [x] C2. Specifically confirm **every slider states its numeric range**. This is
      the defect that cost the whole first round on ICPC and Goldwert: endpoint
      labels printed without the 0–100 range, models answering on an implicit
      0–10 scale, mean control-arm level error 30–39 pp
- [x] C3. Confirm both pre and post batteries are present, in the right places
- [x] C4. Confirm each arm's stimulus text is complete and free of placeholders
- [x] C5. Confirm response options, item order randomisation and display logic
- [x] C6. Confirm the demographic quota matches the study's own sample
- [x] C7. Fix everything found, then re-verify

### D. Sampling
- [x] D1. Smoke run: Qwen2.5-7B, all 13 arms, small n — **check the invalid /
      rejection rate immediately and raise the alarm early if it is high**
- [x] D2. Qwen2.5-7B on CCC, 13,821 respondents, local 4090
- [x] D3. Qwen2.5-72B on CCC, DAIS
- [x] D4. Muse-Glimmer-30B on **all five** studies — Pfänder, Voelkel-SDC, ICPC,
      Goldwert, CCC — because a model is only usable in the cross-validation if it
      has run everywhere. Roughly 51,800 respondents; combine into one DAIS job per
      the cluster-queue rules, sized with ~50% headroom.
      **Submit every job explicitly.** No automated resubmission: the auto-resume
      loop is deleted after it created 44 two-GPU jobs against a run it wrongly
      read as unstarted. Read the previous job's log, then submit one job.
- [~] D5. **Last**: DeepSeek-V4-Flash on CCC — **running**, DAIS 425064,
      4xH200, 14 h limit, submitted by hand 2026-08-28

### E. Evaluation at the pre-V4 state
- [x] E1. Score the currently locked-in recipe on CCC as a held-out study
- [x] E2. Nested leave-one-study-out CV over four studies, same pre-committed
      lexicographic rule, now including Muse-Glimmer in the membership candidates
- [x] E3. Report: what the fourth fold and the fourth model do to the conclusions,
      including the honest possibility that they make the recipe look worse
- [ ] E4. Revise the Pfänder prediction

### F. Evaluation after V4-Flash on CCC
- [ ] F1. Re-run the nested CV with everything
- [ ] F2. Update the reports and, if the constants move, rebuild the submission

### G. New calibration routes from the item overlap
CCC's concern battery is **verbatim identical** to Pfänder's, its general-policy
item is verbatim identical, and three of six behavioural-intention items are.
Things that become possible, to be tested rather than assumed:

- [x] G1. Level and dispersion anchors for six climate outcomes that currently
      have none — the submission anchors three, all from the trust battery
- [x] G2. Replace the party-gap anchors with directly measured ones. The shipped
      values are wrong by up to 2.4x on `behavior_mean` (10.0 against 24.0)
- [x] G3. **NEGATIVE RESULT — does not transfer.** Quantile mapping on the identical items. With the same question on
      the same scale in the same population, the whole human response distribution
      is known, not just its mean and sd — so our answers can be mapped onto it.
      This is the first calibration available that could move OVL, KS and W1
      directly rather than through a variance ratio
- [ ] G4. Effect-magnitude prior: mean absolute human effect is 1.70 pp across
      CCC's ten framings, the closest analogue to Pfänder, which should narrow the
      `rmse`, `alpha` and `beta` predictions
- [ ] G5. **Arm-level effect anchor.** Pfänder has a `Consensus` condition and CCC
      has `Consensus Framing 1` and `2`. Check whether any Pfänder arm matches a
      CCC arm closely enough to anchor a single condition effect directly
- [ ] G6. Pre-post structure: CCC measures outcomes before and after. Check what
      the human pre-post relationship says about plausible effect sizes, and
      whether Pfänder's design supports the same

## Sequencing

A first, because it can block and the answer changes what to build. B and C next
and they are the critical path — the fidelity audit is the standing reason not to
rush them. D2 and D3 can start as soon as C passes; D4 runs on DAIS in parallel;
D5 is deliberately last. E once D1–D4 are in, F when D5 lands, G alongside E.

## Status

Started 2026-08-26. Updated 2026-08-28.

**Done:** the CCC study package and its verification (B, C); sampling for
Qwen2.5-7B, Qwen2.5-72B and Muse-Glimmer on all studies (D1–D4); the four-study
nested cross-validation and the CCC structural hold-out (E1–E3); the level and
party anchors from CCC (G1, G2); and three calibration routes tested and rejected
(G3, plus level expansion and a Goldwert donation anchor). Written up in
[docs/reports/four_study_cross_validation.md](../reports/four_study_cross_validation.md).

**Found along the way, not in the plan:** `qwen25_72b_seed3` was sampled against a
copy of `qwen25_72b_seed2`'s profiles and is the same draw, not a replicate — 77.1%
identical cells. It is out of `BEST_RANKERS` and a draw-level guard test now
catches the class.

**Running:** DAIS 425064 (V4-Flash on CCC, the last sampling job) and DAIS 425081
(two Muse-Glimmer replicate seeds on Pfänder, needed before Muse can join the
ensemble at all).

**Remaining:** E4 revise the prediction; F1/F2 re-run the CV after V4 lands and
rebuild if constants move; the Muse membership decision once its seeds are in;
G5 and G6.
