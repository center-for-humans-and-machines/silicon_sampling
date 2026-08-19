# [DONE] Resample both studies with DeepSeek-V4-Flash-Base on DAIS

Task: [docs/tasks/v4_flash_on_DAIS.md](../tasks/v4_flash_on_DAIS.md)

Rerun the Pfänder and Voelkel silicon samples with `deepseek-ai/DeepSeek-V4-Flash-Base`
on DAIS, pull the completions back, and redo both analyses so they compare
V4-Flash against Qwen2.5-7B and — for Voelkel — against real humans. The question
being answered is **does a much bigger base model make silicon sampling more
faithful**, so every deliverable is a paired comparison, not a fresh set of
numbers.

## Outcome

**Scaling the base model ~40x bought a much more realistic sample and no better
prediction of which interventions work.**

| | measured |
| --- | --- |
| Voelkel level error | 22.9 -> **8.0** points on a 0-100 scale |
| Voelkel effect over-spread | 2.6x -> **1.7x** the true spread |
| Voelkel rank correlation with real effects | 0.31 -> **0.19** (human replication 0.40) |
| Voelkel RMSE | 3.62 -> **2.81**, the only near-settled delta (p 0.94) |
| Pfänder partisan gap, climate belief | -1.1 -> **-12.4** pp |
| Pfänder moderator R² beyond condition | 0.0020 -> **0.0373** |
| the two models' agreement on which messages work | r = **0.17** on the primary outcome |

Reports: [Voelkel](../reports/voelkel_validation/05_model_comparison.md) (the one
with ground truth) and [Pfänder](../reports/pfander_silicon_sample/05_model_comparison.md).

No effect-recovery delta clears its interval, so the honest reading is "no
improvement, possibly a regression" rather than a demonstrated regression.
Everything that *did* improve — level accuracy, exaggeration, demographic
responsiveness — is a property of how a respondent answers in isolation. The thing
that did not is the one the megastudy scores.

## What the checkpoint actually is

Verified present on DAIS at `/opt/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Base`
(46 shards, `total_size` 294,673,469,000 B = **294.7 GB**). `config.json` says:

| | |
| --- | --- |
| architecture | `DeepseekV4ForCausalLM`, 43 layers, hidden 4096 |
| MoE | 256 routed experts, 6 active + 1 shared, `moe_intermediate_size` 2048, experts **fp8** (e4m3, 128×128 block scales) |
| total / active params | ~290 B / **13.9 B per token** |
| attention | MLA, 64 heads, `head_dim` 512, `num_key_value_heads` 1, q/o LoRA rank 1024 |
| sparse attention | DSA indexer, `index_topk` 512, 64 index heads of dim 128 |
| KV compression | `compress_ratios` per layer: 2 layers full, 21 at ratio 4, 20 at ratio 128 (+1 for the MTP layer) |
| context | 1,048,576 (YaRN ×16 over 65,536) |
| MTP | `num_nextn_predict_layers: 1` — usable for speculative decoding |

vLLM **0.23.0** in the default `glm_52_vllm.sif` registers both
`DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel`, with the implementation in
`vllm/models/deepseek_v4/` (torch 2.11.0+cu130). So no engine surgery is needed —
but three things about this model change the run shape versus Qwen2.5-7B:

1. **294.7 GB of weights need ≥ 4 H200 or ≥ 2 B200.** Two H200s (282 GB) cannot
   hold it.
2. ~~**KV per session collapses by 10×.**~~ **This was wrong — see §4b.** The
   reasoning was that only 2 of 43 layers keep a full-length MLA cache and the
   rest are sliding-window compressor states, giving 2,304 B/token + 5.9 MB fixed
   and so ~23 MB for a 7,500-token transcript. The engine reports otherwise:
   4×H200 holds **27-50 transcripts**, not the ~500 that implies. The hybrid
   compressor/MLA cache is far more expensive per session than the layer ratios
   suggest, and **KV capacity remains the binding constraint on group size** —
   exactly as it was for Qwen2.5-7B.
3. **Tensor parallelism is mandatory**, and `configure_runtime()` currently
   forces `VLLM_ENABLE_V1_MULTIPROCESSING=0`, which is incompatible with TP > 1.

## How long the completions will take

Full derivation in the calculation section at the end. The short version, and the
reason it is not the answer anyone expects:

**The GPU is not the bottleneck. The driver's host-side work is.** Backing the
4090 run apart — 515 ms per round, 15 prompts, ~8.3 decode steps — leaves roughly
**12 ms per prompt per round** that is not decode: the driver hands vLLM the
*entire transcript as a string* every round, so ~7,000 tokens get re-tokenised,
re-hashed into prefix-cache blocks and rebuilt into a request ~78 times per
respondent. That term is CPU-bound, serial with the GPU, and **identical on every
GPU configuration**:

| study | host-side term | GPU term, 8×H200 | GPU term, 8×B200 |
| --- | --- | --- | --- |
| Pfänder (18,000) | 2.6 – 6.5 h | 0.08 – 0.19 h | 0.03 – 0.06 h |
| Voelkel (6,203) | 0.7 – 1.6 h | 0.02 – 0.05 h | 0.01 h |

So with the driver untouched, Pfänder takes **~5 h whether you use 4 H200s or 8
B200s**, and 97% of that is a CPU re-tokenising text the GPU already has cached.
Fixing it (step 1 below) is what makes the hardware matter.

GPU term alone, once the host term is out of the way — group size fixed at the
weight-bound/compute-bound crossover, swept over achieved HBM bandwidth.

> **Superseded: every row below assumes a group size of 512-1024, and the real
> capacity is 27-50.** Multiply the Pfänder columns by roughly 12-19x, or read the
> measured figures in §4b instead. The table is kept because the *shape* of the
> reasoning held up — the run is weight-streaming-bound, step time is flat in
> batch size, and ~40% of peak bandwidth was the right guess — but the one input
> that was never engine-verified is the one that broke it.

| config | group | ms/step | Pfänder @25% bw | @40% | @60% | Voelkel @40% |
| --- | --- | --- | --- | --- | --- | --- |
| 2×H200 | — | — | weights do not fit in 282 GB | | | |
| 4×H200 | 512 | 38.4 | 0.39 h | 0.24 h | 0.16 h | 0.06 h |
| 8×H200 | 512 | 19.2 | 0.19 h | 0.12 h | 0.08 h | 0.03 h |
| 2×B200 | 1024 | 46.0 | 0.23 h | 0.14 h | 0.11 h | 0.04 h |
| 4×B200 | 1024 | 23.0 | 0.12 h | 0.07 h | 0.06 h | 0.02 h |
| 8×B200 | 1024 | 11.5 | 0.06 h | 0.04 h | 0.03 h | 0.01 h |

That conclusion — "the run is not the expensive part of this task" — did not
survive. With the real group size, Pfänder is 8-16 h of 4-GPU time spread over
several resumed jobs, and queue waits add hours on top. The engineering risk was
still mostly in the driver and the analysis, but GPU time was not free.

## Plan

### 0. Pin the one coefficient the estimate rests on (local, 4090, ~20 min)

The host-side term above is a 2.5× band because it was backed out of a total
rather than measured. Measure it directly with the existing Qwen2.5-7B setup: time
`engine.generate()` on 15 warm-prefix prompts of ~7,500 tokens, split into
(a) submit-to-first-token and (b) decode, and repeat at group sizes 15 / 60 / 240
to confirm the term scales linearly with `group_size × prompt_tokens`.

This decides whether step 1 is essential or merely nice, and it costs no cluster
time. Record it in the report; it is a genuinely interesting result about the
existing pipeline regardless of V4-Flash.

### 1. Stop re-sending transcripts as strings

Carry token ids alongside the text in `survey/session.py`: keep the transcript
string as the authoritative artefact written to `raw/*.txt`, and maintain an
incrementally-extended `list[int]` for the prompt, passed to vLLM as
`TokensPrompt(prompt_token_ids=...)`.

**Fidelity gate, not optional.** Incremental tokenisation is not guaranteed to
equal tokenising the whole string, and a boundary difference changes the model's
conditioning — which is exactly the thing this project measures. So:

- Add a test that, for every condition in both studies, walking a session to
  completion and accumulating ids yields **exactly** `tokenizer(transcript)`.
- Every answer in these instruments is followed by `\n`, which should make the
  append boundary clean, but if the assertion fails for any slot, fall back to
  re-tokenising only from the last verified block boundary rather than accepting
  a silent difference.

If step 0 shows the host term is small, keep this change anyway but drop it to the
end — it is what allows group sizes in the hundreds, which is where V4-Flash's
tiny KV footprint pays off.

### 2. Engine: tensor parallelism and expert parallelism

- `EngineConfig`: add `tensor_parallel_size: int = 1` and
  `enable_expert_parallel: bool = False`; pass both through `VLLMEngine.__enter__`.
- `configure_runtime()`: only set `VLLM_ENABLE_V1_MULTIPROCESSING=0` when
  `in_process` is true, and have both CLIs pass `in_process=(tp == 1)`. The
  docstring's reasoning (tens of thousands of tiny calls, errors in the caller's
  traceback) still holds for the single-GPU local case, so keep it there.
- Prefer **expert parallelism** for the MoE at TP=8: each rank then holds 32 whole
  experts and streams 37 GB rather than a narrow slice of all 256. Make it a flag
  and settle it empirically in the pilot.
- Make the cache root overridable (`SILICON_SAMPLING_CACHE`), because
  `paths._cache_root()` writes to `~/.cache` and container homes on DAIS may not
  be writable — the bind list offers `/tmp`, `/opt/outputs`, `/opt/runs`.
- Raise the `max_num_seqs` ceiling: group 512 with 4 draws needs ≥ 2,048.

### 3. Multi-model run layout

Both studies hardcode `SAMPLES = data/<study>/silicon_sampling/qwen25_7b`. Add a
small model registry (`qwen25_7b` → `Qwen/Qwen2.5-7B`, `v4_flash` →
`deepseek-ai/DeepSeek-V4-Flash-Base`) and a `run_dir(key)` helper; add
`--run qwen25_7b|v4_flash` to both CLIs, defaulting to the existing path so
nothing already on disk moves.

**Reuse the existing `profiles.csv` verbatim** — copy it into the new run dir
rather than rebuilding. Profiles are model-independent, and identical respondents
with identical seeds make the two samples *paired*, which the comparison in step 6
depends on.

Note that `fit_token_budgets()` and `max_transcript_tokens()` re-measure with the
run's own tokenizer, so per-slot budgets and worst-case lengths recompute for
V4-Flash automatically. They need the tokenizer, so they run on the cluster.

### 3a. Two DAIS submission facts, learned the hard way

Both cost a failed job, so they are written down here.

- **`program_call` must start with `-c pass;`.** The slurm template hardcodes
  `export CMD="python $program_call"`, and *neither* allowed container image ships
  a `python` binary — only `python3`. So the real command goes after a
  semicolon: `-c pass; python3 -m silicon_sampling...`, which lets the broken
  `python -c pass` fail harmlessly first. This is already the convention in this
  project's earlier cluster runs (`glm52_annotation_*`), so it is the house style
  rather than an invention.
- **`mem` must be explicit and within 250 GB / 12 cores *per GPU*.** The skill's
  documented `--mem 0` default is rejected outright for shared jobs, and a 4-GPU
  job is capped at 48 cores. A 4×H200 job wants `n_cpu: 48, mem: "800G"`.

Also worth knowing: a pending job reporting *"Nodes required for job are DOWN,
DRAINED or reserved for jobs in higher priority partitions"* is usually just
contention — the same request started a few minutes later. Both `h200` and `b200`
nodes accept jobs.

### 4. Pilot on DAIS (1 node, 4×H200)

`sync_to_cluster`, then a short interactive or batch job sampling ~64 Pfänder
respondents stratified across arms, at group size 64. Confirm and record:

- the model loads under vLLM 0.23 at TP=8 (and whether EP beats TP);
- `kv_cache_tokens()` and measured ms/step against the 19.2 ms prediction;
- respondents/hour, rejection rate, structured-fallback rate;
- that transcripts and `answers.jsonl` look like the Qwen ones (no format drift
  from a different tokenizer or a base model that ignores the layout).

**Decision gate:** if measured throughput is worse than ~10× the estimate, stop
and report rather than burning 24 h of an 8-GPU node — that would mean the model
is being served in a way the calculation does not describe, and the fix belongs
upstream of a full run.

### 4a. Wall-time is a scheduling cost, so ask for the estimate plus headroom

A 24 h request queues behind everything shorter, and on DAIS the queue wait
dominates. So the limit is set to the estimate plus ~60-100%, and being killed at
the limit is treated as normal rather than as a failure: the sampler resumes from
`answers.jsonl`, losing at most the group in flight (~100 respondents, a couple of
minutes), plus one model load.

**Resuming is resubmitting the identical job.** Nothing needs editing — the
wrapper reads how many respondents are already done and asks only for the rest.
The two submissions, verbatim, are in
[`scripts/dais_submit_v4_flash.md`](../../scripts/dais_submit_v4_flash.md).

### 4b. What the measurements said, against the estimate

| | plan estimate | measured |
| --- | --- | --- |
| KV capacity, 4xH200 | group 512 | **group 27-50** (203k-228k tokens) |
| Voelkel, 6,203 | 1.5-3 h | **~2.6 h** |
| Pfänder, 18,000 | 5-9 h | ~8-16 h (group ~31) |
| decode throughput | — | **2,400 respondents/h** at group 50 |

The group-size prediction was the plan's one real error, and it was worth ~19x.
Reading `compress_ratios` and concluding that only 2 of 43 layers keep a
full-length cache gave ~23 MB per session; the hybrid compressor/MLA cache
actually costs enough that 4xH200 holds 27-50 transcripts, not 512. Since
throughput is linear in group size, that single number carried the whole
headline. Everything downstream of an engine-reported KV figure was fine.

### 5. Full runs, then pull down — done

Both complete: Voelkel 6,203 (two jobs, the first killed at its 3 h wall at 88%),
Pfänder 18,000 (two jobs, the first killed at its 14 h wall at 97%). Final
throughput **1,022-1,245 respondents/h** on 4xH200, rejection 5.1-6.7%, zero forced
defaults in either study.

**What actually cost time, in order:** queue waits (up to 10 h for a whole-node
job, ~7 h for a 4-GPU one at its worst), then the two throughput traps
(oversubscribed group size, cold TileLang cache), then the sampling itself. The
plan budgeted for none of the first and all of the last.

### 5-original. Full runs, then pull down

Voelkel first — it is smaller and it is the only one with ground truth, so it
answers the task's question on its own if anything goes wrong with the other.
Then Pfänder. Both use the existing resumable wrapper pattern
(`scripts/run_*_sample.sh`), generalised to take a run key and to skip the
`nvidia-smi`/`pkill` GPU-reclaim loop, which is a single-GPU-local idiom.

Pull down `answers.jsonl`, `draws.jsonl`, `samples.csv`, `run_meta.json`,
`profiles.csv` via `/opt/cluster_transfer` (~60 MB Pfänder, ~15 MB Voelkel).
**Leave `raw/` on the cluster** — it is 443 MB for Pfänder at Qwen lengths and the
skill caps transfers at 1 GB; pull a stratified sample of ~200 transcripts for
inspection instead.

### 6. Redo the analyses as paired comparisons

The existing reports are written for one sample. The changes needed:

- `voelkel/score.py`: `leaderboard()` hardcodes the label
  `"Silicon sample (Qwen2.5-7B)"` and takes exactly one synthetic frame. Change it
  to take `{label: frame}` so the board carries both models, the human
  replication and both baselines.
- `voelkel/report.py`: `generate()` takes `samples_csv` but reads
  `SAMPLES / "run_meta.json"` from module scope — add a `run_dir` parameter.
- `pfander/report.py`: `generate(samples_csv, run_dir, out)` is already
  parameterised; no change needed to run it twice.
- **Add a paired cluster bootstrap** over the difference between models on the
  same conditions. "V4-Flash scores r = 0.51 and Qwen scored 0.41" is not an
  answer without an interval on the *difference*; resampling conditions
  independently for each model throws away the pairing that makes the comparison
  precise. This is new code in `benchmark/metrics.py`.

Metrics that carry the verdict, all V4-Flash against Qwen2.5-7B against the human
yardstick:

1. **Effect recovery** (Voelkel, ground truth): directional %, Pearson r and
   noise-adjusted r, RMSE, calibration β and β_adj — against Human 2's 66.7% /
   0.514 / 1.682 / 0.436, not against 1.0.
2. **Level error and distribution shape** (Voelkel): the mean absolute level error
   of 23 points on a 0–100 scale is Qwen's most damning number. Per-outcome level
   error, variance ratio, OVL, W1.
3. **Demographic conditioning** (both studies): the headline weakness — subgroup
   R² ≤ 0.002 over condition alone, and a 1.1-point synthetic partisan gap on
   climate belief where the real gap is tens of points. If a 290 B base model
   improves anything, this is where it must show. Re-run the same moderator
   variance decomposition and the Voelkel seen-vs-unseen moderator test
   (pooled r 0.26 seen vs 0.24 unseen).
4. **Survey-likeness** (both): modal-answer share, multiples-of-10 share,
   Cronbach's α, flat-profile share, rejection rate.

Deliverables: extend both report trees with a `05_model_comparison.md`, update
both READMEs' headline tables to carry both model rows, and add
`docs/reports/model_comparison/README.md` that answers the task's actual question
across the two studies in one place. Regenerate `tier1_submission.csv` from the
V4-Flash Pfänder sample so the competition submission can use whichever model
scores better.

### 7. Finish

`black .`, `flake8 . --max-line-length=200 --extend-ignore=E203,W503`, push to
main, move this plan to `docs/plans/archive/` and the task to
`docs/tasks/archive/`.

## Risks and open questions

- **The host-side term is the whole schedule.** If step 0 shows it is 16 ms
  rather than 6, Pfänder is a 6.5 h job on any hardware until step 1 lands.
  Sequencing step 0 first is deliberate.
- **Incremental tokenisation could change the conditioning.** Gated by an
  exact-equality test (step 1). If it fails and cannot be made to hold, the
  honest options are to keep string prompts and accept ~5 h, or to parallelise the
  host work across processes feeding one engine — not to accept a silent
  difference in what the model sees.
- **A base model at temperature 1.0 with an unfamiliar tokenizer** may follow the
  transcript layout better or worse than Qwen; the rejection rate and
  structured-fallback count in the pilot will say. Sampling stays untruncated
  (temperature 1.0, no top-p/top-k) — changing it would make the two samples
  incomparable and reshape the distributions being measured.
- **B200 availability on DAIS is unverified.** The command server exposes no
  `sinfo`, and `submit_slurm_job` defaults to `gpu_type: h200` with 1–2 nodes of
  1–8 GPUs, so the B200 rows above are hypothetical. Nothing in the plan depends
  on them; 8×H200 is the target.
- **A `HF_TOKEN` is exported in the DAIS container environment** and was printed
  into a command-server log during this investigation
  (`/var/cc_output/20260817T094142Z_*.log`). Worth rotating if that log is
  retained anywhere shared.
- **Open question for you:** the estimate says both runs together are well under
  an hour of 8-GPU time once the driver is fixed. If that holds, is it worth
  raising N above 18,000 / 6,203? Keeping them identical is what makes the
  comparison paired, so my default is to leave them alone and spend nothing on
  more respondents.

## The calculation

Reproduced by [`scripts/estimate_v4_flash_throughput.py`](../../scripts/estimate_v4_flash_throughput.py).
Inputs are measured, read off the checkpoint, or vendor spec; two parameters are
swept.

**Per decode step**, the cost is streaming weights, not arithmetic. At batch
`B` tokens each routing to 6 of 256 experts, the share of experts touched is
`1 − (1 − 6/256)^B`, which is 78% at B=64 and >99.7% at B=256 — so past a group
of ~64 every step reads essentially all 294.7 GB. Against that,
`2 × 13.9e9 × B` FLOPs of compute; the step is `max(memory, compute)`, with a
4 ms floor for ~650 kernel launches. Achieved HBM bandwidth is swept at 25/40/60%
of peak (4.8 TB/s per H200, 8.0 per B200); compute at 35% of fp8 dense peak.
Group size is fixed per configuration at the crossover where compute starts to
bind, because beyond it more concurrency buys nothing.

**Per respondent**, the loop is sequential by construction: 78.1 prompt-rounds
for Pfänder and 57.3 for Voelkel (measured, `run_meta.json` `calls` ÷
respondents), at ~7.5 decode steps per round (Pfänder's 98 slot budgets sum to
737 tokens), ×1.10 for rejection re-issues and structured fallbacks — so ~644 and
~473 sequential steps respectively. Throughput is then
`group / (steps × ms_per_step)`, and the group is affordable because KV is 23 MB
per session: 512 sessions cost 12 GB of the ~800 GB left over on 8×H200.

**Sanity check against the 4090.** Qwen2.5-7B, 15.2 GB of weights, group 15:
measured 1,343.5 respondents/h = 40.2 s per chain over ~644 steps = 62 ms/step,
against a ~15 ms bandwidth floor and ~40 ms of realistic vLLM step time at batch
60. The residual ~183 ms per round over 15 prompts is the 12 ms/prompt host term,
and 18,000 × 78.1 × 12 ms = 4.8 h of the measured 13.4 h run. The other 8.6 h is
decode. Both halves are accounted for, which is why the host term is trusted
enough to headline — and why step 0 measures it directly rather than trusting it.
