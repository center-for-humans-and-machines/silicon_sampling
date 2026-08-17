# [ACTIVE] Resample both studies with DeepSeek-V4-Flash-Base on DAIS

Task: [docs/tasks/v4_flash_on_DAIS.md](../tasks/v4_flash_on_DAIS.md)

Rerun the Pfänder and Voelkel silicon samples with `deepseek-ai/DeepSeek-V4-Flash-Base`
on DAIS, pull the completions back, and redo both analyses so they compare
V4-Flash against Qwen2.5-7B and — for Voelkel — against real humans. The question
being answered is **does a much bigger base model make silicon sampling more
faithful**, so every deliverable is a paired comparison, not a fresh set of
numbers.

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
2. **KV per session collapses by 10×.** Only 2 of 43 layers keep a full-length
   MLA cache; the rest are sliding-window compressor states of 8 or 128
   positions (`CompressorStateCache.sliding_window = coff * compress_ratio`).
   That is **2,304 B/token + 5.9 MB fixed**, so a 7,500-token transcript costs
   **23 MB** against Qwen2.5-7B's 229 MB. The KV cache stops being the binding
   constraint on group size, which is what the whole driver was built around.
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
weight-bound/compute-bound crossover, swept over achieved HBM bandwidth:

| config | group | ms/step | Pfänder @25% bw | @40% | @60% | Voelkel @40% |
| --- | --- | --- | --- | --- | --- | --- |
| 2×H200 | — | — | weights do not fit in 282 GB | | | |
| 4×H200 | 512 | 38.4 | 0.39 h | 0.24 h | 0.16 h | 0.06 h |
| 8×H200 | 512 | 19.2 | 0.19 h | 0.12 h | 0.08 h | 0.03 h |
| 2×B200 | 1024 | 46.0 | 0.23 h | 0.14 h | 0.11 h | 0.04 h |
| 4×B200 | 1024 | 23.0 | 0.12 h | 0.07 h | 0.06 h | 0.02 h |
| 8×B200 | 1024 | 11.5 | 0.06 h | 0.04 h | 0.03 h | 0.01 h |

Both numbers are small enough that **the run is not the expensive part of this
task** — even at 5× my estimate, Pfänder on 8×H200 is a one-hour job inside the
24 h DAIS cap. That flips the plan's priorities: the engineering risk is in the
driver and the analysis, not in GPU budget.

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

### 4. Pilot on DAIS (1 node, 8×H200)

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

### 5. Full runs, then pull down

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
