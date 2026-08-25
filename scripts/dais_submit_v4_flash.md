# Submitting (and resuming) the DeepSeek-V4-Flash runs on DAIS

Both runs go through `scripts/run_cluster_sample.sh`, which re-enters the sampler
until the target count is reached and then builds the analysis CSVs on the
cluster, so the pull-down is tens of MB rather than the whole transcript tree.

**Resuming is resubmitting the identical job.** `answers.jsonl` is the source of
truth for what is finished; the wrapper asks only for the rest. That makes an
unexpected death cheap — at most the group in flight plus one model load.

**It is not a licence to set short wall-time limits, and an earlier version of
this file said otherwise.** It argued that because resuming is cheap and the
queue wait dominates, the limit should sit near the estimate so the job is
scheduled ahead of longer ones. That inverts
[the cluster-queue guidance](../.claude/skills/handling_cluster_queues/SKILL.md):
the **primary** driver of DAIS queue time is the number of GPUs requested, and
the time limit is only secondary. Setting a 3 h limit on a 16 h job does not buy
a cheaper queue slot — it splits one job into six, each paying its own full queue
wait and its own model load, which is exactly the "don't split tasks up into
multiple cluster jobs" anti-pattern.

Measured, when this was live: three runs configured that way sat `PD (Priority)`
for over an hour together, and DeepSeek-V4-Flash reached 3,810 of 18,000
respondents in about ten hours of wall clock.

So: **set the limit at the estimated runtime plus roughly 50%**, and combine work
that shares a model into one job rather than submitting it twice. The two
Qwen2.5-72B runs go in a single job for that reason — one queue wait instead of
two. `scripts/dais_auto_resume.sh` stays useful as a safety net for a job that
dies unexpectedly, but it must not be the mechanism that a run is *expected* to
need.

Post these to the CC command server from the agent container.

## Voelkel — 6,203 respondents, estimated 1.5-3 h on 4 GPUs

```json
{"cluster":"dais",
 "program_call":"-c pass; bash scripts/run_cluster_sample.sh voelkel v4_flash 6203 --tensor-parallel-size 4 --expert-parallel --group-size 0 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.94",
 "time":"03:00:00","n_gpu":4,"n_nodes":1,"n_cpu":48,"mem":"800G","launcher":"python",
 "run_group":"v4voelkel","project":"silicon_sampling","gpu_type":"h200"}
```

## Pfänder — 18,000 respondents, estimated 5-9 h on 4 GPUs, so expect 2-4 resumes

```json
{"cluster":"dais",
 "program_call":"-c pass; bash scripts/run_cluster_sample.sh pfander v4_flash 18000 --tensor-parallel-size 4 --expert-parallel --group-size 0 --max-num-seqs 512 --gpu-memory-utilization 0.94",
 "time":"03:00:00","n_gpu":4,"n_nodes":1,"n_cpu":48,"mem":"800G","launcher":"python",
 "run_group":"v4pfander","project":"silicon_sampling","gpu_type":"h200"}
```

## Why each flag is what it is

| flag | reason |
| --- | --- |
| `-c pass;` prefix | the slurm template hardcodes `python $program_call`, and neither container image ships a `python` binary — only `python3`. This burns off the broken prefix. Same convention as this project's earlier `glm52_annotation_*` runs. |
| `--tensor-parallel-size 4` | 294.7 GB of weights need ≥ 4 H200. **Do not raise this to 8 without a reason to wait.** A 4-GPU job lands in the `gpu1` partition and starts in 10-25 minutes; a whole-node 8-GPU job lands in `gpu` and sat on `(Priority)` for over ten hours. 8 GPUs would be ~3.6x the concurrency and half the step time — roughly 2.5 h instead of 5 h of *runtime* — but that is not worth ten hours of *queue*. Short 4-GPU jobs plus resume beat one wide job here. |
| `--expert-parallel` | each rank then holds 32 whole experts and streams them, rather than a narrow slice of all 256. |
| `--group-size 0` | size the group to the KV cache the engine actually gets. Forcing a larger number oversubscribes the cache and every session pays a full re-prefill on its next step — a run at group 64 against a capacity of 27 managed 81 respondents/hour. |
| `--gpu-memory-utilization 0.94` | more KV, and therefore more concurrency, which is the binding constraint. |
| `kv_cache_dtype` | not passed: `models.ENGINE_DEFAULTS` supplies the required `fp8_ds_mla`. |
| `n_cpu 48`, `mem 800G` | the node caps a job at 12 cores and 250 GB **per GPU**, and `--mem 0` is refused outright for shared jobs. |

## Checking a run without waiting for it

```bash
# respondents done, and the live rate
curl -s -X POST http://cc_command_server:8765/run_in_container \
  -H 'Content-Type: application/json' \
  -d '{"cluster":"dais","command":"wc -l < data/pfander/silicon_sampling/v4_flash/answers.jsonl; tail -3 data/pfander/silicon_sampling/v4_flash/sample.log"}'
```

The sampler prints `[run] <done>/<todo> respondents | <rate>/h | eta <hours> h`
after every group, so the first two of those lines are enough to decide whether
the settings are right or the job should be cancelled.

## Pulling the results down

The outputs live in the bind-mounted repo, which `/sync_from_cluster` does not
cover — it only pulls `/opt/cluster_transfer`. So copy the small files there
first and leave `raw/` on the cluster:

```bash
for s in pfander Voelkel; do
  d=/opt/cluster_transfer/$s/v4_flash; mkdir -p $d
  cp data/$s/silicon_sampling/v4_flash/{answers.jsonl,draws.jsonl,samples.csv,run_meta.json,profiles.csv} $d/ 2>/dev/null
done
cp data/pfander/silicon_sampling/v4_flash/tier1_submission.csv /opt/cluster_transfer/pfander/v4_flash/
```

## Letting it resume unattended

`scripts/dais_auto_resume.sh` polls, and whenever no job for a run group is in the
queue and the target is not reached, resubmits the identical body. Start it from
the agent container *after* confirming a run produces sane completions — an
automatic resubmit loop pointed at bad settings just burns allocation:

```bash
scripts/dais_auto_resume.sh v4voelkel \
    data/Voelkel/silicon_sampling/v4_flash/answers.jsonl 6203 voelkel.json &
scripts/dais_auto_resume.sh v4pfander \
    data/pfander/silicon_sampling/v4_flash/answers.jsonl 18000 pfander.json &
```

## Measured throughput (2026-08-18, 4xH200, warm TileLang cache)

**2,400 respondents/hour** on Voelkel at group size 50 — 200 completions in 300 s,
measured incrementally so the ~8 minutes of engine startup is excluded. That puts
Voelkel at ~2.6 h and Pfänder at roughly 8-16 h (its group is ~31 rather than 50,
because its transcripts are longer, and throughput tracks group size).

Two earlier numbers, for contrast, both from the same hardware:

| | rate | why |
| --- | --- | --- |
| first pilot | 81/h | cold TileLang cache (96 JIT compiles) *and* group 64 forced against a capacity of 27, so every session was evicted and re-prefilled |
| this run | 2,400/h | warm cache, `--group-size 0` |

The cumulative rate the sampler prints on its *first* group is not the number to
read — it divides by wall time including model load, so it printed 405/h at a
moment when the steady rate was already 2,400/h.
