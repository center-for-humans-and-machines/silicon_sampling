---
name: run_on_cluster
description: Run cluster jobs on the DAIS cluster via the CC external command server. Use whenever the user asks to submit, query, cancel, monitor, or interactively run something on DAIS (slurm, salloc, GPU, apptainer container).
---

# Running on DAIS

DAIS is reachable through a small HTTP server (`CC_command_server`) that runs in a sibling Docker container. It SSHes into the login node, executes whitelisted commands, and writes merged stdout/stderr to a log file. **You never SSH the cluster directly — always go through this server.**

This project has **only DAIS**. There is no MareNostrum / raven fallback.

## Cluster

Every endpoint that touches a cluster takes `cluster: "dais"`:
- POST endpoints: JSON body field.
- GET endpoints: query string parameter.

Locked-down settings (slurm template, bind mounts, env vars) are baked into the server. On DAIS the main container image may be chosen from a fixed allowlist via optional `container`.

Default image: `/dais/fs/scratch/ykeller/containers/apptainer/glm_52_vllm.sif`.
Optional: `/dais/fs/scratch/ykeller/containers/apptainer/vllm-openai_latest.sif`.

Jobs may use **1 or 2 nodes** (`n_nodes` 1–2). GPUs per node: 1–8. Omit `gpu_type` to request any GPU; default `--mem 0` (whole node). Wall-time cap on DAIS is 24h.

Connection to DAIS requires an active SSH connection established by a human operator. If commands prompt for a password or SSH fails, stop and tell the user. Do not try to work around a missing DAIS connection.

## Connection

- Base URL: `http://cc_command_server:8765/`.
- Output logs: `/var/cc_output/<run_id>.log` (mounted into this container at the same path; you can `Read` them directly).
- Source code (read-only here, for reference): `/opt/claude_tools/CC_command_server/`.

Every endpoint that *starts work* returns:

```json
{"run_id": "<id>", "output_file": "/var/cc_output/<id>.log"}
```

The work runs asynchronously. Poll `GET /runs/{run_id}` for status (`running` / `exited`), `exit_code`, and a tail of the log. Read the full log from `output_file` when needed.

## Example Workflow

1. After changing the local `silicon_sampling` checkout, sync it with `POST /sync_to_cluster` (`{"cluster": "dais"}`) before using it on DAIS.
2. Hit the endpoint → get a `run_id`.
3. Poll `GET /runs/{run_id}` (sleep a few seconds first; the SSH handshake takes a moment).
4. Inspect `tail` for quick output, or `Read` the `output_file` for the full log.
5. For long-running things (interactive GPU `salloc`, monitoring), the run stays `running` until killed. Use `POST /runs/{run_id}/kill` to stop it.

## Important: When something doesn't work as expected

If the cluster is down, the container is misconfigured, essential commands fail independently of your input, or permissions block the task: **do not work around the configuration issue**. Return and inform the user.

Inside the cluster container the repo is at `/opt/silicon_sampling` (`PYTHONPATH` / cwd). Host-side it lives at `/u/ykeller/github_repos/silicon_sampling`. Copy files into `/opt/cluster_transfer` on the cluster to pull them locally via `/sync_from_cluster`.

## Endpoints

### `POST /run_in_container`
Run an arbitrary command **on the login node** inside the Apptainer container. **Login node only — no GPU.**

Body: `{"cluster": "dais", "command": "<shell command>", "container"?: "<sif path>"}` (command max 4096 bytes, no newlines / NULs / control chars).

`container` is optional. Omit for `glm_52_vllm.sif`. Allowlist: that default plus `vllm-openai_latest.sif`.

### `POST /run_in_job`
Run a command **inside an already-running slurm job** via `srun --overlap`. Use this to exec into an interactive GPU allocation from `/request_single_gpu`, or to attach to a batch job.

Body: `{"cluster": "dais", "jobid": "<slurm_jobid>", "command": "<shell command>", "container"?: "<sif path>"}`

`jobid` matches `^[0-9]+(_[0-9]+)?$`. Same optional `container` allowlist as `/run_in_container`.

### `GET /jobs`
Run `squeue --me` on the login node. Returns a `run_id`; poll it to see queued/running jobs.

Query params: `cluster=dais` (required).

### `GET /sinfo`
Run bare `sinfo` on the login node (no extra flags). Returns a `run_id`; poll it or `Read` the log to see partition/node state.

Query params: `cluster=dais` (required).

### `POST /cancel_job`
Run `scancel <jobid>` on the login node.

Body: `{"cluster": "dais", "jobid": "<slurm_jobid>"}`

### `POST /submit_slurm_job`
Submit a batch job through the project's `slurm_job` wrapper. The template `run_slurm_dais_apptainer_silicon_sampling.sh` is hardcoded.

Body:
| Field | Type | Notes |
|---|---|---|
| `cluster` | str | Must be `dais`. |
| `program_call` | str | Command to run inside the container (passed to the chosen launcher). |
| `time` | str | Wall-time `HH:MM:SS`, server-side cap 72h; DAIS slurm cap is 24h. |
| `n_gpu` | int | GPUs per node, 1–8. |
| `n_nodes` | int | **1–2**. |
| `n_cpu` | int | CPUs per task, 1–640. |
| `launcher` | str | One of `python`, `accelerate`, `torchrun`. |
| `run_group` | str | Identifier, `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`. |
| `project` | str | Same identifier rule. |
| `acc_config` | str? | Optional, `configs/accelerate/<name>.ya?ml`. |
| `mem` | str? | SLURM `--mem` (e.g. `0` for whole node, `64G`). Defaults to `0`. |
| `gpu_type` | str? | Pin `h200` or `b200`. Omit to request any GPU (`--gres=gpu:N`). |
| `container` | str? | Main job image; allowlist: `glm_52_vllm.sif` (default) + `vllm-openai_latest.sif`. |
| `vllm_servers` | object[]? | Optional vLLM sidecar servers in `vllm-openai_latest.sif` before the main container. **Requires `n_nodes=1`.** |

Pre-flight check: server refuses with **HTTP 409** if you already have ≥ `CC_MAX_JOBS` queued/running jobs.

### `POST /request_single_gpu`
Allocates one interactive GPU for 2h:

`salloc --partition=gpu --gres=gpu:1 --mem=120GB --time=02:00:00 -J started sleep 7200`

Body: `{"cluster": "dais"}`.

The run stays `running` while the allocation lives — kill it (or let `sleep 7200` expire / `scancel` the slurm job) to release the GPU. Same `CC_MAX_JOBS` quota as `/submit_slurm_job`. The slurm jobid appears in the log tail; pass it to `/run_in_job` to exec on the GPU.

Use this when you expect to run several different GPU scripts and want to avoid waiting for allocation each time. As long as it is only one GPU, don't worry about GPU idle time.

### `GET /monitor_newest_slurm_run`
Run `monitor_run` against the cluster's slurm output dir. Main way to watch jobs started by `/submit_slurm_job`.

Query params:
- `cluster=dais` (required).
- `oldness` (int, 0–1000, default 0): 0 = newest, 1 = second-newest, etc.
- `script` (bool, default false): if true, view `slurm_script.sh` instead of `slurm-*.out`.

### `POST /sync_from_cluster`
Pull DAIS `cluster_transfer` into local `/var/cluster_transfer` via `rsync`.

Body: `{"cluster": "dais"}`.

On the cluster the folder is `/opt/cluster_transfer`. Copy files there before running this. Connection speed is limited — do not copy >1GB of data.

### `POST /sync_to_cluster`
Push the local `silicon_sampling` checkout to DAIS (`/opt/silicon_sampling/` → `/u/ykeller/github_repos/silicon_sampling/`, excluding `.git`).

Body: `{"cluster": "dais"}`.

### `GET /runs/{run_id}`
Status of a previously-started run. Returns `run_id`, `endpoint`, `output_file`, `status` (`running` | `exited`), `exit_code`, `started_at`, and `tail`.

### `POST /runs/{run_id}/kill`
Terminate the underlying SSH process for a run. Killing SSH does **not** `scancel` a slurm job — use `/cancel_job` with the slurm jobid for that.

## Patterns

**Submit and confirm pending:**
```bash
curl -s -X POST http://cc_command_server:8765/request_single_gpu \
  -H 'Content-Type: application/json' \
  -d '{"cluster": "dais"}'
# -> {"run_id": "...", "output_file": "..."}
sleep 5
curl -s http://cc_command_server:8765/runs/<run_id>           # "salloc: Pending job allocation <jobid>"
curl -s 'http://cc_command_server:8765/jobs?cluster=dais'     # poll returned run_id; state PD = pending
curl -s 'http://cc_command_server:8765/sinfo?cluster=dais'    # then poll; node/partition idle vs allocated
```

**Run a command on a live GPU allocation:**
```bash
curl -s -X POST http://cc_command_server:8765/run_in_job \
  -H 'Content-Type: application/json' \
  -d '{"cluster": "dais", "jobid": "12345", "command": "nvidia-smi"}'
```

**Submit a 2-node job (default glm_52_vllm.sif, any GPU):**
```bash
curl -s -X POST http://cc_command_server:8765/submit_slurm_job \
  -H 'Content-Type: application/json' \
  -d '{"cluster": "dais", "program_call": "python -m silicon_sampling ...", "time": "04:00:00",
       "n_gpu": 8, "n_nodes": 2, "n_cpu": 96, "launcher": "torchrun",
       "run_group": "exp1", "project": "silicon_sampling"}'
```

Pin a GPU type with `"gpu_type": "h200"` or `"gpu_type": "b200"`.

**Sync repo to DAIS, then monitor:**
```bash
curl -s -X POST http://cc_command_server:8765/sync_to_cluster \
  -H 'Content-Type: application/json' -d '{"cluster": "dais"}'
curl -s 'http://cc_command_server:8765/monitor_newest_slurm_run?cluster=dais&oldness=0'
```
