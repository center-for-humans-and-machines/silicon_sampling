---
name: waiting_for_jobs
description: Block on a slurm job's state via the CC external command server. Use whenever the user asks to wait until a previously submitted slurm job starts running, finishes, is cancelled, or otherwise leaves the queue, instead of busy-polling `/jobs` yourself.
---

# Waiting on a slurm job

`wait_for_job.sh` is a thin polling loop around the `CC_command_server`'s `/jobs` endpoint. It is on `PATH`, so just invoke it as `wait_for_job.sh`.

Use this instead of manually polling `/jobs` + `/runs/<id>` from the agent: the script handles the async two-step (submit → poll run → parse `squeue --me` tail), tolerates transient SSH / network failures, and sleeps 2 minutes between cluster polls so it doesn't hammer the login node.

For very long running jobs (>6h), avoid Wait-until-done mode until you are sure that everything is working alright. Instead use long sleeps (E.g. sleep for 3-9 hours at a time) to check in regularly to restart the job early if bugs / issues make continuing the current job pointless.

## Usage

```bash
wait_for_job.sh <URL> <CLUSTER> <JOB_ID> [WAIT_UNTIL_DONE]
```

- `URL` — base URL of the server, almost always `http://cc_command_server:8765` from inside the project container.
- `CLUSTER` — `marenostrum` or `dais`. Must match the cluster the job was submitted to (the script polls `/jobs?cluster=<CLUSTER>`).
- `JOB_ID` — slurm jobid (`12345` or array task `12345_0`). Get it from the log tail of the run that submitted the job (e.g. `/request_single_gpu` prints `salloc: Pending job allocation <jobid>`; `/submit_slurm_job` prints the jobid via `sbatch`).
- `WAIT_UNTIL_DONE` — optional. If absent, the script exits as soon as the job is no longer pending (running / stopping / disappeared). If present and non-empty (any value, e.g. `1`, `wait`, `--wait-until-done`), the script keeps polling while the job is running and only exits on stopping or disappeared.

The script writes timestamped progress lines to stdout while it sleeps, then prints exactly one of:

- `wait_for_job: job <id> is running`
- `wait_for_job: job <id> is stopping (state=<code>)`
- `wait_for_job: job <id> has disappeared from the queue`

It exits 0 on any of those, 2 on bad arguments (including unknown `CLUSTER`), 3 on missing tools (`curl`, `jq`, `awk`).

## When to use which mode

- **Default mode** — you want to do something as soon as the job has resources, e.g. wait for `/request_single_gpu` to actually allocate before you `/run_in_job` on it.
- **Wait-until-done mode** — you want to block until a batch job has finished so you can inspect its outputs, e.g. after `/submit_slurm_job` for a training run.

## Tunables

Mostly relevant for tests; don't override in normal use.

- `WAIT_FOR_JOB_SLEEP` — seconds between cluster polls (default `120`).
- `WAIT_FOR_JOB_RUN_POLL` — seconds between `/runs/<id>` polls within a single `/jobs` call (default `3`).
- `WAIT_FOR_JOB_RUN_POLL_MAX` — max `/runs/<id>` polls per `/jobs` call before giving up and retrying (default `40`).
