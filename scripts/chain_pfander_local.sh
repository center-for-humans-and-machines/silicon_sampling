#!/usr/bin/env bash
# Run Pfander sampling jobs back to back on the single local GPU.
#
# Usage: scripts/chain_pfander_local.sh "<run>:<profiles>" ["<run>:<profiles>" ...]
#
# Waiting on a running sampler is done by matching its command line, not by
# `kill -0`: that call SUCCEEDS on a zombie, and a sampler whose parent shell has
# exited stays a zombie until something reaps it, which is exactly what launching
# with nohup produces.  An earlier version of this pattern spun for hours
# believing a finished run was still going.  A zombie has no cmdline, so pgrep
# will not match it.
#
# The card also has to be genuinely free before the next run starts: the sampler
# wants essentially all 24 GB, so starting while anything still holds it just
# fails.  Hence the wait is on the GPU, not on a retry counter.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TARGET=18000

wait_for_gpu () {
    while pgrep -f "silicon_sampling\..*\.cli sample" > /dev/null; do sleep 60; done
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-999999}" -lt 2000 ] && return 0
        sleep 30
    done
    echo "[chain] GPU never freed"; return 1
}

for spec in "$@"; do
    RUN="${spec%%:*}"; PROFILES="${spec#*:}"
    OUT="data/pfander/silicon_sampling/$RUN"
    mkdir -p "$OUT"
    for attempt in 1 2 3 4 5 6 7 8; do
        done_now=0
        [ -f "$OUT/answers.jsonl" ] && done_now=$(wc -l < "$OUT/answers.jsonl")
        if [ "$done_now" -ge "$TARGET" ]; then
            echo "[chain] $RUN complete at $done_now/$TARGET ($(date -Is))"
            python3 -m silicon_sampling.pfander.cli build-csv --run "$RUN" >> "$OUT/sample.log" 2>&1
            break
        fi
        wait_for_gpu || exit 1
        echo "[chain] $RUN attempt $attempt from $done_now/$TARGET ($(date -Is))"
        python3 -m silicon_sampling.pfander.cli sample --run "$RUN" --profiles "$PROFILES" \
            >> "$OUT/sample.log" 2>&1
        after=0; [ -f "$OUT/answers.jsonl" ] && after=$(wc -l < "$OUT/answers.jsonl")
        # No progress twice running means a real failure, not a wall-clock kill.
        if [ "$after" -le "$done_now" ] && [ "$attempt" -ge 2 ]; then
            echo "[chain] $RUN made no progress in two attempts; see $OUT/sample.log"
            break
        fi
        sleep 15
    done
done
echo "[chain] all done ($(date -Is))"
