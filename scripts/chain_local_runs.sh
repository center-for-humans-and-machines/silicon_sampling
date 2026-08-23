#!/usr/bin/env bash
# Run the local studies back to back, so the single 4090 never idles.
#
# The GPU holds one engine at a time and the current occupant must not be
# disturbed, so this waits for a named PID to exit rather than reclaiming memory
# by force.  Each sampler is resumable, so a killed step costs the group in
# flight and one model load, and re-running this script picks up where it left
# off.
#
# Usage: scripts/chain_local_runs.sh <pid-to-wait-for>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
    echo "[chain] waiting for pid $WAIT_PID to finish ($(date -Is))"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[chain] pid $WAIT_PID gone ($(date -Is))"
fi

run_study () {
    local study="$1" target="$2"
    local out="data/$( [ "$study" = icpc ] && echo ICPC || echo Goldwert )/silicon_sampling/qwen25_7b"
    mkdir -p "$out"
    echo "[chain] === $study -> $out ($(date -Is))"
    for attempt in 1 2 3 4 5 6; do
        local done_now=0
        [ -f "$out/answers.jsonl" ] && done_now=$(wc -l < "$out/answers.jsonl")
        if [ "$done_now" -ge "$target" ]; then
            echo "[chain] $study complete at $done_now/$target"
            python3 -m "silicon_sampling.$study.cli" build-csv --run qwen25_7b >> "$out/sample.log" 2>&1
            return 0
        fi
        echo "[chain] $study attempt $attempt from $done_now/$target"
        python3 -m "silicon_sampling.$study.cli" sample --run qwen25_7b >> "$out/sample.log" 2>&1
        sleep 10
    done
    echo "[chain] $study gave up after 6 attempts"
}

run_study icpc 4200
run_study goldwert 9570
echo "[chain] all done ($(date -Is))"
