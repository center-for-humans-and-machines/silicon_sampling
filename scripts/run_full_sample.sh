#!/usr/bin/env bash
# Sample the full 18,000-respondent silicon sample.
#
# The run takes many hours and does not fit in one shell invocation, so this
# wrapper simply re-enters the sampler until every respondent is done. The
# sampler is resumable — answers.jsonl is the source of truth — so a restart
# costs at most the group that was in flight plus one model load.
#
# Usage: scripts/run_full_sample.sh [target_n] [group_size]
#   group_size 0 (the default) sizes the group to the KV cache the engine gets.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-18000}"
GROUP="${2:-0}"
OUT="$ROOT/data/pfander/silicon_sampling/qwen25_7b"
LOG="$OUT/sample.log"

mkdir -p "$OUT"
cd "$ROOT"

for attempt in $(seq 1 40); do
    done_now=0
    [ -f "$OUT/answers.jsonl" ] && done_now=$(wc -l < "$OUT/answers.jsonl")
    if [ "$done_now" -ge "$TARGET" ]; then
        echo "[wrapper] $done_now/$TARGET complete" | tee -a "$LOG"
        break
    fi
    echo "[wrapper] attempt $attempt starting at $done_now/$TARGET ($(date -Is))" | tee -a "$LOG"

    # A stale engine from a killed attempt would hold the GPU and make the next
    # start fail on memory; clear it before trying again.
    # A previous attempt's engine can still hold the GPU; starting before it
    # lets go fails on memory and burns an attempt. Wait for the card, don't
    # guess how long it takes.
    pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-9999}" -lt 2000 ] && break
        sleep 5
    done

    python -m silicon_sampling.pfander.cli sample --group-size "$GROUP" --gpu-memory-utilization 0.96 >> "$LOG" 2>&1
    status=$?
    echo "[wrapper] attempt $attempt exited $status" | tee -a "$LOG"
    sleep 10
done

final=0
[ -f "$OUT/answers.jsonl" ] && final=$(wc -l < "$OUT/answers.jsonl")
echo "[wrapper] finished with $final/$TARGET respondents" | tee -a "$LOG"
