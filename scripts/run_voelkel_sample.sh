#!/usr/bin/env bash
# Sample the full Voelkel silicon sample.
#
# Re-enters the sampler until every respondent is done. The sampler is resumable
# from answers.jsonl, so an unexpected kill costs at most the group in flight
# plus one model load.
#
# Usage: scripts/run_voelkel_sample.sh [target_n]

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-6203}"
OUT="$ROOT/data/Voelkel/silicon_sampling/qwen25_7b"
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

    pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
    sleep 5

    python -m silicon_sampling.voelkel.cli sample >> "$LOG" 2>&1
    echo "[wrapper] attempt $attempt exited $?" | tee -a "$LOG"
    sleep 10
done

final=0
[ -f "$OUT/answers.jsonl" ] && final=$(wc -l < "$OUT/answers.jsonl")
echo "[wrapper] finished with $final/$TARGET respondents" | tee -a "$LOG"
