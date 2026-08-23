#!/usr/bin/env bash
# Run the local studies back to back, so the single 4090 never idles.
#
# Three things this has to get right, each of which cost real GPU time to learn.
#
# **Waiting.** `kill -0` is the wrong liveness test: it SUCCEEDS on a zombie, and a
# finished sampler whose parent shell has already exited stays a zombie until
# something reaps it — which launching with nohup from a shell that then exits
# produces every time. An earlier version spun for hours believing a run was still
# going when it had finished all 18,000 respondents. Matching on the command line
# instead is robust, because a zombie has no cmdline for pgrep to match.
#
# **Not fighting for the card.** The sampler wants essentially the whole 24 GB, so
# starting while anything else holds the GPU just fails. Retries are only useful
# once the card is actually free, so the wait is on the GPU, not on a retry
# counter: an earlier version burned all twelve of its attempts in three minutes
# against a busy card and gave up on both studies.
#
# **Resumability.** `answers.jsonl` is the source of truth, so a killed step costs
# the group in flight plus one model load, and re-running this script continues
# rather than restarting.
#
# Usage: scripts/chain_local_runs.sh [study ...]      (default: icpc goldwert)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

#: Respondent target per study, matching the profile sets on disk.
target_for () {
    case "$1" in
        icpc) echo 4200 ;;
        goldwert) echo 9570 ;;
        pfander) echo 18000 ;;
        voelkel) echo 6203 ;;
        *) echo 0 ;;
    esac
}

out_for () {
    case "$1" in
        icpc) echo "data/ICPC/silicon_sampling/qwen25_7b" ;;
        goldwert) echo "data/Goldwert/silicon_sampling/qwen25_7b" ;;
        pfander) echo "data/pfander/silicon_sampling/qwen25_7b" ;;
        voelkel) echo "data/Voelkel/silicon_sampling/qwen25_7b" ;;
    esac
}

# Wait until no sampler of any study is running, and the card is actually free.
wait_for_gpu () {
    local waited=0
    while pgrep -f "silicon_sampling\..*\.cli sample" > /dev/null; do
        [ $((waited % 20)) -eq 0 ] && echo "[chain] another sampler is running; waiting ($(date -Is))"
        sleep 60
        waited=$((waited + 1))
    done
    for _ in $(seq 1 40); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-999999}" -lt 2000 ] && return 0
        echo "[chain] GPU still holds ${used} MiB; waiting"
        sleep 30
    done
    echo "[chain] GPU never freed; giving up"
    return 1
}

run_study () {
    local study="$1"
    local target out done_now
    target=$(target_for "$study")
    out=$(out_for "$study")
    if [ "$target" -eq 0 ]; then
        echo "[chain] unknown study: $study"
        return 2
    fi
    mkdir -p "$out"
    echo "[chain] === $study -> $out ($(date -Is))"
    for attempt in 1 2 3 4 5 6 7 8; do
        done_now=0
        [ -f "$out/answers.jsonl" ] && done_now=$(wc -l < "$out/answers.jsonl")
        if [ "$done_now" -ge "$target" ]; then
            echo "[chain] $study complete at $done_now/$target ($(date -Is))"
            python3 -m "silicon_sampling.$study.cli" build-csv --run qwen25_7b \
                >> "$out/sample.log" 2>&1
            return 0
        fi
        wait_for_gpu || return 1
        echo "[chain] $study attempt $attempt from $done_now/$target ($(date -Is))"
        python3 -m "silicon_sampling.$study.cli" sample --run qwen25_7b \
            >> "$out/sample.log" 2>&1
        # A step that advanced nothing means a real failure, not a wall-clock kill,
        # so stop rather than retrying into the same error.
        local after=0
        [ -f "$out/answers.jsonl" ] && after=$(wc -l < "$out/answers.jsonl")
        if [ "$after" -le "$done_now" ] && [ "$attempt" -ge 2 ]; then
            echo "[chain] $study made no progress in two attempts; see $out/sample.log"
            return 1
        fi
        sleep 15
    done
    echo "[chain] $study exhausted its attempts at $after/$target"
}

for study in "${@:-icpc goldwert}"; do
    run_study "$study" || echo "[chain] $study did not finish"
done
echo "[chain] all done ($(date -Is))"
