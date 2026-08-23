#!/usr/bin/env bash
# Sample one study with one model, re-entering the sampler until it is done.
#
# The local wrapper (run_full_sample.sh) reclaims a single GPU between attempts
# with pkill + nvidia-smi polling.  On the cluster that is wrong: the allocation
# is ours for the whole job, the engine's worker processes die with their parent,
# and a stray pkill would be aimed at the wrong thing.  So this wrapper only
# retries; the sampler's own resumability does the rest.
#
# Usage: scripts/run_cluster_sample.sh <pfander|voelkel|icpc|goldwert> <run_key> <target_n> [extra sampler args...]
#
# Called from a slurm job as, e.g.:
#   -c pass; bash scripts/run_cluster_sample.sh pfander v4_flash 18000 \
#       --tensor-parallel-size 4 --expert-parallel --group-size 256

set -u

STUDY="${1:?study: pfander, voelkel, icpc or goldwert}"
RUN="${2:?run key, e.g. v4_flash}"
TARGET="${3:?target respondent count}"
shift 3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The directory case is the only per-study knowledge in this wrapper: the CLI
# module path is "silicon_sampling.<study>.cli" for all four, and every one of them
# accepts the same --run and writes answers.jsonl one line per finished respondent,
# which is what the retry loop below counts.
case "$STUDY" in
    pfander)  OUT="$ROOT/data/pfander/silicon_sampling/$RUN" ;;
    voelkel)  OUT="$ROOT/data/Voelkel/silicon_sampling/$RUN" ;;
    icpc)     OUT="$ROOT/data/ICPC/silicon_sampling/$RUN" ;;
    goldwert) OUT="$ROOT/data/Goldwert/silicon_sampling/$RUN" ;;
    *) echo "unknown study: $STUDY" >&2; exit 2 ;;
esac
LOG="$OUT/sample.log"
mkdir -p "$OUT"

for attempt in $(seq 1 12); do
    done_now=0
    [ -f "$OUT/answers.jsonl" ] && done_now=$(wc -l < "$OUT/answers.jsonl")
    if [ "$done_now" -ge "$TARGET" ]; then
        echo "[wrapper] $done_now/$TARGET complete" | tee -a "$LOG"
        break
    fi
    echo "[wrapper] attempt $attempt starting at $done_now/$TARGET ($(date -Is))" | tee -a "$LOG"

    python3 -m "silicon_sampling.$STUDY.cli" sample --run "$RUN" "$@" >> "$LOG" 2>&1
    echo "[wrapper] attempt $attempt exited $?" | tee -a "$LOG"
    sleep 5
done

final=0
[ -f "$OUT/answers.jsonl" ] && final=$(wc -l < "$OUT/answers.jsonl")
echo "[wrapper] finished with $final/$TARGET respondents" | tee -a "$LOG"

# Build the analysis CSVs here, while the cluster still has the answers, so the
# pull-down is a few tens of MB rather than the whole raw transcript tree.
if [ "$final" -ge "$TARGET" ]; then
    python3 -m "silicon_sampling.$STUDY.cli" build-csv --run "$RUN" 2>&1 | tee -a "$LOG"
fi
