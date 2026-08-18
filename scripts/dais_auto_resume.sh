#!/usr/bin/env bash
# Keep one study's run alive until it reaches its target, one short job at a time.
#
# DAIS schedules a 4-GPU job in minutes and a whole-node job in hours, so the
# fastest route to a finished run is a sequence of short jobs rather than one long
# one. That only works because the sampler resumes: answers.jsonl says what is
# done, and resubmitting the identical job asks for the rest.
#
# Usage: scripts/dais_auto_resume.sh <run_group> <answers_path> <target> <submit_json>
#
# Runs in the agent container, not on the cluster: it talks to the CC command
# server. The submit_json is the same body documented in dais_submit_v4_flash.md.

set -u
GROUP="${1:?run_group}"
ANSWERS="${2:?path to answers.jsonl on the cluster}"
TARGET="${3:?target respondent count}"
JSON="${4:?submission json file}"
API=http://cc_command_server:8765

remote() {
    local rid
    rid=$(curl -s -X POST $API/run_in_container -H 'Content-Type: application/json' \
          -d "{\"cluster\":\"dais\",\"command\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
    for _ in $(seq 1 40); do
        [ "$(curl -s $API/runs/$rid | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')" = "exited" ] && break
        sleep 3
    done
    grep -v gocryptfs /var/cc_output/$rid.log 2>/dev/null
}

queued() {  # is a job of this run_group in the queue?
    local rid
    rid=$(curl -s "$API/jobs?cluster=dais" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
    for _ in $(seq 1 25); do
        [ "$(curl -s $API/runs/$rid | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')" = "exited" ] && break
        sleep 3
    done
    curl -s $API/runs/$rid | python3 -c "import json,sys; print(json.load(sys.stdin)['tail'])" \
        | grep -qE "[0-9]+ +\S+ +${GROUP:0:8}"
}

done_now() { remote "n=0; [ -f $ANSWERS ] && n=\$(wc -l < $ANSWERS); echo COUNT=\$n" \
             | grep -oE 'COUNT=[0-9]+' | cut -d= -f2 | tail -1; }

submissions=0
for i in $(seq 1 500); do
    n=$(done_now); n=${n:-0}
    if [ "$n" -ge "$TARGET" ]; then
        echo "[$(date -u +%H:%M:%S)] $GROUP COMPLETE: $n/$TARGET after $submissions submissions"
        exit 0
    fi
    if queued; then
        echo "[$(date -u +%H:%M:%S)] $GROUP $n/$TARGET — a job is queued or running"
    else
        out=$(curl -s -X POST $API/submit_slurm_job -H 'Content-Type: application/json' -d @"$JSON")
        submissions=$((submissions + 1))
        echo "[$(date -u +%H:%M:%S)] $GROUP $n/$TARGET — submitted #$submissions: $(echo "$out" | head -c 120)"
        sleep 60
    fi
    sleep 120
done
echo "[$(date -u +%H:%M:%S)] $GROUP gave up at $(done_now)/$TARGET"
