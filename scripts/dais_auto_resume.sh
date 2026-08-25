#!/usr/bin/env bash
# Keep one study's run alive until it reaches its target, resubmitting if a job
# dies before finishing.
#
# **A safety net, not a plan.** An earlier version of this header claimed that a
# sequence of short jobs was the fastest route to a finished run, and the
# submission JSONs were written with 3 h limits to match. That is wrong and it
# cost real time: per the cluster-queue guidance the primary driver of DAIS queue
# wait is the number of GPUs requested, not the time limit, so a short limit does
# not buy a cheaper slot -- it just splits one job into several, each paying a full
# queue wait and a model load. Size the job for the work it has to do, with about
# 50% headroom, and let this script cover only the case where it dies anyway.
#
# It works because the sampler resumes: answers.jsonl says what is done, and
# resubmitting the identical job asks for the rest.
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

# Is a job of this run_group in the queue?  Three answers, not two: 0 yes, 1 no,
# 2 "cannot tell".
#
# The third one matters.  During a six-hour cluster outage every squeue returned
# `exit 255: Connection closed by UNKNOWN port 65535`, which matches no job name,
# so a two-valued check read it as "nothing is queued" and tried to submit six
# times.  Those submissions happened to fail at the same precheck, so no
# duplicates were created -- but that was luck, and a link that came back between
# the check and the submit would have produced a second job against a run that
# was already going.  A failed lookup now blocks submission instead of licensing
# it.
queued() {
    local rid ec tail
    rid=$(curl -s --max-time 40 "$API/jobs?cluster=dais" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin).get("run_id",""))' 2>/dev/null)
    [ -z "$rid" ] && return 2
    for _ in $(seq 1 25); do
        [ "$(curl -s --max-time 20 $API/runs/$rid | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)" = "exited" ] && break
        sleep 3
    done
    ec=$(curl -s --max-time 20 $API/runs/$rid \
         | python3 -c 'import json,sys; print(json.load(sys.stdin).get("exit_code"))' 2>/dev/null)
    [ "$ec" = "0" ] || return 2
    tail=$(curl -s --max-time 20 $API/runs/$rid \
           | python3 -c "import json,sys; print(json.load(sys.stdin).get('tail',''))" 2>/dev/null)
    echo "$tail" | grep -qE "[0-9]+ +\S+ +${GROUP:0:8}"
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
    queued; q=$?
    if [ "$q" -eq 0 ]; then
        echo "[$(date -u +%H:%M:%S)] $GROUP $n/$TARGET — a job is queued or running"
    elif [ "$q" -eq 2 ]; then
        echo "[$(date -u +%H:%M:%S)] $GROUP $n/$TARGET — cannot reach the cluster; not submitting"
    else
        out=$(curl -s -X POST $API/submit_slurm_job -H 'Content-Type: application/json' -d @"$JSON")
        submissions=$((submissions + 1))
        echo "[$(date -u +%H:%M:%S)] $GROUP $n/$TARGET — submitted #$submissions: $(echo "$out" | head -c 120)"
        sleep 60
    fi
    sleep 120
done
echo "[$(date -u +%H:%M:%S)] $GROUP gave up at $(done_now)/$TARGET"
