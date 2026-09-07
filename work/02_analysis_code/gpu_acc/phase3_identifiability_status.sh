#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
available_4090(){ local table count; table=$(sinfo -p gpu_4090 2>&1 || true); count=$(printf '%s\n' "$table" | awk '$1=="gpu_4090" && $2 ~ /^[0-9]+$/ {print $2; exit}'); case "$count" in ''|*[!0-9]*) count=0;; esac; printf '%s\n' "$count"; }
if [ "${1:-}" = --watch ]; then
    JOB_ID="${2:?missing JOBID}"; RUN_ROOT="${3:?missing RUN_ROOT}"; WAIT_SECONDS="${TMM_GPU_FALLBACK_WAIT_SECONDS:-600}"
    sleep "$WAIT_SECONDS"
    while true; do
        STATUS=$(squeue -h -j "$JOB_ID" -o '%T|%P' 2>/dev/null || true)
        [ "$STATUS" = 'PENDING|gpu_5090' ] || { printf '[INFO] No fallback: %s\n' "${STATUS:-job left queue}"; exit 0; }
        AVAILABLE=$(available_4090); printf '[INFO] gpu_4090 available=%s\n' "$AVAILABLE"
        if [ "$AVAILABLE" -gt 0 ]; then
            scancel --state=PENDING "$JOB_ID" 2>/dev/null || true; sleep 2
            STATUS=$(squeue -h -j "$JOB_ID" -o '%T|%P' 2>/dev/null || true)
            [ -z "$STATUS" ] || { printf '[INFO] Original job remains %s; no fallback.\n' "$STATUS"; exit 0; }
            printf '[PASS] Cancelled pending 5090 JOBID %s; resubmitting on 4090.\n' "$JOB_ID"
            TMM_GPU_PARTITION=gpu_4090 TMM_DISABLE_FALLBACK=1 TMM_CLUSTER_RUN_ROOT="$RUN_ROOT" bash "$SCRIPT_DIR/submit_phase3_identifiability.sh"
            exit $?
        fi
        sleep 60
    done
fi
parajobs 2>&1 || true
JOB_FILE="$SCRIPT_DIR/last_phase3_identifiability_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[INFO] No identifiability JOBID.\n'; exit 0; }
IFS='|' read -r JOB_ID RUN_ROOT < "$JOB_FILE"; RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/cluster_runs}"
printf '=== Last Phase 3 identifiability JOBID: %s ===\n' "$JOB_ID"
for log in "$RUN_ROOT/phase3_identifiability_slurm_${JOB_ID}.out" "$RUN_ROOT/phase3_identifiability_slurm_${JOB_ID}.err" "$RUN_ROOT/$JOB_ID/stdout.log" "$RUN_ROOT/$JOB_ID/stderr.log"; do if [ -f "$log" ]; then printf '=== Last 30 lines: %s ===\n' "$log"; tail -n 30 "$log" || true; else printf '[INFO] Not available: %s\n' "$log"; fi; done
exit 0
