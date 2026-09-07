#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
printf '=== parajobs ===
'
parajobs 2>&1 || true
JOB_FILE="$SCRIPT_DIR/last_phase2_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[INFO] No Phase 2 JOBID yet.
'; exit 0; }
JOB_ID=$(tr -d '[:space:]' < "$JOB_FILE")
printf '=== Last Phase 2 JOBID: %s ===
' "$JOB_ID"
for log in "$SCRIPT_DIR/phase2_slurm_${JOB_ID}.out" "$SCRIPT_DIR/phase2_slurm_${JOB_ID}.err" "$SCRIPT_DIR/cluster_runs/$JOB_ID/stdout.log" "$SCRIPT_DIR/cluster_runs/$JOB_ID/stderr.log"; do
 if [ -f "$log" ]; then printf '=== Last 30 lines: %s ===
' "$log"; tail -n 30 "$log" || true; else printf '[INFO] Log not available yet: %s
' "$log"; fi
done
exit 0
