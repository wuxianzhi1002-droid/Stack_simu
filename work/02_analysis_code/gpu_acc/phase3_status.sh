#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; parajobs 2>&1 || true; JOB_FILE="$SCRIPT_DIR/last_phase3_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[INFO] No Phase 3 JOBID.
'; exit 0; }
JOB_ID=$(tr -d '[:space:]' < "$JOB_FILE"); printf '=== Last Phase 3 JOBID: %s ===
' "$JOB_ID"
for log in "$SCRIPT_DIR/phase3_slurm_${JOB_ID}.out" "$SCRIPT_DIR/phase3_slurm_${JOB_ID}.err" "$SCRIPT_DIR/cluster_runs/$JOB_ID/stdout.log" "$SCRIPT_DIR/cluster_runs/$JOB_ID/stderr.log"; do if [ -f "$log" ]; then printf '=== Last 30 lines: %s ===
' "$log"; tail -n 30 "$log" || true; else printf '[INFO] Not available: %s
' "$log"; fi; done
exit 0
