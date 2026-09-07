#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_FILE="$SCRIPT_DIR/last_phase2_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[FAIL] No Phase 2 JOBID.
' >&2; exit 1; }
JOB_ID=$(tr -d '[:space:]' < "$JOB_FILE")
RUN_DIR="$SCRIPT_DIR/cluster_runs/$JOB_ID"
printf 'Phase 2 JOBID: %s
' "$JOB_ID"
[ -f "$RUN_DIR/phase2_result.json" ] || { printf '[INFO] Result JSON is not available yet.
'; exit 0; }
printf '%s
' "=== $RUN_DIR/phase2_result_summary.md ==="
cat "$RUN_DIR/phase2_result_summary.md"
printf 'Machine JSON: %s
' "$RUN_DIR/phase2_result.json"
printf 'Phase 3 was not started.
'
