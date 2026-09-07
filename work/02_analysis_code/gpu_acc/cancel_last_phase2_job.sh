#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_FILE="$SCRIPT_DIR/last_phase2_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[FAIL] No Phase 2 JOBID.
' >&2; exit 1; }
JOB_ID=$(tr -d '[:space:]' < "$JOB_FILE")
printf 'Phase 2 JOBID: %s
' "$JOB_ID"
scancel "$JOB_ID"
parajobs 2>&1 || true
