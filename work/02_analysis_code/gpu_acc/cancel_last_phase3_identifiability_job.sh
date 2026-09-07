#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; JOB_FILE="$SCRIPT_DIR/last_phase3_identifiability_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[FAIL] No identifiability JOBID.\n' >&2; exit 1; }
IFS='|' read -r JOB_ID RUN_ROOT < "$JOB_FILE"; printf 'Phase 3 identifiability JOBID: %s\n' "$JOB_ID"; scancel "$JOB_ID"; parajobs 2>&1 || true
