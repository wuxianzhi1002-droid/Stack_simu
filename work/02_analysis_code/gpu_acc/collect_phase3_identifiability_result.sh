#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; JOB_FILE="$SCRIPT_DIR/last_phase3_identifiability_job_id.txt"
[ -f "$JOB_FILE" ] || { printf '[FAIL] No identifiability JOBID.\n' >&2; exit 1; }
IFS='|' read -r JOB_ID RUN_ROOT < "$JOB_FILE"; RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/cluster_runs}"; RUN_DIR="$RUN_ROOT/$JOB_ID"
printf 'Phase 3 identifiability JOBID: %s\n' "$JOB_ID"
[ -f "$RUN_DIR/phase3_identifiability.json" ] || { printf '[INFO] Result not available yet.\n'; exit 0; }
cat "$RUN_DIR/phase3_identifiability_summary.md"
printf 'Machine JSON: %s\n' "$RUN_DIR/phase3_identifiability.json"
printf 'Jacobian archive: %s\n' "$RUN_DIR/phase3_identifiability_jacobians.npz"
printf 'Phase 4 was not started.\n'
