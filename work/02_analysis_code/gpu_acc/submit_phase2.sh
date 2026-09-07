#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ROOT="$SCRIPT_DIR/cluster_runs"
LAST_JOB_FILE="$SCRIPT_DIR/last_phase2_job_id.txt"
command -v sbatch >/dev/null 2>&1 || { printf '[FAIL] sbatch unavailable.
' >&2; exit 1; }
mkdir -p "$RUN_ROOT"
cd "$SCRIPT_DIR"
printf '[INFO] Submitting Phase 2 exactly one GPU to gpu_5090.
'
set +e
SUBMIT_OUTPUT=$(sbatch --gpus=1 -p gpu_5090 ./run_phase2_gpu.sh 2>&1)
SUBMIT_CODE=$?
set -e
printf '%s
' "$SUBMIT_OUTPUT"
[ "$SUBMIT_CODE" -eq 0 ] || exit "$SUBMIT_CODE"
JOB_ID=$(printf '%s
' "$SUBMIT_OUTPUT" | awk '/Submitted batch job/ {print $4; exit}')
case "$JOB_ID" in ''|*[!0-9]*) printf '[FAIL] Could not parse JOBID.
' >&2; exit 1;; esac
mkdir -p "$RUN_ROOT/$JOB_ID"
printf '%s
' "$SUBMIT_OUTPUT" > "$RUN_ROOT/$JOB_ID/submission.txt"
printf '%s
' "$JOB_ID" > "$LAST_JOB_FILE"
printf '[PASS] Saved Phase 2 JOBID %s.
' "$JOB_ID"
printf 'Status: %s/phase2_cluster.sh status
' "$SCRIPT_DIR"
printf 'Result: %s/phase2_cluster.sh result
' "$SCRIPT_DIR"
