#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ROOT="$SCRIPT_DIR/cluster_runs"
LAST_JOB_FILE="$SCRIPT_DIR/last_job_id.txt"

command -v sbatch >/dev/null 2>&1 || {
    printf '[FAIL] sbatch is unavailable. Run this on the SSH login node.\n' >&2
    exit 1
}
mkdir -p "$RUN_ROOT"
cd "$SCRIPT_DIR"

printf '[INFO] Submitting exactly one GPU to gpu_5090.\n'
set +e
SUBMIT_OUTPUT=$(sbatch --gpus=1 -p gpu_5090 ./run_phase1_gpu.sh 2>&1)
SUBMIT_CODE=$?
set -e
printf '%s\n' "$SUBMIT_OUTPUT"
[ "$SUBMIT_CODE" -eq 0 ] || {
    printf '[FAIL] sbatch submission failed.\n' >&2
    exit "$SUBMIT_CODE"
}

JOB_ID=$(printf '%s\n' "$SUBMIT_OUTPUT" | awk '/Submitted batch job/ {print $4; exit}')
case "$JOB_ID" in
    ''|*[!0-9]*)
        printf '[FAIL] Could not parse JOBID from sbatch output.\n' >&2
        exit 1
        ;;
esac

mkdir -p "$RUN_ROOT/$JOB_ID"
printf '%s\n' "$SUBMIT_OUTPUT" > "$RUN_ROOT/$JOB_ID/submission.txt"
printf '%s\n' "$JOB_ID" > "$LAST_JOB_FILE"

printf '[PASS] Saved JOBID %s to %s\n' "$JOB_ID" "$LAST_JOB_FILE"
printf 'Status:  %s/cluster.sh status\n' "$SCRIPT_DIR"
printf 'Cancel:  %s/cluster.sh cancel\n' "$SCRIPT_DIR"
printf 'Results: %s/cluster.sh result\n' "$SCRIPT_DIR"
