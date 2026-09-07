#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; RUN_ROOT="$SCRIPT_DIR/cluster_runs"; JOB_FILE="$SCRIPT_DIR/last_phase3_job_id.txt"
command -v sbatch >/dev/null 2>&1 || { printf '[FAIL] sbatch unavailable.
' >&2; exit 1; }
mkdir -p "$RUN_ROOT"; cd "$SCRIPT_DIR"
set +e; OUT=$(sbatch --gpus=1 -p gpu_5090 ./run_phase3_gpu.sh 2>&1); CODE=$?; set -e
printf '%s
' "$OUT"; [ "$CODE" -eq 0 ] || exit "$CODE"
JOB_ID=$(printf '%s
' "$OUT" | awk '/Submitted batch job/ {print $4; exit}')
case "$JOB_ID" in ''|*[!0-9]*) printf '[FAIL] Could not parse JOBID.
' >&2; exit 1;; esac
mkdir -p "$RUN_ROOT/$JOB_ID"; printf '%s
' "$OUT" > "$RUN_ROOT/$JOB_ID/submission.txt"; printf '%s
' "$JOB_ID" > "$JOB_FILE"
printf '[PASS] Saved Phase 3 JOBID %s.
' "$JOB_ID"
