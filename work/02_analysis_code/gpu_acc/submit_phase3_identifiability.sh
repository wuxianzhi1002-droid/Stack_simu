#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_FILE="$SCRIPT_DIR/last_phase3_identifiability_job_id.txt"
command -v sbatch >/dev/null 2>&1 || { printf '[FAIL] sbatch unavailable.\n' >&2; exit 1; }
PARTITION="${TMM_GPU_PARTITION:-gpu_5090}"
case "$PARTITION" in gpu_5090|gpu_4090) ;; *) printf '[FAIL] Unsupported partition: %s\n' "$PARTITION" >&2; exit 2;; esac
RUN_ROOT="${TMM_CLUSTER_RUN_ROOT:-$SCRIPT_DIR/cluster_runs}"
mkdir -p "$RUN_ROOT"
cd "$SCRIPT_DIR"
printf '[PASS] Selected partition: %s\n' "$PARTITION"
printf '[PASS] Run root: %s\n' "$RUN_ROOT"
set +e
OUT=$(sbatch --gpus=1 -p "$PARTITION" --export=ALL,TMM_CLUSTER_RUN_ROOT="$RUN_ROOT" --output="$RUN_ROOT/phase3_identifiability_slurm_%j.out" --error="$RUN_ROOT/phase3_identifiability_slurm_%j.err" ./run_phase3_identifiability_gpu.sh 2>&1)
CODE=$?
set -e
printf '%s\n' "$OUT"; [ "$CODE" -eq 0 ] || exit "$CODE"
JOB_ID=$(printf '%s\n' "$OUT" | awk '/Submitted batch job/ {print $4; exit}')
case "$JOB_ID" in ''|*[!0-9]*) printf '[FAIL] Could not parse JOBID.\n' >&2; exit 1;; esac
mkdir -p "$RUN_ROOT/$JOB_ID"
{ printf 'selected_partition=%s\n' "$PARTITION"; printf 'run_root=%s\n' "$RUN_ROOT"; printf '%s\n' "$OUT"; } > "$RUN_ROOT/$JOB_ID/submission.txt"
printf '%s|%s\n' "$JOB_ID" "$RUN_ROOT" > "$JOB_FILE"
printf '[PASS] Saved JOBID %s on %s.\n' "$JOB_ID" "$PARTITION"
if [ "$PARTITION" = gpu_5090 ] && [ "${TMM_DISABLE_FALLBACK:-0}" != 1 ]; then
    WAIT_SECONDS="${TMM_GPU_FALLBACK_WAIT_SECONDS:-600}"
    nohup env TMM_GPU_FALLBACK_WAIT_SECONDS="$WAIT_SECONDS" bash "$SCRIPT_DIR/phase3_identifiability_status.sh" --watch "$JOB_ID" "$RUN_ROOT" > "$RUN_ROOT/$JOB_ID/fallback_monitor.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$RUN_ROOT/$JOB_ID/fallback_monitor.pid"
    printf '[PASS] 4090 fallback monitor started; threshold=%ss.\n' "$WAIT_SECONDS"
fi
