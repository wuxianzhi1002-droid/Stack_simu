#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; JOB_FILE="$SCRIPT_DIR/last_phase5_job_id.txt"; PARTITION="${TMM_GPU_PARTITION:-gpu_5090}"; RUN_ROOT="${TMM_CLUSTER_RUN_ROOT:-/vast/$USER/gpu_acc_phase5_runs}"
case "$PARTITION" in gpu_5090) CPUS_PER_TASK=8 ;; gpu_4090) CPUS_PER_TASK=6 ;; *) printf '[FAIL] Unsupported partition %s\n' "$PARTITION" >&2; exit 2;; esac
mkdir -p "$RUN_ROOT"; cd "$SCRIPT_DIR"
set +e; OUT=$(sbatch --gpus=1 -p "$PARTITION" --cpus-per-task="$CPUS_PER_TASK" --export=ALL,TMM_CLUSTER_RUN_ROOT="$RUN_ROOT" --output="$RUN_ROOT/phase5_slurm_%j.out" --error="$RUN_ROOT/phase5_slurm_%j.err" ./run_phase5_gpu.sh 2>&1); CODE=$?; set -e
printf '%s\n' "$OUT"; [ "$CODE" -eq 0 ] || exit "$CODE"; JOB_ID=$(printf '%s\n' "$OUT"|awk '/Submitted batch job/{print $4;exit}'); case "$JOB_ID" in ''|*[!0-9]*) exit 2;; esac
mkdir -p "$RUN_ROOT/$JOB_ID"; printf '%s|%s\n' "$JOB_ID" "$RUN_ROOT" > "$JOB_FILE"; printf 'partition=%s\nrun_root=%s\n%s\n' "$PARTITION" "$RUN_ROOT" "$OUT" > "$RUN_ROOT/$JOB_ID/submission.txt"
printf '[PASS] Phase 5 JOBID %s on %s with %s CPU cores.\n' "$JOB_ID" "$PARTITION" "$CPUS_PER_TASK"
if [ "$PARTITION" = gpu_5090 ] && [ "${TMM_DISABLE_FALLBACK:-0}" != 1 ]; then WAIT="${TMM_GPU_FALLBACK_WAIT_SECONDS:-600}"; nohup env TMM_GPU_FALLBACK_WAIT_SECONDS="$WAIT" bash "$SCRIPT_DIR/phase5_status.sh" --watch "$JOB_ID" "$RUN_ROOT" > "$RUN_ROOT/$JOB_ID/fallback_monitor.log" 2>&1 < /dev/null & printf '%s\n' "$!" > "$RUN_ROOT/$JOB_ID/fallback_monitor.pid"; printf '[PASS] 4090 fallback threshold=%ss.\n' "$WAIT"; fi
