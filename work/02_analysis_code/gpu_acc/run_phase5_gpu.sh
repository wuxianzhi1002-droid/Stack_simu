#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v10_phase5_10
#SBATCH --output=phase5_slurm_%j.out
#SBATCH --error=phase5_slurm_%j.err
set -Eeo pipefail
printf '=== V10 Phase 5 deterministic 10-NPZ bootstrap ===\n'
printf 'JOBID=%s\nSLURM_SUBMIT_DIR=%s\n' "${SLURM_JOB_ID:-<unset>}" "${SLURM_SUBMIT_DIR:-<unset>}"; hostname; pwd; printf 'script=%s\n' "$0"
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] Must run through sbatch.\n' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"; RUN_ROOT="${TMM_CLUSTER_RUN_ROOT:-/vast/$USER/gpu_acc_phase5_runs}"; RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/matplotlib_config"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/matplotlib_config"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] Phase 5 completed.\n'; else printf '[FAIL] Phase 5 exit=%s.\n' "$code" >&2; fi; }; trap finish EXIT
case "$SLURM_JOB_PARTITION" in gpu_5090|gpu_4090) ;; *) printf '[FAIL] Unsupported partition.\n' >&2; exit 2;; esac
source "$PROJECT_DIR/cluster_modules.sh"; tmm_ensure_cuda_12_8; tmm_ensure_conda; eval "$(conda shell.bash hook)"; conda activate tmm5090; cd "$PROJECT_DIR"
python -m v10_gpu.verify_phase1_freeze | tee "$RUN_DIR/phase1_freeze_validation.json"
python -m v10_gpu.verify_phase2_freeze | tee "$RUN_DIR/phase2_freeze_validation.json"
python -m v10_gpu.verify_phase3_freeze | tee "$RUN_DIR/phase3_freeze_validation.json"
python -m v10_gpu.verify_phase3_diagnostics_freeze | tee "$RUN_DIR/phase3_diagnostics_freeze_validation.json"
hostname > "$RUN_DIR/hostname.txt"; nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"; python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
INPUT_DIR="${TMM_PHASE5_INPUT_DIR:-$HOME/gpu_acc/04_results_and_datasets/static_stackrt_v10_20260825_232804}"
CPU_BASELINE="${TMM_PHASE5_CPU_BASELINE:-$PROJECT_DIR/cluster_runs/1482539/phase4_result.json}"
[ -d "$INPUT_DIR" ] || { printf '[FAIL] Missing input directory: %s\n' "$INPUT_DIR" >&2; exit 2; }
[ -f "$CPU_BASELINE" ] || { printf '[FAIL] Missing CPU baseline: %s\n' "$CPU_BASELINE" >&2; exit 2; }
export TMM_PHASE5_PROCESS_START_EPOCH_NS="$(date +%s%N)"
set +e
python -m v10_gpu.phase5_runner --input-dir "$INPUT_DIR" --output-dir "$RUN_DIR" --machine "$RUN_DIR/machine_info.json" --backend gpu --count 10 --global-forward-model fast_no_ils --global-popsize 8 --global-maxiter 40 --multistarts 8 --max-nfev 600 --random-seed 20260825 --cpu-baseline "$CPU_BASELINE" | tee "$RUN_DIR/phase5_console.json"
PHASE5_CODE=${PIPESTATUS[0]}
set -e
[ "$PHASE5_CODE" -eq 0 ] || exit "$PHASE5_CODE"
printf '[PASS] Deterministic first-10 Phase 5 finished; 401-NPZ run was not started.\n'
