#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v10_phase6_diag
#SBATCH --output=phase6_slurm_%j.out
#SBATCH --error=phase6_slurm_%j.err
set -Eeo pipefail
printf '=== V10 Phase 6 single-case CPU/GPU convergence diagnostic ===\n'
printf 'JOBID=%s\nSLURM_SUBMIT_DIR=%s\n' "${SLURM_JOB_ID:-<unset>}" "${SLURM_SUBMIT_DIR:-<unset>}"; hostname; pwd; printf 'script=%s\n' "$0"
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] Must run through sbatch.\n' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"; RUN_ROOT="${TMM_CLUSTER_RUN_ROOT:-/vast/$USER/gpu_acc_phase6_runs}"; RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/matplotlib_config"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/matplotlib_config"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] Phase 6 completed.\n'; else printf '[FAIL] Phase 6 exit=%s.\n' "$code" >&2; fi; }; trap finish EXIT
case "$SLURM_JOB_PARTITION" in gpu_5090|gpu_4090) ;; *) printf '[FAIL] Unsupported partition.\n' >&2; exit 2;; esac
source "$PROJECT_DIR/cluster_modules.sh"; tmm_ensure_cuda_12_8; tmm_ensure_conda; eval "$(conda shell.bash hook)"; conda activate tmm5090; cd "$PROJECT_DIR"
python -m v10_gpu.verify_phase1_freeze | tee "$RUN_DIR/phase1_freeze_validation.json"
python -m v10_gpu.verify_phase2_freeze | tee "$RUN_DIR/phase2_freeze_validation.json"
python -m v10_gpu.verify_phase3_freeze | tee "$RUN_DIR/phase3_freeze_validation.json"
python -m v10_gpu.verify_phase3_diagnostics_freeze | tee "$RUN_DIR/phase3_diagnostics_freeze_validation.json"
python -m v10_gpu.verify_phase5_freeze | tee "$RUN_DIR/phase5_freeze_validation.json"
hostname > "$RUN_DIR/hostname.txt"; nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"; python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
CASE="$HOME/gpu_acc/04_results_and_datasets/static_stackrt_v10_20260825_232804/static_spectrum_angle_in_spec_r0005_seed22260830.npz"
[ -f "$CASE" ] || { printf '[FAIL] Missing diagnostic input: %s\n' "$CASE" >&2; exit 2; }
python -m v10_gpu.phase6_case_diagnostic --input "$CASE" --seed 20306230 --output "$RUN_DIR/phase6_case46_cpu_gpu_diagnostic.json" | tee "$RUN_DIR/phase6_case46_console.json"
printf '[PASS] Phase 6 case-46 CPU/GPU convergence diagnostic finished.\n'
