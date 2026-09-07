#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --job-name=v10_phase4_one
#SBATCH --output=phase4_slurm_%j.out
#SBATCH --error=phase4_slurm_%j.err
set -Eeo pipefail
printf '=== V10 Phase 4 single-NPZ bootstrap ===\n'
printf 'JOBID=%s\n' "${SLURM_JOB_ID:-<unset>}"; printf 'SLURM_SUBMIT_DIR=%s\n' "${SLURM_SUBMIT_DIR:-<unset>}"; hostname; pwd; printf 'script=%s\n' "$0"
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] Must run through sbatch.\n' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"; RUN_ROOT="${TMM_CLUSTER_RUN_ROOT:-$PROJECT_DIR/cluster_runs}"; RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/matplotlib_config"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/matplotlib_config"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] Phase 4 completed.\n'; else printf '[FAIL] Phase 4 exit=%s.\n' "$code" >&2; fi; }; trap finish EXIT
case "$SLURM_JOB_PARTITION" in gpu_5090|gpu_4090) ;; *) printf '[FAIL] Unsupported partition.\n' >&2; exit 2;; esac
source "$PROJECT_DIR/cluster_modules.sh"; tmm_ensure_cuda_12_8; tmm_ensure_conda; eval "$(conda shell.bash hook)"; conda activate tmm5090; cd "$PROJECT_DIR"
python -m v10_gpu.verify_phase1_freeze | tee "$RUN_DIR/phase1_freeze_validation.json"
python -m v10_gpu.verify_phase2_freeze | tee "$RUN_DIR/phase2_freeze_validation.json"
python -m v10_gpu.verify_phase3_freeze | tee "$RUN_DIR/phase3_freeze_validation.json"
python -m v10_gpu.verify_phase3_diagnostics_freeze | tee "$RUN_DIR/phase3_diagnostics_freeze_validation.json"
hostname > "$RUN_DIR/hostname.txt"; nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"; python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
INPUT_NPZ="${TMM_PHASE4_INPUT:-$HOME/gpu_acc/04_results_and_datasets/static_stackrt_v10_20260825_232804/static_spectrum_clean_r0000_seed20260825.npz}"
[ -f "$INPUT_NPZ" ] || { printf '[FAIL] Missing input: %s\n' "$INPUT_NPZ" >&2; exit 2; }
set +e
python -m v10_gpu.phase4_optimizer --input "$INPUT_NPZ" --machine "$RUN_DIR/machine_info.json" --output-dir "$RUN_DIR" | tee "$RUN_DIR/phase4_console.json"
PHASE4_CODE=${PIPESTATUS[0]}
set -e
if [ -f "$RUN_DIR/phase4_result.json" ] && [ -f "$RUN_DIR/phase4_result_summary.md" ]; then
  python -m v10_gpu.phase4_report_audit --result "$RUN_DIR/phase4_result.json" --summary "$RUN_DIR/phase4_result_summary.md" --identifiability "$PROJECT_DIR/cluster_runs/1482351/phase3_identifiability.json"
  printf '[PASS] Phase 4 audit fields written.\n'
fi
[ "$PHASE4_CODE" -eq 0 ] || exit "$PHASE4_CODE"
printf '[PASS] Single-NPZ Phase 4 finished; no dataset batch or later phase started.\n'
