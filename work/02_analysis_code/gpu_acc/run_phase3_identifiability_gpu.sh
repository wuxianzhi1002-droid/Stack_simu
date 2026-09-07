#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --job-name=v10_p3_ident
#SBATCH --output=phase3_identifiability_slurm_%j.out
#SBATCH --error=phase3_identifiability_slurm_%j.err
set -Eeo pipefail
printf '=== V10 Phase 3 identifiability bootstrap ===\n'
printf 'JOBID=%s\n' "${SLURM_JOB_ID:-<unset>}"
printf 'SLURM_SUBMIT_DIR=%s\n' "${SLURM_SUBMIT_DIR:-<unset>}"
printf 'hostname='; hostname
printf 'pwd='; pwd
printf 'script_path=%s\n' "$0"
printf 'bash_version=%s\n' "$BASH_VERSION"
[ -n "${SLURM_JOB_ID:-}" ] || { printf '[FAIL] Must use sbatch.\n' >&2; exit 2; }
[ -n "${SLURM_JOB_PARTITION:-}" ] || { printf '[FAIL] Missing partition.\n' >&2; exit 2; }
[ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] Missing submit dir.\n' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
RUN_ROOT="${TMM_CLUSTER_RUN_ROOT:-$PROJECT_DIR/cluster_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp"
export TEMP="$RUN_DIR/tmp"
export TMP="$RUN_DIR/tmp"
printf 'PROJECT_DIR=%s\n' "$PROJECT_DIR"
printf 'CUPY_CACHE_DIR=%s\n' "$CUPY_CACHE_DIR"
printf 'TMPDIR=%s\n' "$TMPDIR"
printf '[PASS] RUN_DIR created: %s\n' "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] Phase 3 identifiability diagnostics completed.\n'; else printf '[FAIL] Diagnostics exited with code %s.\n' "$code" >&2; fi; }
trap finish EXIT
case "$SLURM_JOB_PARTITION" in
    gpu_5090|gpu_4090) printf '[PASS] Supported GPU partition: %s\n' "$SLURM_JOB_PARTITION";;
    *) printf '[FAIL] Expected gpu_5090 or gpu_4090.\n' >&2; exit 2;;
esac
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"
export MPLCONFIGDIR="$RUN_DIR/matplotlib_config"
python -m v10_gpu.verify_phase1_freeze | tee "$RUN_DIR/phase1_freeze_validation.json"
python -m v10_gpu.verify_phase2_freeze | tee "$RUN_DIR/phase2_freeze_validation.json"
python -m v10_gpu.verify_phase3_freeze | tee "$RUN_DIR/phase3_freeze_validation.json"
printf '[PASS] Phase 1/2/3 freezes verified.\n'
hostname > "$RUN_DIR/hostname.txt"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python -m v10_gpu.validate_phase3_reference | tee "$RUN_DIR/reference_validation.json"
python -m v10_gpu.phase3_identifiability --machine "$RUN_DIR/machine_info.json" --output-dir "$RUN_DIR" | tee "$RUN_DIR/identifiability_console.json"
[ -f "$RUN_DIR/phase3_identifiability.json" ] || { printf '[FAIL] Missing diagnostic JSON.\n' >&2; exit 2; }
[ -f "$RUN_DIR/phase3_identifiability_summary.md" ] || { printf '[FAIL] Missing diagnostic summary.\n' >&2; exit 2; }
[ -f "$RUN_DIR/phase3_identifiability_jacobians.npz" ] || { printf '[FAIL] Missing Jacobian archive.\n' >&2; exit 2; }
printf '[PASS] Machine-readable data, summary, Jacobian archive and figures generated.\n'
printf 'No scipy least_squares, JAX, autodiff or Phase 4 execution.\n'
