#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gpu_global_b48
#SBATCH --output=gpu_global_b48_slurm_%j.out
#SBATCH --error=gpu_global_b48_slurm_%j.err
set -Eeo pipefail
printf '=== GPU population-batched full-ILS global benchmark ===\n'
printf 'JOBID=%s\nhostname=%s\npwd=%s\nSLURM_SUBMIT_DIR=%s\nscript=%s\nbash=%s\n' "${SLURM_JOB_ID:-<unset>}" "$(hostname)" "$PWD" "${SLURM_SUBMIT_DIR:-<unset>}" "$0" "$BASH_VERSION"
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
RUN_ROOT="${TMM_V11_RUN_ROOT:-/vast/$USER/v11_optimizer_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl"
printf '[PASS] RUN_DIR=%s\n' "$RUN_DIR"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then echo '[PASS] completed'; else echo "[FAIL] exit=$code" >&2; fi; }
trap finish EXIT
case "${SLURM_JOB_PARTITION:-}" in gpu_5090|gpu_4090);; *) echo '[FAIL] unsupported partition' >&2; exit 2;; esac
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"
which python
python --version
which conda
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ] || { echo '[FAIL] expected one GPU' >&2; exit 2; }
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
INPUT="${TMM_GPU_GLOBAL_INPUT:?TMM_GPU_GLOBAL_INPUT required}"
[ -f "$INPUT" ] || { echo "[FAIL] missing input $INPUT" >&2; exit 2; }
case "$(basename "$INPUT")" in *clean_r0000*) ;; *) echo '[FAIL] benchmark input must be clean_r0000' >&2; exit 2;; esac
printf '[PASS] one clean NPZ; full_ils=0.002 nm; population=48; vectorized=true; updating=deferred\n'
python -m v11_optimizer.gpu_global_batch_benchmark --input "$INPUT" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json"
