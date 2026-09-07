#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v11_220_gpu401
#SBATCH --output=v11_220_gpu401_slurm_%j.out
#SBATCH --error=v11_220_gpu401_slurm_%j.err

set -Eeo pipefail
printf '=== V11 220-580 nm fresh GPU-global 401-case run ===\n'
printf 'JOBID=%s\nhostname=%s\npwd=%s\nSLURM_SUBMIT_DIR=%s\nscript=%s\nbash=%s\n' \
  "${SLURM_JOB_ID:-<unset>}" "$(hostname)" "$PWD" "${SLURM_SUBMIT_DIR:-<unset>}" "$0" "$BASH_VERSION"

[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || {
  echo '[FAIL] This script must be launched by sbatch.' >&2
  exit 2
}
set -u

PROJECT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
RUN_ROOT="${TMM_V11_RUN_ROOT:-/vast/$USER/v11_optimizer_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl"
printf '[PASS] RUN_DIR created: %s\n' "$RUN_DIR"

export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp"
export TEMP="$RUN_DIR/tmp"
export TMP="$RUN_DIR/tmp"
export MPLCONFIGDIR="$RUN_DIR/mpl"

exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

finish() {
  code=$?
  printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"
  if [ "$code" -eq 0 ]; then
    echo '[PASS] V11 220-580 nm 401-case job completed.'
  else
    printf '[FAIL] Job exited with code %s.\n' "$code" >&2
  fi
}
trap finish EXIT

case "${SLURM_JOB_PARTITION:-}" in
  gpu_5090|gpu_4090) ;;
  *) printf '[FAIL] Unsupported partition: %s\n' "${SLURM_JOB_PARTITION:-<unset>}" >&2; exit 2 ;;
esac

printf '[BOOTSTRAP] Sourcing cluster_modules.sh from %s\n' "$PROJECT_DIR"
source "$PROJECT_DIR/cluster_modules.sh"
printf '[PASS] cluster_modules.sh sourced.\n'

printf '[BOOTSTRAP] Loading CUDA 12.8.\n'
tmm_ensure_cuda_12_8
printf '[PASS] CUDA bootstrap complete.\n'

printf '[BOOTSTRAP] Loading Miniforge/conda.\n'
tmm_ensure_conda
printf '[PASS] Miniforge/conda bootstrap complete.\n'

eval "$(conda shell.bash hook)"
conda activate tmm5090
printf '[PASS] Activated conda environment tmm5090.\n'
which python
python --version
which conda

cd "$PROJECT_DIR"
printf '[PASS] Working directory: %s\n' "$PWD"

nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "$GPU_COUNT" -eq 1 ] || {
  printf '[FAIL] Expected exactly one visible GPU, found %s.\n' "$GPU_COUNT" >&2
  exit 2
}
printf '[PASS] Exactly one allocated GPU is visible.\n'

python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"

INPUT_DIR="${TMM_WIDEBAND_DATASET:?TMM_WIDEBAND_DATASET is required}"
[ -d "$INPUT_DIR" ] || {
  printf '[FAIL] Dataset directory not found: %s\n' "$INPUT_DIR" >&2
  exit 2
}
NPZ_COUNT="$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l)"
[ "$NPZ_COUNT" -eq 401 ] || {
  printf '[FAIL] Expected 401 NPZ files, found %s in %s.\n' "$NPZ_COUNT" "$INPUT_DIR" >&2
  exit 2
}
printf '[PASS] Dataset contract: 401 NPZ files at %s\n' "$INPUT_DIR"
printf '[PASS] Fresh per-case DE; full_ils; population B=48; vectorized=true; updating=deferred; multistarts=8.\n'
printf '[PASS] No old population/start reuse; no fast_no_ils/coarse-ILS/JAX/autodiff/multi-GPU.\n'

python -m v11_optimizer.wideband_gpu_global_401_runner \
  --input-dir "$INPUT_DIR" \
  --output-dir "$RUN_DIR/results" \
  --machine "$RUN_DIR/machine_info.json" \
  --count 401
