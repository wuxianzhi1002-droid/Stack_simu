#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tmm_band_jac
#SBATCH --output=band_sensitivity_slurm_%j.out
#SBATCH --error=band_sensitivity_slurm_%j.err

set -Eeo pipefail
printf '=== TMM 220-580 vs 450-580 GPU Jacobian sensitivity ===\n'
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
    echo '[PASS] Band-sensitivity diagnostic completed.'
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
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
printf '[PASS] Activated tmm5090.\n'
which python
python --version
which conda

cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "$GPU_COUNT" -eq 1 ] || {
  printf '[FAIL] Expected exactly one visible GPU, found %s.\n' "$GPU_COUNT" >&2
  exit 2
}
printf '[PASS] Exactly one allocated GPU is visible.\n'
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"

DATASET_220="${TMM_DATASET_220:?TMM_DATASET_220 is required}"
DATASET_450="${TMM_DATASET_450:?TMM_DATASET_450 is required}"
for dataset in "$DATASET_220" "$DATASET_450"; do
  [ -d "$dataset" ] || { printf '[FAIL] Missing dataset: %s\n' "$dataset" >&2; exit 2; }
  count="$(find "$dataset" -maxdepth 1 -type f -name '*.npz' | wc -l)"
  [ "$count" -eq 401 ] || { printf '[FAIL] Expected 401 NPZ files in %s, found %s.\n' "$dataset" "$count" >&2; exit 2; }
done
printf '[PASS] Matched dataset file-count contract: 401 and 401 NPZ files.\n'
printf '[PASS] Diagnostic contract: same physical points; strict full ILS; GPU B=13; no optimizer.\n'

python -m v10_gpu.band_sensitivity \
  --dataset-220 "$DATASET_220" \
  --dataset-450 "$DATASET_450" \
  --output-dir "$RUN_DIR/results" \
  --machine "$RUN_DIR/machine_info.json"
