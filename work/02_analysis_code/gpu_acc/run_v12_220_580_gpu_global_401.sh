#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v12_fixed401
#SBATCH --output=v12_fixed401_slurm_%j.out
#SBATCH --error=v12_fixed401_slurm_%j.err

set -Eeo pipefail
printf '=== V12 fixed-angle 220-580 nm GPU production ===\n'
printf 'JOBID=%s\nhostname=%s\npwd=%s\nSLURM_SUBMIT_DIR=%s\nscript=%s\nbash=%s\n' \
  "${SLURM_JOB_ID:-<unset>}" "$(hostname)" "$PWD" "${SLURM_SUBMIT_DIR:-<unset>}" "$0" "$BASH_VERSION"

[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || {
  echo '[FAIL] This script must be launched by sbatch.' >&2
  exit 2
}
set -u

PROJECT_DIR="${TMM_V12_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}"
[ -d "$PROJECT_DIR" ] || {
  printf '[FAIL] Project directory not found: %s\n' "$PROJECT_DIR" >&2
  exit 2
}
RUN_ROOT="${TMM_V12_RUN_ROOT:-/data/home/$USER/v12_optimizer_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl"
printf '[PASS] RUN_DIR created: %s\n' "$RUN_DIR"

export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp"
export TEMP="$RUN_DIR/tmp"
export TMP="$RUN_DIR/tmp"
export MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

finish() {
  code=$?
  printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"
  if [ "$code" -eq 0 ]; then
    echo '[PASS] V12 fixed-angle 401-case job completed.'
  else
    printf '[FAIL] Job exited with code %s.\n' "$code" >&2
  fi
}
trap finish EXIT

case "${SLURM_JOB_PARTITION:-}" in
  gpu_5090|gpu_4090) ;;
  *) printf '[FAIL] Unsupported partition: %s\n' "${SLURM_JOB_PARTITION:-<unset>}" >&2; exit 2 ;;
esac

source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
printf '[PASS] Activated tmm5090.\n'
python --version

cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "$GPU_COUNT" -eq 1 ] || {
  printf '[FAIL] Expected exactly one visible GPU, found %s.\n' "$GPU_COUNT" >&2
  exit 2
}
printf '[PASS] Exactly one allocated GPU is visible.\n'

python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"

CASE_COUNT="${TMM_V12_CASE_COUNT:-401}"
GLOBAL_MAXITER="${TMM_V12_GLOBAL_MAXITER:-40}"
MAX_NFEV="${TMM_V12_MAX_NFEV:-600}"
case "$CASE_COUNT" in
  1|401) ;;
  *) printf '[FAIL] TMM_V12_CASE_COUNT must be 1 or 401, got %s.\n' "$CASE_COUNT" >&2; exit 2 ;;
esac

if [ -n "${TMM_V12_DATASET:-}" ]; then
  INPUT_DIR="$TMM_V12_DATASET"
  [ -d "$INPUT_DIR" ] || {
    printf '[FAIL] Dataset directory not found: %s\n' "$INPUT_DIR" >&2
    exit 2
  }
  printf '[PASS] Using explicit dataset: %s\n' "$INPUT_DIR"
else
  NODE_DATA_ROOT="${SLURM_TMPDIR:-/tmp/$USER/v12_$SLURM_JOB_ID}"
  mkdir -p "$NODE_DATA_ROOT"
  INPUT_DIR="$NODE_DATA_ROOT/static_stackrt_v12_220_580"
  GENERATOR="$PROJECT_DIR/wideband_generator/main_v12.py"
  [ -f "$GENERATOR" ] || {
    printf '[FAIL] V12 generator not found: %s\n' "$GENERATOR" >&2
    exit 2
  }
  if [ "$CASE_COUNT" -eq 401 ]; then
    CASES=all
    REPEATS=10
  else
    CASES=clean
    REPEATS=1
  fi
  printf '[BOOTSTRAP] Generating %s V12 case(s) in node-local storage: %s\n' "$CASE_COUNT" "$INPUT_DIR"
  python "$GENERATOR" \
    --backend tmm \
    --cases "$CASES" \
    --repeats "$REPEATS" \
    --clean-repeats 1 \
    --seed 20260831 \
    --wavelength-start-nm 220 \
    --wavelength-stop-nm 580 \
    --output-sampling-nm 0.02 \
    --output-dir "$INPUT_DIR"
  mkdir -p "$RUN_DIR/dataset_audit"
  cp "$INPUT_DIR/simulation_manifest.json" "$INPUT_DIR/dataset_index.csv" "$RUN_DIR/dataset_audit/"
  find "$INPUT_DIR" -maxdepth 1 -type f -name 'static_spectrum_*.npz' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$RUN_DIR/dataset_audit/npz_sha256.txt"
  sha256sum "$GENERATOR" > "$RUN_DIR/dataset_audit/generator_sha256.txt"
fi
NPZ_COUNT="$(find "$INPUT_DIR" -maxdepth 1 -type f -name 'static_spectrum_*.npz' | wc -l)"
[ "$NPZ_COUNT" -eq "$CASE_COUNT" ] || {
  printf '[FAIL] Expected %s V12 NPZ files, found %s.\n' "$CASE_COUNT" "$NPZ_COUNT" >&2
  exit 2
}
printf '[PASS] V12 dataset contract: %s NPZ files.\n' "$NPZ_COUNT"
printf '[PASS] Angle fixed to measured b; no MAP prior; free N=5; global B=40; local B=11.\n'

RUNNER_ARGS=(
  --input-dir "$INPUT_DIR" \
  --output-dir "$RUN_DIR/results" \
  --machine "$RUN_DIR/machine_info.json" \
  --count "$CASE_COUNT" \
  --global-maxiter "$GLOBAL_MAXITER" \
  --max-nfev "$MAX_NFEV"
)
if [ "$CASE_COUNT" -eq 401 ]; then
  RUNNER_ARGS+=(--require-production-count)
fi
python -m v12_optimizer.gpu_runner "${RUNNER_ARGS[@]}"
