#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v11_wideband
#SBATCH --output=v11_wideband_slurm_%j.out
#SBATCH --error=v11_wideband_slurm_%j.err
set -Eeo pipefail
printf '=== V11 matched-bandwidth diagnostics bootstrap ===\n'
printf 'JOBID=%s\nSLURM_SUBMIT_DIR=%s\npwd=%s\nscript=%s\nbash=%s\n' "${SLURM_JOB_ID:-<unset>}" "${SLURM_SUBMIT_DIR:-<unset>}" "$PWD" "$0" "$BASH_VERSION"
hostname
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
RUN_ROOT="${TMM_V11_RUN_ROOT:-/vast/$USER/v11_optimizer_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl"
printf '[PASS] RUN_DIR=%s\n' "$RUN_DIR"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ c=$?; printf '%s\n' "$c" > "$RUN_DIR/exit_code.txt"; if [ "$c" -eq 0 ]; then echo '[PASS] completed'; else echo "[FAIL] exit=$c" >&2; fi; }
trap finish EXIT
case "${SLURM_JOB_PARTITION:-}" in gpu_5090|gpu_4090);; *) echo '[FAIL] unsupported GPU partition' >&2; exit 2;; esac
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"
which python
python --version
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ] || { echo '[FAIL] expected one GPU' >&2; exit 2; }
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
DATA450="${TMM_V11_DATASET_450:?TMM_V11_DATASET_450 required}"
DATA350="${TMM_V11_DATASET_350:?TMM_V11_DATASET_350 required}"
DATA220="${TMM_V11_DATASET_220:?TMM_V11_DATASET_220 required}"
PHASE6="${TMM_V11_PHASE6_JSON:-$PROJECT_DIR/cluster_runs/1496648/phase6_401npz_results.json}"
if [ ! -f "$PHASE6" ] && [ -f "$PROJECT_DIR/phase6_401npz_results_1496648.json" ]; then PHASE6="$PROJECT_DIR/phase6_401npz_results_1496648.json"; fi
for target in "$DATA450" "$DATA350" "$DATA220"; do [ -d "$target" ] || { echo "[FAIL] missing dataset $target" >&2; exit 2; }; done
[ -f "$PHASE6" ] || { echo "[FAIL] missing Phase6 JSON $PHASE6" >&2; exit 2; }
python -m v11_optimizer.wideband_runner \
  --dataset-450 "$DATA450" \
  --dataset-350 "$DATA350" \
  --dataset-220 "$DATA220" \
  --phase6-json "$PHASE6" \
  --output-dir "$RUN_DIR/results" \
  --machine "$RUN_DIR/machine_info.json"
