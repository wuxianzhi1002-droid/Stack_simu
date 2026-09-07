#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v9_gpu_301
#SBATCH --output=v9_gpu_slurm_%j.out
#SBATCH --error=v9_gpu_slurm_%j.err
set -Eeo pipefail
printf '=== Formal V9 GPU-accelerated 301-NPZ bootstrap ===\n'
printf 'JOBID=%s\n' "${SLURM_JOB_ID:-<unset>}"
printf 'SLURM_SUBMIT_DIR=%s\n' "${SLURM_SUBMIT_DIR:-<unset>}"
printf 'pwd=%s\nscript_path=%s\nbash_version=%s\n' "$PWD" "$0" "$BASH_VERSION"
hostname
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] Must run through sbatch.\n' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
RUN_ROOT="${TMM_V9_RUN_ROOT:-/vast/$USER/gpu_acc_v9_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/matplotlib_config"
printf '[PASS] RUN_DIR created: %s\n' "$RUN_DIR"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp"
export MPLCONFIGDIR="$RUN_DIR/matplotlib_config"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] V9 GPU production completed.\n'; else printf '[FAIL] V9 GPU production exit=%s.\n' "$code" >&2; fi; }
trap finish EXIT
case "${SLURM_JOB_PARTITION:-}" in gpu_5090|gpu_4090) ;; *) printf '[FAIL] Unsupported partition: %s\n' "${SLURM_JOB_PARTITION:-<unset>}" >&2; exit 2;; esac
printf '[BOOTSTRAP] source cluster_modules.sh: start\n'
source "$PROJECT_DIR/cluster_modules.sh"
printf '[PASS] source cluster_modules.sh: done\n'
printf '[BOOTSTRAP] CUDA 12.8 setup: start\n'
tmm_ensure_cuda_12_8
printf '[PASS] CUDA 12.8 setup: done\n'
printf '[BOOTSTRAP] Miniforge setup: start\n'
tmm_ensure_conda
printf '[PASS] Miniforge setup: done\n'
eval "$(conda shell.bash hook)"
conda activate tmm5090
printf '[PASS] Activated conda environment tmm5090\n'
which python
python --version
which conda
cd "$PROJECT_DIR"
printf '[PASS] Project directory: %s\n' "$PROJECT_DIR"
hostname | tee "$RUN_DIR/hostname.txt"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)"
[ "$VISIBLE_GPU_COUNT" -eq 1 ] || { printf '[FAIL] Expected exactly one visible GPU, got %s.\n' "$VISIBLE_GPU_COUNT" >&2; exit 2; }
printf '[PASS] Exactly one allocated GPU is visible.\n'
python - "$RUN_DIR/machine_info.json" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
import cupy as cp
props = cp.cuda.runtime.getDeviceProperties(0)
name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
info = {
    'hostname': platform.node(), 'gpu_name': name, 'driver_version': driver,
    'cuda_runtime_version': int(cp.cuda.runtime.runtimeGetVersion()),
    'cupy_version': cp.__version__, 'python_version': platform.python_version(),
    'visible_gpu_count': int(cp.cuda.runtime.getDeviceCount()),
}
Path(sys.argv[1]).write_text(json.dumps(info, indent=2), encoding='utf-8')
print(json.dumps(info, ensure_ascii=False))
PY
INPUT_DIR="${TMM_V9_INPUT_DIR:-$HOME/gpu_acc/04_results_and_datasets/static_stackrt_v9_20260819_121305}"
[ -d "$INPUT_DIR" ] || { printf '[FAIL] Missing V9 input directory: %s\n' "$INPUT_DIR" >&2; exit 2; }
NPZ_COUNT="$(find "$INPUT_DIR" -maxdepth 1 -type f -name 'static_spectrum_*.npz' | wc -l)"
[ "$NPZ_COUNT" -eq 301 ] || { printf '[FAIL] Expected exactly 301 V9 NPZ files, found %s in %s.\n' "$NPZ_COUNT" "$INPUT_DIR" >&2; exit 2; }
printf '[PASS] Found exactly 301 V9 NPZ files.\n'
python -m v9_gpu.production_runner \
    --input-dir "$INPUT_DIR" \
    --output-dir "$RUN_DIR/results" \
    --count 301 \
    --machine "$RUN_DIR/machine_info.json"
printf '[PASS] V9 GPU production runner finished.\n'
