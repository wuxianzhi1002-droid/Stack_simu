#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v12_srt401
#SBATCH --output=v12_srt401_slurm_%j.out
#SBATCH --error=v12_srt401_slurm_%j.err

set -Eeo pipefail
printf '=== V12 fixed-angle StackRT 450-580 nm GPU run ===\n'
printf 'JOBID=%s\nhostname=%s\npwd=%s\nSLURM_SUBMIT_DIR=%s\nscript=%s\nbash=%s\n' \
  "${SLURM_JOB_ID:-<unset>}" "$(hostname)" "$PWD" "${SLURM_SUBMIT_DIR:-<unset>}" "$0" "$BASH_VERSION"

[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || {
  echo '[FAIL] This script must be launched by sbatch.' >&2
  exit 2
}
set -u

PROJECT_DIR="${TMM_V12_PROJECT_DIR:?Set TMM_V12_PROJECT_DIR to gpu_acc project directory}"
INPUT_DIR="${TMM_V12_DATASET:?Set TMM_V12_DATASET to the uploaded StackRT dataset}"
RUN_ROOT="${TMM_V12_RUN_ROOT:-/data/home/$USER/v12_stackrt_optimizer_runs}"
CASE_COUNT="${TMM_V12_CASE_COUNT:-401}"
GLOBAL_MAXITER="${TMM_V12_GLOBAL_MAXITER:-40}"
MAX_NFEV="${TMM_V12_MAX_NFEV:-600}"

[ -d "$PROJECT_DIR" ] || { printf '[FAIL] Project directory not found: %s\n' "$PROJECT_DIR" >&2; exit 2; }
[ -d "$INPUT_DIR" ] || { printf '[FAIL] Dataset directory not found: %s\n' "$INPUT_DIR" >&2; exit 2; }
case "$CASE_COUNT" in
  1|401) ;;
  *) printf '[FAIL] TMM_V12_CASE_COUNT must be 1 or 401, got %s.\n' "$CASE_COUNT" >&2; exit 2 ;;
esac

RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
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
    echo '[PASS] V12 StackRT fixed-angle GPU job completed.'
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

python - "$INPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "simulation_manifest.json").read_text(encoding="utf-8"))
npz_files = sorted(root.glob("static_spectrum_*.npz"))
checks = {
    "version": manifest.get("version") == "main_v12",
    "backend": manifest.get("backend") == "api",
    "dataset_count": manifest.get("dataset_count") == 401,
    "npz_count": len(npz_files) == 401,
    "failures": manifest.get("failures") == [],
    "wavelength_start_nm": manifest["config"]["WAVELENGTH_START_UM"] == 0.45,
    "wavelength_stop_nm": manifest["config"]["WAVELENGTH_STOP_UM"] == 0.58,
    "angle_mode": manifest["angle_contract"]["MODE"] == "fixed_independent_measurement",
}
print(json.dumps({"checks": checks, "manifest_backend": manifest.get("backend")}, indent=2))
if not all(checks.values()):
    raise SystemExit("[FAIL] StackRT dataset manifest gate failed")
PY
printf '[PASS] StackRT dataset gate: backend=api, 401 NPZ, 450-580 nm, failures=0.\n'

cp "$INPUT_DIR/simulation_manifest.json" "$INPUT_DIR/dataset_index.csv" "$RUN_DIR/dataset_audit/"
sha256sum "$INPUT_DIR"/*.npz | sort -k2 > "$RUN_DIR/dataset_audit/npz_sha256.txt"
sha256sum "$INPUT_DIR/simulation_manifest.json" "$INPUT_DIR/dataset_index.csv" \
  > "$RUN_DIR/dataset_audit/metadata_sha256.txt"

printf '[PASS] Angle fixed to measured b; no MAP prior; free N=5; global B=40; local B=11.\n'
RUNNER_ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$RUN_DIR/results"
  --machine "$RUN_DIR/machine_info.json"
  --count "$CASE_COUNT"
  --wavelength-min-nm 450
  --wavelength-max-nm 580
  --global-maxiter "$GLOBAL_MAXITER"
  --max-nfev "$MAX_NFEV"
)
if [ "$CASE_COUNT" -eq 401 ]; then
  RUNNER_ARGS+=(--require-production-count)
fi
python -m v12_optimizer.gpu_runner "${RUNNER_ARGS[@]}"
