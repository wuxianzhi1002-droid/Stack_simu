#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v12_wide_a001
#SBATCH --output=v12_wide_a001_slurm_%j.out
#SBATCH --error=v12_wide_a001_slurm_%j.err

set -Eeo pipefail
printf '=== V12 StackRT 220-580 nm, fixed measured angle sigma=0.001 deg ===\n'
printf 'JOBID=%s\nhostname=%s\npwd=%s\nSLURM_SUBMIT_DIR=%s\nscript=%s\nbash=%s\n' \
  "${SLURM_JOB_ID:-<unset>}" "$(hostname)" "$PWD" "${SLURM_SUBMIT_DIR:-<unset>}" "$0" "$BASH_VERSION"
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || {
  echo '[FAIL] This script must be launched by sbatch.' >&2
  exit 2
}
set -u

PROJECT_DIR="${TMM_V12_PROJECT_DIR:?Set TMM_V12_PROJECT_DIR}"
INPUT_DIR="${TMM_V12_DATASET:?Set TMM_V12_DATASET}"
RUN_ROOT="${TMM_V12_RUN_ROOT:-/ssd/$USER/v12_stackrt_220_sigma0001_runs}"
CASE_COUNT="${TMM_V12_CASE_COUNT:-401}"
GLOBAL_MAXITER="${TMM_V12_GLOBAL_MAXITER:-40}"
MAX_NFEV="${TMM_V12_MAX_NFEV:-600}"

[ -d "$PROJECT_DIR" ] || { printf '[FAIL] Project not found: %s\n' "$PROJECT_DIR" >&2; exit 2; }
[ -d "$INPUT_DIR" ] || { printf '[FAIL] Dataset not found: %s\n' "$INPUT_DIR" >&2; exit 2; }
case "$CASE_COUNT" in 1|401) ;; *) echo '[FAIL] CASE_COUNT must be 1 or 401.' >&2; exit 2 ;; esac

RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
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
  [ "$code" -eq 0 ] && echo '[PASS] V12 wideband sigma=0.001 job completed.' \
    || printf '[FAIL] Job exited with code %s.\n' "$code" >&2
}
trap finish EXIT

case "${SLURM_JOB_PARTITION:-}" in gpu_5090|gpu_4090) ;; *) echo '[FAIL] unsupported partition' >&2; exit 2 ;; esac
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ] || exit 2
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"

python - "$INPUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "simulation_manifest.json").read_text(encoding="utf-8"))
npz_files = sorted(root.glob("static_spectrum_*.npz"))
with (root / "dataset_index.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
with np.load(npz_files[0], allow_pickle=False) as sample:
    required = {
        "measured_reflector_angle_deg", "angle_measurement_sigma_deg",
        "angle_measurement_error_deg", "angle_measurement_mode",
    }
    sample_fields = set(sample.files)
    sample_start = float(sample["reported_wavelengths_nm"][0])
    sample_stop = float(sample["reported_wavelengths_nm"][-1])
    sample_points = len(sample["reported_wavelengths_nm"])
checks = {
    "version_main_v12": manifest.get("version") == "main_v12",
    "backend_stackrt_api": manifest.get("backend") == "api",
    "manifest_count_401": manifest.get("dataset_count") == 401,
    "npz_count_401": len(npz_files) == 401,
    "csv_count_401": len(rows) == 401,
    "failures_empty": manifest.get("failures") == [],
    "wavelength_220_580": sample_start == 220.0 and sample_stop == 580.0,
    "sample_points_18001": sample_points == 18001,
    "angle_mode_fixed": manifest["angle_contract"]["MODE"] == "fixed_independent_measurement",
    "angle_sigma_0001": manifest["angle_contract"]["STANDARD_UNCERTAINTY_DEG"] == 0.001,
    "npz_angle_fields": required.issubset(sample_fields),
    "config_isolated": manifest.get("configuration_isolation", {}).get("depends_on_main_v10_config") is False,
}
print(json.dumps({"checks": checks, "sample": npz_files[0].name}, indent=2))
if not all(checks.values()):
    raise SystemExit("[FAIL] V12 StackRT wideband dataset gate failed")
PY

cp "$INPUT_DIR/simulation_manifest.json" "$INPUT_DIR/dataset_index.csv" "$RUN_DIR/dataset_audit/"
sha256sum "$INPUT_DIR"/*.npz | sort -k2 > "$RUN_DIR/dataset_audit/npz_sha256.txt"
sha256sum "$INPUT_DIR/simulation_manifest.json" "$INPUT_DIR/dataset_index.csv" \
  > "$RUN_DIR/dataset_audit/metadata_sha256.txt"
printf '[PASS] StackRT API, 401 NPZ, 220-580 nm, sigma=0.001, isolated V12 config.\n'

RUNNER_ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$RUN_DIR/results"
  --machine "$RUN_DIR/machine_info.json"
  --count "$CASE_COUNT"
  --wavelength-min-nm 220
  --wavelength-max-nm 580
  --global-maxiter "$GLOBAL_MAXITER"
  --max-nfev "$MAX_NFEV"
)
[ "$CASE_COUNT" -eq 401 ] && RUNNER_ARGS+=(--require-production-count)
python -m v12_optimizer.gpu_runner "${RUNNER_ARGS[@]}"
