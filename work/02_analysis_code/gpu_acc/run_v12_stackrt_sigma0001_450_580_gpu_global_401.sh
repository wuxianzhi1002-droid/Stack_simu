#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v12_a001
#SBATCH --output=v12_a001_slurm_%j.out
#SBATCH --error=v12_a001_slurm_%j.err

set -Eeo pipefail
printf '=== V12 StackRT 450-580 nm, angle sigma 0.001 deg ===\n'
printf 'JOBID=%s\nhostname=%s\npwd=%s\nSLURM_SUBMIT_DIR=%s\nscript=%s\nbash=%s\n' \
  "${SLURM_JOB_ID:-<unset>}" "$(hostname)" "$PWD" "${SLURM_SUBMIT_DIR:-<unset>}" "$0" "$BASH_VERSION"
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || {
  echo '[FAIL] This script must be launched by sbatch.' >&2
  exit 2
}
set -u

PROJECT_DIR="${TMM_V12_PROJECT_DIR:?Set TMM_V12_PROJECT_DIR}"
CODE_OVERLAY="${TMM_V12_CODE_OVERLAY:?Set TMM_V12_CODE_OVERLAY}"
SOURCE_DATASET="${TMM_V12_SOURCE_DATASET:?Set TMM_V12_SOURCE_DATASET}"
ANGLE_OVERRIDES="${TMM_V12_ANGLE_OVERRIDES:?Set TMM_V12_ANGLE_OVERRIDES}"
TARGET_MANIFEST="${TMM_V12_TARGET_MANIFEST:?Set TMM_V12_TARGET_MANIFEST}"
RUN_ROOT="${TMM_V12_RUN_ROOT:-/data/home/$USER/v12_stackrt_sigma0001_runs}"
CASE_COUNT="${TMM_V12_CASE_COUNT:-401}"
GLOBAL_MAXITER="${TMM_V12_GLOBAL_MAXITER:-40}"
MAX_NFEV="${TMM_V12_MAX_NFEV:-600}"

for path in "$PROJECT_DIR" "$CODE_OVERLAY" "$SOURCE_DATASET"; do
  [ -d "$path" ] || { printf '[FAIL] Directory not found: %s\n' "$path" >&2; exit 2; }
done
for path in "$ANGLE_OVERRIDES" "$TARGET_MANIFEST"; do
  [ -f "$path" ] || { printf '[FAIL] File not found: %s\n' "$path" >&2; exit 2; }
done
case "$CASE_COUNT" in 1|401) ;; *) echo '[FAIL] CASE_COUNT must be 1 or 401.' >&2; exit 2 ;; esac

RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp"
export TEMP="$RUN_DIR/tmp"
export TMP="$RUN_DIR/tmp"
export MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$CODE_OVERLAY:$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

finish() {
  code=$?
  printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"
  [ "$code" -eq 0 ] && echo '[PASS] V12 sigma=0.001 deg job completed.' \
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

python - "$SOURCE_DATASET" "$ANGLE_OVERRIDES" "$TARGET_MANIFEST" <<'PY'
import csv
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
overrides = pathlib.Path(sys.argv[2])
target_manifest = pathlib.Path(sys.argv[3])
source_manifest = json.loads((source / "simulation_manifest.json").read_text(encoding="utf-8"))
target = json.loads(target_manifest.read_text(encoding="utf-8"))
npz_names = {path.name for path in source.glob("static_spectrum_*.npz")}
with overrides.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
override_names = {row["npz_path"].replace("\\", "/").rsplit("/", 1)[-1] for row in rows}
sigmas = {float(row["angle_measurement_sigma_deg"]) for row in rows}
checks = {
    "source_backend_api": source_manifest.get("backend") == "api",
    "source_count_401": len(npz_names) == 401,
    "source_sigma_0010": source_manifest["angle_contract"]["STANDARD_UNCERTAINTY_DEG"] == 0.01,
    "target_backend_api": target.get("backend") == "api",
    "target_count_401": target.get("dataset_count") == 401,
    "target_failures_empty": target.get("failures") == [],
    "target_sigma_0001": target["angle_contract"]["STANDARD_UNCERTAINTY_DEG"] == 0.001,
    "override_count_401": len(rows) == 401 and len(override_names) == 401,
    "override_filenames_match": override_names == npz_names,
    "override_sigma_0001": sigmas == {0.001},
}
print(json.dumps({"checks": checks, "override_sigmas": sorted(sigmas)}, indent=2))
if not all(checks.values()):
    raise SystemExit("[FAIL] sigma=0.001 dataset overlay gate failed")
PY

cp "$TARGET_MANIFEST" "$RUN_DIR/dataset_audit/target_simulation_manifest.json"
cp "$ANGLE_OVERRIDES" "$RUN_DIR/dataset_audit/target_dataset_index.csv"
cp "$SOURCE_DATASET/simulation_manifest.json" "$RUN_DIR/dataset_audit/source_simulation_manifest.json"
sha256sum "$TARGET_MANIFEST" "$ANGLE_OVERRIDES" "$SOURCE_DATASET/simulation_manifest.json" \
  > "$RUN_DIR/dataset_audit/metadata_sha256.txt"
sha256sum "$SOURCE_DATASET"/*.npz | sort -k2 > "$RUN_DIR/dataset_audit/source_npz_sha256.txt"
printf '[PASS] Exact-spectrum source + sigma=0.001 measured-angle overlay gate passed.\n'

RUNNER_ARGS=(
  --input-dir "$SOURCE_DATASET"
  --angle-overrides-csv "$ANGLE_OVERRIDES"
  --output-dir "$RUN_DIR/results"
  --machine "$RUN_DIR/machine_info.json"
  --count "$CASE_COUNT"
  --wavelength-min-nm 450
  --wavelength-max-nm 580
  --global-maxiter "$GLOBAL_MAXITER"
  --max-nfev "$MAX_NFEV"
)
[ "$CASE_COUNT" -eq 401 ] && RUNNER_ARGS+=(--require-production-count)
cd "$CODE_OVERLAY"
python -m v12_optimizer.gpu_runner "${RUNNER_ARGS[@]}"
