#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_4090
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=v14_band
#SBATCH --output=v14_band_slurm_%j.out
#SBATCH --error=v14_band_slurm_%j.err

set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || {
  echo '[FAIL] This script must be submitted with sbatch.' >&2
  exit 2
}
set -u

PROJECT_DIR="${TMM_V14_PROJECT_DIR:?Set TMM_V14_PROJECT_DIR}"
INPUT_DIR="${TMM_V14_DATASET:?Set TMM_V14_DATASET}"
CALIBRATION_DIR="${TMM_V14_CALIBRATION:?Set TMM_V14_CALIBRATION}"
BAND="${TMM_V14_BAND:?Set TMM_V14_BAND}"
RUN_ROOT="${TMM_V14_RUN_ROOT:-/ssd/$USER/v14_stage1_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
CASE_COUNT="${TMM_V14_CASE_COUNT:-100}"

case "$BAND" in
  220-580|200-600|200-650|200-700|200-800) ;;
  *) echo '[FAIL] Unsupported Stage 1 band.' >&2; exit 2 ;;
esac
case "$CASE_COUNT" in 1|100) ;; *) echo '[FAIL] CASE_COUNT must be 1 or 100.' >&2; exit 2 ;; esac
[ "${SLURM_JOB_PARTITION:-}" = 'gpu_4090' ] || {
  echo '[FAIL] V14 Stage 1 is locked to gpu_4090.' >&2
  exit 2
}

mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish() { code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; exit "$code"; }
trap finish EXIT

source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"

nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ] || exit 2
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '4090' || {
  echo '[FAIL] Allocated GPU is not an RTX 4090.' >&2
  exit 2
}
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"

python - "$INPUT_DIR" "$CALIBRATION_DIR" <<'PY'
import pathlib, sys, numpy as np
source = pathlib.Path(sys.argv[1])
calibration = pathlib.Path(sys.argv[2])
paths = sorted(source.glob('static_spectrum_*_typical_*.npz'))
assert len(paths) == 100, len(paths)
for path in paths:
    with np.load(path, allow_pickle=False) as data:
        axis = np.asarray(data['reported_wavelengths_nm'])
        assert axis.shape == (30001,)
        assert float(axis[0]) == 200.0 and float(axis[-1]) == 800.0
        assert str(data['generator_version']) == 'main_v12'
        assert str(data['optical_backend']) == 'api'
        assert float(data['angle_measurement_sigma_deg']) == 0.001
groups = sorted(calibration.glob('multiframe_g*.npz'))
assert len(groups) == 10, len(groups)
with np.load(groups[0], allow_pickle=False) as data:
    assert data['spectra_measured'].shape == (16, 30001)
    assert 'StackRT' in str(data['multiframe_provenance'])
print('[PASS] 100 paired StackRT typical files and 10x16 detector calibration frames; 200-800 nm')
PY

sha256sum "$INPUT_DIR"/static_spectrum_*_typical_*.npz | sort -k2 > "$RUN_DIR/dataset_audit/source_npz_sha256.txt"
sha256sum "$CALIBRATION_DIR"/multiframe_g*.npz "$CALIBRATION_DIR"/multiframe_manifest.json \
  | sort -k2 > "$RUN_DIR/dataset_audit/calibration_sha256.txt"

RUNNER_ARGS=(
  --input-dir "$INPUT_DIR"
  --noise-calibration-dir "$CALIBRATION_DIR"
  --output-dir "$RUN_DIR/results"
  --machine "$RUN_DIR/machine_info.json"
  --band "$BAND"
  --count "$CASE_COUNT"
  --global-maxiter "${TMM_V14_GLOBAL_MAXITER:-40}"
  --max-nfev "${TMM_V14_MAX_NFEV:-600}"
)
[ "$CASE_COUNT" -eq 100 ] && RUNNER_ARGS+=(--require-production-count)
[ -n "${TMM_V14_NOISE_CASE:-}" ] && RUNNER_ARGS+=(--noise-case "$TMM_V14_NOISE_CASE")
python -m v14_stage1.band_runner "${RUNNER_ARGS[@]}"
