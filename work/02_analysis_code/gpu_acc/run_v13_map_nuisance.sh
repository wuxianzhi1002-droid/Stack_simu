#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v13_map
#SBATCH --output=v13_map_slurm_%j.out
#SBATCH --error=v13_map_slurm_%j.err

set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${TMM_V13_PROJECT_DIR:?Set TMM_V13_PROJECT_DIR}"
INPUT_DIR="${TMM_V13_DATASET:?Set TMM_V13_DATASET}"
NUISANCE="${TMM_V13_NUISANCE:?Set TMM_V13_NUISANCE}"
RUN_ROOT="${TMM_V13_RUN_ROOT:-/ssd/$USER/v13_map_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
CASE_COUNT="${TMM_V13_CASE_COUNT:-100}"
case "$CASE_COUNT" in 1|100) ;; *) echo '[FAIL] CASE_COUNT must be 1 or 100' >&2; exit 2 ;; esac
case "$NUISANCE" in axis_offset_nm|axis_scale_ppm|source_center_drift_nm) ;; *) echo '[FAIL] bad nuisance' >&2; exit 2 ;; esac
case "${SLURM_JOB_PARTITION:-}" in gpu_5090|gpu_4090) ;; *) echo '[FAIL] unsupported partition' >&2; exit 2 ;; esac
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; exit "$code"; }; trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh"; tmm_ensure_cuda_12_8; tmm_ensure_conda
eval "$(conda shell.bash hook)"; conda activate tmm5090; cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ] || exit 2
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python - "$INPUT_DIR" <<'PY'
import pathlib,sys,numpy as np
root=pathlib.Path(sys.argv[1]); paths=sorted(root.glob('static_spectrum_*_typical_*.npz'))
assert len(paths)==100, len(paths)
with np.load(paths[0],allow_pickle=False) as d:
 assert float(d['reported_wavelengths_nm'][0])==220.0
 assert float(d['reported_wavelengths_nm'][-1])==580.0
 assert len(d['reported_wavelengths_nm'])==18001
 assert str(d['generator_version'])=='main_v12'
 assert float(d['angle_measurement_sigma_deg'])==0.001
print('[PASS] 100 typical StackRT V12 files; 220-580 nm; 18001 points; angle sigma=0.001 deg')
PY
sha256sum "$INPUT_DIR"/static_spectrum_*_typical_*.npz | sort -k2 > "$RUN_DIR/dataset_audit/typical_npz_sha256.txt"
RUNNER_ARGS=(--input-dir "$INPUT_DIR" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json"
  --nuisance "$NUISANCE" --count "$CASE_COUNT" --global-maxiter "${TMM_V13_GLOBAL_MAXITER:-40}"
  --max-nfev "${TMM_V13_MAX_NFEV:-600}")
[ "$CASE_COUNT" -eq 100 ] && RUNNER_ARGS+=(--require-typical-count)
[ -n "${TMM_V13_NOISE_CASE:-}" ] && RUNNER_ARGS+=(--noise-case "$TMM_V13_NOISE_CASE")
python -m v13_optimizer.nuisance_runner "${RUNNER_ARGS[@]}"
