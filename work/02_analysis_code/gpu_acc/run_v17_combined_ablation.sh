#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_4090
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --job-name=v17_ablation
#SBATCH --output=v17_ablation_slurm_%j.out
#SBATCH --error=v17_ablation_slurm_%j.err
set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${TMM_V17_PROJECT_DIR:?Set TMM_V17_PROJECT_DIR}"
INPUT_DIR="${TMM_V17_ABLATION_DATASET:?Set TMM_V17_ABLATION_DATASET}"
CALIBRATION_DIR="${TMM_V17_CALIBRATION:?Set TMM_V17_CALIBRATION}"
RUN_ROOT="${TMM_V17_RUN_ROOT:-/ssd/$USER/v17_ablation_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
CASE_COUNT="${TMM_V17_CASE_COUNT:-100}"
case "$CASE_COUNT" in 1|100);; *) echo '[FAIL] count must be 1 or 100' >&2; exit 2;; esac
[ "${SLURM_JOB_PARTITION:-}" = gpu_4090 ] || { echo '[FAIL] gpu_4090 required' >&2; exit 2; }
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; exit "$code"; }; trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh"
if tmm_require_module_command; then
  tmm_ensure_cuda_12_8
  tmm_ensure_conda
  eval "$(conda shell.bash hook)"
  conda activate tmm5090
else
  echo '[BOOTSTRAP] Falling back to audited absolute CUDA and Conda paths.'
  export CUDA_HOME=/data/apps/cuda/12.8
  export PATH="$CUDA_HOME/bin:/data/home/scxl838/.conda/envs/tmm5090/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  source /data/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh
  conda activate /data/home/scxl838/.conda/envs/tmm5090
  nvcc --version | tail -n 1
  python -c "import cupy; print('[PASS] CuPy', cupy.__version__)"
fi
cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 4090
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python - "$INPUT_DIR" "$CALIBRATION_DIR" <<'PY'
import collections,json,pathlib,sys,numpy as np
root=pathlib.Path(sys.argv[1]); paths=sorted(root.glob('*__static_spectrum_combined_typical_r*.npz')); assert len(paths)==100,len(paths)
counts=collections.Counter()
for p in paths:
 with np.load(p,allow_pickle=False) as d:
  scenario=str(d['ablation_scenario']);counts[scenario]+=1
  axis=np.asarray(d['reported_wavelengths_nm']);assert axis.shape==(20001,) and axis[0]==200.0 and axis[-1]==600.0
  angle=float(d['true_reflector_angle_deg']);assert 0.1<=angle<=0.2
  expected=0.001 if scenario=='reduce_angle_measurement' else 0.01;assert abs(float(d['angle_measurement_sigma_deg'])-expected)<1e-15
  assert str(d['generator_version'])=='main_v15' and str(d['optical_backend'])=='api'
assert len(counts)==10 and set(counts.values())=={10},counts
cal=sorted(pathlib.Path(sys.argv[2]).glob('multiframe_g*.npz'));assert len(cal)==10
for p in cal:
 with np.load(p,allow_pickle=False) as d: assert d.files
print('[PASS] V17 StackRT ablation matrix and calibration gates',counts)
PY
sha256sum "$INPUT_DIR"/*.npz | sort -k2 > "$RUN_DIR/dataset_audit/source_npz_sha256.txt"
sha256sum "$CALIBRATION_DIR"/multiframe_g*.npz "$CALIBRATION_DIR"/multiframe_manifest.json | sort -k2 > "$RUN_DIR/dataset_audit/calibration_sha256.txt"
ARGS=(--input-dir "$INPUT_DIR" --noise-calibration-dir "$CALIBRATION_DIR" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json" --count "$CASE_COUNT" --global-maxiter "${TMM_V17_GLOBAL_MAXITER:-40}" --max-nfev "${TMM_V17_MAX_NFEV:-600}")
[ "$CASE_COUNT" -eq 100 ] && ARGS+=(--require-production-count)
python -m v17_optimizer.ablation_runner "${ARGS[@]}"
