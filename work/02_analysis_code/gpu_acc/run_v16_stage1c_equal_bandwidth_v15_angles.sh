#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_4090
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=v16_s1c_v15ang
#SBATCH --output=v16_s1c_v15ang_slurm_%j.out
#SBATCH --error=v16_s1c_v15ang_slurm_%j.err
set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${TMM_V16_PROJECT_DIR:?Set TMM_V16_PROJECT_DIR}"
INPUT_DIR="${TMM_V15_STAGE1_DATASET:?Set TMM_V15_STAGE1_DATASET}"
CALIBRATION_DIR="${TMM_V16_CALIBRATION:?Set TMM_V16_CALIBRATION}"
RUN_ROOT="${TMM_V16_RUN_ROOT:-/ssd/$USER/v16_stage1c_v15_angles_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
[ "${SLURM_JOB_PARTITION:-}" = gpu_4090 ] || { echo '[FAIL] gpu_4090 required' >&2; exit 2; }
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s
' "$code" > "$RUN_DIR/exit_code.txt"; exit "$code"; };trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh";tmm_ensure_cuda_12_8;tmm_ensure_conda;eval "$(conda shell.bash hook)";conda activate tmm5090;cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt";[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ];nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 4090
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python - "$INPUT_DIR" <<'PY'
import pathlib,sys,numpy as np
ps=sorted(pathlib.Path(sys.argv[1]).glob('static_spectrum_*_typical_*.npz'));assert len(ps)==100,len(ps)
angles=[]
for p in ps:
 with np.load(p,allow_pickle=False) as d:
  assert d['reported_wavelengths_nm'].shape==(30001,);assert str(d['generator_version'])=='main_v15';assert str(d['angle_measurement_mode'])=='fixed_independent_measurement';angles.append(float(d['measured_reflector_angle_deg']))
assert np.all(np.isfinite(angles))
print('[PASS] 100 V15 Stage 1 per-NPZ fixed measured angles',min(angles),max(angles))
PY
sha256sum "$INPUT_DIR"/static_spectrum_*_typical_*.npz | sort -k2 > "$RUN_DIR/dataset_audit/source_npz_sha256.txt"
sha256sum "$CALIBRATION_DIR"/multiframe_g*.npz "$CALIBRATION_DIR"/multiframe_manifest.json | sort -k2 > "$RUN_DIR/dataset_audit/calibration_sha256.txt"
python -m v16_optimizer.equal_bandwidth_v15_angles --input-dir "$INPUT_DIR" --noise-calibration-dir "$CALIBRATION_DIR" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json"
