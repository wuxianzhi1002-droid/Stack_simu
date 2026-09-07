#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_4090
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH --job-name=v16_s1b_info
#SBATCH --output=v16_s1b_info_slurm_%j.out
#SBATCH --error=v16_s1b_info_slurm_%j.err
set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${TMM_V16_PROJECT_DIR:?Set TMM_V16_PROJECT_DIR}"
INPUT_DIR="${TMM_V16_MULTIANGLE_DATASET:?Set TMM_V16_MULTIANGLE_DATASET}"
CALIBRATION_DIR="${TMM_V16_CALIBRATION:?Set TMM_V16_CALIBRATION}"
RUN_ROOT="${TMM_V16_RUN_ROOT:-/ssd/$USER/v16_stage1b_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
[ "${SLURM_JOB_PARTITION:-}" = gpu_4090 ] || { echo '[FAIL] gpu_4090 required' >&2; exit 2; }
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache"
export TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"; exit "$code"; }
trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 4090
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python - "$INPUT_DIR" "$CALIBRATION_DIR" <<'PY'
import json,pathlib,sys,numpy as np
root=pathlib.Path(sys.argv[1]);paths=sorted(root.glob('multiangle_*_typical_*.npz'));assert len(paths)==100,len(paths)
with np.load(paths[0],allow_pickle=False) as d:
 assert d['spectra_measured'].shape==(2,30001);assert np.allclose(d['true_reflector_angles_deg'],[0.02,0.2],rtol=0,atol=1e-12);assert str(d['generator_version'])=='main_v16_multiangle_nonzero';config=json.loads(str(d['config_json']));assert config['SOURCE_MODEL']=='eq99x_digitized_peak_normalized'
assert len(sorted(pathlib.Path(sys.argv[2]).glob('multiframe_g*.npz')))==10
print('[PASS] V16 Stage 1B input and calibration gates')
PY
first_npz="$(find "$INPUT_DIR" -maxdepth 1 -type f -name 'multiangle_*_typical_*.npz' | sort | head -n 1)"
sha256sum "$first_npz" "$INPUT_DIR/simulation_manifest.json" > "$RUN_DIR/dataset_audit/source_sha256.txt"
sha256sum "$CALIBRATION_DIR"/multiframe_g*.npz "$CALIBRATION_DIR"/multiframe_manifest.json | sort -k2 > "$RUN_DIR/dataset_audit/calibration_sha256.txt"
python -m v16_optimizer.longwave_information --input-dir "$INPUT_DIR" --noise-calibration-dir "$CALIBRATION_DIR" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json"
