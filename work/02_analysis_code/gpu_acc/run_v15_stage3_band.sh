#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_4090
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --job-name=v15_stage3
#SBATCH --output=v15_stage3_slurm_%j.out
#SBATCH --error=v15_stage3_slurm_%j.err
set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${TMM_V15_PROJECT_DIR:?Set TMM_V15_PROJECT_DIR}"
INPUT_DIR="${TMM_V15_MULTIANGLE_DATASET:?Set TMM_V15_MULTIANGLE_DATASET}"
CALIBRATION_DIR="${TMM_V15_CALIBRATION:?Set TMM_V15_CALIBRATION}"
BAND="${TMM_V15_BAND:?Set TMM_V15_BAND}"
RUN_ROOT="${TMM_V15_RUN_ROOT:-/ssd/$USER/v15_stage3_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID";CASE_COUNT="${TMM_V15_CASE_COUNT:-100}"
case "$BAND" in 220-580|200-800);; *) echo '[FAIL] unsupported band' >&2;exit 2;;esac
case "$CASE_COUNT" in 1|100);; *) echo '[FAIL] count must be 1 or 100' >&2;exit 2;;esac
[ "${SLURM_JOB_PARTITION:-}" = gpu_4090 ] || { echo '[FAIL] gpu_4090 required' >&2;exit 2; }
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl" "$RUN_DIR/dataset_audit"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache";export TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?;printf '%s\n' "$code">"$RUN_DIR/exit_code.txt";exit "$code";};trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh";tmm_ensure_cuda_12_8;tmm_ensure_conda;eval "$(conda shell.bash hook)";conda activate tmm5090;cd "$PROJECT_DIR"
nvidia-smi|tee "$RUN_DIR/nvidia-smi.txt";[ "$(nvidia-smi --query-gpu=name --format=csv,noheader|wc -l)" -eq 1 ];nvidia-smi --query-gpu=name --format=csv,noheader|grep -q 4090
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python - "$INPUT_DIR" "$CALIBRATION_DIR" <<'PY'
import json,pathlib,sys,numpy as np
root=pathlib.Path(sys.argv[1]);paths=sorted(root.glob('multiangle_*_typical_*.npz'));assert len(paths)==100,len(paths)
for p in paths:
 with np.load(p,allow_pickle=False) as d:
  assert d['spectra_measured'].shape==(2,30001);assert np.array_equal(d['true_reflector_angles_deg'],[0.,0.2]);assert str(d['generator_version'])=='main_v15_multiangle';c=json.loads(str(d['config_json']));assert c['SOURCE_MODEL']=='eq99x_digitized_peak_normalized' and c['EQ99X_SOURCE_CSV_SHA256']
groups=sorted(pathlib.Path(sys.argv[2]).glob('multiframe_g*.npz'));assert len(groups)==10
print('[PASS] V15 EQ-99X two-angle StackRT dataset and calibration gates')
PY
sha256sum "$INPUT_DIR"/multiangle_*_typical_*.npz|sort -k2>"$RUN_DIR/dataset_audit/source_npz_sha256.txt"
sha256sum "$CALIBRATION_DIR"/multiframe_g*.npz "$CALIBRATION_DIR"/multiframe_manifest.json|sort -k2>"$RUN_DIR/dataset_audit/calibration_sha256.txt"
ARGS=(--input-dir "$INPUT_DIR" --noise-calibration-dir "$CALIBRATION_DIR" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json" --band "$BAND" --count "$CASE_COUNT" --global-maxiter "${TMM_V15_GLOBAL_MAXITER:-40}" --max-nfev "${TMM_V15_MAX_NFEV:-600}")
[ "$CASE_COUNT" -eq 100 ]&&ARGS+=(--require-production-count)
python -m v15_optimizer.multiangle_runner "${ARGS[@]}"
