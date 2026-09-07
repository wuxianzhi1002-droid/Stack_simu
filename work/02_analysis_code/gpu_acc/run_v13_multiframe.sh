#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v13_frames
#SBATCH --output=v13_frames_slurm_%j.out
#SBATCH --error=v13_frames_slurm_%j.err

set -Eeo pipefail
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${TMM_V13_PROJECT_DIR:?Set TMM_V13_PROJECT_DIR}"
INPUT_DIR="${TMM_V13_DATASET:?Set TMM_V13_DATASET}"
RUN_ROOT="${TMM_V13_RUN_ROOT:-/ssd/$USER/v13_multiframe_runs}"
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
CASE_COUNT="${TMM_V13_CASE_COUNT:-10}"
case "$CASE_COUNT" in 1|10) ;; *) echo '[FAIL] CASE_COUNT must be 1 or 10' >&2; exit 2 ;; esac
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
import json,pathlib,sys,numpy as np
root=pathlib.Path(sys.argv[1]); paths=sorted(root.glob('multiframe_g*.npz'))
assert len(paths)==10, len(paths)
manifest=json.loads((root/'multiframe_manifest.json').read_text(encoding='utf-8'))
assert manifest['wavelength_nm']==[220.0,580.0]
with np.load(paths[0],allow_pickle=False) as d:
 assert d['spectra_measured'].shape==(16,18001)
 assert float(d['reported_wavelengths_nm'][0])==220.0 and float(d['reported_wavelengths_nm'][-1])==580.0
 assert 'StackRT' in str(d['multiframe_provenance'])
print('[PASS] 10 StackRT-derived groups; 16 frames; 220-580 nm; 18001 points')
PY
sha256sum "$INPUT_DIR"/multiframe_g*.npz "$INPUT_DIR/multiframe_manifest.json" | sort -k2 > "$RUN_DIR/dataset_audit/sha256.txt"
python -m v13_optimizer.multiframe_runner --input-dir "$INPUT_DIR" --output-dir "$RUN_DIR/results" \
  --machine "$RUN_DIR/machine_info.json" --count "$CASE_COUNT" --global-maxiter "${TMM_V13_GLOBAL_MAXITER:-40}" \
  --max-nfev "${TMM_V13_MAX_NFEV:-600}"
