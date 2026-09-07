#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v10_wideband_220
#SBATCH --output=v10_wideband_220_slurm_%j.out
#SBATCH --error=v10_wideband_220_slurm_%j.err
set -Eeo pipefail
printf '=== V10 matched 220-580 nm dataset generation ===\n'
printf 'JOBID=%s\nSLURM_SUBMIT_DIR=%s\npwd=%s\nscript=%s\nbash=%s\n' "${SLURM_JOB_ID:-<unset>}" "${SLURM_SUBMIT_DIR:-<unset>}" "$PWD" "$0" "$BASH_VERSION"
hostname
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
RUN_DIR="/vast/$USER/v11_optimizer_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/mpl"
export MPLCONFIGDIR="$RUN_DIR/mpl"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ c=$?; printf '%s\n' "$c" > "$RUN_DIR/exit_code.txt"; if [ "$c" -eq 0 ]; then echo '[PASS] completed'; else echo "[FAIL] exit=$c" >&2; fi; }
trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
GEN="$PROJECT_DIR/wideband_generator/main_v10.py"
[ -f "$GEN" ] || { echo '[FAIL] generator missing' >&2; exit 2; }
ROOT="/vast/$USER/wideband_datasets"
mkdir -p "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
DATASET="$ROOT/static_stackrt_v10_wideband_220_580_$STAMP"
printf '{"job_id":"%s","dataset_220_580":"%s"}\n' "$SLURM_JOB_ID" "$DATASET" > "$RUN_DIR/wideband_220_path.json"
python "$GEN" --backend tmm --cases all --repeats 10 --clean-repeats 1 --seed 20260825 --wavelength-start-nm 220 --wavelength-stop-nm 580 --output-sampling-nm 0.02 --output-dir "$DATASET"
count="$(find "$DATASET" -maxdepth 1 -type f -name 'static_spectrum_*.npz' | wc -l)"
[ "$count" -eq 401 ] || { echo "[FAIL] dataset count=$count" >&2; exit 2; }
du -sh "$DATASET"
sha256sum "$GEN" > "$RUN_DIR/generator_sha256.txt"
echo '[PASS] matched 401-NPZ 220-580 nm dataset generated'
