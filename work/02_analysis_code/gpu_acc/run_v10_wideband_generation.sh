#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v10_wideband_data
#SBATCH --output=v10_wideband_slurm_%j.out
#SBATCH --error=v10_wideband_slurm_%j.err
set -Eeo pipefail
printf '=== V10 matched wideband dataset generation ===\nJOBID=%s\nSLURM_SUBMIT_DIR=%s\npwd=%s\nscript=%s\nbash=%s\n' "${SLURM_JOB_ID:-<unset>}" "${SLURM_SUBMIT_DIR:-<unset>}" "$PWD" "$0" "$BASH_VERSION";hostname
[ -n "${SLURM_JOB_ID:-}" ]&&[ -n "${SLURM_SUBMIT_DIR:-}" ]||exit 2
set -u;PROJECT_DIR="${SLURM_SUBMIT_DIR:?}";RUN_DIR="/vast/$USER/v11_optimizer_runs/$SLURM_JOB_ID";mkdir -p "$RUN_DIR" "$RUN_DIR/mpl";export MPLCONFIGDIR="$RUN_DIR/mpl"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ c=$?;printf '%s\n' "$c">"$RUN_DIR/exit_code.txt";};trap finish EXIT
source "$PROJECT_DIR/cluster_modules.sh";tmm_ensure_conda;eval "$(conda shell.bash hook)";conda activate tmm5090
GEN="$PROJECT_DIR/wideband_generator/main_v10.py";test -f "$GEN"
ROOT="/vast/$USER/wideband_datasets"
mkdir -p "$ROOT";STAMP="$(date +%Y%m%d_%H%M%S)"
D350="$ROOT/static_stackrt_v10_wideband_350_580_$STAMP";D220="$ROOT/static_stackrt_v10_wideband_220_580_$STAMP"
printf '{"job_id":"%s","dataset_350_580":"%s","dataset_220_580":"%s"}\n' "$SLURM_JOB_ID" "$D350" "$D220" > "$RUN_DIR/wideband_paths.json"
python "$GEN" --backend tmm --cases all --repeats 10 --clean-repeats 1 --seed 20260825 --wavelength-start-nm 350 --wavelength-stop-nm 580 --output-sampling-nm 0.02 --output-dir "$D350"
python "$GEN" --backend tmm --cases all --repeats 10 --clean-repeats 1 --seed 20260825 --wavelength-start-nm 220 --wavelength-stop-nm 580 --output-sampling-nm 0.02 --output-dir "$D220"
for d in "$D350" "$D220";do n="$(find "$d" -maxdepth 1 -type f -name 'static_spectrum_*.npz'|wc -l)";[ "$n" -eq 401 ]||{ echo "[FAIL] $d count=$n" >&2;exit 2;};du -sh "$d";done
sha256sum "$GEN">"$RUN_DIR/generator_sha256.txt";echo '[PASS] two matched 401-NPZ wideband datasets generated'
