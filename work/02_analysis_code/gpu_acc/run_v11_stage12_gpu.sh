#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v11_stage12
#SBATCH --output=v11_stage12_slurm_%j.out
#SBATCH --error=v11_stage12_slurm_%j.err
set -Eeo pipefail
printf '=== V11 Stage 1 ranking + Stage 2 q=theta^2 bootstrap ===\n'
printf 'JOBID=%s\nSLURM_SUBMIT_DIR=%s\npwd=%s\nscript=%s\nbash=%s\n' "${SLURM_JOB_ID:-<unset>}" "${SLURM_SUBMIT_DIR:-<unset>}" "$PWD" "$0" "$BASH_VERSION";hostname
[ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo '[FAIL] sbatch required' >&2;exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}";RUN_ROOT="${TMM_V11_RUN_ROOT:-/vast/$USER/v11_optimizer_runs}";RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/cupy_cache" "$RUN_DIR/tmp" "$RUN_DIR/mpl";printf '[PASS] RUN_DIR=%s\n' "$RUN_DIR"
export CUPY_CACHE_DIR="$RUN_DIR/cupy_cache" TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" MPLCONFIGDIR="$RUN_DIR/mpl"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ c=$?;printf '%s\n' "$c">"$RUN_DIR/exit_code.txt";if [ "$c" -eq 0 ];then echo '[PASS] completed';else echo "[FAIL] exit=$c" >&2;fi;};trap finish EXIT
case "${SLURM_JOB_PARTITION:-}" in gpu_5090|gpu_4090);;*) echo '[FAIL] partition' >&2;exit 2;;esac
source "$PROJECT_DIR/cluster_modules.sh";tmm_ensure_cuda_12_8;tmm_ensure_conda;eval "$(conda shell.bash hook)";conda activate tmm5090;cd "$PROJECT_DIR"
which python;python --version;nvidia-smi|tee "$RUN_DIR/nvidia-smi.txt"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader|wc -l)" -eq 1 ]||{ echo '[FAIL] expected one GPU' >&2;exit 2; }
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
INPUT="${TMM_V11_INPUT_DIR:-$HOME/gpu_acc/04_results_and_datasets/static_stackrt_v10_20260825_232804}"
BASELINE="${TMM_V11_PHASE6_JSON:-$PROJECT_DIR/cluster_runs/1496648/phase6_401npz_results.json}"
[ -d "$INPUT" ]&&[ -f "$BASELINE" ]||{ echo '[FAIL] input or Phase6 JSON missing' >&2;exit 2; }
python -m v11_optimizer.stage12_runner --phase6-json "$BASELINE" --input-dir "$INPUT" --output-dir "$RUN_DIR/results" --machine "$RUN_DIR/machine_info.json" --count 401
