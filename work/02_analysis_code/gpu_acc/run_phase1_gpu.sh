#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --job-name=v10_phase1_tmm
#SBATCH --output=phase1_slurm_%j.out
#SBATCH --error=phase1_slurm_%j.err

set -Eeo pipefail

printf '=== V10 Phase 1 bootstrap ===\n'
printf 'JOBID=%s\n' "${SLURM_JOB_ID:-<unset>}"
printf 'SLURM_SUBMIT_DIR=%s\n' "${SLURM_SUBMIT_DIR:-<unset>}"
printf 'hostname='
hostname
printf 'pwd='
pwd
printf 'script_path=%s\n' "$0"
printf 'bash_version=%s\n' "$BASH_VERSION"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    printf '[FAIL] run_phase1_gpu.sh must be launched by sbatch.\n' >&2
    exit 2
fi
if [ -z "${SLURM_JOB_PARTITION:-}" ]; then
    printf '[FAIL] SLURM_JOB_PARTITION is unavailable.\n' >&2
    exit 2
fi
if [ -z "${SLURM_SUBMIT_DIR:-}" ]; then
    printf '[FAIL] SLURM_SUBMIT_DIR is unavailable.\n' >&2
    exit 2
fi

set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
printf 'PROJECT_DIR=%s\n' "$PROJECT_DIR"
[ -d "$PROJECT_DIR" ] || {
    printf '[FAIL] PROJECT_DIR does not exist: %s\n' "$PROJECT_DIR" >&2
    exit 2
}
RUN_DIR="$PROJECT_DIR/cluster_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
printf '[PASS] RUN_DIR created: %s\n' "$RUN_DIR"

exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

finish() {
    code=$?
    printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"
    if [ "$code" -eq 0 ]; then
        printf '[PASS] Phase 1 job completed successfully.\n'
    else
        printf '[FAIL] Phase 1 job exited with code %s.\n' "$code" >&2
    fi
}
trap finish EXIT

printf '=== V10 Phase 1 strict TMM GPU job ===\n'
printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
printf 'PARTITION=%s\n' "$SLURM_JOB_PARTITION"
printf 'RUN_DIR=%s\n' "$RUN_DIR"
date --iso-8601=seconds

[ "$SLURM_JOB_PARTITION" = "gpu_5090" ] || {
    printf '[FAIL] Expected partition gpu_5090, got %s.\n' "$SLURM_JOB_PARTITION" >&2
    exit 2
}
printf '[PASS] Partition is gpu_5090.\n'

MODULE_HELPER="$PROJECT_DIR/cluster_modules.sh"
[ -f "$MODULE_HELPER" ] || {
    printf '[FAIL] Missing cluster_modules.sh: %s\n' "$MODULE_HELPER" >&2
    exit 2
}

printf '[BOOTSTRAP] About to source cluster_modules.sh: %s\n' "$MODULE_HELPER"
source "$MODULE_HELPER"
printf '[PASS] cluster_modules.sh sourced.\n'

printf '[BOOTSTRAP] Preparing CUDA 12.8; preferred module is cuda/12.8.\n'
tmm_ensure_cuda_12_8
printf '[PASS] CUDA 12.8 environment ready.\n'

printf '[BOOTSTRAP] Preparing conda; preferred module is miniforge3/26.3.2-3.\n'
tmm_ensure_conda
printf '[PASS] Miniforge/conda environment ready.\n'

printf '[BOOTSTRAP] Initializing conda shell integration.\n'
eval "$(conda shell.bash hook)"
printf '[PASS] Conda shell integration initialized.\n'

printf '[BOOTSTRAP] Activating tmm5090.\n'
conda activate tmm5090
printf '[PASS] Activated tmm5090.\n'
printf 'which python: '
which python
python --version
printf 'which conda: '
which conda

cd "$PROJECT_DIR"
[ -f "v10_gpu/cpu_reference/forward_reference.npz" ] || {
    printf '[FAIL] Missing CPU reference NPZ.\n' >&2
    exit 2
}
[ -f "v10_gpu/cpu_reference/cpu_benchmark.json" ] || {
    printf '[FAIL] Missing existing CPU benchmark JSON.\n' >&2
    exit 2
}
printf '[PASS] Phase 0 reference and CPU baseline exist.\n'

hostname > "$RUN_DIR/hostname.txt"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
printf '[PASS] Recorded hostname and nvidia-smi.\n'

python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
printf '[PASS] Recorded machine-readable GPU/runtime information.\n'

python -m v10_gpu.validate_reference | tee "$RUN_DIR/reference_validation.log"
printf '[PASS] CPU reference replay passed on the compute node.\n'

python -m v10_gpu.benchmark_forward --backend both --batch-sizes 1 7 13 32 64 --warmup 1 --repeat 3 --output "$RUN_DIR/phase1_benchmark.json" | tee "$RUN_DIR/benchmark_console.json"
printf '[PASS] Completed strict TMM CPU/GPU closure and benchmark.\n'

python -m v10_gpu.cluster_report --machine "$RUN_DIR/machine_info.json" --benchmark "$RUN_DIR/phase1_benchmark.json" --baseline "$PROJECT_DIR/v10_gpu/cpu_reference/cpu_benchmark.json" --output-json "$RUN_DIR/phase1_result.json" --output-md "$RUN_DIR/phase1_result_summary.md" | tee "$RUN_DIR/result_validation.json"

printf '[PASS] Numerical thresholds and one-GPU checks passed.\n'
printf 'Human summary: %s\n' "$RUN_DIR/phase1_result_summary.md"
printf 'Machine JSON: %s\n' "$RUN_DIR/phase1_result.json"
printf 'Phase 2 was not started.\n'
