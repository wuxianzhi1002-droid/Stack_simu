#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --job-name=v10_phase2_ils
#SBATCH --output=phase2_slurm_%j.out
#SBATCH --error=phase2_slurm_%j.err

set -Eeo pipefail
printf '=== V10 Phase 2 bootstrap ===
'
printf 'JOBID=%s
' "${SLURM_JOB_ID:-<unset>}"
printf 'SLURM_SUBMIT_DIR=%s
' "${SLURM_SUBMIT_DIR:-<unset>}"
printf 'hostname='; hostname
printf 'pwd='; pwd
printf 'script_path=%s
' "$0"
printf 'bash_version=%s
' "$BASH_VERSION"
[ -n "${SLURM_JOB_ID:-}" ] || { printf '[FAIL] Must be launched by sbatch.
' >&2; exit 2; }
[ -n "${SLURM_JOB_PARTITION:-}" ] || { printf '[FAIL] SLURM_JOB_PARTITION unavailable.
' >&2; exit 2; }
[ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] SLURM_SUBMIT_DIR unavailable.
' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unavailable}"
RUN_DIR="$PROJECT_DIR/cluster_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
printf 'PROJECT_DIR=%s
' "$PROJECT_DIR"
printf '[PASS] RUN_DIR created: %s
' "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s
' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] Phase 2 completed successfully.
'; else printf '[FAIL] Phase 2 exited with code %s.
' "$code" >&2; fi; }
trap finish EXIT
[ "$SLURM_JOB_PARTITION" = gpu_5090 ] || { printf '[FAIL] Expected gpu_5090.
' >&2; exit 2; }
printf '[PASS] Partition is gpu_5090.
'
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
printf '[PASS] CUDA environment ready.
'
tmm_ensure_conda
printf '[PASS] Conda environment ready.
'
eval "$(conda shell.bash hook)"
conda activate tmm5090
printf 'which python: '; which python
python --version
cd "$PROJECT_DIR"
python -m v10_gpu.verify_phase1_freeze | tee "$RUN_DIR/phase1_freeze_validation.json"
printf '[PASS] Phase 1 freeze and timing audit verified.\n'
[ -f v10_gpu/cpu_reference_phase2/spectrometer_reference.npz ] || { printf '[FAIL] Missing Phase 2 reference.
' >&2; exit 2; }
[ -f v10_gpu/cpu_reference_phase2/cpu_benchmark.json ] || { printf '[FAIL] Missing Phase 2 CPU baseline.
' >&2; exit 2; }
hostname > "$RUN_DIR/hostname.txt"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python -m v10_gpu.validate_phase2_reference | tee "$RUN_DIR/reference_validation.log"
printf '[PASS] Phase 2 CPU reference replay passed.
'
python -m v10_gpu.benchmark_spectrometer --backend both --batch-sizes 1 7 13 32 64 --warmup 1 --repeat 3 --output "$RUN_DIR/phase2_benchmark.json" | tee "$RUN_DIR/benchmark_console.json"
printf '[PASS] Phase 2 CPU/GPU response closure and benchmark completed.
'
python -m v10_gpu.phase2_report --machine "$RUN_DIR/machine_info.json" --benchmark "$RUN_DIR/phase2_benchmark.json" --baseline "$PROJECT_DIR/v10_gpu/cpu_reference_phase2/cpu_benchmark.json" --output-json "$RUN_DIR/phase2_result.json" --output-md "$RUN_DIR/phase2_result_summary.md" | tee "$RUN_DIR/result_validation.json"
printf '[PASS] Phase 2 acceptance passed.
'
printf 'Phase 3/Jacobian, scipy least_squares changes and JAX were not started.
'
