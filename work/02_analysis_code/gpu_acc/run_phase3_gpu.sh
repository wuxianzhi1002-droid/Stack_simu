#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --job-name=v10_phase3_jac
#SBATCH --output=phase3_slurm_%j.out
#SBATCH --error=phase3_slurm_%j.err
set -Eeo pipefail
printf '=== V10 Phase 3 bootstrap ===
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
[ -n "${SLURM_JOB_ID:-}" ] || { printf '[FAIL] Must use sbatch.
' >&2; exit 2; }
[ -n "${SLURM_JOB_PARTITION:-}" ] || { printf '[FAIL] Missing partition.
' >&2; exit 2; }
[ -n "${SLURM_SUBMIT_DIR:-}" ] || { printf '[FAIL] Missing submit dir.
' >&2; exit 2; }
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
RUN_DIR="$PROJECT_DIR/cluster_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
printf 'PROJECT_DIR=%s
' "$PROJECT_DIR"
printf '[PASS] RUN_DIR created: %s
' "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
finish(){ code=$?; printf '%s
' "$code" > "$RUN_DIR/exit_code.txt"; if [ "$code" -eq 0 ]; then printf '[PASS] Phase 3 completed successfully.
'; else printf '[FAIL] Phase 3 exited with code %s.
' "$code" >&2; fi; }
trap finish EXIT
[ "$SLURM_JOB_PARTITION" = gpu_5090 ] || { printf '[FAIL] Expected gpu_5090.
' >&2; exit 2; }
source "$PROJECT_DIR/cluster_modules.sh"
tmm_ensure_cuda_12_8
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
cd "$PROJECT_DIR"
python -m v10_gpu.verify_phase1_freeze | tee "$RUN_DIR/phase1_freeze_validation.json"
python -m v10_gpu.verify_phase2_freeze | tee "$RUN_DIR/phase2_freeze_validation.json"
printf '[PASS] Phase 1/2 freezes verified.
'
[ -f v10_gpu/cpu_reference_phase3/jacobian_reference.npz ] || { printf '[FAIL] Missing Phase 3 reference.
' >&2; exit 2; }
hostname > "$RUN_DIR/hostname.txt"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt"
python -m v10_gpu.cluster_machine_info --output "$RUN_DIR/machine_info.json"
python -m v10_gpu.validate_phase3_reference | tee "$RUN_DIR/reference_validation.json"
printf '[PASS] Phase 3 CPU reference hashes and shape verified.
'
python -m v10_gpu.benchmark_jacobian --gpu-repeat 3 --output "$RUN_DIR/phase3_benchmark.json" | tee "$RUN_DIR/benchmark_console.json"
printf '[PASS] Completed 20-case CPU/GPU Jacobian closure.
'
python -m v10_gpu.phase3_report --machine "$RUN_DIR/machine_info.json" --benchmark "$RUN_DIR/phase3_benchmark.json" --output-json "$RUN_DIR/phase3_result.json" --output-md "$RUN_DIR/phase3_result_summary.md" | tee "$RUN_DIR/result_validation.json"
printf '[PASS] Phase 3 acceptance passed.
'
printf 'scipy least_squares was not changed or connected; JAX and Phase 4 were not started.
'
