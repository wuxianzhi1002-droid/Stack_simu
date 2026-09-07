#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAST_JOB_FILE="$SCRIPT_DIR/last_job_id.txt"
BASELINE="$SCRIPT_DIR/v10_gpu/cpu_reference/cpu_benchmark.json"

[ -f "$LAST_JOB_FILE" ] || {
    printf '[FAIL] Missing last_job_id.txt. Submit Phase 1 first.\n' >&2
    exit 1
}

JOB_ID=$(tr -d '[:space:]' < "$LAST_JOB_FILE")
case "$JOB_ID" in
    ''|*[!0-9]*)
        printf '[FAIL] Invalid JOBID in last_job_id.txt.\n' >&2
        exit 1
        ;;
esac

RUN_DIR="$SCRIPT_DIR/cluster_runs/$JOB_ID"
STDOUT_LOG="$RUN_DIR/stdout.log"
STDERR_LOG="$RUN_DIR/stderr.log"
MACHINE_JSON="$RUN_DIR/machine_info.json"
BENCHMARK_JSON="$RUN_DIR/phase1_benchmark.json"
RESULT_JSON="$RUN_DIR/phase1_result.json"
SUMMARY_MD="$RUN_DIR/phase1_result_summary.md"

printf 'JOBID: %s\n' "$JOB_ID"
printf 'stdout: %s\n' "$STDOUT_LOG"
printf 'stderr: %s\n' "$STDERR_LOG"

for REQUIRED in "$MACHINE_JSON" "$BENCHMARK_JSON" "$BASELINE"; do
    [ -f "$REQUIRED" ] || {
        printf '[FAIL] Result is not ready; missing %s\n' "$REQUIRED" >&2
        printf 'Use ./cluster.sh status and retry after the job finishes.\n' >&2
        exit 1
    }
done

MODULE_HELPER="$SCRIPT_DIR/cluster_modules.sh"
[ -f "$MODULE_HELPER" ] || {
    printf '[FAIL] Missing cluster_modules.sh.\n' >&2
    exit 1
}
source "$MODULE_HELPER"
tmm_ensure_conda
eval "$(conda shell.bash hook)"
conda activate tmm5090
PYTHON_CMD=$(command -v python)

set +e
"$PYTHON_CMD" "$SCRIPT_DIR/v10_gpu/cluster_report.py" --machine "$MACHINE_JSON" --benchmark "$BENCHMARK_JSON" --baseline "$BASELINE" --output-json "$RESULT_JSON" --output-md "$SUMMARY_MD"
REPORT_CODE=$?
set -e

printf '%s\n' "=== $SUMMARY_MD ==="
cat "$SUMMARY_MD"
printf 'Machine-readable result: %s\n' "$RESULT_JSON"
printf 'Phase 2 was not started.\n'
exit "$REPORT_CODE"
