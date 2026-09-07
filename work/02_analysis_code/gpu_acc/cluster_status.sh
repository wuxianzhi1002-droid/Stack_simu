#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAST_JOB_FILE="$SCRIPT_DIR/last_job_id.txt"

printf '=== parajobs ===\n'
if command -v parajobs >/dev/null 2>&1; then
    parajobs || true
else
    printf '[INFO] parajobs is unavailable in the current shell.\n'
fi

if [ ! -f "$LAST_JOB_FILE" ]; then
    printf '[INFO] No last_job_id.txt yet.\n'
    exit 0
fi

JOB_ID=$(tr -d '[:space:]' < "$LAST_JOB_FILE")
case "$JOB_ID" in
    ''|*[!0-9]*)
        printf '[INFO] last_job_id.txt does not contain a valid numeric JOBID.\n'
        exit 0
        ;;
esac

RUN_DIR="$SCRIPT_DIR/cluster_runs/$JOB_ID"
SLURM_OUT="$SCRIPT_DIR/phase1_slurm_$JOB_ID.out"
SLURM_ERR="$SCRIPT_DIR/phase1_slurm_$JOB_ID.err"
printf '=== Last JOBID: %s ===\n' "$JOB_ID"
printf 'Run directory: %s\n' "$RUN_DIR"

for LOG_FILE in "$SLURM_OUT" "$SLURM_ERR" "$RUN_DIR/stdout.log" "$RUN_DIR/stderr.log"; do
    if [ -f "$LOG_FILE" ]; then
        printf '%s\n' "=== Last 30 lines: $LOG_FILE ==="
        tail -n 30 "$LOG_FILE" || true
    else
        printf '[INFO] Log not available yet: %s\n' "$LOG_FILE"
    fi
done

exit 0
