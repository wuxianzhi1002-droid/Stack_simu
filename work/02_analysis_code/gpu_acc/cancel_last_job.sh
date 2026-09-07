#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAST_JOB_FILE="$SCRIPT_DIR/last_job_id.txt"

[ -f "$LAST_JOB_FILE" ] || {
    printf '[FAIL] Missing last_job_id.txt.\n' >&2
    exit 1
}
JOB_ID=$(tr -d '[:space:]' < "$LAST_JOB_FILE")
case "$JOB_ID" in
    ''|*[!0-9]*)
        printf '[FAIL] Invalid JOBID in last_job_id.txt.\n' >&2
        exit 1
        ;;
esac

printf 'JOBID to cancel: %s\n' "$JOB_ID"
command -v scancel >/dev/null 2>&1 || {
    printf '[FAIL] scancel is unavailable.\n' >&2
    exit 1
}
scancel "$JOB_ID"
printf '[PASS] scancel was issued for JOBID %s.\n' "$JOB_ID"

printf '=== parajobs confirmation ===\n'
if command -v parajobs >/dev/null 2>&1; then
    parajobs || true
else
    printf '[INFO] parajobs is unavailable in the current shell.\n'
fi
