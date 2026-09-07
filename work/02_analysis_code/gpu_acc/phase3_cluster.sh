#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; chmod u+x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
[ "$#" -eq 1 ] || { printf 'Usage: %s submit|status|result|cancel
' "$0" >&2; exit 2; }
case "$1" in submit) exec "$SCRIPT_DIR/submit_phase3.sh";; status) exec "$SCRIPT_DIR/phase3_status.sh";; result) exec "$SCRIPT_DIR/collect_phase3_result.sh";; cancel) exec "$SCRIPT_DIR/cancel_last_phase3_job.sh";; *) exit 2;; esac
