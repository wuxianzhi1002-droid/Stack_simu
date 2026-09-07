#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
chmod u+x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
[ "$#" -eq 1 ] || { printf 'Usage: %s submit|status|result|cancel
' "$0" >&2; exit 2; }
case "$1" in
 submit) exec "$SCRIPT_DIR/submit_phase2.sh";;
 status) exec "$SCRIPT_DIR/phase2_status.sh";;
 result) exec "$SCRIPT_DIR/collect_phase2_result.sh";;
 cancel) exec "$SCRIPT_DIR/cancel_last_phase2_job.sh";;
 *) printf 'Usage: %s submit|status|result|cancel
' "$0" >&2; exit 2;;
esac
