#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
chmod u+x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

if [ "$#" -ne 1 ]; then
    printf 'Usage: %s setup|submit|status|result|cancel\n' "$0" >&2
    exit 2
fi

ACTION="$1"
case "$ACTION" in
    setup)
        exec "$SCRIPT_DIR/setup_cluster_env.sh"
        ;;
    submit)
        exec "$SCRIPT_DIR/submit_phase1.sh"
        ;;
    status)
        exec "$SCRIPT_DIR/cluster_status.sh"
        ;;
    result)
        exec "$SCRIPT_DIR/collect_phase1_result.sh"
        ;;
    cancel)
        exec "$SCRIPT_DIR/cancel_last_job.sh"
        ;;
    *)
        printf 'Unknown action: %s\n' "$ACTION" >&2
        printf 'Usage: %s setup|submit|status|result|cancel\n' "$0" >&2
        exit 2
        ;;
esac
