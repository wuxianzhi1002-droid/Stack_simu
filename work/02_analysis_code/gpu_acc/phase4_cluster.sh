#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "$0")"&&pwd)"; [ "$#" -eq 1 ]||exit 2; case "$1" in submit) exec bash "$D/submit_phase4.sh";; status) exec bash "$D/phase4_status.sh";; result) exec bash "$D/collect_phase4_result.sh";; cancel) exec bash "$D/cancel_last_phase4_job.sh";; *) exit 2;; esac
