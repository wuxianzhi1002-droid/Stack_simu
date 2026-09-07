#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "$0")"&&pwd)";case "${1:-}" in submit)exec bash "$D/submit_phase5.sh";;status)exec bash "$D/phase5_status.sh";;result)exec bash "$D/collect_phase5_result.sh";;cancel)exec bash "$D/cancel_last_phase5_job.sh";;*)printf 'Usage: %s {submit|status|result|cancel}\n' "$0";exit 2;;esac
