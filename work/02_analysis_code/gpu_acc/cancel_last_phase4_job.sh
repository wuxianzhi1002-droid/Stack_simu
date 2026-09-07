#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "$0")"&&pwd)"; F="$D/last_phase4_job_id.txt"; [ -f "$F" ]||exit 1; IFS='|' read -r J R < "$F"; printf 'Phase 4 JOBID: %s\n' "$J"; scancel "$J"; parajobs 2>&1||true
