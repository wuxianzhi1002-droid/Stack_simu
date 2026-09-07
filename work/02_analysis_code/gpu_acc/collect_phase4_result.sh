#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "$0")"&&pwd)"; F="$D/last_phase4_job_id.txt"; [ -f "$F" ]||exit 1; IFS='|' read -r J R < "$F"; O="$R/$J"; printf 'Phase 4 JOBID: %s\n' "$J"; [ -f "$O/phase4_result_summary.md" ]||{ printf '[INFO] Result pending.\n'; exit 0; }; cat "$O/phase4_result_summary.md"; printf 'JSON: %s\n' "$O/phase4_result.json"
