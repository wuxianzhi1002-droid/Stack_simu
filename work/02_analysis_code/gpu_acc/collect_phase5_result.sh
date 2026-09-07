#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "$0")"&&pwd)";F="$D/last_phase5_job_id.txt";[ -f "$F" ]||{ printf '[FAIL] No Phase 5 JOBID.\n' >&2;exit 1;};IFS='|' read -r J R < "$F";O="$R/$J";printf 'Phase 5 JOBID: %s\n' "$J";[ -f "$O/phase5_10npz_summary.md" ]||{ printf '[INFO] Result pending.\n';exit 0;};cat "$O/phase5_10npz_summary.md";printf 'JSON: %s\nCSV: %s\n' "$O/phase5_10npz_results.json" "$O/phase5_10npz_table.csv"
