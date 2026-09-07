#!/usr/bin/env bash
set -u
D="$(cd "$(dirname "$0")"&&pwd)";F="$D/last_phase6_job_id.txt";[ -f "$F" ]||exit 0;IFS='|' read -r J R < "$F";printf 'Phase 6 JOBID: %s\n' "$J";scancel "$J" 2>&1||true;parajobs 2>&1||true
