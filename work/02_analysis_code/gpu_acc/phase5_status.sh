#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
avail4090(){ local t n; t=$(sinfo -p gpu_4090 2>&1||true); n=$(printf '%s\n' "$t"|awk '$1=="gpu_4090"&&$2~/^[0-9]+$/{print $2;exit}'); case "$n" in ''|*[!0-9]*) n=0;; esac; printf '%s\n' "$n"; }
if [ "${1:-}" = --watch ]; then JOB_ID="$2"; RUN_ROOT="$3"; sleep "${TMM_GPU_FALLBACK_WAIT_SECONDS:-600}"; while true; do S=$(squeue -h -j "$JOB_ID" -o '%T|%P' 2>/dev/null||true); [ "$S" = 'PENDING|gpu_5090' ]||exit 0; N=$(avail4090); if [ "$N" -gt 0 ]; then scancel --state=PENDING "$JOB_ID" 2>/dev/null||true; sleep 2; S=$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null||true); [ -z "$S" ]||exit 0; TMM_GPU_PARTITION=gpu_4090 TMM_DISABLE_FALLBACK=1 TMM_CLUSTER_RUN_ROOT="$RUN_ROOT" bash "$SCRIPT_DIR/submit_phase5.sh"; exit $?; fi; sleep 60; done; fi
parajobs 2>&1||true; F="$SCRIPT_DIR/last_phase5_job_id.txt"; [ -f "$F" ]||exit 0; IFS='|' read -r J R < "$F"; printf '=== Phase 5 JOBID %s ===\n' "$J"; for f in "$R/phase5_slurm_${J}.out" "$R/phase5_slurm_${J}.err" "$R/$J/stdout.log" "$R/$J/stderr.log"; do [ -f "$f" ]&&{ printf '=== %s ===\n' "$f";tail -n 30 "$f"||true;};done
