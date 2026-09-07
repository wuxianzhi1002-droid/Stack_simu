#!/usr/bin/env bash
set -euo pipefail
PROJECT=/ssd/scxl838/v15_project_20260905/gpu_acc
DATA=/ssd/scxl838/v15_project_20260905/v15_eq99x_compact_remote_inputs_20260905_012202/stage1
CAL=/ssd/scxl838/v15_project_20260905/v15_eq99x_compact_remote_inputs_20260905_012202/calibration
MULTI=/ssd/scxl838/v15_project_20260905/static_stackrt_v15_eq99x_stage3_multiangle_200_800_20260905_011431
STAGE1_ROOT=/ssd/scxl838/v15_stage1_production
STAGE3_ROOT=/ssd/scxl838/v15_stage3_production
STAGE3_SMOKE=1554195
cd "$PROJECT"
submit_stage1(){
 local band="$1"
 local dep="${2:-}"
 local args=(--parsable --export="ALL,TMM_V15_PROJECT_DIR=$PROJECT,TMM_V15_DATASET=$DATA,TMM_V15_CALIBRATION=$CAL,TMM_V15_BAND=$band,TMM_V15_RUN_ROOT=$STAGE1_ROOT,TMM_V15_CASE_COUNT=100")
 [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
 timeout 180 sbatch "${args[@]}" run_v15_stage1_band.sh
}
submit_stage3(){
 local band="$1" dep="$2"; timeout 180 sbatch --parsable --dependency="afterok:$dep" --export="ALL,TMM_V15_PROJECT_DIR=$PROJECT,TMM_V15_MULTIANGLE_DATASET=$MULTI,TMM_V15_CALIBRATION=$CAL,TMM_V15_BAND=$band,TMM_V15_RUN_ROOT=$STAGE3_ROOT,TMM_V15_CASE_COUNT=100" run_v15_stage3_band.sh
}
j1=$(submit_stage1 220-580);echo "stage1_220_580=$j1"
j2=$(submit_stage1 200-600 "$j1");echo "stage1_200_600=$j2"
j3=$(submit_stage1 200-650 "$j2");echo "stage1_200_650=$j3"
j4=$(submit_stage1 200-700 "$j3");echo "stage1_200_700=$j4"
j5=$(submit_stage1 200-800 "$j4");echo "stage1_200_800=$j5"
s3a=$(submit_stage3 220-580 "$STAGE3_SMOKE:$j5");echo "stage3_220_580=$s3a"
s3b=$(submit_stage3 200-800 "$s3a");echo "stage3_200_800=$s3b"
printf '%s\n' "stage1_220_580=$j1" "stage1_200_600=$j2" "stage1_200_650=$j3" "stage1_200_700=$j4" "stage1_200_800=$j5" "stage3_smoke=$STAGE3_SMOKE" "stage3_220_580=$s3a" "stage3_200_800=$s3b" > /ssd/scxl838/v15_project_20260905/v15_job_chain.txt
