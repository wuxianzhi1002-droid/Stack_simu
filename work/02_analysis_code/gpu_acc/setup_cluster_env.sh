#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="tmm5090"
REQUIREMENTS="$SCRIPT_DIR/requirements-cuda12x.txt"
MODULE_HELPER="$SCRIPT_DIR/cluster_modules.sh"

pass() {
    printf '[PASS] %s\n' "$1"
}

fail() {
    printf '[FAIL] %s\n' "$1" >&2
    exit 1
}

trap 'rc=$?; printf "[FAIL] command failed (exit=%s): %s\n" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR

printf '=== V10 Phase 0/1 cluster environment setup ===\n'
printf 'Script directory: %s\n' "$SCRIPT_DIR"
printf 'This script performs environment/file/import checks only.\n'
printf 'It does not run reference replay, benchmark or simulation.\n'

[ -f "$REQUIREMENTS" ] || fail "Missing requirements-cuda12x.txt."
[ -f "$MODULE_HELPER" ] || fail "Missing cluster_modules.sh."
[ -f "$SCRIPT_DIR/v10_gpu/cpu_reference/forward_reference.npz" ] || fail "Missing CPU reference NPZ."
[ -f "$SCRIPT_DIR/v10_gpu/cpu_reference/cpu_benchmark.json" ] || fail "Missing CPU benchmark JSON."
[ -f "$SCRIPT_DIR/../tmm_joint_inversion_v10.py" ] || fail "Missing formal V10 source."
[ -f "$SCRIPT_DIR/run_phase1_gpu.sh" ] || fail "Missing GPU Slurm script."
pass "Required project files exist."

source "$MODULE_HELPER"
tmm_ensure_conda
eval "$(conda shell.bash hook)"
pass "Initialized conda shell integration."

if conda env list | awk '$1 == "tmm5090" { found=1 } END { exit !found }'; then
    pass "Conda environment tmm5090 already exists."
else
    printf '[INFO] Creating conda environment tmm5090 with Python 3.11.\n'
    conda create -y -n "$ENV_NAME" python=3.11
    pass "Created conda environment tmm5090."
fi

conda activate "$ENV_NAME"
pass "Activated conda environment tmm5090."

python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
pass "Python version is 3.11."

tmm_ensure_cuda_12_8

python -m pip install -r "$REQUIREMENTS"
pass "Installed requirements-cuda12x.txt."

python -c 'import numpy; print("NumPy", numpy.__version__)'
pass "NumPy import."

python -c 'import scipy; print("SciPy", scipy.__version__)'
pass "SciPy import."

python -c 'import cupy; print("CuPy", cupy.__version__)'
pass "CuPy import (no CUDA computation was requested)."

python -c 'import pandas, matplotlib; print("Pandas", pandas.__version__, "Matplotlib", matplotlib.__version__)'
pass "Formal V10 support imports."

chmod u+x "$SCRIPT_DIR"/*.sh
pass "Cluster shell scripts are executable."

printf '=== SETUP COMPLETE: PASS ===\n'
printf 'Next command: %s/cluster.sh submit\n' "$SCRIPT_DIR"
