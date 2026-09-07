#!/usr/bin/env bash
# Shared module discovery for N50R5 login and gpu_5090 compute nodes.
# This file is sourced by other scripts and performs no work on its own.

tmm_module_fail() {
    printf '[FAIL] %s\n' "$1" >&2
    return 1
}

tmm_require_module_command() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi

    printf '[BOOTSTRAP] Environment Modules is not initialized; probing standard init scripts.\n'
    for module_init in \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/modules/init/bash \
        /usr/share/modules/init/sh \
        /usr/share/Modules/init/bash \
        /usr/share/lmod/lmod/init/bash \
        /opt/ohpc/admin/lmod/lmod/init/bash
    do
        if [ -r "$module_init" ]; then
            printf '[BOOTSTRAP] Sourcing module init: %s\n' "$module_init"
            # shellcheck disable=SC1090
            source "$module_init"
            if command -v module >/dev/null 2>&1; then
                printf '[PASS] Environment Modules initialized from %s.\n' "$module_init"
                return 0
            fi
        fi
    done

    tmm_module_fail "Environment Modules command is unavailable after probing standard init scripts."
    return 1
}

tmm_module_inventory() {
    MODULES_COLOR=false module -t avail 2>&1 || true
}

tmm_inventory_has_exact() {
    inventory_text="$1"
    target_name="$2"
    printf '%s\n' "$inventory_text" | awk -v target="$target_name" '
        {
            line=$0
            sub(/^[[:space:]]+/, "", line)
            sub(/[[:space:]]+$/, "", line)
            sub(/\(.*/, "", line)
            if (line == target) found=1
        }
        END { exit(found ? 0 : 1) }
    '
}

tmm_print_conda_identity() {
    printf '[PASS] conda is available.\n'
    conda --version
    which conda
}

tmm_ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        printf '[PASS] Using conda already present in the current shell.\n'
        tmm_print_conda_identity
        return 0
    fi

    tmm_require_module_command || return 1

    module load miniforge3/26.3.2-3 || {
        tmm_module_fail "Failed to load miniforge3/26.3.2-3."
        return 1
    }

    printf '[PASS] Loaded Miniforge: miniforge3/26.3.2-3\n'

    command -v conda >/dev/null 2>&1 || {
        tmm_module_fail "conda is unavailable after loading miniforge3/26.3.2-3."
        return 1
    }

    tmm_print_conda_identity
}

tmm_select_cuda_12_8_module() {
    module_inventory="$1"

    for candidate in         cuda/12.8         cuda/12.8.0         CUDA/12.8         CUDA/12.8.0
    do
        if tmm_inventory_has_exact "$module_inventory" "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    printf '%s\n' "$module_inventory" | awk '
        {
            line=$0
            sub(/^[[:space:]]+/, "", line)
            sub(/[[:space:]]+$/, "", line)
            sub(/\(.*/, "", line)
            lower=tolower(line)
            if (
                lower ~ /12\.8/ &&
                lower ~ /(^|\/)cuda[^\/]*(\/|$)/ &&
                lower !~ /cudnn/
            ) {
                print line
                exit
            }
        }
    '
}

tmm_ensure_cuda_12_8() {
    if command -v nvcc >/dev/null 2>&1 && nvcc --version 2>&1 | grep -Eq 'release[[:space:]]+12\.8'; then
        printf '[PASS] Using CUDA 12.8 already present in the current shell.\n'
        command -v nvcc
        nvcc --version | tail -n 1
        return 0
    fi

    tmm_require_module_command || return 1

    module load cuda/12.8 || {
        tmm_module_fail "Failed to load cuda/12.8."
        return 1
    }

    printf '[PASS] Loaded CUDA: cuda/12.8\n'

    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        nvcc --version | tail -n 1
    else
        printf '[INFO] nvcc is not exposed; GPU runtime validation will run on the compute node.\n'
    fi
}
