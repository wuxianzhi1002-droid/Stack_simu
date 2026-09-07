"""Record allocated-GPU and Python runtime metadata on a Slurm compute node."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import scipy


def cuda_version(value: int) -> str:
    return f"{value // 1000}.{(value % 1000) // 10}"


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect() -> dict:
    device_count = int(cp.cuda.runtime.getDeviceCount())
    devices = []
    for index in range(device_count):
        with cp.cuda.Device(index):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            devices.append(
                {
                    "visible_index": index,
                    "name": str(name),
                    "total_memory_bytes": int(props["totalGlobalMem"]),
                    "compute_capability": (
                        f"{int(props['major'])}.{int(props['minor'])}"
                    ),
                }
            )

    query_lines = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    driver_version = query_lines[0].split(",")[1].strip() if query_lines else "unknown"

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
            "job_name": os.environ.get("SLURM_JOB_NAME", ""),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "visible_gpu_count": device_count,
        "visible_devices": devices,
        "driver_version": driver_version,
        "cuda_runtime": cuda_version(int(cp.cuda.runtime.runtimeGetVersion())),
        "cuda_driver_api": cuda_version(int(cp.cuda.runtime.driverGetVersion())),
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "nvidia_smi_query": query_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["visible_gpu_count"] != 1:
        print(
            "FAIL: CuPy must see exactly one allocated GPU; "
            f"detected {info['visible_gpu_count']}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print("PASS: CuPy sees exactly one allocated GPU.")


if __name__ == "__main__":
    main()
