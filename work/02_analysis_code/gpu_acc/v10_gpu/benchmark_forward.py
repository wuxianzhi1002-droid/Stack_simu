"""Independent Phase 1 strict-TMM benchmark for B=1,7,13,32,64."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .backend import (
    BackendUnavailableError,
    CupyStrictTMMBackend,
    NumpyStrictTMMBackend,
)
from .backend.v10_source import source_sha256
from .reference import BOUNDS, REFERENCE_SEED, deterministic_parameters

DEFAULT_BATCH_SIZES = (1, 7, 13, 32, 64)


def timed_min(function, repeat: int) -> tuple[float, Any]:
    durations = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return min(durations), result


def make_parameter_bank(count: int, seed: int) -> np.ndarray:
    return deterministic_parameters(count=count, seed=seed)


def benchmark_backend(
    backend: Any,
    params_bank: np.ndarray,
    batch_sizes: tuple[int, ...],
    warmup: int,
    repeat: int,
) -> dict:
    if warmup:
        warm_params = params_bank[: min(1, len(params_bank))]
        for _ in range(warmup):
            backend.predict_batch(warm_params)

    rows = []
    for batch_size in batch_sizes:
        params = params_bank[:batch_size]

        upload_s, params_device = timed_min(
            lambda: backend.parameters_to_device(params), repeat
        )

        def run_kernel():
            result = backend.predict_batch_device(params_device)
            backend.synchronize()
            return result

        kernel_s, result_device = timed_min(run_kernel, repeat)

        def run_download():
            result = backend.to_host(result_device)
            backend.synchronize()
            return result

        download_s, result_host = timed_min(run_download, repeat)
        total_s = upload_s + kernel_s + download_s
        finite = np.isfinite(result_host)
        rows.append(
            {
                "batch_size": int(batch_size),
                "upload_s": upload_s,
                "kernel_s": kernel_s,
                "download_s": download_s,
                "total_s": total_s,
                "candidates_per_s": float(batch_size / total_s),
                "nan_count": int(np.isnan(result_host).sum()),
                "inf_count": int(np.isinf(result_host).sum()),
                "min_reflectance": float(np.min(result_host[finite])),
                "max_reflectance": float(np.max(result_host[finite])),
                "checksum_sum": float(np.sum(result_host)),
            }
        )
    memory = backend.memory_stats()
    report = {
        "status": "ok",
        "backend": backend.backend_name,
        "memory": memory,
        "rows": rows,
    }
    if backend.backend_name == "cuda_cupy":
        # CuPy's retained pool total is the allocator high-water reservation
        # after ascending B=1..64 and is used as the Phase 1 peak-memory metric.
        report["peak_gpu_memory_bytes"] = int(
            memory.get("memory_pool_total_bytes", 0)
        )
    return report


def compare_backends(
    cpu: NumpyStrictTMMBackend,
    gpu: CupyStrictTMMBackend,
    params_bank: np.ndarray,
    batch_sizes: tuple[int, ...],
) -> list[dict]:
    comparisons = []
    for batch_size in batch_sizes:
        cpu_values = cpu.predict_batch(params_bank[:batch_size])
        gpu_values = gpu.predict_batch(params_bank[:batch_size])
        difference = cpu_values - gpu_values
        rmse = float(np.sqrt(np.mean(difference**2)))
        max_abs = float(np.max(np.abs(difference)))
        comparisons.append(
            {
                "batch_size": int(batch_size),
                "rmse": rmse,
                "max_abs": max_abs,
                "nan_inf_count": int(
                    np.size(gpu_values) - np.count_nonzero(np.isfinite(gpu_values))
                ),
                "pass": bool(
                    rmse <= 1.0e-10
                    and max_abs <= 1.0e-8
                    and np.all(np.isfinite(gpu_values))
                ),
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Benchmark frozen CPU/CuPy V10 strict TMM."
    )
    parser.add_argument(
        "--backend",
        choices=("cpu_numpy", "cuda_cupy", "both"),
        default="cpu_numpy",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=package_dir / "cpu_reference" / "forward_reference.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_dir / "cpu_reference" / "cpu_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_sizes = tuple(args.batch_sizes)
    if any(size < 1 for size in batch_sizes):
        raise ValueError("All batch sizes must be positive.")
    with np.load(args.reference, allow_pickle=False) as reference:
        wavelengths_um = np.asarray(reference["wavelengths_um"], dtype=np.float64)

    params_bank = make_parameter_bank(max(batch_sizes), args.seed)
    report = {
        "phase": "Phase 1 strict-TMM benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_v10_sha256": source_sha256(),
        "reference_file": str(args.reference.resolve()),
        "batch_sizes": list(batch_sizes),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "wavelength_points": int(wavelengths_um.size),
        "dtype_real": "float64",
        "dtype_complex": "complex128",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "backends": {},
    }

    cpu = None
    gpu = None
    if args.backend in {"cpu_numpy", "both"}:
        cpu = NumpyStrictTMMBackend(wavelengths_um)
        report["backends"]["cpu_numpy"] = benchmark_backend(
            cpu, params_bank, batch_sizes, args.warmup, args.repeat
        )

    if args.backend in {"cuda_cupy", "both"}:
        availability = CupyStrictTMMBackend.availability()
        if not availability.get("available", False):
            report["backends"]["cuda_cupy"] = {
                "status": "unavailable",
                **availability,
            }
        else:
            try:
                gpu = CupyStrictTMMBackend(wavelengths_um, args.device_id)
                report["backends"]["cuda_cupy"] = benchmark_backend(
                    gpu, params_bank, batch_sizes, args.warmup, args.repeat
                )
            except BackendUnavailableError as exc:
                report["backends"]["cuda_cupy"] = {
                    "status": "unavailable",
                    "reason": str(exc),
                }

    if cpu is not None and gpu is not None:
        report["cpu_gpu_validation"] = compare_backends(
            cpu, gpu, params_bank, batch_sizes
        )
        report["cpu_gpu_validation_pass"] = all(
            row["pass"] for row in report["cpu_gpu_validation"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
