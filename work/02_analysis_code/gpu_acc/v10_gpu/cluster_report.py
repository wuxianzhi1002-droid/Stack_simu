"""Merge Phase 1 machine, benchmark and CPU-baseline JSON into final reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_BATCHES = (1, 7, 13, 32, 64)
RMSE_LIMIT = 1.0e-10
MAX_ABS_LIMIT = 1.0e-8


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_batch(rows: list[dict]) -> dict[int, dict]:
    return {int(row["batch_size"]): row for row in rows}


def build_report(machine: dict, benchmark: dict, baseline: dict) -> dict:
    errors: list[str] = []
    if int(machine.get("visible_gpu_count", 0)) != 1:
        errors.append("CuPy visible GPU count is not exactly one.")
    if machine.get("slurm", {}).get("partition") != "gpu_5090":
        errors.append("Job did not run in the gpu_5090 partition.")

    cpu_status = benchmark.get("backends", {}).get("cpu_numpy", {})
    gpu_status = benchmark.get("backends", {}).get("cuda_cupy", {})
    if cpu_status.get("status") != "ok":
        errors.append("Phase 1 CPU backend did not complete.")
    if gpu_status.get("status") != "ok":
        errors.append("Phase 1 CuPy backend did not complete.")

    cpu_rows = rows_by_batch(cpu_status.get("rows", []))
    gpu_rows = rows_by_batch(gpu_status.get("rows", []))
    baseline_rows = rows_by_batch(
        baseline.get("backends", {}).get("cpu_numpy", {}).get("rows", [])
    )
    validation_rows = rows_by_batch(benchmark.get("cpu_gpu_validation", []))

    batches = []
    for batch_size in EXPECTED_BATCHES:
        if batch_size not in cpu_rows:
            errors.append(f"Missing cluster CPU result for B={batch_size}.")
            continue
        if batch_size not in gpu_rows:
            errors.append(f"Missing GPU result for B={batch_size}.")
            continue
        if batch_size not in baseline_rows:
            errors.append(f"Missing existing CPU baseline for B={batch_size}.")
            continue
        if batch_size not in validation_rows:
            errors.append(f"Missing CPU/GPU validation for B={batch_size}.")
            continue

        gpu = gpu_rows[batch_size]
        old_cpu = baseline_rows[batch_size]
        validation = validation_rows[batch_size]
        nan_inf = int(gpu.get("nan_count", 0)) + int(gpu.get("inf_count", 0))
        rmse = float(validation["rmse"])
        max_abs = float(validation["max_abs"])
        if rmse > RMSE_LIMIT:
            errors.append(f"B={batch_size}: RMSE {rmse} exceeds {RMSE_LIMIT}.")
        if max_abs > MAX_ABS_LIMIT:
            errors.append(
                f"B={batch_size}: max abs {max_abs} exceeds {MAX_ABS_LIMIT}."
            )
        if nan_inf > 0 or int(validation.get("nan_inf_count", 0)) > 0:
            errors.append(f"B={batch_size}: NaN/Inf count is nonzero.")

        gpu_total = float(gpu["total_s"])
        batches.append(
            {
                "batch_size": batch_size,
                "gpu_kernel_s": float(gpu["kernel_s"]),
                "gpu_total_s": gpu_total,
                "gpu_throughput_candidates_per_s": float(
                    gpu["candidates_per_s"]
                ),
                "existing_cpu_total_s": float(old_cpu["total_s"]),
                "speedup_vs_existing_cpu": (
                    float(old_cpu["total_s"]) / gpu_total
                ),
                "rmse": rmse,
                "max_abs": max_abs,
                "nan_count": int(gpu.get("nan_count", 0)),
                "inf_count": int(gpu.get("inf_count", 0)),
            }
        )

    memory = gpu_status.get("memory", {})
    peak_gpu_memory_bytes = int(
        gpu_status.get(
            "peak_gpu_memory_bytes",
            memory.get("memory_pool_total_bytes", 0),
        )
    )
    if peak_gpu_memory_bytes <= 0:
        errors.append("Peak CuPy memory-pool allocation was not recorded.")

    return {
        "phase": "Phase 1 strict TMM only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pass": not errors,
        "errors": errors,
        "limits": {
            "rmse_max": RMSE_LIMIT,
            "max_abs_max": MAX_ABS_LIMIT,
            "nan_inf_allowed": 0,
        },
        "machine": {
            "hostname": machine.get("hostname"),
            "gpu_name": (
                machine.get("visible_devices", [{}])[0].get("name")
                if machine.get("visible_devices")
                else "unknown"
            ),
            "visible_gpu_count": machine.get("visible_gpu_count"),
            "driver_version": machine.get("driver_version"),
            "cuda_runtime": machine.get("cuda_runtime"),
            "cuda_driver_api": machine.get("cuda_driver_api"),
            "cupy_version": machine.get("cupy_version"),
            "python_version": machine.get("python_version"),
            "slurm": machine.get("slurm"),
        },
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_gib": peak_gpu_memory_bytes / (1024**3),
        "batches": batches,
        "benchmark_source": benchmark.get("reference_file"),
        "formal_v10_sha256": benchmark.get("formal_v10_sha256"),
        "scope_guard": (
            "No ILS, Jacobian, optimizer, global search, JAX, autodiff or "
            "multi-GPU work is included."
        ),
    }


def markdown(report: dict) -> str:
    machine = report["machine"]
    status = "PASS" if report["pass"] else "FAIL"
    lines = [
        "# V10 Phase 1 RTX5090 strict TMM result",
        "",
        f"- Overall: **{status}**",
        f"- GPU: {machine['gpu_name']}",
        f"- Visible GPU count: {machine['visible_gpu_count']}",
        f"- NVIDIA driver: {machine['driver_version']}",
        f"- CUDA runtime: {machine['cuda_runtime']}",
        f"- CuPy: {machine['cupy_version']}",
        f"- Python: {machine['python_version'].splitlines()[0]}",
        f"- Peak CuPy memory pool: {report['peak_gpu_memory_gib']:.3f} GiB",
        "",
        "| B | GPU kernel (s) | GPU total (s) | Throughput (candidate/s) | Speedup vs existing CPU | RMSE | Max abs | NaN/Inf |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["batches"]:
        lines.append(
            "| {batch_size} | {gpu_kernel_s:.6f} | {gpu_total_s:.6f} | "
            "{gpu_throughput_candidates_per_s:.3f} | "
            "{speedup_vs_existing_cpu:.2f}x | {rmse:.3e} | "
            "{max_abs:.3e} | {nan_inf} |".format(
                nan_inf=row["nan_count"] + row["inf_count"],
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- RMSE <= {RMSE_LIMIT:.1e}",
            f"- max abs <= {MAX_ABS_LIMIT:.1e}",
            "- NaN/Inf = 0",
            "- Exactly one GPU visible to CuPy",
            "- Partition = gpu_5090",
            "",
            "## Scope",
            "",
            report["scope_guard"],
        ]
    )
    if report["errors"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        load_json(args.machine),
        load_json(args.benchmark),
        load_json(args.baseline),
    )
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.output_md.write_text(
        markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
