#!/usr/bin/env python3
"""Recover a complete V12 report after quota failure during final JSON serialization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def finite_stats(values):
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    position = (len(data) - 1) * 0.95
    lo, hi = math.floor(position), math.ceil(position)
    p95 = data[lo] if lo == hi else data[lo] * (hi - position) + data[hi] * (position - lo)
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": p95,
        "max": max(data),
    }


def aggregate(rows):
    return {
        "absolute_Air_error_nm": finite_stats(r["errors"]["absolute_Air_error_nm"] for r in rows),
        "film_MAE_nm": finite_stats(r["errors"]["film_MAE_nm"] for r in rows),
        "angle_abs_error_deg": finite_stats(r["errors"]["angle_abs_error_deg"] for r in rows),
        "exact_RMSE": finite_stats(r["strict_exact_RMSE"] for r in rows),
        "fixed_condition_number": finite_stats(r["fixed_jacobian_diagnostics"]["condition_number"] for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--old-report-template", required=True, type=Path)
    parser.add_argument("--machine", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--slurm-elapsed-raw-s", required=True, type=float)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 401:
        raise ValueError(f"Expected 401 progress rows, got {len(rows)}")
    if [row["index"] for row in rows] != list(range(1, 402)):
        raise ValueError("Progress indices are not exactly 1..401")
    if len({row["filename"] for row in rows}) != 401:
        raise ValueError("Progress filenames are not unique")
    gates = {
        "all_closure_pass": all(row["closure"]["pass"] for row in rows),
        "all_prediction_finite": all(row["prediction_finite"] for row in rows),
        "all_fresh_population": all(row["fresh_population"] and not row["population_reused"] for row in rows),
        "all_sigma_0001": all(math.isclose(float(row["metadata"]["angle_measurement_sigma_deg"]), 0.001, abs_tol=1e-12) for row in rows),
        "all_angle_override_audited": all(row["metadata"].get("angle_override_applied") is True for row in rows),
    }
    if not all(gates.values()):
        raise ValueError(f"Recovery gates failed: {gates}")

    old = json.loads(args.old_report_template.read_text(encoding="utf-8"))
    status = Counter(
        "terminated" if row["selected"]["success"] and row["selected"]["status"] > 0
        else "budget" if row["selected"]["status"] == 0 else "other"
        for row in rows
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["noise_case"]].append(row)
    total_global = sum(row["global_runtime_s"] for row in rows)
    total_local = sum(row["local_runtime_s"] for row in rows)
    case_sum = sum(row["case_runtime_s"] for row in rows)
    summary = {
        "processed": 401,
        "terminated": status["terminated"],
        "budget_exhausted": status["budget"],
        "other_finite": status["other"],
        "closure_passed": sum(row["closure"]["pass"] for row in rows),
        "fresh_gpu_de_cases": sum(row["fresh_population"] and not row["population_reused"] for row in rows),
        "backend_initialization_count": 1,
        "resident_reused": True,
        "global_population_size": 40,
        "local_batch_size": 11,
        "cases_with_boundary_hits": sum(bool(row["selected"]["boundary"]["boundary_hits"]) for row in rows),
        "cases_near_boundary": sum(bool(row["selected"]["boundary"]["near_boundary"]) for row in rows),
        "boundary_preference_changed_selection": sum(bool(row["ranking"]["boundary_preference_changed_selection"]) for row in rows),
        "total_runtime_s": args.slurm_elapsed_raw_s,
        "total_runtime_basis": "Slurm ElapsedRaw; includes setup and failed final serialization",
        "compute_case_runtime_sum_s": case_sum,
        "total_global_runtime_s": total_global,
        "total_local_runtime_s": total_local,
        "mean_case_runtime_s": statistics.fmean(row["case_runtime_s"] for row in rows),
        "throughput_cases_per_min": 401 / args.slurm_elapsed_raw_s * 60.0,
        "peak_gpu_memory_bytes": None,
        "peak_gpu_memory_gib": None,
    }
    configuration = dict(old["configuration"])
    configuration.update({
        "angle_overrides_csv": "/data/home/scxl838/v12_sigma0001_inputs/dataset_index.csv",
        "angle_override_count": 401,
        "angle_measurement_sigma_deg": [0.001],
    })
    report = {
        "overall": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_count": 401,
        "configuration": configuration,
        "summary": summary,
        "aggregate_errors": aggregate(rows),
        "noise_case_aggregates": {name: aggregate(group) for name, group in sorted(groups.items())},
        "formal_v10_sha256": old["formal_v10_sha256"],
        "input_dir": "/data/home/scxl838/v12_stackrt_datasets/static_stackrt_v12_20260901_010407",
        "gpu_memory": {"available": False, "reason": "final in-memory report was lost at quota-limited serialization"},
        "machine": json.loads(args.machine.read_text(encoding="utf-8")),
        "cases": rows,
        "scope_guard": "V12 fixed Angle scheme A; audited sigma=0.001 per-file overlay; no MAP prior; fresh B=40 GPU strict full-ILS global plus B=11 GPU strict local.",
        "recovery_audit": {
            "source_job_id": args.slurm_job_id,
            "source_job_terminal_state": "FAILED only at final JSON serialization: OSError 122 disk quota exceeded",
            "source_progress_jsonl": str(args.progress),
            "progress_row_count": 401,
            "gates": gates,
            "numerical_recompute_performed": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "v12_fixed_angle_gpu_results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "index", "filename", "noise_case", "quality_class", "success", "status",
        "measured_angle_deg", "true_angle_deg", "angle_measurement_error_deg", "spectrum_cost",
        "exact_RMSE", "Air_error_nm", "film_MAE_nm", "boundary_hits", "near_boundary",
        "boundary_severity", "boundary_preference_changed_selection", "fixed_condition_number",
        "augmented_condition_number", "rho_air_angle", "global_runtime_s", "local_runtime_s",
        "case_runtime_s", "global_batch_size", "local_batch_size", "closure_rmse", "closure_max_abs",
    ]
    with (args.output_dir / "v12_fixed_angle_gpu_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            selected = row["selected"]
            boundary = selected["boundary"]
            writer.writerow({
                "index": row["index"], "filename": row["filename"], "noise_case": row["metadata"]["noise_case"],
                "quality_class": selected["quality_class"], "success": selected["success"], "status": selected["status"],
                "measured_angle_deg": row["measured_angle_deg"], "true_angle_deg": row["truth_angle_deg"],
                "angle_measurement_error_deg": row["errors"]["angle_measurement_error_deg"],
                "spectrum_cost": selected["spectrum_cost"], "exact_RMSE": row["strict_exact_RMSE"],
                "Air_error_nm": row["errors"]["Air_error_nm"], "film_MAE_nm": row["errors"]["film_MAE_nm"],
                "boundary_hits": ";".join(boundary["boundary_hits"]), "near_boundary": ";".join(boundary["near_boundary"]),
                "boundary_severity": boundary["severity"],
                "boundary_preference_changed_selection": row["ranking"]["boundary_preference_changed_selection"],
                "fixed_condition_number": row["fixed_jacobian_diagnostics"]["condition_number"],
                "augmented_condition_number": row["augmented_six_parameter_diagnostics"]["condition_number"],
                "rho_air_angle": row["augmented_six_parameter_diagnostics"]["rho_air_angle"],
                "global_runtime_s": row["global_runtime_s"], "local_runtime_s": row["local_runtime_s"],
                "case_runtime_s": row["case_runtime_s"],
                "global_batch_size": row["global_profile"]["differential_evolution"]["mean_batch_size"],
                "local_batch_size": row["cache"]["local_batch_size"], "closure_rmse": row["closure"]["rmse"],
                "closure_max_abs": row["closure"]["max_abs"],
            })
    print(json.dumps({"result": str(result_path), "overall": "PASS", "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
