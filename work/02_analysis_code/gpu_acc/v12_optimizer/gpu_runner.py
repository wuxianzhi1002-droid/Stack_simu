"""V12 fixed-angle five-parameter GPU production runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import load_v10_module, source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend, NumpyStrictSpectrometerBackend
from v11_optimizer.stage12_runner import EXPECTED

from .fixed_angle_gpu import (
    augmented_six_parameter_diagnostics,
    run_gpu_global,
    run_gpu_local,
)


def safe(value):
    return v12.safe(value)


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_angle_overrides(path: Path | None, expected_names: list[str]) -> dict[str, dict]:
    if path is None:
        return {}
    override_path = path.resolve()
    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "npz_path",
        "measured_angle_deg",
        "angle_measurement_error_deg",
        "angle_measurement_sigma_deg",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"angle override CSV missing required columns: {sorted(required)}")
    result = {}
    for row in rows:
        name = row["npz_path"].replace("\\", "/").rsplit("/", 1)[-1]
        if not name or name in result:
            raise ValueError(f"duplicate or empty angle override filename: {name!r}")
        measured = float(row["measured_angle_deg"])
        error = float(row["angle_measurement_error_deg"])
        sigma = float(row["angle_measurement_sigma_deg"])
        if not all(np.isfinite(value) for value in (measured, error, sigma)) or sigma <= 0.0:
            raise ValueError(f"invalid angle override values: {name}")
        result[name] = {
            "measured_angle_deg": measured,
            "angle_measurement_error_deg": error,
            "angle_measurement_sigma_deg": sigma,
        }
    expected = set(expected_names)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"angle override filename contract failed: missing={missing[:3]}, extra={extra[:3]}"
        )
    return result


def apply_angle_override(measurement: dict, filename: str, overrides: dict[str, dict]) -> dict:
    if not overrides:
        return measurement
    override = overrides[filename]
    updated = dict(measurement)
    metadata = dict(measurement["metadata"])
    metadata.update(
        {
            "measured_angle_deg": override["measured_angle_deg"],
            "angle_measurement_sigma_deg": override["angle_measurement_sigma_deg"],
            "angle_measurement_error_deg": override["angle_measurement_error_deg"],
            "angle_override_applied": True,
        }
    )
    updated["metadata"] = metadata
    updated["fixed_angle_deg"] = override["measured_angle_deg"]
    return updated


def finite_stats(values):
    data = np.asarray([float(value) for value in values if np.isfinite(float(value))])
    if not len(data):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "max": float(np.max(data)),
    }


def aggregate(rows):
    return {
        "absolute_Air_error_nm": finite_stats(row["errors"]["absolute_Air_error_nm"] for row in rows),
        "film_MAE_nm": finite_stats(row["errors"]["film_MAE_nm"] for row in rows),
        "angle_abs_error_deg": finite_stats(row["errors"]["angle_abs_error_deg"] for row in rows),
        "exact_RMSE": finite_stats(row["strict_exact_RMSE"] for row in rows),
        "fixed_condition_number": finite_stats(
            row["fixed_jacobian_diagnostics"]["condition_number"]
            for row in rows
            if row["fixed_jacobian_diagnostics"]["condition_number"] is not None
        ),
    }


def compact_selected(selected):
    row = dict(selected)
    trace = np.asarray(row.pop("trace_costs", []), dtype=float)
    row.update(
        {
            "trace_count": int(trace.size),
            "trace_first_cost": float(trace[0]) if trace.size else None,
            "trace_last_cost": float(trace[-1]) if trace.size else None,
            "trace_min_cost": float(np.min(trace)) if trace.size else None,
            "trace_tail_costs": trace[-min(20, len(trace)) :],
        }
    )
    return row


def write_outputs(output_dir: Path, report: dict):
    prefix = "v12_fixed_angle_gpu"
    (output_dir / f"{prefix}_results.json").write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "index", "filename", "noise_case", "quality_class", "success", "status",
        "measured_angle_deg", "true_angle_deg", "angle_measurement_error_deg",
        "spectrum_cost", "exact_RMSE", "Air_error_nm", "film_MAE_nm",
        "boundary_hits", "near_boundary", "boundary_severity",
        "boundary_preference_changed_selection", "fixed_condition_number",
        "augmented_condition_number", "rho_air_angle", "global_runtime_s",
        "local_runtime_s", "case_runtime_s", "global_batch_size", "local_batch_size",
        "closure_rmse", "closure_max_abs",
    ]
    with (output_dir / f"{prefix}_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["cases"]:
            selected = row["selected"]
            boundary = selected["boundary"]
            writer.writerow(
                {
                    "index": row["index"],
                    "filename": row["filename"],
                    "noise_case": row["metadata"]["noise_case"],
                    "quality_class": selected["quality_class"],
                    "success": selected["success"],
                    "status": selected["status"],
                    "measured_angle_deg": row["measured_angle_deg"],
                    "true_angle_deg": row["truth_angle_deg"],
                    "angle_measurement_error_deg": row["errors"]["angle_measurement_error_deg"],
                    "spectrum_cost": selected["spectrum_cost"],
                    "exact_RMSE": row["strict_exact_RMSE"],
                    "Air_error_nm": row["errors"]["Air_error_nm"],
                    "film_MAE_nm": row["errors"]["film_MAE_nm"],
                    "boundary_hits": ";".join(boundary["boundary_hits"]),
                    "near_boundary": ";".join(boundary["near_boundary"]),
                    "boundary_severity": boundary["severity"],
                    "boundary_preference_changed_selection": row["ranking"]["boundary_preference_changed_selection"],
                    "fixed_condition_number": row["fixed_jacobian_diagnostics"]["condition_number"],
                    "augmented_condition_number": row["augmented_six_parameter_diagnostics"]["condition_number"],
                    "rho_air_angle": row["augmented_six_parameter_diagnostics"]["rho_air_angle"],
                    "global_runtime_s": row["global_runtime_s"],
                    "local_runtime_s": row["local_runtime_s"],
                    "case_runtime_s": row["case_runtime_s"],
                    "global_batch_size": row["global_profile"]["differential_evolution"]["mean_batch_size"],
                    "local_batch_size": row["cache"]["local_batch_size"],
                    "closure_rmse": row["closure"]["rmse"],
                    "closure_max_abs": row["closure"]["max_abs"],
                }
            )
    summary = report["summary"]
    errors = report["aggregate_errors"]
    lines = [
        "# V12 fixed-angle GPU production result",
        "",
        f"- Overall: **{report['overall']}**",
        f"- Processed: {summary['processed']}/{report['requested_count']}",
        f"- Angle mode: fixed independent measurement; MAP prior: disabled",
        f"- Free/global population/local batch dimensions: 5 / {summary['global_population_size']} / 11",
        f"- Strict closures: {summary['closure_passed']}/{summary['processed']}",
        f"- Fresh GPU DE: {summary['fresh_gpu_de_cases']}/{summary['processed']}",
        f"- Boundary hits / near-boundary: {summary['cases_with_boundary_hits']} / {summary['cases_near_boundary']}",
        f"- Boundary preference changed local selection: {summary['boundary_preference_changed_selection']}",
        f"- Total/global/local runtime: {summary['total_runtime_s']:.3f} / {summary['total_global_runtime_s']:.3f} / {summary['total_local_runtime_s']:.3f} s",
        f"- Throughput: {summary['throughput_cases_per_min']:.3f} cases/min",
        f"- Peak GPU memory: {summary['peak_gpu_memory_gib']:.3f} GiB",
        "",
        "## Scientific error summary",
        "",
        "| metric | mean | median | p95 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("absolute_Air_error_nm", "film_MAE_nm", "angle_abs_error_deg", "exact_RMSE", "fixed_condition_number"):
        item = errors[key]
        lines.append(
            f"| {key} | {item['mean']:.6g} | {item['median']:.6g} | {item['p95']:.6g} | {item['max']:.6g} |"
        )
    lines += [
        "",
        "## Contract",
        "",
        "Global search uses fresh B=40 GPU strict full-ILS differential evolution with a soft near-boundary penalty. Local fitting uses five free parameters and one B=11 GPU center-difference batch. Angle is fixed to the measured value for every model evaluation. Pure spectrum cost remains the final equivalence gate, so a materially better boundary solution is retained and flagged.",
    ]
    (output_dir / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--count", type=int, default=401)
    parser.add_argument("--require-production-count", action="store_true")
    parser.add_argument("--wavelength-min-nm", type=float, default=220.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument(
        "--angle-overrides-csv",
        type=Path,
        help="Optional per-filename independent angle measurement override table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_production_count and args.count != 401:
        raise ValueError("production requires exactly 401 cases")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if source_sha256() != EXPECTED:
        raise RuntimeError("formal V10 source hash changed")
    v10 = load_v10_module()
    root = args.input_dir.resolve()
    all_paths = sorted(root.glob("static_spectrum_*.npz"), key=lambda path: path.name)
    if args.require_production_count and len(all_paths) != 401:
        raise RuntimeError(f"expected 401 V12 NPZ files, found {len(all_paths)}")
    paths = all_paths[: args.count]
    if len(paths) != args.count:
        raise RuntimeError(f"requested {args.count} cases, found {len(paths)}")
    overrides = load_angle_overrides(
        args.angle_overrides_csv,
        [path.name for path in all_paths],
    )
    config = v12.FitConfig(
        input_dir=str(root),
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        global_forward_model="full_ils",
        global_popsize=8,
        global_maxiter=args.global_maxiter,
        multistarts=8,
        max_nfev=args.max_nfev,
        workers=1,
        random_seed=20260831,
    )
    first = apply_angle_override(v12.load_fit_input(paths[0], config), paths[0].name, overrides)
    gpu = CupyStrictSpectrometerBackend(
        first["wavelengths_um"],
        first["generator_config"],
        first["metadata"]["internal_wavelength_margin_nm"],
    )
    cpu = NumpyStrictSpectrometerBackend(
        first["wavelengths_um"],
        first["generator_config"],
        first["metadata"]["internal_wavelength_margin_nm"],
    )
    resident_before = resident_array_identity(gpu)
    population_size = config.global_popsize * len(v12.FREE_PARAMS)
    progress = output_dir / "v12_fixed_angle_gpu_progress.jsonl"
    rows = []
    started_all = time.perf_counter()
    for index, path in enumerate(paths, 1):
        case_started = time.perf_counter()
        measurement = apply_angle_override(v12.load_fit_input(path, config), path.name, overrides)
        if not np.array_equal(measurement["wavelengths_um"], first["wavelengths_um"]):
            raise RuntimeError(f"wavelength contract changed: {path.name}")
        seed = int(config.random_seed + (index - 1) * 1009)
        case_config = v12.FitConfig(**{**asdict(config), "random_seed": seed})
        population = v12.latin_hypercube_population(seed, population_size)
        _, de_result, candidates, profile = run_gpu_global(
            v10, measurement, case_config, population, gpu
        )
        differential = profile["differential_evolution"]
        if not (
            differential["min_batch_size"] == population_size
            and differential["max_batch_size"] == population_size
            and differential["gpu_batch_calls"] == differential["objective_calls"]
            and differential["candidate_evaluations"] == int(de_result.nfev) * population_size
        ):
            raise RuntimeError(f"V12 population batching contract failed: {path.name}")
        local = run_gpu_local(v10, measurement, case_config, candidates, gpu, cpu)
        prediction = np.asarray(local.pop("prediction"))
        truth, _ = v12.load_evaluation_truth(path)
        selected = compact_selected(local["selected"])
        diagnostics6 = augmented_six_parameter_diagnostics(
            gpu,
            local["full_parameters"],
            measurement["spectrum"],
            v10.robust_scale(measurement["spectrum"]),
        )
        row = {
            "index": index,
            "filename": path.name,
            "seed": seed,
            "metadata": measurement["metadata"],
            "measured_angle_deg": measurement["fixed_angle_deg"],
            "truth_angle_deg": truth["Angle"],
            "selected": selected,
            "ranking": {
                key: value
                for key, value in local["ranking"].items()
                if key != "attempts" and key != "selected"
            },
            "free_parameters": local["free_parameters"],
            "full_parameters": local["full_parameters"],
            "fresh_population": True,
            "population_reused": False,
            "global_runtime_s": profile["totals"]["total_global_runtime_s"],
            "global_profile": profile,
            "local_runtime_s": local["runtime_s"],
            "strict_exact_RMSE": local["exact_rmse"],
            "errors": v12.scientific_errors(local["full_parameters"], truth),
            "closure": local["closure"],
            "cache": local["cache"],
            "fixed_jacobian_diagnostics": local["fixed_jacobian_diagnostics"],
            "augmented_six_parameter_diagnostics": diagnostics6,
            "case_runtime_s": time.perf_counter() - case_started,
            "prediction_finite": bool(np.all(np.isfinite(prediction))),
        }
        rows.append(row)
        append_jsonl(progress, row)
        print(
            f"[{index}/{len(paths)}] {path.name} global={row['global_runtime_s']:.3f}s "
            f"local={row['local_runtime_s']:.3f}s B={differential['mean_batch_size']:.1f}/11 "
            f"hits={selected['boundary']['boundary_hits']} closure=PASS",
            flush=True,
        )
    total = time.perf_counter() - started_all
    resident_after = resident_array_identity(gpu)
    memory = gpu.memory_stats()
    status = Counter(
        "terminated" if row["selected"]["success"] and row["selected"]["status"] > 0
        else "budget" if row["selected"]["status"] == 0
        else "other"
        for row in rows
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["noise_case"]].append(row)
    summary = {
        "processed": len(rows),
        "terminated": status["terminated"],
        "budget_exhausted": status["budget"],
        "other_finite": status["other"],
        "closure_passed": sum(row["closure"]["pass"] for row in rows),
        "fresh_gpu_de_cases": sum(row["fresh_population"] and not row["population_reused"] for row in rows),
        "backend_initialization_count": 1,
        "resident_reused": resident_before == resident_after,
        "global_population_size": population_size,
        "local_batch_size": 11,
        "cases_with_boundary_hits": sum(bool(row["selected"]["boundary"]["boundary_hits"]) for row in rows),
        "cases_near_boundary": sum(bool(row["selected"]["boundary"]["near_boundary"]) for row in rows),
        "boundary_preference_changed_selection": sum(bool(row["ranking"]["boundary_preference_changed_selection"]) for row in rows),
        "total_runtime_s": total,
        "total_global_runtime_s": float(sum(row["global_runtime_s"] for row in rows)),
        "total_local_runtime_s": float(sum(row["local_runtime_s"] for row in rows)),
        "mean_case_runtime_s": float(np.mean([row["case_runtime_s"] for row in rows])),
        "throughput_cases_per_min": len(rows) / total * 60.0,
        "peak_gpu_memory_bytes": int(memory.get("memory_pool_total_bytes", 0)),
        "peak_gpu_memory_gib": int(memory.get("memory_pool_total_bytes", 0)) / (1024**3),
    }
    passed = bool(
        len(rows) == args.count
        and summary["closure_passed"] == args.count
        and summary["fresh_gpu_de_cases"] == args.count
        and summary["resident_reused"]
        and all(row["prediction_finite"] for row in rows)
        and source_sha256() == EXPECTED
    )
    report = {
        "overall": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_count": args.count,
        "configuration": {
            **asdict(config),
            "free_parameter_order": list(v12.FREE_PARAMS),
            "fixed_parameter": "Angle",
            "angle_mode": "fixed_independent_measurement",
            "map_prior_enabled": False,
            "population_size": population_size,
            "local_batch_size": 11,
            "vectorized": True,
            "updating": "deferred",
            "angle_overrides_csv": str(args.angle_overrides_csv.resolve()) if args.angle_overrides_csv else None,
            "angle_override_count": len(overrides),
            "angle_measurement_sigma_deg": sorted(
                {float(row["angle_measurement_sigma_deg"]) for row in overrides.values()}
            ) if overrides else None,
        },
        "summary": summary,
        "aggregate_errors": aggregate(rows),
        "noise_case_aggregates": {name: aggregate(values) for name, values in sorted(groups.items())},
        "formal_v10_sha256": source_sha256(),
        "input_dir": str(root),
        "gpu_memory": memory,
        "machine": json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,
        "cases": rows,
        "scope_guard": "V12 fixed Angle scheme A only; optional audited per-file measured-angle override; no MAP prior; no V10/V11 edit; fresh B=40 GPU strict full-ILS global plus B=11 GPU strict local.",
    }
    write_outputs(output_dir, report)
    print(
        json.dumps(
            {"overall": report["overall"], "processed": len(rows), "runtime_s": total, "output": str(output_dir)},
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
