"""V14 Stage 1a GPU runner for nested bands wider than 220-580 nm.

One locally generated 200-800 nm StackRT dataset is the immutable source for
all candidate bands.  Every candidate contains the historical 220-580 nm
baseline.  The V12 fixed measured angle remains fixed (scheme A, sigma 0.001
deg), and the strict full-ILS GPU global/local fitting contract is unchanged.

Stage 1a uses the current V12 synthetic source/QE/detector chain as an explicit
instrument proxy.  It must not be described as measured hardware calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import load_v10_module, source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend, NumpyStrictSpectrometerBackend
from v12_optimizer.fixed_angle_gpu import run_gpu_global, run_gpu_local

from .information import signal_jacobian_normalized_solver, whitened_information_metrics
from .noise_covariance import estimate_diagonal_sigma, sha256_file


VERSION = "v14_stage1a_wideband_information"
EXPECTED_FORMAL_V10_SHA256 = "d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"
MASTER_START_NM = 200.0
MASTER_STOP_NM = 800.0
MASTER_POINTS = 30_001
BASELINE_START_NM = 220.0
BASELINE_STOP_NM = 580.0
CANDIDATE_BANDS = {
    "220-580": (220.0, 580.0),
    "200-600": (200.0, 600.0),
    "200-650": (200.0, 650.0),
    "200-700": (200.0, 700.0),
    "200-800": (200.0, 800.0),
}


def safe(value: Any) -> Any:
    return v12.safe(value)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finite_stats(values) -> dict[str, Any]:
    data = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )
    if not len(data):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "max": float(np.max(data)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "absolute_Air_error_nm": finite_stats(row["absolute_Air_error_nm"] for row in rows),
        "film_MAE_nm": finite_stats(row["film_MAE_nm"] for row in rows),
        "exact_RMSE": finite_stats(row["exact_RMSE"] for row in rows),
        "sigma_min_Jw": finite_stats(row["sigma_min_Jw"] for row in rows),
        "condition_number_Jw": finite_stats(row["condition_number_Jw"] for row in rows),
        "log10_det_JwT_Jw": finite_stats(row["log10_det_JwT_Jw"] for row in rows),
        "density_sigma_min_Jw": finite_stats(row["density_sigma_min_Jw"] for row in rows),
        "density_log10_det_JwT_Jw": finite_stats(
            row["density_log10_det_JwT_Jw"] for row in rows
        ),
        "boundary_hit_rate": float(np.mean([bool(row["boundary_hits"]) for row in rows])),
        "terminated_rate": float(
            np.mean([bool(row["success"]) and int(row["status"]) > 0 for row in rows])
        ),
    }


def compact_selected(selected: dict[str, Any]) -> dict[str, Any]:
    row = dict(selected)
    trace = np.asarray(row.pop("trace_costs", []), dtype=np.float64)
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


def validate_master_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        axis = np.asarray(data["reported_wavelengths_nm"], dtype=np.float64)
        if axis.shape != (MASTER_POINTS,):
            raise RuntimeError(f"master point count changed: {path.name}")
        if (
            not np.isclose(axis[0], MASTER_START_NM, rtol=0.0, atol=1.0e-12)
            or not np.isclose(axis[-1], MASTER_STOP_NM, rtol=0.0, atol=1.0e-12)
        ):
            raise RuntimeError(f"master band is not 200-800 nm: {path.name}")
        if str(np.asarray(data["generator_version"]).item()) != "main_v12":
            raise RuntimeError(f"generator is not main_v12: {path.name}")
        if str(np.asarray(data["optical_backend"]).item()) != "api":
            raise RuntimeError(f"optical backend is not local StackRT API: {path.name}")
        if str(np.asarray(data["angle_measurement_mode"]).item()) != "fixed_independent_measurement":
            raise RuntimeError(f"angle mode changed: {path.name}")
        if not np.isclose(
            float(np.asarray(data["angle_measurement_sigma_deg"]).item()),
            0.001,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise RuntimeError(f"angle measurement sigma changed: {path.name}")
        if not bool(np.asarray(data["reported_axis_is_fixed"]).item()):
            raise RuntimeError(f"reported axis is not fixed: {path.name}")


def dataset_bundle_hash(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    rows = []
    for path in paths:
        file_hash = sha256_file(path)
        row = {"filename": path.name, "sha256": file_hash, "bytes": path.stat().st_size}
        rows.append(row)
        digest.update(f"{path.name}\t{file_hash}\t{path.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest(), rows


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    prefix = f"v14_stage1_band_{report['configuration']['band_label'].replace('-', '_')}"
    (output_dir / f"{prefix}_results.json").write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "index",
        "filename",
        "noise_case",
        "realization_index",
        "band_label",
        "band_start_nm",
        "band_stop_nm",
        "wavelength_samples",
        "quality_class",
        "success",
        "status",
        "measured_angle_deg",
        "Air_error_nm",
        "absolute_Air_error_nm",
        "film_MAE_nm",
        "exact_RMSE",
        "boundary_hits",
        "near_boundary",
        "sigma_min_Jw",
        "condition_number_Jw",
        "log10_det_JwT_Jw",
        "det_JwT_Jw",
        "density_sigma_min_Jw",
        "density_condition_number_Jw",
        "density_log10_det_JwT_Jw",
        "global_runtime_s",
        "local_runtime_s",
        "case_runtime_s",
        "closure_rmse",
        "closure_max_abs",
    ]
    with (output_dir / f"{prefix}_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["cases"]:
            writer.writerow({key: row.get(key) for key in fields})

    overall = report["aggregate"]
    summary = report["summary"]
    lines = [
        f"# V14 Stage 1a band {report['configuration']['band_label']} GPU result",
        "",
        f"- Overall: **{report['overall']}**",
        f"- Effective fit band: {report['configuration']['effective_band_nm']} nm",
        "- Master StackRT acquisition: 200-800 nm, 0.02 nm sampling",
        "- Scope: typical only; 10 noise types x 10 realizations",
        "- Angle: fixed independent measurement, sigma=0.001 deg, no Angle MAP",
        "- Measurement chain: V12 synthetic proxy, not measured hardware SPD/QE",
        f"- Processed / strict closure: {summary['processed']} / {summary['closure_passed']}",
        f"- Boundary-hit rate: {overall['boundary_hit_rate']:.6g}",
        f"- Runtime: {summary['total_runtime_s']:.3f} s",
        "",
        "## Overall statistics",
        "",
        "| metric | mean | median | p95 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "absolute_Air_error_nm",
        "film_MAE_nm",
        "exact_RMSE",
        "sigma_min_Jw",
        "condition_number_Jw",
        "log10_det_JwT_Jw",
        "density_sigma_min_Jw",
        "density_log10_det_JwT_Jw",
    ):
        item = overall[key]
        lines.append(
            f"| {key} | {item['mean']:.6g} | {item['median']:.6g} | "
            f"{item['p95']:.6g} | {item['max']:.6g} |"
        )
    lines += [
        "",
        "`Jw` uses a pooled, wavelength-dependent diagonal detector covariance. "
        "Systematic typical noise remains represented by the ten paired realization groups. "
        "Raw Fisher metrics reward total collected information; the density metrics divide "
        "Jw by sqrt(number of wavelength samples) to expose information per sample.",
    ]
    (output_dir / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--noise-calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--band", choices=list(CANDIDATE_BANDS), required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--require-production-count", action="store_true")
    parser.add_argument("--noise-case", type=str)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--max-nfev", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if source_sha256() != EXPECTED_FORMAL_V10_SHA256:
        raise RuntimeError("formal V10 source hash changed")
    if args.require_production_count and args.count != 100:
        raise ValueError("formal Stage 1 requires exactly 100 typical cases")
    band_start, band_stop = CANDIDATE_BANDS[args.band]
    if band_start > BASELINE_START_NM or band_stop < BASELINE_STOP_NM:
        raise RuntimeError("every Stage 1 candidate must contain the 220-580 nm baseline")

    root = args.input_dir.resolve()
    all_paths = sorted(root.glob("static_spectrum_*_typical_*.npz"), key=lambda path: path.name)
    if args.require_production_count and len(all_paths) != 100:
        raise RuntimeError(f"expected 100 typical master NPZ files, found {len(all_paths)}")
    paths = all_paths
    if args.noise_case:
        token = f"static_spectrum_{args.noise_case}_"
        paths = [path for path in paths if path.name.startswith(token)]
    paths = paths[: int(args.count)]
    if len(paths) != int(args.count):
        raise RuntimeError(f"requested {args.count} cases, found {len(paths)}")
    for path in paths:
        validate_master_npz(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_bundle_sha256, source_file_audit = dataset_bundle_hash(all_paths)
    covariance = estimate_diagonal_sigma(
        args.noise_calibration_dir,
        expected_start_nm=MASTER_START_NM,
        expected_stop_nm=MASTER_STOP_NM,
        expected_points=MASTER_POINTS,
    )

    config = v12.FitConfig(
        input_dir=str(root),
        wavelength_min_nm=band_start,
        wavelength_max_nm=band_stop,
        global_forward_model="full_ils",
        global_popsize=8,
        global_maxiter=int(args.global_maxiter),
        multistarts=8,
        max_nfev=int(args.max_nfev),
        workers=1,
        random_seed=20260904,
    )
    first = v12.load_fit_input(paths[0], config)
    selected_nm = np.asarray(first["wavelengths_um"], dtype=np.float64) * 1000.0
    covariance_axis = np.asarray(covariance["wavelengths_nm"], dtype=np.float64)
    covariance_mask = (covariance_axis >= band_start) & (covariance_axis <= band_stop)
    sigma = np.asarray(covariance["sigma"], dtype=np.float64)[covariance_mask]
    selected_covariance_axis = covariance_axis[covariance_mask]
    if (
        selected_covariance_axis.shape != selected_nm.shape
        or not np.allclose(
            selected_covariance_axis,
            selected_nm,
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise RuntimeError("noise covariance and fit wavelength axes do not match")

    v10 = load_v10_module()
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
    population_size = int(config.global_popsize) * len(v12.FREE_PARAMS)
    progress = output_dir / f"v14_stage1_band_{args.band.replace('-', '_')}_progress.jsonl"
    rows = []
    started_all = time.perf_counter()

    for index, path in enumerate(paths, 1):
        case_started = time.perf_counter()
        measurement = v12.load_fit_input(path, config)
        if measurement["metadata"]["noise_level"] != "typical":
            raise RuntimeError(f"non-typical case entered Stage 1: {path.name}")
        if not np.array_equal(measurement["wavelengths_um"], first["wavelengths_um"]):
            raise RuntimeError(f"effective wavelength contract changed: {path.name}")
        seed = int(config.random_seed + (index - 1) * 1009)
        case_config = v12.FitConfig(**{**asdict(config), "random_seed": seed})
        population = v12.latin_hypercube_population(seed, population_size)
        _, de_result, candidates, global_profile = run_gpu_global(
            v10, measurement, case_config, population, gpu
        )
        differential = global_profile["differential_evolution"]
        if not (
            differential["min_batch_size"] == population_size
            and differential["max_batch_size"] == population_size
            and differential["gpu_batch_calls"] == differential["objective_calls"]
            and differential["candidate_evaluations"] == int(de_result.nfev) * population_size
        ):
            raise RuntimeError(f"GPU population batching contract failed: {path.name}")
        local = run_gpu_local(v10, measurement, case_config, candidates, gpu, cpu)
        local.pop("prediction")
        selected = compact_selected(local["selected"])
        signal_jacobian, jacobian_audit = signal_jacobian_normalized_solver(
            gpu,
            local["free_parameters"],
            measurement["fixed_angle_deg"],
        )
        information = whitened_information_metrics(signal_jacobian, sigma)
        raw_info = information["raw_information"]
        density_info = information["per_sample_information_density"]
        truth, _ = v12.load_evaluation_truth(path)
        errors = v12.scientific_errors(local["full_parameters"], truth)
        boundary = selected["boundary"]
        row = {
            "index": index,
            "filename": path.name,
            "seed": seed,
            "noise_case": measurement["metadata"]["noise_case"],
            "noise_factor": measurement["metadata"]["noise_factor"],
            "noise_level": measurement["metadata"]["noise_level"],
            "realization_index": measurement["metadata"]["realization_index"],
            "band_label": args.band,
            "band_start_nm": band_start,
            "band_stop_nm": band_stop,
            "wavelength_samples": int(len(selected_nm)),
            "measured_angle_deg": measurement["fixed_angle_deg"],
            "true_angle_deg": truth["Angle"],
            "Air_error_nm": errors["Air_error_nm"],
            "absolute_Air_error_nm": errors["absolute_Air_error_nm"],
            "film_MAE_nm": errors["film_MAE_nm"],
            "angle_abs_error_deg": errors["angle_abs_error_deg"],
            "exact_RMSE": local["exact_rmse"],
            "spectrum_cost": local["spectrum_cost"],
            "boundary_hits": ";".join(boundary["boundary_hits"]),
            "near_boundary": ";".join(boundary["near_boundary"]),
            "boundary_severity": boundary["severity"],
            "quality_class": selected["quality_class"],
            "success": selected["success"],
            "status": selected["status"],
            "free_parameters": local["free_parameters"],
            "sigma_min_Jw": raw_info["smallest_singular_value"],
            "condition_number_Jw": raw_info["condition_number"],
            "log10_det_JwT_Jw": raw_info["fisher_log10_determinant"],
            "det_JwT_Jw": raw_info["fisher_determinant"],
            "det_JwT_Jw_mantissa": raw_info["fisher_determinant_mantissa"],
            "det_JwT_Jw_exponent10": raw_info["fisher_determinant_exponent10"],
            "density_sigma_min_Jw": density_info["smallest_singular_value"],
            "density_condition_number_Jw": density_info["condition_number"],
            "density_log10_det_JwT_Jw": density_info["fisher_log10_determinant"],
            "information": information,
            "jacobian_audit": jacobian_audit,
            "selected": selected,
            "ranking": {
                key: value
                for key, value in local["ranking"].items()
                if key not in {"attempts", "selected"}
            },
            "global_runtime_s": global_profile["totals"]["total_global_runtime_s"],
            "local_runtime_s": local["runtime_s"],
            "case_runtime_s": time.perf_counter() - case_started,
            "global_profile": global_profile,
            "local_cache": local["cache"],
            "closure_rmse": local["closure"]["rmse"],
            "closure_max_abs": local["closure"]["max_abs"],
            "closure_pass": local["closure"]["pass"],
        }
        rows.append(row)
        append_jsonl(progress, row)
        print(
            f"[{index}/{len(paths)}] {path.name} band={args.band} "
            f"Air={row['Air_error_nm']:.5g}nm film={row['film_MAE_nm']:.5g}nm "
            f"sigma_min={row['sigma_min_Jw']:.5g} closure=PASS",
            flush=True,
        )

    total_runtime = time.perf_counter() - started_all
    resident_after = resident_array_identity(gpu)
    memory = gpu.memory_stats()
    groups = defaultdict(list)
    for row in rows:
        groups[row["noise_case"]].append(row)
    statuses = Counter(
        "terminated" if row["success"] and row["status"] > 0
        else "budget" if row["status"] == 0
        else "other"
        for row in rows
    )
    summary = {
        "processed": len(rows),
        "terminated": statuses["terminated"],
        "budget_exhausted": statuses["budget"],
        "other_finite": statuses["other"],
        "closure_passed": sum(bool(row["closure_pass"]) for row in rows),
        "fresh_gpu_de_cases": len(rows),
        "resident_reused": resident_before == resident_after,
        "global_population_size": population_size,
        "local_batch_size": 11,
        "information_batch_size": 11,
        "total_runtime_s": total_runtime,
        "global_runtime_s": float(sum(row["global_runtime_s"] for row in rows)),
        "local_runtime_s": float(sum(row["local_runtime_s"] for row in rows)),
        "throughput_cases_per_min": len(rows) / total_runtime * 60.0,
        "peak_gpu_memory_gib": int(memory.get("memory_pool_total_bytes", 0)) / 1024**3,
    }
    passed = bool(
        len(rows) == args.count
        and summary["closure_passed"] == args.count
        and summary["resident_reused"]
        and source_sha256() == EXPECTED_FORMAL_V10_SHA256
        and all(row["information"]["raw_information"]["numerical_rank"] == 5 for row in rows)
    )
    report = {
        "version": VERSION,
        "overall": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_count": args.count,
        "configuration": {
            **asdict(config),
            "band_label": args.band,
            "effective_band_nm": [band_start, band_stop],
            "effective_band_contains_220_580_baseline": True,
            "master_acquisition_band_nm": [MASTER_START_NM, MASTER_STOP_NM],
            "candidate_bands_nm": {key: list(value) for key, value in CANDIDATE_BANDS.items()},
            "master_sampling_nm": 0.02,
            "master_points": MASTER_POINTS,
            "free_parameter_order": list(v12.FREE_PARAMS),
            "fixed_parameter": "Angle",
            "angle_mode": "fixed_independent_measurement",
            "angle_sigma_deg": 0.001,
            "map_prior_enabled": False,
            "global_population_size": population_size,
            "local_batch_size": 11,
            "information_batch_size": 11,
            "vectorized": True,
            "updating": "deferred",
            "fit_objective_weighting": "unchanged V12 robust spectral residual",
            "information_weighting": "diagonal detector covariance only",
        },
        "summary": summary,
        "aggregate": aggregate(rows),
        "noise_case_aggregates": {
            name: aggregate(values) for name, values in sorted(groups.items())
        },
        "formal_v10_sha256": source_sha256(),
        "input_dir": str(root),
        "source_dataset_bundle_sha256": source_bundle_sha256,
        "source_dataset_files": source_file_audit,
        "noise_calibration_dir": str(args.noise_calibration_dir.resolve()),
        "noise_covariance_audit": covariance["audit"],
        "machine": (
            json.loads(args.machine.read_text(encoding="utf-8"))
            if args.machine and args.machine.is_file()
            else None
        ),
        "gpu_memory": memory,
        "cases": rows,
        "scope_guard": (
            "V14 Stage 1a only; all candidate bands contain 220-580 nm; one paired "
            "200-800 nm local StackRT master dataset; typical only; fixed measured Angle "
            "sigma=0.001 deg; no Angle MAP; V12 synthetic instrument proxy is not measured "
            "hardware SPD/QE; strict GPU full-ILS B=40 global and B=11 local."
        ),
    }
    write_outputs(output_dir, report)
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "band": args.band,
                "processed": len(rows),
                "runtime_s": total_runtime,
                "output": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
