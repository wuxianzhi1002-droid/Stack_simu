"""V17 200-600 nm fixed-angle combined-typical GPU ablation runner."""
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
from typing import Any
import numpy as np
import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import load_v10_module, source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from v12_optimizer.fixed_angle_gpu import run_gpu_global, run_gpu_local
from v16_optimizer.band_runner import append_jsonl, compact_selected, dataset_bundle_hash, finite_stats, load_v15_fit_input, safe
from v16_optimizer.eq99x_spectrometer import Eq99xCupyStrictSpectrometerBackend, Eq99xNumpyStrictSpectrometerBackend
from v16_optimizer.information import signal_jacobian_normalized_solver, whitened_information_metrics
from v16_optimizer.noise_covariance import estimate_diagonal_sigma

VERSION = "v17_combined_typical_one_factor_reduction_gpu"
EXPECTED_V10 = "d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"
START_NM, STOP_NM, POINTS = 200.0, 600.0, 20001
SCENARIOS = (
    "baseline", "reduce_angle_measurement", "reduce_source_center_drift", "reduce_source_power",
    "reduce_axis_offset", "reduce_axis_scale", "reduce_thermal_drift",
    "reduce_absolute_accuracy", "reduce_material", "reduce_detector",
)
REDUCED_KEYS = {
    "baseline": (), "reduce_angle_measurement": (),
    "reduce_source_center_drift": ("source_center_drift_max_nm",),
    "reduce_source_power": ("source_power_curve_peak_rel",),
    "reduce_axis_offset": ("axis_offset_max_nm",),
    "reduce_axis_scale": ("axis_scale_max_ppm",),
    "reduce_thermal_drift": ("thermal_drift_max_nm",),
    "reduce_absolute_accuracy": ("calibration_residual_max_nm",),
    "reduce_material": ("n_real_sigma_rel", "k_sigma_rel"),
    "reduce_detector": ("detector_noise_multiplier",),
}

def scalar(data, key, default=None):
    return np.asarray(data[key]).item() if key in data else default

def ablation_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "ablation_scenario": str(scalar(data, "ablation_scenario")),
            "reduced_factor": str(scalar(data, "reduced_factor")),
            "reduction_fraction": float(scalar(data, "reduction_fraction")),
            "angle_measurement_sigma_deg": float(scalar(data, "angle_measurement_sigma_deg")),
            "angle_measurement_error_deg": float(scalar(data, "angle_measurement_error_deg")),
            "true_angle_deg": float(scalar(data, "true_reflector_angle_deg")),
            "measured_angle_deg": float(scalar(data, "measured_reflector_angle_deg")),
            "realization_index": int(scalar(data, "realization_index")),
            "paired_realization_seed": int(scalar(data, "paired_realization_seed")),
            "baseline_profile": json.loads(str(scalar(data, "baseline_typical_profile_json"))),
            "scenario_profile": json.loads(str(scalar(data, "scenario_typical_profile_json"))),
        }

def validate_input(path: Path) -> dict[str, Any]:
    meta = ablation_metadata(path)
    with np.load(path, allow_pickle=False) as data:
        axis = np.asarray(data["reported_wavelengths_nm"], dtype=np.float64)
        if axis.shape != (POINTS,) or axis[0] != START_NM or axis[-1] != STOP_NM:
            raise RuntimeError(f"bad 200-600 axis: {path.name}")
        if str(scalar(data, "generator_version")) != "main_v15" or str(scalar(data, "optical_backend")) != "api":
            raise RuntimeError(f"bad StackRT provenance: {path.name}")
        if str(scalar(data, "noise_case")) != "combined_typical":
            raise RuntimeError(f"not combined typical: {path.name}")
        config = json.loads(str(scalar(data, "config_json")))
        if config.get("SOURCE_MODEL") != "eq99x_digitized_peak_normalized" or not config.get("EQ99X_SOURCE_CSV_SHA256"):
            raise RuntimeError(f"bad EQ-99X config: {path.name}")
    scenario = meta["ablation_scenario"]
    if scenario not in SCENARIOS:
        raise RuntimeError(f"bad scenario: {scenario}")
    expected_sigma = 0.001 if scenario == "reduce_angle_measurement" else 0.01
    if not np.isclose(meta["angle_measurement_sigma_deg"], expected_sigma, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"bad angle sigma: {path.name}")
    if not (0.1 <= meta["true_angle_deg"] <= 0.2):
        raise RuntimeError(f"bad true angle: {path.name}")
    expected_fraction = 1.0 if scenario == "baseline" else 0.1
    if not np.isclose(meta["reduction_fraction"], expected_fraction, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"bad reduction fraction: {path.name}")
    baseline_profile = meta["baseline_profile"]
    scenario_profile = meta["scenario_profile"]
    if set(baseline_profile) != set(scenario_profile):
        raise RuntimeError(f"profile keys changed: {path.name}")
    expected_keys = set(REDUCED_KEYS[scenario])
    for key, baseline_value in baseline_profile.items():
        expected_value = float(baseline_value) * (0.1 if key in expected_keys else 1.0)
        if not np.isclose(float(scenario_profile[key]), expected_value, rtol=0.0, atol=1e-18):
            raise RuntimeError(f"profile mismatch {scenario} {key}: {path.name}")
    return meta

def aggregate(rows):
    metrics = ("absolute_Air_error_nm", "film_MAE_nm", "exact_RMSE", "sigma_min_Jw", "condition_number_Jw", "log10_det_JwT_Jw")
    return {key: finite_stats(row[key] for row in rows) for key in metrics} | {
        "boundary_hit_rate": float(np.mean([bool(row["boundary_hits"]) for row in rows])),
        "terminated_rate": float(np.mean([bool(row["success"]) and int(row["status"]) > 0 for row in rows])),
    }

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--noise-calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--require-production-count", action="store_true")
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--max-nfev", type=int, default=600)
    return parser.parse_args()

def main():
    args = parse_args()
    if source_sha256() != EXPECTED_V10:
        raise RuntimeError("formal V10 source changed")
    if args.require_production_count and args.count != 100:
        raise ValueError("production count must be 100")
    root = args.input_dir.resolve()
    all_paths = sorted(root.glob("*__static_spectrum_combined_typical_r*.npz"), key=lambda item: item.name)
    if len(all_paths) != 100:
        raise RuntimeError(f"expected complete 100-file source matrix, got {len(all_paths)}")
    paths = all_paths[: int(args.count)]
    if len(paths) != args.count:
        raise RuntimeError(f"requested {args.count} files, got {len(paths)}")
    metadata = {path: validate_input(path) for path in paths}
    counts = Counter(item["ablation_scenario"] for item in metadata.values())
    if args.count == 100 and counts != Counter({scenario: 10 for scenario in SCENARIOS}):
        raise RuntimeError(f"scenario counts changed: {counts}")
    paired = defaultdict(list)
    for path, item in metadata.items():
        paired[item["realization_index"]].append((path, item))
    if args.count == 100 and (set(paired) != set(range(10)) or any(len(items) != 10 for items in paired.values())):
        raise RuntimeError("paired realization matrix incomplete")
    for realization, items in paired.items():
        if args.count != 100:
            continue
        true_angles = np.asarray([item[1]["true_angle_deg"] for item in items])
        seeds = {item[1]["paired_realization_seed"] for item in items}
        if len(seeds) != 1 or not np.allclose(true_angles, true_angles[0], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"pairing failed at realization {realization}")
        by_scenario = {item[1]["ablation_scenario"]: item[1] for item in items}
        base_error = by_scenario["baseline"]["angle_measurement_error_deg"]
        reduced_error = by_scenario["reduce_angle_measurement"]["angle_measurement_error_deg"]
        if not np.isclose(reduced_error, 0.1 * base_error, rtol=0.0, atol=1e-15):
            raise RuntimeError(f"angle error pairing failed at realization {realization}")
        for scenario, item in by_scenario.items():
            if scenario != "reduce_angle_measurement" and not np.isclose(item["angle_measurement_error_deg"], base_error, rtol=0.0, atol=1e-15):
                raise RuntimeError(f"measurement pairing failed: {scenario} r{realization}")
    source_bundle, source_rows = dataset_bundle_hash(paths)
    covariance = estimate_diagonal_sigma(args.noise_calibration_dir)
    cov_axis = np.asarray(covariance["wavelengths_nm"], dtype=np.float64)
    mask = (cov_axis >= START_NM) & (cov_axis <= STOP_NM)
    sigma = np.asarray(covariance["sigma"], dtype=np.float64)[mask]
    config = v12.FitConfig(
        input_dir=str(root), wavelength_min_nm=START_NM, wavelength_max_nm=STOP_NM,
        global_forward_model="full_ils", global_popsize=8, global_maxiter=int(args.global_maxiter),
        multistarts=8, max_nfev=int(args.max_nfev), workers=1, random_seed=20260906,
    )
    first = load_v15_fit_input(paths[0], config)
    selected_nm = np.asarray(first["wavelengths_um"]) * 1000.0
    if sigma.shape != selected_nm.shape or not np.allclose(cov_axis[mask], selected_nm, rtol=0.0, atol=1e-10):
        raise RuntimeError("calibration axis mismatch")
    gpu = Eq99xCupyStrictSpectrometerBackend(first["wavelengths_um"], first["generator_config"], first["metadata"]["internal_wavelength_margin_nm"])
    cpu = Eq99xNumpyStrictSpectrometerBackend(first["wavelengths_um"], first["generator_config"], first["metadata"]["internal_wavelength_margin_nm"])
    resident_before = resident_array_identity(gpu)
    v10 = load_v10_module(); population_size = config.global_popsize * len(v12.FREE_PARAMS)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    progress = output / "v17_ablation_progress.jsonl"
    rows = []; started = time.perf_counter()
    for index, path in enumerate(paths, 1):
        case_started = time.perf_counter(); meta = metadata[path]
        measurement = load_v15_fit_input(path, config)
        if not np.array_equal(measurement["wavelengths_um"], first["wavelengths_um"]):
            raise RuntimeError(f"wavelength contract changed: {path.name}")
        seed = int(config.random_seed + meta["realization_index"] * 1009)
        case_config = v12.FitConfig(**{**asdict(config), "random_seed": seed})
        population = v12.latin_hypercube_population(seed, population_size)
        _, de_result, candidates, global_profile = run_gpu_global(v10, measurement, case_config, population, gpu)
        differential = global_profile["differential_evolution"]
        if differential["min_batch_size"] != population_size or differential["max_batch_size"] != population_size or differential["candidate_evaluations"] != int(de_result.nfev) * population_size:
            raise RuntimeError(f"global batch contract failed: {path.name}")
        local = run_gpu_local(v10, measurement, case_config, candidates, gpu, cpu); local.pop("prediction")
        selected = compact_selected(local["selected"])
        signal_jacobian, jac_audit = signal_jacobian_normalized_solver(gpu, local["free_parameters"], measurement["fixed_angle_deg"])
        information = whitened_information_metrics(signal_jacobian, sigma)
        truth, _ = v12.load_evaluation_truth(path); errors = v12.scientific_errors(local["full_parameters"], truth)
        boundary = selected["boundary"]
        row = {
            "index": index, "filename": path.name, "ablation_scenario": meta["ablation_scenario"],
            "reduced_factor": meta["reduced_factor"], "reduction_fraction": meta["reduction_fraction"],
            "realization_index": meta["realization_index"], "data_seed": meta["paired_realization_seed"], "optimizer_seed": seed,
            "true_angle_deg": meta["true_angle_deg"], "measured_angle_deg": measurement["fixed_angle_deg"],
            "angle_measurement_error_deg": meta["angle_measurement_error_deg"], "angle_measurement_sigma_deg": meta["angle_measurement_sigma_deg"],
            "Air_error_nm": errors["Air_error_nm"], "absolute_Air_error_nm": errors["absolute_Air_error_nm"],
            "film_MAE_nm": errors["film_MAE_nm"], "exact_RMSE": local["exact_rmse"],
            "boundary_hits": ";".join(boundary["boundary_hits"]), "near_boundary": ";".join(boundary["near_boundary"]),
            "success": selected["success"], "status": selected["status"], "quality_class": selected["quality_class"],
            "sigma_min_Jw": information["raw_information"]["smallest_singular_value"],
            "condition_number_Jw": information["raw_information"]["condition_number"],
            "log10_det_JwT_Jw": information["raw_information"]["fisher_log10_determinant"],
            "free_parameters": local["free_parameters"], "selected": selected,
            "global_runtime_s": global_profile["totals"]["total_global_runtime_s"], "local_runtime_s": local["runtime_s"],
            "case_runtime_s": time.perf_counter() - case_started, "closure_rmse": local["closure"]["rmse"],
            "closure_max_abs": local["closure"]["max_abs"], "closure_pass": local["closure"]["pass"],
            "global_profile": global_profile, "local_cache": local["cache"], "information": information, "jacobian_audit": jac_audit,
        }
        rows.append(row); append_jsonl(progress, row)
        print(f"[{index}/100] {meta['ablation_scenario']} r={meta['realization_index']} Air={row['absolute_Air_error_nm']:.5g} film={row['film_MAE_nm']:.5g}", flush=True)
    elapsed = time.perf_counter() - started; resident_after = resident_array_identity(gpu)
    groups = defaultdict(list)
    for row in rows: groups[row["ablation_scenario"]].append(row)
    summary = {
        "processed": len(rows), "terminated": sum(bool(row["success"]) and int(row["status"]) > 0 for row in rows),
        "closure_passed": sum(bool(row["closure_pass"]) for row in rows), "resident_reused": resident_before == resident_after,
        "runtime_s": elapsed, "global_runtime_s": float(sum(row["global_runtime_s"] for row in rows)),
        "local_runtime_s": float(sum(row["local_runtime_s"] for row in rows)), "throughput_cases_per_min": len(rows) / elapsed * 60.0,
        "population_size": population_size, "fresh_population_case_count": len(rows),
        "paired_optimizer_seed_policy": "same seed for same realization across scenarios; a fresh population object per case",
        "peak_gpu_memory_gib": int(gpu.memory_stats().get("memory_pool_total_bytes", 0)) / 1024**3,
    }
    passed = len(rows) == args.count and summary["closure_passed"] == args.count and summary["resident_reused"] and all(len(values) == (10 if args.count == 100 else 1) for values in groups.values())
    report = {
        "version": VERSION, "overall": "PASS" if passed else "FAIL", "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {**asdict(config), "effective_band_nm": [START_NM, STOP_NM], "source_model": "digitized EQ-99X", "angle_mode": "fixed independent measurement", "true_angle_range_deg": [0.1, 0.2], "baseline_angle_sigma_deg": 0.01, "reduced_angle_sigma_deg": 0.001, "reduction_fraction": 0.1, "scenarios": list(SCENARIOS), "vectorized": True, "updating": "deferred", "angle_MAP": False},
        "summary": summary, "scenario_aggregates": {name: aggregate(values) for name, values in sorted(groups.items())},
        "source_bundle_sha256": source_bundle, "source_files": source_rows, "noise_covariance_audit": covariance["audit"],
        "machine": json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,
        "cases": rows, "scope_guard": "V17 single-angle 200-600 nm combined-typical paired one-factor 0.1x reduction; ten realizations per scenario; fixed measured angle; no multi-angle and no Angle MAP; strict GPU full ILS.",
    }
    (output / "v17_ablation_results.json").write_text(json.dumps(safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["index","filename","ablation_scenario","reduced_factor","realization_index","data_seed","optimizer_seed","true_angle_deg","measured_angle_deg","angle_measurement_error_deg","angle_measurement_sigma_deg","Air_error_nm","absolute_Air_error_nm","film_MAE_nm","exact_RMSE","boundary_hits","near_boundary","success","status","quality_class","sigma_min_Jw","condition_number_Jw","log10_det_JwT_Jw","global_runtime_s","local_runtime_s","case_runtime_s","closure_rmse","closure_max_abs","closure_pass"]
    with (output / "v17_ablation_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in rows)
    baseline = report["scenario_aggregates"]["baseline"]
    lines = ["# V17 combined typical one-factor reduction", "", f"- Overall: **{report['overall']}**", f"- Cases/closure: {len(rows)}/{summary['closure_passed']}", f"- Runtime: {elapsed:.3f} s", "", "| scenario | Air abs mean nm | film MAE mean nm | Air improvement vs baseline nm | film improvement vs baseline nm |", "|---|---:|---:|---:|---:|"]
    for scenario in sorted(report["scenario_aggregates"]):
        item = report["scenario_aggregates"][scenario]
        lines.append(f"| {scenario} | {item['absolute_Air_error_nm']['mean']:.6g} | {item['film_MAE_nm']['mean']:.6g} | {baseline['absolute_Air_error_nm']['mean']-item['absolute_Air_error_nm']['mean']:.6g} | {baseline['film_MAE_nm']['mean']-item['film_MAE_nm']['mean']:.6g} |")
    (output / "v17_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "processed": len(rows), "runtime_s": elapsed}, indent=2))
    if not passed: raise SystemExit(2)

if __name__ == "__main__":
    main()
