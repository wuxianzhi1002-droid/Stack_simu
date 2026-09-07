"""Generate 50 paired V17 StackRT cases with wavelength-calibration and angle errors reduced together."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

START_NM = 200.0
STOP_NM = 600.0
SAMPLING_NM = 0.02
POINTS = 20001
REALIZATION_START = 0
REALIZATION_STOP = 50
REALIZATIONS_PER_SCENARIO = REALIZATION_STOP - REALIZATION_START
TOTAL_CASES = REALIZATIONS_PER_SCENARIO
BASE_ANGLE_SIGMA_DEG = 0.01
REDUCED_ANGLE_SIGMA_DEG = 0.001
REDUCTION_FRACTION = 0.1
SCENARIOS = {
    "reduce_absolute_accuracy_and_angle_measurement": ("calibration_residual_max_nm",),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_main_v15():
    source = Path(__file__).resolve().parents[3] / "01_simulation_models" / "01_Lumerical_Workflow" / "main_v15.py"
    spec = importlib.util.spec_from_file_location("_v17_ablation_main_v15", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source

def augment(path: Path, scenario: str, reduced_keys: tuple[str, ...], baseline: dict, active: dict, seed: int) -> None:
    with np.load(path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    config = json.loads(str(payload["config_json"].item()))
    payload.update({
        "eq99x_source_csv_sha256": np.asarray(config["EQ99X_SOURCE_CSV_SHA256"]),
        "ablation_scenario": np.asarray(scenario),
        "reduced_factor": np.asarray(scenario.removeprefix("reduce_") if scenario.startswith("reduce_") else "none"),
        "reduced_profile_keys": np.asarray(reduced_keys, dtype="U64"),
        "reduction_fraction": np.asarray(1.0 if scenario == "baseline" else REDUCTION_FRACTION),
        "baseline_typical_profile_json": np.asarray(json.dumps(baseline, sort_keys=True)),
        "scenario_typical_profile_json": np.asarray(json.dumps(active, sort_keys=True)),
        "baseline_angle_measurement_sigma_deg": np.asarray(BASE_ANGLE_SIGMA_DEG),
        "paired_realization_seed": np.asarray(seed),
        "ablation_pairing_contract": np.asarray("same base seed and RNG draw order as baseline; calibration residual and angle measurement amplitudes both reduced to 0.1x"),
    })
    temporary = path.with_name(path.stem + ".v17tmp.npz")
    np.savez_compressed(temporary, **payload)
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)

INDEX_FIELDS = (
    "noise_case", "noise_factor", "noise_level", "realization_index", "random_seed",
    "true_reflector_angle_deg", "reflector_angle_setpoint_deg", "reflector_angle_deviation_from_nominal_deg",
    "measured_reflector_angle_deg", "measured_angle_deg", "angle_measurement_error_deg",
    "angle_measurement_error_definition", "angle_measurement_sigma_deg", "angle_measurement_mode",
    "angle_measurement_visible_to_inversion", "true_angle_visible_to_inversion",
    "legacy_kernel_noise_angle_deg_audit_only", "reported_axis_is_fixed", "axis_offset_nm", "axis_scale_ppm",
    "thermal_drift_nm", "calibration_residual_peak_nm", "spectrometer_axis_error_max_abs_nm",
    "spectrometer_axis_error_rms_nm", "physical_axis_shift_max_abs_nm", "axis_estimation_error_max_abs_nm",
    "axis_estimation_error_rms_nm", "wavelength_accuracy_within_spec", "source_center_drift_nm",
    "source_power_curve_peak_rel", "detector_noise_multiplier", "target_reference_peak_electrons",
    "sample_saturation_fraction", "reference_saturation_fraction", "invalid_reference_pixel_fraction",
    "output_sampling_nm", "internal_wavelength_step_nm", "optical_solver_runtime_s", "generation_runtime_s",
    "npz_path", "ablation_scenario", "reduced_factor", "reduction_fraction", "sha256",
)

def row_from_path(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"ablation_scenario", "reduced_factor", "reduction_fraction", "ablation_pairing_contract"}
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"incomplete resumed NPZ {path}: {sorted(missing)}")
        audit = json.loads(str(data["noise_realization_json"]))
        row = {key: audit.get(key, "") for key in INDEX_FIELDS}
        row.update({
            "true_reflector_angle_deg": float(data["true_reflector_angle_deg"]),
            "reflector_angle_setpoint_deg": float(data["reflector_angle_setpoint_deg"]),
            "reflector_angle_deviation_from_nominal_deg": float(data["reflector_angle_deviation_from_nominal_deg"]),
            "measured_reflector_angle_deg": float(data["measured_reflector_angle_deg"]),
            "measured_angle_deg": float(data["measured_angle_deg"]),
            "angle_measurement_error_deg": float(data["angle_measurement_error_deg"]),
            "angle_measurement_sigma_deg": float(data["angle_measurement_sigma_deg"]),
            "angle_measurement_mode": str(data["angle_measurement_mode"]),
            "reported_axis_is_fixed": bool(data["reported_axis_is_fixed"]),
            "realization_index": int(data["realization_index"]),
            "random_seed": int(data["random_seed"]),
            "npz_path": str(path.resolve()),
            "ablation_scenario": str(data["ablation_scenario"]),
            "reduced_factor": str(data["reduced_factor"]),
            "reduction_fraction": float(data["reduction_fraction"]),
            "sha256": sha256(path),
        })
    return {key: row[key] for key in INDEX_FIELDS}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()

def main():
    args = parse_args()
    main_v15, source_path = load_main_v15()
    main_v15.configure_v15()
    main_v15.apply_wavelength_cli_overrides(argparse.Namespace(
        wavelength_start_nm=START_NM, wavelength_stop_nm=STOP_NM, output_sampling_nm=SAMPLING_NM
    ))
    main_v15.configure_v15()
    kernel = main_v15._kernel
    baseline_profile = copy.deepcopy(kernel.NOISE_LEVELS["typical"])
    original_sigma = float(main_v15.V15_CONFIG["ANGLE_MEASUREMENT"]["STANDARD_UNCERTAINTY_DEG"])
    combined_index = kernel.all_case_names().index("combined_typical")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    try:
        with kernel.OpticalSolver("api", output / "_stackrt_batch_bridge") as solver:
            for scenario, reduced_keys in SCENARIOS.items():
                scenario_dir = output / scenario
                scenario_dir.mkdir(exist_ok=True)
                profile = copy.deepcopy(baseline_profile)
                for key in reduced_keys:
                    profile[key] = float(profile[key]) * REDUCTION_FRACTION
                angle_sigma = REDUCED_ANGLE_SIGMA_DEG
                kernel.NOISE_LEVELS["typical"] = profile
                main_v15.V15_CONFIG["ANGLE_MEASUREMENT"]["STANDARD_UNCERTAINTY_DEG"] = angle_sigma
                main_v15.configure_v15()
                generator = main_v15.V15DatasetGenerator("api", scenario_dir)
                for realization in range(REALIZATION_START, REALIZATION_STOP):
                    seed = int(args.seed + combined_index * 1_000_000 + realization)
                    target = scenario_dir / f"static_spectrum_combined_typical_r{realization:04d}_seed{seed}.npz"
                    if target.is_file():
                        row = row_from_path(target)
                        if row["ablation_scenario"] != scenario or row["realization_index"] != realization or row["random_seed"] != seed:
                            raise RuntimeError(f"resume identity mismatch: {target}")
                        rows.append(row)
                        print(f"[{len(rows)}/{TOTAL_CASES}] REUSE {scenario} r={realization}", flush=True)
                        continue
                    try:
                        data, metadata = generator.generate_one(solver, "combined_typical", realization, seed)
                        path = generator.save_one(data)
                        augment(path, scenario, reduced_keys, baseline_profile, profile, seed)
                        row = row_from_path(path)
                        rows.append(row)
                        print(f"[{len(rows)}/{TOTAL_CASES}] {scenario} r={realization} angle={metadata['reflector_angle_deg']:.6f} measured={metadata['measured_reflector_angle_deg']:.6f}", flush=True)
                    except Exception as exc:
                        failures.append({"scenario": scenario, "realization": realization, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                        raise
    finally:
        kernel.NOISE_LEVELS["typical"] = baseline_profile
        main_v15.V15_CONFIG["ANGLE_MEASUREMENT"]["STANDARD_UNCERTAINTY_DEG"] = original_sigma
        main_v15.configure_v15()
    counts = Counter(row["ablation_scenario"] for row in rows)
    if len(rows) != TOTAL_CASES or set(counts.values()) != {REALIZATIONS_PER_SCENARIO} or failures:
        raise RuntimeError({"rows": len(rows), "counts": counts, "failures": failures})
    for realization in range(REALIZATION_START, REALIZATION_STOP):
        paired = [row for row in rows if row["realization_index"] == realization]
        true_angles = np.asarray([row["true_reflector_angle_deg"] for row in paired], dtype=float)
        if not np.allclose(true_angles, true_angles[0], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"true-angle pairing failed at realization {realization}: {true_angles}")
    with (output / "dataset_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_FIELDS))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "version": "v17_combined_typical_joint_calibration_angle_reduction_stackrt_r0_r49",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": "PASS",
        "dataset_count": len(rows),
        "scenario_count": len(SCENARIOS),
        "realizations_per_scenario": REALIZATIONS_PER_SCENARIO,
        "realization_start_inclusive": REALIZATION_START,
        "realization_stop_exclusive": REALIZATION_STOP,
        "target_total_realizations_per_scenario_after_merge": 50,
        "scenarios": {key: list(value) for key, value in SCENARIOS.items()},
        "baseline_profile": baseline_profile,
        "reduction_fraction": REDUCTION_FRACTION,
        "angle_sigma_baseline_deg": BASE_ANGLE_SIGMA_DEG,
        "angle_sigma_reduced_deg": REDUCED_ANGLE_SIGMA_DEG,
        "true_angle_range_deg": [0.1, 0.2],
        "angle_mode": "fixed_independent_measurement",
        "wavelength_nm": [START_NM, STOP_NM],
        "sampling_nm": SAMPLING_NM,
        "points": POINTS,
        "noise_case": "combined_typical",
        "source_model": "digitized EQ-99X source_shape_peak_normalized",
        "eq99x_source_csv_sha256": main_v15.V15_CONFIG["EQ99X_SOURCE_CSV_SHA256"],
        "optical_backend": "Lumerical lumapi.FDTD.stackrt",
        "seed_base": args.seed,
        "combined_case_index": combined_index,
        "pairing": "same data seed and realization indices as the existing 50-pair baseline; calibration residual and angle measurement uncertainty both reduced to 0.1x",
        "main_v15": str(source_path),
        "main_v15_sha256": sha256(source_path),
        "failures": failures,
    }
    (output / "simulation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": "PASS", "count": len(rows), "output": str(output)}, indent=2))

if __name__ == "__main__":
    main()
