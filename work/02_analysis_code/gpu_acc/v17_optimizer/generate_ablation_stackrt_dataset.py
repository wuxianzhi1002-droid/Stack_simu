"""Generate paired V17 combined-typical one-factor-reduction StackRT data."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

START_NM = 200.0
STOP_NM = 600.0
SAMPLING_NM = 0.02
POINTS = 20001
BASE_ANGLE_SIGMA_DEG = 0.01
REDUCED_ANGLE_SIGMA_DEG = 0.001
REDUCTION_FRACTION = 0.1
SCENARIOS = {
    "baseline": (),
    "reduce_angle_measurement": (),
    "reduce_source_center_drift": ("source_center_drift_max_nm",),
    "reduce_source_power": ("source_power_curve_peak_rel",),
    "reduce_axis_offset": ("axis_offset_max_nm",),
    "reduce_axis_scale": ("axis_scale_max_ppm",),
    "reduce_thermal_drift": ("thermal_drift_max_nm",),
    "reduce_absolute_accuracy": ("calibration_residual_max_nm",),
    "reduce_material": ("n_real_sigma_rel", "k_sigma_rel"),
    "reduce_detector": ("detector_noise_multiplier",),
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
        "ablation_pairing_contract": np.asarray("same base seed and RNG draw order across scenarios; one amplitude family changed per scenario"),
    })
    temporary = path.with_name(path.stem + ".v17tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.repeats != 10:
        raise ValueError("formal V17 requires ten realizations per scenario")
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
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    failures = []
    try:
        with kernel.OpticalSolver("api", output / "_stackrt_batch_bridge") as solver:
            for scenario, reduced_keys in SCENARIOS.items():
                scenario_dir = output / scenario
                scenario_dir.mkdir()
                profile = copy.deepcopy(baseline_profile)
                for key in reduced_keys:
                    profile[key] = float(profile[key]) * REDUCTION_FRACTION
                angle_sigma = REDUCED_ANGLE_SIGMA_DEG if scenario == "reduce_angle_measurement" else BASE_ANGLE_SIGMA_DEG
                kernel.NOISE_LEVELS["typical"] = profile
                main_v15.V15_CONFIG["ANGLE_MEASUREMENT"]["STANDARD_UNCERTAINTY_DEG"] = angle_sigma
                main_v15.configure_v15()
                generator = main_v15.V15DatasetGenerator("api", scenario_dir)
                for realization in range(args.repeats):
                    seed = int(args.seed + combined_index * 1_000_000 + realization)
                    try:
                        data, metadata = generator.generate_one(solver, "combined_typical", realization, seed)
                        path = generator.save_one(data)
                        augment(path, scenario, reduced_keys, baseline_profile, profile, seed)
                        row = main_v15.dataset_index_row(metadata, path)
                        row.update({
                            "ablation_scenario": scenario,
                            "reduced_factor": scenario.removeprefix("reduce_") if scenario.startswith("reduce_") else "none",
                            "reduction_fraction": 1.0 if scenario == "baseline" else REDUCTION_FRACTION,
                            "angle_measurement_sigma_deg": angle_sigma,
                            "sha256": sha256(path),
                        })
                        rows.append(row)
                        print(f"[{len(rows)}/100] {scenario} r={realization} angle={metadata['reflector_angle_deg']:.6f} measured={metadata['measured_reflector_angle_deg']:.6f}", flush=True)
                    except Exception as exc:
                        failures.append({"scenario": scenario, "realization": realization, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                        raise
    finally:
        kernel.NOISE_LEVELS["typical"] = baseline_profile
        main_v15.V15_CONFIG["ANGLE_MEASUREMENT"]["STANDARD_UNCERTAINTY_DEG"] = original_sigma
        main_v15.configure_v15()
    counts = Counter(row["ablation_scenario"] for row in rows)
    if len(rows) != 100 or set(counts.values()) != {10} or failures:
        raise RuntimeError({"rows": len(rows), "counts": counts, "failures": failures})
    for realization in range(args.repeats):
        paired = [row for row in rows if row["realization_index"] == realization]
        true_angles = np.asarray([row["true_reflector_angle_deg"] for row in paired], dtype=float)
        if not np.allclose(true_angles, true_angles[0], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"true-angle pairing failed at realization {realization}: {true_angles}")
    with (output / "dataset_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "version": "v17_combined_typical_one_factor_reduction_stackrt",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": "PASS",
        "dataset_count": len(rows),
        "scenario_count": len(SCENARIOS),
        "realizations_per_scenario": args.repeats,
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
        "pairing": "same data seed for the same realization across all scenarios",
        "main_v15": str(source_path),
        "main_v15_sha256": sha256(source_path),
        "failures": failures,
    }
    (output / "simulation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": "PASS", "count": len(rows), "output": str(output)}, indent=2))

if __name__ == "__main__":
    main()
