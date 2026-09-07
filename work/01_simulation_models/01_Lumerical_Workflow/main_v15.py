"""V15 single-frame dataset generator with an independent angle measurement.

V15 owns a complete configuration.  It reuses the audited V10 physical/noise
implementation through a private module instance, but never reads or mutates
``main_v10.CONFIG`` in the normal imported module.

Default V15 contract:
* clean: true set angle a and measured angle b are both 0 deg
* non-clean: a is sampled uniformly in [0.1, 0.2] deg
* measured angle is b = a + e with e ~ N(0, sigma^2), sigma = 0.001 deg
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "main_v15"
EQ99X_COLUMN = "source_shape_peak_normalized"
ANGLE_SETPOINT_MIN_DEG = 0.1
ANGLE_SETPOINT_MAX_DEG = 0.2
ANGLE_MEASUREMENT_SIGMA_DEG = 0.001


def _load_private_v10_kernel():
    """Load V10 code under a private name so its globals cannot leak into V10."""
    module_name = "_stackrt_main_v10_kernel_for_v15"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = Path(__file__).with_name("main_v10.py")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load private V10 kernel from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_kernel = _load_private_v10_kernel()

# ---------------------------------------------------------------------------
# Complete V15-owned config.  Do not derive this dictionary from V10 CONFIG.
# ---------------------------------------------------------------------------
V15_CONFIG = {
    "MODEL_TYPE": "stackrt_v15",
    "SOURCE_MODEL": "eq99x_digitized_peak_normalized",
    "WAVELENGTH_AXIS_SCENARIO": "fixed_reported_axis_hidden_true_pixel_centers",
    "WAVELENGTH_START_UM": 0.220,
    "WAVELENGTH_STOP_UM": 0.580,
    "OUTPUT_SAMPLING_NM": 0.02,
    "INTERNAL_WAVELENGTH_STEP_NM": 0.002,
    "OUTPUT_SAMPLE_POINTS": 18_001,
    "REFLECTOR_ANGLE_NOMINAL_DEG": 0.0,
    "TIME_SERIES_ENABLED": False,
    "FRAMES_PER_REALIZATION": 1,
    "SOURCE_REFERENCE_CENTER_NM": 515.0,
    "SOURCE_ENVELOPE_SIGMA_NM": 55.0,
    "SOURCE_ENVELOPE_FLOOR_REL": 0.15,
    "EQ99X_WAVELENGTH_NM": [],
    "EQ99X_SOURCE_SHAPE": [],
    "EQ99X_SOURCE_CSV_SHA256": "",
    "EQ99X_SOURCE_COLUMN": EQ99X_COLUMN,
    "SOURCE_POWER_CORRELATION_LENGTH_NM": 12.0,
    "CALIBRATION_RESIDUAL_CORRELATION_LENGTH_NM": 30.0,
    "SPECTROMETER": {
        "MODEL": "generic_array_spectrometer",
        "REPORTED_AXIS_POLICY": "fixed_factory_calibration",
        "AXIS_ERROR_SIGN_CONVENTION": "reported_minus_true",
        "THERMAL_DRIFT_TARGET": "spectrometer_pixel_mapping",
        "ILS_ENABLED": True,
        "ILS_SHAPE": "gaussian",
        "ILS_FWHM_NM": 0.02,
        "ILS_TRUNCATE_SIGMA": 4.0,
        "WAVELENGTH_ACCURACY_SPEC_NM": 0.05,
        "QE_MODEL": "quadratic",
        "QE_CENTER_NM": 515.0,
        "QE_PEAK": 0.70,
        "QE_EDGE": 0.45,
        "OPTICAL_THROUGHPUT": 0.25,
        "SAMPLE_EXPOSURE_S": 0.010,
        "REFERENCE_EXPOSURE_S": 0.010,
        "DARK_EXPOSURE_S": 0.010,
        "SAMPLE_AVERAGES": 1,
        "REFERENCE_AVERAGES": 1,
        "DARK_AVERAGES": 1,
        "REFERENCE_PEAK_ELECTRONS": 56_000.0,
        "FULL_WELL_ELECTRONS": 80_000.0,
        "READ_NOISE_E_RMS": 5.0,
        "DARK_CURRENT_E_PER_S": 0.1,
        "PIXEL_RESPONSE_NONUNIFORMITY_SIGMA_REL": 0.002,
        "ADC_BITS": 16,
        "ADC_BIAS_COUNTS": 100.0,
        "ADC_NONLINEARITY_REL": 0.001,
        "MIN_REFERENCE_NET_COUNTS": 10.0,
        "QUANTIZATION_ALWAYS_ENABLED": True,
        "SAVE_INTERNAL_AUDIT_ARRAYS": False,
    },
    "LAYERS": [
        ("RefReflector", 0.0),
        ("Air", 100.0),
        ("HSQ", 0.030),
        ("PSS", 0.010),
        ("SOC", 0.040),
        ("TiO2", 0.040),
        ("Cu", 0.0),
    ],
    "ANGLE_MEASUREMENT": {
        "MODE": "fixed_independent_measurement",
        "CLEAN_SETPOINT_DEG": 0.0,
        "NONCLEAN_SETPOINT_MIN_DEG": ANGLE_SETPOINT_MIN_DEG,
        "NONCLEAN_SETPOINT_MAX_DEG": ANGLE_SETPOINT_MAX_DEG,
        "STANDARD_UNCERTAINTY_DEG": ANGLE_MEASUREMENT_SIGMA_DEG,
        "ERROR_DISTRIBUTION": "normal",
        "RELATION": "b=a+e",
        "TRUE_FIELD": "true_reflector_angle_deg",
        "INVERSION_INPUT_FIELD": "measured_reflector_angle_deg",
        "CSV_COMPATIBILITY_ALIAS": "measured_angle_deg",
        "MEASUREMENT_ERROR_FIELD": "angle_measurement_error_deg",
        "MEASUREMENT_ERROR_DEFINITION": "measured_minus_true",
    },
}


def eq99x_csv_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "wiki"
        / "02_Literature"
        / "Materials"
        / "EQ-99X_datasheet_digitized_220_580nm_5nm.csv"
    )


def load_eq99x_curve() -> tuple[np.ndarray, np.ndarray, Path]:
    path = eq99x_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"EQ-99X source curve is missing: {path}")
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8-sig")
    names = tuple(table.dtype.names or ())
    if "wavelength_nm" not in names or EQ99X_COLUMN not in names:
        raise ValueError(f"EQ-99X CSV requires wavelength_nm and {EQ99X_COLUMN}: {names}")
    wavelengths = np.asarray(table["wavelength_nm"], dtype=np.float64)
    shape = np.asarray(table[EQ99X_COLUMN], dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size < 2 or shape.shape != wavelengths.shape:
        raise ValueError("EQ-99X source curve must contain matching 1-D arrays")
    if np.any(~np.isfinite(wavelengths)) or np.any(~np.isfinite(shape)) or np.any(shape <= 0.0):
        raise ValueError("EQ-99X source curve must be finite and strictly positive")
    if np.any(np.diff(wavelengths) <= 0.0) or wavelengths[0] > 200.0 or wavelengths[-1] < 800.0:
        raise ValueError("EQ-99X source curve must monotonically cover 200-800 nm")
    shape = shape / float(np.max(shape))
    return wavelengths, shape, path


def source_power_eq99x(wavelengths_nm: np.ndarray, center_nm: float) -> np.ndarray:
    """Interpolate the digitized EQ-99X SPD; center changes are horizontal shifts."""
    target = np.asarray(wavelengths_nm, dtype=np.float64)
    source_axis = np.asarray(V15_CONFIG["EQ99X_WAVELENGTH_NM"], dtype=np.float64)
    source_shape = np.asarray(V15_CONFIG["EQ99X_SOURCE_SHAPE"], dtype=np.float64)
    if source_axis.size < 2 or source_shape.shape != source_axis.shape:
        raise RuntimeError("EQ-99X source curve has not been configured")
    shift_nm = float(center_nm) - float(V15_CONFIG["SOURCE_REFERENCE_CENTER_NM"])
    values = np.interp(
        target - shift_nm,
        source_axis,
        source_shape,
        left=float(source_shape[0]),
        right=float(source_shape[-1]),
    )
    maximum = float(np.max(values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise FloatingPointError("interpolated EQ-99X source curve is invalid")
    return values / maximum

def validate_v15_config() -> None:
    start_nm = float(V15_CONFIG["WAVELENGTH_START_UM"]) * 1000.0
    stop_nm = float(V15_CONFIG["WAVELENGTH_STOP_UM"]) * 1000.0
    sampling_nm = float(V15_CONFIG["OUTPUT_SAMPLING_NM"])
    internal_step_nm = float(V15_CONFIG["INTERNAL_WAVELENGTH_STEP_NM"])

    if not np.isfinite(start_nm) or not np.isfinite(stop_nm):
        raise ValueError("V15 wavelength range must be finite.")
    if not np.isfinite(sampling_nm) or sampling_nm <= 0.0:
        raise ValueError("V15 output sampling must be positive.")
    if not np.isfinite(internal_step_nm) or internal_step_nm <= 0.0:
        raise ValueError("V15 internal step must be positive.")

    points = int(round((stop_nm - start_nm) / sampling_nm)) + 1
    if points < 2:
        raise ValueError("Wavelength range must contain at least two output samples.")

    actual_sampling_nm = (stop_nm - start_nm) / (points - 1)
    if not math.isclose(actual_sampling_nm, sampling_nm, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"Requested span is not an integer multiple of output sampling: "
            f"requested={sampling_nm}, actual={actual_sampling_nm} nm."
        )

    V15_CONFIG["OUTPUT_SAMPLE_POINTS"] = points
    if V15_CONFIG["SOURCE_MODEL"] != "eq99x_digitized_peak_normalized":
        raise ValueError("V15 requires the digitized EQ-99X source model")
    source_axis = np.asarray(V15_CONFIG["EQ99X_WAVELENGTH_NM"], dtype=np.float64)
    source_shape = np.asarray(V15_CONFIG["EQ99X_SOURCE_SHAPE"], dtype=np.float64)
    if source_axis.size < 2 or source_shape.shape != source_axis.shape:
        raise ValueError("V15 EQ-99X arrays are not configured")


# ---------------------------------------------------------------------------
# V15 angle contract
# ---------------------------------------------------------------------------
_KERNEL_REALIZE_NOISE = _kernel.realize_noise


def realize_noise_v15(
    case_name: str,
    reported_wavelengths_nm: np.ndarray,
    internal_absolute_wavelengths_nm: np.ndarray,
    rng: np.random.Generator,
):
    """Reuse V10 noise terms, then add the V15 angle contract."""
    metadata, components = _KERNEL_REALIZE_NOISE(
        case_name,
        reported_wavelengths_nm,
        internal_absolute_wavelengths_nm,
        rng,
    )

    legacy_angle = float(metadata.get("reflector_angle_deg", 0.0))
    nominal_angle = float(V15_CONFIG["REFLECTOR_ANGLE_NOMINAL_DEG"])
    angle_config = V15_CONFIG["ANGLE_MEASUREMENT"]

    if case_name == "clean":
        setpoint = 0.0
        measurement_error = 0.0
    else:
        setpoint = float(rng.uniform(
            angle_config["NONCLEAN_SETPOINT_MIN_DEG"],
            angle_config["NONCLEAN_SETPOINT_MAX_DEG"],
        ))
        measurement_error = float(rng.normal(
            0.0, angle_config["STANDARD_UNCERTAINTY_DEG"]
        ))

    measured = setpoint + measurement_error
    if not np.isfinite(measured):
        raise FloatingPointError("Independent angle measurement is not finite.")

    metadata.update(
        {
            "legacy_kernel_noise_angle_deg_audit_only": legacy_angle,
            "reflector_angle_deg": setpoint,
            "reflector_angle_setpoint_deg": setpoint,
            "reflector_angle_deviation_from_nominal_deg": setpoint - nominal_angle,
            # Deprecated compatibility field.  It now has one explicit meaning:
            # physical true-angle deviation from nominal, never measurement error.
            "reflector_angle_error_deg": setpoint - nominal_angle,
            "measured_reflector_angle_deg": measured,
            "measured_angle_deg": measured,
            "angle_measurement_error_deg": measurement_error,
            "angle_measurement_error_definition": "measured_minus_true",
            "angle_measurement_sigma_deg": float(
                angle_config["STANDARD_UNCERTAINTY_DEG"]
            ),
            "angle_measurement_mode": angle_config["MODE"],
            "angle_measurement_visible_to_inversion": True,
            "true_angle_visible_to_inversion": False,
        }
    )

    return metadata, components


class V15DatasetGenerator(_kernel.StaticDatasetGenerator):
    """V15 derivative of the V10 dataset generator.

    It keeps the V10 forward model and signal chain, but overwrites the
    angle contract to the V15 independent measurement model.
    """
    def save_one(self, data: dict) -> Path:
        """Save the kernel payload, then atomically add V15 inversion inputs."""
        path = super().save_one(data)
        metadata = data["metadata"]
        with np.load(path, allow_pickle=False) as source:
            payload = {name: np.asarray(source[name]) for name in source.files}
        payload.update({
            "measured_reflector_angle_deg": np.asarray(
                metadata["measured_reflector_angle_deg"]
            ),
            "measured_angle_deg": np.asarray(metadata["measured_angle_deg"]),
            "measured_angle_semantics": np.asarray(
                "compatibility_alias_of_measured_reflector_angle_deg"
            ),
            "angle_measurement_sigma_deg": np.asarray(
                metadata["angle_measurement_sigma_deg"]
            ),
            "angle_measurement_error_deg": np.asarray(
                metadata["angle_measurement_error_deg"]
            ),
            "angle_measurement_error_definition": np.asarray(
                metadata["angle_measurement_error_definition"]
            ),
            "angle_measurement_mode": np.asarray(
                metadata["angle_measurement_mode"]
            ),
            "angle_measurement_visible_to_inversion": np.asarray(True),
            "true_angle_visible_to_inversion": np.asarray(False),
            "reflector_angle_setpoint_deg": np.asarray(
                metadata["reflector_angle_setpoint_deg"]
            ),
            "reflector_angle_deviation_from_nominal_deg": np.asarray(
                metadata["reflector_angle_deviation_from_nominal_deg"]
            ),
            "reflector_angle_error_semantics": np.asarray(
                "deprecated_alias_of_reflector_angle_deviation_from_nominal_deg"
            ),
        })
        temporary = path.with_name(path.name + ".v15tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path


def configure_v15() -> None:
    """Configure the private kernel with the digitized EQ-99X source model."""
    wavelengths, shape, source_path = load_eq99x_curve()
    V15_CONFIG["EQ99X_WAVELENGTH_NM"] = wavelengths.tolist()
    V15_CONFIG["EQ99X_SOURCE_SHAPE"] = shape.tolist()
    V15_CONFIG["EQ99X_SOURCE_CSV_SHA256"] = source_sha256(source_path)
    validate_v15_config()
    _kernel.CONFIG = copy.deepcopy(V15_CONFIG)
    _kernel.source_power_envelope = source_power_eq99x
    _kernel.GENERATOR_VERSION = GENERATOR_VERSION
    _kernel.WAVELENGTH_ACCURACY_SPEC_NM = float(
        V15_CONFIG["SPECTROMETER"]["WAVELENGTH_ACCURACY_SPEC_NM"]
    )
    _kernel.realize_noise = realize_noise_v15


def apply_wavelength_cli_overrides(args: argparse.Namespace) -> None:
    start_nm = float(V15_CONFIG["WAVELENGTH_START_UM"]) * 1000.0
    stop_nm = float(V15_CONFIG["WAVELENGTH_STOP_UM"]) * 1000.0
    sampling_nm = float(V15_CONFIG["OUTPUT_SAMPLING_NM"])

    if args.wavelength_start_nm is not None:
        start_nm = float(args.wavelength_start_nm)
    if args.wavelength_stop_nm is not None:
        stop_nm = float(args.wavelength_stop_nm)
    if args.output_sampling_nm is not None:
        sampling_nm = float(args.output_sampling_nm)

    if not np.isfinite(start_nm) or not np.isfinite(stop_nm) or not np.isfinite(sampling_nm):
        raise ValueError("Wavelength overrides must be finite.")
    if start_nm <= 0.0 or stop_nm <= start_nm or sampling_nm <= 0.0:
        raise ValueError("Require 0 < wavelength start < stop and positive output sampling.")

    points = int(round((stop_nm - start_nm) / sampling_nm)) + 1
    if points < 2:
        raise ValueError("Wavelength range must contain at least two output samples.")

    actual_sampling_nm = (stop_nm - start_nm) / (points - 1)
    if not math.isclose(actual_sampling_nm, sampling_nm, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"Requested span is not an integer multiple of output sampling: "
            f"requested={sampling_nm}, actual={actual_sampling_nm} nm."
        )

    V15_CONFIG["WAVELENGTH_START_UM"] = start_nm / 1000.0
    V15_CONFIG["WAVELENGTH_STOP_UM"] = stop_nm / 1000.0
    V15_CONFIG["OUTPUT_SAMPLING_NM"] = sampling_nm
    V15_CONFIG["OUTPUT_SAMPLE_POINTS"] = points
    validate_v15_config()


def wavelength_budget_audit() -> dict:
    start_nm = float(V15_CONFIG["WAVELENGTH_START_UM"]) * 1000.0
    stop_nm = float(V15_CONFIG["WAVELENGTH_STOP_UM"]) * 1000.0
    sampling_nm = float(V15_CONFIG["OUTPUT_SAMPLING_NM"])
    span_nm = stop_nm - start_nm
    points = int(round(span_nm / sampling_nm)) + 1
    return {
        "wavelength_span_nm": span_nm,
        "output_sample_points": points,
        "output_sampling_nm": sampling_nm,
        "wavelength_start_nm": start_nm,
        "wavelength_stop_nm": stop_nm,
        "angle_contract": V15_CONFIG["ANGLE_MEASUREMENT"],
    }


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_index_row(metadata: dict, npz_path: Path) -> dict:
    """Return the complete scalar audit row; keep truth and input fields separate."""
    return {
        "noise_case": metadata["noise_case"],
        "noise_factor": metadata["noise_factor"],
        "noise_level": metadata["noise_level"],
        "realization_index": metadata["realization_index"],
        "random_seed": metadata["random_seed"],
        "true_reflector_angle_deg": metadata["reflector_angle_deg"],
        "reflector_angle_setpoint_deg": metadata["reflector_angle_setpoint_deg"],
        "reflector_angle_deviation_from_nominal_deg": metadata[
            "reflector_angle_deviation_from_nominal_deg"
        ],
        "measured_reflector_angle_deg": metadata["measured_reflector_angle_deg"],
        "measured_angle_deg": metadata["measured_angle_deg"],
        "angle_measurement_error_deg": metadata["angle_measurement_error_deg"],
        "angle_measurement_error_definition": metadata[
            "angle_measurement_error_definition"
        ],
        "angle_measurement_sigma_deg": metadata["angle_measurement_sigma_deg"],
        "angle_measurement_mode": metadata["angle_measurement_mode"],
        "angle_measurement_visible_to_inversion": metadata[
            "angle_measurement_visible_to_inversion"
        ],
        "true_angle_visible_to_inversion": metadata["true_angle_visible_to_inversion"],
        "legacy_kernel_noise_angle_deg_audit_only": metadata[
            "legacy_kernel_noise_angle_deg_audit_only"
        ],
        "reported_axis_is_fixed": metadata["reported_axis_is_fixed"],
        "axis_offset_nm": metadata["axis_offset_nm"],
        "axis_scale_ppm": metadata["axis_scale_ppm"],
        "thermal_drift_nm": metadata["thermal_drift_nm"],
        "calibration_residual_peak_nm": metadata["calibration_residual_peak_nm"],
        "spectrometer_axis_error_max_abs_nm": metadata[
            "spectrometer_axis_error_max_abs_nm"
        ],
        "spectrometer_axis_error_rms_nm": metadata["spectrometer_axis_error_rms_nm"],
        "physical_axis_shift_max_abs_nm": metadata["physical_axis_shift_max_abs_nm"],
        "axis_estimation_error_max_abs_nm": metadata[
            "axis_estimation_error_max_abs_nm"
        ],
        "axis_estimation_error_rms_nm": metadata["axis_estimation_error_rms_nm"],
        "wavelength_accuracy_within_spec": metadata["wavelength_accuracy_within_spec"],
        "source_center_drift_nm": metadata["source_center_drift_nm"],
        "source_power_curve_peak_rel": metadata["source_power_curve_peak_rel"],
        "detector_noise_multiplier": metadata["detector_noise_multiplier"],
        "target_reference_peak_electrons": metadata["target_reference_peak_electrons"],
        "sample_saturation_fraction": metadata["sample_saturation_fraction"],
        "reference_saturation_fraction": metadata["reference_saturation_fraction"],
        "invalid_reference_pixel_fraction": metadata[
            "invalid_reference_pixel_fraction"
        ],
        "output_sampling_nm": metadata["output_sampling_nm"],
        "internal_wavelength_step_nm": metadata["internal_wavelength_step_nm"],
        "optical_solver_runtime_s": metadata["optical_solver_runtime_s"],
        "generation_runtime_s": metadata["generation_runtime_s"],
        "npz_path": str(npz_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V15 dataset generator with a fixed independent angle measurement."
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--clean-repeats", type=int, default=1)
    parser.add_argument("--backend", choices=["tmm", "api", "batch"], default="api")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wavelength-start-nm", type=float, default=None)
    parser.add_argument("--wavelength-stop-nm", type=float, default=None)
    parser.add_argument("--output-sampling-nm", type=float, default=None)
    parser.add_argument("--describe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_v15()
    apply_wavelength_cli_overrides(args)
    configure_v15()

    if args.describe:
        print(json.dumps({
            "version": GENERATOR_VERSION,
            "config": V15_CONFIG,
            "wavelength_budget_audit": wavelength_budget_audit(),
            "angle_contract": V15_CONFIG["ANGLE_MEASUREMENT"],
        }, indent=2, ensure_ascii=False))
        return

    if args.repeats < 1 or args.clean_repeats < 1:
        raise ValueError("Repeat counts must be positive.")

    cases = _kernel.select_cases(args.cases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(__file__).resolve().parents[2] / "04_results_and_datasets"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else output_root / f"static_stackrt_v15_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    generator = V15DatasetGenerator(args.backend, output_dir)
    rows: list[dict] = []
    failures: list[dict] = []
    representatives: dict[str, dict] = {}

    with _kernel.OpticalSolver(args.backend, output_dir / "_stackrt_batch_bridge") as solver:
        for case_index, case_name in enumerate(cases):
            repeats = args.clean_repeats if case_name == "clean" else args.repeats
            for realization_index in range(repeats):
                seed = int(args.seed + case_index * 1_000_000 + realization_index)
                try:
                    data, metadata = generator.generate_one(
                        solver, case_name, realization_index, seed
                    )
                    npz_path = generator.save_one(data)
                    rows.append(dataset_index_row(metadata, npz_path))
                    representatives.setdefault(case_name, data)
                except Exception as exc:
                    failures.append({
                        "case": case_name,
                        "realization": realization_index,
                        "seed": seed,
                        "error": str(exc),
                    })

    index_path = output_dir / "dataset_index.csv"
    if rows:
        with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plot_path = (
        _kernel.save_representative_plot(output_dir, representatives)
        if representatives else None
    )
    kernel_source = Path(__file__).with_name("main_v10.py")
    generator_source = Path(__file__).resolve()

    manifest = {
        "version": GENERATOR_VERSION,
        "created": timestamp,
        "backend": args.backend,
        "cases": cases,
        "repeats_non_clean": args.repeats,
        "repeats_clean": args.clean_repeats,
        "random_seed": args.seed,
        "config": V15_CONFIG,
        "noise_levels": _kernel.NOISE_LEVELS,
        "noise_factors": _kernel.NOISE_FACTORS,
        "level_amplitude_fraction_range": _kernel.LEVEL_AMPLITUDE_FRACTION_RANGE,
        "wavelength_axis_scenario": V15_CONFIG["WAVELENGTH_AXIS_SCENARIO"],
        "reported_axis_is_fixed": True,
        "axis_error_sign_convention": V15_CONFIG["SPECTROMETER"][
            "AXIS_ERROR_SIGN_CONVENTION"
        ],
        "wavelength_accuracy_spec_nm": V15_CONFIG["SPECTROMETER"][
            "WAVELENGTH_ACCURACY_SPEC_NM"
        ],
        "wavelength_budget_audit": wavelength_budget_audit(),
        "angle_contract": V15_CONFIG["ANGLE_MEASUREMENT"],
        "configuration_isolation": {
            "depends_on_main_v10_config": False,
            "mutates_public_main_v10_module": False,
            "physical_kernel_source": str(kernel_source.resolve()),
            "physical_kernel_sha256": source_sha256(kernel_source),
            "generator_source": str(generator_source),
            "generator_sha256": source_sha256(generator_source),
        },
        "dataset_count": len(rows),
        "dataset_index": str(index_path) if rows else None,
        "representative_plot": str(plot_path) if plot_path else None,
        "failures": failures,
    }
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT_DIR={output_dir}")
    print(f"MANIFEST={manifest_path}")
    if failures:
        raise RuntimeError(f"Simulation failures: {failures}")


if __name__ == "__main__":
    main()
