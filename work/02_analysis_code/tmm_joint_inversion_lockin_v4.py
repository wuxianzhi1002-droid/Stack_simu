from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tmm_joint_inversion_lockin_v3 as core


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"
DEFAULT_INPUT_DIR = OUTPUT_ROOT / "dynamic_stackrt_lockin_v5"
VERSION = "tmm_joint_inversion_lockin_v4"
THICKNESS_PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2"]
PARAMS = THICKNESS_PARAMS + ["Angle"]
ANGLE_LIMIT_DEG = 0.1

core.PARAMS = PARAMS
core.BOUNDS = {
    "Air": (998.0, 1002.0),
    "HSQ": (20.0, 40.0),
    "PSS": (1.0, 20.0),
    "SOC": (30.0, 50.0),
    "TiO2": (30.0, 50.0),
    "Angle": (0.0, ANGLE_LIMIT_DEG),
}
core.PRIOR_CENTER = {
    name: 0.5 * (bounds[0] + bounds[1]) for name, bounds in core.BOUNDS.items()
}
core.PRIOR_SIGMA = {
    "Air": 2.0,
    "HSQ": 10.0,
    "PSS": 5.0,
    "SOC": 10.0,
    "TiO2": 10.0,
    "Angle": 0.05,
}


def parameter_unit(name: str) -> str:
    return "um" if name == "Air" else "deg" if name == "Angle" else "nm"


def air_phase_length_um(air_um: float, incident_angle_deg: float) -> float:
    theta_air = np.arcsin(5.8284 * np.sin(np.deg2rad(incident_angle_deg)))
    return float(air_um * np.cos(theta_air))


def propagation_cosines(n_matrix: np.ndarray, theta_deg: float) -> np.ndarray:
    theta_rad = np.deg2rad(float(theta_deg))
    tangential_index = n_matrix[0] * np.sin(theta_rad)
    cos_values = np.sqrt(1.0 - (tangential_index[None, :] / n_matrix) ** 2)
    cos_values[np.real(cos_values) < 0.0] *= -1.0
    return cos_values


def oblique_tmm_reflectance(
    wavelengths_um: np.ndarray,
    thicknesses_um: dict[str, float],
    theta_deg: float,
) -> np.ndarray:
    if not 0.0 <= theta_deg <= ANGLE_LIMIT_DEG:
        raise ValueError(f"Angle {theta_deg} is outside [0, {ANGLE_LIMIT_DEG}] deg.")

    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    frequency_hz = core.FREQUENCY_AXIS_C_M_S / (wavelengths_um * 1.0e-6)
    phase_wavelength_m = core.C0_M_S / frequency_hz
    n_matrix = np.vstack([core.material_n(name, wavelengths_um) for name in core.LAYER_NAMES])
    cos_values = propagation_cosines(n_matrix, theta_deg)
    q_values = n_matrix / cos_values
    thicknesses_m = np.asarray([
        thicknesses_um.get(name, 0.0) * 1.0e-6 for name in core.LAYER_NAMES
    ])
    k0 = 2.0 * np.pi / phase_wavelength_m

    m11 = np.ones(len(wavelengths_um), dtype=complex)
    m12 = np.zeros(len(wavelengths_um), dtype=complex)
    m21 = np.zeros(len(wavelengths_um), dtype=complex)
    m22 = np.ones(len(wavelengths_um), dtype=complex)
    for layer_idx in range(1, len(core.LAYER_NAMES) - 1):
        thickness = float(thicknesses_m[layer_idx])
        if thickness <= 0.0:
            continue
        delta = k0 * n_matrix[layer_idx] * cos_values[layer_idx] * thickness
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        q_layer = q_values[layer_idx]
        a11 = c_delta
        a12 = -1j * s_delta / q_layer
        a21 = -1j * q_layer * s_delta
        a22 = c_delta
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21,
            m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21,
            m21 * a12 + m22 * a22,
        )

    q0 = q_values[0]
    qs = q_values[-1]
    numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
    denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
    return np.abs(numerator / denominator) ** 2


def fit_vector_to_um(values: np.ndarray) -> dict[str, float]:
    return {
        "Air": float(values[0]),
        "HSQ": float(values[1]) / 1000.0,
        "PSS": float(values[2]) / 1000.0,
        "SOC": float(values[3]) / 1000.0,
        "TiO2": float(values[4]) / 1000.0,
    }


def finite_modulation_observables(
    wavelengths_um: np.ndarray,
    values: np.ndarray,
    amplitude_um: float,
    phase_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    phases = 2.0 * np.pi * np.arange(phase_samples, dtype=float) / phase_samples
    sin_ref = np.sin(phases)
    spectra = np.empty((phase_samples, len(wavelengths_um)), dtype=float)
    for idx, sin_value in enumerate(sin_ref):
        modulated = np.asarray(values, dtype=float).copy()
        modulated[0] += amplitude_um * sin_value
        spectra[idx] = oblique_tmm_reflectance(
            wavelengths_um,
            fit_vector_to_um(modulated),
            float(modulated[5]),
        )
    return np.mean(spectra, axis=0), 2.0 * (spectra.T @ sin_ref) / phase_samples / amplitude_um


core.fit_vector_to_um = fit_vector_to_um
core.finite_modulation_observables = finite_modulation_observables


def scalar(npz, key: str, default):
    return np.asarray(npz[key]).item() if key in npz else default


def read_metadata_and_truth(npz_path: Path) -> tuple[dict, dict[str, float]]:
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = {
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "noise_level": str(scalar(data, "noise_level", "unknown")),
            "noise_factor": str(scalar(data, "noise_factor", "unknown")),
            "generator_version": str(scalar(data, "generator_version", "unknown")),
            "random_seed": int(scalar(data, "random_seed", -1)),
            "nominal_amplitude_nm": float(scalar(data, "nominal_amplitude_nm", 5.0)),
            "actual_amplitude_nm": float(scalar(data, "actual_amplitude_nm", 5.0)),
            "actual_angle_deg": float(scalar(data, "actual_angle_deg", 0.0)),
            "wavelength_offset_nm": float(scalar(data, "wavelength_offset_nm", 0.0)),
            "reflectance_clip_fraction": float(scalar(data, "reflectance_clip_fraction", 0.0)),
        }
        truth = {
            "Air": 1000.0,
            "HSQ": 30.0,
            "PSS": 10.0,
            "SOC": 40.0,
            "TiO2": 40.0,
            "Angle": metadata["actual_angle_deg"],
        }
        if "layer_names" in data and "layer_thickness_um" in data:
            layers = dict(zip(
                [str(value) for value in np.asarray(data["layer_names"])],
                np.asarray(data["layer_thickness_um"], dtype=float),
            ))
            truth["Air"] = float(layers.get("Air", truth["Air"]))
            for name in THICKNESS_PARAMS[1:]:
                if name in layers:
                    truth[name] = float(layers[name]) * 1000.0
    return metadata, truth


def save_tables(
    output_dir: Path,
    metadata: dict,
    truth: dict[str, float],
    fits: dict[str, dict],
    diagnostics: dict,
) -> list[dict]:
    rows = []
    for mode, fit in fits.items():
        row = {
            **{key: metadata[key] for key in ("noise_case", "noise_factor", "noise_level")},
            "mode": mode,
            "success": fit["success"],
            "cost": fit["cost"],
            "rmse_normalized": fit["rmse_normalized"],
            "rmse_I": fit["rmse_I"],
            "rmse_dIdL": fit["rmse_dIdL"],
            "condition_number": diagnostics[mode]["condition_number"],
            "local_success_fraction": float(np.mean([x["success"] for x in fit["attempts"]])),
        }
        lower, upper = core.bounds_arrays()
        span = upper - lower
        distances = [
            float(np.linalg.norm((attempt["x"] - fit["x"]) / span))
            for attempt in fit["attempts"]
        ]
        row["multistart_distance_median"] = float(np.median(distances))
        row["multistart_distance_max"] = float(np.max(distances))
        tolerance = 1.0e-4 * span
        boundary_hits = [
            PARAMS[idx] for idx, value in enumerate(fit["x"])
            if value - lower[idx] <= tolerance[idx] or upper[idx] - value <= tolerance[idx]
        ]
        row["boundary_hit_count"] = len(boundary_hits)
        row["boundary_hits"] = ";".join(boundary_hits)
        for name, value in zip(PARAMS, fit["x"]):
            unit = parameter_unit(name)
            row[f"fit_{name}_{unit}"] = value
            row[f"error_{name}_{unit}"] = value - truth[name]
        row["film_mae_nm"] = float(np.mean([
            abs(row[f"error_{name}_nm"]) for name in THICKNESS_PARAMS[1:]
        ]))
        row["air_abs_error_um"] = abs(row["error_Air_um"])
        row["angle_abs_error_deg"] = abs(row["error_Angle_deg"])
        row["fit_air_phase_length_um"] = air_phase_length_um(
            row["fit_Air_um"], row["fit_Angle_deg"]
        )
        row["truth_air_phase_length_um"] = air_phase_length_um(
            truth["Air"], truth["Angle"]
        )
        row["air_phase_abs_error_um"] = abs(
            row["fit_air_phase_length_um"] - row["truth_air_phase_length_um"]
        )
        attempt_air = np.asarray([attempt["x"][0] for attempt in fit["attempts"]])
        attempt_angle = np.asarray([attempt["x"][5] for attempt in fit["attempts"]])
        row["multistart_air_angle_correlation"] = (
            float(np.corrcoef(attempt_air, attempt_angle)[0, 1])
            if np.std(attempt_air) > 0.0 and np.std(attempt_angle) > 0.0 else float("nan")
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_dir / "fit_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    truth_vector = np.asarray([truth[name] for name in PARAMS])
    lower, upper = core.bounds_arrays()
    span = upper - lower
    attempts = []
    for mode, fit in fits.items():
        for rank, attempt in enumerate(fit["attempts"], start=1):
            row = {
                "mode": mode,
                "rank": rank,
                "success": attempt["success"],
                "cost": attempt["cost"],
                "rmse_normalized": attempt["rmse_normalized"],
                "nfev": attempt["nfev"],
                "global_population_rank": attempt["global_population_rank"],
                "global_energy": attempt["global_energy"],
                "distance_to_best_scaled": float(np.linalg.norm((attempt["x"] - fit["x"]) / span)),
                "benchmark_distance_to_truth_scaled": float(
                    np.linalg.norm((attempt["x"] - truth_vector) / span)
                ),
            }
            for name, value, start in zip(PARAMS, attempt["x"], attempt["x0"]):
                unit = parameter_unit(name)
                row[f"fit_{name}_{unit}"] = value
                row[f"benchmark_error_{name}_{unit}"] = value - truth[name]
                row[f"x0_{name}_{unit}"] = start
            attempts.append(row)
    pd.DataFrame(attempts).to_csv(
        output_dir / "multistart_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    return rows


def run_case(
    npz_path: Path,
    root: Path,
    base_config: core.FitConfig,
    modes: list[str],
    case_index: int,
) -> list[dict]:
    metadata, truth = read_metadata_and_truth(npz_path)
    amplitude_nm = (
        metadata["nominal_amplitude_nm"]
        if base_config.amplitude_nm <= 0.0
        else base_config.amplitude_nm
    )
    config = replace(base_config, input_npz=str(npz_path), amplitude_nm=amplitude_nm)
    measurement = core.load_measurement(npz_path, config)

    # Keep benchmark truth unavailable until optimization and ranking finish.
    core.EVALUATION_TRUTH = {}
    fits = {
        mode: core.fit_mode(
            measurement,
            config,
            mode,
            config.random_seed + case_index * 10000 + mode_index * 1000,
        )
        for mode_index, mode in enumerate(modes)
    }
    diagnostics = core.diagnostics(measurement, config, fits)
    core.EVALUATION_TRUTH = truth

    case_dir = root / metadata["noise_case"]
    case_dir.mkdir(parents=True, exist_ok=False)
    rows = save_tables(case_dir, metadata, truth, fits, diagnostics)
    plot_paths = core.plot_fits(case_dir, measurement, fits)
    plot_paths.update(core.save_diagnostics_plots(case_dir, diagnostics))
    summary = {
        "version": VERSION,
        "input_npz_resolved": str(npz_path),
        "metadata": metadata,
        "config": asdict(config),
        "params": PARAMS,
        "bounds": core.BOUNDS,
        "evaluation_truth": truth,
        "truth_usage_policy": "loaded for reporting but hidden from optimization, initialization, residuals, priors, and ranking",
        "postfit_initialization_audit": core.postfit_initialization_audit(fits),
        "global_search": {mode: fit["global_search"] for mode, fit in fits.items()},
        "diagnostics": diagnostics,
        "plot_paths": plot_paths,
        "tmm_convention": {
            "frequency_axis": "f = 3e8 / lambda_nominal",
            "phase_wavelength": "lambda_phase = 299792458 / f",
            "polarization": "p",
            "oblique_admittance": "q_p = n / cos(theta_layer)",
            "snell_invariant": "n0*sin(theta0) = nj*sin(thetaj)",
            "angle_parameter": "nonnegative StackRT incident-medium angle, bounded 0 to 0.1 deg",
            "observation_model": "finite modulation mean I and lockin 1f X / nominal A",
        },
    }
    (case_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"[{metadata['noise_case']}/{row['mode']}] "
            f"film_MAE={row['film_mae_nm']:.5g} nm, "
            f"Air_error={row['air_abs_error_um']:.5g} um, "
            f"angle_error={row['angle_abs_error_deg']:.5g} deg, "
            f"boundaries={row['boundary_hits'] or 'none'}"
        )
    return rows


def save_batch_summary(root: Path, rows: list[dict]) -> None:
    table = pd.DataFrame(rows)
    table.to_csv(
        root / "batch_fit_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    selected = table[table["mode"] == "joint"].copy()
    if selected.empty:
        selected = table.copy()
    factors = ["angle", "wavelength", "material", "amplitude", "detector", "combined"]
    levels = ["low", "medium", "high"]
    metrics = [
        ("film_mae_nm", "Film thickness MAE (nm)"),
        ("air_phase_abs_error_um", "Air phase-length absolute error (um)"),
        ("angle_abs_error_deg", "Angle absolute error (deg)"),
        ("rmse_normalized", "Normalized residual RMSE"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (column, title) in zip(axs.flat, metrics):
        for factor in factors:
            subset = selected[selected["noise_factor"] == factor].set_index("noise_level")
            values = [
                float(subset.loc[level, column]) if level in subset.index else np.nan
                for level in levels
            ]
            ax.plot(levels, values, marker="o", label=factor)
        clean = selected[selected["noise_level"] == "clean"]
        if not clean.empty:
            ax.axhline(float(clean.iloc[0][column]), color="black", ls="--", label="clean")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=7)
    fig.suptitle("Angle-aware joint inversion accuracy by noise factor")
    fig.savefig(root / "noise_accuracy_overview.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (column, title) in zip(
        axs,
        (("film_mae_nm", "Film MAE (nm)"), ("rmse_normalized", "Normalized RMSE")),
    ):
        matrix = np.full((len(factors), len(levels)), np.nan)
        for i, factor in enumerate(factors):
            for j, level in enumerate(levels):
                match = selected[
                    (selected["noise_factor"] == factor)
                    & (selected["noise_level"] == level)
                ]
                if not match.empty:
                    matrix[i, j] = float(match.iloc[0][column])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(levels)), labels=levels)
        ax.set_yticks(range(len(factors)), labels=factors)
        ax.set_title(title)
        for i in range(len(factors)):
            for j in range(len(levels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3g}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(root / "noise_factor_heatmaps.png", dpi=200)
    plt.close(fig)

    summary = {
        "version": VERSION,
        "case_count": int(selected["noise_case"].nunique()),
        "modes": sorted(table["mode"].unique().tolist()),
        "best_film_mae_case": str(selected.loc[selected["film_mae_nm"].idxmin(), "noise_case"]),
        "worst_film_mae_case": str(selected.loc[selected["film_mae_nm"].idxmax(), "noise_case"]),
        "boundary_hit_cases": selected.loc[
            selected["boundary_hit_count"] > 0, "noise_case"
        ].tolist(),
        "median_condition_number": float(selected["condition_number"].median()),
        "median_multistart_distance": float(selected["multistart_distance_median"].median()),
    }
    (root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Angle-aware v4 TMM inversion for v5 StackRT ablation datasets."
    )
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--pattern", default="dynamic_spectra_*.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--amplitude-nm", type=float, default=-1.0)
    parser.add_argument("--wavelength-min-nm", type=float, default=220.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--multistarts", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=250)
    parser.add_argument("--global-popsize", type=int, default=12)
    parser.add_argument("--global-maxiter", type=int, default=50)
    parser.add_argument("--global-stride", type=int, default=50)
    parser.add_argument("--global-phase-samples", type=int, default=8)
    parser.add_argument("--local-phase-samples", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260717)
    parser.add_argument("--use-prior", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="soft_l1",
    )
    parser.add_argument("--modes", nargs="+", choices=["I", "D", "joint"], default=["joint"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inputs:
        inputs = [Path(value).resolve() for value in args.inputs]
    else:
        inputs = sorted(Path(args.input_dir).resolve().glob(args.pattern))
    if args.max_files is not None:
        inputs = inputs[:args.max_files]
    if not inputs:
        raise FileNotFoundError("No input NPZ files matched.")

    root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else OUTPUT_ROOT / f"{VERSION}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    root.mkdir(parents=True, exist_ok=False)
    config = core.FitConfig(
        input_npz="",
        amplitude_nm=args.amplitude_nm,
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        stride=args.stride,
        multistarts=args.multistarts,
        max_nfev=args.max_nfev,
        global_popsize=args.global_popsize,
        global_maxiter=args.global_maxiter,
        global_stride=args.global_stride,
        global_phase_samples=args.global_phase_samples,
        local_phase_samples=args.local_phase_samples,
        random_seed=args.random_seed,
        use_prior=args.use_prior,
        loss=args.loss,
    )
    rows = []
    failures = []
    for index, npz_path in enumerate(inputs):
        try:
            rows.extend(run_case(npz_path, root, config, list(args.modes), index))
        except Exception as exc:
            failures.append({"input": str(npz_path), "error": str(exc)})
            print(f"ERROR {npz_path}: {exc}")
    if rows:
        save_batch_summary(root, rows)
    (root / "inversion_manifest.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "inputs": [str(path) for path in inputs],
                "config": asdict(config),
                "modes": args.modes,
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={root}")
    if failures:
        raise RuntimeError(f"Inversion failures: {failures}")


if __name__ == "__main__":
    main()
