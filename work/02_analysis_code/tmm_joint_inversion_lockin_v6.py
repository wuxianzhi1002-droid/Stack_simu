from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"
DEFAULT_INPUT_DIR = OUTPUT_ROOT / "static_stackrt_v6"
VERSION = "tmm_joint_inversion_lockin_v6"
THICKNESS_PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2"]
PARAMS = THICKNESS_PARAMS + ["Angle"]
ANGLE_LIMIT_DEG = 0.1
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
INCIDENT_MEDIUM_N = 5.8284
C0_M_S = 299_792_458.0
FREQUENCY_AXIS_C_M_S = 3.0e8

# Simulation truth is loaded only after optimization and is used exclusively
# for reporting and initialization audits.
EVALUATION_TRUTH: dict[str, float] = {}

BOUNDS = {
    "Air": (998.0, 1002.0),
    "HSQ": (20.0, 40.0),
    "PSS": (1.0, 20.0),
    "SOC": (30.0, 50.0),
    "TiO2": (30.0, 50.0),
    "Angle": (0.0, ANGLE_LIMIT_DEG),
}
PRIOR_CENTER = {
    name: 0.5 * (bounds[0] + bounds[1]) for name, bounds in BOUNDS.items()
}
PRIOR_SIGMA = {
    "Air": 2.0,
    "HSQ": 10.0,
    "PSS": 5.0,
    "SOC": 10.0,
    "TiO2": 10.0,
    "Angle": 0.05,
}


@dataclass
class FitConfig:
    input_npz: str
    wavelength_min_nm: float = 220.0
    wavelength_max_nm: float = 580.0
    stride: int = 1
    multistarts: int = 16
    max_nfev: int = 250
    global_popsize: int = 12
    global_maxiter: int = 80
    global_stride: int = 50
    random_seed: int = 20260721
    use_prior: bool = False
    loss: str = "soft_l1"


def material_n(name: str, wavelengths_um: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=np.float64)
    if name == "RefReflector":
        return np.full_like(w, INCIDENT_MEDIUM_N, dtype=np.complex128)
    if name == "Air":
        return np.full_like(w, 1.0, dtype=np.complex128)
    if name == "HSQ":
        return np.full_like(w, 1.41, dtype=np.complex128)
    if name == "PSS":
        return np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    if name == "SOC":
        return (1.55 + 0.005 / (w**2)).astype(np.complex128)
    if name == "TiO2":
        return (2.4 + 0.02 / (w**2)).astype(np.complex128)
    if name == "Cu":
        return np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    raise ValueError(f"Unknown material: {name}")

def parameter_unit(name: str) -> str:
    return "um" if name == "Air" else "deg" if name == "Angle" else "nm"


def air_phase_length_um(air_um: float, incident_angle_deg: float) -> float:
    theta_air = np.arcsin(INCIDENT_MEDIUM_N * np.sin(np.deg2rad(incident_angle_deg)))
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
    frequency_hz = FREQUENCY_AXIS_C_M_S / (wavelengths_um * 1.0e-6)
    phase_wavelength_m = C0_M_S / frequency_hz
    n_matrix = np.vstack([material_n(name, wavelengths_um) for name in LAYER_NAMES])
    cos_values = propagation_cosines(n_matrix, theta_deg)
    q_values = n_matrix / cos_values
    thicknesses_m = np.asarray([
        thicknesses_um.get(name, 0.0) * 1.0e-6 for name in LAYER_NAMES
    ])
    k0 = 2.0 * np.pi / phase_wavelength_m

    m11 = np.ones(len(wavelengths_um), dtype=complex)
    m12 = np.zeros(len(wavelengths_um), dtype=complex)
    m21 = np.zeros(len(wavelengths_um), dtype=complex)
    m22 = np.ones(len(wavelengths_um), dtype=complex)
    for layer_idx in range(1, len(LAYER_NAMES) - 1):
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


def static_spectrum_model(
    wavelengths_um: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return oblique_tmm_reflectance(
        wavelengths_um,
        fit_vector_to_um(values),
        float(values[5]),
    )


def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([BOUNDS[name][0] for name in PARAMS], dtype=float)
    upper = np.array([BOUNDS[name][1] for name in PARAMS], dtype=float)
    return lower, upper


def robust_sigma(x: np.ndarray, floor: float) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < floor:
        sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma < floor:
        sigma = floor
    return float(sigma)


def load_measurement(npz_path: Path, config: FitConfig):
    with np.load(npz_path, allow_pickle=False) as data:
        wavelengths_um = np.asarray(data["wavelengths"], dtype=float)
        modulation_enabled = bool(scalar(data, "modulation_enabled", False))
        if modulation_enabled:
            raise ValueError(
                "Static v6 inversion does not accept modulated data. "
                "Use an NPZ generated by main_dynamic_v6.py."
            )
        if "spectrum_measured" in data:
            spectrum = np.asarray(data["spectrum_measured"], dtype=float)
        else:
            spectra = np.asarray(data["spectra"], dtype=float)
            if spectra.ndim == 1:
                spectrum = spectra
            elif spectra.ndim == 2:
                spectrum = np.mean(spectra, axis=0)
            else:
                raise ValueError("spectra must have shape (N_lambda,) or (N_frames, N_lambda).")

    if spectrum.ndim != 1 or len(spectrum) != len(wavelengths_um):
        raise ValueError("Static spectrum and wavelength axis must be aligned 1D arrays.")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("Static spectrum contains non-finite values.")

    wavelengths_nm = wavelengths_um * 1000.0
    mask = (wavelengths_nm >= config.wavelength_min_nm) & (wavelengths_nm <= config.wavelength_max_nm)
    masked_idx = np.where(mask)[0]
    idx = masked_idx[:: max(1, int(config.stride))]
    global_idx = masked_idx[:: max(1, int(config.global_stride))]
    if len(idx) < len(PARAMS) * 4:
        raise ValueError("Too few wavelength samples after mask/stride. Reduce --stride or widen wavelength range.")
    if len(global_idx) < len(PARAMS) * 4:
        raise ValueError("Too few global-search samples. Reduce --global-stride or widen wavelength range.")

    return {
        "wavelengths_um_full": wavelengths_um,
        "I_meas_full": spectrum,
        "idx": idx,
        "wavelengths_um": wavelengths_um[idx],
        "I_meas": spectrum[idx],
        "global_idx": global_idx,
        "global_wavelengths_um": wavelengths_um[global_idx],
        "global_I_meas": spectrum[global_idx],
    }


def make_residual(
    wavelengths_um: np.ndarray,
    I_meas: np.ndarray,
    config: FitConfig,
):
    sigma_i = robust_sigma(I_meas, floor=1.0e-4)
    prior_sigma = np.array([PRIOR_SIGMA[name] for name in PARAMS], dtype=float)
    prior_center = np.array([PRIOR_CENTER[name] for name in PARAMS], dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        model_i = static_spectrum_model(wavelengths_um, values)
        blocks = [(model_i - I_meas) / sigma_i]
        if config.use_prior:
            blocks.append((values - prior_center) / prior_sigma)
        return np.concatenate(blocks)

    return residual, {"sigma_I": sigma_i}


def robust_objective(residual_values: np.ndarray, loss: str) -> float:
    z = np.asarray(residual_values, dtype=float) ** 2
    if loss == "linear":
        rho = z
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(z)
    elif loss == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported loss: {loss}")
    return float(np.mean(rho))


def global_candidate_pool(
    measurement: dict,
    config: FitConfig,
    seed: int,
) -> tuple[list[dict], dict]:
    population_size = config.global_popsize * len(PARAMS)
    if population_size < config.multistarts:
        raise ValueError(
            "global_popsize * parameter_count must be >= multistarts so every "
            "local start comes from the optimized global population."
        )

    global_residual, global_scales = make_residual(
        measurement["global_wavelengths_um"],
        measurement["global_I_meas"],
        config,
    )
    lower, upper = bounds_arrays()

    result = differential_evolution(
        lambda values: robust_objective(global_residual(values), config.loss),
        bounds=list(zip(lower, upper)),
        strategy="best1bin",
        maxiter=config.global_maxiter,
        popsize=config.global_popsize,
        tol=1.0e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        init="latinhypercube",
        updating="immediate",
        workers=1,
    )

    order = np.argsort(result.population_energies)
    candidates = []
    for population_rank, index in enumerate(order[: config.multistarts], start=1):
        values = np.asarray(result.population[index], dtype=float)
        candidates.append(
            {
                "x0": values,
                "global_population_rank": population_rank,
                "global_energy": float(result.population_energies[index]),
            }
        )

    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "best_energy": float(result.fun),
        "best_vector": np.asarray(result.x, dtype=float).tolist(),
        "population_size": int(len(result.population)),
        "scales": global_scales,
        "initialization": "differential_evolution with Latin-hypercube population; no truth or nominal seed",
    }
    return candidates, summary


def postfit_initialization_audit(fits: dict[str, dict]) -> dict:
    """Benchmark-only audit run after all optimizations have completed."""
    truth = np.array([EVALUATION_TRUTH[name] for name in PARAMS], dtype=float)
    lower, upper = bounds_arrays()
    span = upper - lower
    audit = {}
    for mode, fit in fits.items():
        starts = np.vstack([attempt["x0"] for attempt in fit["attempts"]])
        scaled_distances = np.linalg.norm((starts - truth) / span, axis=1)
        audit[mode] = {
            "start_count": int(len(starts)),
            "exact_truth_start_count": int(np.sum(np.all(starts == truth, axis=1))),
            "minimum_scaled_distance_to_truth": float(np.min(scaled_distances)),
            "maximum_scaled_distance_to_truth": float(np.max(scaled_distances)),
        }
    return audit


def fit_static(measurement: dict, config: FitConfig, seed: int) -> dict:
    residual, scales = make_residual(
        measurement["wavelengths_um"],
        measurement["I_meas"],
        config,
    )
    lower, upper = bounds_arrays()
    candidates, global_summary = global_candidate_pool(measurement, config, seed)
    attempts = []
    for candidate in candidates:
        guess = candidate["x0"]
        result = least_squares(
            residual,
            guess,
            bounds=(lower, upper),
            method="trf",
            loss=config.loss,
            max_nfev=config.max_nfev,
            x_scale="jac",
        )
        res = residual(result.x)
        selection_cost = 0.5 * float(np.dot(res, res))
        attempts.append(
            {
                "x0": np.asarray(guess, dtype=float),
                "x": np.asarray(result.x, dtype=float),
                "cost": selection_cost,
                "optimizer_robust_cost": float(result.cost),
                "rmse_normalized": math.sqrt(float(np.mean(res**2))),
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "optimality": float(result.optimality),
                "global_population_rank": candidate["global_population_rank"],
                "global_energy": candidate["global_energy"],
            }
        )
    attempts.sort(key=lambda row: (not row["success"], row["cost"]))
    best = attempts[0]
    if not best["success"]:
        raise RuntimeError("No local optimization start converged successfully.")
    values = best["x"]
    model_i = static_spectrum_model(measurement["wavelengths_um"], values)
    best.update(
        {
            "mode": "I",
            "model_I": model_i,
            "rmse_I": math.sqrt(float(np.mean((model_i - measurement["I_meas"]) ** 2))),
            "scales": scales,
            "attempts": attempts,
            "global_search": global_summary,
        }
    )
    return best


def approximate_jacobian(residual_fn, x: np.ndarray, rel_step: float = 1.0e-5) -> np.ndarray:
    base = residual_fn(x)
    jac = np.empty((len(base), len(x)), dtype=float)
    lower, upper = bounds_arrays()
    for j in range(len(x)):
        step = max(abs(x[j]) * rel_step, 1.0e-4)
        xp = x.copy()
        xm = x.copy()
        xp[j] = min(upper[j], xp[j] + step)
        xm[j] = max(lower[j], xm[j] - step)
        actual = xp[j] - xm[j]
        if actual <= 0:
            jac[:, j] = np.nan
        else:
            jac[:, j] = (residual_fn(xp) - residual_fn(xm)) / actual
    return jac


def diagnostics(measurement: dict, config: FitConfig, fits: dict[str, dict]) -> dict:
    out = {}
    for mode, fit in fits.items():
        residual_fn, _ = make_residual(
            measurement["wavelengths_um"],
            measurement["I_meas"],
            config,
        )
        jac = approximate_jacobian(residual_fn, fit["x"])
        finite = np.all(np.isfinite(jac), axis=0)
        if np.any(finite):
            s = np.linalg.svd(jac[:, finite], compute_uv=False)
            cond = float(s[0] / s[-1]) if len(s) and s[-1] > 0 else float("inf")
        else:
            s = np.array([], dtype=float)
            cond = float("nan")
        out[mode] = {"singular_values": s.tolist(), "condition_number": cond}
    return out


def plot_fits(output_dir: Path, measurement: dict, fits: dict[str, dict]) -> dict[str, str]:
    w_nm = measurement["wavelengths_um"] * 1000.0
    fit = fits["I"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axs[0].plot(w_nm, measurement["I_meas"], color="black", lw=1.0, label="measured static I")
    axs[0].plot(w_nm, fit["model_I"], lw=1.0, label="static TMM fit")
    axs[0].set_title("Static TMM fit to I(lambda)")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Reflectance")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(w_nm, fit["model_I"] - measurement["I_meas"], lw=1.0)
    axs[1].set_title("Static I(lambda) residual")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Reflectance")
    axs[1].grid(True)
    path = output_dir / "best_fit_static_spectrum.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"best_fit_static_spectrum": str(path)}


def save_diagnostics_plots(output_dir: Path, diag: dict) -> dict[str, str]:
    paths = {}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for mode, row in diag.items():
        s = np.asarray(row["singular_values"], dtype=float)
        if len(s):
            ax.semilogy(np.arange(1, len(s) + 1), s, marker="o", label=f"{mode}, cond={row['condition_number']:.3g}")
    ax.set_title("Jacobian singular values")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.grid(True)
    ax.legend()
    path = output_dir / "jacobian_singular_values.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["jacobian_singular_values"] = str(path)
    return paths


def scalar(npz, key: str, default):
    return np.asarray(npz[key]).item() if key in npz else default


def read_metadata_and_truth(npz_path: Path) -> tuple[dict, dict[str, float]]:
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = {
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "noise_level": str(scalar(data, "noise_level", "unknown")),
            "noise_factor": str(scalar(data, "noise_factor", "unknown")),
            "generator_version": str(scalar(data, "generator_version", "unknown")),
            "error_realization_policy": str(scalar(data, "error_realization_policy", "unknown")),
            "modulation_enabled": bool(scalar(data, "modulation_enabled", False)),
            "actual_angle_deg": float(scalar(data, "actual_angle_deg", 0.0)),
            "wavelength_offset_nm": float(scalar(data, "wavelength_offset_nm", 0.0)),
            "reflectance_offset_abs": float(scalar(data, "reflectance_offset_abs", 0.0)),
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
            "condition_number": diagnostics[mode]["condition_number"],
            "local_success_fraction": float(np.mean([x["success"] for x in fit["attempts"]])),
        }
        lower, upper = bounds_arrays()
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
    lower, upper = bounds_arrays()
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
    base_config: FitConfig,
    case_index: int,
) -> list[dict]:
    global EVALUATION_TRUTH
    metadata, truth = read_metadata_and_truth(npz_path)
    if metadata["modulation_enabled"]:
        raise ValueError("Input metadata reports modulation_enabled=True; static v6 data is required.")
    config = replace(base_config, input_npz=str(npz_path))
    measurement = load_measurement(npz_path, config)

    # Keep benchmark truth unavailable until optimization and ranking finish.
    EVALUATION_TRUTH = {}
    fit = fit_static(
        measurement,
        config,
        config.random_seed + case_index * 10000,
    )
    fits = {"I": fit}
    diagnostic_results = diagnostics(measurement, config, fits)
    EVALUATION_TRUTH = truth

    case_dir = root / metadata["noise_case"]
    case_dir.mkdir(parents=True, exist_ok=False)
    rows = save_tables(case_dir, metadata, truth, fits, diagnostic_results)
    plot_paths = plot_fits(case_dir, measurement, fits)
    plot_paths.update(save_diagnostics_plots(case_dir, diagnostic_results))
    summary = {
        "version": VERSION,
        "input_npz_resolved": str(npz_path),
        "metadata": metadata,
        "config": asdict(config),
        "mode": "I",
        "params": PARAMS,
        "bounds": BOUNDS,
        "evaluation_truth": truth,
        "truth_usage_policy": "loaded for reporting but hidden from optimization, initialization, residuals, priors, and ranking",
        "postfit_initialization_audit": postfit_initialization_audit(fits),
        "global_search": fit["global_search"],
        "diagnostics": diagnostic_results,
        "plot_paths": plot_paths,
        "tmm_convention": {
            "frequency_axis": "f = 3e8 / lambda_nominal",
            "phase_wavelength": "lambda_phase = 299792458 / f",
            "polarization": "p",
            "oblique_admittance": "q_p = n / cos(theta_layer)",
            "snell_invariant": "n0*sin(theta0) = nj*sin(thetaj)",
            "angle_parameter": "nonnegative StackRT incident-medium angle, bounded 0 to 0.1 deg",
            "observation_model": "single static reflectance spectrum I(lambda); no modulation or lock-in observable",
        },
    }
    (case_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"[{metadata['noise_case']}/static-I] "
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
    selected = table.copy()
    factors = ["angle", "wavelength", "material", "detector", "combined"]
    levels = ["low", "medium", "high"]
    metrics = [
        ("film_mae_nm", "Film thickness MAE (nm)"),
        ("air_phase_abs_error_um", "Air phase-equivalent cavity absolute error (um)"),
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
    fig.suptitle("Angle-aware static-spectrum inversion accuracy by error factor")
    fig.savefig(root / "noise_accuracy_overview.png", dpi=200)
    plt.close(fig)

    excel_cmap = LinearSegmentedColormap.from_list(
        "excel_green_yellow_red",
        ["#63BE7B", "#FFEB84", "#F8696B"],
    )
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    heatmap_metrics = (
        (
            "air_phase_abs_error_um",
            "Phase-equivalent cavity length absolute error (nm)",
            1000.0,
        ),
        ("film_mae_nm", "Film MAE (nm)", 1.0),
    )
    for ax, (column, title, unit_scale) in zip(axs, heatmap_metrics):
        matrix = np.full((len(factors), len(levels)), np.nan)
        for i, factor in enumerate(factors):
            for j, level in enumerate(levels):
                match = selected[
                    (selected["noise_factor"] == factor)
                    & (selected["noise_level"] == level)
                ]
                if not match.empty:
                    matrix[i, j] = float(match.iloc[0][column]) * unit_scale
        image = ax.imshow(matrix, aspect="auto", cmap=excel_cmap)
        ax.set_xticks(np.arange(-0.5, len(levels), 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, len(factors), 1.0), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(range(len(levels)), labels=levels)
        ax.set_yticks(range(len(factors)), labels=factors)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title(title, fontsize=17, pad=12)
        for i in range(len(factors)):
            for j in range(len(levels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j, i, f"{matrix[i, j]:.3g}",
                        ha="center", va="center", color="black", fontsize=14,
                    )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
        colorbar.ax.tick_params(labelsize=13)
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


def select_latest_inputs_by_case(paths: list[Path]) -> list[Path]:
    latest: dict[str, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            case_name = str(scalar(data, "noise_case", path.stem))
        previous = latest.get(case_name)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest[case_name] = path
    return sorted(latest.values(), key=lambda path: path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Angle-aware v6 static TMM inversion for main_dynamic_v6 StackRT datasets."
    )
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--pattern", default="static_spectrum_*.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--wavelength-min-nm", type=float, default=220.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--multistarts", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=250)
    parser.add_argument("--global-popsize", type=int, default=12)
    parser.add_argument("--global-maxiter", type=int, default=80)
    parser.add_argument("--global-stride", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260721)
    parser.add_argument("--use-prior", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="soft_l1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inputs:
        inputs = [Path(value).resolve() for value in args.inputs]
    else:
        matched = sorted(Path(args.input_dir).resolve().glob(args.pattern))
        inputs = select_latest_inputs_by_case(matched)
    if args.max_files is not None:
        inputs = inputs[:args.max_files]
    if not inputs:
        raise FileNotFoundError("No static input NPZ files matched.")

    root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else OUTPUT_ROOT / f"{VERSION}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    root.mkdir(parents=True, exist_ok=False)
    config = FitConfig(
        input_npz="",
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        stride=args.stride,
        multistarts=args.multistarts,
        max_nfev=args.max_nfev,
        global_popsize=args.global_popsize,
        global_maxiter=args.global_maxiter,
        global_stride=args.global_stride,
        random_seed=args.random_seed,
        use_prior=args.use_prior,
        loss=args.loss,
    )
    rows = []
    failures = []
    for index, npz_path in enumerate(inputs):
        try:
            rows.extend(run_case(npz_path, root, config, index))
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
                "mode": "static_I",
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
