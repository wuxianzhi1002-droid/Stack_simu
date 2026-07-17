from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"
DEFAULT_INPUT_DIR = OUTPUT_ROOT / "dynamic_stackrt_lockin_v4"
DEFAULT_INPUT_NPZ = DEFAULT_INPUT_DIR / "dynamic_spectra_low_20260716_164051.npz"

PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2"]

# Simulation truth is used only after optimization to calculate benchmark
# errors. It must never enter initial guesses, priors, residuals, or ranking.
EVALUATION_TRUTH = {
    "Air": 1000.0,
    "HSQ": 30.0,
    "PSS": 10.0,
    "SOC": 40.0,
    "TiO2": 40.0,
}

BOUNDS = {
    "Air": (998.0, 1002.0),
    "HSQ": (20.0, 40.0),
    "PSS": (1.0, 20.0),
    "SOC": (30.0, 50.0),
    "TiO2": (30.0, 50.0),
}

# Optional process prior, derived only from bounds and disabled by default.
PRIOR_CENTER = {name: 0.5 * (bounds[0] + bounds[1]) for name, bounds in BOUNDS.items()}
PRIOR_SIGMA = {
    "Air": 2.0,
    "HSQ": 10.0,
    "PSS": 5.0,
    "SOC": 10.0,
    "TiO2": 10.0,
}
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
C0_M_S = 299_792_458.0
FREQUENCY_AXIS_C_M_S = 3.0e8


@dataclass
class FitConfig:
    input_npz: str
    amplitude_nm: float
    wavelength_min_nm: float = 220.0
    wavelength_max_nm: float = 580.0
    stride: int = 10
    multistarts: int = 32
    max_nfev: int = 300
    global_popsize: int = 16
    global_maxiter: int = 60
    global_stride: int = 50
    global_phase_samples: int = 8
    local_phase_samples: int = 16
    random_seed: int = 20260715
    use_prior: bool = False
    loss: str = "soft_l1"


def material_n(name: str, wavelengths_um: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=np.float64)
    if name == "RefReflector":
        return np.full_like(w, 5.8284, dtype=np.complex128)
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


def characteristic_admittance(n_values: np.ndarray, cos_values: np.ndarray, pol: str) -> np.ndarray:
    if pol.lower() == "s":
        return n_values * cos_values
    if pol.lower() == "p":
        return cos_values / n_values
    raise ValueError(f"Unsupported polarization: {pol}")


def tmm_reflectance(
    wavelengths_um: np.ndarray,
    thicknesses_um: dict[str, float],
    pol: str = "p",
    theta_rad: float = 0.0,
) -> np.ndarray:
    """StackRT-matched normal-incidence TMM for the existing dynamic NPZ.

    main_dynamic_v2.py generated frequencies with 3e8/lambda_nominal, while
    StackRT propagates with the physical speed of light. Material indices are
    still evaluated at lambda_nominal, matching the original n_matrix input.
    """
    if pol.lower() not in {"s", "p"}:
        raise ValueError(f"Unsupported polarization: {pol}")
    if not np.isclose(theta_rad, 0.0):
        raise ValueError("The StackRT-matched v3 TMM currently supports normal incidence only.")

    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
    frequency_hz = FREQUENCY_AXIS_C_M_S / (wavelengths_um * 1e-6)
    phase_wavelength_m = C0_M_S / frequency_hz
    n_matrix = np.vstack([material_n(name, wavelengths_um) for name in LAYER_NAMES])
    thicknesses_m = np.array([thicknesses_um.get(name, 0.0) * 1e-6 for name in LAYER_NAMES])
    k0 = 2.0 * np.pi / phase_wavelength_m

    # At normal incidence s and p power reflectance are identical. This q=n,
    # -i convention matches StackRT for passive materials expressed as n+i*k.
    q_values = n_matrix

    m11 = np.ones(len(wavelengths_um), dtype=np.complex128)
    m12 = np.zeros(len(wavelengths_um), dtype=np.complex128)
    m21 = np.zeros(len(wavelengths_um), dtype=np.complex128)
    m22 = np.ones(len(wavelengths_um), dtype=np.complex128)

    for layer_idx in range(1, len(LAYER_NAMES) - 1):
        thickness = float(thicknesses_m[layer_idx])
        if thickness <= 0.0:
            continue
        delta = k0 * n_matrix[layer_idx, :] * thickness
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        q_layer = q_values[layer_idx, :]
        a11 = c_delta
        a12 = -1j * s_delta / q_layer
        a21 = -1j * q_layer * s_delta
        a22 = c_delta
        new11 = m11 * a11 + m12 * a21
        new12 = m11 * a12 + m12 * a22
        new21 = m21 * a11 + m22 * a21
        new22 = m21 * a12 + m22 * a22
        m11, m12, m21, m22 = new11, new12, new21, new22

    q0 = q_values[0, :]
    qs = q_values[-1, :]
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


def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([BOUNDS[name][0] for name in PARAMS], dtype=float)
    upper = np.array([BOUNDS[name][1] for name in PARAMS], dtype=float)
    return lower, upper


def finite_modulation_observables(
    wavelengths_um: np.ndarray,
    values: np.ndarray,
    amplitude_um: float,
    phase_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same finite-amplitude averaging and 1f operator as the data."""
    if amplitude_um <= 0.0:
        raise ValueError("amplitude_um must be positive.")
    if phase_samples < 8:
        raise ValueError("phase_samples must be at least 8.")

    phases = 2.0 * np.pi * np.arange(phase_samples, dtype=float) / phase_samples
    sin_ref = np.sin(phases)
    spectra = np.empty((phase_samples, len(wavelengths_um)), dtype=float)
    for idx, sin_value in enumerate(sin_ref):
        modulated = np.asarray(values, dtype=float).copy()
        modulated[0] += amplitude_um * sin_value
        spectra[idx] = tmm_reflectance(wavelengths_um, fit_vector_to_um(modulated))

    mean_spectrum = np.mean(spectra, axis=0)
    lockin_x = 2.0 * (spectra.T @ sin_ref) / phase_samples
    return mean_spectrum, lockin_x / amplitude_um


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
    with np.load(npz_path) as z:
        wavelengths_um = z["wavelengths"].astype(float)
        spectra = z["spectra"].astype(float)
        lockin_x = z["lockin_1f_X"].astype(float)
        lockin_y = z["lockin_1f_Y"].astype(float) if "lockin_1f_Y" in z.files else np.zeros_like(lockin_x)
        lockin_r = z["lockin_1f_R"].astype(float) if "lockin_1f_R" in z.files else np.sqrt(lockin_x**2 + lockin_y**2)

    wavelengths_nm = wavelengths_um * 1000.0
    mask = (wavelengths_nm >= config.wavelength_min_nm) & (wavelengths_nm <= config.wavelength_max_nm)
    masked_idx = np.where(mask)[0]
    idx = masked_idx[:: max(1, int(config.stride))]
    global_idx = masked_idx[:: max(1, int(config.global_stride))]
    if len(idx) < len(PARAMS) * 4:
        raise ValueError("Too few wavelength samples after mask/stride. Reduce --stride or widen wavelength range.")
    if len(global_idx) < len(PARAMS) * 4:
        raise ValueError("Too few global-search samples. Reduce --global-stride or widen wavelength range.")

    amplitude_um = config.amplitude_nm / 1000.0
    if amplitude_um <= 0.0:
        raise ValueError("--amplitude-nm must be positive.")

    return {
        "wavelengths_um_full": wavelengths_um,
        "I_meas_full": np.mean(spectra, axis=0),
        "dIdL_x_full": lockin_x / amplitude_um,
        "dIdL_y_full": lockin_y / amplitude_um,
        "dIdL_r_full": lockin_r / amplitude_um,
        "amplitude_um": amplitude_um,
        "idx": idx,
        "wavelengths_um": wavelengths_um[idx],
        "I_meas": np.mean(spectra, axis=0)[idx],
        "dIdL_meas": (lockin_x / amplitude_um)[idx],
        "global_idx": global_idx,
        "global_wavelengths_um": wavelengths_um[global_idx],
        "global_I_meas": np.mean(spectra, axis=0)[global_idx],
        "global_dIdL_meas": (lockin_x / amplitude_um)[global_idx],
    }


def make_residual(
    wavelengths_um: np.ndarray,
    I_meas: np.ndarray,
    dIdL_meas: np.ndarray,
    amplitude_um: float,
    phase_samples: int,
    config: FitConfig,
    mode: str,
):
    sigma_i = robust_sigma(I_meas, floor=1.0e-4)
    sigma_d = robust_sigma(dIdL_meas, floor=1.0e-3)
    prior_sigma = np.array([PRIOR_SIGMA[name] for name in PARAMS], dtype=float)
    prior_center = np.array([PRIOR_CENTER[name] for name in PARAMS], dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        model_i, model_d = finite_modulation_observables(
            wavelengths_um,
            values,
            amplitude_um,
            phase_samples,
        )
        blocks = []
        if mode in {"I", "joint"}:
            blocks.append((model_i - I_meas) / sigma_i)
        if mode in {"D", "joint"}:
            blocks.append((model_d - dIdL_meas) / sigma_d)
        if config.use_prior:
            blocks.append((values - prior_center) / prior_sigma)
        return np.concatenate(blocks)

    return residual, {"sigma_I": sigma_i, "sigma_dIdL": sigma_d}


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
    mode: str,
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
        measurement["global_dIdL_meas"],
        measurement["amplitude_um"],
        config.global_phase_samples,
        config,
        mode,
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


def fit_mode(measurement: dict, config: FitConfig, mode: str, seed: int) -> dict:
    residual, scales = make_residual(
        measurement["wavelengths_um"],
        measurement["I_meas"],
        measurement["dIdL_meas"],
        measurement["amplitude_um"],
        config.local_phase_samples,
        config,
        mode,
    )
    lower, upper = bounds_arrays()
    candidates, global_summary = global_candidate_pool(measurement, config, mode, seed)
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
    # Rank by a numerically stable raw residual cost. SciPy soft_l1 cost can
    # round to exactly zero for near-machine-precision synthetic closures.
    attempts.sort(key=lambda row: (not row["success"], row["cost"]))
    best = attempts[0]
    values = best["x"]
    model_i, model_d = finite_modulation_observables(
        measurement["wavelengths_um"],
        values,
        measurement["amplitude_um"],
        config.local_phase_samples,
    )
    best.update(
        {
            "mode": mode,
            "model_I": model_i,
            "model_dIdL": model_d,
            "rmse_I": math.sqrt(float(np.mean((model_i - measurement["I_meas"]) ** 2))),
            "rmse_dIdL": math.sqrt(float(np.mean((model_d - measurement["dIdL_meas"]) ** 2))),
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
            measurement["dIdL_meas"],
            measurement["amplitude_um"],
            config.local_phase_samples,
            config,
            mode,
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


def save_fit_results(output_dir: Path, fits: dict[str, dict], diag: dict) -> Path:
    rows = []
    for mode, fit in fits.items():
        row = {
            "mode": mode,
            "success": fit["success"],
            "cost": fit["cost"],
            "optimizer_robust_cost": fit["optimizer_robust_cost"],
            "rmse_normalized": fit["rmse_normalized"],
            "rmse_I": fit["rmse_I"],
            "rmse_dIdL": fit["rmse_dIdL"],
            "nfev": fit["nfev"],
            "optimality": fit["optimality"],
            "condition_number": diag[mode]["condition_number"],
        }
        for name, value in zip(PARAMS, fit["x"]):
            row[f"fit_{name}_{'um' if name == 'Air' else 'nm'}"] = value
            row[f"error_{name}_{'um' if name == 'Air' else 'nm'}"] = value - EVALUATION_TRUTH[name]
        rows.append(row)
    path = output_dir / "fit_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
    return path


def save_multistart_results(output_dir: Path, fits: dict[str, dict]) -> Path:
    rows = []
    lower, upper = bounds_arrays()
    span = upper - lower
    truth = np.array([EVALUATION_TRUTH[name] for name in PARAMS], dtype=float)
    for mode, fit in fits.items():
        for idx, attempt in enumerate(fit["attempts"]):
            distance_to_best = float(np.linalg.norm((attempt["x"] - fit["x"]) / span))
            distance_to_truth = float(np.linalg.norm((attempt["x"] - truth) / span))
            row = {
                "mode": mode,
                "rank": idx + 1,
                "success": attempt["success"],
                "cost": attempt["cost"],
                "optimizer_robust_cost": attempt["optimizer_robust_cost"],
                "rmse_normalized": attempt["rmse_normalized"],
                "nfev": attempt["nfev"],
                "optimality": attempt["optimality"],
                "message": attempt["message"],
                "global_population_rank": attempt["global_population_rank"],
                "global_energy": attempt["global_energy"],
                "distance_to_best_scaled": distance_to_best,
                "benchmark_distance_to_truth_scaled": distance_to_truth,
            }
            for name, value in zip(PARAMS, attempt["x"]):
                row[f"fit_{name}_{'um' if name == 'Air' else 'nm'}"] = value
                row[f"benchmark_error_{name}_{'um' if name == 'Air' else 'nm'}"] = value - EVALUATION_TRUTH[name]
            for name, value in zip(PARAMS, attempt["x0"]):
                row[f"x0_{name}_{'um' if name == 'Air' else 'nm'}"] = value
            rows.append(row)
    path = output_dir / "multistart_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
    return path


def plot_fits(output_dir: Path, measurement: dict, fits: dict[str, dict]) -> dict[str, str]:
    w_nm = measurement["wavelengths_um"] * 1000.0
    paths = {}

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axs[0].plot(w_nm, measurement["I_meas"], color="black", lw=1.0, label="measured I")
    for mode, fit in fits.items():
        axs[0].plot(w_nm, fit["model_I"], lw=1.0, label=f"{mode} model")
    axs[0].set_title("TMM fit to I(lambda)")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Reflectance")
    axs[0].grid(True)
    axs[0].legend()

    for mode, fit in fits.items():
        axs[1].plot(w_nm, fit["model_I"] - measurement["I_meas"], lw=1.0, label=f"{mode} residual")
    axs[1].set_title("I(lambda) residual")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Reflectance")
    axs[1].grid(True)
    axs[1].legend()
    path = output_dir / "best_fit_spectrum.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["best_fit_spectrum"] = str(path)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axs[0].plot(w_nm, measurement["dIdL_meas"], color="black", lw=1.0, label="measured lockin_1f_X / A_um")
    for mode, fit in fits.items():
        axs[0].plot(w_nm, fit["model_dIdL"], lw=1.0, label=f"{mode} model")
    axs[0].set_title("TMM fit to dI/dL(lambda)")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Reflectance / um")
    axs[0].grid(True)
    axs[0].legend()

    for mode, fit in fits.items():
        axs[1].plot(w_nm, fit["model_dIdL"] - measurement["dIdL_meas"], lw=1.0, label=f"{mode} residual")
    axs[1].set_title("dI/dL residual")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Reflectance / um")
    axs[1].grid(True)
    axs[1].legend()
    path = output_dir / "best_fit_dIdL.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["best_fit_dIdL"] = str(path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    modes = list(fits.keys())
    axs[0].bar(modes, [fits[m]["rmse_I"] for m in modes])
    axs[0].set_title("I RMSE")
    axs[0].set_ylabel("Reflectance")
    axs[0].grid(True, axis="y")
    axs[1].bar(modes, [fits[m]["rmse_dIdL"] for m in modes])
    axs[1].set_title("dI/dL RMSE")
    axs[1].set_ylabel("Reflectance / um")
    axs[1].grid(True, axis="y")
    path = output_dir / "joint_residual.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["joint_residual"] = str(path)

    return paths


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


def write_summary(output_dir: Path, config: FitConfig, fits: dict[str, dict], diag: dict, paths: dict, fit_csv: Path, multistart_csv: Path) -> Path:
    lines = [
        "# TMM Joint Inversion With Lock-in dI/dL (v3)",
        "",
        "## Method",
        "",
        "- Input channels: `I(lambda)` from mean dynamic spectra and `dI/dL(lambda)` from `lockin_1f_X / A_um`.",
        "- Forward model: StackRT-matched coherent TMM with `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`.",
        "- Observation model: finite-amplitude sinusoidal modulation, time averaging, and digital 1f demodulation are reproduced inside every residual evaluation.",
        "- Convention: `lambda_phase = 299792458 / (3e8 / lambda_nominal)`, `q = n`, and `-i` matrix off-diagonal terms for `n+i*k`.",
        "- Initialization: differential evolution with a Latin-hypercube population; no truth or nominal vector is inserted as a start.",
        f"- Local multistarts: top {config.multistarts} candidates from each mode's independent global population.",
        f"- Prior enabled: `{config.use_prior}`; v3 defaults to no prior.",
        "- Selection: converged attempts are ranked before failed attempts, then by robust cost.",
        "- Fitted parameters: Air in um; HSQ/PSS/SOC/TiO2 in nm.",
        "- Comparison modes: I-only, dI/dL-only, joint.",
        "",
        "## Best Fit",
        "",
        "| mode | Air um | HSQ nm | PSS nm | SOC nm | TiO2 nm | RMSE I | RMSE dI/dL | cond |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, fit in fits.items():
        vals = fit["x"]
        lines.append(
            f"| {mode} | {vals[0]:.6g} | {vals[1]:.6g} | {vals[2]:.6g} | {vals[3]:.6g} | {vals[4]:.6g} | "
            f"{fit['rmse_I']:.6g} | {fit['rmse_dIdL']:.6g} | {diag[mode]['condition_number']:.6g} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Fit results: `{fit_csv.name}`",
        f"- Multistart results: `{multistart_csv.name}`",
    ]
    for key, value in paths.items():
        lines.append(f"- {key}: `{Path(value).name}`")
    lines += ["", "## Config", "", "```json", json.dumps(asdict(config), indent=2, ensure_ascii=False), "```", ""]
    path = output_dir / "summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(config: FitConfig, modes: list[str]) -> Path:
    input_npz = Path(config.input_npz)
    if not input_npz.is_absolute():
        input_npz = DEFAULT_INPUT_DIR / input_npz
    if not input_npz.exists():
        raise FileNotFoundError(input_npz)

    output_dir = OUTPUT_ROOT / f"tmm_joint_inversion_lockin_v3_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)

    measurement = load_measurement(input_npz, config)
    fits = {
        mode: fit_mode(measurement, config, mode, config.random_seed + mode_index * 1000)
        for mode_index, mode in enumerate(modes)
    }
    initialization_audit = postfit_initialization_audit(fits)
    diag = diagnostics(measurement, config, fits)

    fit_csv = save_fit_results(output_dir, fits, diag)
    multistart_csv = save_multistart_results(output_dir, fits)
    plot_paths = plot_fits(output_dir, measurement, fits)
    plot_paths.update(save_diagnostics_plots(output_dir, diag))

    config_path = output_dir / "fit_summary.json"
    config_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "input_npz_resolved": str(input_npz),
                "params": PARAMS,
                "evaluation_truth_fit_units": EVALUATION_TRUTH,
                "bounds_fit_units": BOUNDS,
                "prior_center_fit_units": PRIOR_CENTER,
                "prior_sigma_fit_units": PRIOR_SIGMA,
                "truth_usage_policy": "evaluation only after optimization; never used in initialization, residuals, priors, or ranking",
                "postfit_initialization_audit": initialization_audit,
                "global_search": {mode: fit["global_search"] for mode, fit in fits.items()},
                "tmm_convention": {
                    "frequency_axis": "f = 3e8 / lambda_nominal",
                    "phase_wavelength": "lambda_phase = 299792458 / f",
                    "normal_incidence_admittance": "q = n",
                    "complex_index": "n + i*k",
                    "matrix_off_diagonal_sign": "-i",
                    "observation_model": "finite sinusoidal modulation, mean spectrum, and lockin_1f_X/A",
                    "initialization": "independent differential evolution per mode with Latin-hypercube population",
                    "multistart_selection": "top global-population candidates; successful local fits first, then ascending cost",
                },
                "fit_results_csv": str(fit_csv),
                "multistart_results_csv": str(multistart_csv),
                "plot_paths": plot_paths,
                "diagnostics": diag,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = write_summary(output_dir, config, fits, diag, plot_paths, fit_csv, multistart_csv)

    print(f"OUTPUT_DIR={output_dir}")
    print(f"FIT_RESULTS={fit_csv}")
    print(f"REPORT={report}")
    for mode, fit in fits.items():
        values = ", ".join(f"{name}={value:.6g}" for name, value in zip(PARAMS, fit["x"]))
        print(f"{mode}: {values}; rmse_I={fit['rmse_I']:.6g}; rmse_dIdL={fit['rmse_dIdL']:.6g}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3 blind joint inversion with finite-modulation TMM and global-to-local multistart search."
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        default=str(DEFAULT_INPUT_NPZ),
        help="Input dynamic_spectra_*.npz. Defaults to the explicit DEFAULT_INPUT_NPZ near the top of this script.",
    )
    parser.add_argument("--amplitude-nm", type=float, default=5.0, help="Height/air-gap modulation amplitude used to compute lockin_1f_X / A_um.")
    parser.add_argument("--wavelength-min-nm", type=float, default=220.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=10, help="Use every Nth wavelength sample for fitting. Use 1 for full spectrum.")
    parser.add_argument("--multistarts", type=int, default=32, help="Number of global-population candidates refined per mode.")
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--global-popsize", type=int, default=16, help="Differential-evolution population multiplier.")
    parser.add_argument("--global-maxiter", type=int, default=60)
    parser.add_argument("--global-stride", type=int, default=50)
    parser.add_argument("--global-phase-samples", type=int, default=8)
    parser.add_argument("--local-phase-samples", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260715)
    parser.add_argument("--use-prior", action="store_true", help="Enable the optional bounds-derived process prior. Disabled by default.")
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--modes", nargs="+", choices=["I", "D", "joint"], default=["I", "D", "joint"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FitConfig(
        input_npz=str(args.input_npz),
        amplitude_nm=float(args.amplitude_nm),
        wavelength_min_nm=float(args.wavelength_min_nm),
        wavelength_max_nm=float(args.wavelength_max_nm),
        stride=int(args.stride),
        multistarts=int(args.multistarts),
        max_nfev=int(args.max_nfev),
        global_popsize=int(args.global_popsize),
        global_maxiter=int(args.global_maxiter),
        global_stride=int(args.global_stride),
        global_phase_samples=int(args.global_phase_samples),
        local_phase_samples=int(args.local_phase_samples),
        random_seed=int(args.random_seed),
        use_prior=bool(args.use_prior),
        loss=str(args.loss),
    )
    run(config, list(args.modes))


if __name__ == "__main__":
    main()
