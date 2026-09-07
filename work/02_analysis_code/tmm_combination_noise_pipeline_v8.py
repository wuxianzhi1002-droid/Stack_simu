"""Lumerical StackRT combination-noise generation and Python TMM inversion."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

LUMERICAL_API_PATH = Path(
    os.environ.get(
        "LUMERICAL_API_PATH",
        r"D:\Program Files\Lumerical\v241\api\python",
    )
)
LUMERICAL_BIN_PATH = Path(
    os.environ.get(
        "LUMERICAL_BIN_PATH",
        r"D:\Program Files\Lumerical\v241\bin",
    )
)
if LUMERICAL_API_PATH.exists():
    if str(LUMERICAL_API_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_API_PATH))
    os.environ["PATH"] = (
        os.environ.get("PATH", "")
        + os.pathsep
        + str(LUMERICAL_BIN_PATH)
    )

try:
    import lumapi
except ImportError as exc:
    lumapi = None
    LUMAPI_IMPORT_ERROR = exc
else:
    LUMAPI_IMPORT_ERROR = None


VERSION = "tmm_combination_noise_pipeline_v8_stackrt"
C0_M_S = 299_792_458.0
INCIDENT_MEDIUM_N = 5.8284
MAX_WAVELENGTH_ERROR_NM = 0.2
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2", "Angle"]
FILM_PARAMS = ["HSQ", "PSS", "SOC", "TiO2"]
BOUNDS = {
    "Air": (95.0, 105.0), "HSQ": (20.0, 40.0), "PSS": (1.0, 20.0),
    "SOC": (30.0, 50.0), "TiO2": (30.0, 50.0), "Angle": (0.0, 0.1),
}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]   
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"

@dataclass
class FitConfig:
    input_dir: str
    wavelength_min_nm: float = 450.0
    wavelength_max_nm: float = 580.0
    stride: int = 5
    global_stride: int = 4
    global_popsize: int = 8
    global_maxiter: int = 40
    multistarts: int = 8
    max_nfev: int = 600
    local_gtol: float = 1.0e-5
    workers: int = 1
    random_seed: int = 20260810
    loss: str = "soft_l1"

def scalar(npz, key: str, default):
    return np.asarray(npz[key]).item() if key in npz else default

def parameter_unit(name: str) -> str:
    return "um" if name == "Air" else "deg" if name == "Angle" else "nm"

def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([BOUNDS[name][0] for name in PARAMS], dtype=float),
        np.asarray([BOUNDS[name][1] for name in PARAMS], dtype=float),
    )

def material_n(name: str, wavelengths_um: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=float)
    if name == "RefReflector": return np.full_like(w, INCIDENT_MEDIUM_N, dtype=np.complex128)
    if name == "Air": return np.ones_like(w, dtype=np.complex128)
    if name == "HSQ": return np.full_like(w, 1.41, dtype=np.complex128)
    if name == "PSS": return np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    if name == "SOC": return (1.55 + 0.005 / (w**2)).astype(np.complex128)
    if name == "TiO2": return (2.4 + 0.02 / (w**2)).astype(np.complex128)
    if name == "Cu": return np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    raise ValueError(f"Unknown material: {name}")

def propagation_cosines(n_matrix: np.ndarray, reflector_angle_deg: float) -> np.ndarray:
    tangential_index = n_matrix[0] * np.sin(np.deg2rad(float(reflector_angle_deg)))
    cos_values = np.sqrt(1.0 - (tangential_index[None, :] / n_matrix) ** 2)
    cos_values[np.real(cos_values) < 0.0] *= -1.0
    return cos_values

def vector_to_thickness_um(values: np.ndarray) -> tuple[dict[str, float], float]:
    values = np.asarray(values, dtype=float)
    return {
        "RefReflector": 0.0, "Air": float(values[0]), "HSQ": float(values[1]) / 1000.0,
        "PSS": float(values[2]) / 1000.0, "SOC": float(values[3]) / 1000.0,
        "TiO2": float(values[4]) / 1000.0, "Cu": 0.0,
    }, float(values[5])

def tmm_reflectance(wavelengths_um: np.ndarray, values: np.ndarray) -> np.ndarray:
    thicknesses_um, reflector_angle_deg = vector_to_thickness_um(values)
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    n_matrix = np.vstack([material_n(name, wavelengths_um) for name in LAYER_NAMES])
    cos_values = propagation_cosines(n_matrix, reflector_angle_deg)
    q_values = n_matrix / cos_values
    k0 = 2.0 * np.pi / (wavelengths_um * 1.0e-6)
    m11 = np.ones(len(wavelengths_um), dtype=complex)
    m12 = np.zeros(len(wavelengths_um), dtype=complex)
    m21 = np.zeros(len(wavelengths_um), dtype=complex)
    m22 = np.ones(len(wavelengths_um), dtype=complex)
    for layer_index, name in enumerate(LAYER_NAMES[1:-1], start=1):
        thickness_m = thicknesses_um[name] * 1.0e-6
        if thickness_m <= 0.0: continue
        delta = k0 * n_matrix[layer_index] * cos_values[layer_index] * thickness_m
        c_delta, s_delta, q_layer = np.cos(delta), np.sin(delta), q_values[layer_index]
        a11, a12 = c_delta, -1j * s_delta / q_layer
        a21, a22 = -1j * q_layer * s_delta, c_delta
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21, m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21, m21 * a12 + m22 * a22,
        )
    q0, qs = q_values[0], q_values[-1]
    numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
    denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
    return np.abs(numerator / denominator) ** 2

def reflector_to_air_angle_deg(reflector_angle_deg: float) -> float:
    invariant = INCIDENT_MEDIUM_N * math.sin(math.radians(float(reflector_angle_deg)))
    if abs(invariant) > 1.0: raise ValueError("Reflector angle has no Air-layer solution.")
    return math.degrees(math.asin(invariant))

def load_fit_input(npz_path: Path, config: FitConfig) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        if bool(scalar(data, "ils_enabled", False)): raise ValueError("ILS must be disabled.")
        if bool(scalar(data, "time_series_enabled", False)): raise ValueError("Static data required.")
        if int(scalar(data, "frames_per_realization", 1)) != 1: raise ValueError("One frame required.")
        if bool(scalar(data, "modulation_enabled", False)): raise ValueError("No modulation accepted.")
        wavelengths_um = np.asarray(data["wavelengths"], dtype=float)
        spectrum = np.asarray(data["spectrum_measured"], dtype=float)
        wavelength_error = np.asarray(data["wavelength_error_total_nm"], dtype=float)
        if wavelengths_um.ndim != 1 or spectrum.shape != wavelengths_um.shape:
            raise ValueError("Wavelength and spectrum arrays must be matching 1D arrays.")
        if float(np.max(np.abs(wavelength_error))) > MAX_WAVELENGTH_ERROR_NM + 1.0e-9:
            raise ValueError("Input wavelength error exceeds 0.2 nm.")
        wavelength_nm = wavelengths_um * 1000.0
        mask = (wavelength_nm >= config.wavelength_min_nm) & (wavelength_nm <= config.wavelength_max_nm)
        indices = np.where(mask)[0][::max(1, int(config.stride))]
        if len(indices) < 50: raise ValueError("Too few samples after mask and stride.")
        metadata = {
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "noise_factor": str(scalar(data, "noise_factor", "unknown")),
            "noise_level": str(scalar(data, "noise_level", "unknown")),
            "realization_index": int(scalar(data, "realization_index", 0)),
            "random_seed": int(scalar(data, "random_seed", 0)),
            "generator_version": str(scalar(data, "generator_version", "unknown")),
            "optical_backend": str(scalar(data, "optical_backend", "unknown")),
            "wavelength_error_max_abs_nm": float(np.max(np.abs(wavelength_error))),
            "wavelength_error_rms_nm": float(np.sqrt(np.mean(wavelength_error**2))),
        }
    selected_wavelengths = wavelengths_um[indices]
    return {
        "path": str(npz_path.resolve()), "wavelengths_um": selected_wavelengths,
        "spectrum": spectrum[indices],
        "actual_step_nm": float(np.median(np.diff(selected_wavelengths))) * 1000.0,
        "metadata": metadata,
    }

def load_evaluation_truth(npz_path: Path) -> tuple[dict[str, float], dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        names = [str(value) for value in np.asarray(data["layer_names"])]
        layers = dict(zip(names, np.asarray(data["layer_thickness_um"], dtype=float)))
        truth = {
            "Air": float(layers["Air"]), "HSQ": float(layers["HSQ"]) * 1000.0,
            "PSS": float(layers["PSS"]) * 1000.0, "SOC": float(layers["SOC"]) * 1000.0,
            "TiO2": float(layers["TiO2"]) * 1000.0,
            "Angle": float(scalar(data, "true_reflector_angle_deg", 0.0)),
        }
        audit = json.loads(str(scalar(data, "noise_realization_json", "{}")))
    return truth, audit

def validate_sampling(measurement: dict, config: FitConfig) -> dict:
    shortest_period_nm = config.wavelength_min_nm**2 / (2.0 * BOUNDS["Air"][1] * 1000.0)
    nyquist_max_step_nm = 0.5 * shortest_period_nm
    local_step_nm = measurement["actual_step_nm"]
    global_step_nm = local_step_nm * max(1, int(config.global_stride))
    if local_step_nm > nyquist_max_step_nm or global_step_nm > nyquist_max_step_nm:
        raise ValueError(
            f"Sampling violates Nyquist: local={local_step_nm:.6g}, global={global_step_nm:.6g}, "
            f"limit={nyquist_max_step_nm:.6g} nm."
        )
    return {"local_step_nm": local_step_nm, "global_step_nm": global_step_nm,
            "shortest_fringe_period_nm": shortest_period_nm,
            "nyquist_max_step_nm": nyquist_max_step_nm}


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, float(np.std(values)), 1.0e-4)

def robust_objective(residual_values: np.ndarray, loss: str) -> float:
    z = np.asarray(residual_values, dtype=float) ** 2
    if loss == "linear": rho = z
    elif loss == "soft_l1": rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber": rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy": rho = np.log1p(z)
    elif loss == "arctan": rho = np.arctan(z)
    else: raise ValueError(f"Unknown loss: {loss}")
    return float(np.sum(rho))

def latin_hypercube_population(seed: int, size: int) -> np.ndarray:
    lower, upper = bounds_arrays()
    sample = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n=size)
    return qmc.scale(sample, lower, upper)

def select_diverse_candidates(
    population: np.ndarray,
    energies: np.ndarray,
    count: int,
) -> list[tuple[np.ndarray, int, float]]:
    lower, upper = bounds_arrays()
    span = upper - lower
    selected = []
    for index in np.argsort(energies):
        candidate = np.asarray(population[index], dtype=float)
        scaled = (candidate - lower) / span
        if all(np.linalg.norm(scaled - previous[0]) >= 0.03 for previous in selected):
            selected.append((scaled, int(index), float(energies[index])))
        if len(selected) >= count: break
    return [(lower + scaled * span, index, energy) for scaled, index, energy in selected]

def approximate_jacobian(residual_fn, values: np.ndarray) -> np.ndarray:
    lower, upper = bounds_arrays()
    span = upper - lower
    jacobian = np.empty((len(residual_fn(values)), len(values)), dtype=float)
    for index in range(len(values)):
        step = max(1.0e-6 * span[index], 1.0e-8)
        plus, minus = values.copy(), values.copy()
        plus[index] = min(upper[index], plus[index] + step)
        minus[index] = max(lower[index], minus[index] - step)
        denominator = plus[index] - minus[index]
        jacobian[:, index] = (residual_fn(plus) - residual_fn(minus)) / denominator
    return jacobian

def fit_measurement(measurement: dict, config: FitConfig, seed: int) -> dict:
    wavelengths = measurement["wavelengths_um"]
    observed = measurement["spectrum"]
    scale = robust_scale(observed)
    global_indices = np.arange(0, len(wavelengths), max(1, int(config.global_stride)))
    global_wavelengths = wavelengths[global_indices]
    global_observed = observed[global_indices]
    lower, upper = bounds_arrays()

    def model(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        axis = global_wavelengths if use_global else wavelengths
        return tmm_reflectance(axis, values)

    def residual(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        target = global_observed if use_global else observed
        return (model(values, use_global) - target) / scale

    started = time.perf_counter()
    population_size = max(5, int(config.global_popsize) * len(PARAMS))
    initial_population = latin_hypercube_population(seed, population_size)
    initial_energies = np.asarray([
        robust_objective(residual(values, True), config.loss) for values in initial_population
    ])
    global_result = differential_evolution(
        lambda values: robust_objective(residual(values, True), config.loss),
        bounds=[BOUNDS[name] for name in PARAMS],
        strategy="best1bin",
        maxiter=int(config.global_maxiter),
        popsize=int(config.global_popsize),
        tol=1.0e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        init=initial_population,
        updating="immediate",
        workers=1,
    )
    evolved_population = np.asarray(global_result.population, dtype=float)
    evolved_energies = np.asarray(global_result.population_energies, dtype=float)
    candidates = select_diverse_candidates(
        evolved_population, evolved_energies, int(config.multistarts)
    )
    if not candidates:
        raise RuntimeError("Global search returned no candidate starts.")

    span = upper - lower
    angle_index = PARAMS.index("Angle")

    def physical_to_unit(values: np.ndarray) -> np.ndarray:
        unit_values = np.clip((np.asarray(values, dtype=float) - lower) / span, 0.0, 1.0)
        unit_values[angle_index] = unit_values[angle_index] ** 2
        return unit_values

    def unit_to_physical(unit_values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(unit_values, dtype=float).copy()
        transformed[angle_index] = np.sqrt(max(0.0, transformed[angle_index]))
        return lower + transformed * span

    attempts = []
    for start_rank, (x0, population_index, global_energy) in enumerate(candidates, start=1):
        x0_unit = physical_to_unit(x0)
        local = least_squares(
            lambda unit_values: residual(unit_to_physical(unit_values), False),
            x0=x0_unit,
            bounds=(np.zeros(len(PARAMS)), np.ones(len(PARAMS))),
            loss=config.loss,
            max_nfev=int(config.max_nfev),
            x_scale=1.0,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=float(config.local_gtol),
        )
        fitted_values = unit_to_physical(local.x)
        fitted = model(fitted_values, False)
        attempts.append({
            "start_rank": start_rank,
            "success": bool(local.success) and np.isfinite(local.cost),
            "message": str(local.message),
            "status": int(local.status),
            "optimality": float(local.optimality),
            "cost": float(local.cost),
            "rmse_reflectance": float(np.sqrt(np.mean((fitted - observed) ** 2))),
            "nfev": int(local.nfev),
            "x0": np.asarray(x0, dtype=float),
            "x": fitted_values,
            "global_population_index": int(population_index),
            "global_energy": float(global_energy),
        })
    attempts.sort(key=lambda row: (not row["success"], row["cost"]))
    successful = [row for row in attempts if row["success"]]
    runtime_s = time.perf_counter() - started
    if not successful:
        return {
            "success": False, "runtime_s": runtime_s, "attempts": attempts,
            "global_summary": {},
        }
    best = successful[0]
    fitted_values = np.asarray(best["x"], dtype=float)
    jacobian = approximate_jacobian(lambda values: residual(values, False), fitted_values)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).eps]
    condition_number = float(positive[0] / positive[-1]) if len(positive) else float("inf")
    tolerance = 1.0e-5 * (upper - lower)
    boundary_hits = [
        PARAMS[index] for index, value in enumerate(fitted_values)
        if value - lower[index] <= tolerance[index] or upper[index] - value <= tolerance[index]
    ]
    return {
        "success": True,
        "runtime_s": runtime_s,
        "x": fitted_values,
        "cost": float(best["cost"]),
        "rmse_reflectance": float(best["rmse_reflectance"]),
        "nfev": int(best["nfev"]),
        "attempts": attempts,
        "condition_number": condition_number,
        "singular_values": singular_values.tolist(),
        "boundary_hits": boundary_hits,
        "fitted_spectrum": model(fitted_values, False),
        "global_summary": {
            "initial_population_source": "LatinHypercube; no truth, nominal truth, or forced center point",
            "initial_population_size": int(len(initial_population)),
            "initial_best_energy": float(np.min(initial_energies)),
            "initial_median_energy": float(np.median(initial_energies)),
            "evolved_population_size": int(len(evolved_population)),
            "evolved_best_energy": float(np.min(evolved_energies)),
            "evolved_median_energy": float(np.median(evolved_energies)),
            "global_success": bool(global_result.success),
            "global_message": str(global_result.message),
            "global_nit": int(global_result.nit),
            "global_nfev": int(global_result.nfev),
        },
    }


def result_row(measurement: dict, fit: dict, truth: dict[str, float]) -> dict:
    metadata = measurement["metadata"]
    row = {
        "input_npz": measurement["path"], **metadata,
        "success": bool(fit["success"]), "fit_runtime_s": float(fit["runtime_s"]),
    }
    if not fit["success"]:
        return row
    for index, name in enumerate(PARAMS):
        unit = parameter_unit(name)
        value = float(fit["x"][index])
        row[f"fit_{name}_{unit}"] = value
        row[f"truth_{name}_{unit}"] = float(truth[name])
        row[f"error_{name}_{unit}"] = value - float(truth[name])
    row["cavity_error_nm"] = row["error_Air_um"] * 1000.0
    row["cavity_abs_error_nm"] = abs(row["cavity_error_nm"])
    row["film_mae_nm"] = float(np.mean([abs(row[f"error_{name}_nm"]) for name in FILM_PARAMS]))
    row["angle_abs_error_deg"] = abs(row["error_Angle_deg"])
    row["fit_air_angle_deg"] = reflector_to_air_angle_deg(row["fit_Angle_deg"])
    row["truth_air_angle_deg"] = reflector_to_air_angle_deg(row["truth_Angle_deg"])
    row.update({
        "cost": fit["cost"], "rmse_reflectance": fit["rmse_reflectance"],
        "nfev": fit["nfev"], "condition_number": fit["condition_number"],
        "boundary_hits": ";".join(fit["boundary_hits"]),
        "boundary_hit_count": len(fit["boundary_hits"]),
        "selected_start_rank": fit["attempts"][0]["start_rank"],
    })
    return row

def attempt_rows(measurement: dict, fit: dict, truth: dict[str, float]) -> list[dict]:
    rows = []
    for rank, attempt in enumerate(fit.get("attempts", []), start=1):
        row = {
            "input_npz": measurement["path"],
            "noise_case": measurement["metadata"]["noise_case"],
            "realization_index": measurement["metadata"]["realization_index"],
            "rank": rank, "success": attempt["success"], "status": attempt["status"],
            "message": attempt["message"], "optimality": attempt["optimality"],
            "cost": attempt["cost"], "rmse_reflectance": attempt["rmse_reflectance"],
            "nfev": attempt["nfev"],
            "global_population_index": attempt["global_population_index"],
            "global_energy": attempt["global_energy"],
        }
        for index, name in enumerate(PARAMS):
            unit = parameter_unit(name)
            row[f"x0_{name}_{unit}"] = float(attempt["x0"][index])
            row[f"fit_{name}_{unit}"] = float(attempt["x"][index])
            row[f"benchmark_error_{name}_{unit}"] = float(attempt["x"][index] - truth[name])
        rows.append(row)
    return rows

def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_无可用汇总结果。_"
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize_results(table: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for case, group in table.groupby("noise_case", sort=False):
        successful = group[group["success"]].copy()
        row = {
            "noise_case": case,
            "noise_factor": str(group.iloc[0]["noise_factor"]),
            "noise_level": str(group.iloc[0]["noise_level"]),
            "runs": int(len(group)),
            "successful_runs": int(len(successful)),
            "fit_success_rate": float(len(successful) / len(group)),
        }
        if not successful.empty:
            for name in PARAMS:
                unit = parameter_unit(name)
                errors = successful[f"error_{name}_{unit}"].to_numpy(dtype=float)
                if name == "Air": errors = errors * 1000.0; output_unit = "nm"
                else: output_unit = unit
                row[f"{name}_bias_{output_unit}"] = float(np.mean(errors))
                row[f"{name}_mae_{output_unit}"] = float(np.mean(np.abs(errors)))
                row[f"{name}_rmse_{output_unit}"] = float(np.sqrt(np.mean(errors**2)))
                row[f"{name}_p95_abs_{output_unit}"] = float(np.percentile(np.abs(errors), 95.0))
            row.update({
                "film_mae_nm_mean": float(successful["film_mae_nm"].mean()),
                "film_mae_nm_p95": float(successful["film_mae_nm"].quantile(0.95)),
                "cavity_max_abs_error_nm": float(successful["cavity_abs_error_nm"].max()),
                "angle_max_abs_error_deg": float(successful["angle_abs_error_deg"].max()),
                "fit_runtime_mean_s": float(successful["fit_runtime_s"].mean()),
                "fit_runtime_median_s": float(successful["fit_runtime_s"].median()),
                "fit_runtime_p95_s": float(successful["fit_runtime_s"].quantile(0.95)),
                "boundary_hit_rate": float((successful["boundary_hit_count"] > 0).mean()),
                "condition_number_median": float(successful["condition_number"].median()),
            })
        summaries.append(row)
    return pd.DataFrame(summaries)

def save_plots(output_dir: Path, table: pd.DataFrame, summary: pd.DataFrame, representatives: dict) -> list[str]:
    paths = []
    successful = table[table["success"]].copy()
    if successful.empty: return paths
    cases = successful["noise_case"].drop_duplicates().tolist()
    values = [successful.loc[successful["noise_case"] == case, "cavity_error_nm"].to_numpy() for case in cases]
    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    ax.boxplot(values, labels=cases, showfliers=True)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Cavity error (nm)")
    ax.set_title("V8 six-parameter single-frame inversion")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / "cavity_error_boxplot.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    factors = [value for value in summary["noise_factor"].drop_duplicates() if value != "clean"]
    levels = ["low", "medium", "high"]
    metrics = [
        ("Air_mae_nm", "Cavity MAE (nm)"),
        ("film_mae_nm_mean", "Film MAE (nm)"),
        ("Angle_mae_deg", "Reflector angle MAE (deg)"),
        ("fit_runtime_mean_s", "Mean inversion runtime (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, metrics):
        matrix = np.full((len(factors), len(levels)), np.nan)
        for i, factor in enumerate(factors):
            for j, level in enumerate(levels):
                match = summary[(summary["noise_factor"] == factor) & (summary["noise_level"] == level)]
                if not match.empty and column in match: matrix[i, j] = float(match.iloc[0][column])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(levels)), labels=levels)
        ax.set_yticks(range(len(factors)), labels=factors)
        ax.set_title(title)
        for i in range(len(factors)):
            for j in range(len(levels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3g}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, shrink=0.8)
    path = output_dir / "noise_factor_accuracy_heatmaps.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    if representatives:
        selected_names = [name for name in representatives if name == "clean" or name.endswith("_high")]
        fig, axes = plt.subplots(len(selected_names), 1, figsize=(13, max(5, 2.8 * len(selected_names))), squeeze=False, constrained_layout=True)
        for index, name in enumerate(selected_names):
            item = representatives[name]; ax = axes[index, 0]
            ax.plot(item["wavelengths_um"] * 1000.0, item["observed"], lw=0.7, label="measured")
            ax.plot(item["wavelengths_um"] * 1000.0, item["fitted"], lw=0.7, label="fit")
            ax.set_title(name); ax.set_ylabel("Reflectance"); ax.grid(True, alpha=0.3); ax.legend()
        axes[-1, 0].set_xlabel("Reported wavelength (nm)")
        path = output_dir / "representative_fits.png"
        fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths



def process_fit_task(payload: tuple[int, str, FitConfig]) -> dict:
    index, npz_path_text, config = payload
    npz_path = Path(npz_path_text)
    try:
        measurement = load_fit_input(npz_path, config)
        sampling_audit = validate_sampling(measurement, config)
        fit = fit_measurement(measurement, config, config.random_seed + index * 1009)
        return {
            "ok": True,
            "index": index,
            "npz_path": npz_path_text,
            "measurement": measurement,
            "sampling_audit": sampling_audit,
            "fit": fit,
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": index,
            "npz_path": npz_path_text,
            "error": f"{type(exc).__name__}: {exc}",
        }





REALIZATION_MIN_FRACTION = 0.8
WAVELENGTH_ERROR_HARD_MAX_NM = 0.2
PERTURBED_MATERIALS = FILM_PARAMS
COMBINATION_LEVELS = ["low", "medium", "high"]
COMBINATION_NOISE_LEVELS = {
    "low": {
        "n_real_sigma_rel": 0.0005, "k_sigma_rel": 0.01,
        "angle_max_deg": 0.01, "laser_residual_max_nm": 0.005,
        "axis_offset_max_nm": 0.0002, "axis_scale_max_ppm": 10.0,
        "thermal_drift_max_nm": 0.001, "absolute_accuracy_max_nm": 0.02,
        "frame_gain_sigma_rel": 0.0002,
        "reflectance_offset_sigma_abs": 0.0002,
    },
    "medium": {
        "n_real_sigma_rel": 0.002, "k_sigma_rel": 0.05,
        "angle_max_deg": 0.05, "laser_residual_max_nm": 0.02,
        "axis_offset_max_nm": 0.001, "axis_scale_max_ppm": 30.0,
        "thermal_drift_max_nm": 0.005, "absolute_accuracy_max_nm": 0.08,
        "frame_gain_sigma_rel": 0.001,
        "reflectance_offset_sigma_abs": 0.001,
    },
    "high": {
        "n_real_sigma_rel": 0.005, "k_sigma_rel": 0.1,
        "angle_max_deg": 0.1, "laser_residual_max_nm": 0.05,
        "axis_offset_max_nm": 0.005, "axis_scale_max_ppm": 60.0,
        "thermal_drift_max_nm": 0.02, "absolute_accuracy_max_nm": 0.12,
        "frame_gain_sigma_rel": 0.005,
        "reflectance_offset_sigma_abs": 0.005,
    },
}
COMBINATIONS = {
    "wavelength_calibration": {
        "axis_offset", "axis_scale", "thermal_drift", "absolute_accuracy",
    },
    "wavelength_all": {
        "laser_wavelength", "axis_offset", "axis_scale",
        "thermal_drift", "absolute_accuracy",
    },
    "angle_wavelength": {
        "angle", "laser_wavelength", "axis_offset", "axis_scale",
        "thermal_drift", "absolute_accuracy",
    },
    "material_wavelength": {
        "material", "laser_wavelength", "axis_offset", "axis_scale",
        "thermal_drift", "absolute_accuracy",
    },
    "detector_wavelength": {
        "detector", "laser_wavelength", "axis_offset", "axis_scale",
        "thermal_drift", "absolute_accuracy",
    },
    "non_wavelength": {"angle", "material", "detector"},
}
GENERATION_TRUTH = {
    "Air": 100.0, "HSQ": 30.0, "PSS": 10.0,
    "SOC": 40.0, "TiO2": 40.0, "Angle": 0.0,
}


def gen_material_n(
    name: str,
    wavelengths_um: np.ndarray,
    n_real_rel_delta: dict[str, float] | None = None,
    k_rel_delta: dict[str, float] | None = None,
) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=float)
    if name == "RefReflector":
        values = np.full_like(w, INCIDENT_MEDIUM_N, dtype=np.complex128)
    elif name == "Air":
        values = np.ones_like(w, dtype=np.complex128)
    elif name == "HSQ":
        values = np.full_like(w, 1.41, dtype=np.complex128)
    elif name == "PSS":
        values = np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    elif name == "SOC":
        values = (1.55 + 0.005 / (w**2)).astype(np.complex128)
    elif name == "TiO2":
        values = (2.4 + 0.02 / (w**2)).astype(np.complex128)
    elif name == "Cu":
        values = np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown material: {name}")
    if name in PERTURBED_MATERIALS:
        dn = 0.0 if n_real_rel_delta is None else float(n_real_rel_delta.get(name, 0.0))
        dk = 0.0 if k_rel_delta is None else float(k_rel_delta.get(name, 0.0))
        values = values.real * (1.0 + dn) + 1j * values.imag * (1.0 + dk)
    return values

def signed_level_value(maximum: float, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    magnitude = float(maximum) * rng.uniform(REALIZATION_MIN_FRACTION, 1.0)
    return magnitude if rng.integers(0, 2) else -magnitude

def positive_level_value(maximum: float, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    return float(maximum) * rng.uniform(REALIZATION_MIN_FRACTION, 1.0)

def remove_constant_and_linear_terms(values: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    x = (wavelengths_nm - np.mean(wavelengths_nm)) / max(float(np.ptp(wavelengths_nm)), 1.0)
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - design @ coefficients

def smooth_curve_with_peak(
    wavelengths_nm: np.ndarray,
    target_peak_nm: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_peak_nm <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)
    spacing_nm = float(np.median(np.diff(wavelengths_nm)))
    sigma_pixels = float(correlation_length_nm) / spacing_nm
    for _ in range(50):
        raw = gaussian_filter1d(rng.normal(size=len(wavelengths_nm)), sigma=sigma_pixels, mode="reflect")
        curve = remove_constant_and_linear_terms(raw, wavelengths_nm)
        peak = float(np.max(np.abs(curve)))
        if peak > 1.0e-12:
            return curve * (float(target_peak_nm) / peak)
    raise RuntimeError("Could not generate a nonzero smooth wavelength curve.")


def realize_combination_noise(
    combination: str,
    level: str,
    wavelengths_nm: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    active = COMBINATIONS[combination]
    profile = COMBINATION_NOISE_LEVELS[level]
    zeros = np.zeros_like(wavelengths_nm, dtype=float)
    n_delta = {
        name: signed_level_value(profile["n_real_sigma_rel"], rng)
        if "material" in active else 0.0
        for name in FILM_PARAMS
    }
    k_delta = {
        name: signed_level_value(profile["k_sigma_rel"], rng)
        if "material" in active else 0.0
        for name in FILM_PARAMS
    }
    angle_deg = (
        positive_level_value(profile["angle_max_deg"], rng)
        if "angle" in active else 0.0
    )
    laser_curve = (
        smooth_curve_with_peak(
            wavelengths_nm,
            positive_level_value(profile["laser_residual_max_nm"], rng),
            8.0,
            rng,
        )
        if "laser_wavelength" in active else zeros.copy()
    )
    absolute_curve = (
        smooth_curve_with_peak(
            wavelengths_nm,
            positive_level_value(profile["absolute_accuracy_max_nm"], rng),
            30.0,
            rng,
        )
        if "absolute_accuracy" in active else zeros.copy()
    )
    axis_offset_nm = (
        signed_level_value(profile["axis_offset_max_nm"], rng)
        if "axis_offset" in active else 0.0
    )
    thermal_drift_nm = (
        signed_level_value(profile["thermal_drift_max_nm"], rng)
        if "thermal_drift" in active else 0.0
    )
    axis_scale_ppm = (
        signed_level_value(profile["axis_scale_max_ppm"], rng)
        if "axis_scale" in active else 0.0
    )
    center_nm = 0.5 * (wavelengths_nm[0] + wavelengths_nm[-1])
    offset_curve = np.full_like(wavelengths_nm, axis_offset_nm)
    thermal_curve = np.full_like(wavelengths_nm, thermal_drift_nm)
    scale_curve = axis_scale_ppm * 1.0e-6 * (wavelengths_nm - center_nm)
    total_curve = (
        laser_curve + absolute_curve + offset_curve + thermal_curve + scale_curve
    )
    total_max = float(np.max(np.abs(total_curve)))
    if total_max > WAVELENGTH_ERROR_HARD_MAX_NM + 1.0e-12:
        raise ValueError(
            f"Total wavelength error {total_max:.9g} nm exceeds "
            f"{WAVELENGTH_ERROR_HARD_MAX_NM} nm."
        )
    physical_nm = wavelengths_nm + total_curve
    if np.any(np.diff(physical_nm) <= 0.0):
        raise ValueError("Physical wavelength mapping is not monotonic.")
    gain_error = (
        signed_level_value(profile["frame_gain_sigma_rel"], rng)
        if "detector" in active else 0.0
    )
    reflectance_offset = (
        signed_level_value(profile["reflectance_offset_sigma_abs"], rng)
        if "detector" in active else 0.0
    )
    metadata = {
        "combination": combination,
        "noise_case": f"{combination}_{level}",
        "noise_level": level,
        "active_factors": sorted(active),
        "material_n_real_rel_delta": n_delta,
        "material_k_rel_delta": k_delta,
        "reflector_angle_deg": angle_deg,
        "laser_residual_peak_nm": float(np.max(np.abs(laser_curve))),
        "axis_offset_nm": axis_offset_nm,
        "axis_scale_ppm": axis_scale_ppm,
        "axis_scale_edge_max_nm": float(np.max(np.abs(scale_curve))),
        "thermal_drift_nm": thermal_drift_nm,
        "absolute_accuracy_peak_nm": float(np.max(np.abs(absolute_curve))),
        "frame_gain_error_rel": gain_error,
        "reflectance_offset_abs": reflectance_offset,
        "total_wavelength_error_max_abs_nm": total_max,
        "total_wavelength_error_rms_nm": float(np.sqrt(np.mean(total_curve**2))),
    }
    components = {
        "laser_residual_nm": laser_curve,
        "axis_offset_nm": offset_curve,
        "axis_scale_nm": scale_curve,
        "thermal_drift_nm": thermal_curve,
        "absolute_accuracy_nm": absolute_curve,
        "total_nm": total_curve,
        "physical_nm": physical_nm,
    }
    return metadata, components


class StackRTSolver:
    """One reusable lumapi FDTD session for StackRT dataset generation."""

    def __init__(self, hide: bool = True):
        self.hide = bool(hide)
        self.fdtd = None

    def __enter__(self):
        if lumapi is None:
            raise RuntimeError(
                "lumapi is unavailable. Check LUMERICAL_API_PATH and the "
                "Lumerical Python API installation."
            ) from LUMAPI_IMPORT_ERROR
        self.fdtd = lumapi.FDTD(
            hide=self.hide,
            serverArgs={"use-solve": True},
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    def reflectance(
        self,
        wavelengths_um: np.ndarray,
        thicknesses_um: dict[str, float],
        reflector_angle_deg: float,
        n_real_rel_delta: dict[str, float],
        k_rel_delta: dict[str, float],
    ) -> np.ndarray:
        if self.fdtd is None:
            raise RuntimeError("The lumapi FDTD session is not open.")
        wavelengths_um = np.asarray(wavelengths_um, dtype=float)
        n_matrix = np.vstack([
            gen_material_n(
                name,
                wavelengths_um,
                n_real_rel_delta,
                k_rel_delta,
            )
            for name in LAYER_NAMES
        ])
        thicknesses_m = np.asarray(
            [thicknesses_um[name] for name in LAYER_NAMES],
            dtype=float,
        ) * 1.0e-6
        frequencies_hz = C0_M_S / (wavelengths_um * 1.0e-6)
        result = self.fdtd.stackrt(
            n_matrix,
            thicknesses_m,
            frequencies_hz,
            float(reflector_angle_deg),
        )
        reflectance = np.asarray(result["Rp"], dtype=float).reshape(-1)
        if reflectance.shape != wavelengths_um.shape:
            raise RuntimeError(
                "StackRT Rp length mismatch: "
                f"{reflectance.shape} versus {wavelengths_um.shape}."
            )
        if not np.all(np.isfinite(reflectance)):
            raise RuntimeError("StackRT Rp contains non-finite values.")
        return reflectance


def generation_thicknesses_um() -> dict[str, float]:
    return {
        "RefReflector": 0.0,
        "Air": GENERATION_TRUTH["Air"],
        "HSQ": GENERATION_TRUTH["HSQ"] / 1000.0,
        "PSS": GENERATION_TRUTH["PSS"] / 1000.0,
        "SOC": GENERATION_TRUTH["SOC"] / 1000.0,
        "TiO2": GENERATION_TRUTH["TiO2"] / 1000.0,
        "Cu": 0.0,
    }


def generate_combination_dataset(
    dataset_dir: Path,
    repeats: int,
    base_seed: int,
    solver: StackRTSolver,
) -> tuple[list[Path], pd.DataFrame]:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    wavelengths_nm = np.arange(450.0, 580.0 + 0.01, 0.02)
    paths: list[Path] = []
    rows = []
    representatives = {}
    total_count = len(COMBINATIONS) * len(COMBINATION_LEVELS) * repeats
    for combination_index, combination in enumerate(COMBINATIONS):
        for level_index, level in enumerate(COMBINATION_LEVELS):
            case_name = f"{combination}_{level}"
            for realization_index in range(repeats):
                seed = (
                    int(base_seed)
                    + combination_index * 1_000_000
                    + level_index * 100_000
                    + realization_index
                )
                rng = np.random.default_rng(seed)
                started = time.perf_counter()
                metadata, components = realize_combination_noise(
                    combination, level, wavelengths_nm, rng
                )
                physical_um = components["physical_nm"] / 1000.0
                clean = solver.reflectance(
                    physical_um,
                    generation_thicknesses_um(),
                    metadata["reflector_angle_deg"],
                    metadata["material_n_real_rel_delta"],
                    metadata["material_k_rel_delta"],
                )
                measured = np.clip(
                    clean * (1.0 + metadata["frame_gain_error_rel"])
                    + metadata["reflectance_offset_abs"],
                    0.0,
                    1.0,
                )
                runtime_s = time.perf_counter() - started
                metadata.update({
                    "version": VERSION,
                    "backend": "stackrt_lumapi",
                    "random_seed": seed,
                    "realization_index": realization_index,
                    "generation_runtime_s": runtime_s,
                })
                npz_path = dataset_dir / (
                    f"combination_spectrum_{case_name}_"
                    f"r{realization_index:04d}_seed{seed}.npz"
                )
                np.savez_compressed(
                    npz_path,
                    wavelengths=wavelengths_nm / 1000.0,
                    spectrum_measured=measured,
                    spectrum_clean=clean,
                    wavelength_error_total_nm=components["total_nm"],
                    wavelength_error_laser_residual_nm=components["laser_residual_nm"],
                    wavelength_error_axis_offset_nm=components["axis_offset_nm"],
                    wavelength_error_axis_scale_nm=components["axis_scale_nm"],
                    wavelength_error_thermal_drift_nm=components["thermal_drift_nm"],
                    wavelength_error_absolute_accuracy_nm=components["absolute_accuracy_nm"],
                    physical_wavelengths_nm=components["physical_nm"],
                    layer_names=np.asarray(LAYER_NAMES),
                    layer_thickness_um=np.asarray([
                        generation_thicknesses_um()[name] for name in LAYER_NAMES
                    ]),
                    true_reflector_angle_deg=np.asarray(metadata["reflector_angle_deg"]),
                    noise_case=np.asarray(case_name),
                    noise_factor=np.asarray(combination),
                    noise_level=np.asarray(level),
                    realization_index=np.asarray(realization_index),
                    random_seed=np.asarray(seed),
                    generator_version=np.asarray(VERSION),
                    optical_backend=np.asarray("stackrt_lumapi"),
                    stackrt_polarization=np.asarray("Rp"),
                    speed_of_light_m_s=np.asarray(C0_M_S),
                    ils_enabled=np.asarray(False),
                    time_series_enabled=np.asarray(False),
                    frames_per_realization=np.asarray(1),
                    modulation_enabled=np.asarray(False),
                    noise_realization_json=np.asarray(
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                    ),
                )
                paths.append(npz_path)
                rows.append({
                    "noise_case": case_name,
                    "noise_factor": combination,
                    "noise_level": level,
                    "realization_index": realization_index,
                    "random_seed": seed,
                    "reflector_angle_deg": metadata["reflector_angle_deg"],
                    "total_wavelength_error_max_abs_nm":
                        metadata["total_wavelength_error_max_abs_nm"],
                    "total_wavelength_error_rms_nm":
                        metadata["total_wavelength_error_rms_nm"],
                    "generation_runtime_s": runtime_s,
                    "npz_path": str(npz_path.resolve()),
                })
                representatives.setdefault(
                    case_name,
                    (wavelengths_nm.copy(), components["total_nm"].copy()),
                )
                print(
                    f"[generate {len(paths)}/{total_count}] {case_name} "
                    f"r={realization_index:02d} "
                    f"wavelength_max="
                    f"{metadata['total_wavelength_error_max_abs_nm']:.6g} nm",
                    flush=True,
                )
    index = pd.DataFrame(rows)
    index.to_csv(
        dataset_dir / "dataset_index.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    save_generation_outputs(
        dataset_dir, representatives, repeats, len(paths)
    )
    return paths, index


def save_generation_outputs(
    dataset_dir: Path,
    representatives: dict,
    repeats: int,
    dataset_count: int,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(17, 13), constrained_layout=True)
    for ax, combination in zip(axes.flat, COMBINATIONS):
        for level in COMBINATION_LEVELS:
            wavelength_nm, curve_nm = representatives[f"{combination}_{level}"]
            ax.plot(wavelength_nm, curve_nm, label=level)
        ax.set_title(combination)
        ax.set_xlabel("Nominal wavelength (nm)")
        ax.set_ylabel("Total wavelength error (nm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plot_path = dataset_dir / "representative_combination_wavelength_errors.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    manifest = {
        "version": VERSION,
        "backend": "stackrt_lumapi",
        "stackrt_polarization": "Rp",
        "lumerical_api_path": str(LUMERICAL_API_PATH),
        "combinations": {
            name: sorted(factors) for name, factors in COMBINATIONS.items()
        },
        "levels": COMBINATION_NOISE_LEVELS,
        "repeats_per_case": repeats,
        "case_count": len(COMBINATIONS) * len(COMBINATION_LEVELS),
        "dataset_count": dataset_count,
        "hard_wavelength_limit_nm": WAVELENGTH_ERROR_HARD_MAX_NM,
        "dataset_index": str((dataset_dir / "dataset_index.csv").resolve()),
        "representative_plot": str(plot_path.resolve()),
    }
    (dataset_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_combination_noise_ranges(
    dataset_dir: Path,
    inversion_dir: Path,
) -> pd.DataFrame:
    rows = []
    for npz_path in sorted(dataset_dir.glob("combination_spectrum_*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = json.loads(
                str(np.asarray(data["noise_realization_json"]).item())
            )
        n_values = [
            abs(float(value))
            for value in metadata["material_n_real_rel_delta"].values()
        ]
        k_values = [
            abs(float(value))
            for value in metadata["material_k_rel_delta"].values()
        ]
        rows.append({
            "noise_case": metadata["noise_case"],
            "noise_factor": metadata["combination"],
            "noise_level": metadata["noise_level"],
            "angle_abs_deg": abs(float(metadata["reflector_angle_deg"])),
            "laser_residual_peak_nm":
                abs(float(metadata["laser_residual_peak_nm"])),
            "axis_offset_abs_nm": abs(float(metadata["axis_offset_nm"])),
            "axis_scale_abs_ppm": abs(float(metadata["axis_scale_ppm"])),
            "thermal_drift_abs_nm": abs(float(metadata["thermal_drift_nm"])),
            "absolute_accuracy_peak_nm":
                abs(float(metadata["absolute_accuracy_peak_nm"])),
            "material_n_max_abs_rel": max(n_values, default=0.0),
            "material_k_max_abs_rel": max(k_values, default=0.0),
            "detector_gain_abs_rel":
                abs(float(metadata["frame_gain_error_rel"])),
            "detector_offset_abs":
                abs(float(metadata["reflectance_offset_abs"])),
            "total_wavelength_error_max_abs_nm":
                abs(float(metadata["total_wavelength_error_max_abs_nm"])),
        })
    raw = pd.DataFrame(rows)
    metric_columns = [
        column for column in raw.columns
        if column not in {"noise_case", "noise_factor", "noise_level"}
    ]
    range_rows = []
    for keys, group in raw.groupby(
        ["noise_case", "noise_factor", "noise_level"], sort=False
    ):
        row = {
            "noise_case": keys[0],
            "noise_factor": keys[1],
            "noise_level": keys[2],
            "realizations": int(len(group)),
        }
        for metric in metric_columns:
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        range_rows.append(row)
    ranges = pd.DataFrame(range_rows)
    ranges.to_csv(
        inversion_dir / "noise_realization_ranges.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    return ranges


def save_combination_plots(
    inversion_dir: Path,
    table: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[str]:
    paths = []
    metrics = [
        ("Air_mae_nm", "Cavity MAE (nm)"),
        ("film_mae_nm_mean", "Film MAE (nm)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, (column, title) in zip(axes, metrics):
        matrix = np.full(
            (len(COMBINATIONS), len(COMBINATION_LEVELS)),
            np.nan,
        )
        for row_index, combination in enumerate(COMBINATIONS):
            for column_index, level in enumerate(COMBINATION_LEVELS):
                match = summary[
                    (summary["noise_factor"] == combination)
                    & (summary["noise_level"] == level)
                ]
                if not match.empty:
                    matrix[row_index, column_index] = float(
                        match.iloc[0][column]
                    )
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(
            range(len(COMBINATION_LEVELS)),
            labels=COMBINATION_LEVELS,
        )
        ax.set_yticks(range(len(COMBINATIONS)), labels=list(COMBINATIONS))
        ax.set_title(title, fontsize=16)
        for row_index in range(len(COMBINATIONS)):
            for column_index in range(len(COMBINATION_LEVELS)):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.3g}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=11,
                    )
        fig.colorbar(image, ax=ax, shrink=0.8)
    heatmap_path = inversion_dir / "combination_accuracy_heatmaps.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    paths.append(str(heatmap_path.resolve()))

    successful = table[table["success"]].copy()
    cases = successful["noise_case"].drop_duplicates().tolist()
    values = [
        successful.loc[
            successful["noise_case"] == case,
            "cavity_error_nm",
        ].to_numpy()
        for case in cases
    ]
    fig, ax = plt.subplots(figsize=(19, 8), constrained_layout=True)
    ax.boxplot(values, labels=cases, showfliers=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Cavity error (nm)")
    ax.set_title("Combination-noise cavity error distributions")
    ax.grid(True, axis="y", alpha=0.3)
    boxplot_path = inversion_dir / "combination_cavity_error_boxplot.png"
    fig.savefig(boxplot_path, dpi=180)
    plt.close(fig)
    paths.append(str(boxplot_path.resolve()))
    return paths


def run_combination_inversion(
    npz_paths: list[Path],
    dataset_dir: Path,
    inversion_dir: Path,
    config: FitConfig,
) -> dict:
    inversion_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    all_attempts = []
    sampling_audits = []
    global_summaries = []
    failures = []
    tasks = [
        (index, str(path.resolve()), config)
        for index, path in enumerate(sorted(npz_paths))
    ]
    for outcome in map(process_fit_task, tasks):
        index = int(outcome["index"])
        npz_path = Path(outcome["npz_path"])
        if not outcome["ok"]:
            failures.append({
                "input": str(npz_path),
                "error": outcome["error"],
            })
            print(f"ERROR {npz_path}: {outcome['error']}", flush=True)
            continue
        measurement = outcome["measurement"]
        sampling = outcome["sampling_audit"]
        fit = outcome["fit"]
        truth, _noise_audit = load_evaluation_truth(npz_path)
        row = result_row(measurement, fit, truth)
        rows.append(row)
        all_attempts.extend(attempt_rows(measurement, fit, truth))
        sampling_audits.append({
            "input_npz": str(npz_path.resolve()),
            **sampling,
        })
        global_summaries.append({
            "input_npz": str(npz_path.resolve()),
            "noise_case": measurement["metadata"]["noise_case"],
            **fit.get("global_summary", {}),
        })
        if fit["success"]:
            print(
                f"[invert {index + 1}/{len(npz_paths)} "
                f"{measurement['metadata']['noise_case']}] "
                f"Air_error={row['cavity_error_nm']:.6g} nm, "
                f"film_MAE={row['film_mae_nm']:.6g} nm, "
                f"angle_error={row['error_Angle_deg']:.6g} deg, "
                f"runtime={row['fit_runtime_s']:.3f}s",
                flush=True,
            )
        else:
            print(
                f"[invert {index + 1}/{len(npz_paths)}] "
                f"NO CONVERGED LOCAL RESULT: {npz_path}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No inversion result was produced: {failures}")
    table.to_csv(
        inversion_dir / "fit_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    pd.DataFrame(all_attempts).to_csv(
        inversion_dir / "multistart_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    summary = summarize_results(table)
    summary.to_csv(
        inversion_dir / "case_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    successful = table[table["success"]].copy()
    if successful.empty:
        level_summary = pd.DataFrame(columns=[
            "noise_level",
            "runs",
            "cavity_mae_nm",
            "film_mae_nm",
            "angle_mae_deg",
            "boundary_hit_rate",
            "fit_runtime_mean_s",
        ])
    else:
        level_summary = successful.groupby("noise_level").agg(
            runs=("success", "size"),
            cavity_mae_nm=("cavity_abs_error_nm", "mean"),
            film_mae_nm=("film_mae_nm", "mean"),
            angle_mae_deg=("angle_abs_error_deg", "mean"),
            boundary_hit_rate=(
                "boundary_hit_count",
                lambda values: float((values > 0).mean()),
            ),
            fit_runtime_mean_s=("fit_runtime_s", "mean"),
        ).reset_index()
    level_summary.to_csv(
        inversion_dir / "level_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    pd.DataFrame(sampling_audits).to_csv(
        inversion_dir / "sampling_audit.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    with (inversion_dir / "global_search_summary.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in global_summaries:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    ranges = save_combination_noise_ranges(dataset_dir, inversion_dir)
    plot_paths = save_combination_plots(inversion_dir, table, summary)
    sampling_table = pd.DataFrame(sampling_audits)
    audit = {
        "version": VERSION,
        "input_count": len(npz_paths),
        "result_count": len(table),
        "successful_count": int(table["success"].sum()),
        "case_count": int(table["noise_case"].nunique()),
        "config": asdict(config),
        "parameters": PARAMS,
        "bounds": BOUNDS,
        "combinations": {
            name: sorted(factors) for name, factors in COMBINATIONS.items()
        },
        "dataset_backend": sorted(
            str(value) for value in table["optical_backend"].unique()
        ),
        "inversion_backend": "python_tmm",
        "stackrt_polarization": "Rp",
        "truth_usage_policy": (
            "Truth is loaded only after optimization, convergence filtering, "
            "and result ranking complete."
        ),
        "overall_boundary_hit_rate": float(
            (successful["boundary_hit_count"] > 0).mean()
        ),
        "max_realized_wavelength_error_nm": float(
            table["wavelength_error_max_abs_nm"].max()
        ),
        "wavelength_limit_passed": bool(
            table["wavelength_error_max_abs_nm"].max()
            <= WAVELENGTH_ERROR_HARD_MAX_NM
        ),
        "nyquist_passed": bool(
            sampling_table["global_step_nm"].max()
            <= sampling_table["nyquist_max_step_nm"].min()
        ),
        "failures": failures,
        "plots": plot_paths,
        "noise_range_rows": len(ranges),
    }
    (inversion_dir / "run_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_lines = [
        "# Combination-noise V8 simulation and inversion",
        "",
        "## Scope",
        "",
        f"- Combination cases: {len(COMBINATIONS) * len(COMBINATION_LEVELS)}.",
        f"- Realizations: {len(npz_paths)}.",
        f"- Successful inversions: "
        f"{int(table['success'].sum())}/{len(table)}.",
        f"- Maximum realized wavelength error: "
        f"{audit['max_realized_wavelength_error_nm']:.6g} nm.",
        f"- Overall boundary-hit rate: "
        f"{100.0 * audit['overall_boundary_hit_rate']:.3f}%.",
        "- Dataset backend: Lumerical StackRT through lumapi (Rp).",
        "- Inversion backend: independent Python TMM.",
        "- Fitted parameters: Air, HSQ, PSS, SOC, TiO2, reflector angle.",
        "- Single-factor and previous all-factor combined cases are excluded.",
        "",
        "## Output files",
        "",
        "- fit_results.csv",
        "- multistart_results.csv",
        "- case_summary.csv",
        "- level_summary.csv",
        "- noise_realization_ranges.csv",
        "- sampling_audit.csv",
        "- run_audit.json",
    ]
    (inversion_dir / "analysis_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8-sig",
    )
    if failures:
        raise RuntimeError(f"Inversion failures: {failures}")
    return audit


def parse_pipeline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-file lumapi StackRT combination-noise generation and "
            "six-parameter Python TMM inversion pipeline."
        )
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--show-fdtd",
        action="store_true",
        help="Show the reused FDTD session instead of running it hidden.",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--local-gtol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_pipeline_args()
    repeats = 1 if args.smoke else max(1, int(args.repeats))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = (
        f"{VERSION}_smoke_{timestamp}"
        if args.smoke
        else f"{VERSION}_{timestamp}"
    )
    root = (
        Path(args.output_root).resolve()
        if args.output_root
        else OUTPUT_ROOT / default_name
    )
    root.mkdir(parents=True, exist_ok=False)
    dataset_dir = root / "dataset"
    inversion_dir = root / "inversion"
    config = FitConfig(
        input_dir=str(dataset_dir.resolve()),
        global_popsize=int(args.global_popsize),
        global_maxiter=int(args.global_maxiter),
        multistarts=int(args.multistarts),
        max_nfev=int(args.max_nfev),
        local_gtol=float(args.local_gtol),
        workers=1,
    )
    with StackRTSolver(hide=not args.show_fdtd) as solver:
        npz_paths, _dataset_index = generate_combination_dataset(
            dataset_dir,
            repeats,
            int(args.seed),
            solver,
        )
    audit = run_combination_inversion(
        npz_paths,
        dataset_dir,
        inversion_dir,
        config,
    )
    summary = {
        "version": VERSION,
        "root": str(root.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "inversion_dir": str(inversion_dir.resolve()),
        "smoke": bool(args.smoke),
        "dataset_backend": "stackrt_lumapi",
        "inversion_backend": "python_tmm",
        "stackrt_polarization": "Rp",
        "lumerical_api_path": str(LUMERICAL_API_PATH),
        "repeats_per_case": repeats,
        "combinations": {
            name: sorted(factors) for name, factors in COMBINATIONS.items()
        },
        "audit": audit,
    }
    (root / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"OUTPUT_ROOT={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
