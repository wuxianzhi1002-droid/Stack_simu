from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

VERSION = "tmm_compare_L_P_MAP_v1"
C0_M_S = 299_792_458.0
INCIDENT_MEDIUM_N = 5.8284
MAX_WAVELENGTH_ERROR_NM = 0.2
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2", "Angle"]
FILM_PARAMS = ["HSQ", "PSS", "SOC", "TiO2"]
BOUNDS = {
    "Air": (95.0, 105.0), "HSQ": (20.0, 40.0), "PSS": (1.0, 20.0),
    "SOC": (30.0, 50.0), "TiO2": (30.0, 50.0), "Angle": (-0.1, 0.1),
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
    instances_per_case: int = 2
    theta_prior_deg: float = 0.0
    theta_prior_sigma_deg: float = 0.01

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


METHODS = ("L_theta", "P_fixed", "P_theta_MAP")
METHOD_LABELS = {
    "L_theta": "Joint L and theta",
    "P_fixed": "Direct P with fixed prior angle",
    "P_theta_MAP": "Joint P and theta with MAP prior",
}
BASE_FILM_BOUNDS = {name: BOUNDS[name] for name in FILM_PARAMS}


def air_phase_cosine(reflector_angle_deg: float) -> float:
    return math.cos(math.radians(reflector_to_air_angle_deg(reflector_angle_deg)))


def geometric_to_phase_um(air_um: float, reflector_angle_deg: float) -> float:
    return float(air_um) * air_phase_cosine(reflector_angle_deg)


def phase_to_geometric_um(phase_um: float, reflector_angle_deg: float) -> float:
    cosine = air_phase_cosine(reflector_angle_deg)
    if cosine <= 0.0:
        raise ValueError("Air-layer propagation cosine must be positive.")
    return float(phase_um) / cosine


def method_spec(method: str, config: FitConfig) -> dict:
    if method == "L_theta":
        names = ["L", *FILM_PARAMS, "Angle"]
        bounds = [BOUNDS["Air"], *[BASE_FILM_BOUNDS[name] for name in FILM_PARAMS], BOUNDS["Angle"]]
    elif method == "P_fixed":
        names = ["P", *FILM_PARAMS]
        p_values = [
            geometric_to_phase_um(BOUNDS["Air"][i], config.theta_prior_deg)
            for i in range(2)
        ]
        bounds = [(min(p_values), max(p_values)), *[BASE_FILM_BOUNDS[name] for name in FILM_PARAMS]]
    elif method == "P_theta_MAP":
        # The broad P bounds cover every L/angle combination allowed by the common physical bounds.
        angle_candidates = [BOUNDS["Angle"][0], BOUNDS["Angle"][1], 0.0]
        p_values = [
            geometric_to_phase_um(length, angle)
            for length in BOUNDS["Air"] for angle in angle_candidates
        ]
        names = ["P", *FILM_PARAMS, "Angle"]
        bounds = [(min(p_values), max(p_values)), *[BASE_FILM_BOUNDS[name] for name in FILM_PARAMS], BOUNDS["Angle"]]
    else:
        raise ValueError(f"Unknown method: {method}")
    return {"names": names, "bounds": bounds}


def method_values_to_physical(method: str, values: np.ndarray, config: FitConfig) -> dict:
    spec = method_spec(method, config)
    item = dict(zip(spec["names"], np.asarray(values, dtype=float)))
    if method == "L_theta":
        length_um = item["L"]
        angle_deg = item["Angle"]
        phase_um = geometric_to_phase_um(length_um, angle_deg)
        angle_status = "fitted"
    elif method == "P_fixed":
        phase_um = item["P"]
        angle_deg = float(config.theta_prior_deg)
        length_um = phase_to_geometric_um(phase_um, angle_deg)
        angle_status = "fixed_to_prior"
    else:
        phase_um = item["P"]
        angle_deg = item["Angle"]
        length_um = phase_to_geometric_um(phase_um, angle_deg)
        angle_status = "fitted_with_MAP_prior"
    physical = {
        "L_um": float(length_um),
        "P_um": float(phase_um),
        "Angle_deg": float(angle_deg),
        "angle_status": angle_status,
    }
    physical.update({name: float(item[name]) for name in FILM_PARAMS})
    return physical


def physical_model_vector(physical: dict) -> np.ndarray:
    return np.asarray([
        physical["L_um"],
        physical["HSQ"], physical["PSS"], physical["SOC"], physical["TiO2"],
        physical["Angle_deg"],
    ], dtype=float)


def method_bounds_arrays(method: str, config: FitConfig) -> tuple[np.ndarray, np.ndarray]:
    bounds = method_spec(method, config)["bounds"]
    return (
        np.asarray([item[0] for item in bounds], dtype=float),
        np.asarray([item[1] for item in bounds], dtype=float),
    )


def latin_hypercube_for_method(method: str, config: FitConfig, seed: int, size: int) -> np.ndarray:
    lower, upper = method_bounds_arrays(method, config)
    sample = qmc.LatinHypercube(d=len(lower), seed=seed).random(n=size)
    return qmc.scale(sample, lower, upper)


def select_diverse_method_candidates(
    population: np.ndarray,
    energies: np.ndarray,
    count: int,
    method: str,
    config: FitConfig,
) -> list[tuple[np.ndarray, int, float]]:
    lower, upper = method_bounds_arrays(method, config)
    span = upper - lower
    selected = []
    for index in np.argsort(energies):
        candidate = np.asarray(population[index], dtype=float)
        scaled = (candidate - lower) / span
        if all(np.linalg.norm(scaled - previous[0]) >= 0.03 for previous in selected):
            selected.append((scaled, int(index), float(energies[index])))
        if len(selected) >= count:
            break
    return [(lower + scaled * span, index, energy) for scaled, index, energy in selected]


def fit_method(measurement: dict, method: str, config: FitConfig, seed: int) -> dict:
    wavelengths = measurement["wavelengths_um"]
    observed = measurement["spectrum"]
    scale = robust_scale(observed)
    global_indices = np.arange(0, len(wavelengths), max(1, int(config.global_stride)))
    global_wavelengths = wavelengths[global_indices]
    global_observed = observed[global_indices]
    spec = method_spec(method, config)
    lower, upper = method_bounds_arrays(method, config)
    span = upper - lower

    def model(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        axis = global_wavelengths if use_global else wavelengths
        physical = method_values_to_physical(method, values, config)
        return tmm_reflectance(axis, physical_model_vector(physical))

    def spectral_residual(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        target = global_observed if use_global else observed
        return (model(values, use_global) - target) / scale

    def residual(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        spectral = spectral_residual(values, use_global)
        if method != "P_theta_MAP":
            return spectral
        physical = method_values_to_physical(method, values, config)
        prior = (physical["Angle_deg"] - config.theta_prior_deg) / config.theta_prior_sigma_deg
        # P and theta have a coupled feasible region because L=P/cos(theta_air).
        # These zero-inside residuals enforce the same physical L bounds as L_theta.
        lower_violation = max(0.0, BOUNDS["Air"][0] - physical["L_um"])
        upper_violation = max(0.0, physical["L_um"] - BOUNDS["Air"][1])
        bound_scale_um = 1.0e-9
        return np.concatenate([
            spectral,
            np.asarray([
                prior,
                lower_violation / bound_scale_um,
                upper_violation / bound_scale_um,
            ], dtype=float),
        ])

    started = time.perf_counter()
    population_size = max(5, int(config.global_popsize) * len(lower))
    initial_population = latin_hypercube_for_method(method, config, seed, population_size)
    initial_energies = np.asarray([
        robust_objective(residual(values, True), config.loss)
        for values in initial_population
    ])
    global_result = differential_evolution(
        lambda values: robust_objective(residual(values, True), config.loss),
        bounds=spec["bounds"],
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
    candidates = select_diverse_method_candidates(
        evolved_population, evolved_energies, int(config.multistarts), method, config
    )
    if not candidates:
        raise RuntimeError("Global search returned no candidate starts.")

    angle_index = spec["names"].index("Angle") if "Angle" in spec["names"] else None

    def physical_to_unit(values: np.ndarray) -> np.ndarray:
        unit_values = np.clip((np.asarray(values, dtype=float) - lower) / span, 0.0, 1.0)
        if angle_index is not None:
            unit_values[angle_index] = unit_values[angle_index] ** 2
        return unit_values

    def unit_to_physical(unit_values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(unit_values, dtype=float).copy()
        if angle_index is not None:
            transformed[angle_index] = np.sqrt(max(0.0, transformed[angle_index]))
        return lower + transformed * span

    attempts = []
    for start_rank, (x0, population_index, global_energy) in enumerate(candidates, start=1):
        local = least_squares(
            lambda unit_values: residual(unit_to_physical(unit_values), False),
            x0=physical_to_unit(x0),
            bounds=(np.zeros(len(lower)), np.ones(len(lower))),
            loss=config.loss,
            max_nfev=int(config.max_nfev),
            x_scale=1.0,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=float(config.local_gtol),
        )
        fitted_values = unit_to_physical(local.x)
        fitted_spectrum = model(fitted_values, False)
        physical = method_values_to_physical(method, fitted_values, config)
        attempts.append({
            "start_rank": start_rank,
            "success": bool(local.success) and np.isfinite(local.cost),
            "message": str(local.message),
            "status": int(local.status),
            "optimality": float(local.optimality),
            "cost": float(local.cost),
            "rmse_reflectance": float(np.sqrt(np.mean((fitted_spectrum - observed) ** 2))),
            "nfev": int(local.nfev),
            "x0": np.asarray(x0, dtype=float),
            "x": fitted_values,
            "physical": physical,
            "global_population_index": int(population_index),
            "global_energy": float(global_energy),
        })
    attempts.sort(key=lambda row: (not row["success"], row["cost"]))
    successful = [row for row in attempts if row["success"]]
    runtime_s = time.perf_counter() - started
    if not successful:
        return {"success": False, "runtime_s": runtime_s, "attempts": attempts}
    best = successful[0]
    return {
        "success": True,
        "runtime_s": runtime_s,
        "method": method,
        "x": np.asarray(best["x"], dtype=float),
        "physical": best["physical"],
        "cost": float(best["cost"]),
        "rmse_reflectance": float(best["rmse_reflectance"]),
        "nfev": int(best["nfev"]),
        "attempts": attempts,
        "selected_start_rank": int(best["start_rank"]),
        "fitted_spectrum": model(best["x"], False),
        "global_summary": {
            "initial_population_source": "LatinHypercube; no truth or forced center point",
            "initial_population_size": int(len(initial_population)),
            "initial_best_energy": float(np.min(initial_energies)),
            "evolved_best_energy": float(np.min(evolved_energies)),
            "global_success": bool(global_result.success),
            "global_message": str(global_result.message),
            "global_nit": int(global_result.nit),
            "global_nfev": int(global_result.nfev),
        },
    }


def select_inputs(input_dir: Path, pattern: str, instances_per_case: int) -> tuple[list[Path], pd.DataFrame]:
    groups: dict[tuple[str, str], list[Path]] = {}
    metadata_rows = []
    for path in sorted(input_dir.glob(pattern), key=lambda item: item.name):
        with np.load(path, allow_pickle=False) as data:
            factor = str(scalar(data, "noise_factor", "unknown"))
            level = str(scalar(data, "noise_level", "unknown"))
            case = str(scalar(data, "noise_case", path.stem))
            realization = int(scalar(data, "realization_index", 0))
            seed = int(scalar(data, "random_seed", 0))
        if factor == "clean" or level == "clean":
            continue
        groups.setdefault((factor, level), []).append(path)
        metadata_rows.append({
            "input_npz": str(path.resolve()), "filename": path.name,
            "noise_case": case, "noise_factor": factor, "noise_level": level,
            "realization_index": realization, "random_seed": seed,
        })
    selected = []
    selected_rows = []
    metadata = {row["input_npz"]: row for row in metadata_rows}
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda item: item.name)
        if len(ordered) < instances_per_case:
            raise ValueError(f"Case {key} has only {len(ordered)} files.")
        for order, path in enumerate(ordered[:instances_per_case], start=1):
            selected.append(path)
            row = dict(metadata[str(path.resolve())])
            row["selection_order_within_case"] = order
            row["selection_rule"] = "filename ascending, first N; no fit-dependent selection"
            selected_rows.append(row)
    return selected, pd.DataFrame(selected_rows)


def truth_metrics(npz_path: Path) -> tuple[dict, dict]:
    truth, audit = load_evaluation_truth(npz_path)
    truth["P"] = geometric_to_phase_um(truth["Air"], truth["Angle"])
    truth["AirAngle"] = reflector_to_air_angle_deg(truth["Angle"])
    return truth, audit


def multistart_dispersion(attempts: list[dict]) -> dict:
    successful = [item for item in attempts if item["success"]]
    if not successful:
        return {
            "multistart_success_count": 0,
            "multistart_L_std_nm": np.nan, "multistart_L_range_nm": np.nan,
            "multistart_P_std_nm": np.nan, "multistart_P_range_nm": np.nan,
            "multistart_angle_std_deg": np.nan, "multistart_angle_range_deg": np.nan,
        }
    lengths = np.asarray([item["physical"]["L_um"] for item in successful]) * 1000.0
    phases = np.asarray([item["physical"]["P_um"] for item in successful]) * 1000.0
    angles = np.asarray([item["physical"]["Angle_deg"] for item in successful])
    fixed_angle = all(item["physical"]["angle_status"] == "fixed_to_prior" for item in successful)
    return {
        "multistart_success_count": len(successful),
        "multistart_L_std_nm": float(np.std(lengths)),
        "multistart_L_range_nm": float(np.ptp(lengths)),
        "multistart_P_std_nm": float(np.std(phases)),
        "multistart_P_range_nm": float(np.ptp(phases)),
        "multistart_angle_std_deg": np.nan if fixed_angle else float(np.std(angles)),
        "multistart_angle_range_deg": np.nan if fixed_angle else float(np.ptp(angles)),
    }


def result_row(measurement: dict, fit: dict, method: str, truth: dict, config: FitConfig) -> dict:
    metadata = measurement["metadata"]
    attempt_list = fit.get("attempts", [])
    converged_attempts = [item for item in attempt_list if item["success"]]
    row = {
        "input_npz": measurement["path"],
        "filename": Path(measurement["path"]).name,
        "method": method,
        "method_label": METHOD_LABELS[method],
        **metadata,
        "theta_prior_deg": config.theta_prior_deg,
        "theta_prior_sigma_deg": config.theta_prior_sigma_deg,
        "success": bool(fit["success"]),
        "convergence_status": "converged" if fit["success"] else "no_converged_local_result",
        "fit_runtime_s": float(fit["runtime_s"]),
        "multistart_attempt_count": len(attempt_list),
        "converged_multistart_count": len(converged_attempts),
        "total_local_nfev": int(sum(item["nfev"] for item in attempt_list)),
    }
    if not fit["success"]:
        row["nfev"] = np.nan
        return row
    physical = fit["physical"]
    row.update({
        "fit_L_um": physical["L_um"],
        "truth_L_um": truth["Air"],
        "L_error_nm": (physical["L_um"] - truth["Air"]) * 1000.0,
        "L_abs_error_nm": abs(physical["L_um"] - truth["Air"]) * 1000.0,
        "fit_P_um": physical["P_um"],
        "truth_P_um": truth["P"],
        "P_error_nm": (physical["P_um"] - truth["P"]) * 1000.0,
        "P_abs_error_nm": abs(physical["P_um"] - truth["P"]) * 1000.0,
        "fit_reflector_angle_deg": np.nan if method == "P_fixed" else physical["Angle_deg"],
        "model_reflector_angle_deg": physical["Angle_deg"],
        "truth_reflector_angle_deg": truth["Angle"],
        "angle_error_deg": (
            np.nan if method == "P_fixed"
            else physical["Angle_deg"] - truth["Angle"]
        ),
        "angle_abs_error_deg": (
            np.nan if method == "P_fixed"
            else abs(physical["Angle_deg"] - truth["Angle"])
        ),
        "angle_output_status": physical["angle_status"],
        "fixed_prior_angle_deg": config.theta_prior_deg if method == "P_fixed" else np.nan,
        "fit_air_angle_deg": (
            np.nan if method == "P_fixed"
            else reflector_to_air_angle_deg(physical["Angle_deg"])
        ),
        "model_air_angle_deg": reflector_to_air_angle_deg(physical["Angle_deg"]),
        "truth_air_angle_deg": truth["AirAngle"],
        "rmse_reflectance": fit["rmse_reflectance"],
        "cost": fit["cost"],
        "nfev": fit["nfev"],
        "selected_start_rank": fit["selected_start_rank"],
    })
    for name in FILM_PARAMS:
        row[f"fit_{name}_nm"] = physical[name]
        row[f"truth_{name}_nm"] = truth[name]
        row[f"error_{name}_nm"] = physical[name] - truth[name]
    row["film_mae_nm"] = float(np.mean([abs(row[f"error_{name}_nm"]) for name in FILM_PARAMS]))
    row.update(multistart_dispersion(fit["attempts"]))
    return row


def attempt_rows(measurement: dict, fit: dict, method: str, truth: dict, config: FitConfig) -> list[dict]:
    rows = []
    spec = method_spec(method, config)
    for rank, attempt in enumerate(fit.get("attempts", []), start=1):
        physical = attempt["physical"]
        row = {
            "input_npz": measurement["path"],
            "filename": Path(measurement["path"]).name,
            "noise_case": measurement["metadata"]["noise_case"],
            "method": method,
            "rank_after_convergence_filter": rank,
            "start_rank": attempt["start_rank"],
            "success": attempt["success"],
            "status": attempt["status"],
            "message": attempt["message"],
            "cost": attempt["cost"],
            "rmse_reflectance": attempt["rmse_reflectance"],
            "nfev": attempt["nfev"],
            "fit_L_um": physical["L_um"],
            "fit_P_um": physical["P_um"],
            "fit_reflector_angle_deg": physical["Angle_deg"],
            "angle_output_status": physical["angle_status"],
            "benchmark_L_error_nm": (physical["L_um"] - truth["Air"]) * 1000.0,
            "benchmark_P_error_nm": (physical["P_um"] - truth["P"]) * 1000.0,
            "benchmark_angle_error_deg": (
                np.nan if method == "P_fixed" else physical["Angle_deg"] - truth["Angle"]
            ),
            "global_population_index": attempt["global_population_index"],
            "global_energy": attempt["global_energy"],
        }
        for index, name in enumerate(spec["names"]):
            row[f"x0_{name}"] = float(attempt["x0"][index])
            row[f"fit_parameter_{name}"] = float(attempt["x"][index])
        rows.append(row)
    return rows


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    successful = table[table["success"]].copy()
    method_rows = []
    for method in METHODS:
        group = successful[successful["method"] == method]
        all_group = table[table["method"] == method]
        row = {
            "method": method,
            "runs": len(all_group),
            "successful_runs": len(group),
            "success_rate": len(group) / max(1, len(all_group)),
        }
        if len(group):
            row.update({
                "L_bias_nm": float(group["L_error_nm"].mean()),
                "L_mae_nm": float(group["L_abs_error_nm"].mean()),
                "L_p95_abs_nm": float(group["L_abs_error_nm"].quantile(0.95)),
                "P_bias_nm": float(group["P_error_nm"].mean()),
                "P_mae_nm": float(group["P_abs_error_nm"].mean()),
                "P_p95_abs_nm": float(group["P_abs_error_nm"].quantile(0.95)),
                "angle_mae_deg": float(group["angle_abs_error_deg"].mean()) if method != "P_fixed" else np.nan,
                "rmse_reflectance_mean": float(group["rmse_reflectance"].mean()),
                "runtime_mean_s": float(group["fit_runtime_s"].mean()),
                "runtime_median_s": float(group["fit_runtime_s"].median()),
                "nfev_mean": float(group["nfev"].mean()),
                "multistart_L_range_nm_mean": float(group["multistart_L_range_nm"].mean()),
                "multistart_P_range_nm_mean": float(group["multistart_P_range_nm"].mean()),
                "multistart_angle_range_deg_mean": float(group["multistart_angle_range_deg"].mean()),
            })
        method_rows.append(row)
    case_rows = []
    for (case, method), group in successful.groupby(["noise_case", "method"], sort=True):
        case_rows.append({
            "noise_case": case,
            "noise_factor": group.iloc[0]["noise_factor"],
            "noise_level": group.iloc[0]["noise_level"],
            "method": method,
            "runs": len(group),
            "L_mae_nm": float(group["L_abs_error_nm"].mean()),
            "P_mae_nm": float(group["P_abs_error_nm"].mean()),
            "angle_mae_deg": float(group["angle_abs_error_deg"].mean()) if method != "P_fixed" else np.nan,
            "rmse_reflectance_mean": float(group["rmse_reflectance"].mean()),
            "runtime_mean_s": float(group["fit_runtime_s"].mean()),
            "multistart_L_range_nm_mean": float(group["multistart_L_range_nm"].mean()),
            "multistart_P_range_nm_mean": float(group["multistart_P_range_nm"].mean()),
            "multistart_angle_range_deg_mean": float(group["multistart_angle_range_deg"].mean()),
        })
    return pd.DataFrame(method_rows), pd.DataFrame(case_rows)


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
            )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, table: pd.DataFrame, method_summary: pd.DataFrame, config: FitConfig) -> None:
    success = table[table["success"]].copy()
    indexed = method_summary.set_index("method")
    lt = indexed.loc["L_theta"]
    pf = indexed.loc["P_fixed"]
    pm = indexed.loc["P_theta_MAP"]
    true_zero = success[np.isclose(success["truth_reflector_angle_deg"], 0.0)]
    lt_zero = true_zero[true_zero["method"] == "L_theta"]
    map_zero = true_zero[true_zero["method"] == "P_theta_MAP"]
    lines = [
        "# L, P and MAP parameterization comparison", "",
        "## Protocol", "",
        f"- Input dataset: `{config.input_dir}`.",
        f"- The first {config.instances_per_case} files in filename order are selected per noise factor and level.",
        f"- Unique NPZ files: {table['input_npz'].nunique()}; total fits: {len(table)}.",
        "- All methods share the same TMM, materials, film parameters, 450-580 nm band, preprocessing, optimizer settings, and physical parameter bounds.",
        f"- Nominal/prior reflector angle: {config.theta_prior_deg:.6g} deg; MAP sigma: {config.theta_prior_sigma_deg:.6g} deg.",
        "- Solver angle is defined in RefReflector; air-cavity angle is derived by Snell law.",
        "- Truth is loaded only after fitting, convergence filtering, and result ranking.",
        "", "## Method summary", "", dataframe_to_markdown(method_summary), "",
        "## Answers", "",
        "### 1. Is direct P fitting more stable and faster?", "",
        f"P-fixed mean runtime is {pf['runtime_mean_s']:.4g} s versus {lt['runtime_mean_s']:.4g} s for L-theta. "
        f"Their mean multistart P ranges are {pf['multistart_P_range_nm_mean']:.4g} nm and {lt['multistart_P_range_nm_mean']:.4g} nm.",
        "The speed difference mainly reflects removal of one fitted angle parameter. Stability must be judged using both multistart dispersion and estimation error.",
        "", "### 2. Accuracy of L and P", "",
        f"L-theta: L MAE={lt['L_mae_nm']:.4g} nm, P MAE={lt['P_mae_nm']:.4g} nm.",
        f"P-fixed: L MAE={pf['L_mae_nm']:.4g} nm, P MAE={pf['P_mae_nm']:.4g} nm.",
        f"P-theta-MAP: L MAE={pm['L_mae_nm']:.4g} nm, P MAE={pm['P_mae_nm']:.4g} nm.",
        "", "### 3. Does the MAP prior suppress Air-Angle non-identifiability?", "",
        f"L-theta angle MAE={lt['angle_mae_deg']:.4g} deg and mean multistart angle range={lt['multistart_angle_range_deg_mean']:.4g} deg; "
        f"MAP gives {pm['angle_mae_deg']:.4g} deg and {pm['multistart_angle_range_deg_mean']:.4g} deg.",
        "A lower MAP spread indicates regularization of the degeneracy. Bias on truly nonzero-angle cases indicates prior shrinkage toward the nominal angle.",
        "", "### 4. Spurious nonzero angle when truth is 0 deg", "",
        f"Across {lt_zero['input_npz'].nunique()} zero-angle files, L-theta has mean absolute fitted angle "
        f"{lt_zero['fit_reflector_angle_deg'].abs().mean():.4g} deg and maximum {lt_zero['fit_reflector_angle_deg'].abs().max():.4g} deg; "
        f"MAP gives mean {map_zero['fit_reflector_angle_deg'].abs().mean():.4g} deg and maximum {map_zero['fit_reflector_angle_deg'].abs().max():.4g} deg.",
        "", "### 5. Does P add physical information?", "",
        "No. P parameterization adds no observation or physical information. It exposes the phase-sensitive combination explicitly and can improve numerical conditioning. "
        "Independent angle information or a defensible prior is still required to convert P into a reliable geometric L.",
        "", "## Limitations", "",
        "- Two predetermined files per case support a controlled method comparison, not a full Monte Carlo confidence interval.",
        "- P-fixed reports angle status as fixed_to_prior; fitted-angle and angle-error fields are intentionally blank.",
        "- StackRT generated the data and Python TMM performed inversion, so results include cross-model mismatch.",
    ]
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare L-theta, P-fixed, and P-theta MAP inversion.")
    parser.add_argument(
        "--input-dir", type=Path,
        default=REPO_ROOT / "work" / "04_results_and_datasets" / "static_stackrt_v8_20260811_142049",
    )
    parser.add_argument("--pattern", default="static_spectrum_*.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instances-per-case", type=int, default=2)
    parser.add_argument("--theta-prior-deg", type=float, default=0.0)
    parser.add_argument("--theta-prior-sigma-deg", type=float, default=0.01)
    parser.add_argument("--wavelength-min-nm", type=float, default=450.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--global-stride", type=int, default=4)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--local-gtol", type=float, default=1.0e-5)
    parser.add_argument("--random-seed", type=int, default=20260810)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--smoke", action="store_true", help="Fit the first selected NPZ with all three methods.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.theta_prior_sigma_deg <= 0.0:
        raise ValueError("theta_prior_sigma_deg must be positive.")
    input_dir = args.input_dir.resolve()
    config = FitConfig(
        input_dir=str(input_dir),
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        stride=args.stride,
        global_stride=args.global_stride,
        global_popsize=args.global_popsize,
        global_maxiter=args.global_maxiter,
        multistarts=args.multistarts,
        max_nfev=args.max_nfev,
        local_gtol=args.local_gtol,
        workers=1,
        random_seed=args.random_seed,
        loss=args.loss,
        instances_per_case=args.instances_per_case,
        theta_prior_deg=args.theta_prior_deg,
        theta_prior_sigma_deg=args.theta_prior_sigma_deg,
    )
    selected, selection = select_inputs(input_dir, args.pattern, config.instances_per_case)
    if args.smoke:
        selected = selected[:1]
        selection = selection[selection["input_npz"].isin([str(selected[0].resolve())])].copy()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else OUTPUT_ROOT / f"{VERSION}{suffix}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    selection.to_csv(output_dir / "selected_inputs.csv", index=False, encoding="utf-8-sig")

    rows = []
    attempts = []
    failures = []
    sampling_rows = []
    total = len(selected) * len(METHODS)
    counter = 0
    for input_index, npz_path in enumerate(selected):
        measurement = load_fit_input(npz_path, config)
        sampling = validate_sampling(measurement, config)
        sampling_rows.append({"input_npz": str(npz_path.resolve()), **sampling})
        for method_index, method in enumerate(METHODS):
            counter += 1
            seed = config.random_seed + input_index * 1009 + method_index * 1_000_003
            try:
                fit = fit_method(measurement, method, config, seed)
                # Evaluation-only truth is intentionally loaded after fit and ranking.
                truth, _audit = truth_metrics(npz_path)
                row = result_row(measurement, fit, method, truth, config)
                rows.append(row)
                attempts.extend(attempt_rows(measurement, fit, method, truth, config))
                print(
                    f"[{counter}/{total} {measurement['metadata']['noise_case']} {method}] "
                    f"success={fit['success']} L_error={row.get('L_error_nm', np.nan):.6g} nm "
                    f"P_error={row.get('P_error_nm', np.nan):.6g} nm "
                    f"runtime={fit['runtime_s']:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append({
                    "input_npz": str(npz_path.resolve()), "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"ERROR [{counter}/{total} {method}] {npz_path}: {exc}", flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No comparison results were produced: {failures}")
    table.to_csv(output_dir / "comparison_results.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(attempts).to_csv(output_dir / "multistart_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(sampling_rows).drop_duplicates("input_npz").to_csv(
        output_dir / "sampling_audit.csv", index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    method_summary, case_summary = summarize(table)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    case_summary.to_csv(output_dir / "case_method_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    audit = {
        "version": VERSION,
        "input_dir": str(input_dir),
        "selected_unique_npz": int(table["input_npz"].nunique()),
        "fit_count": len(table),
        "successful_fit_count": int(table["success"].sum()),
        "methods": METHODS,
        "selection_rule": "Group by noise_factor and noise_level; filename ascending; first N.",
        "config": asdict(config),
        "truth_usage_policy": "Truth loaded only after fit, convergence filtering, and best-result ranking.",
        "P_definition": "P=L*cos(theta_air); L=P/cos(theta_air) before TMM evaluation.",
        "angle_definition": "Reflector incident-medium solver angle; air angle derived by Snell law.",
        "failures": failures,
    }
    (output_dir / "comparison_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir, table, method_summary, config)
    print(f"OUTPUT_DIR={output_dir.resolve()}", flush=True)
    if failures:
        raise RuntimeError(f"Comparison failures: {failures}")


if __name__ == "__main__":
    main()
