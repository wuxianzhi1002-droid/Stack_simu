"""V12 fixed-angle five-parameter strict TMM inversion.

The fitted parameters are ``[Air, HSQ, PSS, SOC, TiO2]``.  Angle is read from
the independent experiment-visible measurement stored by ``main_v12.py`` and
is fixed in every global and local forward evaluation.

V12 also adds a soft boundary-proximity term to the global objective and a
cost-equivalence-aware local candidate ranking.  Pure spectrum cost, boundary
penalty, proximity flags, and hard boundary hits are all retained separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import tmm_joint_inversion_v10 as v10  # noqa: E402


VERSION = "tmm_joint_inversion_v12"
FREE_PARAMS = ("Air", "HSQ", "PSS", "SOC", "TiO2")
FULL_PARAMS = FREE_PARAMS + ("Angle",)
EXPECTED_GENERATOR = "main_v12"
EXPECTED_ANGLE_MODE = "fixed_independent_measurement"
BOUNDARY_MARGIN_FRACTION = 0.02
BOUNDARY_PENALTY_WEIGHT = 0.05
BOUNDARY_HIT_FRACTION = 1.0e-5
SPECTRUM_EQUIVALENCE_RTOL = 1.0e-3
SPECTRUM_EQUIVALENCE_ATOL = 1.0e-12


@dataclass
class FitConfig:
    input_dir: str
    wavelength_min_nm: float = 220.0
    wavelength_max_nm: float = 580.0
    stride: int = 1
    global_stride: int = 1
    global_forward_model: str = "full_ils"
    global_popsize: int = 8
    global_maxiter: int = 40
    multistarts: int = 8
    max_nfev: int = 600
    local_gtol: float = 1.0e-5
    workers: int = 1
    random_seed: int = 20260831
    loss: str = "soft_l1"
    boundary_margin_fraction: float = BOUNDARY_MARGIN_FRACTION
    boundary_penalty_weight: float = BOUNDARY_PENALTY_WEIGHT
    spectrum_equivalence_rtol: float = SPECTRUM_EQUIVALENCE_RTOL


def scalar(npz, key: str, default):
    return np.asarray(npz[key]).item() if key in npz else default


def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([v10.BOUNDS[name][0] for name in FREE_PARAMS], dtype=np.float64),
        np.asarray([v10.BOUNDS[name][1] for name in FREE_PARAMS], dtype=np.float64),
    )


def full_parameters(free_values: np.ndarray, measured_angle_deg: float) -> np.ndarray:
    free = np.asarray(free_values, dtype=np.float64)
    if free.shape != (5,) or not np.all(np.isfinite(free)):
        raise ValueError("five finite free parameters are required")
    angle = float(measured_angle_deg)
    if not np.isfinite(angle):
        raise ValueError("measured angle must be finite")
    return np.concatenate((free, np.asarray([angle], dtype=np.float64)))


def boundary_metrics(
    free_values: np.ndarray,
    margin_fraction: float = BOUNDARY_MARGIN_FRACTION,
) -> dict[str, Any]:
    lower, upper = bounds_arrays()
    values = np.asarray(free_values, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("boundary metrics require five free parameters")
    normalized = (values - lower) / (upper - lower)
    distance = np.minimum(normalized, 1.0 - normalized)
    margin = float(margin_fraction)
    if not 0.0 < margin < 0.5:
        raise ValueError("boundary margin fraction must lie in (0, 0.5)")
    severity_by_param = np.square(np.clip((margin - distance) / margin, 0.0, 1.0))
    hit_mask = distance <= BOUNDARY_HIT_FRACTION
    near_mask = distance < margin
    return {
        "normalized_distance": distance,
        "severity_by_parameter": severity_by_param,
        "severity": float(np.sum(severity_by_param)),
        "boundary_hits": [FREE_PARAMS[i] for i in np.where(hit_mask)[0]],
        "near_boundary": [FREE_PARAMS[i] for i in np.where(near_mask)[0]],
    }


def penalized_objective(
    spectrum_objective: float,
    free_values: np.ndarray,
    margin_fraction: float = BOUNDARY_MARGIN_FRACTION,
    penalty_weight: float = BOUNDARY_PENALTY_WEIGHT,
) -> tuple[float, float, dict[str, Any]]:
    spectrum = float(spectrum_objective)
    if not np.isfinite(spectrum) or spectrum < 0.0:
        raise ValueError("spectrum objective must be finite and nonnegative")
    metrics = boundary_metrics(free_values, margin_fraction)
    penalty = max(spectrum, 1.0e-12) * float(penalty_weight) * metrics["severity"]
    return spectrum + penalty, penalty, metrics


def load_fit_input(npz_path: Path, config: FitConfig) -> dict[str, Any]:
    """Load only experiment-visible arrays/configuration and measured angle."""
    with np.load(npz_path, allow_pickle=False) as data:
        generator_version = str(scalar(data, "generator_version", "unknown"))
        if generator_version != EXPECTED_GENERATOR:
            raise ValueError(
                f"V12 inversion requires {EXPECTED_GENERATOR}, got {generator_version!r}"
            )
        required = {
            "wavelengths",
            "reported_wavelengths_nm",
            "spectrum_measured",
            "config_json",
            "reported_axis_is_fixed",
            "internal_wavelength_margin_nm",
            "measured_reflector_angle_deg",
            "angle_measurement_sigma_deg",
            "angle_measurement_mode",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"V12 NPZ missing required fields: {missing}")
        wavelengths_um = np.asarray(data["wavelengths"], dtype=np.float64)
        wavelengths_nm = np.asarray(data["reported_wavelengths_nm"], dtype=np.float64)
        spectrum = np.asarray(data["spectrum_measured"], dtype=np.float64)
        if wavelengths_um.ndim != 1 or wavelengths_nm.shape != wavelengths_um.shape:
            raise ValueError("reported wavelength arrays must be matching 1-D arrays")
        if spectrum.shape != wavelengths_um.shape:
            raise ValueError("measured spectrum must match the reported wavelength axis")
        if not np.all(np.isfinite(wavelengths_um)) or not np.all(np.isfinite(spectrum)):
            raise ValueError("reported wavelength and spectrum arrays must be finite")
        if np.any(np.diff(wavelengths_um) <= 0.0):
            raise ValueError("reported wavelength axis must be strictly increasing")
        if not np.allclose(wavelengths_um * 1000.0, wavelengths_nm, atol=1.0e-10):
            raise ValueError("wavelengths and reported_wavelengths_nm disagree")
        if not bool(scalar(data, "reported_axis_is_fixed", False)):
            raise ValueError("V12 requires a fixed reported wavelength axis")
        mode = str(scalar(data, "angle_measurement_mode", "unknown"))
        if mode != EXPECTED_ANGLE_MODE:
            raise ValueError(f"unexpected angle measurement mode: {mode!r}")
        measured_angle = float(scalar(data, "measured_reflector_angle_deg", np.nan))
        sigma_angle = float(scalar(data, "angle_measurement_sigma_deg", np.nan))
        if not np.isfinite(measured_angle) or not np.isfinite(sigma_angle) or sigma_angle <= 0:
            raise ValueError("invalid independent angle measurement fields")
        generator_config = json.loads(str(scalar(data, "config_json", "{}")))
        mask = (
            (wavelengths_nm >= float(config.wavelength_min_nm))
            & (wavelengths_nm <= float(config.wavelength_max_nm))
        )
        indices = np.where(mask)[0][:: max(1, int(config.stride))]
        if len(indices) < 50:
            raise ValueError("too few samples after wavelength mask and stride")
        selected_um = wavelengths_um[indices]
        metadata = {
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "noise_factor": str(scalar(data, "noise_factor", "unknown")),
            "noise_level": str(scalar(data, "noise_level", "unknown")),
            "realization_index": int(scalar(data, "realization_index", 0)),
            "random_seed": int(scalar(data, "random_seed", 0)),
            "generator_version": generator_version,
            "angle_measurement_mode": mode,
            "measured_angle_deg": measured_angle,
            "angle_measurement_sigma_deg": sigma_angle,
            "angle_fixed_in_inversion": True,
            "true_angle_used_for_fit": False,
            "internal_wavelength_margin_nm": float(
                scalar(data, "internal_wavelength_margin_nm", np.nan)
            ),
        }
    return {
        "path": str(npz_path.resolve()),
        "wavelengths_um": selected_um,
        "spectrum": spectrum[indices],
        "actual_step_nm": float(np.median(np.diff(selected_um))) * 1000.0,
        "generator_config": generator_config,
        "metadata": metadata,
        "fixed_angle_deg": measured_angle,
    }


def load_evaluation_truth(npz_path: Path) -> tuple[dict[str, float], dict]:
    """Evaluation-only truth loader; call after fitting and candidate selection."""
    with np.load(npz_path, allow_pickle=False) as data:
        names = [str(value) for value in np.asarray(data["layer_names"])]
        layers = dict(zip(names, np.asarray(data["layer_thickness_um"], dtype=float)))
        truth = {
            "Air": float(layers["Air"]),
            "HSQ": float(layers["HSQ"]) * 1000.0,
            "PSS": float(layers["PSS"]) * 1000.0,
            "SOC": float(layers["SOC"]) * 1000.0,
            "TiO2": float(layers["TiO2"]) * 1000.0,
            "Angle": float(scalar(data, "true_reflector_angle_deg", 0.0)),
        }
        audit = json.loads(str(scalar(data, "noise_realization_json", "{}")))
    return truth, audit


def latin_hypercube_population(seed: int, size: int) -> np.ndarray:
    lower, upper = bounds_arrays()
    sample = qmc.LatinHypercube(d=len(FREE_PARAMS), seed=seed).random(n=int(size))
    return qmc.scale(sample, lower, upper)


def select_diverse_candidates(
    population: np.ndarray,
    energies: np.ndarray,
    count: int,
) -> list[tuple[np.ndarray, int, float]]:
    lower, upper = bounds_arrays()
    span = upper - lower
    selected: list[tuple[np.ndarray, int, float]] = []
    scaled_selected: list[np.ndarray] = []
    for index in np.argsort(np.asarray(energies, dtype=float)):
        candidate = np.asarray(population[index], dtype=float)
        scaled = (candidate - lower) / span
        if all(np.linalg.norm(scaled - previous) >= 0.03 for previous in scaled_selected):
            selected.append((candidate, int(index), float(energies[index])))
            scaled_selected.append(scaled)
        if len(selected) >= int(count):
            break
    return selected


def matrix_diagnostics(jacobian: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(jacobian, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    norms = np.linalg.norm(matrix, axis=0)
    return {
        "parameter_order": list(FREE_PARAMS),
        "singular_values": singular,
        "condition_number": None
        if singular[-1] == 0.0
        else float(singular[0] / singular[-1]),
        "smallest_singular_value": float(singular[-1]),
        "numerical_rank": int(np.sum(singular > tolerance)),
        "column_norms": {FREE_PARAMS[i]: float(norms[i]) for i in range(5)},
    }


def _quality(attempt: dict[str, Any], config: FitConfig) -> tuple[int, str]:
    if bool(attempt["success"]) and int(attempt["status"]) > 0:
        return 0, "terminated"
    if int(attempt["status"]) == 0 and int(attempt["nfev"]) >= int(config.max_nfev):
        return 1, "budget_exhausted"
    return 2, "finite_other"


def select_local_candidate(
    attempts: list[dict[str, Any]], config: FitConfig
) -> dict[str, Any]:
    enriched = []
    for item in attempts:
        row = dict(item)
        metrics = boundary_metrics(
            np.asarray(row["final_free"], dtype=float), config.boundary_margin_fraction
        )
        grade, quality = _quality(row, config)
        row.update(
            {
                "quality_grade": grade,
                "quality_class": quality,
                "boundary": metrics,
            }
        )
        enriched.append(row)
    valid = [
        row
        for row in enriched
        if np.isfinite(float(row["spectrum_cost"]))
        and np.isfinite(float(row["optimality"]))
        and int(row["status"]) >= 0
    ]
    if not valid:
        return {"selected": None, "attempts": enriched, "reason": "no valid candidate"}
    minimum = min(float(row["spectrum_cost"]) for row in valid)
    tolerance = max(
        abs(minimum) * float(config.spectrum_equivalence_rtol),
        SPECTRUM_EQUIVALENCE_ATOL,
    )
    equivalent = [
        row for row in valid if float(row["spectrum_cost"]) <= minimum + tolerance
    ]
    minimum_cost_candidate = min(
        valid, key=lambda row: (float(row["spectrum_cost"]), int(row["call_index"]))
    )
    selected = min(
        equivalent,
        key=lambda row: (
            int(row["quality_grade"]),
            float(row["boundary"]["severity"]),
            len(row["boundary"]["boundary_hits"]),
            float(row["optimality"]),
            float(row["spectrum_cost"]),
            int(row["call_index"]),
        ),
    )
    return {
        "selected": selected,
        "attempts": enriched,
        "minimum_spectrum_cost": minimum,
        "equivalent_candidate_count": len(equivalent),
        "spectrum_equivalence_rtol": float(config.spectrum_equivalence_rtol),
        "boundary_preference_changed_selection": bool(
            int(selected["call_index"]) != int(minimum_cost_candidate["call_index"])
        ),
        "selected_spectrum_cost_increase_fraction": (
            float(selected["spectrum_cost"]) - minimum
        )
        / max(abs(minimum), np.finfo(float).eps),
        "reason": (
            "minimum spectrum-cost equivalence set, then termination quality, "
            "boundary severity, hard hits, optimality, and spectrum cost"
        ),
    }


def fit_measurement(measurement: dict[str, Any], config: FitConfig, seed: int) -> dict:
    if config.global_forward_model != "full_ils":
        raise ValueError("V12 production solver supports strict full_ils only")
    observed = np.asarray(measurement["spectrum"], dtype=np.float64)
    scale = v10.robust_scale(observed)
    fixed_angle = float(measurement["fixed_angle_deg"])
    global_indices = np.arange(0, len(observed), max(1, int(config.global_stride)))
    model = v10.SpectrometerForwardModel(
        measurement["wavelengths_um"],
        measurement["generator_config"],
        measurement["metadata"]["internal_wavelength_margin_nm"],
    )

    def prediction(free_values: np.ndarray) -> np.ndarray:
        return model.predict(full_parameters(free_values, fixed_angle))

    def residual(free_values: np.ndarray) -> np.ndarray:
        return (prediction(free_values) - observed) / scale

    def global_objective(free_values: np.ndarray) -> float:
        residual_values = residual(free_values)[global_indices]
        spectrum = v10.robust_objective(residual_values, config.loss)
        total, _, _ = penalized_objective(
            spectrum,
            free_values,
            config.boundary_margin_fraction,
            config.boundary_penalty_weight,
        )
        return total

    started = time.perf_counter()
    population_size = max(5, int(config.global_popsize) * len(FREE_PARAMS))
    initial_population = latin_hypercube_population(seed, population_size)
    global_started = time.perf_counter()
    global_result = differential_evolution(
        global_objective,
        bounds=[v10.BOUNDS[name] for name in FREE_PARAMS],
        strategy="best1bin",
        maxiter=int(config.global_maxiter),
        popsize=int(config.global_popsize),
        tol=1.0e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=int(seed),
        polish=False,
        init=initial_population,
        updating="immediate",
        workers=1,
    )
    global_runtime = time.perf_counter() - global_started
    candidates = select_diverse_candidates(
        np.asarray(global_result.population),
        np.asarray(global_result.population_energies),
        config.multistarts,
    )
    if not candidates:
        raise RuntimeError("global search produced no diverse candidates")
    lower, upper = bounds_arrays()
    span = upper - lower

    def to_unit(values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values) - lower) / span, 0.0, 1.0)

    def from_unit(values: np.ndarray) -> np.ndarray:
        return lower + np.asarray(values) * span

    local_started = time.perf_counter()
    attempts = []
    for rank, (start, population_index, global_energy) in enumerate(candidates, 1):
        trace = []

        def fun(unit_values):
            values = from_unit(unit_values)
            current = residual(values)
            trace.append(0.5 * v10.robust_objective(current, config.loss))
            return current

        result = least_squares(
            fun,
            x0=to_unit(start),
            bounds=(np.zeros(5), np.ones(5)),
            loss=config.loss,
            max_nfev=int(config.max_nfev),
            x_scale=1.0,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=float(config.local_gtol),
        )
        final_free = from_unit(result.x)
        attempts.append(
            {
                "call_index": rank,
                "population_index": population_index,
                "global_total_objective": global_energy,
                "x0_free": start,
                "final_solver": np.asarray(result.x),
                "final_free": final_free,
                "final_full": full_parameters(final_free, fixed_angle),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "spectrum_cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "trace_costs": trace,
            }
        )
    local_runtime = time.perf_counter() - local_started
    ranked = select_local_candidate(attempts, config)
    selected = ranked["selected"]
    if selected is None:
        raise RuntimeError("no valid V12 local candidate")
    selected_free = np.asarray(selected["final_free"], dtype=float)
    selected_prediction = prediction(selected_free)
    jacobian = np.empty((len(observed), 5), dtype=float)
    for index in range(5):
        step = max(1.0e-6 * span[index], 1.0e-8)
        plus, minus = selected_free.copy(), selected_free.copy()
        plus[index] = min(upper[index], plus[index] + step)
        minus[index] = max(lower[index], minus[index] - step)
        jacobian[:, index] = (
            residual(plus) - residual(minus)
        ) / (plus[index] - minus[index])
    return {
        "success": True,
        "fixed_angle_deg": fixed_angle,
        "free_parameters": selected_free,
        "full_parameters": full_parameters(selected_free, fixed_angle),
        "spectrum_cost": float(selected["spectrum_cost"]),
        "exact_rmse": float(np.sqrt(np.mean((selected_prediction - observed) ** 2))),
        "boundary": selected["boundary"],
        "ranking": ranked,
        "diagnostics": matrix_diagnostics(jacobian),
        "global_runtime_s": global_runtime,
        "local_runtime_s": local_runtime,
        "runtime_s": time.perf_counter() - started,
        "global_summary": {
            "population_size": population_size,
            "global_nit": int(global_result.nit),
            "global_nfev": int(global_result.nfev),
            "global_success": bool(global_result.success),
            "global_message": str(global_result.message),
            "boundary_margin_fraction": config.boundary_margin_fraction,
            "boundary_penalty_weight": config.boundary_penalty_weight,
        },
        "fitted_spectrum": selected_prediction,
    }


def scientific_errors(full_values: np.ndarray, truth: dict[str, float]) -> dict[str, float]:
    values = np.asarray(full_values, dtype=float)
    return {
        "Air_error_nm": 1000.0 * (float(values[0]) - truth["Air"]),
        "absolute_Air_error_nm": abs(1000.0 * (float(values[0]) - truth["Air"])),
        "film_MAE_nm": float(
            np.mean([abs(float(values[i]) - truth[name]) for i, name in enumerate(FREE_PARAMS[1:5], 1)])
        ),
        "angle_measurement_error_deg": float(values[5]) - float(truth["Angle"]),
        "angle_abs_error_deg": abs(float(values[5]) - float(truth["Angle"])),
    }


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="static_spectrum_*.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--wavelength-min-nm", type=float, default=220.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--global-stride", type=int, default=1)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--local-gtol", type=float, default=1.0e-5)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--loss", default="soft_l1")
    parser.add_argument("--boundary-margin-fraction", type=float, default=0.02)
    parser.add_argument("--boundary-penalty-weight", type=float, default=0.05)
    parser.add_argument("--spectrum-equivalence-rtol", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.resolve().glob(args.pattern), key=lambda path: path.name)
    if args.max_files is not None:
        paths = paths[: int(args.max_files)]
    if not paths:
        raise FileNotFoundError("no V12 NPZ files matched")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = FitConfig(
        input_dir=str(args.input_dir.resolve()),
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        global_stride=args.global_stride,
        global_popsize=args.global_popsize,
        global_maxiter=args.global_maxiter,
        multistarts=args.multistarts,
        max_nfev=args.max_nfev,
        local_gtol=args.local_gtol,
        random_seed=args.random_seed,
        loss=args.loss,
        boundary_margin_fraction=args.boundary_margin_fraction,
        boundary_penalty_weight=args.boundary_penalty_weight,
        spectrum_equivalence_rtol=args.spectrum_equivalence_rtol,
    )
    started = time.perf_counter()
    rows = []
    for index, path in enumerate(paths, 1):
        measurement = load_fit_input(path, config)
        seed = int(config.random_seed + (index - 1) * 1009)
        fit = fit_measurement(measurement, config, seed)
        truth, _ = load_evaluation_truth(path)
        errors = scientific_errors(fit["full_parameters"], truth)
        row = {
            "index": index,
            "filename": path.name,
            "seed": seed,
            "metadata": measurement["metadata"],
            "fit": fit,
            "errors": errors,
        }
        rows.append(row)
        print(
            f"[{index}/{len(paths)}] {path.name} cost={fit['spectrum_cost']:.6g} "
            f"hits={fit['boundary']['boundary_hits']} angle={fit['fixed_angle_deg']:.6f}",
            flush=True,
        )
    report = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "summary": {
            "processed": len(rows),
            "cases_with_boundary_hits": sum(bool(row["fit"]["boundary"]["boundary_hits"]) for row in rows),
            "cases_near_boundary": sum(bool(row["fit"]["boundary"]["near_boundary"]) for row in rows),
            "boundary_preference_changed_selection": sum(bool(row["fit"]["ranking"]["boundary_preference_changed_selection"]) for row in rows),
            "mean_absolute_Air_error_nm": float(np.mean([row["errors"]["absolute_Air_error_nm"] for row in rows])),
            "mean_film_MAE_nm": float(np.mean([row["errors"]["film_MAE_nm"] for row in rows])),
            "mean_angle_abs_error_deg": float(np.mean([row["errors"]["angle_abs_error_deg"] for row in rows])),
            "runtime_s": time.perf_counter() - started,
        },
        "cases": rows,
        "scope_guard": "Angle fixed to independent measurement; no MAP prior; five fitted parameters; strict full ILS.",
    }
    (output_dir / "v12_cpu_results.json").write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "index", "filename", "noise_case", "measured_angle_deg", "angle_abs_error_deg",
        "spectrum_cost", "exact_rmse", "Air_error_nm", "film_MAE_nm",
        "boundary_hits", "near_boundary", "condition_number", "runtime_s",
    ]
    with (output_dir / "v12_cpu_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            fit = row["fit"]
            writer.writerow(
                {
                    "index": row["index"],
                    "filename": row["filename"],
                    "noise_case": row["metadata"]["noise_case"],
                    "measured_angle_deg": fit["fixed_angle_deg"],
                    "angle_abs_error_deg": row["errors"]["angle_abs_error_deg"],
                    "spectrum_cost": fit["spectrum_cost"],
                    "exact_rmse": fit["exact_rmse"],
                    "Air_error_nm": row["errors"]["Air_error_nm"],
                    "film_MAE_nm": row["errors"]["film_MAE_nm"],
                    "boundary_hits": ";".join(fit["boundary"]["boundary_hits"]),
                    "near_boundary": ";".join(fit["boundary"]["near_boundary"]),
                    "condition_number": fit["diagnostics"]["condition_number"],
                    "runtime_s": fit["runtime_s"],
                }
            )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
