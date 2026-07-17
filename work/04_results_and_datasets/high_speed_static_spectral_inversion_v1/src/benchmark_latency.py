from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import numpy as np
import pandas as pd

from model_config import LOWER_BOUNDS, PARAMETER_NAMES, UPPER_BOUNDS
from objective_functions import FitProblem
from optimizer_cmaes import optimize_cmaes
from optimizer_common import InversionResult, result_dict
from optimizer_de import optimize_de_best1bin, optimize_de_rand1bin
from optimizer_direct import optimize_direct
from optimizer_fft_hybrid import optimize_fft_hybrid
from optimizer_local import optimize_local
from optimizer_sobol import optimize_sobol
from spectrum_preprocess import preprocess_spectrum

Optimizer = Callable[..., InversionResult]
OPTIMIZERS: dict[str, Optimizer] = {
    "local": optimize_local,
    "sobol": optimize_sobol,
    "de_best1bin": optimize_de_best1bin,
    "de_rand1bin": optimize_de_rand1bin,
    "cmaes": optimize_cmaes,
    "direct": optimize_direct,
    "fft_hybrid": optimize_fft_hybrid,
}


def load_dataset_timed(path: Path) -> tuple[dict[str, np.ndarray], float]:
    begin = perf_counter_ns()
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    elapsed_ms = (perf_counter_ns() - begin) / 1e6
    required = {"wavelengths_um", "spectra", "air_cavity_um", "film_thicknesses_nm", "noise_sigma"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Dataset is missing fields: {sorted(missing)}")
    if data["spectra"].ndim != 2 or data["spectra"].shape[1] != data["wavelengths_um"].size:
        raise ValueError("spectra must have shape (N_samples, N_wavelengths).")
    return data, elapsed_ms


def run_benchmark(
    data: dict[str, np.ndarray],
    config: dict,
    algorithms: list[str],
    modes: list[str],
    max_samples: int | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    unknown = sorted(set(algorithms).difference(OPTIMIZERS))
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")
    sample_count = min(data["spectra"].shape[0], max_samples or data["spectra"].shape[0])
    fit_config = dict(config["fit"])
    seeds = [int(value) for value in config["benchmark"]["random_seeds"]]
    truth = np.column_stack((data["air_cavity_um"][:sample_count], data["film_thicknesses_nm"][:sample_count]))
    rows: list[dict] = []
    fitted_spectra: list[np.ndarray] = []
    fit_keys: list[str] = []
    cold_begin = perf_counter_ns()
    cold_measurement = preprocess_spectrum(data["wavelengths_um"], data["spectra"][0], float(np.asarray(data["noise_sigma"])[0]))
    FitProblem(cold_measurement, str(fit_config["loss"]))
    cold_start_ms = (perf_counter_ns() - cold_begin) / 1e6

    for mode in modes:
        if mode not in {"absolute", "tracking"}:
            raise ValueError("modes must be absolute and/or tracking.")
        for algorithm in algorithms:
            for seed in seeds:
                previous: np.ndarray | None = None
                for sample_index in range(sample_count):
                    preprocess_begin = perf_counter_ns()
                    measurement = preprocess_spectrum(
                        data["wavelengths_um"], data["spectra"][sample_index], float(np.asarray(data["noise_sigma"])[sample_index])
                    )
                    problem = FitProblem(measurement, str(fit_config["loss"]))
                    preprocess_ms = (perf_counter_ns() - preprocess_begin) / 1e6
                    online_begin = perf_counter_ns()
                    if mode == "tracking" and previous is not None:
                        maximum_step = float(fit_config["max_frame_step_um"])
                        lower = LOWER_BOUNDS.copy()
                        upper = UPPER_BOUNDS.copy()
                        lower[0] = max(lower[0], previous[0] - maximum_step)
                        upper[0] = min(upper[0], previous[0] + maximum_step)
                        result = optimize_local(problem, fit_config, seed + sample_index, initial=previous, lower=lower, upper=upper)
                        result.algorithm = algorithm
                    else:
                        result = OPTIMIZERS[algorithm](problem, fit_config, seed + sample_index)
                    base = result_dict(result)
                    total_online_ms = preprocess_ms + (perf_counter_ns() - online_begin) / 1e6
                    measured_stages = result.coarse_search_ms + result.global_search_ms + result.local_refine_ms
                    result_pack_ms = max(0.0, total_online_ms - preprocess_ms - measured_stages)
                    previous = result.values.copy() if result.success else None

                    # Truth is joined only after the optimizer result has been finalized.
                    error = result.values - truth[sample_index]
                    base.update({
                        "sample_index": sample_index,
                        "run_mode": mode,
                        "random_seed": seed,
                        "preprocess_ms": preprocess_ms,
                        "result_pack_ms": result_pack_ms,
                        "total_online_ms": total_online_ms,
                        "timeout": bool(total_online_ms > 1000.0 * float(fit_config["timeout_s"])),
                        "correct_air_order": bool(abs(error[0]) <= float(fit_config["correct_air_threshold_um"])),
                    })
                    for index, name in enumerate(PARAMETER_NAMES):
                        base[f"truth_{name}"] = float(truth[sample_index, index])
                        base[f"error_{name}"] = float(error[index])
                    rows.append(base)

                    # Fitted arrays are generated outside the online timing boundary.
                    evaluation = problem.evaluate(result.values)
                    fitted_spectra.append(evaluation.fitted)
                    fit_keys.append(f"{mode}|{algorithm}|{seed}|{sample_index}")

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "cold_start_ms": cold_start_ms,
        "sample_count": sample_count,
        "wavelength_count": int(data["wavelengths_um"].size),
        "timing_boundary": "in-memory spectrum -> preprocess -> search -> local refine -> result object",
        "truth_isolation": "truth arrays are joined only after optimizer completion",
    }
    arrays = {"keys": np.asarray(fit_keys), "fitted_spectra": np.asarray(fitted_spectra), "wavelengths_um": data["wavelengths_um"]}
    return pd.DataFrame(rows), arrays, metadata


def write_run_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    fitted: dict[str, np.ndarray],
    metadata: dict,
    config: dict,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=False)
    begin = perf_counter_ns()
    results["disk_output_ms"] = 0.0
    results.to_csv(output_dir / "per_spectrum_results.csv", index=False)
    latency_columns = ["sample_index", "algorithm", "run_mode", "random_seed", "preprocess_ms", "coarse_search_ms", "global_search_ms", "local_refine_ms", "result_pack_ms", "total_online_ms", "disk_output_ms", "n_forward_evaluations"]
    results[latency_columns].to_csv(output_dir / "latency_breakdown.csv", index=False)
    np.savez_compressed(output_dir / "fitted_spectra.npz", **fitted)
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    disk_output_ms = (perf_counter_ns() - begin) / 1e6
    results["disk_output_ms"] = disk_output_ms / max(1, len(results))
    results.to_csv(output_dir / "per_spectrum_results.csv", index=False)
    results[latency_columns].to_csv(output_dir / "latency_breakdown.csv", index=False)
    metadata["disk_output_ms"] = disk_output_ms
    with (output_dir / "benchmark_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return disk_output_ms
