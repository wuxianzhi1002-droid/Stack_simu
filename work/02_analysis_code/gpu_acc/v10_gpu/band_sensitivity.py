"""Matched-parameter GPU Jacobian sensitivity comparison for two wavelength bands.

This is a diagnostic-only runner.  It evaluates the frozen strict reported
response R(lambda), including the formal spectrometer response and ILS, with
the validated B=13 CuPy center-difference path.  It does not run an optimizer
or change the formal V10 source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backend.v10_source import load_v10_module, source_sha256
from .identifiability import correlation_as_json, normalized_column_correlation
from .jacobian import BatchedCenterDifferenceJacobian, formal_cpu_jacobian
from .jacobian.contract import PARAMETER_NAMES
from .jacobian.metrics import column_metrics
from .optimizer_contract import effective_physical_scales, solver_jacobian_from_physical
from .spectrometer import CupyStrictSpectrometerBackend, NumpyStrictSpectrometerBackend


EXPECTED_V10_SHA256 = "d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"
BANDS = (("220_580", 220.0, 580.0), ("450_580", 450.0, 580.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _stats(values) -> dict:
    array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _svd_diagnostics(jacobian: np.ndarray) -> dict:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    largest = float(singular[0])
    smallest = float(singular[-1])
    tolerance = float(max(jacobian.shape) * np.finfo(np.float64).eps * largest)
    condition = float(np.linalg.cond(jacobian))
    gram = jacobian.T @ jacobian
    condition_gram = float(np.linalg.cond(gram))
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "singular_values": singular.tolist(),
        "condition_number_J": _finite(condition),
        "condition_number_J_is_infinite": not np.isfinite(condition),
        "condition_number_JTJ": _finite(condition_gram),
        "condition_number_JTJ_is_infinite": not np.isfinite(condition_gram),
        "condition_number_J_squared": _finite(condition * condition),
        "numerical_rank": int(np.sum(singular > tolerance)),
        "rank_tolerance": tolerance,
        "smallest_singular_value": smallest,
        "largest_singular_value": largest,
        "gram_matrix": gram.tolist(),
        "gram_eigenvalues": eigenvalues.tolist(),
    }


def _case_metrics(jacobian: np.ndarray, parameters: np.ndarray, wavelengths_nm: np.ndarray) -> dict:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    norms, correlation = normalized_column_correlation(jacobian)
    rms = np.sqrt(np.mean(jacobian * jacobian, axis=0))
    integrated = np.sqrt(np.trapz(jacobian * jacobian, x=wavelengths_nm, axis=0))
    normalized = np.full_like(jacobian, np.nan)
    valid = norms > 0.0
    normalized[:, valid] = jacobian[:, valid] / norms[valid]
    normalized_gram = normalized[:, valid].T @ normalized[:, valid]
    normalized_condition = float(np.linalg.cond(normalized_gram)) if valid.all() else float("inf")

    scales = effective_physical_scales(parameters)
    scaled = solver_jacobian_from_physical(jacobian, parameters) if scales.defined else None
    result = {
        "parameters": {name: float(parameters[index]) for index, name in enumerate(PARAMETER_NAMES)},
        "parameter_vector": parameters.tolist(),
        "angle_deg": float(parameters[5]),
        "reported_point_count": int(jacobian.shape[0]),
        "column_l2_norms": {name: float(norms[index]) for index, name in enumerate(PARAMETER_NAMES)},
        "column_rms_sensitivity": {name: float(rms[index]) for index, name in enumerate(PARAMETER_NAMES)},
        "column_integrated_l2": {name: float(integrated[index]) for index, name in enumerate(PARAMETER_NAMES)},
        "correlation_matrix": correlation_as_json(correlation),
        "correlation_parameter_order": list(PARAMETER_NAMES),
        "rho_air_angle": _finite(correlation[0, 5]),
        "normalized_gram_condition": _finite(normalized_condition),
        "normalized_gram_condition_is_infinite": not np.isfinite(normalized_condition),
        "raw": _svd_diagnostics(jacobian),
        "optimizer_scaling": {
            "defined": bool(scales.defined),
            "undefined_reason": scales.reason,
            "effective_physical_scales": {
                name: _finite(scales.values[index]) for index, name in enumerate(PARAMETER_NAMES)
            },
            "least_squares_x_scale": 1.0,
            "coordinate_contract": "formal V10 [0,1]^6 local coordinates; Angle uses the frozen square/square-root map",
        },
        "scaled": _svd_diagnostics(scaled) if scaled is not None else None,
    }
    return result


def _load_truth_grid(v10, root_220: Path, root_450: Path) -> tuple[np.ndarray, dict]:
    path_sets = [sorted(root.glob("static_spectrum_*.npz"), key=lambda path: path.name) for root in (root_220, root_450)]
    if [path.name for path in path_sets[0]] != [path.name for path in path_sets[1]]:
        raise RuntimeError("The two band datasets do not have identical sorted NPZ filenames.")
    if len(path_sets[0]) != 401:
        raise RuntimeError(f"Expected 401 matched NPZ files, found {len(path_sets[0])}.")
    arrays = []
    for paths in path_sets:
        rows = []
        for path in paths:
            truth, _ = v10.load_evaluation_truth(path)
            rows.append([truth[name] for name in PARAMETER_NAMES])
        arrays.append(np.asarray(rows, dtype=np.float64))
    if not np.array_equal(arrays[0], arrays[1]):
        raise RuntimeError("Matched files do not contain identical physical truth parameters.")
    unique = np.unique(arrays[0], axis=0)
    return unique, {
        "matched_npz_count": len(path_sets[0]),
        "unique_parameter_count": int(unique.shape[0]),
        "filenames_identical": True,
        "truth_parameters_identical": True,
        "source_example_files": [str(paths[0].resolve()) for paths in path_sets],
    }


def _load_band(v10, root: Path, low_nm: float, high_nm: float):
    paths = sorted(root.glob("static_spectrum_*.npz"), key=lambda path: path.name)
    clean = [path for path in paths if "clean" in path.name]
    source = clean[0] if clean else paths[0]
    config = v10.FitConfig(
        input_dir=str(root),
        wavelength_min_nm=low_nm,
        wavelength_max_nm=high_nm,
        global_forward_model="full_ils",
        workers=1,
    )
    measurement = v10.load_fit_input(source, config)
    wavelengths_um = np.asarray(measurement["wavelengths_um"], dtype=np.float64)
    margin = float(measurement["metadata"]["internal_wavelength_margin_nm"])
    gpu = CupyStrictSpectrometerBackend(wavelengths_um, measurement["generator_config"], margin)
    cpu = NumpyStrictSpectrometerBackend(wavelengths_um, measurement["generator_config"], margin)
    return {
        "source": source,
        "measurement": measurement,
        "wavelengths_um": wavelengths_um,
        "wavelengths_nm": wavelengths_um * 1000.0,
        "gpu": gpu,
        "cpu": cpu,
        "jacobian": BatchedCenterDifferenceJacobian(gpu),
    }


def _closure(cpu_backend, gpu_jacobian, parameters: np.ndarray) -> dict:
    observed = np.zeros(cpu_backend.reported_wavelengths_um.shape, dtype=np.float64)
    started = time.perf_counter()
    cpu = formal_cpu_jacobian(cpu_backend.model, parameters, observed, 1.0)
    cpu_s = time.perf_counter() - started
    gpu_jacobian.response_backend.synchronize()
    started = time.perf_counter()
    gpu = gpu_jacobian.evaluate(parameters, 1.0).jacobian
    gpu_jacobian.response_backend.synchronize()
    gpu_s = time.perf_counter() - started
    metrics = column_metrics(cpu, gpu)
    defined_rel = [row["relative_l2_error"] for row in metrics if row["relative_error_defined"]]
    cosines = [row["cosine_similarity"] for row in metrics if row["cosine_similarity"] is not None]
    nan_inf = int(np.isnan(gpu).sum() + np.isinf(gpu).sum())
    passed = bool(
        nan_inf == 0
        and max(defined_rel, default=0.0) <= 1.0e-5
        and min(cosines, default=1.0) >= 0.99999
    )
    return {
        "pass": passed,
        "parameters": parameters.tolist(),
        "columns": metrics,
        "maximum_defined_relative_l2_error": max(defined_rel, default=None),
        "minimum_defined_cosine_similarity": min(cosines, default=None),
        "maximum_absolute_difference": max(row["max_abs_difference"] for row in metrics),
        "nan_inf_count": nan_inf,
        "cpu_runtime_s": cpu_s,
        "gpu_runtime_s": gpu_s,
    }


def _band_aggregate(cases: list[dict]) -> dict:
    result = {
        "raw_condition_number_J": _stats(case["raw"]["condition_number_J"] for case in cases),
        "raw_condition_number_JTJ": _stats(case["raw"]["condition_number_JTJ"] for case in cases),
        "scaled_condition_number_J": _stats(case["scaled"]["condition_number_J"] if case["scaled"] else None for case in cases),
        "scaled_condition_number_JTJ": _stats(case["scaled"]["condition_number_JTJ"] if case["scaled"] else None for case in cases),
        "normalized_gram_condition": _stats(case["normalized_gram_condition"] for case in cases),
        "raw_smallest_singular_value": _stats(case["raw"]["smallest_singular_value"] for case in cases),
        "scaled_smallest_singular_value": _stats(case["scaled"]["smallest_singular_value"] if case["scaled"] else None for case in cases),
        "absolute_rho_air_angle": _stats(abs(case["rho_air_angle"]) if case["rho_air_angle"] is not None else None for case in cases),
        "minimum_raw_rank": min(case["raw"]["numerical_rank"] for case in cases),
        "minimum_scaled_rank": min(case["scaled"]["numerical_rank"] for case in cases if case["scaled"]),
    }
    for metric in ("column_l2_norms", "column_rms_sensitivity", "column_integrated_l2"):
        result[metric] = {
            name: _stats(case[metric][name] for case in cases) for name in PARAMETER_NAMES
        }
    return result


def _representative_index(parameters: np.ndarray) -> int:
    return int(np.argmin(np.abs(parameters[:, 5] - 0.05)))


def _strongest_pairs(correlation: list[list[float | None]], count: int = 5) -> list[dict]:
    rows = []
    for i in range(6):
        for j in range(i + 1, 6):
            value = correlation[i][j]
            if value is not None:
                rows.append({"parameter_i": PARAMETER_NAMES[i], "parameter_j": PARAMETER_NAMES[j], "rho": float(value), "abs_rho": abs(float(value))})
    return sorted(rows, key=lambda row: row["abs_rho"], reverse=True)[:count]


def _render_figures(output: Path, parameters: np.ndarray, band_results: dict, representative_jacobians: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes.flat, PARAMETER_NAMES)):
        for band, color in (("220_580", "tab:blue"), ("450_580", "tab:orange")):
            payload = representative_jacobians[band]
            axis.plot(payload["wavelengths_nm"], payload["jacobian"][:, index], label=band.replace("_", "-"), color=color, linewidth=1.0)
        axis.set_title(name)
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel(f"dR/d{name}")
        axis.grid(alpha=0.25)
        axis.legend()
    name = "figure1_jacobian_columns_vs_wavelength.png"
    fig.savefig(figure_dir / name, dpi=180)
    plt.close(fig)
    names.append(f"figures/{name}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, band in zip(axes, ("220_580", "450_580")):
        matrix = np.asarray(band_results[band]["representative_case"]["correlation_matrix"], dtype=float)
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(6), PARAMETER_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(6), PARAMETER_NAMES)
        axis.set_title(band.replace("_", "-") + " nm")
        for i in range(6):
            for j in range(6):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.85, label="Normalized column correlation")
    name = "figure2_correlation_matrices.png"
    fig.savefig(figure_dir / name, dpi=180)
    plt.close(fig)
    names.append(f"figures/{name}")

    x = np.arange(6)
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, band, color in ((-width / 2, "220_580", "tab:blue"), (width / 2, "450_580", "tab:orange")):
        medians = [band_results[band]["aggregate"]["column_integrated_l2"][name]["median"] for name in PARAMETER_NAMES]
        axis.bar(x + offset, medians, width, label=band.replace("_", "-"), color=color)
    axis.set_yscale("log")
    axis.set_xticks(x, PARAMETER_NAMES)
    axis.set_ylabel("Median integrated column sensitivity")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    name = "figure3_column_sensitivity_comparison.png"
    fig.savefig(figure_dir / name, dpi=180)
    plt.close(fig)
    names.append(f"figures/{name}")

    angles = parameters[:, 5]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for band, color in (("220_580", "tab:blue"), ("450_580", "tab:orange")):
        cases = band_results[band]["cases"]
        axes[0].plot(angles, [case["scaled"]["condition_number_J"] for case in cases], marker=".", linewidth=1.0, label=band.replace("_", "-"), color=color)
        axes[1].plot(angles, [case["normalized_gram_condition"] for case in cases], marker=".", linewidth=1.0, label=band.replace("_", "-"), color=color)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("cond(J_scaled)")
    axes[1].set_ylabel("cond(normalized J^T J)")
    axes[1].set_xlabel("Reflector angle (deg)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    name = "figure4_conditioning_vs_angle.png"
    fig.savefig(figure_dir / name, dpi=180)
    plt.close(fig)
    names.append(f"figures/{name}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for band, color in (("220_580", "tab:blue"), ("450_580", "tab:orange")):
        cases = band_results[band]["cases"]
        axes[0].plot(angles, [case["scaled"]["smallest_singular_value"] for case in cases], marker=".", linewidth=1.0, label=band.replace("_", "-"), color=color)
        axes[1].plot(angles, [abs(case["rho_air_angle"]) if case["rho_air_angle"] is not None else np.nan for case in cases], marker=".", linewidth=1.0, label=band.replace("_", "-"), color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Smallest singular value(J_scaled)")
    axes[1].set_ylabel("|rho(Air, Angle)|")
    axes[1].set_xlabel("Reflector angle (deg)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    name = "figure5_smallest_singular_and_air_angle.png"
    fig.savefig(figure_dir / name, dpi=180)
    plt.close(fig)
    names.append(f"figures/{name}")
    return names


def _write_csv(path: Path, band_results: dict) -> None:
    fields = [
        "band_nm", "case_index", "angle_deg", "raw_cond_J", "raw_cond_JTJ",
        "scaled_cond_J", "scaled_cond_JTJ", "normalized_gram_condition",
        "raw_smallest_singular", "scaled_smallest_singular", "rho_air_angle",
    ]
    for prefix in ("l2", "rms", "integrated"):
        fields.extend(f"{prefix}_{name}" for name in PARAMETER_NAMES)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for band in ("220_580", "450_580"):
            for index, case in enumerate(band_results[band]["cases"]):
                row = {
                    "band_nm": band.replace("_", "-"),
                    "case_index": index,
                    "angle_deg": case["angle_deg"],
                    "raw_cond_J": case["raw"]["condition_number_J"],
                    "raw_cond_JTJ": case["raw"]["condition_number_JTJ"],
                    "scaled_cond_J": case["scaled"]["condition_number_J"],
                    "scaled_cond_JTJ": case["scaled"]["condition_number_JTJ"],
                    "normalized_gram_condition": case["normalized_gram_condition"],
                    "raw_smallest_singular": case["raw"]["smallest_singular_value"],
                    "scaled_smallest_singular": case["scaled"]["smallest_singular_value"],
                    "rho_air_angle": case["rho_air_angle"],
                }
                for name in PARAMETER_NAMES:
                    row[f"l2_{name}"] = case["column_l2_norms"][name]
                    row[f"rms_{name}"] = case["column_rms_sensitivity"][name]
                    row[f"integrated_{name}"] = case["column_integrated_l2"][name]
                writer.writerow(row)


def _matrix_markdown(matrix: list[list[float | None]]) -> list[str]:
    lines = ["| | " + " | ".join(PARAMETER_NAMES) + " |", "|---|" + "---:|" * 6]
    for name, row in zip(PARAMETER_NAMES, matrix):
        values = ["undefined" if value is None else f"{value:.6f}" for value in row]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return lines


def _write_summary(path: Path, report: dict) -> None:
    bands = report["bands"]
    comparison = report["comparison"]
    lines = [
        "# TMM strict-response Jacobian band sensitivity comparison",
        "",
        f"- Overall: **{report['overall']}**",
        "- Bands: 220-580 nm and 450-580 nm",
        f"- Matched NPZ files: {report['parameter_grid']['matched_npz_count']}",
        f"- Unique matched physical points: {report['parameter_grid']['unique_parameter_count']}",
        "- Jacobian: dR(lambda)/d[Air, HSQ, PSS, SOC, TiO2, Angle]",
        "- Response contract: frozen strict TMM plus source/QE/throughput/photon weighting and full ILS.",
        "- GPU evaluation: one B=13 center-difference batch per point.",
        f"- Formal V10 unchanged: {report['formal_v10_sha256'] == EXPECTED_V10_SHA256}",
        "",
        "## Numerical closure and runtime",
        "",
        "| band | points | internal points | CPU/GPU closure | GPU Jacobian total | peak GPU pool |",
        "|---|---:|---:|:---:|---:|---:|",
    ]
    for band in ("220_580", "450_580"):
        item = bands[band]
        lines.append(
            f"| {band.replace('_','-')} | {item['reported_point_count']} | {item['internal_point_count']} | "
            f"{item['closure']['pass']} | {item['gpu_jacobian_runtime_s']:.6f} s | {item['peak_gpu_memory_gib']:.3f} GiB |"
        )
    lines += [
        "",
        "## Median integrated column sensitivity",
        "",
        "The integrated norm uses the wavelength axis and therefore measures total information over each band. The RMS column sensitivity measures average response per sampled wavelength.",
        "",
        "| parameter | 220-580 | 450-580 | information ratio 220/450 | RMS ratio 220/450 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in PARAMETER_NAMES:
        item = comparison["column_sensitivity"][name]
        lines.append(
            f"| {name} | {item['integrated_median_220_580']:.6e} | {item['integrated_median_450_580']:.6e} | "
            f"{item['integrated_ratio_220_over_450']:.6g} | {item['rms_ratio_220_over_450']:.6g} |"
        )
    lines += [
        "",
        "## Conditioning across matched parameter points",
        "",
        "| metric | 220-580 median | 450-580 median | preferred interpretation |",
        "|---|---:|---:|---|",
        f"| cond(J raw) | {bands['220_580']['aggregate']['raw_condition_number_J']['median']:.6e} | {bands['450_580']['aggregate']['raw_condition_number_J']['median']:.6e} | unit-dependent reference |",
        f"| cond(J_scaled) | {bands['220_580']['aggregate']['scaled_condition_number_J']['median']:.6e} | {bands['450_580']['aggregate']['scaled_condition_number_J']['median']:.6e} | formal optimizer-coordinate metric |",
        f"| cond(normalized J^T J) | {bands['220_580']['aggregate']['normalized_gram_condition']['median']:.6e} | {bands['450_580']['aggregate']['normalized_gram_condition']['median']:.6e} | structural column-collinearity metric |",
        f"| smallest singular(J_scaled) | {bands['220_580']['aggregate']['scaled_smallest_singular_value']['median']:.6e} | {bands['450_580']['aggregate']['scaled_smallest_singular_value']['median']:.6e} | larger is better |",
        f"| median abs rho(Air,Angle) | {bands['220_580']['aggregate']['absolute_rho_air_angle']['median']:.12f} | {bands['450_580']['aggregate']['absolute_rho_air_angle']['median']:.12f} | near one means local degeneracy |",
        "",
        "`cond(J^T J)` is retained in JSON/CSV, but `cond(J)` is primary because forming the Gram matrix squares the condition number and amplifies rounding error.",
        "",
        f"## Representative correlation matrices at angle {report['representative']['angle_deg']:.6g} deg",
        "",
        "### 220-580 nm",
        "",
    ]
    lines.extend(_matrix_markdown(bands["220_580"]["representative_case"]["correlation_matrix"]))
    lines += ["", "### 450-580 nm", ""]
    lines.extend(_matrix_markdown(bands["450_580"]["representative_case"]["correlation_matrix"]))
    lines += [
        "",
        "## Strongest representative column pairs",
        "",
    ]
    for band in ("220_580", "450_580"):
        lines.append(f"- {band.replace('_','-')} nm: " + ", ".join(
            f"{row['parameter_i']}-{row['parameter_j']} rho={row['rho']:.9f}"
            for row in bands[band]["strongest_representative_pairs"]
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        comparison["interpretation"],
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{name}`" for name in report["figures"])
    lines += [
        "",
        "## Scope",
        "",
        "Diagnostic only. No formal V10 physics, parameter definition, bounds, finite-difference step, ILS, residual, optimizer, dataset, JAX, autodiff, or multi-GPU behavior was changed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-220", type=Path, required=True)
    parser.add_argument("--dataset-450", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    args = parser.parse_args()

    if source_sha256() != EXPECTED_V10_SHA256:
        raise RuntimeError("Formal V10 hash changed.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    v10 = load_v10_module()
    parameters, grid_contract = _load_truth_grid(v10, args.dataset_220.resolve(), args.dataset_450.resolve())
    representative_index = _representative_index(parameters)
    representative_parameters = parameters[representative_index]
    bands = {}
    representative_jacobians = {}
    total_started = time.perf_counter()

    for band_name, low_nm, high_nm in BANDS:
        root = args.dataset_220.resolve() if band_name == "220_580" else args.dataset_450.resolve()
        state = _load_band(v10, root, low_nm, high_nm)
        gpu = state["gpu"]
        jacobian_runner = state["jacobian"]
        jacobian_runner.evaluate(representative_parameters, 1.0)
        gpu.synchronize()
        closure = _closure(state["cpu"], jacobian_runner, representative_parameters)
        cases = []
        representative_jacobian = None
        started = time.perf_counter()
        for index, values in enumerate(parameters):
            evaluation = jacobian_runner.evaluate(values, 1.0)
            gpu.synchronize()
            jacobian = evaluation.jacobian
            case = _case_metrics(jacobian, values, state["wavelengths_nm"])
            case["case_index"] = index
            cases.append(case)
            if index == representative_index:
                representative_jacobian = jacobian.copy()
        gpu_runtime = time.perf_counter() - started
        memory = gpu.memory_stats()
        representative_jacobians[band_name] = {
            "wavelengths_nm": state["wavelengths_nm"],
            "jacobian": representative_jacobian,
        }
        bands[band_name] = {
            "wavelength_min_nm": low_nm,
            "wavelength_max_nm": high_nm,
            "reported_point_count": int(state["wavelengths_nm"].size),
            "internal_point_count": int(memory["internal_points"]),
            "source_npz": str(state["source"].resolve()),
            "source_npz_sha256": _sha256(state["source"]),
            "closure": closure,
            "gpu_jacobian_runtime_s": gpu_runtime,
            "mean_gpu_jacobian_runtime_ms": gpu_runtime / len(parameters) * 1000.0,
            "peak_gpu_memory_bytes": int(memory["memory_pool_total_bytes"]),
            "peak_gpu_memory_gib": int(memory["memory_pool_total_bytes"]) / (1024 ** 3),
            "cases": cases,
            "aggregate": _band_aggregate(cases),
            "representative_case": cases[representative_index],
            "strongest_representative_pairs": _strongest_pairs(cases[representative_index]["correlation_matrix"]),
        }

    sensitivity_comparison = {}
    for name in PARAMETER_NAMES:
        integrated_220 = bands["220_580"]["aggregate"]["column_integrated_l2"][name]["median"]
        integrated_450 = bands["450_580"]["aggregate"]["column_integrated_l2"][name]["median"]
        rms_220 = bands["220_580"]["aggregate"]["column_rms_sensitivity"][name]["median"]
        rms_450 = bands["450_580"]["aggregate"]["column_rms_sensitivity"][name]["median"]
        sensitivity_comparison[name] = {
            "integrated_median_220_580": integrated_220,
            "integrated_median_450_580": integrated_450,
            "integrated_ratio_220_over_450": integrated_220 / integrated_450 if integrated_450 else None,
            "rms_median_220_580": rms_220,
            "rms_median_450_580": rms_450,
            "rms_ratio_220_over_450": rms_220 / rms_450 if rms_450 else None,
        }

    scaled_220 = bands["220_580"]["aggregate"]["scaled_condition_number_J"]["median"]
    scaled_450 = bands["450_580"]["aggregate"]["scaled_condition_number_J"]["median"]
    norm_gram_220 = bands["220_580"]["aggregate"]["normalized_gram_condition"]["median"]
    norm_gram_450 = bands["450_580"]["aggregate"]["normalized_gram_condition"]["median"]
    if scaled_220 < scaled_450 and norm_gram_220 < norm_gram_450:
        interpretation = "The 220-580 nm band is better conditioned in both formal optimizer coordinates and the unit-free normalized-column Gram matrix. The added short-wavelength spectrum supplies independent parameter information rather than only increasing sample count."
    elif scaled_220 > scaled_450 and norm_gram_220 > norm_gram_450:
        interpretation = "The 450-580 nm band is better conditioned by both primary metrics for this matched parameter grid. Inspect the column-sensitivity ratios and correlation matrices before attributing this to total information, because a narrower band may have lower total sensitivity despite a lower condition number."
    else:
        interpretation = "The scaled and normalized conditioning metrics do not agree on one uniformly superior band. Interpret total column sensitivity, pairwise correlations, and the smallest singular value together; condition number alone is insufficient."

    archive_path = output / "band_sensitivity_representative_jacobians.npz"
    np.savez_compressed(
        archive_path,
        parameter_order=np.asarray(PARAMETER_NAMES),
        matched_unique_parameters=parameters,
        representative_parameters=representative_parameters,
        wavelengths_nm_220_580=representative_jacobians["220_580"]["wavelengths_nm"],
        jacobian_220_580=representative_jacobians["220_580"]["jacobian"],
        wavelengths_nm_450_580=representative_jacobians["450_580"]["wavelengths_nm"],
        jacobian_450_580=representative_jacobians["450_580"]["jacobian"],
    )
    figures = _render_figures(output, parameters, bands, representative_jacobians)
    machine = json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None
    passed = bool(
        all(bands[name]["closure"]["pass"] for name, _, _ in BANDS)
        and source_sha256() == EXPECTED_V10_SHA256
    )
    report = {
        "overall": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "formal_v10_sha256": source_sha256(),
        "parameter_order": list(PARAMETER_NAMES),
        "jacobian_definition": "d(strict reported response R(lambda))/d physical parameters; residual scale=1",
        "response_contract": "strict TMM + source/QE/throughput/photon weighting + full ILS + reported sampling/reference normalization",
        "finite_difference_contract": "unchanged formal V10 bounded center difference; B=13",
        "parameter_grid": grid_contract,
        "representative": {
            "case_index": representative_index,
            "angle_deg": float(representative_parameters[5]),
            "parameters": {name: float(representative_parameters[index]) for index, name in enumerate(PARAMETER_NAMES)},
        },
        "bands": bands,
        "comparison": {
            "column_sensitivity": sensitivity_comparison,
            "median_scaled_condition_ratio_450_over_220": scaled_450 / scaled_220,
            "median_normalized_gram_condition_ratio_450_over_220": norm_gram_450 / norm_gram_220,
            "interpretation": interpretation,
        },
        "archive": {"path": str(archive_path), "sha256": _sha256(archive_path)},
        "figures": figures,
        "machine": machine,
        "total_runtime_s": time.perf_counter() - total_started,
        "scope_guard": "No formal V10 physics, parameters, bounds, finite-difference step, ILS, residual, optimizer, dataset, JAX, autodiff, or multi-GPU changes.",
    }
    json_path = output / "band_sensitivity.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    _write_csv(output / "band_sensitivity_table.csv", bands)
    _write_summary(output / "band_sensitivity_summary.md", report)
    print(json.dumps({"overall": report["overall"], "unique_parameters": len(parameters), "runtime_s": report["total_runtime_s"], "output": str(output)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
