from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"

NOMINAL_NM = {
    "HSQ": 40.0,
    "PSS": 5.0,
    "SOC": 50.0,
    "TiO2": 20.0,
}

FILM_BOUNDS_NM = {
    "HSQ": (20.0, 60.0),
    "PSS": (1.0, 20.0),
    "SOC": (30.0, 80.0),
    "TiO2": (5.0, 60.0),
}


@dataclass(frozen=True)
class ValidationConfig:
    wavelength_start_um: float = 0.2
    wavelength_stop_um: float = 0.6
    spectral_resolution_nm: float = 0.1
    random_seed: int = 20260707
    noise_sigmas_reflectance: tuple[float, ...] = (0.0, 0.002, 0.005)
    mc_trials_per_noise: int = 3
    truth_samples_per_case: int = 50
    multistarts: int = 12
    max_nfev: int = 120
    fit_affine_intensity: bool = True
    wavelength_offset_sigma_nm: float = 0.03
    wavelength_scale_sigma_ppm: float = 80.0
    material_real_sigma_fraction: float = 0.003
    material_imag_sigma_fraction: float = 0.05
    fixed_layer_sigma_nm: float = 0.5
    near_solution_cost_rel_tol: float = 1.0e-3
    near_solution_cost_abs_tol: float = 1.0e-10
    source_scale_sigma: float = 0.02
    source_offset_sigma: float = 0.002


def wavelength_axis(config: ValidationConfig) -> np.ndarray:
    count = int(
        round(
            (config.wavelength_stop_um - config.wavelength_start_um)
            * 1000.0
            / config.spectral_resolution_nm
        )
    ) + 1
    return np.linspace(config.wavelength_start_um, config.wavelength_stop_um, count)


def material_n(
    name: str,
    wavelengths_um: np.ndarray,
    material_scale: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=np.float64)
    if name == "RefReflector":
        n = np.full_like(w, 5.8284, dtype=np.complex128)
    elif name == "Air":
        n = np.full_like(w, 1.0, dtype=np.complex128)
    elif name == "HSQ":
        n = np.full_like(w, 1.41, dtype=np.complex128)
    elif name == "PSS":
        n = np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    elif name == "SOC":
        n = (1.55 + 0.005 / (w**2)).astype(np.complex128)
    elif name == "TiO2":
        n = (2.4 + 0.02 / (w**2)).astype(np.complex128)
    elif name == "Cu":
        n = np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown material: {name}")

    if material_scale and name in material_scale and name != "Air":
        real_scale, imag_scale = material_scale[name]
        n = np.real(n) * real_scale + 1j * np.imag(n) * imag_scale
    return n


def characteristic_admittance(n_values: np.ndarray, cos_values: np.ndarray, pol: str) -> np.ndarray:
    if pol.lower() == "s":
        return n_values * cos_values
    if pol.lower() == "p":
        return cos_values / n_values
    raise ValueError(f"Unsupported polarization: {pol}")


def tmm_reflectance(
    wavelengths_um: np.ndarray,
    layer_names: list[str],
    thicknesses_um: dict[str, float],
    pol: str = "p",
    theta_rad: float = 0.0,
    material_scale: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    wavelengths_m = wavelengths_um * 1e-6
    n_matrix = np.vstack([material_n(name, wavelengths_um, material_scale) for name in layer_names])
    thicknesses_m = np.array([thicknesses_um.get(name, 0.0) * 1e-6 for name in layer_names])
    k0 = 2.0 * np.pi / wavelengths_m

    kx = n_matrix[0, :] * np.sin(theta_rad)
    cos_values = np.sqrt(1.0 - (kx[None, :] / n_matrix) ** 2 + 0j)
    cos_values = np.where(np.real(cos_values) < 0.0, -cos_values, cos_values)
    cos_values = np.where(
        (np.abs(np.real(cos_values)) < 1.0e-12) & (np.imag(cos_values) < 0.0),
        -cos_values,
        cos_values,
    )
    q_values = characteristic_admittance(n_matrix, cos_values, pol)

    m11 = np.ones(len(wavelengths_um), dtype=np.complex128)
    m12 = np.zeros(len(wavelengths_um), dtype=np.complex128)
    m21 = np.zeros(len(wavelengths_um), dtype=np.complex128)
    m22 = np.ones(len(wavelengths_um), dtype=np.complex128)

    for layer_idx in range(1, len(layer_names) - 1):
        thickness = float(thicknesses_m[layer_idx])
        if thickness <= 0.0:
            continue
        delta = k0 * n_matrix[layer_idx, :] * thickness * cos_values[layer_idx, :]
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        q_layer = q_values[layer_idx, :]
        a11 = c_delta
        a12 = 1j * s_delta / q_layer
        a21 = 1j * q_layer * s_delta
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


def case_layers(case_name: str) -> list[str]:
    if case_name == "A0_cavity_air":
        return ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
    return ["Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]


def nominal_thicknesses_um(case_name: str) -> dict[str, float]:
    return {
        "Air": 1000.0 if case_name == "A0_cavity_air" else 0.0,
        "HSQ": NOMINAL_NM["HSQ"] / 1000.0,
        "PSS": NOMINAL_NM["PSS"] / 1000.0,
        "SOC": NOMINAL_NM["SOC"] / 1000.0,
        "TiO2": NOMINAL_NM["TiO2"] / 1000.0,
    }


def case_definition(case_name: str) -> dict:
    cases = {
        "A0_cavity_air": {
            "label": "with cavity: fit Air cavity length only",
            "fit_params": ["Air"],
            "bounds": {"Air": (998.5, 1002.5)},
        },
        "B1_film_tio2": {
            "label": "film only: fit TiO2 only",
            "fit_params": ["TiO2"],
            "bounds": {"TiO2": FILM_BOUNDS_NM["TiO2"]},
        },
        "B2_film_soc_tio2": {
            "label": "film only: fit SOC and TiO2",
            "fit_params": ["SOC", "TiO2"],
            "bounds": {"SOC": FILM_BOUNDS_NM["SOC"], "TiO2": FILM_BOUNDS_NM["TiO2"]},
        },
        "B3_film_all": {
            "label": "film only: fit HSQ, PSS, SOC, TiO2",
            "fit_params": ["HSQ", "PSS", "SOC", "TiO2"],
            "bounds": {name: FILM_BOUNDS_NM[name] for name in ["HSQ", "PSS", "SOC", "TiO2"]},
        },
    }
    return cases[case_name]


def default_cases() -> list[str]:
    return ["A0_cavity_air", "B1_film_tio2", "B2_film_soc_tio2", "B3_film_all"]


def param_to_um(name: str, value_fit_unit: float) -> float:
    return float(value_fit_unit) if name == "Air" else float(value_fit_unit) / 1000.0


def param_from_um(name: str, value_um: float) -> float:
    return float(value_um) if name == "Air" else float(value_um) * 1000.0


def apply_params(base_um: dict[str, float], params: list[str], values_fit_unit: np.ndarray) -> dict[str, float]:
    result = dict(base_um)
    for name, value in zip(params, values_fit_unit):
        result[name] = param_to_um(name, float(value))
    return result


def bounds_arrays(case_name: str) -> tuple[np.ndarray, np.ndarray]:
    definition = case_definition(case_name)
    lower = []
    upper = []
    for name in definition["fit_params"]:
        lo, hi = definition["bounds"][name]
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def best_affine(model: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([model, np.ones_like(model)])
    coeff, *_ = np.linalg.lstsq(design, observed, rcond=None)
    return float(coeff[0]), float(coeff[1])


def residual_with_affine(model: np.ndarray, observed: np.ndarray, use_affine: bool) -> np.ndarray:
    if use_affine:
        scale, offset = best_affine(model, observed)
        return scale * model + offset - observed
    return model - observed


def draw_truth_samples(case_name: str, config: ValidationConfig, rng: np.random.Generator) -> list[dict[str, float]]:
    definition = case_definition(case_name)
    nominal_um = nominal_thicknesses_um(case_name)
    samples = []
    if case_name == "A0_cavity_air":
        for value in np.linspace(999.4, 1000.6, config.truth_samples_per_case):
            sample = dict(nominal_um)
            sample["Air"] = float(value)
            samples.append(sample)
        return samples

    for idx in range(config.truth_samples_per_case):
        sample = dict(nominal_um)
        for name in definition["fit_params"]:
            lo, hi = definition["bounds"][name]
            if idx == 0:
                value_nm = NOMINAL_NM[name]
            else:
                value_nm = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
            sample[name] = float(value_nm) / 1000.0
        samples.append(sample)
    return samples


def draw_material_scale(config: ValidationConfig, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    scales = {}
    for name in ["RefReflector", "HSQ", "PSS", "SOC", "TiO2", "Cu"]:
        real_scale = 1.0 + rng.normal(0.0, config.material_real_sigma_fraction)
        imag_scale = 1.0 + rng.normal(0.0, config.material_imag_sigma_fraction)
        scales[name] = (float(real_scale), float(max(0.0, imag_scale)))
    return scales


def perturb_fixed_layers(
    case_name: str,
    truth_um: dict[str, float],
    config: ValidationConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    params = set(case_definition(case_name)["fit_params"])
    out = dict(truth_um)
    for name in ["HSQ", "PSS", "SOC", "TiO2"]:
        if name in params:
            continue
        value_nm = param_from_um(name, out[name])
        value_nm += rng.normal(0.0, config.fixed_layer_sigma_nm)
        lo, hi = FILM_BOUNDS_NM[name]
        out[name] = float(np.clip(value_nm, lo, hi)) / 1000.0
    return out


def draw_wavelength_drift(
    wavelengths_um: np.ndarray,
    config: ValidationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    offset_nm = float(rng.normal(0.0, config.wavelength_offset_sigma_nm))
    scale_ppm = float(rng.normal(0.0, config.wavelength_scale_sigma_ppm))
    center = float(np.mean(wavelengths_um))
    true_wavelengths = center + (wavelengths_um - center) * (1.0 + scale_ppm * 1.0e-6)
    true_wavelengths = true_wavelengths + offset_nm / 1000.0
    return true_wavelengths, offset_nm, scale_ppm


def random_initial_guesses(
    case_name: str,
    config: ValidationConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    lower, upper = bounds_arrays(case_name)
    guesses = []
    for _ in range(max(1, config.multistarts)):
        guesses.append(rng.uniform(lower, upper))
    return guesses


def fit_spectrum_multistart(
    wavelengths_um: np.ndarray,
    observed: np.ndarray,
    case_name: str,
    config: ValidationConfig,
    rng: np.random.Generator,
) -> dict:
    definition = case_definition(case_name)
    params = definition["fit_params"]
    base_um = nominal_thicknesses_um(case_name)
    layers = case_layers(case_name)
    bounds = bounds_arrays(case_name)
    def objective(values: np.ndarray) -> np.ndarray:
        thickness = apply_params(base_um, params, values)
        model = tmm_reflectance(wavelengths_um, layers, thickness)
        return residual_with_affine(model, observed, config.fit_affine_intensity)

    attempts = []
    for p0 in random_initial_guesses(case_name, config, rng):
        result = least_squares(
            objective,
            p0,
            bounds=bounds,
            method="trf",
            max_nfev=config.max_nfev,
            x_scale="jac",
        )
        fit_model = tmm_reflectance(wavelengths_um, layers, apply_params(base_um, params, result.x))
        scale, offset = best_affine(fit_model, observed) if config.fit_affine_intensity else (1.0, 0.0)
        residual = scale * fit_model + offset - observed
        attempts.append(
            {
                "p0": np.asarray(p0, dtype=float),
                "x": np.asarray(result.x, dtype=float),
                "success": bool(result.success),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "scale": float(scale),
                "offset": float(offset),
                "rmse_reflectance": math.sqrt(float(np.mean(residual**2))),
                "model_scaled": scale * fit_model + offset,
            }
        )

    attempts.sort(key=lambda row: row["cost"])
    best = attempts[0]
    cost_cutoff = best["cost"] * (1.0 + config.near_solution_cost_rel_tol) + config.near_solution_cost_abs_tol
    near = [row for row in attempts if row["cost"] <= cost_cutoff]
    near_x = np.vstack([row["x"] for row in near])
    best["all_attempts"] = attempts
    best["near_solution_count"] = len(near)
    best["near_solution_spread"] = {
        name: float(np.max(near_x[:, idx]) - np.min(near_x[:, idx])) for idx, name in enumerate(params)
    }
    return best


def summarize_errors(fit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "case", "noise_sigma_reflectance", "param"]
    for group_key, group in fit_df.groupby(keys, dropna=False):
        err = group["error_nm"].to_numpy(dtype=float)
        rows.append(
            {
                "scenario": group_key[0],
                "case": group_key[1],
                "noise_sigma_reflectance": float(group_key[2]),
                "param": group_key[3],
                "n": int(len(err)),
                "success_rate": float(group["success"].mean()),
                "bias_nm": float(np.mean(err)),
                "MAE_nm": float(np.mean(np.abs(err))),
                "RMSE_nm": math.sqrt(float(np.mean(err**2))),
                "P95Abs_nm": float(np.percentile(np.abs(err), 95)),
                "MaxAbs_nm": float(np.max(np.abs(err))),
                "mean_near_solution_count": float(group["near_solution_count"].mean()),
                "max_equivalent_spread_nm": float(group["equivalent_spread_nm"].max()),
                "mean_abs_wavelength_offset_nm": float(np.mean(np.abs(group["wavelength_offset_nm"]))),
                "mean_abs_wavelength_scale_ppm": float(np.mean(np.abs(group["wavelength_scale_ppm"]))),
            }
        )
    return pd.DataFrame(rows)


def save_error_plot(output_dir: Path, summary: pd.DataFrame) -> str:
    cases = list(summary["case"].drop_duplicates())
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, max(3.2, 2.6 * len(cases))), sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = summary[summary["case"] == case]
        for (scenario, param), part in sub.groupby(["scenario", "param"]):
            part = part.sort_values("noise_sigma_reflectance")
            ax.plot(
                part["noise_sigma_reflectance"],
                part["MAE_nm"],
                marker="o",
                label=f"{scenario}:{param}",
            )
        ax.set_title(case)
        ax.set_ylabel("MAE (nm)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("Additive reflectance noise sigma")
    fig.tight_layout()
    path = output_dir / "robust_error_vs_noise_mae.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_example_plot(
    output_dir: Path,
    wavelengths_um: np.ndarray,
    observed: np.ndarray,
    fitted: np.ndarray,
    case_name: str,
    scenario: str,
) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(wavelengths_um, observed, lw=0.7, label="observed synthetic spectrum")
    ax.plot(wavelengths_um, fitted, "--", lw=1.0, label="best inverse fit")
    ax.set_xlabel("Recorded wavelength (um)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"{case_name} / {scenario}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"example_fit_{case_name}_{scenario}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(output_dir: Path, config: ValidationConfig, summary: pd.DataFrame, plot_paths: dict) -> str:
    lines = [
        "# Robust TMM Inverse Validation",
        "",
        "## What changed from the first baseline",
        "",
        "- Synthetic spectra are generated on a drifted true wavelength axis, but fitted on the recorded wavelength axis.",
        "- Truth generation can perturb material n/k and fixed layer thicknesses; inverse fitting still uses the nominal model.",
        "- Each spectrum is fitted from multiple random initial guesses inside the bounds.",
        "- Near-equivalent solutions are counted to expose multi-solution risk.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| scenario | case | noise | param | n | success | MAE_nm | P95Abs_nm | MaxAbs_nm | equiv_count | max_equiv_spread_nm |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["scenario", "case", "param", "noise_sigma_reflectance"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["case"]),
                    f"{row['noise_sigma_reflectance']:.4g}",
                    str(row["param"]),
                    f"{int(row['n'])}",
                    f"{row['success_rate']:.2f}",
                    f"{row['MAE_nm']:.4g}",
                    f"{row['P95Abs_nm']:.4g}",
                    f"{row['MaxAbs_nm']:.4g}",
                    f"{row['mean_near_solution_count']:.2f}",
                    f"{row['max_equivalent_spread_nm']:.4g}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Output files",
        "",
        f"- metrics: `{output_dir / 'metrics_summary.csv'}`",
        f"- fits: `{output_dir / 'fit_results.csv'}`",
        f"- plot: `{plot_paths['error_vs_noise_mae']}`",
        "",
        "## Interpretation",
        "",
        "This is a stricter synthetic validation, not a direct experimental claim. A large equivalent-solution spread means the spectrum admits several parameter sets with nearly identical residuals.",
    ]
    path = output_dir / "summary_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def make_tmm_observation(
    wavelengths_recorded_um: np.ndarray,
    layers: list[str],
    truth_um: dict[str, float],
    config: ValidationConfig,
    rng: np.random.Generator,
    noise_sigma: float,
) -> tuple[np.ndarray, dict]:
    w_true, offset_nm, scale_ppm = draw_wavelength_drift(wavelengths_recorded_um, config, rng)
    material_scale = draw_material_scale(config, rng)
    clean = tmm_reflectance(w_true, layers, truth_um, material_scale=material_scale)
    source_scale = 1.0 + rng.normal(0.0, config.source_scale_sigma)
    source_offset = rng.normal(0.0, config.source_offset_sigma)
    observed = source_scale * clean + source_offset + rng.normal(0.0, noise_sigma, size=clean.shape)
    meta = {
        "wavelength_offset_nm": offset_nm,
        "wavelength_scale_ppm": scale_ppm,
        "source_scale": float(source_scale),
        "source_offset": float(source_offset),
        "material_scale": material_scale,
    }
    return observed, meta


def run_case_grid(
    config: ValidationConfig,
    output_dir: Path,
    cases: list[str],
    observation_factory: Callable[
        [np.ndarray, list[str], dict[str, float], ValidationConfig, np.random.Generator, float],
        tuple[np.ndarray, dict],
    ],
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(config.random_seed)
    wavelengths_um = wavelength_axis(config)
    fit_rows = []
    example_paths = {}
    for case in cases:
        definition = case_definition(case)
        params = definition["fit_params"]
        layers = case_layers(case)
        truth_samples = draw_truth_samples(case, config, rng)
        for truth_id, base_truth_um in enumerate(truth_samples):
            truth_um = perturb_fixed_layers(case, base_truth_um, config, rng)
            for noise_sigma in config.noise_sigmas_reflectance:
                trials = 1 if noise_sigma == 0.0 else config.mc_trials_per_noise
                for trial in range(trials):
                    observed, meta = observation_factory(wavelengths_um, layers, truth_um, config, rng, noise_sigma)
                    fit = fit_spectrum_multistart(wavelengths_um, observed, case, config, rng)
                    true_values = np.array([param_from_um(name, truth_um[name]) for name in params])
                    for idx, (name, true_value, pred_value) in enumerate(zip(params, true_values, fit["x"])):
                        unit_error_nm = (pred_value - true_value) * 1000.0 if name == "Air" else pred_value - true_value
                        spread = fit["near_solution_spread"].get(name, 0.0)
                        spread_nm = spread * 1000.0 if name == "Air" else spread
                        fit_rows.append(
                            {
                                "scenario": scenario,
                                "case": case,
                                "case_label": definition["label"],
                                "truth_id": truth_id,
                                "trial": trial,
                                "noise_sigma_reflectance": float(noise_sigma),
                                "param": name,
                                "true_value": float(true_value),
                                "pred_value": float(pred_value),
                                "error_nm": float(unit_error_nm),
                                "success": bool(fit["success"]),
                                "cost": float(fit["cost"]),
                                "nfev": int(fit["nfev"]),
                                "rmse_reflectance": float(fit["rmse_reflectance"]),
                                "affine_scale": float(fit["scale"]),
                                "affine_offset": float(fit["offset"]),
                                "near_solution_count": int(fit["near_solution_count"]),
                                "equivalent_spread_nm": float(spread_nm),
                                "wavelength_offset_nm": float(meta.get("wavelength_offset_nm", 0.0)),
                                "wavelength_scale_ppm": float(meta.get("wavelength_scale_ppm", 0.0)),
                                "source_scale": float(meta.get("source_scale", 1.0)),
                                "source_offset": float(meta.get("source_offset", 0.0)),
                            }
                        )
                    if truth_id == 0 and trial == 0 and noise_sigma == config.noise_sigmas_reflectance[-1]:
                        example_paths[f"{scenario}_{case}"] = save_example_plot(
                            output_dir, wavelengths_um, observed, fit["model_scaled"], case, scenario
                        )
    return pd.DataFrame(fit_rows), example_paths


def run_validation(config: ValidationConfig, cases: list[str] | None = None) -> Path:
    output_dir = OUTPUT_ROOT / f"tmm_inverse_validation_robust_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = cases or default_cases()
    fit_df, examples = run_case_grid(config, output_dir, cases, make_tmm_observation, "tmm_robust")
    summary = summarize_errors(fit_df)
    fit_csv = output_dir / "fit_results.csv"
    summary_csv = output_dir / "metrics_summary.csv"
    fit_df.to_csv(fit_csv, index=False, encoding="utf-8-sig", float_format="%.10g")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.10g")
    plot_paths = {
        "error_vs_noise_mae": save_error_plot(output_dir, summary),
        "examples": examples,
    }
    (output_dir / "config_and_outputs.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "fit_results_csv": str(fit_csv),
                "metrics_summary_csv": str(summary_csv),
                "plot_paths": plot_paths,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, config, summary, plot_paths)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY={summary_csv}")
    print(f"REPORT={report_path}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust synthetic TMM inverse validation.")
    parser.add_argument("--cases", nargs="+", default=default_cases())
    parser.add_argument("--mc-trials", type=int, default=3)
    parser.add_argument("--truth-samples", type=int, default=3)
    parser.add_argument("--multistarts", type=int, default=12)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument("--spectral-resolution-nm", type=float, default=0.1)
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.0, 0.002, 0.005])
    parser.add_argument("--wavelength-offset-sigma-nm", type=float, default=0.03)
    parser.add_argument("--wavelength-scale-sigma-ppm", type=float, default=80.0)
    parser.add_argument("--material-real-sigma-fraction", type=float, default=0.003)
    parser.add_argument("--material-imag-sigma-fraction", type=float, default=0.05)
    parser.add_argument("--fixed-layer-sigma-nm", type=float, default=0.5)
    parser.add_argument("--no-affine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ValidationConfig(
        spectral_resolution_nm=float(args.spectral_resolution_nm),
        random_seed=int(args.random_seed),
        noise_sigmas_reflectance=tuple(float(v) for v in args.noise_sigmas),
        mc_trials_per_noise=int(args.mc_trials),
        truth_samples_per_case=int(args.truth_samples),
        multistarts=int(args.multistarts),
        max_nfev=int(args.max_nfev),
        fit_affine_intensity=not bool(args.no_affine),
        wavelength_offset_sigma_nm=float(args.wavelength_offset_sigma_nm),
        wavelength_scale_sigma_ppm=float(args.wavelength_scale_sigma_ppm),
        material_real_sigma_fraction=float(args.material_real_sigma_fraction),
        material_imag_sigma_fraction=float(args.material_imag_sigma_fraction),
        fixed_layer_sigma_nm=float(args.fixed_layer_sigma_nm),
    )
    run_validation(config, list(args.cases))


if __name__ == "__main__":
    main()


