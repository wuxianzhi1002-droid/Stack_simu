from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
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


@dataclass
class ValidationConfig:
    wavelength_start_um: float = 0.2
    wavelength_stop_um: float = 0.6
    spectral_resolution_nm: float = 0.02
    random_seed: int = 20260707
    noise_sigmas_reflectance: tuple[float, ...] = (0.0, 0.001, 0.002, 0.005, 0.01)
    mc_trials_per_noise: int = 4
    truth_samples_per_case: int = 4
    max_nfev: int = 80
    fit_affine_intensity: bool = True
    objective: str = "reflectance_with_optional_affine_scale_offset"
    source_model: str = "main_cavity.py simplified material model"


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
    layer_names: list[str],
    thicknesses_um: dict[str, float],
    pol: str = "p",
    theta_rad: float = 0.0,
) -> np.ndarray:
    wavelengths_m = wavelengths_um * 1e-6
    n_matrix = np.vstack([material_n(name, wavelengths_um) for name in layer_names])
    thicknesses_m = np.array([thicknesses_um.get(name, 0.0) * 1e-6 for name in layer_names])
    n_layer, n_lambda = n_matrix.shape
    k0 = 2.0 * np.pi / wavelengths_m

    kx = n_matrix[0, :] * np.sin(theta_rad)
    cos_values = np.sqrt(1.0 - (kx[None, :] / n_matrix) ** 2 + 0j)
    cos_values = np.where(np.real(cos_values) < 0.0, -cos_values, cos_values)
    cos_values = np.where(
        (np.abs(np.real(cos_values)) < 1e-12) & (np.imag(cos_values) < 0.0),
        -cos_values,
        cos_values,
    )
    q_values = characteristic_admittance(n_matrix, cos_values, pol)

    m11 = np.ones(n_lambda, dtype=np.complex128)
    m12 = np.zeros(n_lambda, dtype=np.complex128)
    m21 = np.zeros(n_lambda, dtype=np.complex128)
    m22 = np.ones(n_lambda, dtype=np.complex128)

    for layer_idx in range(1, n_layer - 1):
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
    r = numerator / denominator
    return np.abs(r) ** 2


def case_layers(case_name: str) -> list[str]:
    if case_name == "A0_cavity_air":
        return ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
    return ["Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]


def nominal_thicknesses_um(case_name: str) -> dict[str, float]:
    values = {
        "Air": 1000.0 if case_name == "A0_cavity_air" else 0.0,
        "HSQ": NOMINAL_NM["HSQ"] / 1000.0,
        "PSS": NOMINAL_NM["PSS"] / 1000.0,
        "SOC": NOMINAL_NM["SOC"] / 1000.0,
        "TiO2": NOMINAL_NM["TiO2"] / 1000.0,
    }
    return values


def case_definition(case_name: str) -> dict:
    cases = {
        "A0_cavity_air": {
            "label": "with cavity: fit Air cavity length only",
            "fit_params": ["Air"],
            "bounds": {"Air": (999.8, 1001.2)},
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


def param_to_um(name: str, value_fit_unit: float) -> float:
    if name == "Air":
        return float(value_fit_unit)
    return float(value_fit_unit) / 1000.0


def param_from_um(name: str, value_um: float) -> float:
    if name == "Air":
        return float(value_um)
    return float(value_um) * 1000.0


def apply_params(base_um: dict[str, float], params: list[str], values_fit_unit: np.ndarray) -> dict[str, float]:
    result = dict(base_um)
    for name, value in zip(params, values_fit_unit):
        result[name] = param_to_um(name, float(value))
    return result


def best_affine(model: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([model, np.ones_like(model)])
    coeff, *_ = np.linalg.lstsq(design, observed, rcond=None)
    return float(coeff[0]), float(coeff[1])


def residual_with_affine(model: np.ndarray, observed: np.ndarray, use_affine: bool) -> np.ndarray:
    if use_affine:
        scale, offset = best_affine(model, observed)
        return scale * model + offset - observed
    return model - observed


def fit_spectrum(
    wavelengths_um: np.ndarray,
    observed: np.ndarray,
    case_name: str,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    config: ValidationConfig,
) -> dict:
    definition = case_definition(case_name)
    params = definition["fit_params"]
    base_um = nominal_thicknesses_um(case_name)
    layers = case_layers(case_name)

    def objective(values: np.ndarray) -> np.ndarray:
        thickness = apply_params(base_um, params, values)
        model = tmm_reflectance(wavelengths_um, layers, thickness)
        return residual_with_affine(model, observed, config.fit_affine_intensity)

    result = least_squares(
        objective,
        p0,
        bounds=bounds,
        method="trf",
        max_nfev=config.max_nfev,
        x_scale="jac",
    )
    fitted_thickness = apply_params(base_um, params, result.x)
    fitted_model = tmm_reflectance(wavelengths_um, layers, fitted_thickness)
    scale, offset = best_affine(fitted_model, observed) if config.fit_affine_intensity else (1.0, 0.0)
    residual = scale * fitted_model + offset - observed
    return {
        "x": result.x,
        "success": bool(result.success),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "scale": float(scale),
        "offset": float(offset),
        "rmse_reflectance": math.sqrt(float(np.mean(residual**2))),
        "model_fit": fitted_model,
        "model_scaled": scale * fitted_model + offset,
    }


def make_truth_samples(case_name: str, config: ValidationConfig, rng: np.random.Generator) -> list[dict[str, float]]:
    definition = case_definition(case_name)
    params = definition["fit_params"]
    nominal_um = nominal_thicknesses_um(case_name)
    samples: list[dict[str, float]] = []
    if case_name == "A0_cavity_air":
        values = np.linspace(1000.0, 1001.0, config.truth_samples_per_case)
        for value in values:
            sample = dict(nominal_um)
            sample["Air"] = float(value)
            samples.append(sample)
        return samples

    samples.append(dict(nominal_um))
    while len(samples) < config.truth_samples_per_case:
        sample = dict(nominal_um)
        for name in params:
            lo, hi = definition["bounds"][name]
            span = min((hi - lo) * 0.20, 10.0)
            nominal = NOMINAL_NM[name]
            value_nm = float(np.clip(nominal + rng.uniform(-span, span), lo, hi))
            sample[name] = value_nm / 1000.0
        samples.append(sample)
    return samples


def initial_guess(case_name: str, observed: np.ndarray, wavelengths_um: np.ndarray, config: ValidationConfig) -> np.ndarray:
    definition = case_definition(case_name)
    params = definition["fit_params"]
    if case_name == "A0_cavity_air":
        # Coarse one-dimensional initialization avoids fitting the wrong long-cavity fringe.
        grid = np.linspace(999.8, 1001.2, 71)
        stride = max(1, len(wavelengths_um) // 1200)
        w = wavelengths_um[::stride]
        y = observed[::stride]
        errors = []
        base_um = nominal_thicknesses_um(case_name)
        layers = case_layers(case_name)
        for value in grid:
            thickness = dict(base_um)
            thickness["Air"] = float(value)
            model = tmm_reflectance(w, layers, thickness)
            errors.append(float(np.mean(residual_with_affine(model, y, config.fit_affine_intensity) ** 2)))
        return np.array([float(grid[int(np.argmin(errors))])])
    return np.array([NOMINAL_NM[name] for name in params], dtype=np.float64)


def bounds_arrays(case_name: str) -> tuple[np.ndarray, np.ndarray]:
    definition = case_definition(case_name)
    lower = []
    upper = []
    for name in definition["fit_params"]:
        lo, hi = definition["bounds"][name]
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def jacobian_diagnostics(
    wavelengths_um: np.ndarray,
    case_name: str,
    config: ValidationConfig,
) -> dict:
    definition = case_definition(case_name)
    params = definition["fit_params"]
    nominal = nominal_thicknesses_um(case_name)
    layers = case_layers(case_name)
    base = tmm_reflectance(wavelengths_um, layers, nominal)
    columns = []
    for name in params:
        step = 0.001 if name == "Air" else 0.1  # 1 nm for air, 0.1 nm for films.
        plus = dict(nominal)
        minus = dict(nominal)
        plus[name] = plus[name] + param_to_um(name, step)
        minus[name] = minus[name] - param_to_um(name, step)
        derivative = (
            tmm_reflectance(wavelengths_um, layers, plus)
            - tmm_reflectance(wavelengths_um, layers, minus)
        ) / (2.0 * step)
        columns.append(derivative)
    j = np.column_stack(columns)
    s = np.linalg.svd(j, full_matrices=False, compute_uv=False)
    cond = float(s[0] / s[-1]) if len(s) and s[-1] > 0 else float("inf")
    if len(params) == 1:
        corr = np.ones((1, 1), dtype=float)
    else:
        corr = np.corrcoef(j.T)
    return {
        "params": params,
        "condition_number": cond,
        "singular_values": s.tolist(),
        "correlation": corr,
        "nominal_reflectance": base,
    }


def summarize_errors(fit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = fit_df.groupby(["case", "noise_sigma_reflectance", "param"], dropna=False)
    for (case, noise, param), group in grouped:
        err = group["error_nm"].to_numpy(dtype=float)
        rows.append(
            {
                "case": case,
                "noise_sigma_reflectance": float(noise),
                "param": param,
                "n": int(len(err)),
                "success_rate": float(group["success"].mean()),
                "bias_nm": float(np.mean(err)),
                "MAE_nm": float(np.mean(np.abs(err))),
                "RMSE_nm": math.sqrt(float(np.mean(err**2))),
                "P95Abs_nm": float(np.percentile(np.abs(err), 95)),
                "MaxAbs_nm": float(np.max(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def save_example_plot(
    output_dir: Path,
    wavelengths_um: np.ndarray,
    observed: np.ndarray,
    clean: np.ndarray,
    fitted: np.ndarray,
    case_name: str,
    noise: float,
) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(wavelengths_um, clean, lw=1.0, label="clean TMM")
    ax.plot(wavelengths_um, observed, lw=0.7, alpha=0.65, label="noisy observation")
    ax.plot(wavelengths_um, fitted, "--", lw=1.0, label="fit")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"{case_name}: example fit, noise sigma={noise:g}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"example_fit_{case_name}_noise_{noise:g}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_error_plot(output_dir: Path, summary: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for ax, case in zip(axes, sorted(summary["case"].unique())):
        sub = summary[summary["case"] == case]
        for param in sub["param"].unique():
            part = sub[sub["param"] == param].sort_values("noise_sigma_reflectance")
            ax.plot(part["noise_sigma_reflectance"], part["MAE_nm"], marker="o", label=param)
        ax.set_title(case)
        ax.set_xlabel("Additive reflectance noise sigma")
        ax.set_ylabel("MAE (nm)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "error_vs_noise_mae.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_correlation_plots(output_dir: Path, diagnostics: dict[str, dict]) -> dict[str, str]:
    paths = {}
    for case, diag in diagnostics.items():
        params = diag["params"]
        corr = np.asarray(diag["correlation"], dtype=float)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(params)), params, rotation=45, ha="right")
        ax.set_yticks(range(len(params)), params)
        for i in range(len(params)):
            for j in range(len(params)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"{case}\ncond={diag['condition_number']:.2e}")
        fig.colorbar(im, ax=ax, shrink=0.82)
        fig.tight_layout()
        path = output_dir / f"correlation_{case}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths[case] = str(path)
    return paths


def write_report(output_dir: Path, config: ValidationConfig, summary: pd.DataFrame, diagnostics: dict[str, dict]) -> str:
    lines = [
        "# TMM Inverse Validation Summary",
        "",
        "## Scope",
        "",
        "- Forward model: coherent TMM with the simplified material model from `main_cavity.py`.",
        "- Branch A keeps `RefReflector / Air / films / Cu` and fits the 1 mm air cavity.",
        "- Branch B removes the air cavity and fits film thickness from `Air / films / Cu` reflectance.",
        "- Optional affine intensity scale/offset is fitted analytically inside each spectral residual.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| case | noise_sigma_R | param | n | success | MAE_nm | RMSE_nm | P95Abs_nm | MaxAbs_nm | bias_nm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["case", "param", "noise_sigma_reflectance"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    f"{row['noise_sigma_reflectance']:.4g}",
                    str(row["param"]),
                    f"{int(row['n'])}",
                    f"{row['success_rate']:.2f}",
                    f"{row['MAE_nm']:.4g}",
                    f"{row['RMSE_nm']:.4g}",
                    f"{row['P95Abs_nm']:.4g}",
                    f"{row['MaxAbs_nm']:.4g}",
                    f"{row['bias_nm']:.4g}",
                ]
            )
            + " |"
        )
    lines += ["", "## Identifiability Diagnostics", ""]
    for case, diag in diagnostics.items():
        lines.append(
            f"- `{case}` params={diag['params']}, condition_number={diag['condition_number']:.3e}"
        )
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- Use MAE/P95Abs for practical accuracy; MaxAbs is mainly an outlier-risk indicator.",
        "- High condition number or near +/-1 parameter correlations indicate non-identifiability.",
        "- Film-only results are the main reference for thin-film metrology; the cavity case is a bridge to the existing long-cavity model.",
    ]
    path = output_dir / "summary_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run_validation(config: ValidationConfig) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"tmm_inverse_validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    wavelengths_um = np.linspace(
        config.wavelength_start_um,
        config.wavelength_stop_um,
        int(round((config.wavelength_stop_um - config.wavelength_start_um) * 1000.0 / config.spectral_resolution_nm)) + 1,
    )
    rng = np.random.default_rng(config.random_seed)
    cases = ["A0_cavity_air", "B1_film_tio2", "B2_film_soc_tio2", "B3_film_all"]
    diagnostics = {case: jacobian_diagnostics(wavelengths_um, case, config) for case in cases}

    fit_rows = []
    example_paths = {}
    for case in cases:
        definition = case_definition(case)
        params = definition["fit_params"]
        layers = case_layers(case)
        lower, upper = bounds_arrays(case)
        truth_samples = make_truth_samples(case, config, rng)
        for truth_id, truth_um in enumerate(truth_samples):
            clean = tmm_reflectance(wavelengths_um, layers, truth_um)
            for noise in config.noise_sigmas_reflectance:
                trials = 1 if noise == 0.0 else config.mc_trials_per_noise
                for trial in range(trials):
                    observed = clean + rng.normal(0.0, noise, size=clean.shape)
                    p0 = initial_guess(case, observed, wavelengths_um, config)
                    fit = fit_spectrum(wavelengths_um, observed, case, p0, (lower, upper), config)
                    true_values = np.array([param_from_um(name, truth_um[name]) for name in params])
                    for name, true_value, pred_value in zip(params, true_values, fit["x"]):
                        unit_error_nm = (pred_value - true_value) * 1000.0 if name == "Air" else pred_value - true_value
                        fit_rows.append(
                            {
                                "case": case,
                                "case_label": definition["label"],
                                "truth_id": truth_id,
                                "trial": trial,
                                "noise_sigma_reflectance": float(noise),
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
                            }
                        )
                    if truth_id == 0 and trial == 0 and noise in (0.0, 0.005):
                        example_paths[f"{case}_noise_{noise:g}"] = save_example_plot(
                            output_dir, wavelengths_um, observed, clean, fit["model_scaled"], case, noise
                        )

    fit_df = pd.DataFrame(fit_rows)
    summary = summarize_errors(fit_df)
    fit_csv = output_dir / "fit_results.csv"
    summary_csv = output_dir / "metrics_summary.csv"
    fit_df.to_csv(fit_csv, index=False, encoding="utf-8-sig", float_format="%.10g")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.10g")

    plot_paths = {
        "error_vs_noise_mae": save_error_plot(output_dir, summary),
        "correlations": save_correlation_plots(output_dir, diagnostics),
        "examples": example_paths,
    }
    diagnostics_json = {
        case: {
            "params": diag["params"],
            "condition_number": diag["condition_number"],
            "singular_values": diag["singular_values"],
            "correlation": np.asarray(diag["correlation"]).tolist(),
        }
        for case, diag in diagnostics.items()
    }
    payload = {
        "config": asdict(config),
        "fit_results_csv": str(fit_csv),
        "metrics_summary_csv": str(summary_csv),
        "diagnostics": diagnostics_json,
        "plot_paths": plot_paths,
    }
    (output_dir / "config_and_outputs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path = write_report(output_dir, config, summary, diagnostics_json)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY={summary_csv}")
    print(f"REPORT={report_path}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic TMM inverse validation for cavity and film-only stacks.")
    parser.add_argument("--mc-trials", type=int, default=4)
    parser.add_argument("--truth-samples", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.0, 0.001, 0.002, 0.005, 0.01])
    parser.add_argument("--no-affine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ValidationConfig(
        random_seed=args.random_seed,
        noise_sigmas_reflectance=tuple(float(v) for v in args.noise_sigmas),
        mc_trials_per_noise=int(args.mc_trials),
        truth_samples_per_case=int(args.truth_samples),
        max_nfev=int(args.max_nfev),
        fit_affine_intensity=not bool(args.no_affine),
    )
    run_validation(config)


if __name__ == "__main__":
    main()
