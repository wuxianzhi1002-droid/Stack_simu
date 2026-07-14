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
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"
DEFAULT_INPUT_DIR = OUTPUT_ROOT / "dynamic_stackrt_lockin_v2"

PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2"]
# Fit units: Air in um, films in nm.
NOMINAL = {
    "Air": 1000.0,
    "HSQ": 40.0,
    "PSS": 5.0,
    "SOC": 50.0,
    "TiO2": 20.0,
}
BOUNDS = {
    "Air": (995.0, 1005.0),
    "HSQ": (20.0, 60.0),
    "PSS": (1.0, 20.0),
    "SOC": (30.0, 80.0),
    "TiO2": (5.0, 60.0),
}
PRIOR_SIGMA = {
    "Air": 2.0,
    "HSQ": 10.0,
    "PSS": 5.0,
    "SOC": 10.0,
    "TiO2": 10.0,
}
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]


@dataclass
class FitConfig:
    input_npz: str
    amplitude_nm: float = 1.0
    wavelength_min_nm: float = 220.0
    wavelength_max_nm: float = 580.0
    stride: int = 10
    derivative_step_um: float = 0.001
    multistarts: int = 8
    max_nfev: int = 160
    random_seed: int = 20260713
    use_prior: bool = True
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
    wavelengths_m = wavelengths_um * 1e-6
    n_matrix = np.vstack([material_n(name, wavelengths_um) for name in LAYER_NAMES])
    thicknesses_m = np.array([thicknesses_um.get(name, 0.0) * 1e-6 for name in LAYER_NAMES])
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

    for layer_idx in range(1, len(LAYER_NAMES) - 1):
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


def find_latest_npz(input_dir: Path) -> Path:
    files = sorted(input_dir.glob("dynamic_spectra_*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No dynamic_spectra_*.npz found in {input_dir}")
    return files[-1]


def fit_vector_to_um(values: np.ndarray) -> dict[str, float]:
    return {
        "Air": float(values[0]),
        "HSQ": float(values[1]) / 1000.0,
        "PSS": float(values[2]) / 1000.0,
        "SOC": float(values[3]) / 1000.0,
        "TiO2": float(values[4]) / 1000.0,
    }


def initial_vector() -> np.ndarray:
    return np.array([NOMINAL[name] for name in PARAMS], dtype=float)


def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([BOUNDS[name][0] for name in PARAMS], dtype=float)
    upper = np.array([BOUNDS[name][1] for name in PARAMS], dtype=float)
    return lower, upper


def dI_dair_model(wavelengths_um: np.ndarray, values: np.ndarray, step_um: float) -> np.ndarray:
    plus = np.asarray(values, dtype=float).copy()
    minus = np.asarray(values, dtype=float).copy()
    plus[0] += step_um
    minus[0] -= step_um
    return (
        tmm_reflectance(wavelengths_um, fit_vector_to_um(plus))
        - tmm_reflectance(wavelengths_um, fit_vector_to_um(minus))
    ) / (2.0 * step_um)


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
    idx = np.where(mask)[0][:: max(1, int(config.stride))]
    if len(idx) < len(PARAMS) * 4:
        raise ValueError("Too few wavelength samples after mask/stride. Reduce --stride or widen wavelength range.")

    amplitude_um = config.amplitude_nm / 1000.0
    if amplitude_um <= 0.0:
        raise ValueError("--amplitude-nm must be positive.")

    return {
        "wavelengths_um_full": wavelengths_um,
        "I_meas_full": np.mean(spectra, axis=0),
        "dIdL_x_full": lockin_x / amplitude_um,
        "dIdL_y_full": lockin_y / amplitude_um,
        "dIdL_r_full": lockin_r / amplitude_um,
        "idx": idx,
        "wavelengths_um": wavelengths_um[idx],
        "I_meas": np.mean(spectra, axis=0)[idx],
        "dIdL_meas": (lockin_x / amplitude_um)[idx],
    }


def make_residual(
    wavelengths_um: np.ndarray,
    I_meas: np.ndarray,
    dIdL_meas: np.ndarray,
    config: FitConfig,
    mode: str,
):
    sigma_i = robust_sigma(I_meas, floor=1.0e-4)
    sigma_d = robust_sigma(dIdL_meas, floor=1.0e-3)
    prior_sigma = np.array([PRIOR_SIGMA[name] for name in PARAMS], dtype=float)
    nominal = initial_vector()

    def residual(values: np.ndarray) -> np.ndarray:
        model_i = tmm_reflectance(wavelengths_um, fit_vector_to_um(values))
        model_d = dI_dair_model(wavelengths_um, values, config.derivative_step_um)
        blocks = []
        if mode in {"I", "joint"}:
            blocks.append((model_i - I_meas) / sigma_i)
        if mode in {"D", "joint"}:
            blocks.append((model_d - dIdL_meas) / sigma_d)
        if config.use_prior:
            blocks.append((values - nominal) / prior_sigma)
        return np.concatenate(blocks)

    return residual, {"sigma_I": sigma_i, "sigma_dIdL": sigma_d}


def random_initial_guesses(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    lower, upper = bounds_arrays()
    guesses = [initial_vector()]
    for _ in range(max(0, count - 1)):
        guesses.append(rng.uniform(lower, upper))
    return guesses


def fit_mode(measurement: dict, config: FitConfig, mode: str, rng: np.random.Generator) -> dict:
    residual, scales = make_residual(
        measurement["wavelengths_um"],
        measurement["I_meas"],
        measurement["dIdL_meas"],
        config,
        mode,
    )
    lower, upper = bounds_arrays()
    attempts = []
    for guess in random_initial_guesses(rng, config.multistarts):
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
        attempts.append(
            {
                "x0": np.asarray(guess, dtype=float),
                "x": np.asarray(result.x, dtype=float),
                "cost": float(result.cost),
                "rmse_normalized": math.sqrt(float(np.mean(res**2))),
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "optimality": float(result.optimality),
            }
        )
    attempts.sort(key=lambda row: row["cost"])
    best = attempts[0]
    values = best["x"]
    model_i = tmm_reflectance(measurement["wavelengths_um"], fit_vector_to_um(values))
    model_d = dI_dair_model(measurement["wavelengths_um"], values, config.derivative_step_um)
    best.update(
        {
            "mode": mode,
            "model_I": model_i,
            "model_dIdL": model_d,
            "rmse_I": math.sqrt(float(np.mean((model_i - measurement["I_meas"]) ** 2))),
            "rmse_dIdL": math.sqrt(float(np.mean((model_d - measurement["dIdL_meas"]) ** 2))),
            "scales": scales,
            "attempts": attempts,
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
            "rmse_normalized": fit["rmse_normalized"],
            "rmse_I": fit["rmse_I"],
            "rmse_dIdL": fit["rmse_dIdL"],
            "nfev": fit["nfev"],
            "optimality": fit["optimality"],
            "condition_number": diag[mode]["condition_number"],
        }
        for name, value in zip(PARAMS, fit["x"]):
            row[f"fit_{name}_{'um' if name == 'Air' else 'nm'}"] = value
            row[f"error_{name}_{'um' if name == 'Air' else 'nm'}"] = value - NOMINAL[name]
        rows.append(row)
    path = output_dir / "fit_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
    return path


def save_multistart_results(output_dir: Path, fits: dict[str, dict]) -> Path:
    rows = []
    for mode, fit in fits.items():
        for idx, attempt in enumerate(fit["attempts"]):
            row = {
                "mode": mode,
                "rank": idx + 1,
                "success": attempt["success"],
                "cost": attempt["cost"],
                "rmse_normalized": attempt["rmse_normalized"],
                "nfev": attempt["nfev"],
                "optimality": attempt["optimality"],
                "message": attempt["message"],
            }
            for name, value in zip(PARAMS, attempt["x"]):
                row[f"fit_{name}_{'um' if name == 'Air' else 'nm'}"] = value
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
        "# TMM Joint Inversion With Lock-in dI/dL",
        "",
        "## Method",
        "",
        "- Input channels: `I(lambda)` from mean dynamic spectra and `dI/dL(lambda)` from `lockin_1f_X / A_um`.",
        "- Forward model: coherent TMM with `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`.",
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

    output_dir = OUTPUT_ROOT / f"tmm_joint_inversion_lockin_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)

    measurement = load_measurement(input_npz, config)
    rng = np.random.default_rng(config.random_seed)
    fits = {mode: fit_mode(measurement, config, mode, rng) for mode in modes}
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
                "nominal_fit_units": NOMINAL,
                "bounds_fit_units": BOUNDS,
                "prior_sigma_fit_units": PRIOR_SIGMA,
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
    parser = argparse.ArgumentParser(description="Joint TMM inversion using I(lambda) and lock-in dI/dL(lambda).")
    parser.add_argument("--input-npz", type=str, default=None, help="Input dynamic_spectra_*.npz. Defaults to latest in dynamic_stackrt_lockin_v2.")
    parser.add_argument("--amplitude-nm", type=float, default=1.0, help="Height/air-gap modulation amplitude used to compute lockin_1f_X / A_um.")
    parser.add_argument("--wavelength-min-nm", type=float, default=220.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=10, help="Use every Nth wavelength sample for fitting. Use 1 for full spectrum.")
    parser.add_argument("--derivative-step-um", type=float, default=0.001)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=160)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--no-prior", action="store_true", help="Disable soft nominal priors.")
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--modes", nargs="+", choices=["I", "D", "joint"], default=["I", "D", "joint"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_npz = args.input_npz
    if input_npz is None:
        input_npz = str(find_latest_npz(DEFAULT_INPUT_DIR))
    config = FitConfig(
        input_npz=input_npz,
        amplitude_nm=float(args.amplitude_nm),
        wavelength_min_nm=float(args.wavelength_min_nm),
        wavelength_max_nm=float(args.wavelength_max_nm),
        stride=int(args.stride),
        derivative_step_um=float(args.derivative_step_um),
        multistarts=int(args.multistarts),
        max_nfev=int(args.max_nfev),
        random_seed=int(args.random_seed),
        use_prior=not bool(args.no_prior),
        loss=str(args.loss),
    )
    run(config, list(args.modes))


if __name__ == "__main__":
    main()
