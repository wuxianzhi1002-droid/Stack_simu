import csv
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LUMERICAL_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
if LUMERICAL_PATH.exists():
    if str(LUMERICAL_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_PATH))
    os.environ["PATH"] += os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    lumapi = None

try:
    from solve_npz_fft import FFTSolver
except ImportError:
    FFTSolver = None


@dataclass(frozen=True)
class FilmLayerConfig:
    name: str
    thickness_param: str
    nominal_nm: float
    uncertainty_nm: float = 10.0


@dataclass(frozen=True)
class ValidationConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    lumerical_file_path: Optional[Path] = None
    model_name: str = "PSS_TIO2_MODEL"
    output_root: Path = Path("./0614")
    wavelength_key: str = "wavelengths"
    spectra_key: str = "spectra"

    wavelength_start_um: float = 0.2
    wavelength_stop_um: float = 0.6
    spectral_resolution_nm: float = 0.02
    angle_axis_deg: Tuple[float, ...] = (0.0,)
    polarization: str = "p"

    cavity_layer_name: str = "Air"
    cavity_thickness_param_name: str = "stackrt_thicknesses[Air]"
    cavity_unit: str = "um"
    cavity_min_um: float = 1000.0
    cavity_max_um: float = 1200.0
    num_cavity_points: int = 21

    film_layers: Tuple[FilmLayerConfig, ...] = (
        FilmLayerConfig("HSQ", "stackrt_thicknesses[HSQ]", 40.0, 10.0),
        FilmLayerConfig("PSS", "stackrt_thicknesses[PSS]", 5.0, 10.0),
        FilmLayerConfig("SOC", "stackrt_thicknesses[SOC]", 50.0, 10.0),
        FilmLayerConfig("TiO2", "stackrt_thicknesses[TiO2]", 20.0, 10.0),
    )

    num_processes: int = 20
    film_uncertainty_nm: float = 10.0
    film_sigma_nm: float = 10.0 / 3.0
    random_seed: int = 20260613

    prior_weight: float = 3.0
    fit_l_window_um: float = 0.5
    max_nfev_fixed: int = 80
    max_nfev_prior: int = 150

    zero_pad_factor: int = 8
    fft_ignore_dc_bins: int = 50
    fft_peak_height_ratio: float = 0.2
    fft_peak_distance_bins: int = 100
    fft_peak_type: str = "dominant"

    cache_round_digits: int = 6
    use_interpolated_forward_model: bool = False


CONFIG = ValidationConfig()


def config_to_json_dict(config: ValidationConfig) -> Dict[str, Any]:
    data = asdict(config)
    for key in ("project_root", "output_root", "lumerical_file_path"):
        if data[key] is not None:
            data[key] = str(data[key])
    return data


def normalize_spectrum(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    mean = float(np.nanmean(values[finite]))
    std = float(np.nanstd(values[finite]))
    return (values - mean) / (std + 1e-12)


def build_axes(config: ValidationConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    span_nm = (config.wavelength_stop_um - config.wavelength_start_um) * 1000.0
    n_wavelength = int(round(span_nm / config.spectral_resolution_nm)) + 1
    wavelengths_um = np.linspace(config.wavelength_start_um, config.wavelength_stop_um, n_wavelength)
    freqs_hz = 3e8 / (wavelengths_um * 1e-6)
    cavity_axis_um = np.linspace(config.cavity_min_um, config.cavity_max_um, config.num_cavity_points)
    return wavelengths_um, freqs_hz, cavity_axis_um


def stack_layers(config: ValidationConfig, cavity_um: float, film_nm: Sequence[float]) -> List[Tuple[str, float]]:
    film_map_um = {layer.name: float(value_nm) * 1e-3 for layer, value_nm in zip(config.film_layers, film_nm)}
    return [
        ("RefReflector", 0.0),
        ("Air", float(cavity_um)),
        ("HSQ", film_map_um.get("HSQ", 0.040)),
        ("PSS", film_map_um.get("PSS", 0.005)),
        ("SOC", film_map_um.get("SOC", 0.050)),
        ("TiO2", film_map_um.get("TiO2", 0.020)),
        ("Cu", 0.0),
    ]


def n_matrix_from_layers(
    layers: Sequence[Tuple[str, float]],
    wavelengths_um: np.ndarray,
    freqs_hz: np.ndarray,
    fdtd: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    n_matrix = np.zeros((len(layers), len(freqs_hz)), dtype=complex)
    thicknesses_m: List[float] = []
    cu_n_k = (1.1 + 2.5j) * np.ones_like(wavelengths_um, dtype=complex)
    if fdtd is not None:
        try:
            cu_n_k = np.asarray(fdtd.getindex("Cu (Copper) - Palik", freqs_hz), dtype=complex).flatten()
        except Exception:
            pass

    for i, (mat, thick_um) in enumerate(layers):
        thicknesses_m.append(float(thick_um) * 1e-6)
        if mat == "RefReflector":
            n_matrix[i, :] = 5.8284
        elif mat == "Air":
            n_matrix[i, :] = 1.0
        elif mat == "HSQ":
            n_matrix[i, :] = 1.41
        elif mat == "PSS":
            n_matrix[i, :] = 1.50 + 0.05j
        elif mat == "SOC":
            n_matrix[i, :] = 1.55 + 0.005 / (wavelengths_um**2)
        elif mat == "TiO2":
            n_matrix[i, :] = 2.4 + 0.02 / (wavelengths_um**2)
        elif mat == "Cu":
            n_matrix[i, :] = cu_n_k
        else:
            n_matrix[i, :] = 1.5
    return n_matrix, np.asarray(thicknesses_m, dtype=float)


class ForwardSpectrumModel:
    def evaluate(self, cavity_um: float, film_nm: Sequence[float]) -> np.ndarray:
        raise NotImplementedError


class LumapiForwardModel(ForwardSpectrumModel):
    def __init__(self, config: ValidationConfig, wavelengths_um: np.ndarray, freqs_hz: np.ndarray, fdtd: Any):
        self.config = config
        self.wavelengths_um = wavelengths_um
        self.freqs_hz = freqs_hz
        self.fdtd = fdtd
        self.result_key = "Rp" if config.polarization.lower() == "p" else "Rs"

    def evaluate(self, cavity_um: float, film_nm: Sequence[float]) -> np.ndarray:
        layers = stack_layers(self.config, cavity_um, film_nm)
        n_matrix, thicknesses_m = n_matrix_from_layers(layers, self.wavelengths_um, self.freqs_hz, self.fdtd)
        res = self.fdtd.stackrt(n_matrix, thicknesses_m, self.freqs_hz, float(self.config.angle_axis_deg[0]))
        return np.real(np.asarray(res[self.result_key], dtype=float).reshape(-1))


class InterpolatedForwardModel(ForwardSpectrumModel):
    def __init__(self) -> None:
        raise NotImplementedError("Interpolated surrogate model is reserved for later acceleration.")


class LumapiSpectrumCache:
    def __init__(self, forward_model: ForwardSpectrumModel, cache_path: Path, round_digits: int) -> None:
        self.forward_model = forward_model
        self.cache_path = cache_path
        self.round_digits = round_digits
        self.cache: Dict[Tuple[float, Tuple[float, ...]], np.ndarray] = {}
        self.load_cache()

    def _key(self, cavity_um: float, film_nm: Sequence[float]) -> Tuple[float, Tuple[float, ...]]:
        return (
            round(float(cavity_um), self.round_digits),
            tuple(round(float(v), self.round_digits) for v in film_nm),
        )

    def get_spectrum(self, cavity_um: float, film_nm: Sequence[float]) -> np.ndarray:
        key = self._key(cavity_um, film_nm)
        if key not in self.cache:
            self.cache[key] = self.forward_model.evaluate(cavity_um, film_nm)
        return self.cache[key]

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self.cache, f)

    def load_cache(self) -> None:
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                self.cache = pickle.load(f)


def open_fdtd(config: ValidationConfig) -> Any:
    if lumapi is None:
        raise RuntimeError("lumapi is not available. Check Lumerical installation and Python path.")
    fdtd = lumapi.FDTD(hide=True)
    if config.lumerical_file_path is not None and config.lumerical_file_path.exists():
        fdtd.load(str(config.lumerical_file_path))
    return fdtd


def save_npz(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def generate_lumapi_dataset(config: ValidationConfig, run_dir: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = run_dir / f"prior_joint_lumapi_dataset_{timestamp}.npz"
    checkpoint_path = run_dir / f"prior_joint_lumapi_dataset_{timestamp}_checkpoint.npz"
    failed_cases: List[Dict[str, Any]] = []
    wavelengths_um, freqs_hz, cavity_axis_um = build_axes(config)
    film_nominal_nm = np.asarray([layer.nominal_nm for layer in config.film_layers], dtype=float)

    rng = np.random.default_rng(config.random_seed)
    process_delta_nm = rng.uniform(
        -config.film_uncertainty_nm,
        config.film_uncertainty_nm,
        size=(config.num_processes, len(config.film_layers)),
    )
    process_delta_nm[0, :] = 0.0
    film_true_nm = film_nominal_nm[None, :] + process_delta_nm
    spectra = np.full((config.num_processes, config.num_cavity_points, wavelengths_um.size), np.nan)

    fdtd = open_fdtd(config)
    forward = LumapiForwardModel(config, wavelengths_um, freqs_hz, fdtd)
    try:
        for p in range(config.num_processes):
            print(f"[Dataset] process {p + 1}/{config.num_processes}, film_delta_nm={process_delta_nm[p].tolist()}")
            for c, cavity_um in enumerate(cavity_axis_um):
                try:
                    spectra[p, c, :] = forward.evaluate(float(cavity_um), film_true_nm[p])
                    if not np.all(np.isfinite(spectra[p, c, :])):
                        raise ValueError("spectrum contains non-finite values")
                except Exception as exc:
                    failed_cases.append(
                        {
                            "stage": "dataset",
                            "process_id": p,
                            "cavity_id": c,
                            "cavity_um": float(cavity_um),
                            "film_nm": film_true_nm[p].tolist(),
                            "error": repr(exc),
                        }
                    )
                    print(f"[Dataset] failed process={p}, cavity={c}: {exc}")

            save_npz(
                checkpoint_path,
                wavelengths=wavelengths_um,
                cavity_axis_um=cavity_axis_um,
                process_delta_nm=process_delta_nm,
                film_nominal_nm=film_nominal_nm,
                film_true_nm=film_true_nm,
                spectra=spectra,
                model_name=np.array(config.model_name),
                timestamp=np.array(timestamp),
                config_json=np.array(json.dumps(config_to_json_dict(config), ensure_ascii=False, indent=2)),
            )
            print(f"[Dataset] checkpoint saved: {checkpoint_path}")
    finally:
        try:
            fdtd.close()
        except Exception:
            pass

    save_npz(
        dataset_path,
        wavelengths=wavelengths_um,
        cavity_axis_um=cavity_axis_um,
        process_delta_nm=process_delta_nm,
        film_nominal_nm=film_nominal_nm,
        film_true_nm=film_true_nm,
        spectra=spectra,
        model_name=np.array(config.model_name),
        timestamp=np.array(timestamp),
        config_json=np.array(json.dumps(config_to_json_dict(config), ensure_ascii=False, indent=2)),
    )
    print(f"[Dataset] final saved: {dataset_path}")
    return dataset_path, failed_cases


def solve_fft_dataset(dataset_npz_path: Path, config: ValidationConfig, run_dir: Path) -> Path:
    if FFTSolver is None:
        raise RuntimeError("Could not import FFTSolver from solve_npz_fft.py.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fft_path = run_dir / f"prior_joint_lumapi_fft_solved_{timestamp}.npz"
    data = np.load(dataset_npz_path, allow_pickle=True)
    wavelengths_um = np.asarray(data["wavelengths"], dtype=float)
    spectra = np.asarray(data["spectra"], dtype=float)
    n_process, n_cavity, _ = spectra.shape

    l_fft_um = np.full((n_process, n_cavity), np.nan)
    peak_height = np.full((n_process, n_cavity), np.nan)
    all_peak_distances_um = np.empty((n_process, n_cavity), dtype=object)
    all_peak_heights = np.empty((n_process, n_cavity), dtype=object)
    fft_config = {
        "ZERO_PAD_FACTOR": config.zero_pad_factor,
        "FFT_IGNORE_DC_BINS": config.fft_ignore_dc_bins,
        "FFT_PEAK_HEIGHT_RATIO": config.fft_peak_height_ratio,
        "FFT_PEAK_DISTANCE_BINS": config.fft_peak_distance_bins,
    }

    for p in range(n_process):
        for c in range(n_cavity):
            try:
                res = FFTSolver.solve(wavelengths_um, spectra[p, c, :], fft_config)
                distances = np.asarray(res["peak_distances_um"], dtype=float)
                heights = np.asarray(res["peak_heights"], dtype=float)
                all_peak_distances_um[p, c] = distances
                all_peak_heights[p, c] = heights
                if distances.size:
                    idx = int(np.argmax(heights)) if config.fft_peak_type == "dominant" else 0
                    l_fft_um[p, c] = distances[idx]
                    peak_height[p, c] = heights[idx]
            except Exception as exc:
                all_peak_distances_um[p, c] = np.array([], dtype=float)
                all_peak_heights[p, c] = np.array([], dtype=float)
                print(f"[FFT] failed process={p}, cavity={c}: {exc}")

    save_npz(
        fft_path,
        source_npz=np.array(str(dataset_npz_path.resolve())),
        wavelengths=wavelengths_um,
        cavity_axis_um=np.asarray(data["cavity_axis_um"], dtype=float),
        L_fft_um=l_fft_um,
        peak_height=peak_height,
        all_peak_distances_um=all_peak_distances_um,
        all_peak_heights=all_peak_heights,
        config_json=np.array(json.dumps(fft_config, ensure_ascii=False)),
    )
    print(f"[FFT] saved: {fft_path}")
    return fft_path


def calibrate_linear_on_nominal_process(cavity_axis_um: np.ndarray, l_fft_um: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(l_fft_um[0], dtype=float)
    y = np.asarray(cavity_axis_um, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        raise ValueError("At least two finite nominal-process FFT points are required for calibration.")
    a, b = np.polyfit(x[finite], y[finite], 1)
    print(f"[Linear] L_true = {a:.12g} * L_fft + {b:.12g}")
    return float(a), float(b)


def fit_fixed_film_lsq(
    config: ValidationConfig,
    wavelengths_um: np.ndarray,
    spectra: np.ndarray,
    l_init_um: np.ndarray,
    cache: LumapiSpectrumCache,
    film_nominal_nm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_process, n_cavity, _ = spectra.shape
    l_fit_um = np.full((n_process, n_cavity), np.nan)
    cost = np.full((n_process, n_cavity), np.nan)
    status = np.zeros((n_process, n_cavity), dtype=int)
    nfev = np.zeros((n_process, n_cavity), dtype=int)

    for p in range(n_process):
        for c in range(n_cavity):
            y_meas = spectra[p, c, :]
            if not np.all(np.isfinite(y_meas)) or not np.isfinite(l_init_um[p, c]):
                continue

            def residual(x: np.ndarray) -> np.ndarray:
                model = cache.get_spectrum(float(x[0]), film_nominal_nm)
                return normalize_spectrum(model) - normalize_spectrum(y_meas)

            low = l_init_um[p, c] - config.fit_l_window_um
            high = l_init_um[p, c] + config.fit_l_window_um
            try:
                res = least_squares(
                    residual,
                    x0=np.array([l_init_um[p, c]], dtype=float),
                    bounds=(np.array([low]), np.array([high])),
                    max_nfev=config.max_nfev_fixed,
                )
                l_fit_um[p, c] = float(res.x[0])
                cost[p, c] = float(res.cost)
                status[p, c] = int(res.status)
                nfev[p, c] = int(res.nfev)
            except Exception as exc:
                print(f"[Fixed LSQ] failed process={p}, cavity={c}: {exc}")
            cache.save_cache()
    return l_fit_um, cost, status, nfev


def fit_prior_joint_process_lsq(
    config: ValidationConfig,
    spectra_process: np.ndarray,
    l_init_array_um: np.ndarray,
    film_nominal_nm: np.ndarray,
    cache: LumapiSpectrumCache,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    n_cavity = spectra_process.shape[0]
    n_film = film_nominal_nm.size
    x0 = np.concatenate([l_init_array_um, np.zeros(n_film, dtype=float)])
    lb = np.concatenate([l_init_array_um - config.fit_l_window_um, -np.full(n_film, config.film_uncertainty_nm)])
    ub = np.concatenate([l_init_array_um + config.fit_l_window_um, np.full(n_film, config.film_uncertainty_nm)])

    def residual(x: np.ndarray) -> np.ndarray:
        cavity_values_um = x[:n_cavity]
        delta_nm = x[n_cavity:]
        film_current_nm = film_nominal_nm + delta_nm
        pieces = []
        for i in range(n_cavity):
            model = cache.get_spectrum(float(cavity_values_um[i]), film_current_nm)
            pieces.append(normalize_spectrum(model) - normalize_spectrum(spectra_process[i]))
        prior = delta_nm / config.film_sigma_nm
        pieces.append(config.prior_weight * prior)
        return np.concatenate(pieces)

    finite_init = np.all(np.isfinite(x0))
    finite_spectra = np.all(np.isfinite(spectra_process))
    if not finite_init or not finite_spectra:
        return np.full(n_cavity, np.nan), np.full(n_film, np.nan), np.nan, -1, 0

    res = least_squares(residual, x0=x0, bounds=(lb, ub), max_nfev=config.max_nfev_prior)
    cache.save_cache()
    return res.x[:n_cavity], res.x[n_cavity:], float(res.cost), int(res.status), int(res.nfev)


def compute_metrics(errors_nm: np.ndarray) -> Dict[str, float]:
    values = np.asarray(errors_nm, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ("mean_error_nm", "std_error_nm", "rmse_nm", "mae_nm", "max_abs_error_nm", "p95_abs_error_nm")}
    return {
        "mean_error_nm": float(np.mean(values)),
        "std_error_nm": float(np.std(values)),
        "rmse_nm": float(np.sqrt(np.mean(values**2))),
        "mae_nm": float(np.mean(np.abs(values))),
        "max_abs_error_nm": float(np.max(np.abs(values))),
        "p95_abs_error_nm": float(np.percentile(np.abs(values), 95)),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(run_dir: Path, rows: List[Dict[str, Any]], film_layers: Sequence[FilmLayerConfig], coeff_rows: List[Dict[str, Any]]) -> None:
    methods = ("fft_linear", "fixed_lsq", "prior_joint")
    errors = {m: np.array([row[f"error_{m}_nm"] for row in rows], dtype=float) for m in methods}
    cavity = np.array([row["cavity_true_um"] for row in rows], dtype=float)
    process = np.array([row["process_id"] for row in rows], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.boxplot([errors[m][np.isfinite(errors[m])] for m in methods], labels=methods)
    ax.set_ylabel("Error (nm)")
    ax.set_title("Error Distribution")
    ax.grid(True, axis="y")
    fig.savefig(run_dir / "03_error_boxplot.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for m in methods:
        ax.scatter(cavity, errors[m], s=14, alpha=0.7, label=m)
    ax.axhline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("True cavity length (um)")
    ax.set_ylabel("Error (nm)")
    ax.set_title("Error vs Cavity")
    ax.grid(True)
    ax.legend()
    fig.savefig(run_dir / "04_error_vs_cavity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for m in methods:
        rmse_by_process = []
        ids = sorted(set(process.tolist()))
        for pid in ids:
            e = errors[m][process == pid]
            rmse_by_process.append(np.sqrt(np.nanmean(e**2)))
        ax.plot(ids, rmse_by_process, "o-", ms=3, lw=1, label=m)
    ax.set_xlabel("Process ID")
    ax.set_ylabel("RMSE (nm)")
    ax.set_title("Per-process RMSE")
    ax.grid(True)
    ax.legend()
    fig.savefig(run_dir / "05_error_vs_process.png", dpi=220)
    plt.close(fig)

    if rows:
        target = max(rows, key=lambda r: abs(float(r.get("error_fft_linear_nm", 0.0))))
        wl_nm = np.asarray(target["wavelengths_um"]) * 1000.0
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(wl_nm, target["measured_spectrum"], lw=0.8, label="measured")
        ax.plot(wl_nm, target["fixed_spectrum"], lw=0.8, label="fixed-film LSQ")
        ax.plot(wl_nm, target["prior_spectrum"], lw=0.8, label="prior joint LSQ")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title("Example Spectrum Fit")
        ax.grid(True)
        ax.legend()
        fig.savefig(run_dir / "06_example_spectrum_fit.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(wl_nm, normalize_spectrum(target["fixed_spectrum"]) - normalize_spectrum(target["measured_spectrum"]), lw=0.8, label="fixed residual")
        ax.plot(wl_nm, normalize_spectrum(target["prior_spectrum"]) - normalize_spectrum(target["measured_spectrum"]), lw=0.8, label="prior residual")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized residual")
        ax.set_title("Example Spectrum Residual")
        ax.grid(True)
        ax.legend()
        fig.savefig(run_dir / "06_example_spectrum_fit_residual.png", dpi=220)
        plt.close(fig)

    coeff_process = np.array([r["process_id"] for r in coeff_rows], dtype=int)
    slopes = np.array([r["slope"] for r in coeff_rows], dtype=float)
    intercepts = np.array([r["intercept_um"] for r in coeff_rows], dtype=float)
    for filename, values, ylabel in (
        ("07_linear_coefficients_by_process.png", slopes, "slope"),
        ("07_linear_coefficients_by_process_intercept.png", intercepts, "intercept (um)"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(coeff_process, values, "o-", ms=3, lw=1)
        ax.set_xlabel("Process ID")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.savefig(run_dir / filename, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for j, layer in enumerate(film_layers):
        x = np.array([row[f"film_delta{j + 1}_true_nm"] for row in rows if row["cavity_id"] == 0], dtype=float)
        y = np.array([row[f"d{j + 1}_prior_fit_nm"] for row in rows if row["cavity_id"] == 0], dtype=float)
        ax.scatter(x, y, s=25, label=layer.name)
    ax.axline((0, 0), slope=1, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("True film delta (nm)")
    ax.set_ylabel("Recovered film delta (nm)")
    ax.set_title("Prior Recovered Film Error")
    ax.grid(True)
    ax.legend()
    fig.savefig(run_dir / "08_prior_recovered_film_error.png", dpi=220)
    plt.close(fig)


def run_validation(config: ValidationConfig = CONFIG) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (config.project_root / config.output_root / f"prior_joint_fit_lumapi_result_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] output directory: {run_dir}")
    with (run_dir / "00_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_json_dict(config), f, ensure_ascii=False, indent=2)

    failed_cases: List[Dict[str, Any]] = []
    dataset_path, dataset_failed = generate_lumapi_dataset(config, run_dir)
    failed_cases.extend(dataset_failed)
    fft_path = solve_fft_dataset(dataset_path, config, run_dir)

    dataset = np.load(dataset_path, allow_pickle=True)
    fft = np.load(fft_path, allow_pickle=True)
    wavelengths_um = np.asarray(dataset["wavelengths"], dtype=float)
    cavity_axis_um = np.asarray(dataset["cavity_axis_um"], dtype=float)
    spectra = np.asarray(dataset["spectra"], dtype=float)
    film_nominal_nm = np.asarray(dataset["film_nominal_nm"], dtype=float)
    film_true_nm = np.asarray(dataset["film_true_nm"], dtype=float)
    process_delta_nm = np.asarray(dataset["process_delta_nm"], dtype=float)
    l_fft_um = np.asarray(fft["L_fft_um"], dtype=float)
    peak_height = np.asarray(fft["peak_height"], dtype=float)

    linear_a, linear_b_um = calibrate_linear_on_nominal_process(cavity_axis_um, l_fft_um)
    l_linear_um = linear_a * l_fft_um + linear_b_um

    fdtd = open_fdtd(config)
    forward = LumapiForwardModel(config, wavelengths_um, 3e8 / (wavelengths_um * 1e-6), fdtd)
    cache = LumapiSpectrumCache(forward, run_dir / "lumapi_spectrum_cache.pkl", config.cache_round_digits)
    try:
        l_fixed_um, fixed_cost, fixed_status, fixed_nfev = fit_fixed_film_lsq(
            config, wavelengths_um, spectra, l_linear_um, cache, film_nominal_nm
        )
        n_process = config.num_processes
        n_cavity = config.num_cavity_points
        n_film = film_nominal_nm.size
        l_prior_um = np.full((n_process, n_cavity), np.nan)
        delta_fit_nm = np.full((n_process, n_film), np.nan)
        prior_cost = np.full(n_process, np.nan)
        prior_status = np.zeros(n_process, dtype=int)
        prior_nfev = np.zeros(n_process, dtype=int)

        for p in range(n_process):
            print(f"[Prior LSQ] process {p + 1}/{n_process}")
            try:
                l_p, d_p, cost_p, status_p, nfev_p = fit_prior_joint_process_lsq(
                    config, spectra[p], l_linear_um[p], film_nominal_nm, cache
                )
                l_prior_um[p] = l_p
                delta_fit_nm[p] = d_p
                prior_cost[p] = cost_p
                prior_status[p] = status_p
                prior_nfev[p] = nfev_p
            except Exception as exc:
                failed_cases.append({"stage": "prior_lsq", "process_id": p, "error": repr(exc)})
                print(f"[Prior LSQ] failed process={p}: {exc}")
            cache.save_cache()

        rows: List[Dict[str, Any]] = []
        plot_rows: List[Dict[str, Any]] = []
        for p in range(n_process):
            for c, cavity_um in enumerate(cavity_axis_um):
                row: Dict[str, Any] = {
                    "process_id": p,
                    "cavity_id": c,
                    "cavity_true_um": float(cavity_um),
                    "L_fft_um": float(l_fft_um[p, c]),
                    "fft_peak_height": float(peak_height[p, c]),
                    "L_linear_corrected_um": float(l_linear_um[p, c]),
                    "L_fixed_lsq_um": float(l_fixed_um[p, c]),
                    "L_prior_joint_um": float(l_prior_um[p, c]),
                    "error_fft_linear_nm": float((l_linear_um[p, c] - cavity_um) * 1000.0),
                    "error_fixed_lsq_nm": float((l_fixed_um[p, c] - cavity_um) * 1000.0),
                    "error_prior_joint_nm": float((l_prior_um[p, c] - cavity_um) * 1000.0),
                    "linear_a": linear_a,
                    "linear_b_um": linear_b_um,
                    "fixed_cost": float(fixed_cost[p, c]),
                    "fixed_status": int(fixed_status[p, c]),
                    "fixed_nfev": int(fixed_nfev[p, c]),
                    "prior_cost": float(prior_cost[p]),
                    "prior_status": int(prior_status[p]),
                    "prior_nfev": int(prior_nfev[p]),
                }
                for j, layer in enumerate(config.film_layers):
                    row[f"film_d{j + 1}_true_nm"] = float(film_true_nm[p, j])
                    row[f"film_delta{j + 1}_true_nm"] = float(process_delta_nm[p, j])
                    row[f"d{j + 1}_prior_fit_nm"] = float(delta_fit_nm[p, j])
                rows.append(row)
                plot_row = dict(row)
                plot_row["wavelengths_um"] = wavelengths_um
                plot_row["measured_spectrum"] = spectra[p, c]
                plot_row["fixed_spectrum"] = cache.get_spectrum(l_fixed_um[p, c], film_nominal_nm)
                plot_row["prior_spectrum"] = cache.get_spectrum(l_prior_um[p, c], film_nominal_nm + delta_fit_nm[p])
                plot_rows.append(plot_row)

        coeff_rows: List[Dict[str, Any]] = []
        for p in range(n_process):
            finite = np.isfinite(cavity_axis_um) & np.isfinite(l_fft_um[p])
            if np.count_nonzero(finite) >= 2:
                slope, intercept = np.polyfit(cavity_axis_um[finite], l_fft_um[p, finite], 1)
                pred = slope * cavity_axis_um[finite] + intercept
                rmse = float(np.sqrt(np.mean((l_fft_um[p, finite] - pred) ** 2)))
            else:
                slope, intercept, rmse = np.nan, np.nan, np.nan
            coeff_rows.append(
                {"process_id": p, "slope": float(slope), "intercept_um": float(intercept), "rmse_um": float(rmse)}
            )

        summary = {
            "fft_linear": compute_metrics(np.array([row["error_fft_linear_nm"] for row in rows])),
            "fixed_film_lsq": compute_metrics(np.array([row["error_fixed_lsq_nm"] for row in rows])),
            "prior_joint_lsq": compute_metrics(np.array([row["error_prior_joint_nm"] for row in rows])),
            "linear_coefficients_by_process": {
                "std_slope": float(np.nanstd([r["slope"] for r in coeff_rows])),
                "std_intercept_um": float(np.nanstd([r["intercept_um"] for r in coeff_rows])),
            },
        }

        with (run_dir / "01_summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        write_csv(run_dir / "02_all_results.csv", rows)
        write_csv(run_dir / "09_linear_coefficients_by_process.csv", coeff_rows)
        plot_outputs(run_dir, plot_rows, config.film_layers, coeff_rows)
    finally:
        try:
            fdtd.close()
        except Exception:
            pass

    with (run_dir / "failed_cases.json").open("w", encoding="utf-8") as f:
        json.dump(failed_cases, f, ensure_ascii=False, indent=2)
    shutil.copy2(Path(__file__).resolve(), run_dir / "validate_prior_joint_fit_lumapi.py")

    linear = summary["fft_linear"]
    fixed = summary["fixed_film_lsq"]
    prior = summary["prior_joint_lsq"]
    print("\n[Validation Summary]")
    print(f"Fixed linear calibration: RMSE={linear['rmse_nm']:.6g} nm, MAE={linear['mae_nm']:.6g} nm, Max={linear['max_abs_error_nm']:.6g} nm")
    print(f"Fixed-film LSQ: RMSE={fixed['rmse_nm']:.6g} nm, MAE={fixed['mae_nm']:.6g} nm, Max={fixed['max_abs_error_nm']:.6g} nm")
    print(f"Prior joint LSQ: RMSE={prior['rmse_nm']:.6g} nm, MAE={prior['mae_nm']:.6g} nm, Max={prior['max_abs_error_nm']:.6g} nm")
    print(f"std(slope_p)={summary['linear_coefficients_by_process']['std_slope']:.6g}")
    print(f"std(intercept_p)={summary['linear_coefficients_by_process']['std_intercept_um']:.6g} um")
    if prior["rmse_nm"] < fixed["rmse_nm"] and prior["rmse_nm"] < linear["rmse_nm"]:
        print("PASS: prior joint fitting improves robustness.")
    else:
        print("WARNING: prior joint fitting did not improve. Check identifiability, prior weight, angle diversity, or film model.")
    print("Hint: if L and film thickness are not identifiable at one angle, add angle_axis_deg or s/p polarization for joint I(lambda, theta, pol) fitting.")
    print(f"[Run] completed: {run_dir}")
    return run_dir


def main() -> None:
    run_validation(CONFIG)


if __name__ == "__main__":
    main()
