from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Lumerical API 路径与 main_cavity.py 保持一致。
LUMERICAL_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
LUMERICAL_BIN_PATH = Path(r"D:\Program Files\Lumerical\v241\bin")
if LUMERICAL_PATH.exists():
    lumerical_path_str = str(LUMERICAL_PATH)
    if lumerical_path_str not in sys.path:
        sys.path.append(lumerical_path_str)
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(LUMERICAL_BIN_PATH)

try:
    import lumapi
except ImportError:
    lumapi = None


OUTPUT_MODE = "scalar_plus_optional_spectra_and_full_features"
SPECTRAL_FEATURE_SOURCE = "full_spectrum_before_downsampling"
FILM_LAYER_NAMES = ["HSQ", "PSS", "SOC", "TiO2"]
SPLIT_NAMES = np.asarray(["train", "val", "test"], dtype=str)
SPECTRA_SAVE_MODES = {
    "none",
    "norm_downsampled",
    "norm_full",
    "raw_downsampled",
    "raw_and_norm_downsampled",
}


SPECTRAL_FEATURE_NAMES = np.asarray(
    [
        "spec_mean",
        "spec_std",
        "spec_min",
        "spec_max",
        "spec_ptp",
        "spec_skew",
        "spec_kurtosis",
        "spec_q05",
        "spec_q25",
        "spec_q50",
        "spec_q75",
        "spec_q95",
        "fft_peak_pos_1_um",
        "fft_peak_height_1",
        "fft_peak_width_1",
        "fft_peak_prominence_1",
        "fft_peak_pos_2_um",
        "fft_peak_height_2",
        "fft_peak_width_2",
        "fft_peak_prominence_2",
        "fft_peak_height_ratio_21",
        "fft_peak_distance_21_um",
        "fft_num_peaks",
        "fft_noise_floor",
        "fft_snr_1",
        "fft_spectral_centroid_um",
        "fft_band_energy_low",
        "fft_band_energy_mid",
        "fft_band_energy_high",
        "fringe_visibility_global",
        "fringe_contrast_std",
    ],
    dtype=str,
)


@dataclass
class Config:
    """版本 2 数据集的完整可复现配置。"""

    output_mode: str = OUTPUT_MODE
    model_type: str = "PSS_TiO2"
    run_timestamp: str = ""

    wavelength_start_um: float = 0.2
    wavelength_stop_um: float = 0.6
    spectral_resolution_nm: float = 0.02
    angle_deg: float = 0.0
    polarization: str = "p"

    cavity_start_um: float = 1000.0
    cavity_step_um: float = 0.001
    num_cavity_points: int = 400

    num_nominal_models: int = 100
    num_process_per_nominal: int = 20
    film_uncertainty_nm: float = 5.0
    min_true_film_thickness_nm: float = 0.1
    random_seed: int = 20260620

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    save_checkpoint_every_process: bool = True
    save_csv_index: bool = True
    save_check_plots: bool = True
    compress_npz: bool = True

    save_spectra: bool = True
    spectra_save_mode: str = "norm_downsampled"
    spectra_dtype: str = "float32"
    spectra_downsample_factor: int = 10
    spectra_downsample_method: str = "mean"
    spectrum_normalization: str = "per_spectrum_zscore"
    spectrum_normalization_eps: float = 1e-12
    extract_full_spectral_features: bool = True

    fft_peak_height_ratio: float = 0.2
    fft_ignore_dc_bins: int = 50
    fft_peak_distance_bins: int = 100
    zero_pad_factor: int = 8

    add_noise: bool = False
    noise_std: float = 0.0

    def validate(self) -> None:
        if self.spectra_save_mode not in SPECTRA_SAVE_MODES:
            raise ValueError(f"Unsupported spectra_save_mode={self.spectra_save_mode!r}")
        if self.spectra_dtype not in {"float16", "float32"}:
            raise ValueError("spectra_dtype must be float16 or float32")
        if self.spectra_downsample_method not in {"mean", "slice"}:
            raise ValueError("spectra_downsample_method must be mean or slice")
        if self.spectrum_normalization != "per_spectrum_zscore":
            raise ValueError("Only per_spectrum_zscore is currently supported")
        if self.spectra_downsample_factor < 1:
            raise ValueError("spectra_downsample_factor must be >= 1")
        if self.num_cavity_points < 1 or self.num_nominal_models < 1 or self.num_process_per_nominal < 1:
            raise ValueError("Dataset dimensions must be positive")
        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

        # 参数冲突时以 save_spectra 为最高优先级。
        if not self.save_spectra:
            self.spectra_save_mode = "none"
        if self.spectra_save_mode == "none":
            self.save_spectra = False


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build version-2 scalar + spectral-feature cavity dataset with Lumerical StackRT."
    )
    parser.add_argument("--num-cavity-points", type=int, default=400)
    parser.add_argument("--num-nominal-models", type=int, default=100)
    parser.add_argument("--num-process-per-nominal", type=int, default=20)
    parser.add_argument("--save-spectra", type=str2bool, default=True)
    parser.add_argument("--spectra-save-mode", choices=sorted(SPECTRA_SAVE_MODES), default="norm_downsampled")
    parser.add_argument("--spectra-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--spectra-downsample-factor", type=int, default=10)
    parser.add_argument("--spectra-downsample-method", choices=["mean", "slice"], default="mean")
    parser.add_argument("--spectrum-normalization", default="per_spectrum_zscore")
    parser.add_argument("--extract-full-spectral-features", type=str2bool, default=True)
    parser.add_argument("--random-seed", type=int, default=20260620)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-final-merge",
        type=str2bool,
        default=False,
        help="只生成 checkpoint，不在本次运行结尾合并最终 NPZ/CSV。",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        num_cavity_points=args.num_cavity_points,
        num_nominal_models=args.num_nominal_models,
        num_process_per_nominal=args.num_process_per_nominal,
        save_spectra=args.save_spectra,
        spectra_save_mode=args.spectra_save_mode,
        spectra_dtype=args.spectra_dtype,
        spectra_downsample_factor=args.spectra_downsample_factor,
        spectra_downsample_method=args.spectra_downsample_method,
        spectrum_normalization=args.spectrum_normalization,
        extract_full_spectral_features=args.extract_full_spectral_features,
        random_seed=args.random_seed,
    )
    config.validate()
    return config


def config_from_json(path: Path) -> Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid_fields = {item.name for item in fields(Config)}
    kwargs = {key: value for key, value in payload.items() if key in valid_fields}
    config = Config(**kwargs)
    config.validate()
    return config


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def replace_nonfinite_for_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: replace_nonfinite_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_nonfinite_for_json(item) for item in value]
    return value


def dump_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            replace_nonfinite_for_json(payload),
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )


def allowed_nominal_values_nm() -> dict[str, np.ndarray]:
    return {
        "PSS": np.asarray([10.0, 20.0, 30.0, 40.0]),
        "HSQ": np.asarray([20.0, 30.0, 40.0, 50.0, 60.0]),
        "SOC": np.asarray([40.0, 50.0, 60.0, 70.0, 80.0]),
        "TiO2": np.asarray([30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    }


def make_nominal_stack(index: int, hsq_nm: float, pss_nm: float, soc_nm: float, tio2_nm: float) -> dict:
    return {
        "name": (
            f"model_{index:03d}_hsq{int(hsq_nm)}_pss{int(pss_nm)}_"
            f"soc{int(soc_nm)}_tio2{int(tio2_nm)}"
        ),
        "HSQ": float(hsq_nm),
        "PSS": float(pss_nm),
        "SOC": float(soc_nm),
        "TiO2": float(tio2_nm),
    }


def build_nominal_stacks_nm(config: Config) -> list[dict]:
    values = allowed_nominal_values_nm()
    combinations = [
        (hsq, pss, soc, tio2)
        for hsq in values["HSQ"]
        for pss in values["PSS"]
        for soc in values["SOC"]
        for tio2 in values["TiO2"]
    ]
    if config.num_nominal_models > len(combinations):
        raise ValueError(
            f"num_nominal_models={config.num_nominal_models} exceeds grid size {len(combinations)}"
        )

    selected: list[tuple[float, float, float, float]] = []
    selected_set: set[tuple[float, float, float, float]] = set()

    # 先放入 16 个边界角点，确保每层上下边界都被覆盖。
    for hsq in (values["HSQ"][0], values["HSQ"][-1]):
        for pss in (values["PSS"][0], values["PSS"][-1]):
            for soc in (values["SOC"][0], values["SOC"][-1]):
                for tio2 in (values["TiO2"][0], values["TiO2"][-1]):
                    key = (float(hsq), float(pss), float(soc), float(tio2))
                    selected.append(key)
                    selected_set.add(key)

    # 再放入确定性的中间网格覆盖点。
    coverage_count = max(len(item) for item in values.values())
    for i in range(coverage_count * 2):
        key = (
            float(values["HSQ"][i % len(values["HSQ"])]),
            float(values["PSS"][(2 * i) % len(values["PSS"])]),
            float(values["SOC"][(3 * i) % len(values["SOC"])]),
            float(values["TiO2"][(5 * i) % len(values["TiO2"])]),
        )
        if key not in selected_set:
            selected.append(key)
            selected_set.add(key)

    remaining = [item for item in combinations if item not in selected_set]
    rng = np.random.default_rng(config.random_seed)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, config.num_nominal_models - len(selected))])
    selected = selected[: config.num_nominal_models]

    return [make_nominal_stack(idx, *item) for idx, item in enumerate(selected)]


def nominal_film_array_nm(stack: dict) -> np.ndarray:
    return np.asarray([stack[name] for name in FILM_LAYER_NAMES], dtype=np.float32)


def make_wavelength_axis_um(config: Config) -> np.ndarray:
    span_nm = (config.wavelength_stop_um - config.wavelength_start_um) * 1000.0
    count = int(round(span_nm / config.spectral_resolution_nm)) + 1
    return np.linspace(config.wavelength_start_um, config.wavelength_stop_um, count)


def make_cavity_axis_um(config: Config) -> np.ndarray:
    return np.round(
        config.cavity_start_um
        + config.cavity_step_um * np.arange(config.num_cavity_points, dtype=np.float64),
        12,
    )


def build_process_metadata(config: Config, nominal_stacks_nm: list[dict]) -> dict[str, np.ndarray]:
    total_processes = len(nominal_stacks_nm) * config.num_process_per_nominal
    process_nominal_stack_id = np.empty(total_processes, dtype=np.int16)
    process_film_delta_nm = np.empty((total_processes, len(FILM_LAYER_NAMES)), dtype=np.float32)
    process_film_true_nm = np.empty_like(process_film_delta_nm)
    rng = np.random.default_rng(config.random_seed)

    process_id = 0
    for nominal_stack_id, stack in enumerate(nominal_stacks_nm):
        nominal = nominal_film_array_nm(stack)
        for process_idx in range(config.num_process_per_nominal):
            if process_idx == 0:
                delta = np.zeros(len(FILM_LAYER_NAMES), dtype=np.float32)
            else:
                lower = np.maximum(
                    -float(config.film_uncertainty_nm),
                    float(config.min_true_film_thickness_nm) - nominal,
                )
                upper = np.full_like(nominal, float(config.film_uncertainty_nm))
                delta = rng.uniform(lower, upper).astype(np.float32)

            process_nominal_stack_id[process_id] = nominal_stack_id
            process_film_delta_nm[process_id] = delta
            process_film_true_nm[process_id] = nominal + delta
            process_id += 1

    return {
        "process_nominal_stack_id": process_nominal_stack_id,
        "process_film_delta_nm": process_film_delta_nm,
        "process_film_true_nm": process_film_true_nm,
    }


def split_processes_within_nominal(config: Config, process_metadata: dict[str, np.ndarray]) -> dict:
    """每个 nominal group 内部按 process 切分，避免同一 process 泄漏。"""

    nominal_ids = process_metadata["process_nominal_stack_id"]
    rng = np.random.default_rng(config.random_seed + 1)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for nominal_id in np.unique(nominal_ids):
        pids = np.flatnonzero(nominal_ids == nominal_id)
        rng.shuffle(pids)
        count = len(pids)
        n_train = int(round(count * config.train_ratio))
        n_val = int(round(count * config.val_ratio))
        n_train = min(max(n_train, 1), count - 2)
        n_val = min(max(n_val, 1), count - n_train - 1)
        train.extend(pids[:n_train].tolist())
        val.extend(pids[n_train : n_train + n_val].tolist())
        test.extend(pids[n_train + n_val :].tolist())

    train_ids = np.asarray(sorted(train), dtype=np.int32)
    val_ids = np.asarray(sorted(val), dtype=np.int32)
    test_ids = np.asarray(sorted(test), dtype=np.int32)
    lookup = np.full(len(nominal_ids), -1, dtype=np.int8)
    lookup[train_ids] = 0
    lookup[val_ids] = 1
    lookup[test_ids] = 2
    if np.any(lookup < 0):
        raise RuntimeError("Some process IDs were not assigned to a split")
    return {
        "train_process_ids": train_ids,
        "val_process_ids": val_ids,
        "test_process_ids": test_ids,
        "split_id_by_process": lookup,
    }


def build_layers(stack: dict, cavity_um: float, film_delta_nm: np.ndarray) -> list[tuple[str, float]]:
    delta = dict(zip(FILM_LAYER_NAMES, np.asarray(film_delta_nm, dtype=float)))
    return [
        ("RefReflector", 0.0),
        ("Air", float(cavity_um)),
        ("HSQ", (float(stack["HSQ"]) + delta["HSQ"]) / 1000.0),
        ("PSS", (float(stack["PSS"]) + delta["PSS"]) / 1000.0),
        ("SOC", (float(stack["SOC"]) + delta["SOC"]) / 1000.0),
        ("TiO2", (float(stack["TiO2"]) + delta["TiO2"]) / 1000.0),
        ("Cu", 0.0),
    ]


class StackRTSimulator:
    def __init__(self, config: Config):
        self.config = config
        self.wavelengths_um = make_wavelength_axis_um(config)
        self.freqs = 3e8 / (self.wavelengths_um * 1e-6)
        self.fdtd = None

    def open(self) -> None:
        if lumapi is None:
            raise RuntimeError("lumapi is unavailable; check the configured Lumerical API path")
        print("[StackRT] Opening one hidden Lumerical FDTD API session...")
        self.fdtd = lumapi.FDTD(hide=True)

    def close(self) -> None:
        if self.fdtd is not None:
            print("[StackRT] Closing Lumerical FDTD API session...")
            self.fdtd.close()
            self.fdtd = None

    def n_matrix_and_thicknesses(self, layers: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
        n_matrix = np.zeros((len(layers), len(self.freqs)), dtype=complex)
        thicknesses = np.empty(len(layers), dtype=float)
        w_um = self.wavelengths_um

        for idx, (material, thick_um) in enumerate(layers):
            thicknesses[idx] = float(thick_um) * 1e-6
            if material == "RefReflector":
                n_matrix[idx] = 5.8284
            elif material == "Air":
                n_matrix[idx] = 1.0
            elif material == "HSQ":
                n_matrix[idx] = 1.41
            elif material == "PSS":
                n_matrix[idx] = 1.50 + 0.05j
            elif material == "SOC":
                n_matrix[idx] = 1.55 + 0.005 / (w_um**2)
            elif material == "TiO2":
                n_matrix[idx] = 2.4 + 0.02 / (w_um**2)
            elif material == "Cu":
                n_matrix[idx] = 1.1 + 2.5j
            else:
                raise ValueError(f"Unknown material: {material}")
        return n_matrix, thicknesses

    def simulate_spectrum(self, layers: list[tuple[str, float]]) -> np.ndarray:
        if self.fdtd is None:
            raise RuntimeError("StackRTSimulator.open() must be called first")
        n_matrix, thicknesses = self.n_matrix_and_thicknesses(layers)
        result_key = "Rp" if self.config.polarization.lower() == "p" else "Rs"
        result = self.fdtd.stackrt(
            n_matrix,
            thicknesses,
            self.freqs,
            float(self.config.angle_deg),
        )
        spectrum = np.real(np.asarray(result[result_key]).reshape(-1))
        if spectrum.size != self.wavelengths_um.size:
            raise ValueError(
                f"StackRT returned {spectrum.size} points; expected {self.wavelengths_um.size}"
            )
        return spectrum


def normalize_spectrum(spectrum: np.ndarray, eps: float) -> tuple[np.ndarray, bool]:
    """单条光谱自身 z-score；不会引入 train/val/test 泄漏。"""

    spectrum = np.asarray(spectrum, dtype=np.float64)
    mean = float(np.mean(spectrum))
    std = float(np.std(spectrum))
    if not np.isfinite(std) or std < eps:
        return spectrum - mean, False
    return (spectrum - mean) / std, True


def downsample_vector(values: np.ndarray, factor: int, method: str) -> np.ndarray:
    values = np.asarray(values)
    if factor == 1:
        return values.copy()
    if method == "slice":
        return values[::factor].copy()
    usable = (len(values) // factor) * factor
    if usable == 0:
        raise ValueError("Downsample factor is larger than the spectrum length")
    return values[:usable].reshape(-1, factor).mean(axis=1)


def saved_wavelength_axis(wavelengths_um: np.ndarray, config: Config) -> np.ndarray:
    if not config.save_spectra or config.spectra_save_mode == "none":
        return np.asarray([], dtype=np.float64)
    if config.spectra_save_mode == "norm_full":
        return wavelengths_um.copy()
    return downsample_vector(
        wavelengths_um,
        config.spectra_downsample_factor,
        config.spectra_downsample_method,
    ).astype(np.float64)


def spectra_keys_for_mode(config: Config) -> list[str]:
    if not config.save_spectra or config.spectra_save_mode == "none":
        return []
    mapping = {
        "norm_downsampled": ["spectra_norm_ds"],
        "norm_full": ["spectra_norm"],
        "raw_downsampled": ["spectra_ds"],
        "raw_and_norm_downsampled": ["spectra_ds", "spectra_norm_ds"],
    }
    return mapping[config.spectra_save_mode]


def prepare_spectra_for_save(
    spectrum: np.ndarray,
    normalized: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray]:
    dtype = np.dtype(config.spectra_dtype)
    mode = config.spectra_save_mode
    if not config.save_spectra or mode == "none":
        return {}
    if mode == "norm_full":
        return {"spectra_norm": normalized.astype(dtype)}

    raw_ds = downsample_vector(
        spectrum,
        config.spectra_downsample_factor,
        config.spectra_downsample_method,
    )
    norm_ds = downsample_vector(
        normalized,
        config.spectra_downsample_factor,
        config.spectra_downsample_method,
    )
    if mode == "norm_downsampled":
        return {"spectra_norm_ds": norm_ds.astype(dtype)}
    if mode == "raw_downsampled":
        return {"spectra_ds": raw_ds.astype(dtype)}
    if mode == "raw_and_norm_downsampled":
        return {
            "spectra_ds": raw_ds.astype(dtype),
            "spectra_norm_ds": norm_ds.astype(dtype),
        }
    raise ValueError(f"Unsupported spectra_save_mode={mode}")


def fft_analysis(wavelengths_um: np.ndarray, spectrum: np.ndarray, config: Config) -> dict:
    """基于完整原始光谱执行 k-space FFT，并返回粗解及扩展特征。"""

    wavelengths = np.asarray(wavelengths_um, dtype=np.float64).reshape(-1)
    intensity = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    finite = np.isfinite(wavelengths) & np.isfinite(intensity)
    wavelengths = wavelengths[finite]
    intensity = intensity[finite]
    if wavelengths.size < 4:
        raise ValueError("At least four finite spectrum points are required")

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    intensity = intensity[order]
    k_raw = 2.0 * np.pi / wavelengths
    k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))
    if k_raw[0] > k_raw[-1]:
        i_linear = np.interp(k_linear, k_raw[::-1], intensity[::-1])
    else:
        i_linear = np.interp(k_linear, k_raw, intensity)

    windowed = (i_linear - np.mean(i_linear)) * np.hanning(len(i_linear))
    n_fft = len(windowed) * int(config.zero_pad_factor)
    fft_amp = np.abs(np.fft.rfft(windowed, n=n_fft))
    dk = abs(k_linear[1] - k_linear[0])
    max_range_um = np.pi / dk
    distance_um = np.linspace(0.0, max_range_um / 2.0, len(fft_amp))

    ignore = min(int(config.fft_ignore_dc_bins), max(0, len(fft_amp) - 1))
    search_amp = fft_amp[ignore:]
    if search_amp.size == 0 or not np.isfinite(search_amp).any() or np.nanmax(search_amp) <= 0:
        peaks = np.asarray([], dtype=int)
    else:
        peaks, _ = find_peaks(
            search_amp,
            height=float(np.nanmax(search_amp)) * float(config.fft_peak_height_ratio),
            distance=int(config.fft_peak_distance_bins),
        )
        peaks = peaks + ignore

    noise_floor = float(np.median(search_amp)) if search_amp.size else float("nan")
    search_distance = distance_um[ignore:]
    amplitude_sum = float(np.sum(search_amp))
    centroid = (
        float(np.sum(search_distance * search_amp) / amplitude_sum)
        if amplitude_sum > 0
        else float("nan")
    )
    energy = search_amp**2
    chunks = np.array_split(energy, 3)
    total_energy = float(np.sum(energy))
    band_energy = [
        float(np.sum(chunk) / total_energy) if total_energy > 0 else float("nan")
        for chunk in chunks
    ]

    peak_records: list[dict] = []
    if peaks.size:
        prominences = peak_prominences(fft_amp, peaks)[0]
        widths_samples = peak_widths(fft_amp, peaks, rel_height=0.5)[0]
        distance_step = float(np.mean(np.diff(distance_um)))
        for peak, prominence, width_samples in zip(peaks, prominences, widths_samples):
            peak_records.append(
                {
                    "position_um": float(distance_um[peak]),
                    "height": float(fft_amp[peak]),
                    "width": float(width_samples * distance_step),
                    "prominence": float(prominence),
                }
            )
        peak_records.sort(key=lambda item: item["height"], reverse=True)

    first = peak_records[0] if peak_records else None
    second = peak_records[1] if len(peak_records) > 1 else None
    l_fft_um = first["position_um"] if first else float("nan")
    h_peak = first["height"] if first else float("nan")

    def peak_value(record: dict | None, key: str) -> float:
        return float(record[key]) if record is not None else float("nan")

    return {
        "L_fft_um": l_fft_um,
        "H_peak": h_peak,
        "peak_count": int(len(peaks)),
        "fft_features": {
            "fft_peak_pos_1_um": peak_value(first, "position_um"),
            "fft_peak_height_1": peak_value(first, "height"),
            "fft_peak_width_1": peak_value(first, "width"),
            "fft_peak_prominence_1": peak_value(first, "prominence"),
            "fft_peak_pos_2_um": peak_value(second, "position_um"),
            "fft_peak_height_2": peak_value(second, "height"),
            "fft_peak_width_2": peak_value(second, "width"),
            "fft_peak_prominence_2": peak_value(second, "prominence"),
            "fft_peak_height_ratio_21": (
                peak_value(second, "height") / peak_value(first, "height")
                if first is not None and second is not None and first["height"] != 0
                else float("nan")
            ),
            "fft_peak_distance_21_um": (
                abs(peak_value(second, "position_um") - peak_value(first, "position_um"))
                if first is not None and second is not None
                else float("nan")
            ),
            "fft_num_peaks": float(len(peaks)),
            "fft_noise_floor": noise_floor,
            "fft_snr_1": (
                peak_value(first, "height") / (noise_floor + config.spectrum_normalization_eps)
                if first is not None and np.isfinite(noise_floor)
                else float("nan")
            ),
            "fft_spectral_centroid_um": centroid,
            "fft_band_energy_low": band_energy[0],
            "fft_band_energy_mid": band_energy[1],
            "fft_band_energy_high": band_energy[2],
        },
    }


def extract_full_spectral_features(spectrum: np.ndarray, fft_result: dict, eps: float) -> np.ndarray:
    """从完整原始光谱与完整 k-space FFT 中提取固定顺序的标量特征。"""

    values = np.asarray(spectrum, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(len(SPECTRAL_FEATURE_NAMES), np.nan, dtype=np.float32)

    mean = float(np.mean(finite))
    std = float(np.std(finite))
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if std > eps:
        standardized = (finite - mean) / std
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0
    quantiles = np.quantile(finite, [0.05, 0.25, 0.50, 0.75, 0.95])

    feature_by_name = {
        "spec_mean": mean,
        "spec_std": std,
        "spec_min": minimum,
        "spec_max": maximum,
        "spec_ptp": maximum - minimum,
        "spec_skew": skewness,
        "spec_kurtosis": kurtosis,
        "spec_q05": float(quantiles[0]),
        "spec_q25": float(quantiles[1]),
        "spec_q50": float(quantiles[2]),
        "spec_q75": float(quantiles[3]),
        "spec_q95": float(quantiles[4]),
        **fft_result["fft_features"],
        "fringe_visibility_global": (maximum - minimum) / (maximum + minimum + eps),
        "fringe_contrast_std": std / (abs(mean) + eps),
    }
    return np.asarray([feature_by_name[name] for name in SPECTRAL_FEATURE_NAMES], dtype=np.float32)


def allocate_process_arrays(config: Config, wavelengths_saved_um: np.ndarray) -> dict[str, np.ndarray]:
    n = config.num_cavity_points
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.empty(n, dtype=np.int64),
        "process_id": np.empty(n, dtype=np.int32),
        "nominal_stack_id": np.empty(n, dtype=np.int16),
        "split_id": np.empty(n, dtype=np.int8),
        "valid_mask": np.zeros(n, dtype=bool),
        "simulation_failed_mask": np.zeros(n, dtype=bool),
        "fft_failed_mask": np.zeros(n, dtype=bool),
        "cavity_true_um": np.empty(n, dtype=np.float64),
        "L_true_um": np.empty(n, dtype=np.float64),
        "L_fft_um": np.full(n, np.nan, dtype=np.float64),
        "delta_L_um": np.full(n, np.nan, dtype=np.float64),
        "delta_L_nm": np.full(n, np.nan, dtype=np.float64),
        "H_peak": np.full(n, np.nan, dtype=np.float32),
        "peak_count": np.zeros(n, dtype=np.int16),
        "film_nominal_nm": np.empty((n, len(FILM_LAYER_NAMES)), dtype=np.float32),
        "film_delta_nm": np.empty((n, len(FILM_LAYER_NAMES)), dtype=np.float32),
        "film_true_nm": np.empty((n, len(FILM_LAYER_NAMES)), dtype=np.float32),
        "spectral_features_full": np.full(
            (n, len(SPECTRAL_FEATURE_NAMES) if config.extract_full_spectral_features else 0),
            np.nan,
            dtype=np.float32,
        ),
    }
    spectra_dtype = np.dtype(config.spectra_dtype)
    for key in spectra_keys_for_mode(config):
        arrays[key] = np.full((n, len(wavelengths_saved_um)), np.nan, dtype=spectra_dtype)
    return arrays


def checkpoint_metadata(config: Config, wavelengths_um: np.ndarray, wavelengths_saved_um: np.ndarray) -> dict:
    return {
        "split_names": SPLIT_NAMES,
        "layer_names": np.asarray(FILM_LAYER_NAMES, dtype=str),
        "spectral_feature_names": (
            SPECTRAL_FEATURE_NAMES
            if config.extract_full_spectral_features
            else np.asarray([], dtype=str)
        ),
        "spectral_feature_source": np.array(SPECTRAL_FEATURE_SOURCE),
        "wavelengths_um": wavelengths_um,
        "wavelengths_spectra_saved_um": wavelengths_saved_um,
        "spectra_saved": np.array(bool(config.save_spectra), dtype=bool),
        "spectra_save_mode": np.array(config.spectra_save_mode),
        "spectra_dtype": np.array(config.spectra_dtype),
        "spectra_downsample_factor": np.array(config.spectra_downsample_factor, dtype=np.int32),
        "spectra_downsample_method": np.array(config.spectra_downsample_method),
        "spectrum_normalization": np.array(config.spectrum_normalization),
        "spectra_norm_method": np.array(config.spectrum_normalization),
    }


def save_checkpoint(
    run_dir: Path,
    process_id: int,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    config: Config,
) -> Path:
    path = run_dir / f"checkpoint_process_{process_id:04d}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
    payload = {**arrays, **metadata}
    if config.compress_npz:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    print(f"[Checkpoint] Saved {path.name}")
    return path


def simulate_one_process(
    simulator: StackRTSimulator,
    config: Config,
    run_dir: Path,
    process_id: int,
    nominal_stack_id: int,
    nominal_stack: dict,
    film_delta_nm: np.ndarray,
    film_true_nm: np.ndarray,
    split_id: int,
    cavity_axis_um: np.ndarray,
    wavelengths_saved_um: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    arrays = allocate_process_arrays(config, wavelengths_saved_um)
    nominal_nm = nominal_film_array_nm(nominal_stack)
    failed_sim: list[dict] = []
    failed_fft: list[dict] = []

    for cavity_idx, cavity_um in enumerate(cavity_axis_um):
        sample_id = process_id * config.num_cavity_points + cavity_idx
        arrays["sample_id"][cavity_idx] = sample_id
        arrays["process_id"][cavity_idx] = process_id
        arrays["nominal_stack_id"][cavity_idx] = nominal_stack_id
        arrays["split_id"][cavity_idx] = split_id
        arrays["cavity_true_um"][cavity_idx] = cavity_um
        arrays["L_true_um"][cavity_idx] = cavity_um
        arrays["film_nominal_nm"][cavity_idx] = nominal_nm
        arrays["film_delta_nm"][cavity_idx] = film_delta_nm
        arrays["film_true_nm"][cavity_idx] = film_true_nm

        try:
            layers = build_layers(nominal_stack, float(cavity_um), film_delta_nm)
            spectrum = simulator.simulate_spectrum(layers)
            if config.add_noise and config.noise_std > 0:
                spectrum = spectrum + rng.normal(0.0, config.noise_std, size=spectrum.shape)

            fft_result = fft_analysis(simulator.wavelengths_um, spectrum, config)
            l_fft = float(fft_result["L_fft_um"])
            h_peak = float(fft_result["H_peak"])
            arrays["L_fft_um"][cavity_idx] = l_fft
            arrays["H_peak"][cavity_idx] = h_peak
            arrays["peak_count"][cavity_idx] = int(fft_result["peak_count"])
            if np.isfinite(l_fft):
                arrays["delta_L_um"][cavity_idx] = float(cavity_um) - l_fft
                arrays["delta_L_nm"][cavity_idx] = (float(cavity_um) - l_fft) * 1000.0

            if config.extract_full_spectral_features:
                arrays["spectral_features_full"][cavity_idx] = extract_full_spectral_features(
                    spectrum,
                    fft_result,
                    config.spectrum_normalization_eps,
                )

            normalized, norm_ok = normalize_spectrum(
                spectrum,
                config.spectrum_normalization_eps,
            )
            if not norm_ok:
                print(
                    f"[Warning] Near-zero spectrum std: process_id={process_id}, cavity_idx={cavity_idx}"
                )
            for key, values in prepare_spectra_for_save(spectrum, normalized, config).items():
                arrays[key][cavity_idx] = values

            valid = np.isfinite(l_fft) and np.isfinite(h_peak)
            arrays["valid_mask"][cavity_idx] = valid
            arrays["fft_failed_mask"][cavity_idx] = not valid
            if not valid:
                failed_fft.append(
                    {
                        "sample_id": sample_id,
                        "process_id": process_id,
                        "nominal_stack_id": nominal_stack_id,
                        "cavity_idx": cavity_idx,
                        "cavity_true_um": float(cavity_um),
                        "reason": "no FFT peak detected",
                    }
                )
        except Exception as exc:
            arrays["simulation_failed_mask"][cavity_idx] = True
            failed_sim.append(
                {
                    "sample_id": sample_id,
                    "process_id": process_id,
                    "nominal_stack_id": nominal_stack_id,
                    "nominal_stack_name": nominal_stack["name"],
                    "cavity_idx": cavity_idx,
                    "cavity_true_um": float(cavity_um),
                    "film_nominal_nm": nominal_nm.tolist(),
                    "film_delta_nm": np.asarray(film_delta_nm).tolist(),
                    "film_true_nm": np.asarray(film_true_nm).tolist(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"[Warning] Failed sample process_id={process_id}, cavity_idx={cavity_idx}: {exc}"
            )

    metadata = checkpoint_metadata(config, simulator.wavelengths_um, wavelengths_saved_um)
    save_checkpoint(run_dir, process_id, arrays, metadata, config)
    return failed_sim, failed_fft


def config_to_dict(config: Config, nominal_stacks_nm: list[dict]) -> dict:
    payload = asdict(config)
    payload.update(
        {
            "film_layer_names": FILM_LAYER_NAMES,
            "allowed_nominal_values_nm": {
                name: values.tolist() for name, values in allowed_nominal_values_nm().items()
            },
            "nominal_stacks_nm": nominal_stacks_nm,
            "spectral_feature_names": SPECTRAL_FEATURE_NAMES.tolist(),
            "spectral_feature_source": SPECTRAL_FEATURE_SOURCE,
            "lumerical_path": str(LUMERICAL_PATH),
            "lumerical_bin_path": str(LUMERICAL_BIN_PATH),
        }
    )
    return payload


def save_initial_run_files(run_dir: Path, config: Config, nominal_stacks_nm: list[dict]) -> None:
    dump_json(run_dir / "00_config.json", config_to_dict(config, nominal_stacks_nm))
    for source_name in ["build_nn_cavity_dataset.py", "build_nn_cavity_dataset_v2.py"]:
        source = Path(__file__).resolve().parent / source_name
        if source.exists():
            shutil.copy2(source, run_dir / source.name)
    resume_source = Path(__file__).resolve().parent / "resume_nn_cavity_dataset_v2.py"
    if resume_source.exists():
        shutil.copy2(resume_source, run_dir / "resume_nn_cavity_scalar_results.py")


def existing_checkpoint_ids(run_dir: Path) -> list[int]:
    ids = []
    pattern = re.compile(r"checkpoint_process_(\d+)\.npz$")
    for path in run_dir.glob("checkpoint_process_*.npz"):
        match = pattern.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def validate_checkpoint_sequence(ids: list[int]) -> int:
    if not ids:
        return 0
    expected = list(range(ids[-1] + 1))
    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        raise RuntimeError(f"Checkpoint sequence has gaps; first missing IDs: {missing[:10]}")
    return ids[-1] + 1


def run_simulation(
    config: Config,
    run_dir: Path,
    nominal_stacks_nm: list[dict],
    process_metadata: dict[str, np.ndarray],
    split_info: dict,
) -> dict:
    cavity_axis_um = make_cavity_axis_um(config)
    simulator = StackRTSimulator(config)
    wavelengths_saved_um = saved_wavelength_axis(simulator.wavelengths_um, config)
    total_processes = len(process_metadata["process_nominal_stack_id"])
    start_process = validate_checkpoint_sequence(existing_checkpoint_ids(run_dir))
    rng = np.random.default_rng(config.random_seed + 2)
    failed_sim_all: list[dict] = []
    failed_fft_all: list[dict] = []
    start_time = time.time()

    print(f"[Run] Starting at process_id={start_process}; total_processes={total_processes}")
    print(
        f"[Run] wavelengths full/saved={len(simulator.wavelengths_um)}/{len(wavelengths_saved_um)}; "
        f"spectra_mode={config.spectra_save_mode}; dtype={config.spectra_dtype}"
    )

    if start_process < total_processes:
        simulator.open()
        try:
            for process_id in range(start_process, total_processes):
                nominal_stack_id = int(process_metadata["process_nominal_stack_id"][process_id])
                failed_sim, failed_fft = simulate_one_process(
                    simulator=simulator,
                    config=config,
                    run_dir=run_dir,
                    process_id=process_id,
                    nominal_stack_id=nominal_stack_id,
                    nominal_stack=nominal_stacks_nm[nominal_stack_id],
                    film_delta_nm=process_metadata["process_film_delta_nm"][process_id],
                    film_true_nm=process_metadata["process_film_true_nm"][process_id],
                    split_id=int(split_info["split_id_by_process"][process_id]),
                    cavity_axis_um=cavity_axis_um,
                    wavelengths_saved_um=wavelengths_saved_um,
                    rng=rng,
                )
                failed_sim_all.extend(failed_sim)
                failed_fft_all.extend(failed_fft)
                elapsed = time.time() - start_time
                print(
                    f"[Run] process_id={process_id} complete; "
                    f"failed_sim_session={len(failed_sim_all)}, "
                    f"failed_fft_session={len(failed_fft_all)}, elapsed={elapsed:.1f}s"
                )
        finally:
            simulator.close()

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_sim_path = run_dir / f"failed_cases_{session_stamp}.json"
    failed_fft_path = run_dir / f"failed_fft_cases_{session_stamp}.json"
    dump_json(failed_sim_path, failed_sim_all)
    dump_json(failed_fft_path, failed_fft_all)

    manifest = {
        "status": "completed_checkpoints" if len(existing_checkpoint_ids(run_dir)) == total_processes else "running",
        "last_completed_process_id": total_processes - 1 if total_processes else None,
        "next_process_id": len(existing_checkpoint_ids(run_dir)),
        "completed_checkpoint_count": len(existing_checkpoint_ids(run_dir)),
        "remaining_process_count": total_processes - len(existing_checkpoint_ids(run_dir)),
        "failed_simulation_this_session": len(failed_sim_all),
        "failed_fft_this_session": len(failed_fft_all),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(run_dir / "resume_manifest.json", manifest)
    return {
        "cavity_axis_um": cavity_axis_um,
        "wavelengths_um": simulator.wavelengths_um,
        "wavelengths_spectra_saved_um": wavelengths_saved_um,
        "failed_simulation_this_session": failed_sim_all,
        "failed_fft_this_session": failed_fft_all,
    }


SAMPLE_ARRAY_KEYS = [
    "sample_id",
    "process_id",
    "nominal_stack_id",
    "split_id",
    "valid_mask",
    "simulation_failed_mask",
    "fft_failed_mask",
    "cavity_true_um",
    "L_true_um",
    "L_fft_um",
    "delta_L_um",
    "delta_L_nm",
    "H_peak",
    "peak_count",
    "film_nominal_nm",
    "film_delta_nm",
    "film_true_nm",
    "spectral_features_full",
]


def checkpoint_paths(run_dir: Path, total_processes: int) -> list[Path]:
    paths = [run_dir / f"checkpoint_process_{process_id:04d}.npz" for process_id in range(total_processes)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot merge final dataset; {len(missing)} checkpoints are missing. First: {missing[:10]}"
        )
    return paths


def safe_recreate_merge_dir(run_dir: Path) -> Path:
    temp_dir = (run_dir / ".merge_tmp_v2").resolve()
    if temp_dir.parent != run_dir.resolve():
        raise RuntimeError(f"Unsafe merge temp path: {temp_dir}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    return temp_dir


def allocate_merge_memmaps(
    temp_dir: Path,
    first_checkpoint: np.lib.npyio.NpzFile,
    total_rows: int,
    sample_keys: list[str],
) -> dict[str, np.memmap]:
    memmaps: dict[str, np.memmap] = {}
    for key in sample_keys:
        source = first_checkpoint[key]
        shape = (total_rows, *source.shape[1:])
        memmaps[key] = np.lib.format.open_memmap(
            temp_dir / f"{key}.npy",
            mode="w+",
            dtype=source.dtype,
            shape=shape,
        )
    return memmaps


def fill_merge_memmaps(
    paths: list[Path],
    memmaps: dict[str, np.memmap],
    sample_keys: list[str],
) -> int:
    write_start = 0
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=True) as data:
            n = int(data["sample_id"].shape[0])
            write_stop = write_start + n
            for key in sample_keys:
                memmaps[key][write_start:write_stop] = data[key]
            write_start = write_stop
        if index % 100 == 0 or index == len(paths) - 1:
            print(f"[Merge] Loaded checkpoint {index:04d}/{len(paths) - 1:04d}")
    for array in memmaps.values():
        array.flush()
    return write_start


def finite_stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def estimate_spectra_storage_gb(config: Config, total_rows: int, saved_wavelength_count: int) -> float:
    matrix_count = len(spectra_keys_for_mode(config))
    bytes_per_value = np.dtype(config.spectra_dtype).itemsize
    total_bytes = total_rows * saved_wavelength_count * bytes_per_value * matrix_count
    return float(total_bytes / (1024**3))


def build_summary(
    config: Config,
    memmaps: dict[str, np.memmap],
    nominal_stacks_nm: list[dict],
    split_info: dict,
    wavelengths_um: np.ndarray,
    wavelengths_saved_um: np.ndarray,
) -> dict:
    delta_stats = finite_stats(memmaps["delta_L_nm"])
    cavity_axis = make_cavity_axis_um(config)
    total_rows = int(memmaps["sample_id"].shape[0])
    return {
        "output_mode": OUTPUT_MODE,
        "spectra_saved": bool(config.save_spectra),
        "raw_full_spectra_saved": False,
        "normalized_spectra_saved": config.spectra_save_mode in {
            "norm_downsampled",
            "norm_full",
            "raw_and_norm_downsampled",
        },
        "spectra_save_mode": config.spectra_save_mode,
        "spectra_dtype": config.spectra_dtype,
        "spectra_downsample_factor": int(config.spectra_downsample_factor),
        "spectra_downsample_method": config.spectra_downsample_method,
        "spectrum_normalization": config.spectrum_normalization,
        "num_wavelengths_full": int(len(wavelengths_um)),
        "num_wavelengths_saved": int(len(wavelengths_saved_um)),
        "estimated_spectra_storage_gb": estimate_spectra_storage_gb(
            config, total_rows, len(wavelengths_saved_um)
        ),
        "num_samples_planned": int(
            config.num_nominal_models * config.num_process_per_nominal * config.num_cavity_points
        ),
        "num_samples_total": total_rows,
        "num_samples_valid": int(np.count_nonzero(memmaps["valid_mask"])),
        "num_failed_simulation": int(np.count_nonzero(memmaps["simulation_failed_mask"])),
        "num_failed_fft": int(np.count_nonzero(memmaps["fft_failed_mask"])),
        "num_nominal_stacks": int(len(nominal_stacks_nm)),
        "num_processes": int(config.num_nominal_models * config.num_process_per_nominal),
        "num_train_processes": int(len(split_info["train_process_ids"])),
        "num_val_processes": int(len(split_info["val_process_ids"])),
        "num_test_processes": int(len(split_info["test_process_ids"])),
        "wavelength_start_um": float(config.wavelength_start_um),
        "wavelength_stop_um": float(config.wavelength_stop_um),
        "num_wavelengths": int(len(wavelengths_um)),
        "cavity_start_um": float(cavity_axis[0]),
        "cavity_stop_um": float(cavity_axis[-1]),
        "cavity_step_um": float(config.cavity_step_um),
        "num_cavity_points": int(config.num_cavity_points),
        "film_layer_names": FILM_LAYER_NAMES,
        "film_uncertainty_nm": float(config.film_uncertainty_nm),
        "nominal_ranges_nm": {
            name: [float(values[0]), float(values[-1])]
            for name, values in allowed_nominal_values_nm().items()
        },
        "nominal_step_nm": 10.0,
        "L_fft_nan_count": int(np.count_nonzero(~np.isfinite(memmaps["L_fft_um"]))),
        "delta_L_nm_mean": delta_stats["mean"],
        "delta_L_nm_std": delta_stats["std"],
        "delta_L_nm_min": delta_stats["min"],
        "delta_L_nm_max": delta_stats["max"],
        "num_spectral_features": int(
            len(SPECTRAL_FEATURE_NAMES) if config.extract_full_spectral_features else 0
        ),
        "spectral_feature_names": (
            SPECTRAL_FEATURE_NAMES.tolist() if config.extract_full_spectral_features else []
        ),
        "spectral_feature_source": SPECTRAL_FEATURE_SOURCE,
    }


def save_final_npz(
    run_dir: Path,
    config: Config,
    timestamp: str,
    memmaps: dict[str, np.memmap],
    nominal_stacks_nm: list[dict],
    process_metadata: dict[str, np.ndarray],
    split_info: dict,
    wavelengths_um: np.ndarray,
    wavelengths_saved_um: np.ndarray,
) -> Path:
    path = run_dir / f"nn_cavity_spectral_features_{timestamp}.npz"
    nominal_values = np.asarray(
        [[stack[name] for name in FILM_LAYER_NAMES] for stack in nominal_stacks_nm],
        dtype=np.float32,
    )
    config_json = json.dumps(
        config_to_dict(config, nominal_stacks_nm),
        ensure_ascii=False,
        default=json_default,
    )
    payload = {
        **memmaps,
        "wavelengths_um": wavelengths_um,
        "wavelengths_spectra_saved_um": wavelengths_saved_um,
        "nominal_stack_name_by_id": np.asarray([stack["name"] for stack in nominal_stacks_nm]),
        "split_names": SPLIT_NAMES,
        "layer_names": np.asarray(FILM_LAYER_NAMES),
        "spectral_feature_names": (
            SPECTRAL_FEATURE_NAMES
            if config.extract_full_spectral_features
            else np.asarray([], dtype=str)
        ),
        "spectral_feature_source": np.array(SPECTRAL_FEATURE_SOURCE),
        "cavity_axis_um": make_cavity_axis_um(config),
        "train_process_ids": split_info["train_process_ids"],
        "val_process_ids": split_info["val_process_ids"],
        "test_process_ids": split_info["test_process_ids"],
        "nominal_stack_values_nm": nominal_values,
        "process_nominal_stack_id": process_metadata["process_nominal_stack_id"],
        "process_film_delta_nm": process_metadata["process_film_delta_nm"],
        "process_film_true_nm": process_metadata["process_film_true_nm"],
        "num_samples_planned": np.array(len(memmaps["sample_id"]), dtype=np.int64),
        "num_processes_planned": np.array(
            config.num_nominal_models * config.num_process_per_nominal,
            dtype=np.int32,
        ),
        "spectra_saved": np.array(config.save_spectra, dtype=bool),
        "spectra_save_mode": np.array(config.spectra_save_mode),
        "spectra_dtype": np.array(config.spectra_dtype),
        "spectra_downsample_factor": np.array(config.spectra_downsample_factor, dtype=np.int32),
        "spectra_downsample_method": np.array(config.spectra_downsample_method),
        "spectrum_normalization": np.array(config.spectrum_normalization),
        "spectra_norm_method": np.array(config.spectrum_normalization),
        "config_json": np.array(config_json),
        "timestamp": np.array(timestamp),
    }
    print(f"[Merge] Writing final NPZ: {path}")
    if config.compress_npz:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


CSV_SPECTRAL_FEATURE_NAMES = [
    "spec_mean",
    "spec_std",
    "spec_ptp",
    "fft_peak_pos_1_um",
    "fft_peak_height_1",
    "fft_peak_pos_2_um",
    "fft_peak_height_ratio_21",
    "fft_noise_floor",
    "fft_snr_1",
    "fringe_visibility_global",
    "fringe_contrast_std",
]


def save_csv_index_from_checkpoints(
    run_dir: Path,
    timestamp: str,
    paths: list[Path],
    nominal_stacks_nm: list[dict],
) -> Path:
    path = run_dir / f"nn_cavity_spectral_features_index_{timestamp}.csv"
    feature_index = {name: idx for idx, name in enumerate(SPECTRAL_FEATURE_NAMES.tolist())}
    fieldnames = [
        "sample_id",
        "process_id",
        "nominal_stack_id",
        "nominal_stack_name",
        "split_label",
        "cavity_true_um",
        "L_fft_um",
        "delta_L_nm",
        "H_peak",
        "peak_count",
    ]
    for kind in ["nominal", "delta", "true"]:
        fieldnames.extend([f"film_{name}_{kind}_nm" for name in FILM_LAYER_NAMES])
    fieldnames.extend(CSV_SPECTRAL_FEATURE_NAMES)

    print(f"[Merge] Writing CSV index: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for checkpoint_idx, checkpoint in enumerate(paths):
            with np.load(checkpoint, allow_pickle=True) as data:
                for row_idx in range(len(data["sample_id"])):
                    nominal_id = int(data["nominal_stack_id"][row_idx])
                    split_id = int(data["split_id"][row_idx])
                    row = {
                        "sample_id": int(data["sample_id"][row_idx]),
                        "process_id": int(data["process_id"][row_idx]),
                        "nominal_stack_id": nominal_id,
                        "nominal_stack_name": nominal_stacks_nm[nominal_id]["name"],
                        "split_label": str(SPLIT_NAMES[split_id]),
                        "cavity_true_um": float(data["cavity_true_um"][row_idx]),
                        "L_fft_um": (
                            float(data["L_fft_um"][row_idx])
                            if np.isfinite(data["L_fft_um"][row_idx])
                            else ""
                        ),
                        "delta_L_nm": (
                            float(data["delta_L_nm"][row_idx])
                            if np.isfinite(data["delta_L_nm"][row_idx])
                            else ""
                        ),
                        "H_peak": (
                            float(data["H_peak"][row_idx])
                            if np.isfinite(data["H_peak"][row_idx])
                            else ""
                        ),
                        "peak_count": int(data["peak_count"][row_idx]),
                    }
                    for kind, key in [
                        ("nominal", "film_nominal_nm"),
                        ("delta", "film_delta_nm"),
                        ("true", "film_true_nm"),
                    ]:
                        for layer_idx, layer_name in enumerate(FILM_LAYER_NAMES):
                            row[f"film_{layer_name}_{kind}_nm"] = float(data[key][row_idx, layer_idx])
                    if data["spectral_features_full"].shape[1] > 0:
                        for feature_name in CSV_SPECTRAL_FEATURE_NAMES:
                            value = data["spectral_features_full"][row_idx, feature_index[feature_name]]
                            row[feature_name] = float(value) if np.isfinite(value) else ""
                    writer.writerow(row)
            if checkpoint_idx % 100 == 0 or checkpoint_idx == len(paths) - 1:
                print(f"[Merge] CSV checkpoint {checkpoint_idx:04d}/{len(paths) - 1:04d}")
    return path


def sample_for_plot(values: np.ndarray, max_count: int = 100000) -> np.ndarray:
    if len(values) <= max_count:
        return np.arange(len(values))
    return np.linspace(0, len(values) - 1, max_count, dtype=np.int64)


def safe_hist(ax, values: np.ndarray, max_bins: int = 60) -> None:
    """对常量或近常量数组也能稳定绘制直方图。"""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    value_range = float(np.max(finite) - np.min(finite))
    scale = max(1.0, float(np.max(np.abs(finite))))
    bins = 1 if value_range <= np.finfo(np.float64).eps * scale * 16 else min(max_bins, max(10, int(np.sqrt(finite.size))))
    ax.hist(finite, bins=bins)


def save_check_plots(
    run_dir: Path,
    memmaps: dict[str, np.memmap],
    config: Config,
    split_info: dict,
    nominal_stacks_nm: list[dict],
    first_checkpoint: Path,
) -> None:
    print("[Plot] Saving version-2 validation plots...")
    rows = sample_for_plot(memmaps["sample_id"])

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(memmaps["cavity_true_um"][rows], memmaps["L_fft_um"][rows], s=3, alpha=0.25)
    finite = np.isfinite(memmaps["L_fft_um"][rows])
    if np.any(finite):
        lo = min(
            float(np.min(memmaps["cavity_true_um"][rows])),
            float(np.min(memmaps["L_fft_um"][rows][finite])),
        )
        hi = max(
            float(np.max(memmaps["cavity_true_um"][rows])),
            float(np.max(memmaps["L_fft_um"][rows][finite])),
        )
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="True cavity (um)", ylabel="L_fft (um)", title="FFT Coarse Length vs True Cavity")
    fig.savefig(run_dir / "01_fft_vs_true_cavity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    finite_delta = np.asarray(memmaps["delta_L_nm"])[np.isfinite(memmaps["delta_L_nm"])]
    safe_hist(ax, finite_delta, max_bins=80)
    ax.set(xlabel="delta_L_nm", ylabel="Count", title="delta_L_nm Histogram")
    fig.savefig(run_dir / "02_delta_L_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    finite_h = np.asarray(memmaps["H_peak"])[np.isfinite(memmaps["H_peak"])]
    safe_hist(ax, finite_h, max_bins=80)
    ax.set(xlabel="H_peak", ylabel="Count", title="H_peak Histogram")
    fig.savefig(run_dir / "03_h_peak_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for y, (name, key) in enumerate(
        [("train", "train_process_ids"), ("val", "val_process_ids"), ("test", "test_process_ids")]
    ):
        ids = split_info[key]
        ax.scatter(ids, np.full(len(ids), y), s=10, label=name)
    ax.set_yticks([0, 1, 2], labels=["train", "val", "test"])
    ax.set(xlabel="process_id", title="Process Split")
    fig.savefig(run_dir / "04_process_split.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    process_delta = memmaps["film_delta_nm"][:: config.num_cavity_points]
    for idx, layer in enumerate(FILM_LAYER_NAMES):
        safe_hist(axes.flat[idx], process_delta[:, idx], max_bins=50)
        axes.flat[idx].set(title=f"{layer} process delta", xlabel="nm")
    fig.savefig(run_dir / "05_film_delta_distribution.png", dpi=200)
    plt.close(fig)

    nominal_values = np.asarray(
        [[stack[name] for name in FILM_LAYER_NAMES] for stack in nominal_stacks_nm]
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for idx, layer in enumerate(FILM_LAYER_NAMES):
        safe_hist(axes.flat[idx], nominal_values[:, idx], max_bins=20)
        axes.flat[idx].set(title=f"{layer} nominal coverage", xlabel="nm")
    fig.savefig(run_dir / "06_nominal_model_coverage.png", dpi=200)
    plt.close(fig)

    if config.save_spectra:
        with np.load(first_checkpoint, allow_pickle=True) as data:
            candidate_keys = [
                key
                for key in ["spectra_norm_ds", "spectra_norm", "spectra_ds"]
                if key in data.files
            ]
            if candidate_keys:
                key = candidate_keys[0]
                x = data["wavelengths_spectra_saved_um"]
                fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
                for row_idx in np.linspace(0, len(data[key]) - 1, 6, dtype=int):
                    ax.plot(x, data[key][row_idx], lw=0.8, alpha=0.8)
                ax.set(xlabel="Wavelength (um)", ylabel=key, title=f"Example {key}")
                fig.savefig(run_dir / "07_example_spectra_norm_ds.png", dpi=200)
                plt.close(fig)

    if memmaps["spectral_features_full"].shape[1] > 0:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for idx, feature_name in enumerate(SPECTRAL_FEATURE_NAMES[:4]):
            values = memmaps["spectral_features_full"][rows, idx]
            values = values[np.isfinite(values)]
            safe_hist(axes.flat[idx], values, max_bins=60)
            axes.flat[idx].set_title(str(feature_name))
        fig.savefig(run_dir / "08_spectral_feature_hist.png", dpi=200)
        plt.close(fig)


def validate_merged_shapes(
    memmaps: dict[str, np.memmap],
    config: Config,
    wavelengths_saved_um: np.ndarray,
) -> None:
    n = len(memmaps["sample_id"])
    assert n == len(memmaps["cavity_true_um"]) == len(memmaps["L_fft_um"])
    assert memmaps["film_nominal_nm"].shape == (n, len(FILM_LAYER_NAMES))
    assert memmaps["spectral_features_full"].shape[0] == n
    assert memmaps["spectral_features_full"].shape[1] == (
        len(SPECTRAL_FEATURE_NAMES) if config.extract_full_spectral_features else 0
    )
    for key in spectra_keys_for_mode(config):
        assert memmaps[key].shape == (n, len(wavelengths_saved_um))


def merge_final_outputs(
    run_dir: Path,
    config: Config,
    nominal_stacks_nm: list[dict],
    process_metadata: dict[str, np.ndarray],
    split_info: dict,
    simulation_info: dict,
) -> dict:
    total_processes = config.num_nominal_models * config.num_process_per_nominal
    paths = checkpoint_paths(run_dir, total_processes)
    total_rows = total_processes * config.num_cavity_points
    sample_keys = SAMPLE_ARRAY_KEYS + spectra_keys_for_mode(config)
    temp_dir = safe_recreate_merge_dir(run_dir)
    timestamp = config.run_timestamp

    try:
        with np.load(paths[0], allow_pickle=True) as first:
            memmaps = allocate_merge_memmaps(temp_dir, first, total_rows, sample_keys)
        loaded_rows = fill_merge_memmaps(paths, memmaps, sample_keys)
        if loaded_rows != total_rows:
            raise RuntimeError(f"Merged rows={loaded_rows}, expected={total_rows}")

        validate_merged_shapes(
            memmaps,
            config,
            simulation_info["wavelengths_spectra_saved_um"],
        )
        print("[Validate] Merged array shape checks passed")

        npz_path = save_final_npz(
            run_dir,
            config,
            timestamp,
            memmaps,
            nominal_stacks_nm,
            process_metadata,
            split_info,
            simulation_info["wavelengths_um"],
            simulation_info["wavelengths_spectra_saved_um"],
        )
        csv_path = (
            save_csv_index_from_checkpoints(run_dir, timestamp, paths, nominal_stacks_nm)
            if config.save_csv_index
            else None
        )
        if config.save_check_plots:
            save_check_plots(
                run_dir,
                memmaps,
                config,
                split_info,
                nominal_stacks_nm,
                paths[0],
            )

        summary = build_summary(
            config,
            memmaps,
            nominal_stacks_nm,
            split_info,
            simulation_info["wavelengths_um"],
            simulation_info["wavelengths_spectra_saved_um"],
        )
        summary_path = run_dir / f"summary_{timestamp}.json"
        dump_json(summary_path, summary)
        valid_summary_path = run_dir / "07_valid_mask_summary.json"
        dump_json(
            valid_summary_path,
            {
                "output_mode": OUTPUT_MODE,
                "num_samples_total": summary["num_samples_total"],
                "num_samples_valid": summary["num_samples_valid"],
                "num_failed_simulation": summary["num_failed_simulation"],
                "num_failed_fft": summary["num_failed_fft"],
            },
        )
        return {
            "npz_path": npz_path,
            "csv_path": csv_path,
            "summary_path": summary_path,
            "valid_summary_path": valid_summary_path,
            "summary": summary,
        }
    finally:
        # 先释放 memmap 引用，再删除本脚本创建的临时目录。
        if "memmaps" in locals():
            for array in memmaps.values():
                array.flush()
                mmap_handle = getattr(array, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
            memmaps.clear()
            gc.collect()
        if temp_dir.exists() and temp_dir.parent == run_dir.resolve():
            try:
                shutil.rmtree(temp_dir)
            except PermissionError as exc:
                # Windows 可能仍短暂持有 numpy 视图句柄；保留临时目录比误删更安全。
                print(f"[Warning] Could not remove merge temp directory yet: {exc}")


def print_configuration(config: Config) -> None:
    wavelengths = make_wavelength_axis_um(config)
    saved = saved_wavelength_axis(wavelengths, config)
    total_samples = config.num_nominal_models * config.num_process_per_nominal * config.num_cavity_points
    print("=== Build NN Cavity Dataset Version 2 ===")
    print(f"Output mode: {OUTPUT_MODE}")
    print(f"Nominal models: {config.num_nominal_models}")
    print(f"Processes per nominal: {config.num_process_per_nominal}")
    print(f"Cavity points per process: {config.num_cavity_points}")
    print(f"Total planned samples: {total_samples}")
    print(f"Full wavelengths: {len(wavelengths)}")
    print(f"Saved wavelengths: {len(saved)}")
    print(f"Spectra save mode: {config.spectra_save_mode}")
    print(f"Spectra dtype: {config.spectra_dtype}")
    print(f"Downsample factor/method: {config.spectra_downsample_factor}/{config.spectra_downsample_method}")
    print(
        f"Estimated spectra storage: "
        f"{estimate_spectra_storage_gb(config, total_samples, len(saved)):.3f} GB"
    )
    print(f"Full-spectrum features extracted: {config.extract_full_spectral_features}")
    print(
        f"Number of full-spectrum features: "
        f"{len(SPECTRAL_FEATURE_NAMES) if config.extract_full_spectral_features else 0}"
    )


def print_final_report(run_dir: Path, paths: dict) -> None:
    summary = paths["summary"]
    print("")
    print(f"Dataset saved to: {run_dir}")
    print(f"NPZ path: {paths['npz_path']}")
    print(f"CSV path: {paths['csv_path']}")
    print(f"Summary path: {paths['summary_path']}")
    print(f"Output mode: {summary['output_mode']}")
    print(f"Raw spectra saved: {summary['raw_full_spectra_saved']}")
    print(f"Normalized spectra saved: {summary['normalized_spectra_saved']}")
    print(f"Spectra save mode: {summary['spectra_save_mode']}")
    print(f"Full wavelengths: {summary['num_wavelengths_full']}")
    print(f"Saved wavelengths: {summary['num_wavelengths_saved']}")
    print(f"Spectra dtype: {summary['spectra_dtype']}")
    print(f"Downsample factor: {summary['spectra_downsample_factor']}")
    print(f"Downsample method: {summary['spectra_downsample_method']}")
    print(f"Estimated spectra storage: {summary['estimated_spectra_storage_gb']:.3f} GB")
    print(f"Total planned samples: {summary['num_samples_planned']}")
    print(f"Saved samples: {summary['num_samples_total']}")
    print(f"Valid samples: {summary['num_samples_valid']}")
    print(f"Failed simulations: {summary['num_failed_simulation']}")
    print(f"Failed FFT: {summary['num_failed_fft']}")
    print(
        "Train/Val/Test processes: "
        f"{summary['num_train_processes']}/"
        f"{summary['num_val_processes']}/"
        f"{summary['num_test_processes']}"
    )
    print(
        "delta_L_nm mean/std/min/max: "
        f"{summary['delta_L_nm_mean']}/{summary['delta_L_nm_std']}/"
        f"{summary['delta_L_nm_min']}/{summary['delta_L_nm_max']}"
    )
    print(f"Full-spectrum features extracted: {summary['num_spectral_features'] > 0}")
    print(f"Number of full-spectrum features: {summary['num_spectral_features']}")
    print(f"Spectral feature source: {summary['spectral_feature_source']}")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        config = config_from_json(run_dir / "00_config.json")
        print(f"[Resume] Loaded immutable configuration from {run_dir / '00_config.json'}")
    else:
        config = config_from_args(args)
        config.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = script_dir / f"nn_cavity_spectral_features_{config.run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    config.validate()
    nominal_stacks_nm = build_nominal_stacks_nm(config)
    process_metadata = build_process_metadata(config, nominal_stacks_nm)
    split_info = split_processes_within_nominal(config, process_metadata)
    print_configuration(config)

    if args.resume_run_dir is None:
        save_initial_run_files(run_dir, config, nominal_stacks_nm)

    simulation_info = run_simulation(
        config,
        run_dir,
        nominal_stacks_nm,
        process_metadata,
        split_info,
    )

    total_processes = config.num_nominal_models * config.num_process_per_nominal
    completed = len(existing_checkpoint_ids(run_dir))
    if args.skip_final_merge:
        print(f"[Done] Checkpoints complete: {completed}/{total_processes}; final merge skipped")
        return
    if completed != total_processes:
        raise RuntimeError(f"Only {completed}/{total_processes} checkpoints exist; refusing final merge")

    paths = merge_final_outputs(
        run_dir,
        config,
        nominal_stacks_nm,
        process_metadata,
        split_info,
        simulation_info,
    )
    print_final_report(run_dir, paths)


if __name__ == "__main__":
    main()
