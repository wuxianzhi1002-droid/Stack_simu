import csv
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Lumerical path setup follows 01_Lumerical_Workflow/main_cavity.py.
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


FILM_LAYER_NAMES = ["HSQ", "PSS", "SOC", "TiO2"]

NOMINAL_STACKS_NM = [
    {
        "name": "model1",
        "HSQ": 40.0,
        "PSS": 5.0,
        "SOC": 50.0,
        "TiO2": 20.0,
    },
    {
        "name": "model3_soc60",
        "HSQ": 40.0,
        "PSS": 5.0,
        "SOC": 60.0,
        "TiO2": 20.0,
    },
]

FULL_NOMINAL_STACKS_NM = [
    {
        "name": "model1",
        "HSQ": 40.0,
        "PSS": 5.0,
        "SOC": 50.0,
        "TiO2": 20.0,
    },
    {
        "name": "model2_hsq20",
        "HSQ": 20.0,
        "PSS": 5.0,
        "SOC": 50.0,
        "TiO2": 20.0,
    },
    {
        "name": "model3_soc60",
        "HSQ": 40.0,
        "PSS": 5.0,
        "SOC": 60.0,
        "TiO2": 20.0,
    },
    {
        "name": "model4_hsq60",
        "HSQ": 60.0,
        "PSS": 5.0,
        "SOC": 50.0,
        "TiO2": 20.0,
    },
    {
        "name": "model5_soc80",
        "HSQ": 40.0,
        "PSS": 5.0,
        "SOC": 80.0,
        "TiO2": 20.0,
    },
]


@dataclass
class Config:
    output_root: str = "../ML try"
    model_type: str = "PSS_TiO2"

    wavelength_start_um: float = 0.2
    wavelength_stop_um: float = 0.6
    spectral_resolution_nm: float = 0.02

    angle_deg: float = 0.0
    polarization: str = "p"

    cavity_start_um: float = 1000.0
    cavity_stop_um: float = 1000.03
    cavity_step_um: float = 0.01

    film_uncertainty_nm: float = 10.0
    num_process_per_nominal: int = 2
    random_seed: int = 20260613

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    save_checkpoint_every_process: bool = True

    fft_peak_height_ratio: float = 0.2
    fft_ignore_dc_bins: int = 50
    fft_peak_distance_bins: int = 100
    zero_pad_factor: int = 8

    add_noise: bool = False
    noise_std: float = 0.0


def config_to_dict(config):
    config_dict = asdict(config)
    config_dict["film_layer_names"] = FILM_LAYER_NAMES
    config_dict["nominal_stacks_nm"] = NOMINAL_STACKS_NM
    config_dict["full_nominal_stacks_nm"] = FULL_NOMINAL_STACKS_NM
    config_dict["lumerical_path"] = str(LUMERICAL_PATH)
    config_dict["lumerical_bin_path"] = str(LUMERICAL_BIN_PATH)
    return config_dict


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def dump_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)


def build_layers_from_nominal_stack(nominal_stack_nm, cavity_um, film_delta_nm=None):
    if film_delta_nm is None:
        film_delta_nm = np.zeros(len(FILM_LAYER_NAMES), dtype=float)
    else:
        film_delta_nm = np.asarray(film_delta_nm, dtype=float).reshape(-1)

    if film_delta_nm.size != len(FILM_LAYER_NAMES):
        raise ValueError(
            f"film_delta_nm must contain {len(FILM_LAYER_NAMES)} values, got {film_delta_nm.size}."
        )

    delta_by_layer_nm = dict(zip(FILM_LAYER_NAMES, film_delta_nm))
    hsq_um = (float(nominal_stack_nm["HSQ"]) + delta_by_layer_nm["HSQ"]) / 1000.0
    pss_um = (float(nominal_stack_nm["PSS"]) + delta_by_layer_nm["PSS"]) / 1000.0
    soc_um = (float(nominal_stack_nm["SOC"]) + delta_by_layer_nm["SOC"]) / 1000.0
    tio2_um = (float(nominal_stack_nm["TiO2"]) + delta_by_layer_nm["TiO2"]) / 1000.0

    return [
        ("RefReflector", 0),
        ("Air", float(cavity_um)),
        ("HSQ", hsq_um),
        ("PSS", pss_um),
        ("SOC", soc_um),
        ("TiO2", tio2_um),
        ("Cu", 0),
    ]


class StackRTSimulator:
    def __init__(self, config):
        self.config = config
        span_nm = (config.wavelength_stop_um - config.wavelength_start_um) * 1000.0
        num_wavelengths = int(round(span_nm / config.spectral_resolution_nm)) + 1
        self.wavelengths_um = np.linspace(
            config.wavelength_start_um,
            config.wavelength_stop_um,
            num_wavelengths,
            dtype=float,
        )
        self.freqs = 3e8 / (self.wavelengths_um * 1e-6)
        self.fdtd = None

    def open(self):
        if lumapi is None:
            raise RuntimeError("lumapi is not available. Please check Lumerical path.")
        print("[StackRT] Opening one Lumerical FDTD API session...")
        self.fdtd = lumapi.FDTD(hide=True)

    def close(self):
        if self.fdtd is not None:
            print("[StackRT] Closing Lumerical FDTD API session...")
            self.fdtd.close()
            self.fdtd = None

    def get_n_matrix_and_thicknesses(self, layers):
        n_matrix = np.zeros((len(layers), len(self.freqs)), dtype=complex)
        thicknesses = []

        w_um = self.wavelengths_um
        for i, (material, thick_um) in enumerate(layers):
            thicknesses.append(float(thick_um) * 1e-6)

            if isinstance(material, (int, float, complex)):
                n_matrix[i, :] = material
            elif material == "RefReflector":
                n_matrix[i, :] = 5.8284
            elif material == "Air":
                n_matrix[i, :] = 1.0
            elif material == "HSQ":
                n_matrix[i, :] = 1.41
            elif material == "PSS":
                n_matrix[i, :] = 1.50 + 0.05j
            elif material == "SOC":
                n_matrix[i, :] = 1.55 + 0.005 / (w_um**2)
            elif material == "TiO2":
                n_matrix[i, :] = 2.4 + 0.02 / (w_um**2)
            elif material == "Cu":
                n_matrix[i, :] = 1.1 + 2.5j
            else:
                n_matrix[i, :] = 1.5

        return n_matrix, np.asarray(thicknesses, dtype=float)

    def simulate_spectrum(self, layers):
        if self.fdtd is None:
            raise RuntimeError("StackRTSimulator.open() must be called before simulate_spectrum().")

        n_matrix, thicknesses = self.get_n_matrix_and_thicknesses(layers)
        result_key = "Rp" if self.config.polarization.lower() == "p" else "Rs"
        res = self.fdtd.stackrt(n_matrix, thicknesses, self.freqs, float(self.config.angle_deg))
        spectrum = np.real(np.asarray(res[result_key]).flatten())

        if spectrum.size != self.wavelengths_um.size:
            raise ValueError(
                f"StackRT returned {spectrum.size} points, expected {self.wavelengths_um.size}."
            )

        return spectrum


def nominal_film_array_nm(nominal_stack_nm):
    return np.asarray([float(nominal_stack_nm[name]) for name in FILM_LAYER_NAMES], dtype=float)


def checkpoint_dataset(run_dir, process_id, records, wavelengths_um, cavity_axis_um, failed_cases):
    checkpoint_path = run_dir / f"checkpoint_process_{process_id:04d}.npz"
    partial = records_to_arrays(records, wavelengths_um, cavity_axis_um)
    np.savez_compressed(
        checkpoint_path,
        wavelengths_um=partial["wavelengths_um"],
        spectra=partial["spectra"],
        sample_id=partial["sample_id"],
        process_id=partial["process_id"],
        nominal_stack_id=partial["nominal_stack_id"],
        nominal_stack_name=partial["nominal_stack_name"],
        cavity_true_um=partial["cavity_true_um"],
        film_nominal_nm=partial["film_nominal_nm"],
        film_delta_nm=partial["film_delta_nm"],
        film_true_nm=partial["film_true_nm"],
        layer_names=np.asarray(FILM_LAYER_NAMES, dtype=str),
        cavity_axis_um=partial["cavity_axis_um"],
        num_failed_simulation=np.array(len(failed_cases), dtype=int),
    )
    print(f"[Checkpoint] Saved {checkpoint_path}")


def records_to_arrays(records, wavelengths_um, cavity_axis_um):
    num_samples = len(records)
    num_wavelengths = len(wavelengths_um)
    num_layers = len(FILM_LAYER_NAMES)

    spectra = np.empty((num_samples, num_wavelengths), dtype=float)
    sample_id = np.empty(num_samples, dtype=int)
    process_id = np.empty(num_samples, dtype=int)
    nominal_stack_id = np.empty(num_samples, dtype=int)
    nominal_stack_name = np.empty(num_samples, dtype=object)
    cavity_true_um = np.empty(num_samples, dtype=float)
    film_nominal_nm = np.empty((num_samples, num_layers), dtype=float)
    film_delta_nm = np.empty((num_samples, num_layers), dtype=float)
    film_true_nm = np.empty((num_samples, num_layers), dtype=float)

    for i, record in enumerate(records):
        spectra[i, :] = record["spectrum"]
        sample_id[i] = record["sample_id"]
        process_id[i] = record["process_id"]
        nominal_stack_id[i] = record["nominal_stack_id"]
        nominal_stack_name[i] = record["nominal_stack_name"]
        cavity_true_um[i] = record["cavity_true_um"]
        film_nominal_nm[i, :] = record["film_nominal_nm"]
        film_delta_nm[i, :] = record["film_delta_nm"]
        film_true_nm[i, :] = record["film_true_nm"]

    return {
        "wavelengths_um": np.asarray(wavelengths_um, dtype=float),
        "spectra": spectra,
        "sample_id": sample_id,
        "process_id": process_id,
        "nominal_stack_id": nominal_stack_id,
        "nominal_stack_name": nominal_stack_name.astype(str),
        "cavity_true_um": cavity_true_um,
        "film_nominal_nm": film_nominal_nm,
        "film_delta_nm": film_delta_nm,
        "film_true_nm": film_true_nm,
        "layer_names": np.asarray(FILM_LAYER_NAMES, dtype=str),
        "cavity_axis_um": np.asarray(cavity_axis_um, dtype=float),
    }


def generate_dataset(config, run_dir):
    print("[Dataset] Preparing wavelength and cavity axes...")
    simulator = StackRTSimulator(config)
    cavity_axis_um = np.round(
        np.arange(
            config.cavity_start_um,
            config.cavity_stop_um + config.cavity_step_um / 2.0,
            config.cavity_step_um,
            dtype=float,
        ),
        12,
    )

    print(f"[Dataset] Wavelength points: {simulator.wavelengths_um.size}")
    print(
        "[Dataset] Cavity axis: "
        f"{config.cavity_start_um:g} um to {config.cavity_stop_um:g} um, "
        f"step {config.cavity_step_um:g} um, points {cavity_axis_um.size}"
    )

    rng = np.random.default_rng(config.random_seed)
    records = []
    failed_cases = []
    sample_id = 0
    process_id = 0
    total_processes = len(NOMINAL_STACKS_NM) * config.num_process_per_nominal

    simulator.open()
    start_time = time.time()
    try:
        for nominal_stack_id, nominal_stack_nm in enumerate(NOMINAL_STACKS_NM):
            nominal_name = str(nominal_stack_nm["name"])
            film_nominal_nm = nominal_film_array_nm(nominal_stack_nm)
            print(f"[Dataset] Nominal stack {nominal_stack_id}: {nominal_name}")

            for process_idx in range(config.num_process_per_nominal):
                if process_idx == 0:
                    film_delta_nm = np.zeros(len(FILM_LAYER_NAMES), dtype=float)
                else:
                    film_delta_nm = rng.uniform(
                        -config.film_uncertainty_nm,
                        config.film_uncertainty_nm,
                        size=len(FILM_LAYER_NAMES),
                    )
                film_true_nm = film_nominal_nm + film_delta_nm

                print(
                    "[Dataset] Process "
                    f"{process_id + 1}/{total_processes} "
                    f"(process_id={process_id}, nominal={nominal_name}, "
                    f"delta_nm={np.round(film_delta_nm, 4).tolist()})"
                )

                for cavity_idx, cavity_um in enumerate(cavity_axis_um):
                    try:
                        layers = build_layers_from_nominal_stack(
                            nominal_stack_nm,
                            cavity_um,
                            film_delta_nm,
                        )
                        spectrum = simulator.simulate_spectrum(layers)
                        if config.add_noise:
                            spectrum = spectrum + rng.normal(
                                0.0,
                                config.noise_std,
                                size=spectrum.shape,
                            )

                        records.append(
                            {
                                "sample_id": sample_id,
                                "process_id": process_id,
                                "nominal_stack_id": nominal_stack_id,
                                "nominal_stack_name": nominal_name,
                                "cavity_true_um": float(cavity_um),
                                "film_nominal_nm": film_nominal_nm.copy(),
                                "film_delta_nm": film_delta_nm.copy(),
                                "film_true_nm": film_true_nm.copy(),
                                "spectrum": spectrum,
                            }
                        )
                        sample_id += 1
                    except Exception as exc:
                        failed_case = {
                            "process_id": int(process_id),
                            "nominal_stack_id": int(nominal_stack_id),
                            "nominal_stack_name": nominal_name,
                            "process_idx": int(process_idx),
                            "cavity_idx": int(cavity_idx),
                            "cavity_true_um": float(cavity_um),
                            "film_nominal_nm": film_nominal_nm.tolist(),
                            "film_delta_nm": film_delta_nm.tolist(),
                            "film_true_nm": film_true_nm.tolist(),
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failed_cases.append(failed_case)
                        print(
                            "[Warning] Simulation failed and was skipped: "
                            f"process_id={process_id}, cavity_um={cavity_um:g}, error={exc}"
                        )

                if config.save_checkpoint_every_process:
                    checkpoint_dataset(
                        run_dir,
                        process_id,
                        records,
                        simulator.wavelengths_um,
                        cavity_axis_um,
                        failed_cases,
                    )

                elapsed = time.time() - start_time
                print(
                    f"[Dataset] Finished process_id={process_id}; "
                    f"samples={len(records)}, failed={len(failed_cases)}, elapsed={elapsed:.1f}s"
                )
                process_id += 1
    finally:
        simulator.close()

    dataset = records_to_arrays(records, simulator.wavelengths_um, cavity_axis_um)
    dataset["failed_cases"] = failed_cases
    dataset["num_processes"] = np.array(process_id, dtype=int)
    return dataset


def fft_config_dict(config):
    return {
        "FFT_PEAK_HEIGHT_RATIO": config.fft_peak_height_ratio,
        "FFT_IGNORE_DC_BINS": config.fft_ignore_dc_bins,
        "FFT_PEAK_DISTANCE_BINS": config.fft_peak_distance_bins,
        "ZERO_PAD_FACTOR": config.zero_pad_factor,
    }


def solve_single_fft(wavelengths_um, intensities, config_dict):
    """Copied from solve_npz_fft.py FFTSolver.solve, adapted for dataset generation."""
    wavelengths_um = np.asarray(wavelengths_um, dtype=float).reshape(-1)
    intensities = np.asarray(intensities, dtype=float).reshape(-1)

    if wavelengths_um.size != intensities.size:
        raise ValueError(
            f"wavelengths and intensities must have the same length, got "
            f"{wavelengths_um.size} and {intensities.size}."
        )

    finite_mask = np.isfinite(wavelengths_um) & np.isfinite(intensities)
    wavelengths_um = wavelengths_um[finite_mask]
    intensities = intensities[finite_mask]

    if wavelengths_um.size < 4:
        raise ValueError("At least 4 finite points are required for FFT solving.")

    sort_idx = np.argsort(wavelengths_um)
    wavelengths_um = wavelengths_um[sort_idx]
    intensities = intensities[sort_idx]

    k_raw = 2.0 * np.pi / wavelengths_um
    k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))

    if k_raw[0] > k_raw[-1]:
        i_linear = np.interp(k_linear, k_raw[::-1], intensities[::-1])
    else:
        i_linear = np.interp(k_linear, k_raw, intensities)

    i_detrend = i_linear - np.mean(i_linear)
    i_windowed = i_detrend * np.hanning(len(i_detrend))

    n_fft = len(i_windowed) * int(config_dict["ZERO_PAD_FACTOR"])
    fft_data = np.abs(np.fft.rfft(i_windowed, n=n_fft))

    dk = abs(k_linear[1] - k_linear[0])
    max_range_um = np.pi / dk
    distance_axis_um = np.linspace(0.0, max_range_um / 2.0, len(fft_data))

    ignore = min(int(config_dict["FFT_IGNORE_DC_BINS"]), max(0, len(fft_data) - 1))
    search = fft_data[ignore:]
    if search.size == 0 or np.max(search) <= 0:
        peaks = np.array([], dtype=int)
    else:
        peaks, _ = find_peaks(
            search,
            height=np.max(search) * float(config_dict["FFT_PEAK_HEIGHT_RATIO"]),
            distance=int(config_dict["FFT_PEAK_DISTANCE_BINS"]),
        )
        peaks = peaks + ignore

    return {
        "distance_axis_um": distance_axis_um,
        "fft_data": fft_data,
        "peaks_idx": peaks,
        "peak_distances_um": distance_axis_um[peaks],
        "peak_heights": fft_data[peaks],
        "max_range_um": max_range_um / 2.0,
    }


def solve_fft_for_dataset(dataset, config):
    wavelengths_um = dataset["wavelengths_um"]
    spectra = dataset["spectra"]
    num_samples = spectra.shape[0]

    print(f"[FFT] Solving FFT coarse cavity length for {num_samples} spectra...")
    L_fft_um = np.full(num_samples, np.nan, dtype=float)
    H_peak = np.full(num_samples, np.nan, dtype=float)
    peak_count = np.zeros(num_samples, dtype=int)
    failed_fft_cases = []
    config_dict = fft_config_dict(config)

    for i in range(num_samples):
        try:
            res = solve_single_fft(wavelengths_um, spectra[i, :], config_dict)
            peak_distances_um = np.asarray(res["peak_distances_um"], dtype=float)
            peak_heights = np.asarray(res["peak_heights"], dtype=float)
            peak_count[i] = peak_distances_um.size

            if peak_distances_um.size == 0:
                failed_fft_cases.append(
                    {
                        "sample_id": int(dataset["sample_id"][i]),
                        "process_id": int(dataset["process_id"][i]),
                        "reason": "no peak detected",
                    }
                )
                continue

            dominant_idx = int(np.argmax(peak_heights))
            L_fft_um[i] = peak_distances_um[dominant_idx]
            H_peak[i] = peak_heights[dominant_idx]
        except Exception as exc:
            failed_fft_cases.append(
                {
                    "sample_id": int(dataset["sample_id"][i]),
                    "process_id": int(dataset["process_id"][i]),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        if (i + 1) % max(1, num_samples // 4) == 0:
            print(f"[FFT] Progress: {i + 1}/{num_samples}")

    print(f"[FFT] Done. Failed FFT cases: {len(failed_fft_cases)}")
    return L_fft_um, H_peak, peak_count, failed_fft_cases


def normalize_spectra(spectra):
    if spectra.size == 0:
        return spectra.astype(float, copy=True)
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    return (spectra - mean) / (std + 1e-12)


def split_by_process(process_ids, train_ratio, val_ratio, test_ratio, seed):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {ratio_sum}.")

    unique_process_ids = np.unique(np.asarray(process_ids, dtype=int))
    rng = np.random.default_rng(seed)
    shuffled = unique_process_ids.copy()
    rng.shuffle(shuffled)

    num_processes = shuffled.size
    if num_processes == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )
    if num_processes == 1:
        return np.sort(shuffled), np.array([], dtype=int), np.array([], dtype=int)
    if num_processes == 2:
        return np.sort(shuffled[:1]), np.array([], dtype=int), np.sort(shuffled[1:])

    num_train = int(np.floor(num_processes * train_ratio))
    num_val = int(np.floor(num_processes * val_ratio))

    num_train = min(max(1, num_train), num_processes - 2)
    num_val = min(max(1, num_val), num_processes - num_train - 1)

    train_process_ids = np.sort(shuffled[:num_train])
    val_process_ids = np.sort(shuffled[num_train : num_train + num_val])
    test_process_ids = np.sort(shuffled[num_train + num_val :])
    return train_process_ids, val_process_ids, test_process_ids


def labels_from_process_split(process_ids, train_process_ids, val_process_ids, test_process_ids):
    labels = np.empty(len(process_ids), dtype="<U5")
    train_set = set(np.asarray(train_process_ids, dtype=int).tolist())
    val_set = set(np.asarray(val_process_ids, dtype=int).tolist())
    test_set = set(np.asarray(test_process_ids, dtype=int).tolist())

    for i, process_id in enumerate(np.asarray(process_ids, dtype=int)):
        if process_id in train_set:
            labels[i] = "train"
        elif process_id in val_set:
            labels[i] = "val"
        elif process_id in test_set:
            labels[i] = "test"
        else:
            raise ValueError(f"process_id {process_id} is not present in any split.")

    return labels


def finite_stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def save_dataset_npz(run_dir, timestamp, dataset, config):
    npz_path = run_dir / f"nn_cavity_dataset_{timestamp}.npz"
    config_json = json.dumps(config_to_dict(config), ensure_ascii=False, default=json_default)

    np.savez_compressed(
        npz_path,
        wavelengths_um=dataset["wavelengths_um"],
        spectra=dataset["spectra"],
        spectra_norm=dataset["spectra_norm"],
        sample_id=dataset["sample_id"],
        process_id=dataset["process_id"],
        nominal_stack_id=dataset["nominal_stack_id"],
        nominal_stack_name=dataset["nominal_stack_name"],
        split_label=dataset["split_label"],
        cavity_true_um=dataset["cavity_true_um"],
        L_true_um=dataset["cavity_true_um"],
        L_fft_um=dataset["L_fft_um"],
        delta_L_um=dataset["delta_L_um"],
        delta_L_nm=dataset["delta_L_nm"],
        H_peak=dataset["H_peak"],
        peak_count=dataset["peak_count"],
        film_nominal_nm=dataset["film_nominal_nm"],
        film_delta_nm=dataset["film_delta_nm"],
        film_true_nm=dataset["film_true_nm"],
        layer_names=dataset["layer_names"],
        cavity_axis_um=dataset["cavity_axis_um"],
        train_process_ids=dataset["train_process_ids"],
        val_process_ids=dataset["val_process_ids"],
        test_process_ids=dataset["test_process_ids"],
        valid_mask=dataset["valid_mask"],
        config_json=np.array(config_json),
        timestamp=np.array(timestamp),
    )
    print(f"[Save] NPZ saved: {npz_path}")
    return npz_path


def save_csv_index(run_dir, timestamp, dataset):
    csv_path = run_dir / f"nn_cavity_dataset_index_{timestamp}.csv"
    layer_names = [str(name) for name in dataset["layer_names"]]
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
    fieldnames.extend([f"film_{name}_nominal_nm" for name in layer_names])
    fieldnames.extend([f"film_{name}_delta_nm" for name in layer_names])
    fieldnames.extend([f"film_{name}_true_nm" for name in layer_names])

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(dataset["spectra"].shape[0]):
            row = {
                "sample_id": int(dataset["sample_id"][i]),
                "process_id": int(dataset["process_id"][i]),
                "nominal_stack_id": int(dataset["nominal_stack_id"][i]),
                "nominal_stack_name": str(dataset["nominal_stack_name"][i]),
                "split_label": str(dataset["split_label"][i]),
                "cavity_true_um": float(dataset["cavity_true_um"][i]),
                "L_fft_um": float(dataset["L_fft_um"][i]) if np.isfinite(dataset["L_fft_um"][i]) else "",
                "delta_L_nm": float(dataset["delta_L_nm"][i])
                if np.isfinite(dataset["delta_L_nm"][i])
                else "",
                "H_peak": float(dataset["H_peak"][i]) if np.isfinite(dataset["H_peak"][i]) else "",
                "peak_count": int(dataset["peak_count"][i]),
            }
            for j, name in enumerate(layer_names):
                row[f"film_{name}_nominal_nm"] = float(dataset["film_nominal_nm"][i, j])
                row[f"film_{name}_delta_nm"] = float(dataset["film_delta_nm"][i, j])
                row[f"film_{name}_true_nm"] = float(dataset["film_true_nm"][i, j])
            writer.writerow(row)

    print(f"[Save] CSV saved: {csv_path}")
    return csv_path


def unique_process_film_delta_nm(dataset):
    process_ids = np.asarray(dataset["process_id"], dtype=int)
    if process_ids.size == 0:
        return np.empty((0, len(FILM_LAYER_NAMES)), dtype=float)

    unique_ids, first_indices = np.unique(process_ids, return_index=True)
    order = np.argsort(unique_ids)
    return dataset["film_delta_nm"][first_indices[order], :]


def save_check_plots(run_dir, dataset, config):
    print("[Plot] Saving check plots...")
    wavelengths_nm = dataset["wavelengths_um"] * 1000.0
    spectra = dataset["spectra"]

    rng = np.random.default_rng(config.random_seed + 1)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if spectra.shape[0] > 0:
        selected = rng.choice(spectra.shape[0], size=min(6, spectra.shape[0]), replace=False)
        for idx in np.sort(selected):
            ax.plot(
                wavelengths_nm,
                spectra[idx, :],
                lw=0.8,
                alpha=0.8,
                label=f"sample {int(dataset['sample_id'][idx])}",
            )
        ax.legend(fontsize=8)
    ax.set_title("Example Spectra")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "01_example_spectra.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    split_styles = {
        "train": {"marker": "o", "color": "#1f77b4"},
        "val": {"marker": "s", "color": "#ff7f0e"},
        "test": {"marker": "^", "color": "#2ca02c"},
    }
    for split_name, style in split_styles.items():
        mask = dataset["split_label"] == split_name
        if np.any(mask):
            ax.scatter(
                dataset["cavity_true_um"][mask],
                dataset["L_fft_um"][mask],
                s=24,
                alpha=0.75,
                label=split_name,
                marker=style["marker"],
                color=style["color"],
            )
    if dataset["cavity_true_um"].size > 0:
        finite_y = dataset["L_fft_um"][np.isfinite(dataset["L_fft_um"])]
        if finite_y.size > 0:
            min_axis = min(float(np.min(dataset["cavity_true_um"])), float(np.min(finite_y)))
            max_axis = max(float(np.max(dataset["cavity_true_um"])), float(np.max(finite_y)))
            ax.plot([min_axis, max_axis], [min_axis, max_axis], color="#444444", lw=1.0, ls="--")
    ax.set_title("FFT Coarse Length vs True Cavity")
    ax.set_xlabel("True cavity (um)")
    ax.set_ylabel("L_fft (um)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(run_dir / "02_fft_vs_true_cavity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    delta_L_nm = dataset["delta_L_nm"]
    finite_delta = delta_L_nm[np.isfinite(delta_L_nm)]
    if finite_delta.size > 0:
        ax.hist(finite_delta, bins=min(30, max(5, finite_delta.size)), color="#4c78a8", alpha=0.85)
    ax.set_title("delta_L_nm Histogram")
    ax.set_xlabel("delta_L_nm")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "03_delta_L_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    finite_h = dataset["H_peak"][np.isfinite(dataset["H_peak"])]
    if finite_h.size > 0:
        ax.hist(finite_h, bins=min(30, max(5, finite_h.size)), color="#59a14f", alpha=0.85)
    ax.set_title("H_peak Histogram")
    ax.set_xlabel("H_peak")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "04_h_peak_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    split_processes = [
        ("train", dataset["train_process_ids"], 0, "#1f77b4"),
        ("val", dataset["val_process_ids"], 1, "#ff7f0e"),
        ("test", dataset["test_process_ids"], 2, "#2ca02c"),
    ]
    for split_name, ids, y_value, color in split_processes:
        if len(ids) > 0:
            ax.scatter(ids, np.full(len(ids), y_value), s=60, label=split_name, color=color)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["train", "val", "test"])
    ax.set_title("Process Split")
    ax.set_xlabel("process_id")
    ax.set_ylabel("split")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.savefig(run_dir / "05_process_split.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.reshape(-1)
    process_delta_nm = unique_process_film_delta_nm(dataset)
    for i, layer_name in enumerate(dataset["layer_names"]):
        ax = axes[i]
        values = process_delta_nm[:, i] if process_delta_nm.size else np.array([])
        if values.size > 0:
            ax.hist(values, bins=min(20, max(4, values.size)), color="#b279a2", alpha=0.85)
        ax.set_title(f"{layer_name} delta")
        ax.set_xlabel("film_delta_nm")
        ax.set_ylabel("Process count")
        ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "06_film_delta_distribution.png", dpi=220)
    plt.close(fig)


def build_summary(dataset, config):
    valid_mask = dataset["valid_mask"]
    delta_stats = finite_stats(dataset["delta_L_nm"])
    num_processes = int(np.asarray(dataset.get("num_processes", np.unique(dataset["process_id"]).size)).item())
    return {
        "num_samples_total": int(dataset["spectra"].shape[0]),
        "num_samples_valid": int(np.count_nonzero(valid_mask)),
        "num_failed_simulation": int(len(dataset["failed_cases"])),
        "num_failed_fft": int(len(dataset["failed_fft_cases"])),
        "num_nominal_stacks": int(len(NOMINAL_STACKS_NM)),
        "num_processes": num_processes,
        "num_train_processes": int(len(dataset["train_process_ids"])),
        "num_val_processes": int(len(dataset["val_process_ids"])),
        "num_test_processes": int(len(dataset["test_process_ids"])),
        "wavelength_start_um": float(config.wavelength_start_um),
        "wavelength_stop_um": float(config.wavelength_stop_um),
        "num_wavelengths": int(dataset["wavelengths_um"].size),
        "cavity_start_um": float(config.cavity_start_um),
        "cavity_stop_um": float(config.cavity_stop_um),
        "cavity_step_um": float(config.cavity_step_um),
        "num_cavity_points": int(dataset["cavity_axis_um"].size),
        "film_layer_names": FILM_LAYER_NAMES,
        "film_uncertainty_nm": float(config.film_uncertainty_nm),
        "L_fft_nan_count": int(np.count_nonzero(~np.isfinite(dataset["L_fft_um"]))),
        "delta_L_nm_mean": delta_stats["mean"],
        "delta_L_nm_std": delta_stats["std"],
        "delta_L_nm_min": delta_stats["min"],
        "delta_L_nm_max": delta_stats["max"],
    }


def finalize_and_save(dataset, config, run_dir, timestamp):
    print("[Dataset] Normalizing spectra...")
    dataset["spectra_norm"] = normalize_spectra(dataset["spectra"])

    L_fft_um, H_peak, peak_count, failed_fft_cases = solve_fft_for_dataset(dataset, config)
    dataset["L_fft_um"] = L_fft_um
    dataset["H_peak"] = H_peak
    dataset["peak_count"] = peak_count
    dataset["failed_fft_cases"] = failed_fft_cases
    dataset["delta_L_um"] = dataset["cavity_true_um"] - dataset["L_fft_um"]
    dataset["delta_L_nm"] = dataset["delta_L_um"] * 1000.0

    print("[Split] Splitting dataset by process_id...")
    train_process_ids, val_process_ids, test_process_ids = split_by_process(
        dataset["process_id"],
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed,
    )
    dataset["train_process_ids"] = train_process_ids
    dataset["val_process_ids"] = val_process_ids
    dataset["test_process_ids"] = test_process_ids
    dataset["split_label"] = labels_from_process_split(
        dataset["process_id"],
        train_process_ids,
        val_process_ids,
        test_process_ids,
    )

    dataset["valid_mask"] = np.isfinite(dataset["L_fft_um"]) & np.all(
        np.isfinite(dataset["spectra"]),
        axis=1,
    )

    npz_path = save_dataset_npz(run_dir, timestamp, dataset, config)
    csv_path = save_csv_index(run_dir, timestamp, dataset)

    failed_cases_path = run_dir / f"failed_cases_{timestamp}.json"
    failed_fft_cases_path = run_dir / f"failed_fft_cases_{timestamp}.json"
    dump_json(failed_cases_path, dataset["failed_cases"])
    dump_json(failed_fft_cases_path, dataset["failed_fft_cases"])
    print(f"[Save] Failed simulation JSON saved: {failed_cases_path}")
    print(f"[Save] Failed FFT JSON saved: {failed_fft_cases_path}")

    save_check_plots(run_dir, dataset, config)

    valid_summary = {
        "num_samples_total": int(dataset["spectra"].shape[0]),
        "num_samples_valid": int(np.count_nonzero(dataset["valid_mask"])),
        "num_failed_fft": int(len(dataset["failed_fft_cases"])),
        "num_failed_simulation": int(len(dataset["failed_cases"])),
    }
    valid_summary_path = run_dir / "07_valid_mask_summary.json"
    dump_json(valid_summary_path, valid_summary)
    print(f"[Save] Valid-mask summary saved: {valid_summary_path}")

    summary = build_summary(dataset, config)
    summary_path = run_dir / f"summary_{timestamp}.json"
    dump_json(summary_path, summary)
    print(f"[Save] Summary saved: {summary_path}")

    return {
        "npz_path": npz_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "failed_cases_path": failed_cases_path,
        "failed_fft_cases_path": failed_fft_cases_path,
        "valid_summary_path": valid_summary_path,
        "summary": summary,
    }


def save_initial_config(run_dir, config):
    config_path = run_dir / "00_config.json"
    dump_json(config_path, config_to_dict(config))
    print(f"[Save] Config saved: {config_path}")
    return config_path


def copy_script_to_run_dir(run_dir):
    source_path = Path(__file__).resolve()
    destination_path = run_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    print(f"[Save] Script copy saved: {destination_path}")
    return destination_path


def validate_dataset_shapes(dataset):
    spectra = dataset["spectra"]
    assert spectra.shape[0] == len(dataset["cavity_true_um"]) == len(dataset["L_fft_um"])
    assert spectra.shape[1] == len(dataset["wavelengths_um"])
    assert dataset["film_nominal_nm"].shape[0] == spectra.shape[0]
    assert dataset["film_nominal_nm"].shape[1] == len(dataset["layer_names"])


def print_final_report(run_dir, paths, dataset):
    summary = paths["summary"]
    delta_stats = finite_stats(dataset["delta_L_nm"])
    print("")
    print(f"Dataset saved to: {run_dir}")
    print(f"NPZ path: {paths['npz_path']}")
    print(f"CSV path: {paths['csv_path']}")
    print(f"Summary path: {paths['summary_path']}")
    print(f"Total samples: {summary['num_samples_total']}")
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
        f"{delta_stats['mean']}/{delta_stats['std']}/"
        f"{delta_stats['min']}/{delta_stats['max']}"
    )


def main():
    print("=== Build NN Cavity Dataset with Lumerical StackRT ===")
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = Path(__file__).resolve().parent
    output_base_dir = script_dir
    output_base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_base_dir / f"nn_cavity_dataset_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    save_initial_config(run_dir, config)
    copy_script_to_run_dir(run_dir)

    dataset = generate_dataset(config, run_dir)
    paths = finalize_and_save(dataset, config, run_dir, timestamp)
    validate_dataset_shapes(dataset)
    print("[Validate] Dataset shape checks passed.")
    print_final_report(run_dir, paths, dataset)


if __name__ == "__main__":
    main()
