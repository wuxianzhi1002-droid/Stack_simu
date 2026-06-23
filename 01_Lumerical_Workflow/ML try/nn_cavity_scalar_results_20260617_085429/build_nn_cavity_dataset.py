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
SPLIT_NAMES = np.asarray(["train", "val", "test"], dtype=str)


@dataclass
class Config:
    # The script lives in 01_Lumerical_Workflow/ML try, so all outputs stay there.
    output_root: str = "../ML try"
    model_type: str = "PSS_TiO2"

    wavelength_start_um: float = 0.2
    wavelength_stop_um: float = 0.6
    spectral_resolution_nm: float = 0.02

    angle_deg: float = 0.0
    polarization: str = "p"

    cavity_start_um: float = 1000.0
    cavity_step_um: float = 0.002
    num_cavity_points: int = 1000

    num_nominal_models: int = 100
    num_process_per_nominal: int = 20
    film_uncertainty_nm: float = 10.0
    min_true_film_thickness_nm: float = 0.1
    random_seed: int = 20260613

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    save_checkpoint_every_process: bool = True
    save_csv_index: bool = True
    save_check_plots: bool = True
    compress_npz: bool = True

    # Raw spectra are intentionally not saved for the large run.
    save_raw_spectra: bool = False
    save_normalized_spectra: bool = False

    fft_peak_height_ratio: float = 0.2
    fft_ignore_dc_bins: int = 50
    fft_peak_distance_bins: int = 100
    zero_pad_factor: int = 8

    add_noise: bool = False
    noise_std: float = 0.0


def allowed_nominal_values_nm():
    return {
        "PSS": np.arange(5.0, 30.0 + 0.1, 5.0),
        "HSQ": np.arange(20.0, 60.0 + 0.1, 5.0),
        "SOC": np.arange(40.0, 80.0 + 0.1, 5.0),
        "TiO2": np.arange(20.0, 80.0 + 0.1, 5.0),
    }


def make_nominal_stack(name, hsq_nm, pss_nm, soc_nm, tio2_nm):
    return {
        "name": name,
        "HSQ": float(hsq_nm),
        "PSS": float(pss_nm),
        "SOC": float(soc_nm),
        "TiO2": float(tio2_nm),
    }


def build_nominal_stacks_nm(config):
    values = allowed_nominal_values_nm()
    all_combinations = [
        (hsq_nm, pss_nm, soc_nm, tio2_nm)
        for hsq_nm in values["HSQ"]
        for pss_nm in values["PSS"]
        for soc_nm in values["SOC"]
        for tio2_nm in values["TiO2"]
    ]

    if config.num_nominal_models > len(all_combinations):
        raise ValueError(
            f"num_nominal_models={config.num_nominal_models} exceeds "
            f"the available grid size {len(all_combinations)}."
        )

    selected = []
    selected_keys = set()

    # Add all 16 corners first so every requested layer range boundary is represented.
    for hsq_nm in (values["HSQ"][0], values["HSQ"][-1]):
        for pss_nm in (values["PSS"][0], values["PSS"][-1]):
            for soc_nm in (values["SOC"][0], values["SOC"][-1]):
                for tio2_nm in (values["TiO2"][0], values["TiO2"][-1]):
                    key = (hsq_nm, pss_nm, soc_nm, tio2_nm)
                    selected.append(key)
                    selected_keys.add(key)

    # Add deterministic coverage rows so intermediate 5 nm values appear before random fill.
    coverage_count = max(len(v) for v in values.values())
    for i in range(coverage_count):
        key = (
            values["HSQ"][i % len(values["HSQ"])],
            values["PSS"][(2 * i) % len(values["PSS"])],
            values["SOC"][(3 * i) % len(values["SOC"])],
            values["TiO2"][i % len(values["TiO2"])],
        )
        if key not in selected_keys:
            selected.append(key)
            selected_keys.add(key)

    remaining = [combo for combo in all_combinations if combo not in selected_keys]
    rng = np.random.default_rng(config.random_seed)
    rng.shuffle(remaining)
    selected.extend(remaining[: config.num_nominal_models - len(selected)])
    selected = selected[: config.num_nominal_models]

    return [
        make_nominal_stack(
            name=f"model_{idx:03d}_hsq{int(hsq_nm)}_pss{int(pss_nm)}_soc{int(soc_nm)}_tio2{int(tio2_nm)}",
            hsq_nm=hsq_nm,
            pss_nm=pss_nm,
            soc_nm=soc_nm,
            tio2_nm=tio2_nm,
        )
        for idx, (hsq_nm, pss_nm, soc_nm, tio2_nm) in enumerate(selected)
    ]


def make_cavity_axis_um(config):
    return np.round(
        config.cavity_start_um + config.cavity_step_um * np.arange(config.num_cavity_points, dtype=float),
        12,
    )


def make_wavelength_axis_um(config):
    span_nm = (config.wavelength_stop_um - config.wavelength_start_um) * 1000.0
    num_wavelengths = int(round(span_nm / config.spectral_resolution_nm)) + 1
    return np.linspace(config.wavelength_start_um, config.wavelength_stop_um, num_wavelengths)


def config_to_dict(config, nominal_stacks_nm=None):
    config_dict = asdict(config)
    config_dict["film_layer_names"] = FILM_LAYER_NAMES
    config_dict["allowed_nominal_values_nm"] = {
        key: value.tolist() for key, value in allowed_nominal_values_nm().items()
    }
    config_dict["nominal_stacks_nm"] = nominal_stacks_nm if nominal_stacks_nm is not None else []
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
    def replace_nan(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: replace_nan(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_nan(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as file:
        json.dump(replace_nan(payload), file, ensure_ascii=False, indent=2, default=json_default)


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
        self.wavelengths_um = make_wavelength_axis_um(config)
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
    return np.asarray([float(nominal_stack_nm[name]) for name in FILM_LAYER_NAMES], dtype=np.float32)


def sample_film_delta_nm(film_nominal_nm, config, rng):
    film_nominal_nm = np.asarray(film_nominal_nm, dtype=float)
    lower = np.maximum(
        -float(config.film_uncertainty_nm),
        float(config.min_true_film_thickness_nm) - film_nominal_nm,
    )
    upper = np.full_like(film_nominal_nm, float(config.film_uncertainty_nm), dtype=float)
    return rng.uniform(lower, upper).astype(np.float32)


def fft_config_dict(config):
    return {
        "FFT_PEAK_HEIGHT_RATIO": config.fft_peak_height_ratio,
        "FFT_IGNORE_DC_BINS": config.fft_ignore_dc_bins,
        "FFT_PEAK_DISTANCE_BINS": config.fft_peak_distance_bins,
        "ZERO_PAD_FACTOR": config.zero_pad_factor,
    }


def solve_single_fft(wavelengths_um, intensities, config_dict):
    """Copied from solve_npz_fft.py FFTSolver.solve, adapted for scalar-only output."""
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

    if peaks.size == 0:
        return np.nan, np.nan, 0

    peak_heights = fft_data[peaks]
    dominant_idx = int(np.argmax(peak_heights))
    return float(distance_axis_um[peaks[dominant_idx]]), float(peak_heights[dominant_idx]), int(peaks.size)


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
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    if num_processes == 1:
        return np.sort(shuffled), np.array([], dtype=int), np.array([], dtype=int)
    if num_processes == 2:
        return np.sort(shuffled[:1]), np.array([], dtype=int), np.sort(shuffled[1:])

    num_train = int(np.floor(num_processes * train_ratio))
    num_val = int(np.floor(num_processes * val_ratio))
    num_train = min(max(1, num_train), num_processes - 2)
    num_val = min(max(1, num_val), num_processes - num_train - 1)

    return (
        np.sort(shuffled[:num_train]),
        np.sort(shuffled[num_train : num_train + num_val]),
        np.sort(shuffled[num_train + num_val :]),
    )


def process_split_lookup(process_ids, train_process_ids, val_process_ids, test_process_ids):
    lookup = {}
    for process_id in np.asarray(train_process_ids, dtype=int):
        lookup[int(process_id)] = 0
    for process_id in np.asarray(val_process_ids, dtype=int):
        lookup[int(process_id)] = 1
    for process_id in np.asarray(test_process_ids, dtype=int):
        lookup[int(process_id)] = 2
    missing = sorted(set(np.asarray(process_ids, dtype=int).tolist()) - set(lookup))
    if missing:
        raise ValueError(f"Some process_ids are missing from split lookup: {missing[:10]}")
    return lookup


def allocate_result_arrays(total_samples_planned):
    return {
        "sample_id": np.empty(total_samples_planned, dtype=np.int64),
        "process_id": np.empty(total_samples_planned, dtype=np.int32),
        "nominal_stack_id": np.empty(total_samples_planned, dtype=np.int16),
        "split_id": np.empty(total_samples_planned, dtype=np.int8),
        "cavity_true_um": np.empty(total_samples_planned, dtype=np.float64),
        "L_fft_um": np.full(total_samples_planned, np.nan, dtype=np.float64),
        "delta_L_um": np.full(total_samples_planned, np.nan, dtype=np.float64),
        "delta_L_nm": np.full(total_samples_planned, np.nan, dtype=np.float64),
        "H_peak": np.full(total_samples_planned, np.nan, dtype=np.float32),
        "peak_count": np.zeros(total_samples_planned, dtype=np.int16),
        "film_nominal_nm": np.empty((total_samples_planned, len(FILM_LAYER_NAMES)), dtype=np.float32),
        "film_delta_nm": np.empty((total_samples_planned, len(FILM_LAYER_NAMES)), dtype=np.float32),
        "film_true_nm": np.empty((total_samples_planned, len(FILM_LAYER_NAMES)), dtype=np.float32),
        "valid_mask": np.zeros(total_samples_planned, dtype=bool),
    }


def trim_result_arrays(arrays, count):
    return {
        key: value[:count].copy() if isinstance(value, np.ndarray) and value.shape[:1] == (len(value),) else value
        for key, value in arrays.items()
    }


def checkpoint_process(run_dir, process_id, process_slice, dataset):
    checkpoint_path = run_dir / f"checkpoint_process_{process_id:04d}.npz"
    np.savez_compressed(
        checkpoint_path,
        sample_id=dataset["sample_id"][process_slice],
        process_id=dataset["process_id"][process_slice],
        nominal_stack_id=dataset["nominal_stack_id"][process_slice],
        split_id=dataset["split_id"][process_slice],
        split_names=SPLIT_NAMES,
        cavity_true_um=dataset["cavity_true_um"][process_slice],
        L_fft_um=dataset["L_fft_um"][process_slice],
        delta_L_um=dataset["delta_L_um"][process_slice],
        delta_L_nm=dataset["delta_L_nm"][process_slice],
        H_peak=dataset["H_peak"][process_slice],
        peak_count=dataset["peak_count"][process_slice],
        film_nominal_nm=dataset["film_nominal_nm"][process_slice],
        film_delta_nm=dataset["film_delta_nm"][process_slice],
        film_true_nm=dataset["film_true_nm"][process_slice],
        valid_mask=dataset["valid_mask"][process_slice],
        layer_names=np.asarray(FILM_LAYER_NAMES, dtype=str),
        spectra_saved=np.array(False, dtype=bool),
    )
    print(f"[Checkpoint] Saved scalar-only checkpoint: {checkpoint_path}")


def generate_scalar_results(config, run_dir, simulator_cls=StackRTSimulator):
    nominal_stacks_nm = build_nominal_stacks_nm(config)
    cavity_axis_um = make_cavity_axis_um(config)
    total_processes = len(nominal_stacks_nm) * config.num_process_per_nominal
    total_samples_planned = total_processes * len(cavity_axis_um)
    process_ids_all = np.arange(total_processes, dtype=int)
    train_process_ids, val_process_ids, test_process_ids = split_by_process(
        process_ids_all,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed,
    )
    split_lookup = process_split_lookup(
        process_ids_all,
        train_process_ids,
        val_process_ids,
        test_process_ids,
    )

    print("[Config] Scalar-only output mode: raw spectra and spectra_norm will not be saved.")
    print(f"[Config] Nominal models: {len(nominal_stacks_nm)}")
    print(f"[Config] Processes: {total_processes}")
    print(f"[Config] Cavity points/process: {len(cavity_axis_um)}")
    print(f"[Config] Planned samples: {total_samples_planned}")
    print(
        f"[Config] Cavity range: {cavity_axis_um[0]:.6f} um to "
        f"{cavity_axis_um[-1]:.6f} um, step {config.cavity_step_um:.6f} um"
    )

    simulator = simulator_cls(config)
    rng = np.random.default_rng(config.random_seed)
    arrays = allocate_result_arrays(total_samples_planned)
    process_nominal_stack_id = np.empty(total_processes, dtype=np.int16)
    process_film_delta_nm = np.empty((total_processes, len(FILM_LAYER_NAMES)), dtype=np.float32)
    process_film_true_nm = np.empty((total_processes, len(FILM_LAYER_NAMES)), dtype=np.float32)
    failed_cases = []
    failed_fft_cases = []
    fft_config = fft_config_dict(config)

    write_idx = 0
    process_id = 0
    start_time = time.time()

    simulator.open()
    try:
        for nominal_stack_id, nominal_stack_nm in enumerate(nominal_stacks_nm):
            nominal_name = nominal_stack_nm["name"]
            film_nominal_nm = nominal_film_array_nm(nominal_stack_nm)
            print(f"[Dataset] Nominal {nominal_stack_id + 1}/{len(nominal_stacks_nm)}: {nominal_name}")

            for process_idx in range(config.num_process_per_nominal):
                if process_idx == 0:
                    film_delta_nm = np.zeros(len(FILM_LAYER_NAMES), dtype=np.float32)
                else:
                    film_delta_nm = sample_film_delta_nm(film_nominal_nm, config, rng)
                film_true_nm = film_nominal_nm + film_delta_nm
                split_id = split_lookup[process_id]
                process_start_idx = write_idx
                process_nominal_stack_id[process_id] = nominal_stack_id
                process_film_delta_nm[process_id, :] = film_delta_nm
                process_film_true_nm[process_id, :] = film_true_nm

                print(
                    "[Dataset] Process "
                    f"{process_id + 1}/{total_processes} "
                    f"(process_id={process_id}, split={SPLIT_NAMES[split_id]}, "
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
                            spectrum = spectrum + rng.normal(0.0, config.noise_std, size=spectrum.shape)

                        L_fft_um, H_peak, peak_count = solve_single_fft(
                            simulator.wavelengths_um,
                            spectrum,
                            fft_config,
                        )

                        arrays["sample_id"][write_idx] = write_idx
                        arrays["process_id"][write_idx] = process_id
                        arrays["nominal_stack_id"][write_idx] = nominal_stack_id
                        arrays["split_id"][write_idx] = split_id
                        arrays["cavity_true_um"][write_idx] = cavity_um
                        arrays["L_fft_um"][write_idx] = L_fft_um
                        arrays["delta_L_um"][write_idx] = cavity_um - L_fft_um
                        arrays["delta_L_nm"][write_idx] = (cavity_um - L_fft_um) * 1000.0
                        arrays["H_peak"][write_idx] = H_peak
                        arrays["peak_count"][write_idx] = peak_count
                        arrays["film_nominal_nm"][write_idx, :] = film_nominal_nm
                        arrays["film_delta_nm"][write_idx, :] = film_delta_nm
                        arrays["film_true_nm"][write_idx, :] = film_true_nm
                        arrays["valid_mask"][write_idx] = np.isfinite(L_fft_um) and np.isfinite(H_peak)

                        if not arrays["valid_mask"][write_idx]:
                            failed_fft_cases.append(
                                {
                                    "sample_id": int(write_idx),
                                    "process_id": int(process_id),
                                    "nominal_stack_id": int(nominal_stack_id),
                                    "nominal_stack_name": nominal_name,
                                    "cavity_idx": int(cavity_idx),
                                    "cavity_true_um": float(cavity_um),
                                    "reason": "no FFT peak detected",
                                }
                            )

                        write_idx += 1
                    except Exception as exc:
                        failed_cases.append(
                            {
                                "attempt_sample_id": int(process_id * len(cavity_axis_um) + cavity_idx),
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
                        )
                        print(
                            "[Warning] Simulation/FFT sample failed and was skipped: "
                            f"process_id={process_id}, cavity_idx={cavity_idx}, error={exc}"
                        )

                if config.save_checkpoint_every_process and write_idx > process_start_idx:
                    checkpoint_process(run_dir, process_id, slice(process_start_idx, write_idx), arrays)

                elapsed = time.time() - start_time
                valid_count = int(np.count_nonzero(arrays["valid_mask"][:write_idx]))
                print(
                    f"[Dataset] Finished process_id={process_id}; "
                    f"saved_samples={write_idx}, valid={valid_count}, "
                    f"failed_sim={len(failed_cases)}, failed_fft={len(failed_fft_cases)}, "
                    f"elapsed={elapsed:.1f}s"
                )
                process_id += 1
    finally:
        simulator.close()

    dataset = trim_result_arrays(arrays, write_idx)
    dataset.update(
        {
            "wavelengths_um": simulator.wavelengths_um,
            "cavity_axis_um": cavity_axis_um,
            "layer_names": np.asarray(FILM_LAYER_NAMES, dtype=str),
            "split_names": SPLIT_NAMES,
            "train_process_ids": train_process_ids,
            "val_process_ids": val_process_ids,
            "test_process_ids": test_process_ids,
            "nominal_stack_name_by_id": np.asarray([stack["name"] for stack in nominal_stacks_nm], dtype=str),
            "nominal_stack_values_nm": np.asarray(
                [[stack[name] for name in FILM_LAYER_NAMES] for stack in nominal_stacks_nm],
                dtype=np.float32,
            ),
            "process_nominal_stack_id": process_nominal_stack_id,
            "process_film_delta_nm": process_film_delta_nm,
            "process_film_true_nm": process_film_true_nm,
            "failed_cases": failed_cases,
            "failed_fft_cases": failed_fft_cases,
            "num_samples_planned": np.array(total_samples_planned, dtype=np.int64),
            "num_processes_planned": np.array(total_processes, dtype=np.int32),
            "spectra_saved": np.array(False, dtype=bool),
            "spectra_norm_saved": np.array(False, dtype=bool),
            "nominal_stacks_nm": nominal_stacks_nm,
        }
    )
    return dataset


def finite_stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def save_dataset_npz(run_dir, timestamp, dataset, config):
    npz_path = run_dir / f"nn_cavity_scalar_results_{timestamp}.npz"
    config_json = json.dumps(
        config_to_dict(config, dataset["nominal_stacks_nm"]),
        ensure_ascii=False,
        default=json_default,
    )
    save_kwargs = {
        "wavelengths_um": dataset["wavelengths_um"],
        "sample_id": dataset["sample_id"],
        "process_id": dataset["process_id"],
        "nominal_stack_id": dataset["nominal_stack_id"],
        "nominal_stack_name_by_id": dataset["nominal_stack_name_by_id"],
        "split_id": dataset["split_id"],
        "split_names": dataset["split_names"],
        "cavity_true_um": dataset["cavity_true_um"],
        "L_true_um": dataset["cavity_true_um"],
        "L_fft_um": dataset["L_fft_um"],
        "delta_L_um": dataset["delta_L_um"],
        "delta_L_nm": dataset["delta_L_nm"],
        "H_peak": dataset["H_peak"],
        "peak_count": dataset["peak_count"],
        "film_nominal_nm": dataset["film_nominal_nm"],
        "film_delta_nm": dataset["film_delta_nm"],
        "film_true_nm": dataset["film_true_nm"],
        "layer_names": dataset["layer_names"],
        "cavity_axis_um": dataset["cavity_axis_um"],
        "train_process_ids": dataset["train_process_ids"],
        "val_process_ids": dataset["val_process_ids"],
        "test_process_ids": dataset["test_process_ids"],
        "valid_mask": dataset["valid_mask"],
        "nominal_stack_values_nm": dataset["nominal_stack_values_nm"],
        "process_nominal_stack_id": dataset["process_nominal_stack_id"],
        "process_film_delta_nm": dataset["process_film_delta_nm"],
        "process_film_true_nm": dataset["process_film_true_nm"],
        "num_samples_planned": dataset["num_samples_planned"],
        "num_processes_planned": dataset["num_processes_planned"],
        "spectra_saved": dataset["spectra_saved"],
        "spectra_norm_saved": dataset["spectra_norm_saved"],
        "config_json": np.array(config_json),
        "timestamp": np.array(timestamp),
    }
    if config.compress_npz:
        np.savez_compressed(npz_path, **save_kwargs)
    else:
        np.savez(npz_path, **save_kwargs)
    print(f"[Save] Scalar-only NPZ saved: {npz_path}")
    return npz_path


def save_csv_index(run_dir, timestamp, dataset):
    csv_path = run_dir / f"nn_cavity_scalar_results_index_{timestamp}.csv"
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
        for i in range(dataset["sample_id"].size):
            nominal_stack_id = int(dataset["nominal_stack_id"][i])
            split_id = int(dataset["split_id"][i])
            row = {
                "sample_id": int(dataset["sample_id"][i]),
                "process_id": int(dataset["process_id"][i]),
                "nominal_stack_id": nominal_stack_id,
                "nominal_stack_name": str(dataset["nominal_stack_name_by_id"][nominal_stack_id]),
                "split_label": str(dataset["split_names"][split_id]),
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
    print(f"[Save] Scalar-only CSV saved: {csv_path}")
    return csv_path


def save_check_plots(run_dir, dataset):
    print("[Plot] Saving scalar check plots...")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for split_id, split_name in enumerate(dataset["split_names"]):
        mask = dataset["split_id"] == split_id
        if np.any(mask):
            ax.scatter(
                dataset["cavity_true_um"][mask],
                dataset["L_fft_um"][mask],
                s=4,
                alpha=0.25,
                label=str(split_name),
            )
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
    fig.savefig(run_dir / "01_fft_vs_true_cavity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    finite_delta = dataset["delta_L_nm"][np.isfinite(dataset["delta_L_nm"])]
    if finite_delta.size > 0:
        ax.hist(finite_delta, bins=80, color="#4c78a8", alpha=0.85)
    ax.set_title("delta_L_nm Histogram")
    ax.set_xlabel("delta_L_nm")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "02_delta_L_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    finite_h = dataset["H_peak"][np.isfinite(dataset["H_peak"])]
    if finite_h.size > 0:
        ax.hist(finite_h, bins=80, color="#59a14f", alpha=0.85)
    ax.set_title("H_peak Histogram")
    ax.set_xlabel("H_peak")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "03_h_peak_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    splits = [
        ("train", dataset["train_process_ids"], 0, "#1f77b4"),
        ("val", dataset["val_process_ids"], 1, "#ff7f0e"),
        ("test", dataset["test_process_ids"], 2, "#2ca02c"),
    ]
    for split_name, ids, y_value, color in splits:
        if len(ids) > 0:
            ax.scatter(ids, np.full(len(ids), y_value), s=12, label=split_name, color=color)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["train", "val", "test"])
    ax.set_title("Process Split")
    ax.set_xlabel("process_id")
    ax.set_ylabel("split")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.savefig(run_dir / "04_process_split.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.reshape(-1)
    for i, layer_name in enumerate(dataset["layer_names"]):
        axes[i].hist(dataset["process_film_delta_nm"][:, i], bins=60, color="#b279a2", alpha=0.85)
        axes[i].set_title(f"{layer_name} process delta")
        axes[i].set_xlabel("film_delta_nm")
        axes[i].set_ylabel("Process count")
        axes[i].grid(True, alpha=0.3)
    fig.savefig(run_dir / "05_film_delta_distribution.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.reshape(-1)
    for i, layer_name in enumerate(dataset["layer_names"]):
        axes[i].hist(dataset["nominal_stack_values_nm"][:, i], bins=20, color="#f28e2b", alpha=0.85)
        axes[i].set_title(f"{layer_name} nominal coverage")
        axes[i].set_xlabel("nominal thickness (nm)")
        axes[i].set_ylabel("Model count")
        axes[i].grid(True, alpha=0.3)
    fig.savefig(run_dir / "06_nominal_model_coverage.png", dpi=220)
    plt.close(fig)


def build_summary(dataset, config):
    valid_mask = dataset["valid_mask"]
    delta_stats = finite_stats(dataset["delta_L_nm"])
    cavity_axis_um = dataset["cavity_axis_um"]
    return {
        "output_mode": "scalar_only_no_raw_spectra",
        "spectra_saved": bool(dataset["spectra_saved"]),
        "spectra_norm_saved": bool(dataset["spectra_norm_saved"]),
        "num_samples_planned": int(dataset["num_samples_planned"]),
        "num_samples_total": int(dataset["sample_id"].size),
        "num_samples_valid": int(np.count_nonzero(valid_mask)),
        "num_failed_simulation": int(len(dataset["failed_cases"])),
        "num_failed_fft": int(len(dataset["failed_fft_cases"])),
        "num_nominal_stacks": int(dataset["nominal_stack_values_nm"].shape[0]),
        "num_processes": int(dataset["num_processes_planned"]),
        "num_train_processes": int(len(dataset["train_process_ids"])),
        "num_val_processes": int(len(dataset["val_process_ids"])),
        "num_test_processes": int(len(dataset["test_process_ids"])),
        "wavelength_start_um": float(config.wavelength_start_um),
        "wavelength_stop_um": float(config.wavelength_stop_um),
        "num_wavelengths": int(dataset["wavelengths_um"].size),
        "cavity_start_um": float(cavity_axis_um[0]),
        "cavity_stop_um": float(cavity_axis_um[-1]),
        "cavity_step_um": float(config.cavity_step_um),
        "num_cavity_points": int(cavity_axis_um.size),
        "film_layer_names": FILM_LAYER_NAMES,
        "film_uncertainty_nm": float(config.film_uncertainty_nm),
        "nominal_ranges_nm": {
            "PSS": [5.0, 30.0],
            "HSQ": [20.0, 60.0],
            "SOC": [40.0, 80.0],
            "TiO2": [20.0, 80.0],
        },
        "nominal_step_nm": 5.0,
        "L_fft_nan_count": int(np.count_nonzero(~np.isfinite(dataset["L_fft_um"]))),
        "delta_L_nm_mean": delta_stats["mean"],
        "delta_L_nm_std": delta_stats["std"],
        "delta_L_nm_min": delta_stats["min"],
        "delta_L_nm_max": delta_stats["max"],
    }


def save_all_outputs(run_dir, timestamp, dataset, config):
    npz_path = save_dataset_npz(run_dir, timestamp, dataset, config)
    csv_path = None
    if config.save_csv_index:
        csv_path = save_csv_index(run_dir, timestamp, dataset)

    failed_cases_path = run_dir / f"failed_cases_{timestamp}.json"
    failed_fft_cases_path = run_dir / f"failed_fft_cases_{timestamp}.json"
    dump_json(failed_cases_path, dataset["failed_cases"])
    dump_json(failed_fft_cases_path, dataset["failed_fft_cases"])
    print(f"[Save] Failed simulation JSON saved: {failed_cases_path}")
    print(f"[Save] Failed FFT JSON saved: {failed_fft_cases_path}")

    if config.save_check_plots:
        save_check_plots(run_dir, dataset)

    valid_summary = {
        "output_mode": "scalar_only_no_raw_spectra",
        "num_samples_planned": int(dataset["num_samples_planned"]),
        "num_samples_total": int(dataset["sample_id"].size),
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


def save_initial_config(run_dir, config, nominal_stacks_nm):
    config_path = run_dir / "00_config.json"
    dump_json(config_path, config_to_dict(config, nominal_stacks_nm))
    print(f"[Save] Config saved: {config_path}")
    return config_path


def copy_script_to_run_dir(run_dir):
    source_path = Path(__file__).resolve()
    destination_path = run_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    print(f"[Save] Script copy saved: {destination_path}")
    return destination_path


def validate_dataset_shapes(dataset):
    n_samples = dataset["sample_id"].size
    assert n_samples == len(dataset["cavity_true_um"]) == len(dataset["L_fft_um"])
    assert dataset["film_nominal_nm"].shape == (n_samples, len(dataset["layer_names"]))
    assert dataset["film_delta_nm"].shape == (n_samples, len(dataset["layer_names"]))
    assert dataset["film_true_nm"].shape == (n_samples, len(dataset["layer_names"]))
    assert "spectra" not in dataset
    assert "spectra_norm" not in dataset


def print_final_report(run_dir, paths, dataset):
    summary = paths["summary"]
    delta_stats = finite_stats(dataset["delta_L_nm"])
    print("")
    print(f"Dataset saved to: {run_dir}")
    print(f"NPZ path: {paths['npz_path']}")
    print(f"CSV path: {paths['csv_path']}")
    print(f"Summary path: {paths['summary_path']}")
    print(f"Output mode: {summary['output_mode']}")
    print(f"Raw spectra saved: {summary['spectra_saved']}")
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
        f"{delta_stats['mean']}/{delta_stats['std']}/"
        f"{delta_stats['min']}/{delta_stats['max']}"
    )


def main():
    print("=== Build Scalar-Only NN Cavity Dataset with Lumerical StackRT ===")
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = Path(__file__).resolve().parent
    output_base_dir = script_dir
    output_base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_base_dir / f"nn_cavity_scalar_results_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    nominal_stacks_nm = build_nominal_stacks_nm(config)
    save_initial_config(run_dir, config, nominal_stacks_nm)
    copy_script_to_run_dir(run_dir)

    dataset = generate_scalar_results(config, run_dir)
    paths = save_all_outputs(run_dir, timestamp, dataset, config)
    validate_dataset_shapes(dataset)
    print("[Validate] Scalar-only dataset shape checks passed.")
    print_final_report(run_dir, paths, dataset)


if __name__ == "__main__":
    main()
