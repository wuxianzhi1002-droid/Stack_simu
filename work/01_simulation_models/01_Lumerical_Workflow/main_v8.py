from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


LUMERICAL_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
if LUMERICAL_PATH.exists():
    if str(LUMERICAL_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_PATH))
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    lumapi = None


GENERATOR_VERSION = "main_v8"
C0_M_S = 299_792_458.0
INCIDENT_MEDIUM_N = 5.8284
ANGLE_MAX_DEG = 0.1
WAVELENGTH_ERROR_HARD_MAX_NM = 0.2
REALIZATION_MIN_FRACTION = 0.8
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
PERTURBED_MATERIALS = ("HSQ", "PSS", "SOC", "TiO2")
NOISE_FACTORS = (
    "angle",
    "laser_wavelength",
    "axis_offset",
    "axis_scale",
    "thermal_drift",
    "absolute_accuracy",
    "material",
    "detector",
    "combined",
)

CONFIG = {
    "MODEL_TYPE": "PSS_TiO2",
    "WAVELENGTH_START_UM": 0.450,
    "WAVELENGTH_STOP_UM": 0.580,
    "SPECTRAL_SAMPLING_NM": 0.02,
    "REFLECTOR_ANGLE_NOMINAL_DEG": 0.0,
    "ILS_ENABLED": False,
    "TIME_SERIES_ENABLED": False,
    "FRAMES_PER_REALIZATION": 1,
    "LASER_RESIDUAL_CORRELATION_LENGTH_NM": 8.0,
    "ABSOLUTE_ACCURACY_CORRELATION_LENGTH_NM": 30.0,
    "LAYERS": [
        ("RefReflector", 0.0),
        ("Air", 1000.0),
        ("HSQ", 0.030),
        ("PSS", 0.010),
        ("SOC", 0.040),
        ("TiO2", 0.040),
        ("Cu", 0.0),
    ],
}

# V6 material/angle/detector levels are retained. Wavelength error is split
# into orthogonal physical components. The high-level component maxima sum to
# 0.1989 nm over this wavelength range, below the 0.2 nm hard limit.
NOISE_LEVELS = {
    "low": {
        "n_real_sigma_rel": 5.0e-4,
        "k_sigma_rel": 1.0e-2,
        "angle_max_deg": 0.01,
        "laser_residual_max_nm": 0.005,
        "axis_offset_max_nm": 0.0002,
        "axis_scale_max_ppm": 10.0,
        "thermal_drift_max_nm": 0.001,
        "absolute_accuracy_max_nm": 0.020,
        "frame_gain_sigma_rel": 2.0e-4,
        "reflectance_offset_sigma_abs": 2.0e-4,
    },
    "medium": {
        "n_real_sigma_rel": 2.0e-3,
        "k_sigma_rel": 5.0e-2,
        "angle_max_deg": 0.05,
        "laser_residual_max_nm": 0.020,
        "axis_offset_max_nm": 0.001,
        "axis_scale_max_ppm": 30.0,
        "thermal_drift_max_nm": 0.005,
        "absolute_accuracy_max_nm": 0.080,
        "frame_gain_sigma_rel": 1.0e-3,
        "reflectance_offset_sigma_abs": 1.0e-3,
    },
    "high": {
        "n_real_sigma_rel": 5.0e-3,
        "k_sigma_rel": 1.0e-1,
        "angle_max_deg": 0.10,
        "laser_residual_max_nm": 0.050,
        "axis_offset_max_nm": 0.005,
        "axis_scale_max_ppm": 60.0,
        "thermal_drift_max_nm": 0.020,
        "absolute_accuracy_max_nm": 0.120,
        "frame_gain_sigma_rel": 5.0e-3,
        "reflectance_offset_sigma_abs": 5.0e-3,
    },
}


def zero_profile() -> dict[str, float]:
    return {name: 0.0 for name in next(iter(NOISE_LEVELS.values()))}


def active_profile(level: str, factor: str) -> dict[str, float]:
    if level == "clean":
        return zero_profile()
    source = NOISE_LEVELS[level]
    if factor == "combined":
        return dict(source)
    keys = {
        "angle": ("angle_max_deg",),
        "laser_wavelength": ("laser_residual_max_nm",),
        "axis_offset": ("axis_offset_max_nm",),
        "axis_scale": ("axis_scale_max_ppm",),
        "thermal_drift": ("thermal_drift_max_nm",),
        "absolute_accuracy": ("absolute_accuracy_max_nm",),
        "material": ("n_real_sigma_rel", "k_sigma_rel"),
        "detector": ("frame_gain_sigma_rel", "reflectance_offset_sigma_abs"),
    }[factor]
    profile = zero_profile()
    for key in keys:
        profile[key] = float(source[key])
    return profile


def all_case_names() -> list[str]:
    return ["clean"] + [f"{factor}_{level}" for factor in NOISE_FACTORS for level in NOISE_LEVELS]


def parse_case(case_name: str) -> tuple[str, str]:
    if case_name == "clean":
        return "clean", "clean"
    factor, level = case_name.rsplit("_", 1)
    if factor not in NOISE_FACTORS or level not in NOISE_LEVELS:
        raise ValueError(f"Unknown case: {case_name}")
    return factor, level


def select_cases(value: str) -> list[str]:
    if value == "all":
        return all_case_names()
    if value in NOISE_FACTORS:
        return [f"{value}_{level}" for level in NOISE_LEVELS]
    cases = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(cases) - set(all_case_names()))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    return cases


def signed_level_value(maximum: float, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    magnitude = float(maximum) * rng.uniform(REALIZATION_MIN_FRACTION, 1.0)
    return magnitude if rng.integers(0, 2) else -magnitude


def positive_level_value(maximum: float, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    return float(maximum) * rng.uniform(REALIZATION_MIN_FRACTION, 1.0)


def remove_constant_and_linear_terms(values: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    x = (wavelengths_nm - np.mean(wavelengths_nm)) / max(float(np.ptp(wavelengths_nm)), 1.0)
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - design @ coefficients


def smooth_curve_with_peak(
    wavelengths_nm: np.ndarray,
    target_peak_nm: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_peak_nm <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)
    spacing_nm = float(np.median(np.diff(wavelengths_nm)))
    sigma_pixels = float(correlation_length_nm) / spacing_nm
    for _ in range(50):
        raw = gaussian_filter1d(rng.normal(size=len(wavelengths_nm)), sigma=sigma_pixels, mode="reflect")
        curve = remove_constant_and_linear_terms(raw, wavelengths_nm)
        peak = float(np.max(np.abs(curve)))
        if peak > 1.0e-12:
            return curve * (float(target_peak_nm) / peak)
    raise RuntimeError("Could not generate a nonzero smooth wavelength curve.")


def material_n(
    name: str,
    wavelengths_um: np.ndarray,
    n_real_rel_delta: dict[str, float] | None = None,
    k_rel_delta: dict[str, float] | None = None,
) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=float)
    if name == "RefReflector":
        values = np.full_like(w, INCIDENT_MEDIUM_N, dtype=np.complex128)
    elif name == "Air":
        values = np.ones_like(w, dtype=np.complex128)
    elif name == "HSQ":
        values = np.full_like(w, 1.41, dtype=np.complex128)
    elif name == "PSS":
        values = np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    elif name == "SOC":
        values = (1.55 + 0.005 / (w**2)).astype(np.complex128)
    elif name == "TiO2":
        values = (2.4 + 0.02 / (w**2)).astype(np.complex128)
    elif name == "Cu":
        values = np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown material: {name}")
    if name in PERTURBED_MATERIALS:
        dn = 0.0 if n_real_rel_delta is None else float(n_real_rel_delta.get(name, 0.0))
        dk = 0.0 if k_rel_delta is None else float(k_rel_delta.get(name, 0.0))
        values = values.real * (1.0 + dn) + 1j * values.imag * (1.0 + dk)
    return values


def propagation_cosines(n_matrix: np.ndarray, reflector_angle_deg: float) -> np.ndarray:
    tangential_index = n_matrix[0] * np.sin(np.deg2rad(float(reflector_angle_deg)))
    cos_values = np.sqrt(1.0 - (tangential_index[None, :] / n_matrix) ** 2)
    cos_values[np.real(cos_values) < 0.0] *= -1.0
    return cos_values


def tmm_reflectance(
    wavelengths_um: np.ndarray,
    thicknesses_um: dict[str, float],
    reflector_angle_deg: float,
    n_real_rel_delta: dict[str, float],
    k_rel_delta: dict[str, float],
) -> np.ndarray:
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    n_matrix = np.vstack([
        material_n(name, wavelengths_um, n_real_rel_delta, k_rel_delta) for name in LAYER_NAMES
    ])
    cos_values = propagation_cosines(n_matrix, reflector_angle_deg)
    q_values = n_matrix / cos_values
    k0 = 2.0 * np.pi / (wavelengths_um * 1.0e-6)
    m11 = np.ones(len(wavelengths_um), dtype=complex)
    m12 = np.zeros(len(wavelengths_um), dtype=complex)
    m21 = np.zeros(len(wavelengths_um), dtype=complex)
    m22 = np.ones(len(wavelengths_um), dtype=complex)
    for layer_index, name in enumerate(LAYER_NAMES[1:-1], start=1):
        thickness_m = float(thicknesses_um[name]) * 1.0e-6
        if thickness_m <= 0.0:
            continue
        delta = k0 * n_matrix[layer_index] * cos_values[layer_index] * thickness_m
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        q_layer = q_values[layer_index]
        a11 = c_delta
        a12 = -1j * s_delta / q_layer
        a21 = -1j * q_layer * s_delta
        a22 = c_delta
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21,
            m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21,
            m21 * a12 + m22 * a22,
        )
    q0 = q_values[0]
    qs = q_values[-1]
    numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
    denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
    return np.abs(numerator / denominator) ** 2



def realize_noise(
    case_name: str,
    wavelengths_nm: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    factor, level = parse_case(case_name)
    profile = active_profile(level, factor)
    n_delta = {
        name: signed_level_value(profile["n_real_sigma_rel"], rng)
        for name in PERTURBED_MATERIALS
    }
    k_delta = {
        name: signed_level_value(profile["k_sigma_rel"], rng)
        for name in PERTURBED_MATERIALS
    }
    angle_deg = positive_level_value(profile["angle_max_deg"], rng)
    if angle_deg > ANGLE_MAX_DEG + 1.0e-12:
        raise ValueError("Realized reflector angle exceeds 0.1 deg.")

    laser_peak = positive_level_value(profile["laser_residual_max_nm"], rng)
    absolute_peak = positive_level_value(profile["absolute_accuracy_max_nm"], rng)
    laser_curve = smooth_curve_with_peak(
        wavelengths_nm,
        laser_peak,
        CONFIG["LASER_RESIDUAL_CORRELATION_LENGTH_NM"],
        rng,
    )
    absolute_curve = smooth_curve_with_peak(
        wavelengths_nm,
        absolute_peak,
        CONFIG["ABSOLUTE_ACCURACY_CORRELATION_LENGTH_NM"],
        rng,
    )
    axis_offset_nm = signed_level_value(profile["axis_offset_max_nm"], rng)
    thermal_drift_nm = signed_level_value(profile["thermal_drift_max_nm"], rng)
    axis_scale_ppm = signed_level_value(profile["axis_scale_max_ppm"], rng)
    center_nm = 0.5 * (wavelengths_nm[0] + wavelengths_nm[-1])
    scale_curve = axis_scale_ppm * 1.0e-6 * (wavelengths_nm - center_nm)
    offset_curve = np.full_like(wavelengths_nm, axis_offset_nm)
    thermal_curve = np.full_like(wavelengths_nm, thermal_drift_nm)
    total_curve = laser_curve + offset_curve + scale_curve + thermal_curve + absolute_curve
    total_max = float(np.max(np.abs(total_curve)))  # 在这里已经构造了整体的波长偏移噪声，但是转移到模型中还是用的波长轴偏移吗？
    if total_max > WAVELENGTH_ERROR_HARD_MAX_NM + 1.0e-12:
        raise ValueError(
            f"Combined wavelength error {total_max:.9g} nm exceeds the 0.2 nm hard limit."
        )
    physical_nm = wavelengths_nm + total_curve
    if np.any(np.diff(physical_nm) <= 0.0):
        raise ValueError("Realized physical wavelength mapping is not monotonic.")

    frame_gain_error_rel = signed_level_value(profile["frame_gain_sigma_rel"], rng)
    reflectance_offset_abs = signed_level_value(profile["reflectance_offset_sigma_abs"], rng)
    metadata = {
        "case": case_name,
        "factor": factor,
        "level": level,
        "profile": profile,
        "realization_fraction_range": [REALIZATION_MIN_FRACTION, 1.0],
        "material_n_real_rel_delta": n_delta,
        "material_k_rel_delta": k_delta,
        "reflector_angle_deg": angle_deg,
        "laser_residual_peak_nm": float(np.max(np.abs(laser_curve))),
        "axis_offset_nm": axis_offset_nm,
        "axis_scale_ppm": axis_scale_ppm,
        "axis_scale_edge_max_nm": float(np.max(np.abs(scale_curve))),
        "thermal_drift_nm": thermal_drift_nm,
        "absolute_accuracy_peak_nm": float(np.max(np.abs(absolute_curve))),
        "total_wavelength_error_max_abs_nm": total_max,
        "total_wavelength_error_rms_nm": float(np.sqrt(np.mean(total_curve**2))),
        "frame_gain_error_rel": frame_gain_error_rel,
        "reflectance_offset_abs": reflectance_offset_abs,
    }
    components = {
        "laser_residual_nm": laser_curve,
        "axis_offset_nm": offset_curve,
        "axis_scale_nm": scale_curve,
        "thermal_drift_nm": thermal_curve,
        "absolute_accuracy_nm": absolute_curve,
        "total_nm": total_curve,
    }
    return metadata, components


class OpticalSolver:
    def __init__(self, backend: str, bridge_dir: Path):
        self.backend = backend
        self.bridge_dir = bridge_dir
        self.fdtd = None

    def __enter__(self):
        if self.backend == "api":
            if lumapi is None:
                raise RuntimeError("lumapi is unavailable; use --backend tmm or batch.")
            self.fdtd = lumapi.FDTD(hide=True, serverArgs={ "use-solve": True})
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    def reflectance(
        self,
        wavelengths_um: np.ndarray,
        thicknesses_um: dict[str, float],
        reflector_angle_deg: float,
        n_real_rel_delta: dict[str, float],
        k_rel_delta: dict[str, float],
    ) -> np.ndarray:
        if self.backend == "tmm":
            return tmm_reflectance(
                wavelengths_um,
                thicknesses_um,
                reflector_angle_deg,
                n_real_rel_delta,
                k_rel_delta,
            )
        n_matrix = np.vstack([
            material_n(name, wavelengths_um, n_real_rel_delta, k_rel_delta)
            for name in LAYER_NAMES
        ])
        thicknesses_m = np.asarray([thicknesses_um[name] for name in LAYER_NAMES]) * 1.0e-6
        frequencies_hz = C0_M_S / (np.asarray(wavelengths_um) * 1.0e-6)
        if self.backend == "api":
            result = self.fdtd.stackrt(n_matrix, thicknesses_m, frequencies_hz, reflector_angle_deg)
            return np.asarray(result["Rp"], dtype=float).reshape(-1)
        return self._batch(n_matrix, thicknesses_m, frequencies_hz, reflector_angle_deg)

    def _batch(
        self,
        n_matrix: np.ndarray,
        thicknesses_m: np.ndarray,
        frequencies_hz: np.ndarray,
        reflector_angle_deg: float,
    ) -> np.ndarray:
        stage = self.bridge_dir
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "appdata" / "Ansys").mkdir(parents=True, exist_ok=True)
        np.savetxt(stage / "n_real.txt", n_matrix.real, fmt="%.17g")
        np.savetxt(stage / "n_imag.txt", n_matrix.imag, fmt="%.17g")
        np.savetxt(stage / "freqs.txt", frequencies_hz[:, None], fmt="%.17g")
        np.savetxt(stage / "thicknesses.txt", thicknesses_m[:, None], fmt="%.17g")
        script = (
            'n_real=readdata("n_real.txt");\n'
            'n_imag=readdata("n_imag.txt");\n'
            'n_matrix=n_real+1i*n_imag;\n'
            'freqs=readdata("freqs.txt");\n'
            'thicknesses=readdata("thicknesses.txt");\n'
            f'result=stackrt(n_matrix,thicknesses,freqs,{reflector_angle_deg:.17g});\n'
            'spectra=transpose(result.Rp);\n'
            'matlabsave("stackrt_static_output",spectra);\n'
            'exit;\n'
        )
        (stage / "run_stackrt_static.lsf").write_text(script, encoding="ascii")
        executable = Path(r"D:\Program Files\Lumerical\v241\bin\fdtd-solutions.exe")
        if not executable.exists():
            raise FileNotFoundError(executable)
        environment = os.environ.copy()
        environment["APPDATA"] = str(stage / "appdata")
        started = time.time()
        completed = subprocess.run(
            [str(executable), "-nw", "-trust-script", "-run", "run_stackrt_static.lsf"],
            cwd=stage,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        (stage / "stackrt_batch.log").write_text(
            f"returncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
            encoding="utf-8",
        )
        output_path = stage / "stackrt_static_output.mat"
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_mtime < started - 1.0:
            raise RuntimeError(f"StackRT batch failed; see {stage / 'stackrt_batch.log'}")
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("The batch backend requires h5py.") from exc
        with h5py.File(output_path, "r") as handle:
            return np.asarray(handle["spectra"], dtype=float).reshape(-1)


class StaticDatasetGenerator:
    def __init__(self, backend: str, output_dir: Path):
        self.backend = backend
        self.output_dir = output_dir
        start_nm = CONFIG["WAVELENGTH_START_UM"] * 1000.0
        stop_nm = CONFIG["WAVELENGTH_STOP_UM"] * 1000.0
        step_nm = CONFIG["SPECTRAL_SAMPLING_NM"]
        count = int(round((stop_nm - start_nm) / step_nm)) + 1
        self.reported_nm = np.linspace(start_nm, stop_nm, count) # 这个是名义波长轴
        self.layers = dict(CONFIG["LAYERS"])

    def generate_one(
        self,
        solver: OpticalSolver,
        case_name: str,
        realization_index: int,
        seed: int,
    ) -> tuple[dict, dict]:
        rng = np.random.default_rng(seed)
        noise, components = realize_noise(case_name, self.reported_nm, rng) # 这里施加了波长误差吗？
        physical_nm = self.reported_nm + components["total_nm"] # 这个是发生偏移后的波长轴，最后看解算的文件里面是不是用了偏移后的波长轴
        started = time.perf_counter()
        physical_spectrum = solver.reflectance(  #生成对应的真实光谱，不包括其他类噪声
            physical_nm / 1000.0,
            self.layers,
            noise["reflector_angle_deg"],
            noise["material_n_real_rel_delta"],
            noise["material_k_rel_delta"],
        )
        generation_runtime_s = time.perf_counter() - started
        if CONFIG["ILS_ENABLED"]:
            raise RuntimeError("V8 requires ILS to be disabled.")
        measured = physical_spectrum * (1.0 + noise["frame_gain_error_rel"])  # 这以后叠加的都是幅度噪声
        measured += noise["reflectance_offset_abs"]
        clip_mask = (measured < 0.0) | (measured > 1.0)
        measured = np.clip(measured, 0.0, 1.0)
        metadata = {
            "noise_case": case_name,
            "noise_factor": noise["factor"],
            "noise_level": noise["level"],
            "realization_index": int(realization_index),
            "random_seed": int(seed),
            "generation_runtime_s": float(generation_runtime_s),
            "reflectance_clip_fraction": float(np.mean(clip_mask)),
            "backend": self.backend,
            **noise,
        }
        return {
            "physical_wavelengths_nm": physical_nm,
            "physical_spectrum": np.asarray(physical_spectrum, dtype=float),
            "measured_spectrum": np.asarray(measured, dtype=float),
            "components": components,
            "metadata": metadata,
        }, metadata

    def save_one(self, data: dict) -> Path:
        metadata = data["metadata"]
        path = self.output_dir / (
            f"static_spectrum_{metadata['noise_case']}_r{metadata['realization_index']:04d}_"
            f"seed{metadata['random_seed']}.npz"
        )
        material_names = np.asarray(PERTURBED_MATERIALS, dtype="U16")
        components = data["components"]
        np.savez_compressed(
            path,
            wavelengths=self.reported_nm / 1000.0,
            reported_wavelengths_nm=self.reported_nm,
            physical_wavelengths=data["physical_wavelengths_nm"] / 1000.0,
            wavelength_error_total_nm=components["total_nm"],
            wavelength_error_laser_residual_nm=components["laser_residual_nm"],
            wavelength_error_axis_offset_nm=components["axis_offset_nm"],
            wavelength_error_axis_scale_nm=components["axis_scale_nm"],
            wavelength_error_thermal_drift_nm=components["thermal_drift_nm"],
            wavelength_error_absolute_accuracy_nm=components["absolute_accuracy_nm"],
            spectrum_measured=data["measured_spectrum"],
            spectrum_physical=data["physical_spectrum"],
            layer_names=np.asarray(LAYER_NAMES, dtype="U32"),
            layer_thickness_um=np.asarray([self.layers[name] for name in LAYER_NAMES]),
            true_air_um=np.asarray(self.layers["Air"]),
            true_reflector_angle_deg=np.asarray(metadata["reflector_angle_deg"]),
            perturbed_material_names=material_names,
            material_n_real_rel_delta=np.asarray([
                metadata["material_n_real_rel_delta"][name] for name in material_names
            ]),
            material_k_rel_delta=np.asarray([
                metadata["material_k_rel_delta"][name] for name in material_names
            ]),
            noise_case=np.asarray(metadata["noise_case"]),
            noise_factor=np.asarray(metadata["noise_factor"]),
            noise_level=np.asarray(metadata["noise_level"]),
            realization_index=np.asarray(metadata["realization_index"]),
            random_seed=np.asarray(metadata["random_seed"]),
            frame_gain_error_rel=np.asarray(metadata["frame_gain_error_rel"]),
            reflectance_offset_abs=np.asarray(metadata["reflectance_offset_abs"]),
            total_wavelength_error_max_abs_nm=np.asarray(metadata["total_wavelength_error_max_abs_nm"]),
            total_wavelength_error_rms_nm=np.asarray(metadata["total_wavelength_error_rms_nm"]),
            ils_enabled=np.asarray(False),
            frames_per_realization=np.asarray(1),
            time_series_enabled=np.asarray(False),
            modulation_enabled=np.asarray(False),
            optical_backend=np.asarray(self.backend),
            generator_version=np.asarray(GENERATOR_VERSION),
            speed_of_light_m_s=np.asarray(C0_M_S),
            config_json=np.asarray(json.dumps(CONFIG, ensure_ascii=False, sort_keys=True)),
            noise_realization_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        return path



def save_representative_plot(output_dir: Path, representatives: dict[str, dict]) -> Path:
    cases = list(representatives)
    fig, axes = plt.subplots(
        len(cases), 3, figsize=(19, max(5, 3.1 * len(cases))), squeeze=False, constrained_layout=True,
    )
    for row_index, case_name in enumerate(cases):
        data = representatives[case_name]
        components = data["components"]
        wavelength_nm = data["physical_wavelengths_nm"]
        axes[row_index, 0].plot(wavelength_nm, data["measured_spectrum"], lw=0.7)
        axes[row_index, 0].set_title(f"{case_name}: measured spectrum")
        axes[row_index, 0].set_ylabel("Reflectance")
        axes[row_index, 1].plot(wavelength_nm, components["total_nm"], lw=0.9)
        axes[row_index, 1].axhline(0.0, color="black", lw=0.6)
        axes[row_index, 1].set_title(
            f"total wavelength error, max={np.max(np.abs(components['total_nm'])):.4g} nm"
        )
        axes[row_index, 1].set_ylabel("Wavelength error (nm)")
        for name, values in components.items():
            if name != "total_nm" and np.any(values):
                axes[row_index, 2].plot(wavelength_nm, values, lw=0.8, label=name.replace("_nm", ""))
        axes[row_index, 2].set_title("Wavelength-error components")
        axes[row_index, 2].set_ylabel("Component (nm)")
        if axes[row_index, 2].lines:
            axes[row_index, 2].legend(fontsize=7, ncol=2)
        for ax in axes[row_index]:
            ax.set_xlabel("Wavelength (nm)")
            ax.grid(True, alpha=0.3)
    path = output_dir / "representative_static_noise_realizations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def wavelength_budget_audit() -> dict:
    half_span_nm = 0.5 * (
        CONFIG["WAVELENGTH_STOP_UM"] - CONFIG["WAVELENGTH_START_UM"]
    ) * 1000.0
    audit = {}
    for level, values in NOISE_LEVELS.items():
        scale_edge_nm = values["axis_scale_max_ppm"] * 1.0e-6 * half_span_nm
        worst_sum_nm = (
            values["laser_residual_max_nm"]
            + values["axis_offset_max_nm"]
            + scale_edge_nm
            + values["thermal_drift_max_nm"]
            + values["absolute_accuracy_max_nm"]
        )
        audit[level] = {
            "scale_edge_max_nm": scale_edge_nm,
            "component_triangle_bound_nm": worst_sum_nm,
            "within_0p2_nm": bool(worst_sum_nm <= WAVELENGTH_ERROR_HARD_MAX_NM),
        }
    if not all(item["within_0p2_nm"] for item in audit.values()):
        raise ValueError("Configured wavelength-error budget can exceed 0.2 nm.")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V8 single-frame StackRT/TMM Monte Carlo dataset generator." # 蒙特卡洛是指采用多个repeat对随机过程采样得到统计分布
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--repeats", type=int, default=30, help="Realizations per non-clean case") # 这个参数是去除随机种子的影响，所以设置30次同噪声等级输入下的结果？
    parser.add_argument("--clean-repeats", type=int, default=1) # 对于clean noise case的重复次数，默认是1次
    parser.add_argument("--backend", choices=["tmm", "api", "batch"], default="api") # 这个是计算光学反射率的求解器后端，默认采用lumapi，不可用时会报错，也可以选择采用tmm模型生成光谱
    parser.add_argument("--seed", type=int, default=20260810) # 这个是随机种子，可以改成当前日期
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--describe", action="store_true") # 如果true，则增加输出配置和噪声等级信息
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = wavelength_budget_audit()
    if args.describe:
        print(json.dumps({
            "version": GENERATOR_VERSION,
            "config": CONFIG,
            "noise_levels": NOISE_LEVELS,
            "noise_factors": NOISE_FACTORS,
            "cases": all_case_names(),
            "wavelength_budget": budget,
            "realization_fraction_range": [REALIZATION_MIN_FRACTION, 1.0],
        }, indent=2, ensure_ascii=False))
        return
    if args.repeats < 1 or args.clean_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    cases = select_cases(args.cases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(__file__).resolve().parents[2] / "04_results_and_datasets" # 默认输出结果路径为父路径下的04_results_and_datasets
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else output_root / f"static_stackrt_v8_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    generator = StaticDatasetGenerator(args.backend, output_dir)
    rows = []
    failures = []
    representatives = {}
    with OpticalSolver(args.backend, output_dir / "_stackrt_batch_bridge") as solver: # 光学求解器封装
        for case_index, case_name in enumerate(cases):
            repeats = args.clean_repeats if case_name == "clean" else args.repeats
            for realization_index in range(repeats):
                seed = int(args.seed + case_index * 1_000_000 + realization_index) #不同的noise case下的realization的随机种子不同
                try:
                    data, metadata = generator.generate_one(solver, case_name, realization_index, seed) # 生成一个噪声case下的光谱数据
                    npz_path = generator.save_one(data) # 保存为npz文件，保留配置信息
                    row = {
                        "noise_case": metadata["noise_case"],
                        "noise_factor": metadata["noise_factor"],
                        "noise_level": metadata["noise_level"],
                        "realization_index": metadata["realization_index"],
                        "random_seed": metadata["random_seed"],
                        "reflector_angle_deg": metadata["reflector_angle_deg"],
                        "total_wavelength_error_max_abs_nm": metadata["total_wavelength_error_max_abs_nm"],
                        "total_wavelength_error_rms_nm": metadata["total_wavelength_error_rms_nm"],
                        "frame_gain_error_rel": metadata["frame_gain_error_rel"],
                        "reflectance_offset_abs": metadata["reflectance_offset_abs"],
                        "generation_runtime_s": metadata["generation_runtime_s"],
                        "npz_path": str(npz_path),
                    }
                    rows.append(row)
                    representatives.setdefault(case_name, data)
                    print(
                        f"[{case_name} {realization_index + 1}/{repeats}] "
                        f"wavelength_max={metadata['total_wavelength_error_max_abs_nm']:.6g} nm, "
                        f"angle={metadata['reflector_angle_deg']:.6g} deg, "
                        f"runtime={metadata['generation_runtime_s']:.4f}s"
                    )
                except Exception as exc:
                    failures.append({
                        "case": case_name,
                        "realization": realization_index,
                        "seed": seed,
                        "error": str(exc),
                    })
                    print(f"ERROR [{case_name} {realization_index + 1}/{repeats}]: {exc}")
    index_path = output_dir / "dataset_index.csv"
    if rows:
        with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_path = save_representative_plot(output_dir, representatives) if representatives else None
    manifest = {
        "version": GENERATOR_VERSION,
        "created": timestamp,
        "backend": args.backend,
        "cases": cases,
        "repeats_non_clean": args.repeats,
        "repeats_clean": args.clean_repeats,
        "random_seed": args.seed,
        "config": CONFIG,
        "noise_levels": NOISE_LEVELS,
        "noise_factors": NOISE_FACTORS,
        "realization_fraction_range": [REALIZATION_MIN_FRACTION, 1.0],
        "wavelength_error_hard_max_nm": WAVELENGTH_ERROR_HARD_MAX_NM,
        "wavelength_budget_audit": budget,
        "dataset_count": len(rows),
        "dataset_index": str(index_path) if rows else None,
        "representative_plot": str(plot_path) if plot_path else None,
        "failures": failures,
    }
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"MANIFEST={manifest_path}")
    if failures:
        raise RuntimeError(f"Simulation failures: {failures}")


if __name__ == "__main__":
    main()
