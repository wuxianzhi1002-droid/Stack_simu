from __future__ import annotations

import argparse
import json
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


LUMERICAL_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
if LUMERICAL_PATH.exists():
    if str(LUMERICAL_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_PATH))
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    lumapi = None

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GENERATOR_VERSION = "main_dynamic_v5"
ANGLE_ABS_MAX_DEG = 0.1

CONFIG = {
    "MODEL_TYPE": "PSS_TiO2",
    "WAVELENGTH_START": 0.2,
    "WAVELENGTH_STOP": 0.6,
    "SPECTRAL_RESOLUTION_NM": 0.02,
    "MODULATION": {
        "f_Hz": 1000.0,
        "A_nm": 5.0,
        "T_s": 0.01,
        "fs_Hz": 40000.0,
    },
    "PSS_TIO2_MODEL": {
        "LAYERS": [
            ("RefReflector", 0.0),
            ("Air", 1000.0),
            ("HSQ", 0.030),
            ("PSS", 0.010),
            ("SOC", 0.040),
            ("TiO2", 0.040),
            ("Cu", 0.0),
        ]
    },
}

# Engineering robustness sweep. These are not instrument specifications.
# Wavelength levels are reduced from v4 because a 1 mm cavity has an FSR of
# only about 0.08 nm at 400 nm.
NOISE_LEVELS = {
    "low": {
        "n_real_sigma_rel": 5.0e-4,
        "k_sigma_rel": 1.0e-2,
        "angle_sigma_deg": 0.01,
        "wavelength_offset_sigma_nm": 2.0e-4,
        "amplitude_sigma_rel": 1.0e-3,
        "frame_gain_sigma_rel": 2.0e-4,
        "reflectance_sigma_abs": 2.0e-4,
    },
    "medium": {
        "n_real_sigma_rel": 2.0e-3,
        "k_sigma_rel": 5.0e-2,
        "angle_sigma_deg": 0.05,
        "wavelength_offset_sigma_nm": 1.0e-3,
        "amplitude_sigma_rel": 5.0e-3,
        "frame_gain_sigma_rel": 1.0e-3,
        "reflectance_sigma_abs": 1.0e-3,
    },
    "high": {
        "n_real_sigma_rel": 5.0e-3,
        "k_sigma_rel": 1.0e-1,
        "angle_sigma_deg": 0.10,
        "wavelength_offset_sigma_nm": 5.0e-3,
        "amplitude_sigma_rel": 1.0e-2,
        "frame_gain_sigma_rel": 5.0e-3,
        "reflectance_sigma_abs": 5.0e-3,
    },
}
NOISE_FACTORS = ("angle", "wavelength", "material", "amplitude", "detector", "combined")
PERTURBED_MATERIALS = ("HSQ", "PSS", "SOC", "TiO2")
PROFILE_KEYS = tuple(next(iter(NOISE_LEVELS.values())).keys())


def zero_profile() -> dict[str, float]:
    return {name: 0.0 for name in PROFILE_KEYS}


def active_profile(level: str, factor: str) -> dict[str, float]:
    if level == "clean":
        return zero_profile()
    source = NOISE_LEVELS[level]
    if factor == "combined":
        return dict(source)
    profile = zero_profile()
    keys = {
        "angle": ("angle_sigma_deg",),
        "wavelength": ("wavelength_offset_sigma_nm",),
        "material": ("n_real_sigma_rel", "k_sigma_rel"),
        "amplitude": ("amplitude_sigma_rel",),
        "detector": ("frame_gain_sigma_rel", "reflectance_sigma_abs"),
    }[factor]
    for key in keys:
        profile[key] = source[key]
    return profile


def sample_bounded_angle(rng: np.random.Generator, sigma_deg: float) -> float:
    if sigma_deg <= 0.0:
        return 0.0
    for _ in range(10000):
        value = abs(float(rng.normal(0.0, sigma_deg)))
        if value <= ANGLE_ABS_MAX_DEG:
            return value
    raise RuntimeError("Unable to sample an angle inside the configured bound.")


def realize_noise(level: str, factor: str, rng: np.random.Generator) -> dict:
    profile = active_profile(level, factor)
    n_delta = {
        name: float(rng.normal(0.0, profile["n_real_sigma_rel"]))
        for name in PERTURBED_MATERIALS
    }
    k_delta = {
        name: float(rng.normal(0.0, profile["k_sigma_rel"]))
        for name in PERTURBED_MATERIALS
    }
    return {
        "case": "clean" if level == "clean" else f"{factor}_{level}",
        "level": level,
        "factor": factor,
        "profile_config": profile,
        "material_n_real_rel_delta": n_delta,
        "material_k_rel_delta": k_delta,
        "angle_input_convention": "nonnegative StackRT first-incident-medium angle in degrees",
        "angle_limit_deg": ANGLE_ABS_MAX_DEG,
        "angle_error_deg": sample_bounded_angle(rng, profile["angle_sigma_deg"]),
        "wavelength_offset_nm": float(rng.normal(0.0, profile["wavelength_offset_sigma_nm"])),
        "amplitude_rel_error": float(rng.normal(0.0, profile["amplitude_sigma_rel"])),
    }


def all_case_names() -> list[str]:
    return ["clean"] + [f"{factor}_{level}" for factor in NOISE_FACTORS for level in NOISE_LEVELS]


def select_cases(selector: str) -> list[str]:
    if selector == "all":
        return all_case_names()
    if selector == "combined":
        return ["clean"] + [f"combined_{level}" for level in NOISE_LEVELS]
    if selector in NOISE_FACTORS:
        return ["clean"] + [f"{selector}_{level}" for level in NOISE_LEVELS]
    requested = [item.strip() for item in selector.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(all_case_names()))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    return requested


def split_case(case_name: str) -> tuple[str, str]:
    if case_name == "clean":
        return "clean", "clean"
    factor, level = case_name.rsplit("_", 1)
    return level, factor


class DynamicSimulator:
    def __init__(
        self,
        config: dict,
        case_name: str,
        random_seed: int,
        backend: str = "batch",
        batch_dir: Path | None = None,
    ):
        self.config = config
        self.case_name = case_name
        self.backend = backend
        self.batch_dir = batch_dir
        self.random_seed = int(random_seed)
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        level, factor = split_case(case_name)
        self.noise = realize_noise(level, factor, self.rng)

        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000.0
        count = int(span_nm / config["SPECTRAL_RESOLUTION_NM"]) + 1
        self.wavelengths = np.linspace(config["WAVELENGTH_START"], config["WAVELENGTH_STOP"], count)
        self.physical_wavelengths = self.wavelengths + self.noise["wavelength_offset_nm"] / 1000.0
        self.freqs = 3.0e8 / (self.physical_wavelengths * 1.0e-6)

        modulation = config["MODULATION"]
        self.fs = float(modulation["fs_Hz"])
        self.f_mod = float(modulation["f_Hz"])
        self.duration = float(modulation["T_s"])
        self.nominal_A_um = float(modulation["A_nm"]) / 1000.0
        self.actual_A_um = self.nominal_A_um * (1.0 + self.noise["amplitude_rel_error"])
        self.angle_deg = float(self.noise["angle_error_deg"])
        if not 0.0 <= self.angle_deg <= ANGLE_ABS_MAX_DEG:
            raise ValueError(f"StackRT input angle {self.angle_deg} exceeds [0, {ANGLE_ABS_MAX_DEG}] deg.")

        self.t_axis = np.arange(0.0, self.duration, 1.0 / self.fs)
        self.Nt = len(self.t_axis)
        print(
            f"[{case_name}] seed={self.random_seed}, Nt={self.Nt}, N_lambda={count}, "
            f"angle={self.angle_deg:.8g} deg, wavelength_offset={self.noise['wavelength_offset_nm']:.8g} nm, "
            f"A={self.actual_A_um * 1000.0:.8g} nm"
        )

    def _get_n_matrix(self) -> tuple[np.ndarray, np.ndarray, int]:
        layers = self.config["PSS_TIO2_MODEL"]["LAYERS"]
        n_matrix = np.zeros((len(layers), len(self.freqs)), dtype=np.complex128)
        thicknesses = []
        air_idx = -1
        w_um = self.physical_wavelengths

        for idx, (material, thickness_um) in enumerate(layers):
            thicknesses.append(float(thickness_um) * 1.0e-6)
            if material == "RefReflector":
                values = np.full_like(w_um, 5.8284, dtype=np.complex128)
            elif material == "Air":
                air_idx = idx
                values = np.ones_like(w_um, dtype=np.complex128)
            elif material == "HSQ":
                values = np.full_like(w_um, 1.41, dtype=np.complex128)
            elif material == "PSS":
                values = np.full_like(w_um, 1.50 + 0.05j, dtype=np.complex128)
            elif material == "SOC":
                values = (1.55 + 0.005 / (w_um**2)).astype(np.complex128)
            elif material == "TiO2":
                values = (2.4 + 0.02 / (w_um**2)).astype(np.complex128)
            elif material == "Cu":
                values = np.full_like(w_um, 1.1 + 2.5j, dtype=np.complex128)
            elif isinstance(material, (int, float, complex)):
                values = np.full_like(w_um, material, dtype=np.complex128)
            else:
                raise ValueError(f"Unknown material: {material}")

            if material in PERTURBED_MATERIALS:
                dn = self.noise["material_n_real_rel_delta"][material]
                dk = self.noise["material_k_rel_delta"][material]
                values = values.real * (1.0 + dn) + 1j * values.imag * (1.0 + dk)
            n_matrix[idx] = values

        return n_matrix, np.asarray(thicknesses), air_idx

    def _run_stackrt_batch(
        self,
        n_matrix: np.ndarray,
        base_thicknesses: np.ndarray,
        air_idx: int,
        L_t_m: np.ndarray,
    ) -> np.ndarray:
        if self.batch_dir is None:
            raise ValueError("batch_dir is required for the batch backend.")
        stage = Path(self.batch_dir)
        stage.mkdir(parents=True, exist_ok=True)
        appdata_dir = stage / "appdata" / "Ansys"
        appdata_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(stage / "n_real.txt", n_matrix.real, fmt="%.17g")
        np.savetxt(stage / "n_imag.txt", n_matrix.imag, fmt="%.17g")
        np.savetxt(stage / "freqs.txt", self.freqs[:, None], fmt="%.17g")
        np.savetxt(stage / "thicknesses.txt", base_thicknesses[:, None], fmt="%.17g")
        np.savetxt(stage / "L_t.txt", L_t_m[:, None], fmt="%.17g")

        script = (
            'n_real=readdata("n_real.txt");\n'
            'n_imag=readdata("n_imag.txt");\n'
            'n_matrix=n_real+1i*n_imag;\n'
            'freqs=readdata("freqs.txt");\n'
            'thicknesses=readdata("thicknesses.txt");\n'
            'L_t=readdata("L_t.txt");\n'
            'Nt=length(L_t);\n'
            'Nf=length(freqs);\n'
            'spectra=matrix(Nt,Nf);\n'
            f'for(i=1:Nt){{current=thicknesses;current({air_idx + 1})=L_t(i);'
            f'result=stackrt(n_matrix,current,freqs,{self.angle_deg:.17g});'
            'spectra(i,1:Nf)=transpose(result.Rp);}\n'
            'matlabsave("stackrt_dynamic_output",spectra);\n'
            'exit;\n'
        )
        (stage / "run_stackrt.lsf").write_text(script, encoding="ascii")
        executable = Path(r"D:\Program Files\Lumerical\v241\bin\fdtd-solutions.exe")
        if not executable.exists():
            raise FileNotFoundError(executable)

        environment = os.environ.copy()
        environment["APPDATA"] = str(stage / "appdata")
        started = time.time()
        print(f"[{self.case_name}] launching native StackRT batch backend")
        completed = subprocess.run(
            [str(executable), "-nw", "-trust-script", "-run", "run_stackrt.lsf"],
            cwd=stage,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        log_text = (
            f"returncode={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
        (stage / "stackrt_batch.log").write_text(log_text, encoding="utf-8")
        output_path = stage / "stackrt_dynamic_output.mat"
        output_is_fresh = (
            output_path.exists() and output_path.stat().st_mtime >= started - 1.0
        )
        if completed.returncode != 0 or not output_is_fresh:
            raise RuntimeError(f"StackRT batch failed; see {stage / 'stackrt_batch.log'}")

        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "The batch backend requires h5py. Run with the Anaconda base Python."
            ) from exc
        with h5py.File(output_path, "r") as handle:
            physical_spectra = np.asarray(handle["spectra"], dtype=float).T
        expected = (self.Nt, len(self.wavelengths))
        if physical_spectra.shape != expected:
            raise ValueError(
                f"Unexpected StackRT batch shape {physical_spectra.shape}; expected {expected}."
            )
        print(
            f"[{self.case_name}] native StackRT batch complete, "
            f"elapsed={time.time() - started:.1f}s"
        )
        return physical_spectra

    def run_dynamic_sequence(self) -> dict:
        if self.backend == "api" and lumapi is None:
            raise RuntimeError("lumapi is not available.")

        n_matrix, base_thicknesses, air_idx = self._get_n_matrix()
        if air_idx < 0:
            raise ValueError("Model does not contain an Air layer.")

        L_t_m = base_thicknesses[air_idx] + self.actual_A_um * 1.0e-6 * np.sin(
            2.0 * np.pi * self.f_mod * self.t_axis
        )
        if self.backend == "batch":
            physical_spectra = self._run_stackrt_batch(
                n_matrix, base_thicknesses, air_idx, L_t_m
            )
        elif self.backend == "api":
            physical_spectra = np.empty((self.Nt, len(self.wavelengths)), dtype=np.float64)
            fdtd = lumapi.FDTD(
                hide=True,
                serverArgs={"platform": "offscreen", "use-solve": True},
            )
            started = time.time()
            try:
                for idx, air_length_m in enumerate(L_t_m):
                    current = base_thicknesses.copy()
                    current[air_idx] = air_length_m
                    result = fdtd.stackrt(n_matrix, current, self.freqs, self.angle_deg)
                    physical_spectra[idx] = np.asarray(result["Rp"]).reshape(-1)
                    if (idx + 1) % max(1, self.Nt // 4) == 0:
                        print(
                            f"[{self.case_name}] StackRT {idx + 1}/{self.Nt}, "
                            f"elapsed={time.time() - started:.1f}s"
                        )
            finally:
                fdtd.close()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        profile = self.noise["profile_config"]
        frame_gain_error = self.rng.normal(0.0, profile["frame_gain_sigma_rel"], size=self.Nt)
        measured = physical_spectra * (1.0 + frame_gain_error[:, None])
        if profile["reflectance_sigma_abs"] > 0.0:
            measured += self.rng.normal(0.0, profile["reflectance_sigma_abs"], size=measured.shape)
        clip_mask = (measured < 0.0) | (measured > 1.0)
        clip_fraction = float(np.mean(clip_mask))
        measured = np.clip(measured, 0.0, 1.0)

        layers = self.config["PSS_TIO2_MODEL"]["LAYERS"]
        return {
            "t_axis": self.t_axis,
            "wavelengths": self.wavelengths,
            "physical_wavelengths": self.physical_wavelengths,
            "L_t": L_t_m * 1.0e6,
            "spectra": measured,
            "spectra_physical": physical_spectra,
            "frame_gain_error": frame_gain_error,
            "reflectance_clip_fraction": clip_fraction,
            "noise_case": self.case_name,
            "noise_level": self.noise["level"],
            "noise_factor": self.noise["factor"],
            "random_seed": self.random_seed,
            "noise_realization": self.noise,
            "nominal_amplitude_nm": self.nominal_A_um * 1000.0,
            "actual_amplitude_nm": self.actual_A_um * 1000.0,
            "actual_angle_deg": self.angle_deg,
            "modulation_frequency_hz": self.f_mod,
            "sampling_rate_hz": self.fs,
            "layer_names": [str(layer[0]) for layer in layers],
            "layer_thickness_um": [float(layer[1]) for layer in layers],
        }


class DynamicAnalyzer:
    @staticmethod
    def lockin_harmonics(data: dict, f_mod: float, A_um: float, harmonics=(1, 2, 3)) -> dict:
        t_axis = np.asarray(data["t_axis"], dtype=float)
        spectra = np.asarray(data["spectra"], dtype=float)
        if spectra.ndim != 2 or spectra.shape[0] != len(t_axis):
            raise ValueError("spectra must have shape (Nt, N_lambda).")
        if A_um <= 0.0:
            raise ValueError("A_um must be positive.")

        spectra_ac = spectra - np.mean(spectra, axis=0, keepdims=True)
        results = {}
        for harmonic in harmonics:
            phase = 2.0 * np.pi * harmonic * f_mod * t_axis
            sin_ref = np.sin(phase)
            cos_ref = np.cos(phase)
            X = 2.0 * (spectra_ac.T @ sin_ref) / len(t_axis)
            Y = 2.0 * (spectra_ac.T @ cos_ref) / len(t_axis)
            results[f"{harmonic}f"] = {
                "X": X,
                "Y": Y,
                "R": np.hypot(X, Y),
                "phase": np.arctan2(Y, X),
            }
        results["dIdL_1f"] = results["1f"]["R"] / A_um
        results["dIdL_1f_X"] = results["1f"]["X"] / A_um
        return results

    @staticmethod
    def save_and_plot(data: dict, save_dir: Path) -> dict:
        save_dir.mkdir(parents=True, exist_ok=True)
        case_name = data["noise_case"]
        file_tag = f"{case_name}_{TIMESTAMP}"
        f_mod = data["modulation_frequency_hz"]
        nominal_A_um = data["nominal_amplitude_nm"] / 1000.0
        actual_A_um = data["actual_amplitude_nm"] / 1000.0
        lockin = DynamicAnalyzer.lockin_harmonics(data, f_mod, nominal_A_um)
        physical_lockin = DynamicAnalyzer.lockin_harmonics(
            {"t_axis": data["t_axis"], "spectra": data["spectra_physical"]},
            f_mod,
            actual_A_um,
        )

        material_names = np.asarray(PERTURBED_MATERIALS, dtype="U16")
        n_deltas = np.asarray([
            data["noise_realization"]["material_n_real_rel_delta"][name] for name in material_names
        ])
        k_deltas = np.asarray([
            data["noise_realization"]["material_k_rel_delta"][name] for name in material_names
        ])
        noise_json = json.dumps(data["noise_realization"], ensure_ascii=False, sort_keys=True)

        npz_path = save_dir / f"dynamic_spectra_{file_tag}.npz"
        np.savez_compressed(
            npz_path,
            t_axis=data["t_axis"],
            wavelengths=data["wavelengths"],
            physical_wavelengths=data["physical_wavelengths"],
            L_t=data["L_t"],
            spectra=data["spectra"],
            spectra_physical_mean=np.mean(data["spectra_physical"], axis=0),
            lockin_1f_X=lockin["1f"]["X"],
            lockin_1f_Y=lockin["1f"]["Y"],
            lockin_1f_R=lockin["1f"]["R"],
            lockin_1f_phase=lockin["1f"]["phase"],
            lockin_2f_X=lockin["2f"]["X"],
            lockin_2f_Y=lockin["2f"]["Y"],
            lockin_2f_R=lockin["2f"]["R"],
            lockin_2f_phase=lockin["2f"]["phase"],
            lockin_3f_X=lockin["3f"]["X"],
            lockin_3f_Y=lockin["3f"]["Y"],
            lockin_3f_R=lockin["3f"]["R"],
            lockin_3f_phase=lockin["3f"]["phase"],
            dIdL_1f=lockin["dIdL_1f"],
            dIdL_1f_X=lockin["dIdL_1f_X"],
            dIdL_1f_actual_amplitude=lockin["1f"]["R"] / actual_A_um,
            physical_lockin_1f_X=physical_lockin["1f"]["X"],
            physical_lockin_1f_Y=physical_lockin["1f"]["Y"],
            physical_lockin_1f_R=physical_lockin["1f"]["R"],
            physical_dIdL_1f=physical_lockin["dIdL_1f"],
            noise_case=np.asarray(case_name),
            noise_level=np.asarray(data["noise_level"]),
            noise_factor=np.asarray(data["noise_factor"]),
            random_seed=np.asarray(data["random_seed"], dtype=np.int64),
            generator_version=np.asarray(GENERATOR_VERSION),
            noise_config_json=np.asarray(noise_json),
            nominal_amplitude_nm=np.asarray(data["nominal_amplitude_nm"]),
            actual_amplitude_nm=np.asarray(data["actual_amplitude_nm"]),
            modulation_frequency_hz=np.asarray(data["modulation_frequency_hz"]),
            sampling_rate_hz=np.asarray(data["sampling_rate_hz"]),
            nominal_angle_deg=np.asarray(0.0),
            actual_angle_deg=np.asarray(data["actual_angle_deg"]),
            angle_limit_deg=np.asarray(ANGLE_ABS_MAX_DEG),
            angle_input_convention=np.asarray(data["noise_realization"]["angle_input_convention"]),
            wavelength_offset_nm=np.asarray(data["noise_realization"]["wavelength_offset_nm"]),
            frame_gain_error=data["frame_gain_error"],
            reflectance_clip_fraction=np.asarray(data["reflectance_clip_fraction"]),
            perturbed_material_names=material_names,
            material_n_real_rel_delta=n_deltas,
            material_k_rel_delta=k_deltas,
            layer_names=np.asarray(data["layer_names"], dtype="U32"),
            layer_thickness_um=np.asarray(data["layer_thickness_um"], dtype=float),
        )

        metadata_path = save_dir / f"noise_realization_{file_tag}.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "generator_version": GENERATOR_VERSION,
                    "noise_case": case_name,
                    "noise_level": data["noise_level"],
                    "noise_factor": data["noise_factor"],
                    "random_seed": data["random_seed"],
                    "noise_realization": data["noise_realization"],
                    "nominal_amplitude_nm": data["nominal_amplitude_nm"],
                    "actual_amplitude_nm": data["actual_amplitude_nm"],
                    "actual_angle_deg": data["actual_angle_deg"],
                    "reflectance_clip_fraction": data["reflectance_clip_fraction"],
                    "npz_path": str(npz_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        w_nm = data["wavelengths"] * 1000.0
        fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        panels = [
            ("1f", "tab:blue", "1f R"),
            ("2f", "tab:orange", "2f R"),
            ("3f", "tab:green", "3f R"),
        ]
        for ax, (key, color, title) in zip(axs.flat[:3], panels):
            ax.plot(w_nm, lockin[key]["R"], color=color, lw=0.8)
            ax.set_title(title)
            ax.set_xlabel("Wavelength (nm)")
            ax.grid(True, alpha=0.35)
        axs[1, 1].plot(w_nm, lockin["dIdL_1f_X"], color="tab:red", lw=0.8)
        axs[1, 1].set_title("Lock-in 1f X / nominal A")
        axs[1, 1].set_xlabel("Wavelength (nm)")
        axs[1, 1].set_ylabel("Reflectance / um")
        axs[1, 1].grid(True, alpha=0.35)
        fig.suptitle(f"Lock-in analysis: {case_name}")
        lockin_path = save_dir / f"lockin_analysis_{file_tag}.png"
        fig.savefig(lockin_path, dpi=180)
        plt.close(fig)

        spectra = np.asarray(data["spectra"])
        fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        axs[0, 0].plot(w_nm, np.mean(spectra, axis=0), lw=0.8)
        axs[0, 0].set_title("Mean measured spectrum")
        axs[0, 1].plot(data["t_axis"] * 1000.0, data["L_t"])
        axs[0, 1].set_title("Air gap modulation")
        center = spectra.shape[1] // 2
        axs[1, 0].plot(data["t_axis"] * 1000.0, spectra[:, center])
        axs[1, 0].set_title(f"Time trace at {w_nm[center]:.2f} nm")
        residual = spectra - data["spectra_physical"]
        axs[1, 1].hist(residual.ravel()[::100], bins=80)
        axs[1, 1].set_title("Measurement perturbation")
        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"{case_name}: angle={data['actual_angle_deg']:.5g} deg, "
            f"delta_lambda={data['noise_realization']['wavelength_offset_nm']:.5g} nm"
        )
        dashboard_path = save_dir / f"dynamic_analysis_dashboard_{file_tag}.png"
        fig.savefig(dashboard_path, dpi=180)
        plt.close(fig)

        print(f"[{case_name}] saved {npz_path}")
        return {
            "case": case_name,
            "seed": data["random_seed"],
            "npz": str(npz_path),
            "metadata": str(metadata_path),
            "lockin_plot": str(lockin_path),
            "dashboard": str(dashboard_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V5 StackRT dynamic dataset generator with bounded angle and one-factor ablations."
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="all, combined, a factor name, or a comma-separated list such as clean,angle_low.",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--backend", choices=["batch", "api"], default="batch")
    parser.add_argument("--describe-profiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.describe_profiles:
        print(json.dumps(
            {
                "levels": NOISE_LEVELS,
                "factors": NOISE_FACTORS,
                "angle_limit_deg": ANGLE_ABS_MAX_DEG,
                "cases": all_case_names(),
            },
            indent=2,
        ))
        return

    cases = select_cases(args.cases)
    seed_sequences = np.random.SeedSequence(args.seed).spawn(len(cases))
    output_dir = Path(__file__).resolve().parents[2] / "04_results_and_datasets" / "dynamic_stackrt_lockin_v5"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "_stackrt_batch_bridge"

    artifacts = []
    failures = []
    for case_name, seed_sequence in zip(cases, seed_sequences):
        case_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        print(f"\n=== Generating {case_name} ===")
        try:
            simulator = DynamicSimulator(
                CONFIG, case_name, case_seed, backend=args.backend, batch_dir=batch_dir
            )
            data = simulator.run_dynamic_sequence()
            artifacts.append(DynamicAnalyzer.save_and_plot(data, output_dir))
        except Exception as exc:
            failures.append({"case": case_name, "seed": case_seed, "error": str(exc)})
            print(f"[{case_name}] ERROR: {exc}")
            if "Session not found" in str(exc):
                break

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "timestamp": TIMESTAMP,
        "root_seed": args.seed,
        "backend": args.backend,
        "requested_cases": cases,
        "angle_limit_deg": ANGLE_ABS_MAX_DEG,
        "artifacts": artifacts,
        "failures": failures,
    }
    manifest_path = output_dir / f"simulation_manifest_{TIMESTAMP}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MANIFEST={manifest_path}")
    if failures:
        raise RuntimeError(f"Simulation failures: {failures}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
