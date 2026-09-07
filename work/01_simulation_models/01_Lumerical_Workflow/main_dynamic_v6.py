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
GENERATOR_VERSION = "main_dynamic_v6"
ANGLE_ABS_MAX_DEG = 0.1
ERROR_REALIZATION_POLICY = "deterministic_level_center"

CONFIG = {
    "MODEL_TYPE": "PSS_TiO2",
    "WAVELENGTH_START": 0.2,
    "WAVELENGTH_STOP": 0.6,
    "SPECTRAL_RESOLUTION_NM": 0.02,
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

# Deterministic representative error magnitudes inherited from the v5 levels.
# V5 interpreted these values as Gaussian sigmas; V6 applies each active value
# directly with a positive sign so repeated runs produce identical inputs.
NOISE_LEVELS = {
    "low": {
        "n_real_sigma_rel": 5.0e-4,
        "k_sigma_rel": 1.0e-2,
        "angle_sigma_deg": 0.01,
        "wavelength_offset_sigma_nm": 2.0e-4,
        "frame_gain_sigma_rel": 2.0e-4,
        "reflectance_sigma_abs": 2.0e-4,
    },
    "medium": {
        "n_real_sigma_rel": 2.0e-3,
        "k_sigma_rel": 5.0e-2,
        "angle_sigma_deg": 0.05,
        "wavelength_offset_sigma_nm": 1.0e-3,
        "frame_gain_sigma_rel": 1.0e-3,
        "reflectance_sigma_abs": 1.0e-3,
    },
    "high": {
        "n_real_sigma_rel": 5.0e-3,
        "k_sigma_rel": 1.0e-1,
        "angle_sigma_deg": 0.10,
        "wavelength_offset_sigma_nm": 5.0e-3,
        "frame_gain_sigma_rel": 5.0e-3,
        "reflectance_sigma_abs": 5.0e-3,
    },
}
NOISE_FACTORS = ("angle", "wavelength", "material", "detector", "combined")
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
        "detector": ("frame_gain_sigma_rel", "reflectance_sigma_abs"),
    }[factor]
    for key in keys:
        profile[key] = source[key]
    return profile


def realize_noise(level: str, factor: str) -> dict:
    profile = active_profile(level, factor)
    angle_deg = float(profile["angle_sigma_deg"])
    if not 0.0 <= angle_deg <= ANGLE_ABS_MAX_DEG:
        raise ValueError(f"Deterministic angle {angle_deg} exceeds [0, {ANGLE_ABS_MAX_DEG}] deg.")
    n_delta = {
        name: float(profile["n_real_sigma_rel"])
        for name in PERTURBED_MATERIALS
    }
    k_delta = {
        name: float(profile["k_sigma_rel"])
        for name in PERTURBED_MATERIALS
    }
    return {
        "case": "clean" if level == "clean" else f"{factor}_{level}",
        "level": level,
        "factor": factor,
        "profile_config": profile,
        "realization_policy": ERROR_REALIZATION_POLICY,
        "value_sign": "positive",
        "material_n_real_rel_delta": n_delta,
        "material_k_rel_delta": k_delta,
        "angle_input_convention": "nonnegative StackRT first-incident-medium angle in degrees",
        "angle_limit_deg": ANGLE_ABS_MAX_DEG,
        "angle_error_deg": angle_deg,
        "wavelength_offset_nm": float(profile["wavelength_offset_sigma_nm"]),
        "amplitude_rel_error": 0.0,
        "frame_gain_error_rel": float(profile["frame_gain_sigma_rel"]),
        "reflectance_offset_abs": float(profile["reflectance_sigma_abs"]),
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


class StaticSimulator:
    def __init__(
        self,
        config: dict,
        case_name: str,
        backend: str = "batch",
        batch_dir: Path | None = None,
    ):
        self.config = config
        self.case_name = case_name
        self.backend = backend
        self.batch_dir = batch_dir
        level, factor = split_case(case_name)
        self.noise = realize_noise(level, factor)

        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000.0
        count = int(span_nm / config["SPECTRAL_RESOLUTION_NM"]) + 1
        self.wavelengths = np.linspace(config["WAVELENGTH_START"], config["WAVELENGTH_STOP"], count)
        self.physical_wavelengths = self.wavelengths + self.noise["wavelength_offset_nm"] / 1000.0
        self.freqs = 3.0e8 / (self.physical_wavelengths * 1.0e-6)
        self.angle_deg = float(self.noise["angle_error_deg"])
        self.t_axis = np.asarray([0.0], dtype=float)
        self.Nt = 1
        print(
            f"[{case_name}] static StackRT, N_lambda={count}, "
            f"angle={self.angle_deg:.8g} deg, "
            f"wavelength_offset={self.noise['wavelength_offset_nm']:.8g} nm"
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
        print(f"[{self.case_name}] launching native StackRT batch backend")
        completed = subprocess.run(
            [str(executable), "-nw", "-trust-script", "-run", "run_stackrt_static.lsf"],
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
        output_path = stage / "stackrt_static_output.mat"
        output_is_fresh = (
            output_path.exists() and output_path.stat().st_mtime >= started - 1.0
        )
        if completed.returncode != 0 or not output_is_fresh:
            raise RuntimeError(f"Static StackRT batch failed; see {stage / 'stackrt_batch.log'}")

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

    def run_static_spectrum(self) -> dict:
        if self.backend == "api" and lumapi is None:
            raise RuntimeError("lumapi is not available.")

        n_matrix, base_thicknesses, air_idx = self._get_n_matrix()
        if air_idx < 0:
            raise ValueError("Model does not contain an Air layer.")
        static_air_length_m = np.asarray([base_thicknesses[air_idx]], dtype=float)

        if self.backend == "batch":
            physical_spectra = self._run_stackrt_batch(
                n_matrix, base_thicknesses, air_idx, static_air_length_m
            )
        elif self.backend == "api":
            fdtd = lumapi.FDTD(
                hide=True,
                serverArgs={"platform": "offscreen", "use-solve": True},
            )
            started = time.time()
            try:
                result = fdtd.stackrt(n_matrix, base_thicknesses, self.freqs, self.angle_deg)
                physical_spectra = np.asarray(result["Rp"], dtype=float).reshape(1, -1)
            finally:
                fdtd.close()
            print(
                f"[{self.case_name}] StackRT static spectrum complete, "
                f"elapsed={time.time() - started:.1f}s"
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        profile = self.noise["profile_config"]
        frame_gain_error = np.asarray([profile["frame_gain_sigma_rel"]], dtype=float)
        measured = physical_spectra * (1.0 + frame_gain_error[:, None])
        measured += float(profile["reflectance_sigma_abs"])
        clip_mask = (measured < 0.0) | (measured > 1.0)
        clip_fraction = float(np.mean(clip_mask))
        measured = np.clip(measured, 0.0, 1.0)

        layers = self.config["PSS_TIO2_MODEL"]["LAYERS"]
        return {
            "t_axis": self.t_axis,
            "wavelengths": self.wavelengths,
            "physical_wavelengths": self.physical_wavelengths,
            "L_t": static_air_length_m * 1.0e6,
            "spectra": measured,
            "spectra_physical": physical_spectra,
            "frame_gain_error": frame_gain_error,
            "reflectance_clip_fraction": clip_fraction,
            "noise_case": self.case_name,
            "noise_level": self.noise["level"],
            "noise_factor": self.noise["factor"],
            "noise_realization": self.noise,
            "nominal_amplitude_nm": 0.0,
            "actual_amplitude_nm": 0.0,
            "actual_angle_deg": self.angle_deg,
            "modulation_frequency_hz": 0.0,
            "sampling_rate_hz": 0.0,
            "modulation_enabled": False,
            "layer_names": [str(layer[0]) for layer in layers],
            "layer_thickness_um": [float(layer[1]) for layer in layers],
        }


class StaticAnalyzer:
    @staticmethod
    def save_and_plot(data: dict, save_dir: Path) -> dict:
        save_dir.mkdir(parents=True, exist_ok=True)
        case_name = data["noise_case"]
        file_tag = f"{case_name}_{TIMESTAMP}"
        material_names = np.asarray(PERTURBED_MATERIALS, dtype="U16")
        n_deltas = np.asarray([
            data["noise_realization"]["material_n_real_rel_delta"][name]
            for name in material_names
        ])
        k_deltas = np.asarray([
            data["noise_realization"]["material_k_rel_delta"][name]
            for name in material_names
        ])
        noise_json = json.dumps(data["noise_realization"], ensure_ascii=False, sort_keys=True)
        zero_spectrum = np.zeros_like(data["wavelengths"], dtype=float)

        npz_path = save_dir / f"static_spectrum_{file_tag}.npz"
        np.savez_compressed(
            npz_path,
            t_axis=data["t_axis"],
            wavelengths=data["wavelengths"],
            physical_wavelengths=data["physical_wavelengths"],
            L_t=data["L_t"],
            spectra=data["spectra"],
            spectrum_measured=data["spectra"][0],
            spectrum_physical=data["spectra_physical"][0],
            spectra_physical_mean=data["spectra_physical"][0],
            lockin_available=np.asarray(False),
            lockin_1f_X=zero_spectrum,
            lockin_1f_Y=zero_spectrum,
            lockin_1f_R=zero_spectrum,
            lockin_1f_phase=zero_spectrum,
            lockin_2f_X=zero_spectrum,
            lockin_2f_Y=zero_spectrum,
            lockin_2f_R=zero_spectrum,
            lockin_2f_phase=zero_spectrum,
            lockin_3f_X=zero_spectrum,
            lockin_3f_Y=zero_spectrum,
            lockin_3f_R=zero_spectrum,
            lockin_3f_phase=zero_spectrum,
            dIdL_1f=zero_spectrum,
            dIdL_1f_X=zero_spectrum,
            noise_case=np.asarray(case_name),
            noise_level=np.asarray(data["noise_level"]),
            noise_factor=np.asarray(data["noise_factor"]),
            generator_version=np.asarray(GENERATOR_VERSION),
            error_realization_policy=np.asarray(ERROR_REALIZATION_POLICY),
            noise_config_json=np.asarray(noise_json),
            modulation_enabled=np.asarray(False),
            nominal_amplitude_nm=np.asarray(0.0),
            actual_amplitude_nm=np.asarray(0.0),
            modulation_frequency_hz=np.asarray(0.0),
            sampling_rate_hz=np.asarray(0.0),
            nominal_angle_deg=np.asarray(0.0),
            actual_angle_deg=np.asarray(data["actual_angle_deg"]),
            angle_limit_deg=np.asarray(ANGLE_ABS_MAX_DEG),
            angle_input_convention=np.asarray(data["noise_realization"]["angle_input_convention"]),
            wavelength_offset_nm=np.asarray(data["noise_realization"]["wavelength_offset_nm"]),
            frame_gain_error=data["frame_gain_error"],
            reflectance_offset_abs=np.asarray(data["noise_realization"]["reflectance_offset_abs"]),
            reflectance_clip_fraction=np.asarray(data["reflectance_clip_fraction"]),
            perturbed_material_names=material_names,
            material_n_real_rel_delta=n_deltas,
            material_k_rel_delta=k_deltas,
            layer_names=np.asarray(data["layer_names"], dtype="U32"),
            layer_thickness_um=np.asarray(data["layer_thickness_um"], dtype=float),
        )

        metadata_path = save_dir / f"error_realization_{file_tag}.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "generator_version": GENERATOR_VERSION,
                    "noise_case": case_name,
                    "noise_level": data["noise_level"],
                    "noise_factor": data["noise_factor"],
                    "error_realization_policy": ERROR_REALIZATION_POLICY,
                    "noise_realization": data["noise_realization"],
                    "modulation_enabled": False,
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
        measured = data["spectra"][0]
        physical = data["spectra_physical"][0]
        fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        axs[0, 0].plot(w_nm, physical, lw=0.8, label="physical")
        axs[0, 0].plot(w_nm, measured, lw=0.8, alpha=0.8, label="measured")
        axs[0, 0].set_title("Static spectrum")
        axs[0, 0].legend()
        axs[0, 1].plot(w_nm, measured - physical, lw=0.8)
        axs[0, 1].set_title("Measurement perturbation")
        x = np.arange(len(material_names))
        axs[1, 0].bar(x - 0.18, n_deltas * 100.0, width=0.36, label="real(n)")
        axs[1, 0].bar(x + 0.18, k_deltas * 100.0, width=0.36, label="k")
        axs[1, 0].set_xticks(x, material_names)
        axs[1, 0].set_ylabel("Relative error (%)")
        axs[1, 0].set_title("Deterministic material errors")
        axs[1, 0].legend()
        axs[1, 1].axis("off")
        axs[1, 1].text(
            0.02,
            0.98,
            "Static acquisition (no modulation)\n"
            f"policy: {ERROR_REALIZATION_POLICY}\n"
            f"angle: {data['actual_angle_deg']:.8g} deg\n"
            f"wavelength offset: {data['noise_realization']['wavelength_offset_nm']:.8g} nm\n"
            f"frame gain: {data['noise_realization']['frame_gain_error_rel']:.8g}\n"
            f"reflectance offset: {data['noise_realization']['reflectance_offset_abs']:.8g}",
            va="top",
            family="monospace",
        )
        for ax in axs.flat[:3]:
            ax.set_xlabel("Wavelength (nm)" if ax is not axs[1, 0] else "Material")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"V6 deterministic static case: {case_name}")
        dashboard_path = save_dir / f"static_analysis_dashboard_{file_tag}.png"
        fig.savefig(dashboard_path, dpi=180)
        plt.close(fig)

        print(f"[{case_name}] saved {npz_path}")
        return {
            "case": case_name,
            "npz": str(npz_path),
            "metadata": str(metadata_path),
            "dashboard": str(dashboard_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V6 deterministic static StackRT dataset generator with one-factor ablations."
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="all, combined, a factor name, or a comma-separated list such as clean,angle_low.",
    )
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
                "error_realization_policy": ERROR_REALIZATION_POLICY,
                "modulation_enabled": False,
                "cases": all_case_names(),
            },
            indent=2,
        ))
        return

    cases = select_cases(args.cases)
    output_dir = Path(__file__).resolve().parents[2] / "04_results_and_datasets" / "static_stackrt_v6"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "_stackrt_batch_bridge"

    artifacts = []
    failures = []
    for case_name in cases:
        print(f"\n=== Generating {case_name} ===")
        try:
            simulator = StaticSimulator(
                CONFIG, case_name, backend=args.backend, batch_dir=batch_dir
            )
            data = simulator.run_static_spectrum()
            artifacts.append(StaticAnalyzer.save_and_plot(data, output_dir))
        except Exception as exc:
            failures.append({"case": case_name, "error": str(exc)})
            print(f"[{case_name}] ERROR: {exc}")
            if "Session not found" in str(exc):
                break

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "timestamp": TIMESTAMP,
        "backend": args.backend,
        "requested_cases": cases,
        "angle_limit_deg": ANGLE_ABS_MAX_DEG,
        "error_realization_policy": ERROR_REALIZATION_POLICY,
        "modulation_enabled": False,
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
