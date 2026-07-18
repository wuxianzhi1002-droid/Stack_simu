from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from tmm_stackrt_matched import StackRTMatchedTMM, stackrt_arrays

LUMERICAL_API_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
LUMERICAL_BIN_PATH = Path(r"D:\Program Files\Lumerical\v241\bin")
FDTD_EXECUTABLE = LUMERICAL_BIN_PATH / "fdtd-solutions.exe"
CLI_RUNTIME_DIR = Path(r"C:\Users\wuxianzhi\AppData\Local\Temp\.ansys\stackrt_cli_runtime")


def import_lumapi():
    if LUMERICAL_API_PATH.exists() and str(LUMERICAL_API_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_API_PATH))
    if LUMERICAL_BIN_PATH.exists():
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(LUMERICAL_BIN_PATH)
    try:
        import lumapi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "lumapi is unavailable. Run with the Lumerical Python environment or use "
            "--backend tmm-smoke for a clearly labelled pipeline test."
        ) from exc
    return lumapi


class StaticStackRTGenerator:
    def __init__(self, wavelengths_um: np.ndarray):
        self.wavelengths_um = np.asarray(wavelengths_um, dtype=float)
        self._fdtd = None

    def __enter__(self) -> "StaticStackRTGenerator":
        lumapi = import_lumapi()
        self._fdtd = lumapi.FDTD(hide=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fdtd is not None:
            self._fdtd.close()
            self._fdtd = None

    def spectrum(self, values: np.ndarray) -> np.ndarray:
        if self._fdtd is None:
            raise RuntimeError("StaticStackRTGenerator must be used as a context manager.")
        n_matrix, thickness_m, frequency = stackrt_arrays(self.wavelengths_um, values)
        result = self._fdtd.stackrt(n_matrix, thickness_m, frequency, 0.0)
        return np.asarray(result["Rp"], dtype=float).reshape(-1)


class TMMStaticSmokeGenerator:
    """Pipeline-only fallback; outputs must never be labelled as StackRT."""

    def __init__(self, wavelengths_um: np.ndarray):
        self.model = StackRTMatchedTMM(wavelengths_um)

    def __enter__(self) -> "TMMStaticSmokeGenerator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def spectrum(self, values: np.ndarray) -> np.ndarray:
        return self.model.reflectance(values)


class StaticStackRTCLIGenerator:
    """Batch StackRT through the supported FDTD command-line LSF interface."""

    def __init__(self, runtime_dir: Path = CLI_RUNTIME_DIR, timeout_s: float = 180.0):
        self.runtime_dir = Path(runtime_dir)
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _lsf_path(path: Path) -> str:
        return path.resolve().as_posix()

    def spectra(self, wavelengths_um_by_sample: np.ndarray, parameter_values: np.ndarray) -> np.ndarray:
        axes = np.asarray(wavelengths_um_by_sample, dtype=np.float64)
        values = np.asarray(parameter_values, dtype=np.float64)
        if axes.ndim == 1:
            axes = axes[None, :]
        if values.ndim == 1:
            values = values[None, :]
        if axes.ndim != 2 or values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("Expected wavelength axes (N, N_lambda) and parameter values (N, 5).")
        if axes.shape[0] != values.shape[0] or axes.shape[1] < 2:
            raise ValueError("StackRT batch axes and parameter rows are not aligned.")
        if np.any(axes <= 0.0) or not np.all(np.diff(axes, axis=1) > 0.0):
            raise ValueError("Every wavelength axis must be positive and strictly increasing.")
        if not FDTD_EXECUTABLE.exists():
            raise FileNotFoundError(f"FDTD executable not found: {FDTD_EXECUTABLE}")

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.runtime_dir / "stackrt_batch_input.mat"
        output_path = self.runtime_dir / "stackrt_batch_output.mat"
        script_path = self.runtime_dir / "stackrt_batch_run.lsf"
        if output_path.exists():
            output_path.unlink()
        with h5py.File(input_path, "w") as handle:
            for name, array in (
                ("wavelengths_um_by_sample", axes),
                ("parameter_values", values),
            ):
                dataset = handle.create_dataset(name, data=array.T)
                dataset.attrs["MATLAB_class"] = np.bytes_("double")
        script = f'''clear;
matlabload("{self._lsf_path(input_path)}");
sample_count = size(parameter_values, 1);
wavelength_count = size(wavelengths_um_by_sample, 2);
spectra = matrix(sample_count, wavelength_count);
for (sample_index=1:sample_count) {{
    w_um = pinch(wavelengths_um_by_sample(sample_index, 1:wavelength_count));
    frequency = 3e8 / (w_um * 1e-6);
    n_matrix = matrix(7, wavelength_count);
    n_matrix(1, 1:wavelength_count) = 5.8284;
    n_matrix(2, 1:wavelength_count) = 1.0;
    n_matrix(3, 1:wavelength_count) = 1.41;
    n_matrix(4, 1:wavelength_count) = 1.50 + 0.05i;
    n_matrix(5, 1:wavelength_count) = 1.55 + 0.005 / (w_um^2);
    n_matrix(6, 1:wavelength_count) = 2.4 + 0.02 / (w_um^2);
    n_matrix(7, 1:wavelength_count) = 1.1 + 2.5i;
    thickness = [0;
                 parameter_values(sample_index, 1) * 1e-6;
                 parameter_values(sample_index, 2) * 1e-9;
                 parameter_values(sample_index, 3) * 1e-9;
                 parameter_values(sample_index, 4) * 1e-9;
                 parameter_values(sample_index, 5) * 1e-9;
                 0];
    result = stackrt(n_matrix, thickness, frequency, 0);
    spectra(sample_index, 1:wavelength_count) = transpose(result.Rp);
}}
matlabsave("{self._lsf_path(output_path.with_suffix(''))}", spectra);
exit;
'''
        script_path.write_text(script, encoding="ascii")
        command = [
            str(FDTD_EXECUTABLE),
            "-hide",
            "-trust-script",
            "-run",
            str(script_path),
            "-exit",
        ]
        completed = subprocess.run(
            command,
            cwd=self.runtime_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"FDTD CLI failed with exit code {completed.returncode}. "
                f"stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-2000:]!r}"
            )
        deadline = time.monotonic() + 10.0
        while not output_path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not output_path.exists():
            raise RuntimeError(
                "FDTD CLI exited without creating the StackRT MAT output. "
                f"stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-2000:]!r}"
            )
        with h5py.File(output_path, "r") as handle:
            if "spectra" not in handle:
                raise RuntimeError(f"StackRT output has no 'spectra' variable: {list(handle.keys())}")
            spectra = np.asarray(handle["spectra"], dtype=float)
        if spectra.shape == (axes.shape[1], axes.shape[0]):
            spectra = spectra.T
        if spectra.shape != axes.shape:
            raise RuntimeError(f"Unexpected StackRT output shape {spectra.shape}; expected {axes.shape}.")
        if not np.all(np.isfinite(spectra)):
            raise RuntimeError("StackRT output contains non-finite values.")
        return spectra
