from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from tmm_stackrt_matched import StackRTMatchedTMM, stackrt_arrays

LUMERICAL_API_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
LUMERICAL_BIN_PATH = Path(r"D:\Program Files\Lumerical\v241\bin")


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
