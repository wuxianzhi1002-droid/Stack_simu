"""Backend-neutral Phase 2 strict spectrometer-response interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..backend.base import ForwardBackend


class SpectrometerResponseBackend(ABC):
    """Strict V10 TMM plus nominal spectrometer response.

    Phase 2 ends at normalized spectra on the fixed reported wavelength axis.
    Jacobians, optimizers, JAX, autodiff and global search are out of scope.
    """

    backend_name: str

    def __init__(self, reported_wavelengths_um: np.ndarray):
        reported = np.asarray(reported_wavelengths_um, dtype=np.float64)
        if reported.ndim != 1 or reported.size == 0:
            raise ValueError("reported_wavelengths_um must be a non-empty 1D array.")
        if not np.all(np.isfinite(reported)) or np.any(reported <= 0.0):
            raise ValueError("reported_wavelengths_um must contain finite positive values.")
        if np.any(np.diff(reported) <= 0.0):
            raise ValueError("reported_wavelengths_um must be strictly increasing.")
        self.reported_wavelengths_um = reported

    @staticmethod
    def validate_params_batch(params_batch: Any) -> np.ndarray:
        return ForwardBackend.validate_params_batch(params_batch)

    @abstractmethod
    def parameters_to_device(self, params_batch: Any) -> Any:
        """Validate and transfer parameters to the backend-native device."""

    @abstractmethod
    def predict_batch_device(self, params_device: Any) -> Any:
        """Return backend-native normalized spectra with shape (B, reported)."""

    @abstractmethod
    def to_host(self, values_device: Any) -> np.ndarray:
        """Transfer backend-native spectra to a NumPy float64 array."""

    def predict_batch(self, params_batch: Any) -> np.ndarray:
        params_device = self.parameters_to_device(params_batch)
        result_device = self.predict_batch_device(params_device)
        self.synchronize()
        result_host = self.to_host(result_device)
        self.synchronize()
        return result_host

    def predict(self, params: Any) -> np.ndarray:
        return self.predict_batch(params)[0]

    def synchronize(self) -> None:
        """Synchronize asynchronous backend work; NumPy is a no-op."""

    def memory_stats(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "available": True}

    def residual_and_jacobian(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Jacobian support is outside Phase 2; no numerical or autodiff "
            "Jacobian is implemented."
        )
