"""Common Phase 1 strict-TMM forward interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a requested compute backend cannot be initialized."""


class ForwardBackend(ABC):
    """Backend-neutral strict-TMM interface.

    Phase 1 ends at strict reflectance R(B, wavelength). ILS, Jacobian,
    optimization and global search deliberately remain outside this contract.
    """

    backend_name: str

    def __init__(self, wavelengths_um: np.ndarray):
        wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths_um must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
            raise ValueError("wavelengths_um must contain finite positive values.")
        if np.any(np.diff(wavelengths) <= 0.0):
            raise ValueError("wavelengths_um must be strictly increasing.")
        self.wavelengths_um = wavelengths

    @staticmethod
    def validate_params_batch(params_batch: Any) -> np.ndarray:
        params = np.asarray(params_batch, dtype=np.float64)
        if params.ndim == 1:
            params = params[None, :]
        if params.ndim != 2 or params.shape[1] != 6:
            raise ValueError("params_batch must have shape (B, 6) or (6,).")
        if params.shape[0] == 0 or not np.all(np.isfinite(params)):
            raise ValueError("params_batch must be non-empty and finite.")
        return params

    @abstractmethod
    def parameters_to_device(self, params_batch: Any) -> Any:
        """Validate and transfer parameters to the backend-native device."""

    @abstractmethod
    def predict_batch_device(self, params_device: Any) -> Any:
        """Return backend-native R with shape (B, wavelength)."""

    @abstractmethod
    def to_host(self, values_device: Any) -> np.ndarray:
        """Transfer backend-native values to a NumPy float64 array."""

    def predict_batch(self, params_batch: Any) -> np.ndarray:
        params_device = self.parameters_to_device(params_batch)
        result_device = self.predict_batch_device(params_device)
        self.synchronize()
        return self.to_host(result_device)

    def predict(self, params: Any) -> np.ndarray:
        return self.predict_batch(params)[0]

    def residual(self, params: Any, observed_reflectance: Any) -> np.ndarray:
        observed = np.asarray(observed_reflectance, dtype=np.float64)
        predicted = self.predict(params)
        if observed.shape != predicted.shape:
            raise ValueError("observed_reflectance must match the wavelength axis.")
        return predicted - observed

    def residual_and_jacobian(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Jacobian support starts after Phase 1; no numerical or autodiff "
            "Jacobian is implemented in this package."
        )

    def synchronize(self) -> None:
        """Synchronize asynchronous work; CPU backends are a no-op."""

    def memory_stats(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "available": True}
