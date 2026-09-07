"""Frozen NumPy V10 strict spectrometer-response backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend.v10_source import load_v10_module, source_sha256
from .base import SpectrometerResponseBackend


class NumpyStrictSpectrometerBackend(SpectrometerResponseBackend):
    """CPU oracle that delegates each candidate to formal V10 unchanged."""

    backend_name = "cpu_numpy_spectrometer"

    def __init__(
        self,
        reported_wavelengths_um: np.ndarray,
        generator_config: dict[str, Any],
        internal_margin_nm: float | None = None,
    ):
        super().__init__(reported_wavelengths_um)
        v10 = load_v10_module()
        self.model = v10.SpectrometerForwardModel(
            self.reported_wavelengths_um,
            generator_config,
            internal_margin_nm,
        )
        self.internal_wavelengths_um = np.asarray(
            self.model.internal_um, dtype=np.float64
        )
        self.formal_v10_sha256 = source_sha256()

    def parameters_to_device(self, params_batch: Any) -> np.ndarray:
        return self.validate_params_batch(params_batch)

    def predict_batch_device(self, params_device: Any) -> np.ndarray:
        params = self.validate_params_batch(params_device)
        return np.vstack([self.model.predict(row) for row in params]).astype(
            np.float64, copy=False
        )

    def to_host(self, values_device: Any) -> np.ndarray:
        return np.asarray(values_device, dtype=np.float64)
