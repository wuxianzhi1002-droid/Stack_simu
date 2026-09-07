"""Frozen NumPy adapter around the unchanged formal V10 strict forward."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import ForwardBackend
from .v10_source import load_v10_module, source_sha256


class NumpyStrictTMMBackend(ForwardBackend):
    """The Phase 0 CPU reference backend.

    It calls tmm_joint_inversion_v10.tmm_reflectance directly rather than
    maintaining a second CPU physics implementation.
    """

    backend_name = "cpu_numpy"

    def __init__(self, wavelengths_um: np.ndarray):
        super().__init__(wavelengths_um)
        self._v10 = load_v10_module()
        self.n_matrix = np.vstack(
            [
                self._v10.material_n(name, self.wavelengths_um)
                for name in self._v10.LAYER_NAMES
            ]
        ).astype(np.complex128, copy=False)
        self.k0_m_inv = (
            2.0 * np.pi / (self.wavelengths_um * 1.0e-6)
        ).astype(np.float64, copy=False)
        self.source_sha256 = source_sha256()

    def parameters_to_device(self, params_batch: Any) -> np.ndarray:
        return self.validate_params_batch(params_batch)

    def predict_batch_device(self, params_device: Any) -> np.ndarray:
        params = self.validate_params_batch(params_device)
        output = np.empty(
            (params.shape[0], self.wavelengths_um.size), dtype=np.float64
        )
        for index, values in enumerate(params):
            output[index] = self._v10.tmm_reflectance(
                self.wavelengths_um,
                values,
                n_matrix=self.n_matrix,
                k0_m_inv=self.k0_m_inv,
            )
        return output

    def to_host(self, values_device: Any) -> np.ndarray:
        return np.asarray(values_device, dtype=np.float64)

    def memory_stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "available": True,
            "resident_bytes": int(
                self.wavelengths_um.nbytes
                + self.n_matrix.nbytes
                + self.k0_m_inv.nbytes
            ),
            "formal_v10_sha256": self.source_sha256,
        }
