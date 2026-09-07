"""CuPy V10 strict TMM plus nominal spectrometer response."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend.base import BackendUnavailableError
from ..backend.cupy_backend import CupyStrictTMMBackend, _import_cupy
from ..backend.v10_source import source_sha256
from .base import SpectrometerResponseBackend
from .contract import C0_M_S, PLANCK_CONSTANT_J_S, build_contract


class CupyStrictSpectrometerBackend(SpectrometerResponseBackend):
    """Phase 2 GPU backend; no Jacobian, optimizer, JAX or global search."""

    backend_name = "cuda_cupy_spectrometer"

    @classmethod
    def availability(cls) -> dict[str, Any]:
        status = CupyStrictTMMBackend.availability()
        if not status.get("available"):
            return status
        try:
            from cupyx.scipy.ndimage import gaussian_filter1d  # noqa: F401
        except Exception as exc:
            return {
                "available": False,
                "reason": f"cupyx Gaussian ILS is unavailable: {type(exc).__name__}: {exc}",
            }
        return status

    def __init__(
        self,
        reported_wavelengths_um: np.ndarray,
        generator_config: dict[str, Any],
        internal_margin_nm: float | None = None,
        device_id: int = 0,
    ):
        super().__init__(reported_wavelengths_um)
        self.cp = _import_cupy()
        try:
            from cupyx.scipy.ndimage import gaussian_filter1d
        except Exception as exc:
            raise BackendUnavailableError(
                f"cupyx Gaussian ILS import failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.gaussian_filter1d = gaussian_filter1d
        self.contract = build_contract(
            self.reported_wavelengths_um,
            generator_config,
            internal_margin_nm,
        )
        self.internal_wavelengths_um = self.contract.internal_um
        self.tmm_backend = CupyStrictTMMBackend(
            self.internal_wavelengths_um, device_id=device_id
        )
        self.device = self.tmm_backend.device
        self.device_id = self.tmm_backend.device_id
        self.formal_v10_sha256 = source_sha256()

        cp = self.cp
        c = self.contract
        with self.device:
            internal_nm = cp.asarray(c.internal_nm, dtype=cp.float64)
            source = c.source_floor_rel + (1.0 - c.source_floor_rel) * cp.exp(
                -0.5 * ((internal_nm - c.source_center_nm) / c.source_sigma_nm) ** 2
            )
            source = source / cp.max(source)
            half_span_nm = max(
                c.qe_center_nm - c.wavelength_start_nm,
                c.wavelength_stop_nm - c.qe_center_nm,
            )
            normalized = cp.clip(
                (internal_nm - c.qe_center_nm) / half_span_nm, -1.0, 1.0
            )
            qe = c.qe_edge + (c.qe_peak - c.qe_edge) * (1.0 - normalized**2)
            wavelength_m = internal_nm * 1.0e-9
            photon_energy_j = PLANCK_CONSTANT_J_S * C0_M_S / wavelength_m
            self.electron_weight_device = (
                source * c.optical_throughput * qe / photon_energy_j
            )
            self.interpolation_left_device = cp.asarray(
                c.interpolation_left, dtype=cp.int64
            )
            self.interpolation_alpha_device = cp.asarray(
                c.interpolation_alpha, dtype=cp.float64
            )
            reference_filtered = self._convolve_device(self.electron_weight_device)
            self.reference_sampled_device = self._sample_device(reference_filtered)
            if bool(cp.any(self.reference_sampled_device <= 0.0).item()):
                raise ValueError("Nominal GPU reference response must remain positive.")
            self.synchronize()

    def _convolve_device(self, values_device: Any):
        axis = -1
        return self.gaussian_filter1d(
            values_device,
            sigma=self.contract.ils_sigma_samples,
            axis=axis,
            mode="nearest",
            truncate=self.contract.ils_truncate_sigma,
        )

    def _sample_device(self, values_device: Any):
        left = self.interpolation_left_device
        alpha = self.interpolation_alpha_device
        left_values = values_device[..., left]
        right_values = values_device[..., left + 1]
        return left_values + alpha * (right_values - left_values)

    def parameters_to_device(self, params_batch: Any):
        return self.tmm_backend.parameters_to_device(params_batch)

    def predict_batch_device(self, params_device: Any):
        cp = self.cp
        with self.device:
            reflectance = self.tmm_backend.predict_batch_device(params_device)
            sample_internal = self.electron_weight_device[None, :] * reflectance
            sample_filtered = self._convolve_device(sample_internal)
            sample_sampled = self._sample_device(sample_filtered)
            return (
                self.contract.exposure_ratio
                * sample_sampled
                / cp.maximum(self.reference_sampled_device[None, :], 1.0e-30)
            )

    def to_host(self, values_device: Any) -> np.ndarray:
        with self.device:
            return self.cp.asnumpy(values_device).astype(np.float64, copy=False)

    def synchronize(self) -> None:
        self.tmm_backend.synchronize()

    def memory_stats(self) -> dict[str, Any]:
        stats = self.tmm_backend.memory_stats()
        stats["backend"] = self.backend_name
        stats["internal_points"] = int(self.contract.internal_nm.size)
        stats["reported_points"] = int(self.contract.reported_nm.size)
        return stats
