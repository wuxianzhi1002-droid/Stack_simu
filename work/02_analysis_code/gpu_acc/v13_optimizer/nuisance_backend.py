"""Strict 220-580 nm spectrometer response with three instrument nuisances.

The structural TMM kernel is unchanged.  Nuisance parameters alter only the
experiment-side signal chain:

* axis_offset_nm and axis_scale_ppm move the hidden pixel centers according to
  the generator convention ``reported - true``;
* source_center_drift_nm shifts the sample-source envelope while the reference
  source remains nominal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from v10_gpu.backend.numpy_backend import NumpyStrictTMMBackend
from v10_gpu.spectrometer.contract import (
    C0_M_S,
    PLANCK_CONSTANT_J_S,
    build_contract,
)
from v10_gpu.spectrometer.cupy_backend import CupyStrictSpectrometerBackend


NUISANCE_ORDER = ("axis_offset_nm", "axis_scale_ppm", "source_center_drift_nm")


def _validate_nuisance(values: Any, batch_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (batch_size, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"nuisance values must have shape ({batch_size}, 3)")
    return array


class CupyNuisanceSpectrometerBackend(CupyStrictSpectrometerBackend):
    """CuPy strict response with candidate-specific axis/source nuisances."""

    backend_name = "cuda_cupy_v13_nuisance_spectrometer"

    def __init__(self, reported_wavelengths_um, generator_config, internal_margin_nm=None, device_id=0):
        super().__init__(reported_wavelengths_um, generator_config, internal_margin_nm, device_id)
        cp = self.cp
        c = self.contract
        with self.device:
            internal_nm = cp.asarray(c.internal_nm, dtype=cp.float64)
            reported_nm = cp.asarray(c.reported_nm, dtype=cp.float64)
            half_span_nm = max(
                c.qe_center_nm - c.wavelength_start_nm,
                c.wavelength_stop_nm - c.qe_center_nm,
            )
            normalized = cp.clip((internal_nm - c.qe_center_nm) / half_span_nm, -1.0, 1.0)
            qe = c.qe_edge + (c.qe_peak - c.qe_edge) * (1.0 - normalized**2)
            photon_energy_j = PLANCK_CONSTANT_J_S * C0_M_S / (internal_nm * 1.0e-9)
            detector_weight = c.optical_throughput * qe / photon_energy_j
            nominal_source = c.source_floor_rel + (1.0 - c.source_floor_rel) * cp.exp(
                -0.5 * ((internal_nm - c.source_center_nm) / c.source_sigma_nm) ** 2
            )
            nominal_source /= cp.max(nominal_source)
            self.internal_nm_device = internal_nm
            self.reported_nm_device = reported_nm
            self.detector_weight_device = detector_weight
            self.reference_filtered_device = self._convolve_device(nominal_source * detector_weight)
            self.axis_center_nm = float(0.5 * (c.reported_nm[0] + c.reported_nm[-1]))
            self.internal_start_nm = float(c.internal_nm[0])
            self.internal_step_nm = float(c.internal_step_nm)
            self.synchronize()

    def _sample_candidate_device(self, values_device, true_centers_device):
        cp = self.cp
        position = (true_centers_device - self.internal_start_nm) / self.internal_step_nm
        left = cp.floor(position).astype(cp.int64)
        left = cp.clip(left, 0, self.internal_nm_device.size - 2)
        alpha = position - left.astype(cp.float64)
        lo = cp.take_along_axis(values_device, left, axis=1)
        hi = cp.take_along_axis(values_device, left + 1, axis=1)
        return lo + alpha * (hi - lo)

    def predict_batch_nuisance_device(self, params_device, nuisance_device):
        cp = self.cp
        with self.device:
            # This entry point is device-native: global and local objectives
            # already hold the population on CuPy.  Calling
            # ``parameters_to_device`` here would route the CuPy array through
            # NumPy validation and trigger a forbidden implicit D2H transfer.
            params = cp.asarray(params_device, dtype=cp.float64)
            if params.ndim == 1:
                params = params[None, :]
            if params.ndim != 2 or params.shape[1] != 6 or params.shape[0] == 0:
                raise ValueError("device parameters must have shape (B, 6) or (6,)")
            if not bool(cp.all(cp.isfinite(params)).item()):
                raise ValueError("device parameters contain NaN/Inf")
            nuisance = cp.asarray(nuisance_device, dtype=cp.float64)
            if nuisance.shape != (params.shape[0], 3):
                raise ValueError("device nuisance batch shape mismatch")
            if not bool(cp.all(cp.isfinite(nuisance)).item()):
                raise ValueError("device nuisance values contain NaN/Inf")
            reflectance = self.tmm_backend.predict_batch_device(params)
            drift = nuisance[:, 2:3]
            source = self.contract.source_floor_rel + (1.0 - self.contract.source_floor_rel) * cp.exp(
                -0.5
                * ((self.internal_nm_device[None, :] - (self.contract.source_center_nm + drift))
                   / self.contract.source_sigma_nm) ** 2
            )
            source /= cp.max(source, axis=1, keepdims=True)
            sample_filtered = self._convolve_device(
                source * self.detector_weight_device[None, :] * reflectance
            )
            axis_error = nuisance[:, 0:1] + nuisance[:, 1:2] * 1.0e-6 * (
                self.reported_nm_device[None, :] - self.axis_center_nm
            )
            true_centers = self.reported_nm_device[None, :] - axis_error
            sample = self._sample_candidate_device(sample_filtered, true_centers)
            reference_matrix = cp.broadcast_to(
                self.reference_filtered_device[None, :], sample_filtered.shape
            )
            reference = self._sample_candidate_device(reference_matrix, true_centers)
            return self.contract.exposure_ratio * sample / cp.maximum(reference, 1.0e-30)

    def predict_batch_nuisance(self, params_batch, nuisance_batch) -> np.ndarray:
        params = self.validate_params_batch(params_batch)
        nuisance = _validate_nuisance(nuisance_batch, len(params))
        with self.device:
            result = self.predict_batch_nuisance_device(
                self.cp.asarray(params, dtype=self.cp.float64),
                self.cp.asarray(nuisance, dtype=self.cp.float64),
            )
            self.synchronize()
            return self.cp.asnumpy(result).astype(np.float64, copy=False)


class NumpyNuisanceSpectrometerBackend:
    """Independent NumPy/SciPy oracle for selected-point GPU closure."""

    backend_name = "cpu_numpy_v13_nuisance_spectrometer"

    def __init__(self, reported_wavelengths_um, generator_config, internal_margin_nm=None):
        self.contract = build_contract(reported_wavelengths_um, generator_config, internal_margin_nm)
        self.tmm_backend = NumpyStrictTMMBackend(self.contract.internal_um)
        c = self.contract
        internal_nm = c.internal_nm
        half_span_nm = max(
            c.qe_center_nm - c.wavelength_start_nm,
            c.wavelength_stop_nm - c.qe_center_nm,
        )
        normalized = np.clip((internal_nm - c.qe_center_nm) / half_span_nm, -1.0, 1.0)
        qe = c.qe_edge + (c.qe_peak - c.qe_edge) * (1.0 - normalized**2)
        photon_energy_j = PLANCK_CONSTANT_J_S * C0_M_S / (internal_nm * 1.0e-9)
        self.detector_weight = c.optical_throughput * qe / photon_energy_j
        nominal = c.source_floor_rel + (1.0 - c.source_floor_rel) * np.exp(
            -0.5 * ((internal_nm - c.source_center_nm) / c.source_sigma_nm) ** 2
        )
        nominal /= np.max(nominal)
        self.reference_filtered = gaussian_filter1d(
            nominal * self.detector_weight,
            sigma=c.ils_sigma_samples,
            axis=-1,
            mode="nearest",
            truncate=c.ils_truncate_sigma,
        )
        self.axis_center_nm = float(0.5 * (c.reported_nm[0] + c.reported_nm[-1]))

    def _sample(self, values: np.ndarray, centers: np.ndarray) -> np.ndarray:
        c = self.contract
        position = (centers - c.internal_nm[0]) / c.internal_step_nm
        left = np.clip(np.floor(position).astype(np.int64), 0, c.internal_nm.size - 2)
        alpha = position - left
        lo = np.take_along_axis(values, left, axis=1)
        hi = np.take_along_axis(values, left + 1, axis=1)
        return lo + alpha * (hi - lo)

    def predict_batch_nuisance(self, params_batch, nuisance_batch) -> np.ndarray:
        params = np.asarray(params_batch, dtype=np.float64)
        if params.ndim == 1:
            params = params[None, :]
        nuisance = _validate_nuisance(nuisance_batch, len(params))
        reflectance = self.tmm_backend.predict_batch(params)
        c = self.contract
        source = c.source_floor_rel + (1.0 - c.source_floor_rel) * np.exp(
            -0.5
            * ((c.internal_nm[None, :] - (c.source_center_nm + nuisance[:, 2:3]))
               / c.source_sigma_nm) ** 2
        )
        source /= np.max(source, axis=1, keepdims=True)
        sample_filtered = gaussian_filter1d(
            source * self.detector_weight[None, :] * reflectance,
            sigma=c.ils_sigma_samples,
            axis=-1,
            mode="nearest",
            truncate=c.ils_truncate_sigma,
        )
        axis_error = nuisance[:, 0:1] + nuisance[:, 1:2] * 1.0e-6 * (
            c.reported_nm[None, :] - self.axis_center_nm
        )
        centers = c.reported_nm[None, :] - axis_error
        sample = self._sample(sample_filtered, centers)
        reference = self._sample(
            np.broadcast_to(self.reference_filtered[None, :], sample_filtered.shape), centers
        )
        return c.exposure_ratio * sample / np.maximum(reference, 1.0e-30)
