"""EQ-99X strict CPU/GPU spectrometer backends for V15.

The frozen V10 TMM, QE, throughput, photon conversion, Gaussian ILS, and
reference normalization are preserved.  Only the nominal source envelope is
replaced by the digitized EQ-99X curve embedded in each NPZ config.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v10_gpu.backend.v10_source import load_v10_module
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend, NumpyStrictSpectrometerBackend
from v10_gpu.spectrometer.contract import C0_M_S, PLANCK_CONSTANT_J_S

EXPECTED_MODEL = "eq99x_digitized_peak_normalized"


def eq99x_source_numpy(wavelengths_nm: np.ndarray, generator_config: dict[str, Any]) -> np.ndarray:
    if generator_config.get("SOURCE_MODEL") != EXPECTED_MODEL:
        raise ValueError(f"V15 requires {EXPECTED_MODEL}")
    axis = np.asarray(generator_config["EQ99X_WAVELENGTH_NM"], dtype=np.float64)
    shape = np.asarray(generator_config["EQ99X_SOURCE_SHAPE"], dtype=np.float64)
    target = np.asarray(wavelengths_nm, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or shape.shape != axis.shape:
        raise ValueError("invalid embedded EQ-99X source arrays")
    if np.any(~np.isfinite(axis)) or np.any(~np.isfinite(shape)) or np.any(shape <= 0.0):
        raise ValueError("EQ-99X source arrays must be finite and positive")
    if np.any(np.diff(axis) <= 0.0) or axis[0] > 200.0 or axis[-1] < 800.0:
        raise ValueError("EQ-99X source arrays must cover 200-800 nm")
    result = np.interp(target, axis, shape, left=float(shape[0]), right=float(shape[-1]))
    return result / float(np.max(result))


class Eq99xNumpyStrictSpectrometerBackend(NumpyStrictSpectrometerBackend):
    backend_name = "cpu_numpy_eq99x_spectrometer"

    def __init__(self, reported_wavelengths_um, generator_config, internal_margin_nm=None):
        # The formal module is loaded under a private analysis name.  V15 runs
        # in its own process, so replacing only this module's source function
        # cannot modify main_v10.py or main_v12.py on disk.
        module = load_v10_module()
        module.source_power_envelope = eq99x_source_numpy
        super().__init__(reported_wavelengths_um, generator_config, internal_margin_nm)
        self.eq99x_source_csv_sha256 = str(generator_config["EQ99X_SOURCE_CSV_SHA256"])


class Eq99xCupyStrictSpectrometerBackend(CupyStrictSpectrometerBackend):
    backend_name = "cuda_cupy_eq99x_spectrometer"

    def __init__(self, reported_wavelengths_um, generator_config, internal_margin_nm=None, device_id=0):
        # The parent builds the unchanged TMM/ILS/QE contract.  Replacing the
        # resident photon-weight vector is sufficient and keeps all callbacks
        # on the GPU after one initialization upload.
        super().__init__(reported_wavelengths_um, generator_config, internal_margin_nm, device_id)
        source_host = eq99x_source_numpy(self.contract.internal_nm, generator_config)
        cp = self.cp
        c = self.contract
        with self.device:
            internal_nm = cp.asarray(c.internal_nm, dtype=cp.float64)
            source = cp.asarray(source_host, dtype=cp.float64)
            half_span_nm = max(
                c.qe_center_nm - c.wavelength_start_nm,
                c.wavelength_stop_nm - c.qe_center_nm,
            )
            normalized = cp.clip((internal_nm - c.qe_center_nm) / half_span_nm, -1.0, 1.0)
            qe = c.qe_edge + (c.qe_peak - c.qe_edge) * (1.0 - normalized**2)
            photon_energy_j = PLANCK_CONSTANT_J_S * C0_M_S / (internal_nm * 1.0e-9)
            self.electron_weight_device = source * c.optical_throughput * qe / photon_energy_j
            reference_filtered = self._convolve_device(self.electron_weight_device)
            self.reference_sampled_device = self._sample_device(reference_filtered)
            if bool(cp.any(self.reference_sampled_device <= 0.0).item()):
                raise ValueError("EQ-99X GPU reference response must remain positive")
            self.synchronize()
        self.eq99x_source_csv_sha256 = str(generator_config["EQ99X_SOURCE_CSV_SHA256"])
