"""Frozen V10 nominal spectrometer-response contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

C0_M_S = 299_792_458.0
PLANCK_CONSTANT_J_S = 6.626_070_15e-34


@dataclass(frozen=True)
class SpectrometerContract:
    reported_um: np.ndarray
    reported_nm: np.ndarray
    internal_um: np.ndarray
    internal_nm: np.ndarray
    interpolation_left: np.ndarray
    interpolation_alpha: np.ndarray
    internal_step_nm: float
    internal_margin_nm: float
    ils_fwhm_nm: float
    ils_sigma_nm: float
    ils_sigma_samples: float
    ils_truncate_sigma: float
    source_center_nm: float
    source_sigma_nm: float
    source_floor_rel: float
    wavelength_start_nm: float
    wavelength_stop_nm: float
    qe_center_nm: float
    qe_peak: float
    qe_edge: float
    optical_throughput: float
    exposure_ratio: float


def build_contract(
    reported_wavelengths_um: np.ndarray,
    generator_config: dict[str, Any],
    internal_margin_nm: float | None = None,
) -> SpectrometerContract:
    reported_um = np.asarray(reported_wavelengths_um, dtype=np.float64)
    if reported_um.ndim != 1 or reported_um.size == 0:
        raise ValueError("reported_wavelengths_um must be a non-empty 1D array.")
    if not np.all(np.isfinite(reported_um)) or np.any(np.diff(reported_um) <= 0.0):
        raise ValueError("reported_wavelengths_um must be finite and strictly increasing.")
    reported_nm = reported_um * 1000.0
    settings = generator_config["SPECTROMETER"]
    if not bool(settings["ILS_ENABLED"]):
        raise ValueError("V10 Phase 2 requires ILS_ENABLED=True.")
    if settings["ILS_SHAPE"] != "gaussian":
        raise ValueError("Only the frozen V10 Gaussian ILS is supported.")
    if settings["QE_MODEL"] != "quadratic":
        raise ValueError("Only the frozen V10 quadratic QE model is supported.")

    step_nm = float(generator_config["INTERNAL_WAVELENGTH_STEP_NM"])
    fwhm_nm = float(settings["ILS_FWHM_NM"])
    truncate = float(settings["ILS_TRUNCATE_SIGMA"])
    sigma_nm = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if step_nm <= 0.0 or step_nm > fwhm_nm / 5.0:
        raise ValueError("Internal grid is invalid for the configured V10 ILS.")
    minimum_margin_nm = truncate * sigma_nm
    margin_nm = (
        minimum_margin_nm
        if internal_margin_nm is None
        else max(minimum_margin_nm, float(internal_margin_nm))
    )
    start_nm = float(reported_nm[0]) - margin_nm
    stop_nm = float(reported_nm[-1]) + margin_nm
    count = int(np.ceil((stop_nm - start_nm) / step_nm)) + 1
    internal_nm = start_nm + np.arange(count, dtype=np.float64) * step_nm
    internal_um = internal_nm / 1000.0

    right = np.searchsorted(internal_nm, reported_nm, side="left")
    right = np.clip(right, 1, internal_nm.size - 1)
    left = right - 1
    alpha = (reported_nm - internal_nm[left]) / (
        internal_nm[right] - internal_nm[left]
    )

    reference_exposure_s = float(settings["REFERENCE_EXPOSURE_S"])
    sample_exposure_s = float(settings["SAMPLE_EXPOSURE_S"])
    if reference_exposure_s <= 0.0 or sample_exposure_s <= 0.0:
        raise ValueError("Sample and reference exposure times must be positive.")

    return SpectrometerContract(
        reported_um=reported_um,
        reported_nm=reported_nm,
        internal_um=internal_um,
        internal_nm=internal_nm,
        interpolation_left=left.astype(np.int64, copy=False),
        interpolation_alpha=alpha.astype(np.float64, copy=False),
        internal_step_nm=step_nm,
        internal_margin_nm=margin_nm,
        ils_fwhm_nm=fwhm_nm,
        ils_sigma_nm=sigma_nm,
        ils_sigma_samples=sigma_nm / step_nm,
        ils_truncate_sigma=truncate,
        source_center_nm=float(generator_config["SOURCE_REFERENCE_CENTER_NM"]),
        source_sigma_nm=float(generator_config["SOURCE_ENVELOPE_SIGMA_NM"]),
        source_floor_rel=float(generator_config["SOURCE_ENVELOPE_FLOOR_REL"]),
        wavelength_start_nm=float(generator_config["WAVELENGTH_START_UM"]) * 1000.0,
        wavelength_stop_nm=float(generator_config["WAVELENGTH_STOP_UM"]) * 1000.0,
        qe_center_nm=float(settings["QE_CENTER_NM"]),
        qe_peak=float(settings["QE_PEAK"]),
        qe_edge=float(settings["QE_EDGE"]),
        optical_throughput=float(settings["OPTICAL_THROUGHPUT"]),
        exposure_ratio=sample_exposure_s / reference_exposure_s,
    )
