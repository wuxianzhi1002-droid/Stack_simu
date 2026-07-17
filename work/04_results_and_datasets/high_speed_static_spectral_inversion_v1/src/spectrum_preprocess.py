from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessedSpectrum:
    wavelengths_um: np.ndarray
    intensity: np.ndarray
    sigma: float


def preprocess_spectrum(wavelengths_um: np.ndarray, spectrum: np.ndarray, noise_sigma: float) -> PreprocessedSpectrum:
    wavelength = np.asarray(wavelengths_um, dtype=np.float64)
    intensity = np.asarray(spectrum, dtype=np.float64)
    if wavelength.ndim != 1 or intensity.ndim != 1 or wavelength.shape != intensity.shape:
        raise ValueError("wavelengths and spectrum must be aligned one-dimensional arrays.")
    if wavelength.size < 100 or not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("The full ordered spectrum is required; direct stride subsampling is not supported.")
    if not np.all(np.isfinite(intensity)):
        raise ValueError("spectrum contains non-finite values.")
    sigma = max(float(noise_sigma), 1e-6)
    return PreprocessedSpectrum(wavelength.copy(), intensity.copy(), sigma)


def uniform_wavenumber(preprocessed: PreprocessedSpectrum, count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    sigma = 1.0 / preprocessed.wavelengths_um
    size = int(count or sigma.size)
    uniform_sigma = np.linspace(sigma.min(), sigma.max(), size)
    intensity = np.interp(uniform_sigma, sigma[::-1], preprocessed.intensity[::-1])
    return uniform_sigma, intensity
