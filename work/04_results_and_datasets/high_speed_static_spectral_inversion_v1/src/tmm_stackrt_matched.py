from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_config import (
    C0_M_S,
    LAYER_NAMES,
    STACKRT_FREQUENCY_C_M_S,
    material_n,
    parameters_to_thickness_um,
)


@dataclass
class TMMCache:
    wavelengths_um: np.ndarray
    frequency_hz: np.ndarray
    phase_wavelength_m: np.ndarray
    k0: np.ndarray
    n_matrix: np.ndarray


class StackRTMatchedTMM:
    """Normal-incidence Rp model matching the validated StackRT convention."""

    def __init__(self, wavelengths_um: np.ndarray):
        wavelength = np.asarray(wavelengths_um, dtype=np.float64)
        if wavelength.ndim != 1 or wavelength.size < 2 or np.any(wavelength <= 0.0):
            raise ValueError("wavelengths_um must be a positive one-dimensional array.")
        frequency = STACKRT_FREQUENCY_C_M_S / (wavelength * 1e-6)
        phase_wavelength = C0_M_S / frequency
        n_matrix = np.vstack([material_n(name, wavelength) for name in LAYER_NAMES])
        self.cache = TMMCache(wavelength, frequency, phase_wavelength, 2.0 * np.pi / phase_wavelength, n_matrix)
        self.n_forward_evaluations = 0

    @property
    def wavelengths_um(self) -> np.ndarray:
        return self.cache.wavelengths_um

    def reset_counter(self) -> None:
        self.n_forward_evaluations = 0

    def reflectance(self, values: np.ndarray) -> np.ndarray:
        self.n_forward_evaluations += 1
        thickness_m = parameters_to_thickness_um(values) * 1e-6
        n = self.cache.n_matrix
        m11 = np.ones(n.shape[1], dtype=np.complex128)
        m12 = np.zeros(n.shape[1], dtype=np.complex128)
        m21 = np.zeros(n.shape[1], dtype=np.complex128)
        m22 = np.ones(n.shape[1], dtype=np.complex128)
        for layer_index in range(1, len(LAYER_NAMES) - 1):
            delta = self.cache.k0 * n[layer_index] * thickness_m[layer_index]
            c_delta = np.cos(delta)
            s_delta = np.sin(delta)
            q = n[layer_index]
            a12 = -1j * s_delta / q
            a21 = -1j * q * s_delta
            m11, m12, m21, m22 = (
                m11 * c_delta + m12 * a21,
                m11 * a12 + m12 * c_delta,
                m21 * c_delta + m22 * a21,
                m21 * a12 + m22 * c_delta,
            )
        q0 = n[0]
        qs = n[-1]
        numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
        denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
        return np.abs(numerator / denominator) ** 2

    def reflectance_batch(self, values: np.ndarray) -> np.ndarray:
        candidates = np.asarray(values, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != 5:
            raise ValueError("values must have shape (N_candidates, 5).")
        self.n_forward_evaluations += candidates.shape[0]
        thickness_um = np.column_stack(
            (
                np.zeros(candidates.shape[0]),
                candidates[:, 0],
                candidates[:, 1] / 1000.0,
                candidates[:, 2] / 1000.0,
                candidates[:, 3] / 1000.0,
                candidates[:, 4] / 1000.0,
                np.zeros(candidates.shape[0]),
            )
        )
        thickness_m = thickness_um * 1e-6
        n = self.cache.n_matrix[None, :, :]
        shape = (candidates.shape[0], self.cache.wavelengths_um.size)
        m11 = np.ones(shape, dtype=np.complex128)
        m12 = np.zeros(shape, dtype=np.complex128)
        m21 = np.zeros(shape, dtype=np.complex128)
        m22 = np.ones(shape, dtype=np.complex128)
        for layer_index in range(1, len(LAYER_NAMES) - 1):
            delta = self.cache.k0[None, :] * n[:, layer_index, :] * thickness_m[:, layer_index, None]
            c_delta = np.cos(delta)
            s_delta = np.sin(delta)
            q = n[:, layer_index, :]
            a12 = -1j * s_delta / q
            a21 = -1j * q * s_delta
            m11, m12, m21, m22 = (
                m11 * c_delta + m12 * a21,
                m11 * a12 + m12 * c_delta,
                m21 * c_delta + m22 * a21,
                m21 * a12 + m22 * c_delta,
            )
        q0 = n[:, 0, :]
        qs = n[:, -1, :]
        numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
        denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
        return np.abs(numerator / denominator) ** 2


def stackrt_arrays(wavelengths_um: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelengths_um, dtype=float)
    frequency = STACKRT_FREQUENCY_C_M_S / (wavelength * 1e-6)
    n_matrix = np.vstack([material_n(name, wavelength) for name in LAYER_NAMES])
    thickness_m = parameters_to_thickness_um(values) * 1e-6
    return n_matrix, thickness_m, frequency
