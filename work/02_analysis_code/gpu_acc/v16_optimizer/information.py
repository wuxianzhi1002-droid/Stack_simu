"""Noise-whitened Jacobian and Fisher-information diagnostics for Stage 1."""

from __future__ import annotations

from typing import Any

import numpy as np

import tmm_joint_inversion_v12 as v12


def signal_jacobian_normalized_solver(
    backend,
    free_parameters: np.ndarray,
    fixed_angle_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return d(signal)/d(normalized five-parameter coordinate), using GPU B=11."""
    free = np.asarray(free_parameters, dtype=np.float64)
    lower, upper = v12.bounds_arrays()
    span = upper - lower
    if free.shape != (5,) or np.any(~np.isfinite(free)):
        raise ValueError("five finite fitted free parameters are required")
    solver = np.clip((free - lower) / span, 0.0, 1.0)
    step = np.full(5, 1.0e-6, dtype=np.float64)
    plus = np.repeat(solver[None, :], 5, axis=0)
    minus = np.repeat(solver[None, :], 5, axis=0)
    indices = np.arange(5)
    plus[indices, indices] = np.minimum(1.0, solver + step)
    minus[indices, indices] = np.maximum(0.0, solver - step)
    denominator = plus[indices, indices] - minus[indices, indices]
    if np.any(denominator <= 0.0):
        raise FloatingPointError("finite-difference denominator is not positive")
    solver_batch = np.vstack((solver[None, :], plus, minus))
    free_batch = lower[None, :] + solver_batch * span[None, :]
    full_batch = np.column_stack(
        (free_batch, np.full(11, float(fixed_angle_deg), dtype=np.float64))
    )
    spectra = np.asarray(backend.predict_batch(full_batch), dtype=np.float64)
    jacobian = ((spectra[1:6] - spectra[6:11]) / denominator[:, None]).T
    if spectra.shape[0] != 11 or jacobian.shape[1] != 5:
        raise RuntimeError("unexpected Stage 1 Jacobian batch shape")
    if np.any(~np.isfinite(jacobian)):
        raise FloatingPointError("Stage 1 signal Jacobian contains NaN/Inf")
    return jacobian, {
        "coordinate_system": "normalized [0,1]^5 V15 free-parameter coordinates",
        "parameter_order": list(v12.FREE_PARAMS),
        "batch_size": 11,
        "finite_difference_step": 1.0e-6,
        "one_sided_columns": [
            v12.FREE_PARAMS[index]
            for index in range(5)
            if not np.isclose(denominator[index], 2.0e-6, rtol=1.0e-7, atol=1.0e-15)
        ],
    }


def _safe_power10(log10_value: float) -> float | None:
    if not np.isfinite(log10_value) or log10_value > 308.0 or log10_value < -323.0:
        return None
    return float(10.0**log10_value)


def _matrix_metrics(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    tolerance = max(values.shape) * np.finfo(np.float64).eps * singular[0]
    sigma_min = float(singular[-1])
    condition = None if sigma_min == 0.0 else float(singular[0] / sigma_min)
    fisher = values.T @ values
    sign, logdet = np.linalg.slogdet(fisher)
    log10_det = None if sign <= 0.0 else float(logdet / np.log(10.0))
    if log10_det is None:
        determinant = None
        mantissa = None
        exponent = None
    else:
        determinant = _safe_power10(log10_det)
        exponent = int(np.floor(log10_det))
        mantissa = float(10.0 ** (log10_det - exponent))
    return {
        "singular_values": singular,
        "smallest_singular_value": sigma_min,
        "condition_number": condition,
        "numerical_rank": int(np.sum(singular > tolerance)),
        "fisher_log10_determinant": log10_det,
        "fisher_determinant": determinant,
        "fisher_determinant_mantissa": mantissa,
        "fisher_determinant_exponent10": exponent,
    }


def whitened_information_metrics(
    signal_jacobian: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    """Compute requested Jw metrics and row-count-normalized information density."""
    jacobian = np.asarray(signal_jacobian, dtype=np.float64)
    noise_sigma = np.asarray(sigma, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[1] != 5:
        raise ValueError("signal Jacobian must have shape (wavelengths, 5)")
    if noise_sigma.shape != (jacobian.shape[0],):
        raise ValueError("noise sigma must match the Jacobian wavelength dimension")
    if np.any(~np.isfinite(noise_sigma)) or np.any(noise_sigma <= 0.0):
        raise ValueError("noise sigma must be finite and strictly positive")
    whitened = jacobian / noise_sigma[:, None]
    raw = _matrix_metrics(whitened)
    density = _matrix_metrics(whitened / np.sqrt(float(len(whitened))))
    return {
        "definition": "Jw = diag(1/sigma_lambda) @ d(signal)/d(normalized parameters)",
        "covariance_model": "diagonal",
        "wavelength_samples": int(len(whitened)),
        "sigma_min": float(np.min(noise_sigma)),
        "sigma_median": float(np.median(noise_sigma)),
        "sigma_max": float(np.max(noise_sigma)),
        "raw_information": raw,
        "per_sample_information_density": density,
    }
