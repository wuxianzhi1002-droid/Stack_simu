"""Exact-key B=13 residual/Jacobian cache for formal V9 solver coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import numpy as np

from .source import load_v9_module

PARAMETER_NAMES = ("Air", "HSQ", "PSS", "SOC", "TiO2", "Angle")
ANGLE_INDEX = 5


def bounds_and_span() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = load_v9_module().bounds_arrays()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return lower, upper, upper - lower


def physical_to_solver(values: np.ndarray) -> np.ndarray:
    lower, _upper, span = bounds_and_span()
    normalized = np.clip((np.asarray(values, dtype=np.float64) - lower) / span, 0.0, 1.0)
    solver = normalized.copy()
    solver[ANGLE_INDEX] = normalized[ANGLE_INDEX] ** 2
    return solver


def solver_to_physical(values: np.ndarray) -> np.ndarray:
    lower, _upper, span = bounds_and_span()
    transformed = np.asarray(values, dtype=np.float64).copy()
    transformed[ANGLE_INDEX] = np.sqrt(max(0.0, float(transformed[ANGLE_INDEX])))
    return lower + transformed * span


def build_center_difference_batch(values: np.ndarray):
    lower, upper, span = bounds_and_span()
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("physical values must be a finite six-parameter vector")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("physical values are outside formal V9 bounds")
    steps = np.maximum(1.0e-6 * span, 1.0e-8)
    plus = np.repeat(values[None, :], 6, axis=0)
    minus = np.repeat(values[None, :], 6, axis=0)
    indices = np.arange(6)
    plus[indices, indices] = np.minimum(upper, values + steps)
    minus[indices, indices] = np.maximum(lower, values - steps)
    denominators = plus[indices, indices] - minus[indices, indices]
    batch = np.vstack((values[None, :], plus, minus))
    if batch.shape != (13, 6) or np.any(denominators <= 0.0):
        raise RuntimeError("V9 center-difference B=13 contract failed")
    return batch, denominators


def solver_jacobian_from_physical(
    jacobian: np.ndarray, physical_values: np.ndarray
) -> np.ndarray:
    lower, _upper, span = bounds_and_span()
    normalized = np.clip(
        (np.asarray(physical_values, dtype=np.float64) - lower) / span, 0.0, 1.0
    )
    if normalized[ANGLE_INDEX] <= 0.0:
        raise ValueError("Formal V9 Angle sqrt map derivative is undefined at lower bound")
    scales = span.copy()
    scales[ANGLE_INDEX] = span[ANGLE_INDEX] / (2.0 * normalized[ANGLE_INDEX])
    return np.asarray(jacobian, dtype=np.float64) * scales[None, :]


@dataclass(frozen=True)
class CachedEvaluation:
    key: bytes
    solver: np.ndarray
    physical: np.ndarray
    base_prediction: np.ndarray
    residual: np.ndarray
    physical_jacobian: np.ndarray
    solver_jacobian: np.ndarray
    batch_runtime_s: float


class ExactV9BatchCache:
    def __init__(self, backend: Any, observed: np.ndarray, scale: float):
        self.backend = backend
        self.observed = np.asarray(observed, dtype=np.float64)
        self.scale = float(scale)
        if self.observed.ndim != 1 or not np.all(np.isfinite(self.observed)):
            raise ValueError("observed must be a finite one-dimensional array")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        self._cached: CachedEvaluation | None = None
        self.residual_calls = 0
        self.jacobian_calls = 0
        self.actual_batch_evaluations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_runtime_s = 0.0
        self.residual_callback_runtime_s = 0.0
        self.jacobian_callback_runtime_s = 0.0

    @staticmethod
    def exact_key(values: np.ndarray) -> tuple[bytes, np.ndarray]:
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError("solver vector must contain six finite float64 values")
        return array.tobytes(order="C"), array

    def _evaluate(self, solver: np.ndarray) -> CachedEvaluation:
        key, solver_array = self.exact_key(solver)
        if self._cached is not None and key == self._cached.key:
            self.cache_hits += 1
            return self._cached
        self.cache_misses += 1
        physical = solver_to_physical(solver_array)
        batch, denominators = build_center_difference_batch(physical)
        started = time.perf_counter()
        spectra = np.asarray(self.backend.predict_batch(batch), dtype=np.float64)
        self.backend.synchronize()
        elapsed = time.perf_counter() - started
        expected = (13, self.observed.size)
        if spectra.shape != expected:
            raise RuntimeError(f"Expected V9 B=13 spectra {expected}, got {spectra.shape}")
        physical_jacobian = (
            (spectra[1:7] - spectra[7:13]) / denominators[:, None] / self.scale
        ).T
        solver_jacobian = solver_jacobian_from_physical(
            physical_jacobian, physical
        )
        residual = (spectra[0] - self.observed) / self.scale
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(solver_jacobian)):
            raise FloatingPointError("V9 B=13 evaluation produced NaN/Inf")
        result = CachedEvaluation(
            key,
            solver_array.copy(),
            physical,
            spectra[0],
            residual,
            physical_jacobian,
            solver_jacobian,
            elapsed,
        )
        self._cached = result
        self.actual_batch_evaluations += 1
        self.batch_runtime_s += elapsed
        return result

    def residual(self, solver: np.ndarray) -> np.ndarray:
        self.residual_calls += 1
        started = time.perf_counter()
        result = self._evaluate(solver).residual
        self.residual_callback_runtime_s += time.perf_counter() - started
        return result

    def jacobian(self, solver: np.ndarray) -> np.ndarray:
        self.jacobian_calls += 1
        started = time.perf_counter()
        result = self._evaluate(solver).solver_jacobian
        self.jacobian_callback_runtime_s += time.perf_counter() - started
        return result

    def evaluate(self, solver: np.ndarray) -> CachedEvaluation:
        return self._evaluate(solver)

    def snapshot(self) -> dict[str, Any]:
        return {
            "residual_calls": self.residual_calls,
            "jacobian_calls": self.jacobian_calls,
            "actual_gpu_batch_evaluations": self.actual_batch_evaluations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "gpu_batch_runtime_s": self.batch_runtime_s,
            "residual_callback_runtime_s": self.residual_callback_runtime_s,
            "jacobian_callback_runtime_s": self.jacobian_callback_runtime_s,
            "exact_key_contract": (
                "contiguous float64 solver bytes; one-entry cache; no approximate matching"
            ),
        }
