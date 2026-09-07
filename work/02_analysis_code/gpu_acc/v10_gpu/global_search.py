"""GPU population-batched full-ILS objective for SciPy differential_evolution."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from .backend.v10_source import load_v10_module


def robust_objective_batch_device(residuals_device: Any, loss: str, cp: Any):
    """Return one unchanged robust objective per residual row, on device."""
    residuals = cp.asarray(residuals_device, dtype=cp.float64)
    if residuals.ndim != 2:
        raise ValueError("residuals_device must have shape (B, N_residual)")
    z = residuals * residuals
    if loss == "linear":
        rho = z
    elif loss == "soft_l1":
        rho = 2.0 * (cp.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = cp.where(z <= 1.0, z, 2.0 * cp.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho = cp.log1p(z)
    elif loss == "arctan":
        rho = cp.arctan(z)
    else:
        raise ValueError(f"Unknown loss: {loss}")
    return cp.sum(rho, axis=1, dtype=cp.float64)


class GpuFullIlsPopulationObjective:
    """SciPy-vectorized objective backed by device-resident strict responses."""

    def __init__(
        self,
        backend: Any,
        observed: np.ndarray,
        scale: float,
        global_indices: np.ndarray,
        loss: str,
        max_chunk_size: int | None = None,
    ):
        self.backend = backend
        self.cp = backend.cp
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        self.loss = str(loss)
        self.max_chunk_size = None if max_chunk_size is None else int(max_chunk_size)
        if self.max_chunk_size is not None and self.max_chunk_size < 2:
            raise ValueError("max_chunk_size must be at least two")
        indices = np.asarray(global_indices, dtype=np.int64)
        observed_host = np.asarray(observed, dtype=np.float64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("global_indices must be a non-empty 1-D array")
        if np.any(indices < 0) or np.any(indices >= observed_host.size):
            raise ValueError("global_indices are out of range")
        v10 = load_v10_module()
        self.lower, self.upper = v10.bounds_arrays()
        with backend.device:
            self.observed_global_device = self.cp.asarray(
                observed_host[indices], dtype=self.cp.float64
            )
            self.global_indices_device = self.cp.asarray(indices, dtype=self.cp.int64)
        self.phase = "unspecified"
        self._profiles: dict[str, dict[str, Any]] = defaultdict(self._new_profile)

    @staticmethod
    def _new_profile():
        return {
            "objective_calls": 0,
            "gpu_batch_calls": 0,
            "candidate_evaluations": 0,
            "batch_sizes": [],
            "chunk_sizes": [],
            "h2d_time_s": 0.0,
            "gpu_forward_time_s": 0.0,
            "gpu_robust_objective_time_s": 0.0,
            "d2h_time_s": 0.0,
            "objective_wall_time_s": 0.0,
        }

    def set_phase(self, phase: str):
        self.phase = str(phase)

    def _normalize_scipy_input(self, x: Any) -> np.ndarray:
        array = np.asarray(x, dtype=np.float64)
        if array.ndim == 1:
            if array.shape != (6,):
                raise ValueError("one candidate must have shape (6,)")
            candidates = array[None, :]
        elif array.ndim == 2 and array.shape[0] == 6:
            candidates = array.T
        elif array.ndim == 2 and array.shape[1] == 6:
            candidates = array
        else:
            raise ValueError(
                f"expected SciPy shape (6, S) or backend shape (S, 6), got {array.shape}"
            )
        candidates = np.ascontiguousarray(candidates, dtype=np.float64)
        if not np.all(np.isfinite(candidates)):
            raise ValueError("global candidates contain NaN/Inf")
        if np.any(candidates < self.lower[None, :]) or np.any(
            candidates > self.upper[None, :]
        ):
            raise ValueError("global candidates are outside frozen physical bounds")
        return candidates

    def _events(self):
        return tuple(self.cp.cuda.Event() for _ in range(4))

    def _evaluate_chunk(self, candidates: np.ndarray, profile: dict[str, Any]):
        cp = self.cp
        with self.backend.device:
            e0, e1, e2, e3 = self._events()
            e0.record()
            params_device = self.backend.parameters_to_device(candidates)
            e1.record()
            predictions_device = self.backend.predict_batch_device(params_device)
            e2.record()
            residuals_device = (
                predictions_device[:, self.global_indices_device]
                - self.observed_global_device[None, :]
            ) / self.scale
            energies_device = robust_objective_batch_device(
                residuals_device, self.loss, cp
            )
            e3.record()
            e3.synchronize()
            profile["h2d_time_s"] += float(cp.cuda.get_elapsed_time(e0, e1)) / 1000.0
            profile["gpu_forward_time_s"] += (
                float(cp.cuda.get_elapsed_time(e1, e2)) / 1000.0
            )
            profile["gpu_robust_objective_time_s"] += (
                float(cp.cuda.get_elapsed_time(e2, e3)) / 1000.0
            )
            d2h_started = time.perf_counter()
            energies = cp.asnumpy(energies_device).astype(np.float64, copy=False)
            profile["d2h_time_s"] += time.perf_counter() - d2h_started
        if energies.shape != (len(candidates),) or not np.all(np.isfinite(energies)):
            raise FloatingPointError("GPU global objective produced invalid energies")
        return energies

    def __call__(self, x: Any) -> np.ndarray:
        started = time.perf_counter()
        candidates = self._normalize_scipy_input(x)
        profile = self._profiles[self.phase]
        profile["objective_calls"] += 1
        profile["candidate_evaluations"] += int(len(candidates))
        profile["batch_sizes"].append(int(len(candidates)))
        chunk_size = self.max_chunk_size or int(len(candidates))
        chunks = []
        for begin in range(0, len(candidates), chunk_size):
            chunk = candidates[begin : begin + chunk_size]
            profile["gpu_batch_calls"] += 1
            profile["chunk_sizes"].append(int(len(chunk)))
            chunks.append(self._evaluate_chunk(chunk, profile))
        energies = np.concatenate(chunks)
        profile["objective_wall_time_s"] += time.perf_counter() - started
        return energies

    def profile(self) -> dict[str, Any]:
        result = {}
        for phase, source in self._profiles.items():
            row = dict(source)
            sizes = row.pop("batch_sizes")
            chunks = row.pop("chunk_sizes")
            row.update(
                {
                    "mean_batch_size": float(np.mean(sizes)) if sizes else 0.0,
                    "max_batch_size": int(max(sizes)) if sizes else 0,
                    "min_batch_size": int(min(sizes)) if sizes else 0,
                    "mean_chunk_size": float(np.mean(chunks)) if chunks else 0.0,
                    "max_chunk_size": int(max(chunks)) if chunks else 0,
                    "actual_batch_sizes": sizes,
                    "actual_chunk_sizes": chunks,
                }
            )
            result[phase] = row
        return result
