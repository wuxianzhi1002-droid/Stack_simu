"""GPU population/global and B=11 local primitives for V12 fixed-angle fitting."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares

import tmm_joint_inversion_v12 as v12
from v10_gpu.global_search import robust_objective_batch_device


class FixedAngleGpuPopulationObjective:
    """SciPy-vectorized strict full-ILS objective for five free parameters."""

    def __init__(self, backend, observed, scale, global_indices, fixed_angle_deg, config):
        self.backend = backend
        self.cp = backend.cp
        self.scale = float(scale)
        self.loss = str(config.loss)
        self.fixed_angle_deg = float(fixed_angle_deg)
        self.margin = float(config.boundary_margin_fraction)
        self.weight = float(config.boundary_penalty_weight)
        self.lower, self.upper = v12.bounds_arrays()
        self.span = self.upper - self.lower
        indices = np.asarray(global_indices, dtype=np.int64)
        observed_host = np.asarray(observed, dtype=np.float64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("global_indices must be a non-empty 1-D array")
        with backend.device:
            self.observed_device = self.cp.asarray(observed_host[indices], dtype=self.cp.float64)
            self.indices_device = self.cp.asarray(indices, dtype=self.cp.int64)
            self.lower_device = self.cp.asarray(self.lower, dtype=self.cp.float64)
            self.span_device = self.cp.asarray(self.span, dtype=self.cp.float64)
        self.phase = "unspecified"
        self._profiles = defaultdict(self._new_profile)

    @staticmethod
    def _new_profile():
        return {
            "objective_calls": 0,
            "gpu_batch_calls": 0,
            "candidate_evaluations": 0,
            "batch_sizes": [],
            "h2d_time_s": 0.0,
            "gpu_forward_time_s": 0.0,
            "gpu_objective_time_s": 0.0,
            "d2h_time_s": 0.0,
            "objective_wall_time_s": 0.0,
        }

    def set_phase(self, phase: str):
        self.phase = str(phase)

    def _normalize(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1 and array.shape == (5,):
            candidates = array[None, :]
        elif array.ndim == 2 and array.shape[0] == 5:
            candidates = array.T
        elif array.ndim == 2 and array.shape[1] == 5:
            candidates = array
        else:
            raise ValueError(f"expected (5,), (5,S), or (S,5), got {array.shape}")
        candidates = np.ascontiguousarray(candidates, dtype=np.float64)
        if not np.all(np.isfinite(candidates)):
            raise ValueError("global candidates contain NaN/Inf")
        if np.any(candidates < self.lower) or np.any(candidates > self.upper):
            raise ValueError("global candidates are outside V12 free-parameter bounds")
        return candidates

    def __call__(self, values):
        wall_started = time.perf_counter()
        candidates = self._normalize(values)
        profile = self._profiles[self.phase]
        profile["objective_calls"] += 1
        profile["gpu_batch_calls"] += 1
        profile["candidate_evaluations"] += len(candidates)
        profile["batch_sizes"].append(len(candidates))
        cp = self.cp
        with self.backend.device:
            e0, e1, e2, e3 = (cp.cuda.Event() for _ in range(4))
            e0.record()
            free_device = cp.asarray(candidates, dtype=cp.float64)
            angle_device = cp.full(
                (len(candidates), 1), self.fixed_angle_deg, dtype=cp.float64
            )
            full_device = cp.concatenate((free_device, angle_device), axis=1)
            e1.record()
            predictions = self.backend.predict_batch_device(full_device)
            e2.record()
            residuals = (
                predictions[:, self.indices_device] - self.observed_device[None, :]
            ) / self.scale
            spectrum = robust_objective_batch_device(residuals, self.loss, cp)
            normalized = (free_device - self.lower_device[None, :]) / self.span_device[None, :]
            distance = cp.minimum(normalized, 1.0 - normalized)
            severity = cp.sum(
                cp.square(cp.clip((self.margin - distance) / self.margin, 0.0, 1.0)),
                axis=1,
            )
            total = spectrum * (1.0 + self.weight * severity)
            e3.record()
            e3.synchronize()
            profile["h2d_time_s"] += float(cp.cuda.get_elapsed_time(e0, e1)) / 1000.0
            profile["gpu_forward_time_s"] += float(cp.cuda.get_elapsed_time(e1, e2)) / 1000.0
            profile["gpu_objective_time_s"] += float(cp.cuda.get_elapsed_time(e2, e3)) / 1000.0
            d2h_started = time.perf_counter()
            result = cp.asnumpy(total).astype(np.float64, copy=False)
            profile["d2h_time_s"] += time.perf_counter() - d2h_started
        profile["objective_wall_time_s"] += time.perf_counter() - wall_started
        if result.shape != (len(candidates),) or not np.all(np.isfinite(result)):
            raise FloatingPointError("V12 GPU global objective produced invalid energies")
        return result

    def profile(self):
        result = {}
        for phase, source in self._profiles.items():
            row = dict(source)
            sizes = row.pop("batch_sizes")
            row.update(
                {
                    "actual_batch_sizes": sizes,
                    "min_batch_size": int(min(sizes)) if sizes else 0,
                    "mean_batch_size": float(np.mean(sizes)) if sizes else 0.0,
                    "max_batch_size": int(max(sizes)) if sizes else 0,
                }
            )
            result[phase] = row
        return result


@dataclass(frozen=True)
class FixedAngleEvaluation:
    key: bytes
    solver: np.ndarray
    free: np.ndarray
    full: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray


class ExactFixedAngleBatchCache:
    """Exact-input B=11 cache in normalized five-parameter coordinates."""

    def __init__(self, backend, observed, scale, fixed_angle_deg):
        self.backend = backend
        self.observed = np.asarray(observed, dtype=np.float64)
        self.scale = float(scale)
        self.fixed_angle_deg = float(fixed_angle_deg)
        self.lower, self.upper = v12.bounds_arrays()
        self.span = self.upper - self.lower
        self._cached = None
        self.residual_calls = 0
        self.jacobian_calls = 0
        self.evaluations = 0
        self.hits = 0
        self.gpu_batch_runtime_s = 0.0

    def solver_to_free(self, solver):
        values = np.asarray(solver, dtype=np.float64)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("five finite normalized solver values are required")
        return self.lower + values * self.span

    def free_to_solver(self, free):
        return np.clip((np.asarray(free, dtype=np.float64) - self.lower) / self.span, 0.0, 1.0)

    def _batch(self, solver):
        values = np.asarray(solver, dtype=np.float64)
        steps = np.full(5, 1.0e-6, dtype=np.float64)
        plus = np.repeat(values[None, :], 5, axis=0)
        minus = np.repeat(values[None, :], 5, axis=0)
        indices = np.arange(5)
        plus[indices, indices] = np.minimum(1.0, values + steps)
        minus[indices, indices] = np.maximum(0.0, values - steps)
        denominator = plus[indices, indices] - minus[indices, indices]
        solver_batch = np.vstack((values[None, :], plus, minus))
        free_batch = self.lower[None, :] + solver_batch * self.span[None, :]
        angle = np.full((11, 1), self.fixed_angle_deg, dtype=np.float64)
        return solver_batch, free_batch, np.hstack((free_batch, angle)), denominator

    def evaluate(self, solver):
        values = np.ascontiguousarray(np.asarray(solver, dtype=np.float64))
        key = values.tobytes()
        if self._cached is not None and key == self._cached.key:
            self.hits += 1
            return self._cached
        _, free_batch, full_batch, denominator = self._batch(values)
        started = time.perf_counter()
        spectra = np.asarray(self.backend.predict_batch(full_batch), dtype=np.float64)
        self.gpu_batch_runtime_s += time.perf_counter() - started
        residual = (spectra[0] - self.observed) / self.scale
        jacobian = ((spectra[1:6] - spectra[6:11]) / denominator[:, None] / self.scale).T
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("V12 fixed-angle cache produced NaN/Inf")
        self._cached = FixedAngleEvaluation(
            key,
            values.copy(),
            free_batch[0].copy(),
            full_batch[0].copy(),
            spectra[0].copy(),
            residual,
            jacobian,
        )
        self.evaluations += 1
        return self._cached

    def residual(self, solver):
        self.residual_calls += 1
        return self.evaluate(solver).residual

    def jacobian(self, solver):
        self.jacobian_calls += 1
        return self.evaluate(solver).jacobian

    def snapshot(self):
        return {
            "residual_calls": self.residual_calls,
            "jacobian_calls": self.jacobian_calls,
            "actual_gpu_batch_evaluations": self.evaluations,
            "cache_hits": self.hits,
            "gpu_batch_runtime_s": self.gpu_batch_runtime_s,
            "local_batch_size": 11,
        }


def run_gpu_global(v10, measurement, config, population, backend):
    observed = np.asarray(measurement["spectrum"], dtype=np.float64)
    indices = np.arange(0, len(observed), max(1, int(config.global_stride)))
    objective = FixedAngleGpuPopulationObjective(
        backend,
        observed,
        v10.robust_scale(observed),
        indices,
        measurement["fixed_angle_deg"],
        config,
    )
    kwargs = dict(
        bounds=[v10.BOUNDS[name] for name in v12.FREE_PARAMS],
        strategy="best1bin",
        maxiter=int(config.global_maxiter),
        popsize=int(config.global_popsize),
        tol=1.0e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=int(config.random_seed),
        polish=False,
        init=population,
        workers=1,
        updating="deferred",
        vectorized=True,
    )
    total_started = time.perf_counter()
    objective.set_phase("initial_audit")
    initial = objective(population.T)
    objective.set_phase("differential_evolution")
    de_started = time.perf_counter()
    result = differential_evolution(objective, **kwargs)
    de_runtime = time.perf_counter() - de_started
    profile = objective.profile()
    differential = profile["differential_evolution"]
    differential["scipy_de_wall_time_s"] = de_runtime
    differential["scipy_cpu_overhead_s"] = max(
        0.0, de_runtime - differential["objective_wall_time_s"]
    )
    profile["totals"] = {
        "total_global_runtime_s": time.perf_counter() - total_started,
        "gpu_batch_calls": sum(row["gpu_batch_calls"] for row in profile.values()),
        "candidate_evaluations": sum(row["candidate_evaluations"] for row in profile.values()),
        "h2d_time_s": sum(row["h2d_time_s"] for row in profile.values()),
        "gpu_forward_time_s": sum(row["gpu_forward_time_s"] for row in profile.values()),
        "gpu_objective_time_s": sum(row["gpu_objective_time_s"] for row in profile.values()),
        "d2h_time_s": sum(row["d2h_time_s"] for row in profile.values()),
        "updating": "deferred",
        "vectorized": True,
        "population_size": len(population),
    }
    candidates = v12.select_diverse_candidates(
        result.population, result.population_energies, config.multistarts
    )
    return initial, result, candidates, profile


def _local_attempt(cache, start, config, call_index, population_index, global_energy):
    trace = []

    def fun(values):
        residual = cache.residual(values)
        trace.append(0.5 * v12.v10.robust_objective(residual, config.loss))
        return residual

    started = time.perf_counter()
    result = least_squares(
        fun,
        x0=cache.free_to_solver(start),
        jac=cache.jacobian,
        bounds=(np.zeros(5), np.ones(5)),
        loss=config.loss,
        max_nfev=int(config.max_nfev),
        x_scale=1.0,
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=float(config.local_gtol),
    )
    free = cache.solver_to_free(result.x)
    return {
        "call_index": int(call_index),
        "population_index": int(population_index),
        "global_total_objective": float(global_energy),
        "x0_free": np.asarray(start),
        "final_solver": np.asarray(result.x),
        "final_free": free,
        "final_full": v12.full_parameters(free, cache.fixed_angle_deg),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "spectrum_cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "runtime_s": time.perf_counter() - started,
        "trace_costs": trace,
    }


def run_gpu_local(v10, measurement, config, candidates, gpu_backend, cpu_backend):
    if len(candidates) != int(config.multistarts):
        raise RuntimeError(f"expected {config.multistarts} local starts, got {len(candidates)}")
    cache = ExactFixedAngleBatchCache(
        gpu_backend,
        measurement["spectrum"],
        v10.robust_scale(measurement["spectrum"]),
        measurement["fixed_angle_deg"],
    )
    started = time.perf_counter()
    attempts = [
        _local_attempt(cache, candidate, config, index, population_index, energy)
        for index, (candidate, population_index, energy) in enumerate(candidates, 1)
    ]
    ranked = v12.select_local_candidate(attempts, config)
    selected = ranked["selected"]
    if selected is None:
        raise RuntimeError("no valid V12 GPU local candidate")
    evaluation = cache.evaluate(np.asarray(selected["final_solver"]))
    cpu_prediction = cpu_backend.predict(evaluation.full)
    difference = evaluation.prediction - cpu_prediction
    closure = {
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "nan_inf_count": int(difference.size - np.isfinite(difference).sum()),
    }
    closure["pass"] = bool(
        closure["rmse"] <= 1.0e-10
        and closure["max_abs"] <= 1.0e-8
        and closure["nan_inf_count"] == 0
    )
    if not closure["pass"]:
        raise RuntimeError(f"V12 CPU/GPU strict closure failed: {closure}")
    return {
        "runtime_s": time.perf_counter() - started,
        "selected": selected,
        "ranking": ranked,
        "free_parameters": evaluation.free,
        "full_parameters": evaluation.full,
        "prediction": evaluation.prediction,
        "spectrum_cost": float(selected["spectrum_cost"]),
        "exact_rmse": float(
            np.sqrt(np.mean((evaluation.prediction - measurement["spectrum"]) ** 2))
        ),
        "fixed_jacobian_diagnostics": v12.matrix_diagnostics(evaluation.jacobian),
        "closure": closure,
        "cache": cache.snapshot(),
    }


def augmented_six_parameter_diagnostics(backend, full_values, observed, scale):
    """Diagnostic-only B=13 physical Jacobian, including the fixed Angle column."""
    center = np.asarray(full_values, dtype=np.float64)
    lower, upper = v12.v10.bounds_arrays()
    lower = lower.copy()
    upper = upper.copy()
    lower[5], upper[5] = -0.25, 0.25
    span = upper - lower
    plus = np.repeat(center[None, :], 6, axis=0)
    minus = np.repeat(center[None, :], 6, axis=0)
    for index in range(6):
        step = max(1.0e-6 * span[index], 1.0e-8)
        plus[index, index] = min(upper[index], center[index] + step)
        minus[index, index] = max(lower[index], center[index] - step)
    denominator = plus[np.arange(6), np.arange(6)] - minus[np.arange(6), np.arange(6)]
    spectra = np.asarray(backend.predict_batch(np.vstack((center[None, :], plus, minus))))
    jacobian = ((spectra[1:7] - spectra[7:13]) / denominator[:, None] / float(scale)).T
    singular = np.linalg.svd(jacobian, compute_uv=False)
    norms = np.linalg.norm(jacobian, axis=0)
    rho = None
    if norms[0] > 0.0 and norms[5] > 0.0:
        rho = float(np.dot(jacobian[:, 0], jacobian[:, 5]) / (norms[0] * norms[5]))
    tolerance = max(jacobian.shape) * np.finfo(float).eps * singular[0]
    return {
        "parameter_order": list(v12.FULL_PARAMS),
        "batch_size": 13,
        "condition_number": None if singular[-1] == 0.0 else float(singular[0] / singular[-1]),
        "smallest_singular_value": float(singular[-1]),
        "numerical_rank": int(np.sum(singular > tolerance)),
        "rho_air_angle": rho,
        "angle_column_norm": float(norms[5]),
    }
