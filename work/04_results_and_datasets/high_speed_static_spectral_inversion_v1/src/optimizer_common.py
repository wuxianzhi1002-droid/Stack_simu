from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from model_config import LOWER_BOUNDS, PARAMETER_NAMES, UPPER_BOUNDS
from objective_functions import FitProblem


@dataclass
class InversionResult:
    algorithm: str
    values: np.ndarray
    success: bool
    message: str
    final_cost: float
    spectral_rmse: float
    source_scale: float
    source_offset: float
    n_forward_evaluations: int
    coarse_search_ms: float = 0.0
    global_search_ms: float = 0.0
    local_refine_ms: float = 0.0
    timeout: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def cluster_by_air(candidates: list[tuple[float, np.ndarray]], separation_um: float, keep: int) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    for _, values in sorted(candidates, key=lambda item: item[0]):
        if all(abs(float(values[0] - prior[0])) >= separation_um for prior in selected):
            selected.append(np.asarray(values, dtype=float))
            if len(selected) >= keep:
                break
    return selected


def local_refine(
    problem: FitProblem,
    start: np.ndarray,
    max_nfev: int,
    loss: str,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> tuple[Any, float]:
    lo = LOWER_BOUNDS if lower is None else np.asarray(lower, dtype=float)
    hi = UPPER_BOUNDS if upper is None else np.asarray(upper, dtype=float)
    x0 = np.clip(np.asarray(start, dtype=float), lo + 1e-10, hi - 1e-10)
    begin = perf_counter_ns()
    result = least_squares(problem.residual, x0, bounds=(lo, hi), loss=loss, max_nfev=max_nfev, method="trf")
    elapsed_ms = (perf_counter_ns() - begin) / 1e6
    return result, elapsed_ms


def pack_result(
    algorithm: str,
    problem: FitProblem,
    values: np.ndarray,
    success: bool,
    message: str,
    start_calls: int,
    **timing: float,
) -> InversionResult:
    evaluation = problem.evaluate(values)
    return InversionResult(
        algorithm=algorithm,
        values=np.asarray(values, dtype=float),
        success=bool(success and np.all(np.isfinite(values))),
        message=str(message),
        final_cost=evaluation.cost,
        spectral_rmse=float(np.sqrt(np.mean((evaluation.fitted - problem.measurement.intensity) ** 2))),
        source_scale=evaluation.scale,
        source_offset=evaluation.offset,
        n_forward_evaluations=problem.n_forward_evaluations - start_calls,
        **timing,
    )


def result_dict(result: InversionResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "algorithm": result.algorithm,
        "success": result.success,
        "timeout": result.timeout,
        "message": result.message,
        "final_objective": result.final_cost,
        "spectral_rmse": result.spectral_rmse,
        "source_scale": result.source_scale,
        "source_offset": result.source_offset,
        "n_forward_evaluations": result.n_forward_evaluations,
        "coarse_search_ms": result.coarse_search_ms,
        "global_search_ms": result.global_search_ms,
        "local_refine_ms": result.local_refine_ms,
    }
    row.update({f"fit_{name}": float(value) for name, value in zip(PARAMETER_NAMES, result.values)})
    return row
