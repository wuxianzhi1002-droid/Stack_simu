from __future__ import annotations

from time import perf_counter_ns

import numpy as np

from model_config import LOWER_BOUNDS, UPPER_BOUNDS, bounds_center
from objective_functions import FitProblem
from optimizer_common import InversionResult, cluster_by_air, local_refine, pack_result


def optimize_cmaes(problem: FitProblem, config: dict, seed: int, **kwargs) -> InversionResult:
    """Compact IPOP-style restart CMA-ES implemented with NumPy."""
    del kwargs
    rng = np.random.default_rng(seed)
    start_calls = problem.n_forward_evaluations
    budget = int(config["global_budget"])
    scale = UPPER_BOUNDS - LOWER_BOUNDS
    archive: list[tuple[float, np.ndarray]] = []
    used = 0
    restart = 0
    begin = perf_counter_ns()
    while used < budget:
        population = min(8 * (2**restart), max(4, budget - used))
        mean = bounds_center() if restart == 0 else rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
        covariance = np.eye(5)
        sigma = 0.22
        mu = max(2, population // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        generations = max(1, min(12, (budget - used) // population))
        for _ in range(generations):
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-10)))
            z = rng.normal(size=(population, 5))
            unit_mean = (mean - LOWER_BOUNDS) / scale
            unit = np.clip(unit_mean + sigma * (z @ transform.T), 0.0, 1.0)
            candidates = LOWER_BOUNDS + unit * scale
            costs = np.array([problem.cost(candidate) for candidate in candidates])
            used += population
            order = np.argsort(costs)
            archive.extend((float(costs[index]), candidates[index].copy()) for index in order[: min(3, population)])
            elite_unit = unit[order[:mu]]
            new_unit_mean = np.sum(weights[:, None] * elite_unit, axis=0)
            centered = elite_unit - new_unit_mean
            covariance = 0.75 * covariance + 0.25 * ((centered * weights[:, None]).T @ centered) / max(sigma**2, 1e-8)
            mean = LOWER_BOUNDS + new_unit_mean * scale
            sigma *= 0.88
            if used >= budget:
                break
        restart += 1
    global_ms = (perf_counter_ns() - begin) / 1e6
    representatives = cluster_by_air(archive, float(config["cluster_air_um"]), int(config["refine_clusters"]))
    best = None
    local_ms = 0.0
    for candidate in representatives:
        refined, elapsed = local_refine(problem, candidate, int(config["local_max_nfev"]), str(config["loss"]))
        local_ms += elapsed
        cost = problem.cost(refined.x)
        if best is None or cost < best[0]:
            best = (cost, refined)
    if best is None:
        refined, elapsed = local_refine(problem, bounds_center(), int(config["local_max_nfev"]), str(config["loss"]))
        best = (problem.cost(refined.x), refined)
        local_ms += elapsed
    packed = pack_result("cmaes", problem, best[1].x, best[1].success, best[1].message, start_calls, global_search_ms=global_ms, local_refine_ms=local_ms)
    packed.metadata.update({"restart_strategy": "IPOP", "restarts": restart, "global_candidates": used})
    return packed
