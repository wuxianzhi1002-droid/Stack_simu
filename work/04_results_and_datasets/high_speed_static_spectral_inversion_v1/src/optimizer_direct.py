from __future__ import annotations

from time import perf_counter_ns

import numpy as np
from scipy.optimize import Bounds, direct

from model_config import LOWER_BOUNDS, UPPER_BOUNDS
from objective_functions import FitProblem
from optimizer_common import InversionResult, cluster_by_air, local_refine, pack_result


def optimize_direct(problem: FitProblem, config: dict, seed: int, **kwargs) -> InversionResult:
    del seed, kwargs
    start_calls = problem.n_forward_evaluations
    archive: list[tuple[float, np.ndarray]] = []

    def objective(values: np.ndarray) -> float:
        cost = problem.cost(values)
        archive.append((cost, np.asarray(values, dtype=float).copy()))
        return cost

    begin = perf_counter_ns()
    result = direct(objective, Bounds(LOWER_BOUNDS, UPPER_BOUNDS), maxfun=int(config["global_budget"]), locally_biased=False)
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
        refined, elapsed = local_refine(problem, result.x, int(config["local_max_nfev"]), str(config["loss"]))
        best = (problem.cost(refined.x), refined)
        local_ms += elapsed
    packed = pack_result("direct", problem, best[1].x, best[1].success, best[1].message, start_calls, global_search_ms=global_ms, local_refine_ms=local_ms)
    packed.metadata.update({"direct_nfev": int(result.nfev)})
    return packed
