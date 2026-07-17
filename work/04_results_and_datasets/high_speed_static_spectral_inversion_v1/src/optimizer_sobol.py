from __future__ import annotations

from time import perf_counter_ns

import numpy as np
from scipy.stats import qmc

from model_config import LOWER_BOUNDS, UPPER_BOUNDS
from objective_functions import FitProblem
from optimizer_common import InversionResult, local_refine, pack_result


def optimize_sobol(problem: FitProblem, config: dict, seed: int, **kwargs) -> InversionResult:
    del kwargs
    start_calls = problem.n_forward_evaluations
    count = int(config["sobol_starts"])
    exponent = int(np.ceil(np.log2(max(2, count))))
    unit = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(exponent)[:count]
    starts = qmc.scale(unit, LOWER_BOUNDS, UPPER_BOUNDS)
    best = None
    local_total_ms = 0.0
    begin = perf_counter_ns()
    max_each = max(12, int(config["global_budget"]) // count)
    for start in starts:
        result, elapsed = local_refine(problem, start, max_each, str(config["loss"]))
        local_total_ms += elapsed
        cost = problem.cost(result.x)
        if best is None or cost < best[0]:
            best = (cost, result)
    coarse_ms = (perf_counter_ns() - begin) / 1e6 - local_total_ms
    assert best is not None
    return pack_result(
        "sobol",
        problem,
        best[1].x,
        best[1].success,
        best[1].message,
        start_calls,
        coarse_search_ms=max(0.0, coarse_ms),
        local_refine_ms=local_total_ms,
    )
