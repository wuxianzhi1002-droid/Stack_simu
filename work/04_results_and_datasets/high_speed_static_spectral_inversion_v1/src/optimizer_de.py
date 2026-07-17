from __future__ import annotations

from time import perf_counter_ns

import numpy as np
from scipy.optimize import differential_evolution

from model_config import LOWER_BOUNDS, UPPER_BOUNDS
from objective_functions import FitProblem
from optimizer_common import InversionResult, cluster_by_air, local_refine, pack_result


def optimize_de(problem: FitProblem, config: dict, seed: int, strategy: str = "best1bin", **kwargs) -> InversionResult:
    del kwargs
    start_calls = problem.n_forward_evaluations
    budget = int(config["global_budget"])
    popsize = max(4, min(10, budget // 25))
    population_count = popsize * 5
    maxiter = max(1, budget // population_count - 1)
    begin = perf_counter_ns()
    result = differential_evolution(
        problem.cost,
        list(zip(LOWER_BOUNDS, UPPER_BOUNDS)),
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        init="sobol",
        seed=seed,
        polish=False,
        workers=1,
        updating="immediate",
    )
    global_ms = (perf_counter_ns() - begin) / 1e6
    candidates = [(float(cost), np.asarray(x, dtype=float)) for cost, x in zip(result.population_energies, result.population)]
    representatives = cluster_by_air(candidates, float(config["cluster_air_um"]), int(config["refine_clusters"]))
    representatives = representatives or [np.asarray(result.x, dtype=float)]
    best = None
    local_ms = 0.0
    for candidate in representatives:
        refined, elapsed = local_refine(problem, candidate, int(config["local_max_nfev"]), str(config["loss"]))
        local_ms += elapsed
        cost = problem.cost(refined.x)
        if best is None or cost < best[0]:
            best = (cost, refined)
    assert best is not None
    packed = pack_result(
        f"de_{strategy}", problem, best[1].x, best[1].success, best[1].message,
        start_calls, global_search_ms=global_ms, local_refine_ms=local_ms,
    )
    packed.metadata.update({"strategy": strategy, "popsize_multiplier": popsize, "maxiter": maxiter, "population_count": int(len(result.population))})
    return packed


def optimize_de_best1bin(problem: FitProblem, config: dict, seed: int, **kwargs) -> InversionResult:
    return optimize_de(problem, config, seed, strategy="best1bin", **kwargs)


def optimize_de_rand1bin(problem: FitProblem, config: dict, seed: int, **kwargs) -> InversionResult:
    return optimize_de(problem, config, seed, strategy="rand1bin", **kwargs)
