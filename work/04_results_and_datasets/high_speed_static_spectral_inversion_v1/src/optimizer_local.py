from __future__ import annotations

import numpy as np

from model_config import bounds_center
from objective_functions import FitProblem
from optimizer_common import InversionResult, local_refine, pack_result


def optimize_local(
    problem: FitProblem,
    config: dict,
    seed: int,
    initial: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> InversionResult:
    del seed
    start_calls = problem.n_forward_evaluations
    start = bounds_center() if initial is None else np.asarray(initial, dtype=float)
    result, local_ms = local_refine(problem, start, int(config["local_max_nfev"]), str(config["loss"]), lower, upper)
    return pack_result("local", problem, result.x, result.success, result.message, start_calls, local_refine_ms=local_ms)
