from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

import numpy as np
import pandas as pd

from model_config import LOWER_BOUNDS, UPPER_BOUNDS
from tmm_stackrt_matched import StackRTMatchedTMM


def benchmark_forward_backends(wavelengths_um: np.ndarray, seed: int, candidate_count: int = 32, repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(candidate_count, 5))
    rows: list[dict] = []
    reference = None
    for repeat in range(repeats):
        begin = perf_counter_ns()
        uncached = np.vstack([StackRTMatchedTMM(wavelengths_um).reflectance(candidate) for candidate in candidates])
        elapsed = (perf_counter_ns() - begin) / 1e6
        rows.append({"backend": "uncached_model_per_candidate", "repeat": repeat, "candidate_count": candidate_count, "elapsed_ms": elapsed, "candidates_per_second": 1000.0 * candidate_count / elapsed, "max_abs_vs_reference": 0.0})

        model = StackRTMatchedTMM(wavelengths_um)
        begin = perf_counter_ns()
        sequential = np.vstack([model.reflectance(candidate) for candidate in candidates])
        elapsed = (perf_counter_ns() - begin) / 1e6
        reference = sequential
        rows[-1]["max_abs_vs_reference"] = float(np.max(np.abs(uncached - reference)))
        rows.append({"backend": "single_thread_loop", "repeat": repeat, "candidate_count": candidate_count, "elapsed_ms": elapsed, "candidates_per_second": 1000.0 * candidate_count / elapsed, "max_abs_vs_reference": 0.0})

        model = StackRTMatchedTMM(wavelengths_um)
        begin = perf_counter_ns()
        vectorized = model.reflectance_batch(candidates)
        elapsed = (perf_counter_ns() - begin) / 1e6
        rows.append({"backend": "numpy_vectorized_batch", "repeat": repeat, "candidate_count": candidate_count, "elapsed_ms": elapsed, "candidates_per_second": 1000.0 * candidate_count / elapsed, "max_abs_vs_reference": float(np.max(np.abs(vectorized - reference)))})

        model = StackRTMatchedTMM(wavelengths_um)
        begin = perf_counter_ns()
        with ThreadPoolExecutor(max_workers=4) as pool:
            threaded = np.vstack(list(pool.map(model.reflectance, candidates)))
        elapsed = (perf_counter_ns() - begin) / 1e6
        rows.append({"backend": "scipy_workers_style_thread_map_4", "repeat": repeat, "candidate_count": candidate_count, "elapsed_ms": elapsed, "candidates_per_second": 1000.0 * candidate_count / elapsed, "max_abs_vs_reference": float(np.max(np.abs(threaded - reference)))})
    return pd.DataFrame(rows)
