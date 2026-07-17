from __future__ import annotations

from time import perf_counter_ns

import numpy as np
from scipy.signal import find_peaks

from model_config import LOWER_BOUNDS, UPPER_BOUNDS, bounds_center
from objective_functions import FitProblem
from optimizer_common import InversionResult, local_refine, pack_result
from spectrum_preprocess import uniform_wavenumber


def fft_air_candidates(problem: FitProblem, count: int) -> np.ndarray:
    sigma, intensity = uniform_wavenumber(problem.measurement)
    centered = (intensity - np.mean(intensity)) * np.hanning(intensity.size)
    pad_size = int(2 ** np.ceil(np.log2(intensity.size)) * 16)
    amplitude = np.abs(np.fft.rfft(centered, n=pad_size))
    spacing = float(sigma[1] - sigma[0])
    opd_um = np.fft.rfftfreq(pad_size, d=spacing)
    air_um = opd_um / 2.0
    mask = (air_um >= LOWER_BOUNDS[0]) & (air_um <= UPPER_BOUNDS[0])
    indices = np.flatnonzero(mask)
    peaks, _ = find_peaks(amplitude[indices])
    ranked = indices[peaks[np.argsort(amplitude[indices][peaks])[::-1]]] if peaks.size else indices[np.argsort(amplitude[indices])[::-1]]
    candidates: list[float] = []
    for index in ranked:
        value = float(air_um[index])
        if all(abs(value - prior) >= 0.03 for prior in candidates):
            candidates.append(value)
            if len(candidates) >= count:
                break
    if not candidates:
        candidates = [float(bounds_center()[0])]
    return np.asarray(candidates)


def optimize_fft_hybrid(problem: FitProblem, config: dict, seed: int, **kwargs) -> InversionResult:
    del seed, kwargs
    start_calls = problem.n_forward_evaluations
    begin = perf_counter_ns()
    air_candidates = fft_air_candidates(problem, int(config["fft_candidates"]))
    coarse_ms = (perf_counter_ns() - begin) / 1e6
    best = None
    local_ms = 0.0
    base = bounds_center()
    per_candidate = max(15, int(config["local_max_nfev"]) // 2)
    for air in air_candidates:
        start = base.copy()
        start[0] = air
        lower = LOWER_BOUNDS.copy()
        upper = UPPER_BOUNDS.copy()
        lower[0] = max(LOWER_BOUNDS[0], air - 0.04)
        upper[0] = min(UPPER_BOUNDS[0], air + 0.04)
        refined, elapsed = local_refine(problem, start, per_candidate, str(config["loss"]), lower, upper)
        local_ms += elapsed
        cost = problem.cost(refined.x)
        if best is None or cost < best[0]:
            best = (cost, refined)
    assert best is not None
    final, elapsed = local_refine(problem, best[1].x, int(config["local_max_nfev"]), str(config["loss"]))
    local_ms += elapsed
    packed = pack_result("fft_hybrid", problem, final.x, final.success, final.message, start_calls, coarse_search_ms=coarse_ms, local_refine_ms=local_ms)
    packed.metadata.update({"fft_zero_padding_factor": 16, "fft_candidates_um": air_candidates.tolist()})
    return packed
