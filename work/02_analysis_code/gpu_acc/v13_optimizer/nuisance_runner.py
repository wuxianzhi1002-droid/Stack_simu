"""V13 one-at-a-time MAP nuisance GPU runner for V12 StackRT data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import source_sha256
from v10_gpu.global_search import robust_objective_batch_device

from .nuisance_backend import (
    NUISANCE_ORDER,
    CupyNuisanceSpectrometerBackend,
    NumpyNuisanceSpectrometerBackend,
)


EXPECTED_V10 = "d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"
PRIORS = {
    "axis_offset_nm": {"sigma": 0.001, "bound": 0.003, "truth_key": "axis_offset_nm"},
    "axis_scale_ppm": {"sigma": 5.0, "bound": 15.0, "truth_key": "axis_scale_ppm"},
    "source_center_drift_nm": {
        "sigma": 0.005,
        "bound": 0.015,
        "truth_key": "source_center_drift_nm",
    },
}


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe(row), ensure_ascii=False) + "\n")
        handle.flush()


def source_hash() -> str:
    return source_sha256()


def bounds(active: str):
    lower5, upper5 = v12.bounds_arrays()
    limit = PRIORS[active]["bound"]
    return np.r_[lower5, -limit], np.r_[upper5, limit]


def population(seed: int, size: int, active: str):
    lo, hi = bounds(active)
    sample = qmc.LatinHypercube(d=6, seed=seed).random(n=size)
    return qmc.scale(sample, lo, hi)


def nuisance_matrix(active: str, values) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.zeros((len(q), 3), dtype=np.float64)
    result[:, NUISANCE_ORDER.index(active)] = q
    return result


def quadratic_prior_residual(z):
    """Residual whose soft_l1 rho equals z**2 exactly."""
    z = np.asarray(z, dtype=np.float64)
    return z * np.sqrt(1.0 + 0.25 * z * z)


def select_candidates(pop, energies, count, active):
    lo, hi = bounds(active)
    span = hi - lo
    result, normalized = [], []
    for index in np.argsort(np.asarray(energies, dtype=float)):
        row = np.asarray(pop[index], dtype=np.float64)
        scaled = (row - lo) / span
        if all(np.linalg.norm(scaled - previous) >= 0.03 for previous in normalized):
            result.append((row, int(index), float(energies[index])))
            normalized.append(scaled)
        if len(result) == count:
            break
    return result


class MapPopulationObjective:
    def __init__(self, backend, observed, scale, indices, angle, config, active):
        self.backend = backend
        self.cp = backend.cp
        self.scale = float(scale)
        self.loss = str(config.loss)
        self.angle = float(angle)
        self.active = active
        self.sigma = float(PRIORS[active]["sigma"])
        self.lo, self.hi = bounds(active)
        self.span = self.hi - self.lo
        self.margin = float(config.boundary_margin_fraction)
        self.weight = float(config.boundary_penalty_weight)
        with backend.device:
            self.observed = self.cp.asarray(np.asarray(observed)[indices], dtype=self.cp.float64)
            self.indices = self.cp.asarray(indices, dtype=self.cp.int64)
            self.lo5 = self.cp.asarray(self.lo[:5], dtype=self.cp.float64)
            self.span5 = self.cp.asarray(self.span[:5], dtype=self.cp.float64)
        self.calls = 0
        self.candidates = 0
        self.batch_sizes = []
        self.wall_s = 0.0

    def _normalize(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1 and x.shape == (6,):
            x = x[None, :]
        elif x.ndim == 2 and x.shape[0] == 6:
            x = x.T
        elif not (x.ndim == 2 and x.shape[1] == 6):
            raise ValueError(f"expected (6,), (6,S), or (S,6), got {x.shape}")
        if not np.all(np.isfinite(x)) or np.any(x < self.lo) or np.any(x > self.hi):
            raise ValueError("invalid MAP candidate population")
        return np.ascontiguousarray(x)

    def __call__(self, values):
        started = time.perf_counter()
        x = self._normalize(values)
        self.calls += 1
        self.candidates += len(x)
        self.batch_sizes.append(len(x))
        cp = self.cp
        with self.backend.device:
            physical = cp.asarray(x[:, :5], dtype=cp.float64)
            angle = cp.full((len(x), 1), self.angle, dtype=cp.float64)
            full = cp.concatenate((physical, angle), axis=1)
            q_host = nuisance_matrix(self.active, x[:, 5])
            q = cp.asarray(q_host, dtype=cp.float64)
            prediction = self.backend.predict_batch_nuisance_device(full, q)
            residual = (prediction[:, self.indices] - self.observed[None, :]) / self.scale
            spectrum = robust_objective_batch_device(residual, self.loss, cp)
            prior = cp.square(q[:, NUISANCE_ORDER.index(self.active)] / self.sigma)
            normalized = (physical - self.lo5[None, :]) / self.span5[None, :]
            distance = cp.minimum(normalized, 1.0 - normalized)
            severity = cp.sum(
                cp.square(cp.clip((self.margin - distance) / self.margin, 0.0, 1.0)), axis=1
            )
            # Preserve V12 semantics: the boundary term regularizes only the
            # spectral objective; the Gaussian MAP prior remains independent.
            total = spectrum * (1.0 + self.weight * severity) + prior
            result = cp.asnumpy(total).astype(np.float64, copy=False)
        self.wall_s += time.perf_counter() - started
        return result

    def profile(self):
        return {
            "objective_calls": self.calls,
            "candidate_evaluations": self.candidates,
            "actual_batch_sizes": self.batch_sizes,
            "min_batch_size": min(self.batch_sizes) if self.batch_sizes else 0,
            "max_batch_size": max(self.batch_sizes) if self.batch_sizes else 0,
            "mean_batch_size": float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
            "objective_wall_time_s": self.wall_s,
        }


class MapCache:
    def __init__(self, backend, observed, scale, angle, active):
        self.backend = backend
        self.observed = np.asarray(observed, dtype=np.float64)
        self.scale = float(scale)
        self.angle = float(angle)
        self.active = active
        self.sigma = float(PRIORS[active]["sigma"])
        self.lo, self.hi = bounds(active)
        self.span = self.hi - self.lo
        self.cached = None
        self.evaluations = self.hits = self.residual_calls = self.jacobian_calls = 0
        self.runtime_s = 0.0

    def to_solver(self, physical):
        return np.clip((np.asarray(physical) - self.lo) / self.span, 0.0, 1.0)

    def from_solver(self, solver):
        return self.lo + np.asarray(solver) * self.span

    def evaluate(self, solver):
        solver = np.ascontiguousarray(np.asarray(solver, dtype=np.float64))
        if solver.shape != (6,):
            raise ValueError("MAP local solver requires six normalized values")
        key = solver.tobytes()
        if self.cached is not None and self.cached[0] == key:
            self.hits += 1
            return self.cached[1]
        steps = np.full(6, 1.0e-6)
        plus = np.repeat(solver[None, :], 6, axis=0)
        minus = np.repeat(solver[None, :], 6, axis=0)
        ii = np.arange(6)
        plus[ii, ii] = np.minimum(1.0, solver + steps)
        minus[ii, ii] = np.maximum(0.0, solver - steps)
        denominator = plus[ii, ii] - minus[ii, ii]
        batch_solver = np.vstack((solver[None, :], plus, minus))
        batch = self.from_solver(batch_solver)
        full = np.c_[batch[:, :5], np.full(13, self.angle)]
        nuisance = nuisance_matrix(self.active, batch[:, 5])
        started = time.perf_counter()
        spectra = self.backend.predict_batch_nuisance(full, nuisance)
        self.runtime_s += time.perf_counter() - started
        spectral = (spectra - self.observed[None, :]) / self.scale
        z = batch[:, 5] / self.sigma
        prior = quadratic_prior_residual(z)[:, None]
        residuals = np.concatenate((spectral, prior), axis=1)
        jacobian = ((residuals[1:7] - residuals[7:13]) / denominator[:, None]).T
        result = {
            "physical": batch[0],
            "full": full[0],
            "nuisance": nuisance[0],
            "prediction": spectra[0],
            "spectral_residual": spectral[0],
            "residual": residuals[0],
            "jacobian": jacobian,
        }
        self.cached = (key, result)
        self.evaluations += 1
        return result

    def residual(self, solver):
        self.residual_calls += 1
        return self.evaluate(solver)["residual"]

    def jacobian(self, solver):
        self.jacobian_calls += 1
        return self.evaluate(solver)["jacobian"]

    def snapshot(self):
        return {
            "local_batch_size": 13,
            "actual_gpu_batch_evaluations": self.evaluations,
            "cache_hits": self.hits,
            "residual_calls": self.residual_calls,
            "jacobian_calls": self.jacobian_calls,
            "gpu_batch_runtime_s": self.runtime_s,
        }


def rank_attempts(attempts, config):
    valid = []
    for source in attempts:
        row = dict(source)
        row["boundary"] = v12.boundary_metrics(row["final_physical"][:5])
        row["quality_grade"], row["quality_class"] = v12._quality(
            {**row, "final_free": row["final_physical"][:5]}, config
        )
        if np.isfinite(row["map_cost"]):
            valid.append(row)
    if not valid:
        raise RuntimeError("no finite MAP local attempt")
    minimum = min(row["map_cost"] for row in valid)
    tolerance = max(abs(minimum) * config.spectrum_equivalence_rtol, 1.0e-12)
    equivalent = [row for row in valid if row["map_cost"] <= minimum + tolerance]
    selected = min(
        equivalent,
        key=lambda row: (
            row["quality_grade"],
            row["boundary"]["severity"],
            len(row["boundary"]["boundary_hits"]),
            row["optimality"],
            row["map_cost"],
            row["call_index"],
        ),
    )
    return selected, {
        "minimum_map_cost": minimum,
        "equivalent_candidate_count": len(equivalent),
        "boundary_preference_changed_selection": selected["call_index"]
        != min(valid, key=lambda row: (row["map_cost"], row["call_index"]))["call_index"],
    }


def matrix_diagnostics(jacobian, active):
    matrix = np.asarray(jacobian, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {
        "parameter_order": list(v12.FREE_PARAMS) + [active],
        "coordinate_system": "normalized_solver_with_exact_quadratic_MAP_prior_residual",
        "singular_values": singular,
        "condition_number": None if singular[-1] == 0 else float(singular[0] / singular[-1]),
        "smallest_singular_value": float(singular[-1]),
    }


def fit_case(measurement, config, active, seed, gpu, cpu):
    observed = np.asarray(measurement["spectrum"], dtype=np.float64)
    scale = v12.v10.robust_scale(observed)
    pop_size = 8 * 6
    init = population(seed, pop_size, active)
    indices = np.arange(0, len(observed), max(1, int(config.global_stride)))
    objective = MapPopulationObjective(
        gpu, observed, scale, indices, measurement["fixed_angle_deg"], config, active
    )
    global_started = time.perf_counter()
    de = differential_evolution(
        objective,
        bounds=list(zip(*bounds(active))),
        strategy="best1bin",
        maxiter=int(config.global_maxiter),
        popsize=8,
        tol=1.0e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        init=init,
        updating="deferred",
        vectorized=True,
        workers=1,
    )
    global_runtime = time.perf_counter() - global_started
    candidates = select_candidates(de.population, de.population_energies, config.multistarts, active)
    if len(candidates) != config.multistarts:
        raise RuntimeError("MAP global search did not produce enough diverse candidates")
    cache = MapCache(gpu, observed, scale, measurement["fixed_angle_deg"], active)
    attempts = []
    local_started = time.perf_counter()
    for call_index, (start, population_index, energy) in enumerate(candidates, 1):
        result = least_squares(
            cache.residual,
            x0=cache.to_solver(start),
            jac=cache.jacobian,
            bounds=(np.zeros(6), np.ones(6)),
            loss=config.loss,
            max_nfev=config.max_nfev,
            x_scale=1.0,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=config.local_gtol,
        )
        physical = cache.from_solver(result.x)
        evaluation = cache.evaluate(result.x)
        pure_cost = 0.5 * v12.v10.robust_objective(evaluation["spectral_residual"], config.loss)
        prior_cost = 0.5 * (physical[5] / PRIORS[active]["sigma"]) ** 2
        attempts.append(
            {
                "call_index": call_index,
                "population_index": population_index,
                "global_total_objective": energy,
                "final_solver": np.asarray(result.x),
                "final_physical": physical,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "map_cost": float(result.cost),
                "pure_spectrum_cost": float(pure_cost),
                "prior_cost": float(prior_cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
            }
        )
    local_runtime = time.perf_counter() - local_started
    selected, ranking = rank_attempts(attempts, config)
    evaluation = cache.evaluate(selected["final_solver"])
    cpu_prediction = cpu.predict_batch_nuisance(
        evaluation["full"][None, :], evaluation["nuisance"][None, :]
    )[0]
    delta = evaluation["prediction"] - cpu_prediction
    closure = {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "pass": bool(np.sqrt(np.mean(delta * delta)) <= 1.0e-10 and np.max(np.abs(delta)) <= 1.0e-8),
    }
    if not closure["pass"]:
        raise RuntimeError(f"V13 MAP CPU/GPU closure failed: {closure}")
    return {
        "selected": selected,
        "ranking": ranking,
        "structural_parameters": evaluation["physical"][:5],
        "full_parameters": evaluation["full"],
        "fitted_nuisance": float(evaluation["physical"][5]),
        "prediction": evaluation["prediction"],
        "exact_rmse": float(np.sqrt(np.mean((evaluation["prediction"] - observed) ** 2))),
        "diagnostics": matrix_diagnostics(evaluation["jacobian"], active),
        "closure": closure,
        "cache": cache.snapshot(),
        "global_profile": objective.profile(),
        "global_runtime_s": global_runtime,
        "local_runtime_s": local_runtime,
        "population_size": pop_size,
    }


def truth_nuisance(path: Path, active: str):
    with np.load(path, allow_pickle=False) as data:
        key = PRIORS[active]["truth_key"]
        if key in data:
            return float(np.asarray(data[key]).item())
        audit = json.loads(str(np.asarray(data["noise_realization_json"]).item()))
        return float(audit.get(key, 0.0))


def finite_stats(values):
    a = np.asarray(list(values), dtype=np.float64)
    return {
        "count": len(a),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def write_outputs(output: Path, report: dict):
    prefix = f"v13_map_{report['configuration']['active_nuisance']}"
    (output / f"{prefix}_results.json").write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "filename", "noise_case", "realization_index", "active_nuisance",
        "nuisance_truth", "nuisance_fit", "nuisance_error", "Air_error_nm",
        "film_MAE_nm", "exact_RMSE", "map_cost", "pure_spectrum_cost", "prior_cost",
        "boundary_hits", "success", "status", "global_runtime_s", "local_runtime_s",
        "closure_rmse", "closure_max_abs",
    ]
    with (output / f"{prefix}_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["cases"]:
            writer.writerow({key: row.get(key) for key in fields})
    summary = report["summary"]
    lines = [
        f"# V13 MAP nuisance: {report['configuration']['active_nuisance']}", "",
        f"- Overall: **{report['overall']}**",
        f"- Scope: 220-580 nm, typical only, fixed measured Angle sigma=0.001 deg",
        f"- Processed/closure: {summary['processed']} / {summary['closure_passed']}",
        f"- Global/local runtime: {summary['global_runtime_s']:.3f} / {summary['local_runtime_s']:.3f} s",
        f"- Population/local batch: {summary['population_size']} / 13",
        f"- Boundary-hit cases: {summary['boundary_cases']}", "",
        "| metric | mean | median | p95 | max |", "|---|---:|---:|---:|---:|",
    ]
    for key, item in report["aggregates"].items():
        lines.append(f"| {key} | {item['mean']:.6g} | {item['median']:.6g} | {item['p95']:.6g} | {item['max']:.6g} |")
    (output / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--nuisance", choices=sorted(PRIORS), required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--noise-case", type=str)
    parser.add_argument("--require-typical-count", action="store_true")
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--max-nfev", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    if source_hash() != EXPECTED_V10:
        raise RuntimeError("formal V10 source hash changed")
    root = args.input_dir.resolve()
    paths = sorted(root.glob("static_spectrum_*_typical_*.npz"), key=lambda p: p.name)
    if args.require_typical_count and len(paths) != 100:
        raise RuntimeError(f"expected 100 typical NPZ, found {len(paths)}")
    if args.noise_case:
        token = f"static_spectrum_{args.noise_case}_"
        paths = [path for path in paths if path.name.startswith(token)]
    paths = paths[: args.count]
    if len(paths) != args.count:
        raise RuntimeError(f"requested {args.count}, found {len(paths)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = v12.FitConfig(
        input_dir=str(root), wavelength_min_nm=220.0, wavelength_max_nm=580.0,
        global_forward_model="full_ils", global_popsize=8,
        global_maxiter=args.global_maxiter, multistarts=8, max_nfev=args.max_nfev,
        workers=1, random_seed=20260902,
    )
    first = v12.load_fit_input(paths[0], config)
    gpu = CupyNuisanceSpectrometerBackend(
        first["wavelengths_um"], first["generator_config"],
        first["metadata"]["internal_wavelength_margin_nm"],
    )
    cpu = NumpyNuisanceSpectrometerBackend(
        first["wavelengths_um"], first["generator_config"],
        first["metadata"]["internal_wavelength_margin_nm"],
    )
    progress = output / f"v13_map_{args.nuisance}_progress.jsonl"
    rows = []
    started_all = time.perf_counter()
    for index, path in enumerate(paths, 1):
        measurement = v12.load_fit_input(path, config)
        if measurement["metadata"]["noise_level"] != "typical":
            raise RuntimeError(f"non-typical case entered V13 Stage 3: {path.name}")
        case_started = time.perf_counter()
        fit = fit_case(measurement, config, args.nuisance, 20260902 + index * 1009, gpu, cpu)
        truth, _ = v12.load_evaluation_truth(path)
        q_truth = truth_nuisance(path, args.nuisance)
        structural = np.asarray(fit["structural_parameters"])
        air_error = (structural[0] - truth["Air"]) * 1000.0
        film_mae = float(np.mean([abs(structural[i] - truth[name]) for i, name in enumerate(v12.FREE_PARAMS[1:], 1)]))
        selected = fit["selected"]
        row = {
            "index": index,
            "filename": path.name,
            "noise_case": measurement["metadata"]["noise_case"],
            "realization_index": measurement["metadata"]["realization_index"],
            "active_nuisance": args.nuisance,
            "nuisance_truth": q_truth,
            "nuisance_fit": fit["fitted_nuisance"],
            "nuisance_error": fit["fitted_nuisance"] - q_truth,
            "Air_error_nm": air_error,
            "absolute_Air_error_nm": abs(air_error),
            "film_MAE_nm": film_mae,
            "exact_RMSE": fit["exact_rmse"],
            "map_cost": selected["map_cost"],
            "pure_spectrum_cost": selected["pure_spectrum_cost"],
            "prior_cost": selected["prior_cost"],
            "boundary_hits": ";".join(selected["boundary"]["boundary_hits"]),
            "success": selected["success"],
            "status": selected["status"],
            "global_runtime_s": fit["global_runtime_s"],
            "local_runtime_s": fit["local_runtime_s"],
            "case_runtime_s": time.perf_counter() - case_started,
            "closure_rmse": fit["closure"]["rmse"],
            "closure_max_abs": fit["closure"]["max_abs"],
            "closure_pass": fit["closure"]["pass"],
            "ranking": fit["ranking"],
            "diagnostics": fit["diagnostics"],
            "cache": fit["cache"],
            "global_profile": fit["global_profile"],
        }
        rows.append(row)
        append_jsonl(progress, row)
        print(
            f"[{index}/{len(paths)}] {path.name} q={row['nuisance_fit']:.6g} "
            f"Air={row['Air_error_nm']:.4g}nm closure=PASS", flush=True
        )
    total = time.perf_counter() - started_all
    summary = {
        "processed": len(rows),
        "terminated": sum(bool(row["success"]) and int(row["status"]) > 0 for row in rows),
        "closure_passed": sum(row["closure_pass"] for row in rows),
        "boundary_cases": sum(bool(row["boundary_hits"]) for row in rows),
        "population_size": 48,
        "local_batch_size": 13,
        "global_runtime_s": float(sum(row["global_runtime_s"] for row in rows)),
        "local_runtime_s": float(sum(row["local_runtime_s"] for row in rows)),
        "total_runtime_s": total,
        "mean_case_runtime_s": float(np.mean([row["case_runtime_s"] for row in rows])),
        "throughput_cases_per_min": len(rows) / total * 60.0,
        "peak_gpu_memory_gib": gpu.memory_stats().get("memory_pool_total_bytes", 0) / 1024**3,
    }
    passed = len(rows) == args.count and summary["closure_passed"] == args.count
    report = {
        "overall": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            **asdict(config),
            "stage": 3,
            "wavelength_nm": [220.0, 580.0],
            "noise_level": "typical",
            "noise_case_filter": args.noise_case,
            "angle_mode": "fixed_independent_measurement",
            "angle_sigma_deg": 0.001,
            "active_nuisance": args.nuisance,
            "nuisance_prior": PRIORS[args.nuisance],
            "nuisance_parameter_order": list(NUISANCE_ORDER),
            "map_contract": "robust spectrum objective + exact quadratic Gaussian prior",
            "population_size": 48,
            "local_batch_size": 13,
            "vectorized": True,
            "updating": "deferred",
        },
        "summary": summary,
        "aggregates": {
            "absolute_Air_error_nm": finite_stats(row["absolute_Air_error_nm"] for row in rows),
            "film_MAE_nm": finite_stats(row["film_MAE_nm"] for row in rows),
            "exact_RMSE": finite_stats(row["exact_RMSE"] for row in rows),
            "absolute_nuisance_error": finite_stats(abs(row["nuisance_error"]) for row in rows),
        },
        "formal_v10_sha256": source_hash(),
        "input_dir": str(root),
        "machine": json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,
        "cases": rows,
        "scope_guard": "V13 Stage 3 only; 220-580 nm; typical; fixed measured Angle sigma=0.001 deg; one nuisance at a time; V12/V10 unchanged.",
    }
    write_outputs(output, report)
    print(json.dumps({"overall": report["overall"], "processed": len(rows), "runtime_s": total}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
