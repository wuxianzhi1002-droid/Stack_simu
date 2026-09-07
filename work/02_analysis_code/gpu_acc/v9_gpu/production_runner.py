"""Production V9 GPU local-fitting runner; formal V9 source remains frozen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares as scipy_least_squares

from .backend import CupyV9StrictTMMBackend
from .cache import ExactV9BatchCache, physical_to_solver
from .source import EXPECTED_SHA256, load_v9_module, source_path, source_sha256

v9 = load_v9_module()


class GpuLocalContext:
    """Routes only formal V9 local LS callbacks and final Jacobian to CuPy."""

    def __init__(self, backend: CupyV9StrictTMMBackend):
        self.backend = backend
        self.cache: ExactV9BatchCache | None = None
        self.case_snapshots: list[dict[str, Any]] = []
        self.backend_reinitializations = 0
        self._case_resident_before: dict[str, Any] | None = None
        self.case_resident_reused: list[bool] = []

    def begin_case(self, measurement: dict) -> None:
        axis = np.asarray(measurement["wavelengths_um"], dtype=np.float64)
        if not np.array_equal(axis, self.backend.wavelengths_um):
            self.backend = CupyV9StrictTMMBackend(axis, device_id=self.backend.device_id)
            self.backend_reinitializations += 1
        self._case_resident_before = self.backend.resident_identity()
        self.cache = ExactV9BatchCache(
            self.backend,
            measurement["spectrum"],
            v9.robust_scale(measurement["spectrum"]),
        )

    def least_squares(self, fun, x0, *args, **kwargs):
        if self.cache is None:
            raise RuntimeError("GPU local context is not initialized")
        if "jac" in kwargs:
            raise RuntimeError("Formal V9 unexpectedly supplied a Jacobian")
        return scipy_least_squares(
            self.cache.residual,
            x0,
            *args,
            jac=self.cache.jacobian,
            **kwargs,
        )

    def approximate_jacobian(self, residual_fn, physical_values):
        if self.cache is None:
            raise RuntimeError("GPU local context is not initialized")
        return self.cache.evaluate(physical_to_solver(physical_values)).physical_jacobian

    def finish_case(self) -> dict[str, Any]:
        if self.cache is None:
            return {}
        item = self.cache.snapshot()
        resident_reused = self._case_resident_before == self.backend.resident_identity()
        item["resident_constants_reused_within_case"] = resident_reused
        self.case_resident_reused.append(resident_reused)
        self.case_snapshots.append(item)
        self.cache = None
        self._case_resident_before = None
        return item


class FrozenPatches:
    def __init__(self, context: GpuLocalContext):
        self.context = context
        self.original_least_squares = v9.least_squares
        self.original_approximate_jacobian = v9.approximate_jacobian

    def __enter__(self):
        v9.least_squares = self.context.least_squares
        v9.approximate_jacobian = self.context.approximate_jacobian
        return self

    def __exit__(self, exc_type, exc, traceback):
        v9.least_squares = self.original_least_squares
        v9.approximate_jacobian = self.original_approximate_jacobian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal V9 CPU global + CuPy local inversion")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=301)
    parser.add_argument("--pattern", default="static_spectrum_*.npz")
    parser.add_argument("--machine", type=Path, default=None)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=20260810)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(item), ensure_ascii=False) + "\n")


def closure_metrics(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    cpu = np.asarray(cpu, dtype=np.float64)
    gpu = np.asarray(gpu, dtype=np.float64)
    difference = gpu - cpu
    nan_inf = int(np.size(cpu) - np.count_nonzero(np.isfinite(cpu)))
    nan_inf += int(np.size(gpu) - np.count_nonzero(np.isfinite(gpu)))
    if nan_inf:
        rmse = float("inf")
        max_abs = float("inf")
    else:
        rmse = float(np.sqrt(np.mean(difference * difference)))
        max_abs = float(np.max(np.abs(difference)))
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "nan_inf_count": nan_inf,
        "pass": bool(rmse <= 1.0e-10 and max_abs <= 1.0e-8 and nan_inf == 0),
    }


def aggregate_cache(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "residual_calls", "jacobian_calls", "actual_gpu_batch_evaluations",
        "cache_hits", "cache_misses", "gpu_batch_runtime_s",
        "residual_callback_runtime_s", "jacobian_callback_runtime_s",
    )
    return {key: sum(float(item.get(key, 0)) for item in items) for key in keys}


def build_formal_outputs(
    output_dir: Path,
    inputs: list[Path],
    config: Any,
    rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    global_summaries: list[dict[str, Any]],
    representatives: dict[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No inversion row was produced: {failures}")
    table.to_csv(output_dir / "fit_results.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(attempts).to_csv(
        output_dir / "multistart_results.csv", index=False,
        encoding="utf-8-sig", float_format="%.10g",
    )
    summary = v9.summarize_results(table)
    summary.to_csv(output_dir / "case_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    successful = table[table["success"]].copy()
    if successful.empty:
        level_summary = pd.DataFrame(columns=[
            "noise_level", "runs", "cavity_mae_nm", "film_mae_nm",
            "angle_mae_deg", "fit_runtime_mean_s",
        ])
    else:
        level_summary = successful.groupby("noise_level", dropna=False).agg(
            runs=("success", "size"),
            cavity_mae_nm=("cavity_abs_error_nm", "mean"),
            film_mae_nm=("film_mae_nm", "mean"),
            angle_mae_deg=("angle_abs_error_deg", "mean"),
            fit_runtime_mean_s=("fit_runtime_s", "mean"),
        ).reset_index()
    level_summary.to_csv(output_dir / "level_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    with (output_dir / "global_search_summary.jsonl").open("w", encoding="utf-8") as handle:
        for item in global_summaries:
            handle.write(json.dumps(json_safe(item), ensure_ascii=False) + "\n")
    pd.DataFrame(audits).to_csv(output_dir / "sampling_audit.csv", index=False, encoding="utf-8-sig")
    from matplotlib.axes import Axes
    original_boxplot = Axes.boxplot
    def compatible_boxplot(self, *args, **kwargs):
        if "labels" in kwargs and "tick_labels" not in kwargs:
            kwargs["tick_labels"] = kwargs.pop("labels")
        return original_boxplot(self, *args, **kwargs)
    Axes.boxplot = compatible_boxplot
    try:
        plots = v9.save_plots(output_dir, table, summary, representatives)
    finally:
        Axes.boxplot = original_boxplot
    formal_report = {
        "version": v9.VERSION,
        "execution": "formal V9 CPU global search + CuPy strict local residual and B=13 Jacobian",
        "input_count": len(inputs),
        "result_count": len(table),
        "successful_count": int(table["success"].sum()),
        "config": asdict(config),
        "parameters": v9.PARAMS,
        "bounds": v9.BOUNDS,
        "formal_v9_sha256": source_sha256(),
        "truth_usage_policy": "Truth loaded only after fitting and result ranking.",
        "failures": failures,
        "plots": plots,
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(json_safe(formal_report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# V9 single-frame six-parameter GPU-accelerated inversion", "",
        f"- Inputs: {len(inputs)}",
        f"- Converged: {int(table['success'].sum())}",
        "- Frozen formal V9 model and CPU differential-evolution global search.",
        "- CuPy strict TMM residual plus B=13 center-difference Jacobian for local fitting.",
        "- Truth is read only after optimization and ranking.", "",
        "## Case summary", "", v9.dataframe_to_markdown(summary), "",
        "## Scope", "",
        "- ILS remains disabled as required by V9.",
        "- No changes to parameterization, bounds, residual scaling, loss, tolerances, or rank rules.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table, summary, plots


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    inputs = sorted(input_dir.glob(args.pattern))
    if len(inputs) != int(args.count):
        raise RuntimeError(f"Expected exactly {args.count} V9 NPZ files, found {len(inputs)} in {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "v9_gpu_progress.jsonl"
    source_before = source_sha256()
    if source_before != EXPECTED_SHA256:
        raise RuntimeError(f"Formal V9 source hash mismatch: {source_before}")

    config = v9.FitConfig(
        input_dir=str(input_dir), workers=1,
        global_popsize=int(args.global_popsize),
        global_maxiter=int(args.global_maxiter),
        multistarts=int(args.multistarts),
        max_nfev=int(args.max_nfev),
        random_seed=int(args.random_seed),
    )
    first_measurement = v9.load_fit_input(inputs[0], config)
    v9.validate_sampling(first_measurement, config)
    init_started = time.perf_counter()
    backend = CupyV9StrictTMMBackend(first_measurement["wavelengths_um"])
    backend_init_s = time.perf_counter() - init_started
    resident_before = backend.resident_identity()
    warm_values = np.asarray([(v9.BOUNDS[name][0] + v9.BOUNDS[name][1]) / 2.0 for name in v9.PARAMS])
    warm_started = time.perf_counter()
    warm_gpu = backend.predict_batch(np.repeat(warm_values[None, :], 13, axis=0))[0]
    backend.synchronize()
    warmup_s = time.perf_counter() - warm_started
    warm_cpu = v9.tmm_reflectance(first_measurement["wavelengths_um"], warm_values)
    warm_closure = closure_metrics(warm_cpu, warm_gpu)
    if not warm_closure["pass"]:
        raise RuntimeError(f"V9 CPU/GPU warmup closure failed: {warm_closure}")

    context = GpuLocalContext(backend)
    rows: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sampling_audits: list[dict[str, Any]] = []
    global_summaries: list[dict[str, Any]] = []
    representatives: dict[str, Any] = {}
    closures: list[dict[str, Any]] = []
    case_timings: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    with FrozenPatches(context):
        for index, npz_path in enumerate(inputs):
            case_started = time.perf_counter()
            try:
                measurement = v9.load_fit_input(npz_path, config)
                audit = v9.validate_sampling(measurement, config)
                context.begin_case(measurement)
                fit = v9.fit_measurement(measurement, config, config.random_seed + index * 1009)
                cache_stats = context.finish_case()
                truth, _noise_audit = v9.load_evaluation_truth(npz_path)
                row = v9.result_row(measurement, fit, truth)
                rows.append(row)
                all_attempts.extend(v9.attempt_rows(measurement, fit, truth))
                sampling_audits.append({"input_npz": str(npz_path), **audit})
                global_summaries.append({
                    "input_npz": str(npz_path),
                    "noise_case": measurement["metadata"]["noise_case"],
                    **fit.get("global_summary", {}),
                })
                closure = None
                if fit["success"]:
                    cpu_spectrum = v9.tmm_reflectance(measurement["wavelengths_um"], fit["x"])
                    gpu_spectrum = context.backend.predict(fit["x"])
                    closure = closure_metrics(cpu_spectrum, gpu_spectrum)
                    closure.update({"index": index, "input_npz": npz_path.name})
                    closures.append(closure)
                    if not closure["pass"]:
                        raise RuntimeError(f"Final CPU/GPU closure failed: {closure}")
                    noise_case = measurement["metadata"]["noise_case"]
                    if noise_case not in representatives:
                        representatives[noise_case] = {
                            "wavelengths_um": measurement["wavelengths_um"],
                            "observed": measurement["spectrum"],
                            "fitted": fit["fitted_spectrum"],
                        }
                elapsed = time.perf_counter() - case_started
                timing = {
                    "index": index, "input_npz": npz_path.name,
                    "success": bool(fit["success"]), "wall_runtime_s": elapsed,
                    **cache_stats,
                }
                case_timings.append(timing)
                append_jsonl(progress_path, {
                    "event": "case_complete", **timing,
                    "closure": closure,
                })
                print(
                    f"[{index + 1}/{len(inputs)}] {npz_path.name} "
                    f"success={fit['success']} wall={elapsed:.3f}s", flush=True
                )
            except Exception as exc:
                if context.cache is not None:
                    context.finish_case()
                elapsed = time.perf_counter() - case_started
                item = {"index": index, "input": str(npz_path), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(item)
                append_jsonl(progress_path, {"event": "case_failure", **item, "wall_runtime_s": elapsed})
                print(f"ERROR [{index + 1}/{len(inputs)}] {npz_path}: {item['error']}", flush=True)


    total_runtime_s = time.perf_counter() - total_started
    table, summary, plots = build_formal_outputs(
        output_dir, inputs, config, rows, all_attempts, sampling_audits,
        global_summaries, representatives, failures,
    )
    source_after = source_sha256()
    resident_after = context.backend.resident_identity()
    cache_totals = aggregate_cache(context.case_snapshots)
    memory = context.backend.memory_stats()
    successful_count = int(table["success"].sum())
    closure_pass = bool(closures) and all(item["pass"] for item in closures)
    resident_reused = bool(context.case_resident_reused) and all(context.case_resident_reused)
    accepted = bool(
        not failures
        and len(table) == len(inputs)
        and successful_count > 0
        and len(closures) == successful_count
        and closure_pass
        and source_after == source_before == EXPECTED_SHA256
        and resident_reused
    )
    machine = {}
    if args.machine and args.machine.is_file():
        machine = json.loads(args.machine.read_text(encoding="utf-8"))
    result = {
        "overall": "PASS" if accepted else "FAIL",
        "scope": "V9 formal CPU global search plus GPU strict local residual/B=13 Jacobian",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_count": len(inputs),
        "result_count": len(table),
        "successful_count": successful_count,
        "nonconverged_count": int(len(table) - successful_count),
        "runtime_failure_count": len(failures),
        "formal_v9_path": str(source_path()),
        "formal_v9_sha256_before": source_before,
        "formal_v9_sha256_after": source_after,
        "formal_v9_unchanged": source_before == source_after == EXPECTED_SHA256,
        "backend": backend.backend_name,
        "backend_init_s": backend_init_s,
        "warmup_s": warmup_s,
        "warmup_closure": warm_closure,
        "resident_constants_reused_within_each_npz": resident_reused,
        "wavelength_backend_reinitializations": context.backend_reinitializations,
        "resident_constants_last_case": resident_after,
        "total_runtime_s": total_runtime_s,
        "average_wall_runtime_per_npz_s": total_runtime_s / len(inputs),
        "throughput_npz_per_hour": len(inputs) / total_runtime_s * 3600.0,
        "cache_totals": cache_totals,
        "closure": {
            "tested_count": len(closures),
            "all_pass": closure_pass,
            "max_rmse": max((item["rmse"] for item in closures), default=None),
            "max_abs": max((item["max_abs"] for item in closures), default=None),
            "nan_inf_count": sum(item["nan_inf_count"] for item in closures),
            "thresholds": {"rmse": 1.0e-10, "max_abs": 1.0e-8, "nan_inf": 0},
        },
        "memory": memory,
        "machine": machine,
        "failures": failures,
        "plots": plots,
        "cpu_speedup": None,
        "cpu_speedup_note": "No paired full V9 CPU production run was executed; no unsupported speedup is claimed.",
        "configuration": asdict(config),
    }
    (output_dir / "v9_gpu_results.json").write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(case_timings).to_csv(
        output_dir / "v9_gpu_table.csv", index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    gpu_name = machine.get("gpu_name", memory.get("device_name", "unknown"))
    summary_lines = [
        "# V9 GPU-accelerated 301-NPZ inversion summary", "",
        f"- Overall: **{result['overall']}**",
        f"- GPU: {gpu_name}",
        f"- Inputs/results/converged: {len(inputs)}/{len(table)}/{successful_count}",
        f"- Runtime failures: {len(failures)}",
        f"- Total runtime: {total_runtime_s:.3f} s",
        f"- Mean wall time per NPZ: {total_runtime_s / len(inputs):.3f} s",
        f"- Throughput: {len(inputs) / total_runtime_s * 3600.0:.3f} NPZ/hour",
        f"- GPU B=13 batch runtime: {cache_totals['gpu_batch_runtime_s']:.3f} s",
        f"- Resident constants reused within every NPZ: {resident_reused}",
        f"- Formal V9 unchanged: {source_after == source_before == EXPECTED_SHA256}",
        f"- CPU/GPU closure count/all pass: {len(closures)}/{closure_pass}",
        f"- Maximum closure RMSE: {result['closure']['max_rmse']}",
        f"- Maximum closure absolute difference: {result['closure']['max_abs']}",
        f"- Closure NaN/Inf count: {result['closure']['nan_inf_count']}",
        f"- CuPy pool high/current allocation: {memory.get('memory_pool_total_bytes', 0) / 2**30:.3f} GiB",
        "", "## Accuracy summary", "", v9.dataframe_to_markdown(summary), "",
        "## Execution contract", "",
        "- Formal V9 source and physical model were not modified.",
        "- Differential-evolution global search remained on CPU.",
        "- Local residual used CuPy strict TMM; constants stayed resident within each NPZ optimization.",
        "- A changed estimated calibrated wavelength axis triggers a safe backend rebuild between NPZ files.",
        "- Each Jacobian used one B=13 batch.",
        "- SciPy least_squares remained on CPU with frozen V9 bounds, mapping, loss, scaling and tolerances.",
        "- No ILS, JAX, autodiff, multi-GPU, or historical V10 qualification replay.",
        "- No paired full CPU production run was executed, so this report does not claim a V9 CPU/GPU speedup.",
    ]
    (output_dir / "v9_gpu_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall": result["overall"], "output_dir": str(output_dir),
        "input_count": len(inputs), "successful_count": successful_count,
        "total_runtime_s": total_runtime_s,
    }, ensure_ascii=False), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
