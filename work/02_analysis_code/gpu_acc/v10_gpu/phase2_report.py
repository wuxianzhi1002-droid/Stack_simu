"""Build Phase 2 strict spectrometer-response machine and Markdown reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_BATCHES=(1,7,13,32,64)
RMSE_LIMIT=1.0e-10
MAX_ABS_LIMIT=1.0e-8


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_batch(rows: list[dict]) -> dict[int,dict]:
    return {int(row["batch_size"]):row for row in rows}


def build_report(machine: dict, benchmark: dict, baseline: dict) -> dict:
    errors=[]
    if int(machine.get("visible_gpu_count",0)) != 1:
        errors.append("CuPy visible GPU count is not exactly one.")
    if machine.get("slurm",{}).get("partition") != "gpu_5090":
        errors.append("Job did not run in gpu_5090.")
    cpu=benchmark.get("backends",{}).get("cpu_numpy_spectrometer",{})
    gpu=benchmark.get("backends",{}).get("cuda_cupy_spectrometer",{})
    old=baseline.get("backends",{}).get("cpu_numpy_spectrometer",{})
    if cpu.get("status") != "ok": errors.append("Phase 2 CPU backend did not complete.")
    if gpu.get("status") != "ok": errors.append("Phase 2 CuPy backend did not complete.")
    cpu_rows=rows_by_batch(cpu.get("rows",[])); gpu_rows=rows_by_batch(gpu.get("rows",[])); old_rows=rows_by_batch(old.get("rows",[])); validation=rows_by_batch(benchmark.get("cpu_gpu_validation",[]))
    batches=[]
    for b in EXPECTED_BATCHES:
        if any(b not in rows for rows in (cpu_rows,gpu_rows,old_rows,validation)):
            errors.append(f"Missing Phase 2 result for B={b}.")
            continue
        g=gpu_rows[b]; v=validation[b]; rmse=float(v["rmse"]); max_abs=float(v["max_abs"]); nan_inf=int(v.get("nan_inf_count",0))+int(g.get("nan_count",0))+int(g.get("inf_count",0))
        if rmse > RMSE_LIMIT: errors.append(f"B={b}: RMSE {rmse} exceeds {RMSE_LIMIT}.")
        if max_abs > MAX_ABS_LIMIT: errors.append(f"B={b}: max abs {max_abs} exceeds {MAX_ABS_LIMIT}.")
        if nan_inf: errors.append(f"B={b}: NaN/Inf count is nonzero.")
        total=float(g["total_s"])
        batches.append({"batch_size":b,"gpu_response_s":float(g["kernel_s"]),"gpu_total_s":total,"gpu_throughput_candidates_per_s":float(g["candidates_per_s"]),"existing_cpu_total_s":float(old_rows[b]["total_s"]),"speedup_vs_existing_cpu":float(old_rows[b]["total_s"])/total,"rmse":rmse,"max_abs":max_abs,"nan_inf_count":nan_inf})
    memory=gpu.get("memory",{}); peak=int(gpu.get("peak_gpu_memory_bytes",memory.get("memory_pool_total_bytes",0)))
    if peak <= 0: errors.append("Peak CuPy memory pool was not recorded.")
    return {"phase":"Phase 2 strict GPU ILS / spectrometer response only","created_utc":datetime.now(timezone.utc).isoformat(),"pass":not errors,"errors":errors,"limits":{"rmse_max":RMSE_LIMIT,"max_abs_max":MAX_ABS_LIMIT,"nan_inf_allowed":0},"machine":{"hostname":machine.get("hostname"),"gpu_name":machine.get("visible_devices",[{}])[0].get("name","unknown") if machine.get("visible_devices") else "unknown","visible_gpu_count":machine.get("visible_gpu_count"),"driver_version":machine.get("driver_version"),"cuda_runtime":machine.get("cuda_runtime"),"cuda_driver_api":machine.get("cuda_driver_api"),"cupy_version":machine.get("cupy_version"),"python_version":machine.get("python_version"),"slurm":machine.get("slurm")},"peak_gpu_memory_bytes":peak,"peak_gpu_memory_gib":peak/(1024**3),"batches":batches,"benchmark_source":benchmark.get("reference_file"),"formal_v10_sha256":benchmark.get("formal_v10_sha256"),"timing_contract":"Every interval ends with explicit synchronize; total_s is contiguous H2D + strict response + D2H.","scope_guard":"Strict TMM plus nominal source, throughput, QE, photon weighting, Gaussian ILS, fixed reported-axis sampling and sample/reference exposure normalization only. No Jacobian, scipy least_squares changes, optimizer integration, JAX, autodiff, global search or multi-GPU."}


def markdown(report: dict) -> str:
    m=report["machine"]; status="PASS" if report["pass"] else "FAIL"
    lines=["# V10 Phase 2 RTX5090 strict spectrometer-response result","",f"- Overall: **{status}**",f"- GPU: {m['gpu_name']}",f"- Visible GPU count: {m['visible_gpu_count']}",f"- NVIDIA driver: {m['driver_version']}",f"- CUDA runtime: {m['cuda_runtime']}",f"- CuPy: {m['cupy_version']}",f"- Peak CuPy memory pool: {report['peak_gpu_memory_gib']:.3f} GiB","", "| B | GPU response (s) | End-to-end (s) | Throughput (candidate/s) | Speedup vs CPU | RMSE | Max abs | NaN/Inf |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["batches"]:
        lines.append("| {batch_size} | {gpu_response_s:.6f} | {gpu_total_s:.6f} | {gpu_throughput_candidates_per_s:.3f} | {speedup_vs_existing_cpu:.2f}x | {rmse:.3e} | {max_abs:.3e} | {nan_inf_count} |".format(**row))
    lines.extend(["","## Timing", "", report["timing_contract"], "", "## Acceptance", "", f"- RMSE <= {RMSE_LIMIT:.1e}", f"- max abs <= {MAX_ABS_LIMIT:.1e}", "- NaN/Inf = 0", "- Exactly one GPU visible", "- Partition = gpu_5090", "", "## Scope", "", report["scope_guard"]])
    if report["errors"]: lines.extend(["","## Failures",""]+[f"- {e}" for e in report["errors"]])
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--machine",type=Path,required=True); parser.add_argument("--benchmark",type=Path,required=True); parser.add_argument("--baseline",type=Path,required=True); parser.add_argument("--output-json",type=Path,required=True); parser.add_argument("--output-md",type=Path,required=True); args=parser.parse_args()
    report=build_report(load_json(args.machine),load_json(args.benchmark),load_json(args.baseline))
    args.output_json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n"); args.output_md.write_text(markdown(report),encoding="utf-8",newline="\n"); print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report["pass"]: raise SystemExit(2)


if __name__ == "__main__": main()
