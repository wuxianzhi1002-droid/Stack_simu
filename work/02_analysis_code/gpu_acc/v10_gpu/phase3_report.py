"""Build Phase 3 Jacobian machine-readable and Markdown reports."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path

def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))

def build(machine: dict,benchmark: dict) -> dict:
 errors=list(benchmark.get("errors",[]))
 if not benchmark.get("pass",False): errors.append("Jacobian benchmark acceptance failed.")
 if int(machine.get("visible_gpu_count",0)) != 1: errors.append("Visible GPU count is not one.")
 if machine.get("slurm",{}).get("partition") != "gpu_5090": errors.append("Partition is not gpu_5090.")
 if int(benchmark.get("case_count",0)) < 20: errors.append("Fewer than 20 validation parameter sets.")
 if int(benchmark.get("batch_size",0)) != 13: errors.append("GPU perturbation batch is not B=13.")
 return {"phase":"Phase 3 six-parameter center-difference GPU batch Jacobian only","created_utc":datetime.now(timezone.utc).isoformat(),"pass":not errors,"errors":errors,"machine":{"hostname":machine.get("hostname"),"gpu_name":machine.get("visible_devices",[{}])[0].get("name","unknown") if machine.get("visible_devices") else "unknown","visible_gpu_count":machine.get("visible_gpu_count"),"driver_version":machine.get("driver_version"),"cuda_runtime":machine.get("cuda_runtime"),"cuda_driver_api":machine.get("cuda_driver_api"),"cupy_version":machine.get("cupy_version"),"python_version":machine.get("python_version"),"slurm":machine.get("slurm")},"case_count":benchmark.get("case_count"),"batch_size":benchmark.get("batch_size"),"batch_order":benchmark.get("batch_order"),"limits":benchmark.get("limits"),"timing":benchmark.get("timing"),"columns":benchmark.get("columns"),"cpu_reference_replay_columns":benchmark.get("cpu_reference_replay_columns"),"peak_gpu_memory_bytes":benchmark.get("peak_gpu_memory_bytes",0),"peak_gpu_memory_gib":benchmark.get("peak_gpu_memory_bytes",0)/(1024**3),"formal_v10_sha256":benchmark.get("formal_v10_sha256"),"scope_guard":benchmark.get("scope_guard")}

def markdown(report: dict) -> str:
 m=report["machine"]; t=report["timing"]; status="PASS" if report["pass"] else "FAIL"
 lines=["# V10 Phase 3 RTX5090 center-difference batch Jacobian result","",f"- Overall: **{status}**",f"- GPU: {m['gpu_name']}",f"- Cases: {report['case_count']}",f"- Perturbation batch: B={report['batch_size']} ({report['batch_order']})",f"- CPU Jacobian total: {t['cpu_total_s']:.6f} s",f"- GPU Jacobian total: {t['gpu_total_s']:.6f} s",f"- CPU/GPU speedup: {t['speedup']:.2f}x",f"- CPU mean per Jacobian: {t['cpu_mean_s']:.6f} s",f"- GPU mean per Jacobian: {t['gpu_mean_s']*1000.0:.3f} ms",f"- Peak CuPy memory pool: {report['peak_gpu_memory_gib']:.3f} GiB","","| Column | Max relative L2 error | Min cosine similarity | Max abs difference | Undefined zero-reference cases |","|---|---:|---:|---:|---:|"]
 for row in report["columns"]:
  relative="n/a" if row["max_relative_l2_error"] is None else f"{row['max_relative_l2_error']:.3e}"
  cosine="n/a" if row["min_cosine_similarity"] is None else f"{row['min_cosine_similarity']:.12f}"
  lines.append(f"| {row['parameter']} | {relative} | {cosine} | {row['max_abs_difference']:.3e} | {row['undefined_zero_reference_case_count']} |")
 lines.extend(["","## Timing contract","",t["contract"],"","## Scope","",report["scope_guard"]])
 if report["errors"]: lines.extend(["","## Failures",""]+[f"- {e}" for e in report["errors"]])
 return "\n".join(lines)+"\n"

def main() -> None:
 parser=argparse.ArgumentParser(); parser.add_argument("--machine",type=Path,required=True); parser.add_argument("--benchmark",type=Path,required=True); parser.add_argument("--output-json",type=Path,required=True); parser.add_argument("--output-md",type=Path,required=True); args=parser.parse_args(); report=build(load(args.machine),load(args.benchmark)); args.output_json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n"); args.output_md.write_text(markdown(report),encoding="utf-8",newline="\n"); print(json.dumps(report,ensure_ascii=False,indent=2));
 if not report["pass"]: raise SystemExit(2)
if __name__ == "__main__": main()
