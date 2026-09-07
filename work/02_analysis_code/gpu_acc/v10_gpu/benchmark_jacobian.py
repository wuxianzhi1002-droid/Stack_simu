"""Phase 3 formal CPU versus one-call B=13 GPU Jacobian benchmark."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .backend.v10_source import source_sha256
from .jacobian import BatchedCenterDifferenceJacobian,formal_cpu_jacobian
from .jacobian.metrics import aggregate_case_metrics,column_metrics
from .reference import sha256_file
from .spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend

RELATIVE_LIMIT=2.0e-5
COSINE_MIN=0.999999999
MAX_ABS_LIMIT=1.0e-4

def timed_gpu(jacobian_backend,values,scale,repeat):
 durations=[]; evaluation=None
 for _ in range(repeat):
  jacobian_backend.response_backend.synchronize(); started=time.perf_counter(); current=jacobian_backend.evaluate(values,scale); jacobian_backend.response_backend.synchronize(); durations.append(time.perf_counter()-started)
  if evaluation is None or durations[-1] <= min(durations): evaluation=current
 return min(durations),evaluation

def main() -> None:
 package=Path(__file__).resolve().parent; parser=argparse.ArgumentParser(); parser.add_argument("--reference",type=Path,default=package/"cpu_reference_phase3"/"jacobian_reference.npz"); parser.add_argument("--gpu-repeat",type=int,default=3); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
 if args.gpu_repeat < 1: raise ValueError("gpu-repeat must be positive.")
 with np.load(args.reference,allow_pickle=False) as d:
  reported=np.asarray(d["reported_wavelengths_um"],dtype=np.float64); params=np.asarray(d["parameters"],dtype=np.float64); observed=np.asarray(d["observed"],dtype=np.float64); scale=float(d["residual_scale"].item()); stored=np.asarray(d["jacobians"],dtype=np.float64); config=json.loads(str(d["generator_config_json"].item())); margin=float(d["internal_margin_nm"].item())
 if params.shape[0] < 20: raise ValueError("At least 20 parameter cases are required.")
 cpu=NumpyStrictSpectrometerBackend(reported,config,margin); gpu=CupyStrictSpectrometerBackend(reported,config,margin); gpu_jac=BatchedCenterDifferenceJacobian(gpu)
 gpu_jac.evaluate(params[0],scale)
 cases=[]; gpu_metrics=[]; cpu_reference_metrics=[]; cpu_total=0.0; gpu_total=0.0
 for index,values in enumerate(params):
  started=time.perf_counter(); cpu_jac=formal_cpu_jacobian(cpu.model,values,observed,scale); cpu_s=time.perf_counter()-started
  gpu_s,evaluation=timed_gpu(gpu_jac,values,scale,args.gpu_repeat); current_metrics=column_metrics(cpu_jac,evaluation.jacobian); reference_metrics=column_metrics(stored[index],cpu_jac)
  cpu_total+=cpu_s; gpu_total+=gpu_s; gpu_metrics.append(current_metrics); cpu_reference_metrics.append(reference_metrics)
  cases.append({"case_index":index,"parameters":values.tolist(),"batch_size":13,"cpu_total_s":cpu_s,"gpu_total_s":gpu_s,"speedup":cpu_s/gpu_s,"columns":current_metrics})
 aggregate=aggregate_case_metrics(gpu_metrics); cpu_replay=aggregate_case_metrics(cpu_reference_metrics)
 nan_inf_count=sum(int(not np.isfinite(value)) for case in gpu_metrics for column in case for value in (column["relative_l2_error"], column["cosine_similarity"], column["max_abs_difference"]) if value is not None)
 errors=[]
 if nan_inf_count: errors.append(f"Jacobian metrics contain {nan_inf_count} NaN/Inf values.")
 for row in aggregate:
  if row["max_relative_l2_error"] is not None and row["max_relative_l2_error"] > RELATIVE_LIMIT: errors.append(f"{row['parameter']}: relative error exceeds {RELATIVE_LIMIT}")
  if row["min_cosine_similarity"] is not None and row["min_cosine_similarity"] < COSINE_MIN: errors.append(f"{row['parameter']}: cosine similarity below {COSINE_MIN}")
  if row["max_abs_difference"] > MAX_ABS_LIMIT: errors.append(f"{row['parameter']}: max abs exceeds {MAX_ABS_LIMIT}")
 memory=gpu.memory_stats(); report={"phase":"Phase 3 six-parameter bounded center-difference GPU batch Jacobian","created_utc":datetime.now(timezone.utc).isoformat(),"pass":not errors,"errors":errors,"formal_v10_sha256":source_sha256(),"reference_file":str(args.reference.resolve()),"reference_sha256":sha256_file(args.reference),"case_count":int(params.shape[0]),"parameter_count":6,"batch_size":13,"batch_order":"base + six positive perturbations + six negative perturbations","gpu_repeat":args.gpu_repeat,"limits":{"max_relative_l2_error":RELATIVE_LIMIT,"min_cosine_similarity":COSINE_MIN,"max_abs_difference":MAX_ABS_LIMIT,"zero_reference_l2_norm_max":1.0e-12,"nan_inf_allowed":0},"nan_inf_count":nan_inf_count,"timing":{"cpu_total_s":cpu_total,"gpu_total_s":gpu_total,"speedup":cpu_total/gpu_total,"cpu_mean_s":cpu_total/len(params),"gpu_mean_s":gpu_total/len(params),"contract":"GPU total includes host batch construction, one B=13 H2D+strict response+D2H call, host Jacobian assembly, and ends with explicit synchronize."},"columns":aggregate,"cpu_reference_replay_columns":cpu_replay,"cases":cases,"memory":memory,"peak_gpu_memory_bytes":int(memory.get("memory_pool_total_bytes",0)),"scope_guard":"No strict response, ILS, parameter, residual or scipy least_squares changes. No optimizer integration, JAX, autodiff or Phase 4."}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n"); print(json.dumps(report,ensure_ascii=False,indent=2))
 if not report["pass"]: raise SystemExit(2)
if __name__ == "__main__": main()
