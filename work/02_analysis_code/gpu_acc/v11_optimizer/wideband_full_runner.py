"""Full 401-case V11 q=theta^2 GPU inversion for one matched wideband dataset."""
from __future__ import annotations
import argparse,csv,json,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from v10_gpu.phase6_full_runner import exact_backend_contract
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend
from .q_cache import ExactQBatchCache,physical_to_solver_q
from .ranking import boundary_hits,select_candidate
from .stage12_runner import EXPECTED,NAMES,closure,diagnostics,errors,q_attempt,safe

def finite_stats(values):
    a=np.asarray([float(v) for v in values if np.isfinite(v)],dtype=np.float64)
    if not len(a):return {"count":0,"mean":None,"median":None,"p95":None,"max":None}
    return {"count":int(len(a)),"mean":float(np.mean(a)),"median":float(np.median(a)),"p95":float(np.quantile(a,0.95)),"max":float(np.max(a))}

def aggregate_errors(rows):
    return {
        "absolute_Air_error_nm":finite_stats(abs(r["errors"]["Air_error_nm"]) for r in rows),
        "film_MAE_nm":finite_stats(r["errors"]["film_MAE_nm"] for r in rows),
        "angle_abs_error_deg":finite_stats(r["errors"]["angle_abs_error_deg"] for r in rows),
        "exact_RMSE":finite_stats(r["strict_exact_RMSE"] for r in rows),
    }

def write_outputs(output,report):
    json_path=output/"v11_wideband_401_results.json"
    json_path.write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    fields=["index","filename","noise_case","noise_factor","noise_level","q_success","q_status","quality_class","q_cost","strict_exact_RMSE","Air_error_nm","film_MAE_nm","angle_abs_error_deg","boundary_hits","closure_rmse","closure_max_abs","condition_number","smallest_singular_value","rho_air_q","runtime_s"]
    with (output/"v11_wideband_401_table.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in report["cases"]:
            writer.writerow({"index":row["index"],"filename":row["filename"],"noise_case":row["metadata"]["noise_case"],"noise_factor":row["metadata"]["noise_factor"],"noise_level":row["metadata"]["noise_level"],"q_success":row["selected"]["success"],"q_status":row["selected"]["status"],"quality_class":row["selected"]["quality_class"],"q_cost":row["selected"]["cost"],"strict_exact_RMSE":row["strict_exact_RMSE"],**row["errors"],"boundary_hits":";".join(row["boundary_hits"]),"closure_rmse":row["closure"]["rmse"],"closure_max_abs":row["closure"]["max_abs"],"condition_number":row["diagnostics"]["condition_number"],"smallest_singular_value":row["diagnostics"]["smallest_singular_value"],"rho_air_q":row["diagnostics"]["rho_air_q"],"runtime_s":row["case_runtime_s"]})
    summary=report["summary"];agg=report["aggregate_errors"]
    lines=["# V11 full 401-case wideband GPU inversion","",f"- Overall: **{report['overall']}**",f"- Band: **{report['band']['label']} nm**",f"- Input NPZ: {summary['processed']}/{report['requested_count']}",f"- Selected SciPy terminated: {summary['selected_solver_terminated']}",f"- Selected budget-exhausted: {summary['selected_budget_exhausted']}",f"- Selected other finite: {summary['selected_other_finite']}",f"- CPU/GPU strict closures passed: {summary['closure_passed']}/{summary['processed']}",f"- Backend initialization count: {summary['backend_initialization_count']}",f"- Resident constants reused: {summary['resident_reused']}",f"- Runtime: {summary['total_runtime_s']:.3f} s",f"- Mean runtime: {summary['mean_case_runtime_s']:.3f} s/case",f"- Throughput: {summary['throughput_cases_per_s']:.6f} cases/s",f"- Peak GPU memory pool: {summary['peak_gpu_memory_gib']:.3f} GiB",f"- Formal V10 unchanged: {report['formal_v10_sha256']==EXPECTED}","","## Scientific error summary","","| metric | mean | median | p95 | max |","|---|---:|---:|---:|---:|"]
    for key in ("absolute_Air_error_nm","film_MAE_nm","angle_abs_error_deg","exact_RMSE"):
        row=agg[key];lines.append(f"| {key} | {row['mean']:.6g} | {row['median']:.6g} | {row['p95']:.6g} | {row['max']:.6g} |")
    lines+=["","## Selection quality","",f"- Quality classes: {summary['selected_quality_classes']}",f"- Cases with one or more boundary hits: {summary['cases_with_boundary_hits']}",f"- Failed cases: {summary['failed_cases']}","","## Timing contract","","Runtime includes per-case input loading, eight CPU SciPy local attempts using the exact-key B=13 GPU response/Jacobian cache, final same-parameter CPU/GPU strict verification, diagnostics, and output accounting. It excludes file transfer, Slurm queue time, environment startup, and dataset generation.","","## Scope","","No formal V10 source, strict forward, ILS, residual, bounds, loss, finite-difference physical contract, optimizer tolerances, JAX, autodiff, or multi-GPU changes."]
    (output/"v11_wideband_401_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase6-json",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--band-label",required=True)
    parser.add_argument("--wavelength-min-nm",type=float,required=True)
    parser.add_argument("--wavelength-max-nm",type=float,required=True)
    parser.add_argument("--machine",type=Path)
    parser.add_argument("--count",type=int,default=401)
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    baseline=json.loads(args.phase6_json.read_text(encoding="utf-8"));baseline_cases=baseline["cases"]
    if len(baseline_cases)!=args.count:raise ValueError(f"expected {args.count} baseline cases, got {len(baseline_cases)}")
    input_dir=args.input_dir.resolve()
    expected={row["filename"] for row in baseline_cases};actual={p.name for p in input_dir.glob("static_spectrum_*.npz")}
    if actual!=expected:raise RuntimeError(f"dataset filename contract mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    config=v10.FitConfig(input_dir=str(input_dir),wavelength_min_nm=float(args.wavelength_min_nm),wavelength_max_nm=float(args.wavelength_max_nm),global_forward_model="fast_no_ils",global_popsize=8,global_maxiter=40,multistarts=8,max_nfev=600,workers=1,random_seed=20260825)
    first_path=input_dir/baseline_cases[0]["filename"];first=v10.load_fit_input(first_path,config)
    gpu=CupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
    cpu=NumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
    contract=exact_backend_contract(first);resident_before=resident_array_identity(gpu);backend_initializations=1
    first_scale=v10.robust_scale(first["spectrum"]);ExactQBatchCache(gpu,first["spectrum"],first_scale).evaluate(physical_to_solver_q(np.asarray(baseline_cases[0]["local_attempts"][0]["x0_physical"],dtype=np.float64)));gpu.synchronize()
    rows=[];progress=output/"v11_wideband_401_progress.jsonl";started_all=time.perf_counter()
    for position,baseline_case in enumerate(baseline_cases,1):
        path=input_dir/baseline_case["filename"];case_started=time.perf_counter();measurement=v10.load_fit_input(path,config)
        if exact_backend_contract(measurement)!=contract:raise RuntimeError(f"backend contract changed at {path.name}")
        scale=v10.robust_scale(measurement["spectrum"]);cache=ExactQBatchCache(gpu,measurement["spectrum"],scale)
        attempts=[q_attempt(cache,physical_to_solver_q(np.asarray(a["x0_physical"],dtype=np.float64)),config,index) for index,a in enumerate(baseline_case["local_attempts"],1)]
        ranked=select_candidate(attempts,600,1e-8);selected=ranked["selected"]
        if selected is None:raise RuntimeError(f"no numerically valid q candidate: {path.name}")
        selected_solver=np.asarray(selected["final_solver"],dtype=np.float64);selected_physical=np.asarray(selected["final_physical"],dtype=np.float64);evaluation=cache.evaluate(selected_solver)
        cpu_prediction=cpu.predict(selected_physical);gpu.synchronize();consistency=closure(cpu_prediction,evaluation.prediction)
        if not consistency["pass"]:raise RuntimeError(f"CPU/GPU strict closure failed: {path.name}")
        truth,_=v10.load_evaluation_truth(path)
        row={"index":position,"filename":path.name,"seed":baseline_case["seed"],"metadata":measurement["metadata"],"selected":selected,"ranking_reason":ranked["reason"],"equivalent_candidate_count":ranked["equivalent_candidate_count"],"strict_exact_RMSE":float(np.sqrt(np.mean((evaluation.prediction-measurement["spectrum"])**2))),"errors":errors(selected_physical,truth),"boundary_hits":boundary_hits(selected_physical),"closure":consistency,"diagnostics":diagnostics(evaluation.jacobian),"cache":cache.snapshot(),"case_runtime_s":time.perf_counter()-case_started}
        rows.append(row)
        with progress.open("a",encoding="utf-8") as stream:stream.write(json.dumps(safe(row),ensure_ascii=False)+"\n")
        print(f"[{position}/{args.count}] {path.name} status={selected['status']} quality={selected['quality_class']} cost={selected['cost']:.6g} closure=PASS",flush=True)
    total=time.perf_counter()-started_all;resident_after=resident_array_identity(gpu);memory=gpu.memory_stats()
    statuses=Counter("terminated" if r["selected"]["success"] and int(r["selected"]["status"])>0 else "budget" if int(r["selected"]["status"])==0 else "other" for r in rows)
    quality=Counter(r["selected"]["quality_class"] for r in rows);groups=defaultdict(list)
    for row in rows:groups[row["metadata"]["noise_case"]].append(row)
    summary={"processed":len(rows),"selected_solver_terminated":statuses["terminated"],"selected_budget_exhausted":statuses["budget"],"selected_other_finite":statuses["other"],"selected_quality_classes":dict(quality),"closure_passed":sum(r["closure"]["pass"] for r in rows),"cases_with_boundary_hits":sum(bool(r["boundary_hits"]) for r in rows),"failed_cases":[],"backend_initialization_count":backend_initializations,"resident_reused":resident_before==resident_after,"total_runtime_s":total,"mean_case_runtime_s":float(np.mean([r["case_runtime_s"] for r in rows])),"throughput_cases_per_s":len(rows)/total,"peak_gpu_memory_bytes":int(memory.get("memory_pool_total_bytes",0)),"peak_gpu_memory_gib":int(memory.get("memory_pool_total_bytes",0))/(1024**3)}
    report={"overall":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"band":{"label":args.band_label,"wavelength_min_nm":args.wavelength_min_nm,"wavelength_max_nm":args.wavelength_max_nm,"reported_point_count":int(len(first["wavelengths_um"]))},"requested_count":args.count,"summary":summary,"aggregate_errors":aggregate_errors(rows),"noise_case_aggregates":{name:aggregate_errors(values) for name,values in sorted(groups.items())},"formal_v10_sha256":source_sha256(),"baseline_phase6_json":str(args.phase6_json.resolve()),"input_dir":str(input_dir),"machine":json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,"memory":memory,"cases":rows,"scope_guard":"No formal V10, strict response, ILS, residual, bounds, loss, physical finite-difference contract, optimizer tolerances, JAX, autodiff, or multi-GPU changes."}
    write_outputs(output,report);print(json.dumps({"overall":"PASS","band":args.band_label,"processed":len(rows),"runtime_s":total,"output":str(output)},indent=2),flush=True)
if __name__=="__main__":main()
