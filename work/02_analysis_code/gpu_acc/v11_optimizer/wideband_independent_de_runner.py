"""Independent 401-case 220-580 nm V11 inversion with fresh full-ILS differential evolution."""
from __future__ import annotations
import argparse,csv,json,os,time
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from v10_gpu.phase6_full_runner import exact_backend_contract
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend
from .global_basin_runner import QLeastSquaresAdapter,patched
from .q_cache import ExactQBatchCache,physical_to_solver_q
from .ranking import boundary_hits,select_candidate
from .stage12_runner import EXPECTED,closure,diagnostics,errors,safe
from .wideband_full_runner import aggregate_errors

GLOBAL_MODE="full_ils";GLOBAL_POPSIZE=8;GLOBAL_MAXITER=40;MULTISTARTS=8;MAX_NFEV=600;RANDOM_SEED=20260825

def compact_attempt(item):
    row=dict(item);trace=np.asarray(row.pop("trace_costs",[]),dtype=np.float64);row["trace_count"]=int(trace.size)
    if trace.size:
        tail=trace[-min(20,trace.size):];row.update({"trace_first_cost":float(trace[0]),"trace_last_cost":float(trace[-1]),"trace_min_cost":float(np.min(trace)),"trace_tail_costs":tail.tolist()})
    else:row.update({"trace_first_cost":None,"trace_last_cost":None,"trace_min_cost":None,"trace_tail_costs":[]})
    return row

def match_global_candidate(local_attempt,formal_attempts):
    x0=np.asarray(local_attempt["x0_physical"],dtype=np.float64);matches=[row for row in formal_attempts if np.array_equal(np.asarray(row["x0"],dtype=np.float64),x0)]
    if len(matches)!=1:raise RuntimeError(f"expected one matching fresh DE candidate, found {len(matches)}")
    match=matches[0];return {"start_rank":int(match["start_rank"]),"global_population_index":int(match["global_population_index"]),"global_energy":float(match["global_energy"])}

def append_progress(path,row):
    with path.open("a",encoding="utf-8",newline="\n") as stream:stream.write(json.dumps(safe(row),ensure_ascii=False,separators=(",",":"))+"\n");stream.flush();os.fsync(stream.fileno())

def write_outputs(output,report):
    (output/"v11_wideband_220_580_independent_de_results.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    fields=["index","filename","seed","noise_case","quality_class","q_success","q_status","q_cost","strict_exact_RMSE","Air_error_nm","film_MAE_nm","angle_abs_error_deg","boundary_hits","global_runtime_s","global_nfev","global_nit","local_runtime_s","case_runtime_s","closure_rmse","closure_max_abs","condition_number","smallest_singular_value","rho_air_q"]
    with (output/"v11_wideband_220_580_independent_de_table.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in report["cases"]:
            writer.writerow({"index":row["index"],"filename":row["filename"],"seed":row["seed"],"noise_case":row["metadata"]["noise_case"],"quality_class":row["selected"]["quality_class"],"q_success":row["selected"]["success"],"q_status":row["selected"]["status"],"q_cost":row["selected"]["cost"],"strict_exact_RMSE":row["strict_exact_RMSE"],**row["errors"],"boundary_hits":";".join(row["boundary_hits"]),"global_runtime_s":row["global"]["runtime_s"],"global_nfev":row["global"]["summary"].get("global_nfev"),"global_nit":row["global"]["summary"].get("global_nit"),"local_runtime_s":row["local_runtime_s"],"case_runtime_s":row["case_runtime_s"],"closure_rmse":row["closure"]["rmse"],"closure_max_abs":row["closure"]["max_abs"],"condition_number":row["diagnostics"]["condition_number"],"smallest_singular_value":row["diagnostics"]["smallest_singular_value"],"rho_air_q":row["diagnostics"]["rho_air_q"]})
    summary=report["summary"];agg=report["aggregate_errors"];lines=["# V11 220-580 nm independent full-ILS differential-evolution inversion","",f"- Overall: **{report['overall']}**",f"- Processed: {summary['processed']}/{report['requested_count']}","- Every NPZ reran its own formal differential evolution; no prior population or start was reused.",f"- Global model: {report['configuration']['global_forward_model']}",f"- Fresh DE confirmations: {summary['fresh_de_confirmed']}/{summary['processed']}",f"- Selected SciPy terminated: {summary['selected_solver_terminated']}",f"- Selected budget-exhausted: {summary['selected_budget_exhausted']}",f"- CPU/GPU strict closures: {summary['closure_passed']}/{summary['processed']}",f"- Backend initialization count: {summary['backend_initialization_count']}",f"- Resident constants reused: {summary['resident_reused']}",f"- Total runtime: {summary['total_runtime_s']:.3f} s",f"- Global DE runtime: {summary['total_global_runtime_s']:.3f} s",f"- GPU local runtime: {summary['total_local_runtime_s']:.3f} s",f"- Mean case runtime: {summary['mean_case_runtime_s']:.3f} s",f"- Throughput: {summary['throughput_cases_per_hour']:.6f} cases/hour",f"- Peak GPU memory pool: {summary['peak_gpu_memory_gib']:.3f} GiB",f"- Formal V10 unchanged: {report['formal_v10_sha256']==EXPECTED}","","## Scientific error summary","","| metric | mean | median | p95 | max |","|---|---:|---:|---:|---:|"]
    for key in ("absolute_Air_error_nm","film_MAE_nm","angle_abs_error_deg","exact_RMSE"):
        row=agg[key];lines.append(f"| {key} | {row['mean']:.6g} | {row['median']:.6g} | {row['p95']:.6g} | {row['max']:.6g} |")
    lines+=["","## Timing contract","","Global time is the frozen formal NumPy/SciPy full-ILS differential evolution for each NPZ. Local time is eight CPU SciPy least_squares attempts with GPU strict response and B=13 center-difference Jacobian. Queue and transfer time are excluded.","","## Scope","","Only the 220-580 nm dataset is processed. No previous DE population or start is reused. Formal V10 physics, bounds, loss, tolerances, DE settings, residual, strict ILS, and truth isolation remain unchanged. No JAX, autodiff, or multi-GPU."]
    (output/"v11_wideband_220_580_independent_de_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--machine",type=Path);parser.add_argument("--count",type=int,default=401);parser.add_argument("--wavelength-min-nm",type=float,default=220.0);parser.add_argument("--wavelength-max-nm",type=float,default=580.0);args=parser.parse_args()
    if args.count!=401:raise ValueError("production run is locked to 401 NPZ files")
    if args.wavelength_min_nm!=220.0 or args.wavelength_max_nm!=580.0:raise ValueError("this runner is locked to 220-580 nm")
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);started_all=time.perf_counter();v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    input_dir=args.input_dir.resolve();paths=sorted(input_dir.glob("static_spectrum_*.npz"),key=lambda path:path.name)
    if len(paths)!=args.count:raise RuntimeError(f"expected {args.count} NPZ files, found {len(paths)}")
    config=v10.FitConfig(input_dir=str(input_dir),wavelength_min_nm=220.0,wavelength_max_nm=580.0,global_forward_model=GLOBAL_MODE,global_popsize=GLOBAL_POPSIZE,global_maxiter=GLOBAL_MAXITER,multistarts=MULTISTARTS,max_nfev=MAX_NFEV,workers=1,random_seed=RANDOM_SEED)
    first=v10.load_fit_input(paths[0],config);gpu=CupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"]);cpu=NumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"]);contract=exact_backend_contract(first);resident_before=resident_array_identity(gpu)
    warm_start=np.asarray([3.75,60.0,40.0,50.0,50.0,0.05]);warm_cache=ExactQBatchCache(gpu,first["spectrum"],v10.robust_scale(first["spectrum"]));warm_cache.evaluate(physical_to_solver_q(warm_start));gpu.synchronize()
    rows=[];progress=output/"v11_wideband_220_580_independent_de_progress.jsonl"
    for position,path in enumerate(paths,1):
        case_started=time.perf_counter();measurement=v10.load_fit_input(path,config)
        if exact_backend_contract(measurement)!=contract:raise RuntimeError(f"backend contract changed at {path.name}")
        cache=ExactQBatchCache(gpu,measurement["spectrum"],v10.robust_scale(measurement["spectrum"]));adapter=QLeastSquaresAdapter(cache);seed=RANDOM_SEED+(position-1)*1009
        with patched(v10,adapter):fit=v10.fit_measurement(measurement,config,seed)
        if len(adapter.calls)!=MULTISTARTS:raise RuntimeError(f"{path.name}: expected {MULTISTARTS} fresh DE local starts, got {len(adapter.calls)}")
        if fit["global_summary"].get("global_forward_model")!=GLOBAL_MODE:raise RuntimeError(f"{path.name}: global mode is not {GLOBAL_MODE}")
        if "global_uses_ils" in fit["global_summary"] and not fit["global_summary"]["global_uses_ils"]:raise RuntimeError(f"{path.name}: global strict ILS was not used")
        ranked=select_candidate(adapter.calls,MAX_NFEV,1e-8);selected=ranked["selected"]
        if selected is None:raise RuntimeError(f"{path.name}: no numerically valid local candidate")
        selected_solver=np.asarray(selected["final_solver"],dtype=np.float64);selected_physical=np.asarray(selected["final_physical"],dtype=np.float64);evaluation=cache.evaluate(selected_solver);cpu_prediction=cpu.predict(selected_physical);gpu.synchronize();consistency=closure(cpu_prediction,evaluation.prediction)
        if not consistency["pass"]:raise RuntimeError(f"{path.name}: CPU/GPU strict closure failed")
        compacted=[]
        for item in ranked["attempts"]:
            compact=compact_attempt(item);compact["fresh_global_candidate"]=match_global_candidate(item,fit["attempts"]);compacted.append(compact)
        compact_selected=next(item for item in compacted if int(item["call_index"])==int(selected["call_index"]))
        truth,_=v10.load_evaluation_truth(path);global_summary=dict(fit["global_summary"]);row={"index":position,"filename":path.name,"seed":seed,"metadata":measurement["metadata"],"global":{"rerun_for_this_npz":True,"population_reused":False,"start_reused":False,"runtime_s":float(fit["global_runtime_s"]),"summary":global_summary},"selected":compact_selected,"ranking_reason":ranked["reason"],"equivalent_candidate_count":ranked["equivalent_candidate_count"],"local_attempts":compacted,"local_runtime_s":float(sum(float(item["runtime_s"]) for item in adapter.calls)),"strict_exact_RMSE":float(np.sqrt(np.mean((evaluation.prediction-measurement["spectrum"])**2))),"errors":errors(selected_physical,truth),"boundary_hits":boundary_hits(selected_physical),"closure":consistency,"diagnostics":diagnostics(evaluation.jacobian),"cache":cache.snapshot(),"case_runtime_s":time.perf_counter()-case_started}
        rows.append(row);append_progress(progress,row);print(f"[{position}/{args.count}] {path.name} DE={fit['global_runtime_s']:.3f}s nfev={global_summary.get('global_nfev')} q={row['local_runtime_s']:.3f}s quality={compact_selected['quality_class']} closure=PASS",flush=True)
    total=time.perf_counter()-started_all;resident_after=resident_array_identity(gpu);memory=gpu.memory_stats();statuses=Counter("terminated" if row["selected"]["success"] and int(row["selected"]["status"])>0 else "budget" if int(row["selected"]["status"])==0 else "other" for row in rows);groups=defaultdict(list)
    for row in rows:groups[row["metadata"]["noise_case"]].append(row)
    summary={"processed":len(rows),"fresh_de_confirmed":sum(row["global"]["rerun_for_this_npz"] and not row["global"]["population_reused"] and not row["global"]["start_reused"] for row in rows),"selected_solver_terminated":statuses["terminated"],"selected_budget_exhausted":statuses["budget"],"selected_other_finite":statuses["other"],"closure_passed":sum(row["closure"]["pass"] for row in rows),"cases_with_boundary_hits":sum(bool(row["boundary_hits"]) for row in rows),"backend_initialization_count":1,"resident_reused":resident_before==resident_after,"total_runtime_s":total,"total_global_runtime_s":float(sum(row["global"]["runtime_s"] for row in rows)),"total_local_runtime_s":float(sum(row["local_runtime_s"] for row in rows)),"mean_case_runtime_s":float(np.mean([row["case_runtime_s"] for row in rows])),"throughput_cases_per_hour":len(rows)/total*3600.0,"peak_gpu_memory_bytes":int(memory.get("memory_pool_total_bytes",0)),"peak_gpu_memory_gib":int(memory.get("memory_pool_total_bytes",0))/(1024**3)}
    passed=bool(len(rows)==args.count and summary["fresh_de_confirmed"]==args.count and summary["closure_passed"]==args.count and summary["resident_reused"] and source_sha256()==EXPECTED)
    report={"overall":"PASS" if passed else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"band":{"label":"220-580","wavelength_min_nm":220.0,"wavelength_max_nm":580.0,"reported_point_count":int(len(first["wavelengths_um"]))},"requested_count":args.count,"configuration":asdict(config),"differential_evolution_contract":{"rerun_per_npz":True,"reuse_prior_population":False,"reuse_prior_start":False,"implementation":"frozen formal V10 fit_measurement","objective_backend":"NumPy CPU strict full ILS","local_backend":"CuPy GPU strict full ILS B=13 center difference"},"summary":summary,"aggregate_errors":aggregate_errors(rows),"noise_case_aggregates":{name:aggregate_errors(values) for name,values in sorted(groups.items())},"formal_v10_sha256":source_sha256(),"input_dir":str(input_dir),"machine":json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,"memory":memory,"cases":rows,"scope_guard":"Only 220-580 nm; fresh full-ILS DE per NPZ; no formal V10, physics, bounds, residual, loss, tolerance, or convergence change."}
    write_outputs(output,report);print(json.dumps({"overall":report["overall"],"processed":len(rows),"runtime_s":total,"output":str(output)},indent=2),flush=True)
    if not passed:raise SystemExit(2)
if __name__=="__main__":main()
