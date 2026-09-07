"""V11 full_ils versus fast_no_ils global-basin diagnostic with q local fitting."""
from __future__ import annotations
import argparse,csv,json,time
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.optimizer_contract import physical_to_solver as formal_physical_to_solver,solver_to_physical as formal_solver_to_physical
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend
from .q_cache import ExactQBatchCache,physical_to_solver_q,solver_to_physical_q
from .ranking import robust_cost,select_candidate
from .stage12_runner import EXPECTED,NAMES,closure,errors,safe
CASE_PATTERNS=("clean_r0000","angle_typical_r0000","axis_offset_typical_r0000","material_typical_r0000","combined_typical_r0000")
MODES=("full_ils","fast_no_ils")

class QLeastSquaresAdapter:
    def __init__(self,cache):self.cache=cache;self.calls=[]
    def __call__(self,fun,x0,**kwargs):
        physical_start=formal_solver_to_physical(np.asarray(x0,dtype=np.float64));q0=physical_to_solver_q(physical_start);trace=[]
        def q_fun(q):
            residual=self.cache.residual(q);trace.append(robust_cost(residual,kwargs.get("loss","linear")));return residual
        call_kwargs=dict(kwargs);call_kwargs.pop("jac",None);call_kwargs["bounds"]=(np.zeros(6),np.ones(6));call_kwargs["x_scale"]=1.0
        started=time.perf_counter();result=scipy_least_squares(q_fun,x0=q0,jac=self.cache.jacobian,**call_kwargs);elapsed=time.perf_counter()-started
        q_final=np.asarray(result.x,dtype=np.float64);physical_final=solver_to_physical_q(q_final)
        self.calls.append({"call_index":len(self.calls)+1,"x0_formal_solver":np.asarray(x0),"x0_physical":physical_start,"x0_q_solver":q0,"final_solver":q_final,"final_physical":physical_final,"success":bool(result.success),"status":int(result.status),"message":str(result.message),"cost":float(result.cost),"optimality":float(result.optimality),"nfev":int(result.nfev),"njev":None if result.njev is None else int(result.njev),"runtime_s":elapsed,"trace_costs":trace})
        result.x=formal_physical_to_solver(physical_final);return result

@contextmanager
def patched(v10,adapter):
    original=v10.least_squares;v10.least_squares=adapter
    try:yield
    finally:v10.least_squares=original

def select_paths(root):
    files=sorted(Path(root).glob("static_spectrum_*.npz"));selected=[]
    for token in CASE_PATTERNS:
        matches=[path for path in files if token in path.name]
        if len(matches)!=1:raise RuntimeError(f"expected one {token} case, found {len(matches)}")
        selected.append(matches[0])
    return selected

def run_mode(v10,path,measurement,truth,gpu,cpu,mode,seed):
    config=v10.FitConfig(input_dir=str(path.parent),wavelength_min_nm=450.0,wavelength_max_nm=580.0,global_forward_model=mode,global_popsize=8,global_maxiter=40,multistarts=8,max_nfev=600,workers=1,random_seed=20260825)
    scale=v10.robust_scale(measurement["spectrum"]);cache=ExactQBatchCache(gpu,measurement["spectrum"],scale);adapter=QLeastSquaresAdapter(cache)
    started=time.perf_counter()
    with patched(v10,adapter):fit=v10.fit_measurement(measurement,config,seed)
    elapsed=time.perf_counter()-started
    if len(adapter.calls)!=8:raise RuntimeError(f"{path.name} {mode}: expected eight local starts, got {len(adapter.calls)}")
    ranked=select_candidate(adapter.calls,600,1e-8);selected=ranked["selected"]
    if selected is None:raise RuntimeError(f"{path.name} {mode}: no valid q candidate")
    final_q=np.asarray(selected["final_solver"]);final_physical=np.asarray(selected["final_physical"]);evaluation=cache.evaluate(final_q);cpu_prediction=cpu.predict(final_physical);gpu.synchronize();consistency=closure(cpu_prediction,evaluation.prediction)
    if not consistency["pass"]:raise RuntimeError(f"{path.name} {mode}: CPU/GPU closure failed")
    outer_attempts=[]
    for local,global_attempt in zip(adapter.calls,fit.get("attempts",[])):
        enriched=next(a for a in ranked["attempts"] if a["call_index"]==local["call_index"])
        outer_attempts.append({"call_index":local["call_index"],"x0_physical":local["x0_physical"],"global_population_index":global_attempt.get("global_population_index"),"global_energy":global_attempt.get("global_energy"),"q_final_physical":local["final_physical"],"strict_cost":local["cost"],"quality_class":enriched["quality_class"]})
    return {"mode":mode,"seed":seed,"global_runtime_s":float(fit["global_runtime_s"]),"total_runtime_s":elapsed,"global_summary":fit["global_summary"],"selected":selected,"ranking_reason":ranked["reason"],"parameters":{name:float(final_physical[i]) for i,name in enumerate(NAMES)},"errors":errors(final_physical,truth),"exact_rmse":float(np.sqrt(np.mean((evaluation.prediction-measurement["spectrum"])**2))),"response_cpu_gpu_closure":consistency,"global_to_local_candidates":outer_attempts,"cache":cache.snapshot(),"prediction":evaluation.prediction}

def write_summary(path,report):
    lines=["# V11 global-basin diagnostic","","- Overall: **PASS**","- Compared full_ils and fast_no_ils with identical seeds and frozen global settings.","- Both strategies feed the already-qualified V11 q=theta^2 B=13 GPU strict local optimizer.","",f"- Cases: {len(report['cases'])}",f"- Same final basin: {report['summary']['same_basin_count']}/{len(report['cases'])}",f"- All CPU/GPU strict closures pass: {report['summary']['all_closures_pass']}","","| case | full cost | fast cost | response RMSE | normalized parameter distance | same basin | full global s | fast global s |","|---|---:|---:|---:|---:|:---:|---:|---:|"]
    for row in report["cases"]:
        full=row["modes"]["full_ils"];fast=row["modes"]["fast_no_ils"];cmp=row["comparison"]
        lines.append(f"| {row['filename']} | {full['selected']['cost']:.6e} | {fast['selected']['cost']:.6e} | {cmp['final_response_rmse']:.3e} | {cmp['normalized_parameter_distance']:.3e} | {cmp['same_basin']} | {full['global_runtime_s']:.3f} | {fast['global_runtime_s']:.3f} |")
    lines+=["","same_basin is a diagnostic label requiring both final strict-response RMSE <= 1e-8 and normalized parameter distance <= 1e-3. A different basin is not classified as a GPU failure when both CPU/GPU strict closures pass.","","No formal V10 source, global algorithm, population, seed, strict response, ILS, residual, loss, bounds, tolerance, or convergence rule was changed."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--machine",type=Path);args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    rows=[];started=time.perf_counter()
    for index,path in enumerate(select_paths(args.input_dir.resolve())):
        seed=20260825+index*1000003;config=v10.FitConfig(input_dir=str(args.input_dir),wavelength_min_nm=450.0,wavelength_max_nm=580.0,workers=1)
        measurement=v10.load_fit_input(path,config);truth,_=v10.load_evaluation_truth(path)
        gpu=CupyStrictSpectrometerBackend(measurement["wavelengths_um"],measurement["generator_config"],measurement["metadata"]["internal_wavelength_margin_nm"])
        cpu=NumpyStrictSpectrometerBackend(measurement["wavelengths_um"],measurement["generator_config"],measurement["metadata"]["internal_wavelength_margin_nm"])
        mode_results={mode:run_mode(v10,path,measurement,truth,gpu,cpu,mode,seed) for mode in MODES}
        full=mode_results["full_ils"];fast=mode_results["fast_no_ils"];full_x=np.asarray([full["parameters"][n] for n in NAMES]);fast_x=np.asarray([fast["parameters"][n] for n in NAMES]);lower,upper=v10.bounds_arrays()
        full_prediction=np.asarray(full.pop("prediction"));fast_prediction=np.asarray(fast.pop("prediction"));response_rmse=float(np.sqrt(np.mean((full_prediction-fast_prediction)**2)));distance=float(np.linalg.norm((full_x-fast_x)/(upper-lower)))
        comparison={"final_response_rmse":response_rmse,"normalized_parameter_distance":distance,"same_basin":bool(response_rmse<=1e-8 and distance<=1e-3),"parameter_abs_differences":{name:float(abs(full_x[i]-fast_x[i])) for i,name in enumerate(NAMES)}}
        rows.append({"index":index+1,"filename":path.name,"seed":seed,"modes":mode_results,"comparison":comparison});print(f"[PASS] {path.name}: same_basin={comparison['same_basin']} response_rmse={response_rmse:.3e}",flush=True)
    all_closures=all(mode["response_cpu_gpu_closure"]["pass"] for row in rows for mode in row["modes"].values())
    report={"overall":"PASS" if all_closures else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"formal_v10_sha256":source_sha256(),"cases":rows,"summary":{"same_basin_count":sum(row["comparison"]["same_basin"] for row in rows),"different_basin_count":sum(not row["comparison"]["same_basin"] for row in rows),"all_closures_pass":all_closures,"runtime_s":time.perf_counter()-started},"machine":json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,"scope_guard":"Global models and settings unchanged; V11 q local GPU fitting and candidate-quality ranking only."}
    (output/"v11_global_basin.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    fields=["filename","seed","full_cost","fast_cost","full_exact_rmse","fast_exact_rmse","response_rmse","normalized_parameter_distance","same_basin","full_global_runtime_s","fast_global_runtime_s"]
    with (output/"v11_global_basin_table.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            full=row["modes"]["full_ils"];fast=row["modes"]["fast_no_ils"];cmp=row["comparison"];writer.writerow({"filename":row["filename"],"seed":row["seed"],"full_cost":full["selected"]["cost"],"fast_cost":fast["selected"]["cost"],"full_exact_rmse":full["exact_rmse"],"fast_exact_rmse":fast["exact_rmse"],"response_rmse":cmp["final_response_rmse"],"normalized_parameter_distance":cmp["normalized_parameter_distance"],"same_basin":cmp["same_basin"],"full_global_runtime_s":full["global_runtime_s"],"fast_global_runtime_s":fast["global_runtime_s"]})
    write_summary(output/"v11_global_basin_summary.md",report);print(json.dumps({"overall":report["overall"],"output":str(output),"cases":len(rows)},indent=2))
    if report["overall"]!="PASS":raise SystemExit(2)
if __name__=="__main__":main()
