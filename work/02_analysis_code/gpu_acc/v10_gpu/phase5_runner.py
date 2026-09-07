"""Phase 5 formal V10 GPU local-fit integration for deterministic first 10 NPZ files."""
from __future__ import annotations
_PROCESS_IMPORT_STARTED_NS=__import__("time").time_ns()
import argparse,csv,hashlib,json,os,time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares
from .backend.v10_source import load_v10_module,source_path,source_sha256
from .jacobian.contract import PARAMETER_NAMES
from .optimizer_contract import physical_to_solver,solver_to_physical
from .phase5_cache import ExactBatchResidualJacobianCache
from .spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend

STRICT_RMSE_LIMIT=1.0e-10
STRICT_MAX_ABS_LIMIT=1.0e-8
FORMAL_V10_EXPECTED_SHA256="d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)

def backend_source_hash()->dict:
    package=Path(__file__).resolve().parent
    files=[package/"spectrometer/cupy_backend.py",package/"backend/cupy_backend.py",package/"jacobian/batched.py",package/"jacobian/contract.py",package/"phase5_cache.py",Path(__file__).resolve()]
    return {str(p.relative_to(package.parent)):sha256_file(p) for p in files}

def resident_array_identity(backend)->dict:
    arrays={"electron_weight_device":backend.electron_weight_device,"interpolation_left_device":backend.interpolation_left_device,"interpolation_alpha_device":backend.interpolation_alpha_device,"reference_sampled_device":backend.reference_sampled_device,"wavelengths_device":backend.tmm_backend.wavelengths_device,"n_matrix_device":backend.tmm_backend.n_matrix_device,"k0_device":backend.tmm_backend.k0_device}
    return {name:{"pointer":int(array.data.ptr),"shape":list(array.shape),"dtype":str(array.dtype),"nbytes":int(array.nbytes)} for name,array in arrays.items()}

def exact_backend_contract(measurement)->bytes:
    payload={"generator_config":measurement["generator_config"],"internal_margin_nm":measurement["metadata"]["internal_wavelength_margin_nm"],"wavelength_shape":list(np.asarray(measurement["wavelengths_um"]).shape),"wavelength_sha256":hashlib.sha256(np.ascontiguousarray(measurement["wavelengths_um"],dtype=np.float64).tobytes()).hexdigest()}
    return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")

def select_first_valid(input_dir:Path,pattern:str,config,count:int):
    v10=load_v10_module(); valid=[]; skipped=[]
    for path in sorted(input_dir.glob(pattern),key=lambda p:p.name):
        try:
            measurement=v10.load_fit_input(path,config); sampling=v10.validate_sampling(measurement,config)
            valid.append((path,measurement,sampling))
            if len(valid)==count: break
        except Exception as exc:
            skipped.append({"filename":path.name,"error":f"{type(exc).__name__}: {exc}"})
    if len(valid)<count: raise RuntimeError(f"Only {len(valid)} valid NPZ files found; need {count}")
    return valid,skipped

class GpuLeastSquaresAdapter:
    def __init__(self,cache:ExactBatchResidualJacobianCache):
        self.cache=cache; self.calls=[]
    def __call__(self,fun,x0,**kwargs):
        before=self.cache.snapshot(); started=time.perf_counter()
        result=scipy_least_squares(fun=self.cache.residual,x0=x0,jac=self.cache.jacobian,**kwargs)
        elapsed=time.perf_counter()-started; after=self.cache.snapshot()
        delta={k:after[k]-before[k] for k in ("residual_calls","jacobian_calls","actual_gpu_batch_evaluations","cache_hits","cache_misses","gpu_batch_runtime_s","residual_callback_runtime_s","jacobian_callback_runtime_s")}
        record={"call_index":len(self.calls)+1,"x0_solver":np.asarray(x0,dtype=float).tolist(),"x0_physical":solver_to_physical(x0).tolist(),"final_solver":np.asarray(result.x,dtype=float).tolist(),"final_physical":solver_to_physical(result.x).tolist(),"success":bool(result.success),"status":int(result.status),"message":str(result.message),"cost":float(result.cost),"optimality":float(result.optimality),"nfev":int(result.nfev),"njev":None if result.njev is None else int(result.njev),"runtime_s":elapsed,"cache_delta":delta}
        self.calls.append(record); return result

@contextmanager
def patched_formal_least_squares(v10,adapter):
    original=v10.least_squares; v10.least_squares=adapter
    try: yield
    finally: v10.least_squares=original

def matching_adapter_record(attempt,records):
    x0=np.asarray(attempt["x0"],dtype=float)
    exact=[r for r in records if np.array_equal(np.asarray(r["x0_physical"],dtype=float),x0)]
    if len(exact)==1:return exact[0]
    if not records:return None
    return min(records,key=lambda r:float(np.linalg.norm(np.asarray(r["x0_physical"])-x0)))

def rank_diagnostic(jacobian):
    s=np.linalg.svd(np.asarray(jacobian,dtype=float),compute_uv=False); largest=float(s[0]); tol=float(max(jacobian.shape)*np.finfo(float).eps*largest)
    return {"numerical_rank":int(np.sum(s>tol)),"rank_tolerance":tol,"singular_values":s.tolist(),"smallest_singular_value":float(s[-1]),"condition_number":None if s[-1]==0.0 else float(s[0]/s[-1])}

def boundary_hits(values):
    v10=load_v10_module(); lower,upper=v10.bounds_arrays(); tol=1.0e-5*(upper-lower)
    return [name for i,name in enumerate(PARAMETER_NAMES) if values[i]-lower[i]<=tol[i] or upper[i]-values[i]<=tol[i]]

def truth_errors(values,truth):
    return {"Air_error_nm":1000.0*(float(values[0])-truth["Air"]),"film_MAE_nm":float(np.mean([abs(float(values[i])-truth[name]) for i,name in enumerate(PARAMETER_NAMES[1:5],start=1)])),"angle_error_deg":float(values[5])-truth["Angle"]}

def flatten_case(row):
    fit=row.get("gpu_optimizer",{}); verify=row.get("final_cpu_strict_verification",{}); stats=row.get("gpu_backend",{}); errors=row.get("scientific_errors",{}); params=fit.get("fitted_parameters",{})
    base={"index":row.get("index"),"filename":row.get("filename"),"seed":row.get("seed"),"success":fit.get("success"),"status":fit.get("status"),"message":fit.get("message"),"cost":fit.get("cost"),"optimality":fit.get("optimality"),"exact_RMSE":fit.get("exact_RMSE"),"nfev":fit.get("nfev"),"njev":fit.get("njev"),"rank":fit.get("rank"),"optimizer_runtime_s":fit.get("runtime_s"),"global_runtime_s":row.get("global_runtime_s"),"residual_calls":stats.get("residual_calls"),"jacobian_calls":stats.get("jacobian_calls"),"actual_gpu_batch_evaluations":stats.get("actual_gpu_batch_evaluations"),"cache_hits":stats.get("cache_hits"),"cache_misses":stats.get("cache_misses"),"cpu_strict_RMSE":verify.get("CPU_strict_RMSE"),"gpu_strict_RMSE":verify.get("GPU_strict_RMSE"),"same_parameter_response_RMSE":verify.get("CPU_GPU_response_RMSE"),"max_abs_difference":verify.get("max_abs_difference"),"nan_inf_count":verify.get("nan_inf_count"),"closure_pass":verify.get("pass"),**errors}
    for name in PARAMETER_NAMES:base[name]=params.get(name)
    base["boundary_hits"]=";".join(fit.get("boundary_hits",[]));return base

def write_outputs(output:Path,report:dict):
    output.mkdir(parents=True,exist_ok=True)
    (output/"phase5_10npz_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=json_default)+"\n",encoding="utf-8",newline="\n")
    rows=[flatten_case(r) for r in report["cases"]]; fields=list(rows[0].keys()) if rows else []
    with (output/"phase5_10npz_table.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    p=report["performance"]; a=report["acceptance"]
    lines=["# Phase 5 formal V10 GPU local-fit integration: first 10 NPZ","",f"- Overall: **{'PASS' if a['pass'] else 'FAIL'}**",f"- Effective backend: `{report['backend']['effective']}`",f"- Selected/completed: `{report['selection']['selected_count']}/{len(report['cases'])}`",f"- GPU optimization successes: `{a['optimization_success_count']}/10`",f"- Same-parameter CPU/GPU closure passes: `{a['closure_pass_count']}/10`",f"- One GPU backend initialization: `{report['backend']['initialization_count']}`",f"- Total measured 10-NPZ runtime: `{p['total_10npz_runtime_s']:.3f} s`",f"- Mean/median optimizer runtime per NPZ: `{p['mean_optimizer_runtime_s']:.3f}` / `{p['median_optimizer_runtime_s']:.3f} s`",f"- Total GPU optimizer runtime: `{p['total_gpu_optimizer_runtime_s']:.3f} s`",f"- Effective throughput: `{p['effective_throughput_npz_per_min']:.3f} NPZ/min`",f"- Measured projection for 401 NPZ: `{p['estimated_401_runtime_s']:.3f} s` (`{p['estimated_401_runtime_hours']:.3f} h`)",f"- Process/CUDA startup: `{p['process_cuda_startup_s']:.3f} s`; backend init: `{p['backend_initialization_s']:.3f} s`; first warmup: `{p['first_warmup_s']:.3f} s`",f"- Existing CPU baseline local optimizer: `{p['existing_cpu_baseline_optimizer_s']:.3f} s`; startup-amortized Phase 5 GPU mean per local attempt: `{p['startup_amortized_gpu_runtime_per_local_attempt_s']:.3f} s`; reference speedup: `{p['speedup_vs_existing_cpu_local_attempt']:.2f}x`","","## Per-NPZ results","","| # | filename | success | exact RMSE | closure RMSE | max abs | batches | hits/misses | optimizer s | Air error nm | film MAE nm | angle error deg |","|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["cases"]:
        f=c.get("gpu_optimizer",{});v=c.get("final_cpu_strict_verification",{});b=c.get("gpu_backend",{});e=c.get("scientific_errors",{})
        lines.append(f"| {c['index']} | `{c['filename']}` | {f.get('success')} | {f.get('exact_RMSE',float('nan')):.3e} | {v.get('CPU_GPU_response_RMSE',float('nan')):.3e} | {v.get('max_abs_difference',float('nan')):.3e} | {b.get('actual_gpu_batch_evaluations')} | {b.get('cache_hits')}/{b.get('cache_misses')} | {f.get('runtime_s',float('nan')):.3f} | {e.get('Air_error_nm',float('nan')):.3f} | {e.get('film_MAE_nm',float('nan')):.3f} | {e.get('angle_error_deg',float('nan')):.6g} |")
    abnormal=[c for c in report["cases"] if c.get("gpu_optimizer",{}).get("boundary_hits")]
    lines += ["","## Scientific abnormal cases","",f"- Boundary-hit cases: `{len(abnormal)}/{len(report['cases'])}`. Same-parameter CPU/GPU closure still passes for these cases, so they are inversion/optimization issues rather than GPU-kernel failures."]
    for c in abnormal: lines.append(f"- Case {c['index']} `{c['filename']}`: boundary hits `{c['gpu_optimizer']['boundary_hits']}`; Air error `{c['scientific_errors']['Air_error_nm']:.3f} nm`; film MAE `{c['scientific_errors']['film_MAE_nm']:.3f} nm`; angle error `{c['scientific_errors']['angle_error_deg']:.6g} deg`.")
    lines += ["","## Acceptance","",f"- A, all optimizations complete without GPU/runtime failure: `{a['A_all_optimizations_complete']}`",f"- B, all same-parameter strict closures pass: `{a['B_all_same_parameter_closures_pass']}`",f"- C, formal semantics/freeze guards pass: `{a['C_formal_semantics_preserved']}`",f"- D/E, backend reused and initialized once: `{a['D_backend_reused']}/{a['E_no_per_npz_reinitialization']}`",f"- F, final CPU strict verification recorded for every case: `{a['F_final_cpu_verification_complete']}`",f"- G, materially faster than existing CPU local baseline after amortization: `{a['G_material_speedup']}`","","CPU/GPU optimizer endpoint equality is not an acceptance criterion. Scientifically poor fits are reported as inversion issues when same-parameter strict closure passes.","","## Scope","","Exactly the first 10 valid sorted NPZ files were processed in one Python process. No 401-NPZ run, JAX, autodiff, multi-GPU, parameter change, physics change, or formal V10 source edit was performed."]
    if report.get("fatal_error"):lines += ["","## Fatal error","",f"`{report['fatal_error']}`"]
    (output/"phase5_10npz_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--input-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--machine",type=Path);parser.add_argument("--pattern",default="static_spectrum_*.npz");parser.add_argument("--backend",choices=["cpu","gpu"],default="gpu");parser.add_argument("--allow-cpu-fallback",action="store_true");parser.add_argument("--count",type=int,default=10);parser.add_argument("--global-forward-model",choices=["full_ils","fast_no_ils"],default="fast_no_ils");parser.add_argument("--global-popsize",type=int,default=8);parser.add_argument("--global-maxiter",type=int,default=40);parser.add_argument("--multistarts",type=int,default=8);parser.add_argument("--max-nfev",type=int,default=600);parser.add_argument("--random-seed",type=int,default=20260825);parser.add_argument("--cpu-baseline",type=Path);args=parser.parse_args()
    if args.count!=10: raise ValueError("Phase 5 is locked to exactly 10 NPZ files")
    process_epoch_ns=int(os.environ.get("TMM_PHASE5_PROCESS_START_EPOCH_NS",_PROCESS_IMPORT_STARTED_NS)); process_startup_s=(time.time_ns()-process_epoch_ns)/1.0e9
    main_started=time.perf_counter();v10=load_v10_module();config=v10.FitConfig(input_dir=str(args.input_dir),global_forward_model=args.global_forward_model,global_popsize=args.global_popsize,global_maxiter=args.global_maxiter,multistarts=args.multistarts,max_nfev=args.max_nfev,workers=1,random_seed=args.random_seed)
    if source_sha256()!=FORMAL_V10_EXPECTED_SHA256:raise RuntimeError("Formal V10 SHA256 changed")
    selection_started=time.perf_counter(); selected,skipped=select_first_valid(args.input_dir,args.pattern,config,args.count); selection_runtime_s=time.perf_counter()-selection_started; contract=exact_backend_contract(selected[0][1])
    for path,m,_ in selected[1:]:
        if exact_backend_contract(m)!=contract:raise RuntimeError(f"GPU backend contract differs for {path.name}")
    effective=args.backend;fallback_reason=None
    cuda_probe_s=0.0
    if effective=="gpu":
        t=time.perf_counter();availability=CupyStrictSpectrometerBackend.availability();cuda_probe_s=time.perf_counter()-t
        if not availability.get("available"):
            if not args.allow_cpu_fallback:raise RuntimeError(f"GPU initialization unavailable and fallback disabled: {availability}")
            effective="cpu";fallback_reason=availability.get("reason")
    first=selected[0][1]; backend_init_started=time.perf_counter();cpu_verify=NumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
    gpu=None
    if effective=="gpu":
        try: gpu=CupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
        except Exception as exc:
            if not args.allow_cpu_fallback: raise RuntimeError(f"GPU backend initialization failed and fallback disabled: {type(exc).__name__}: {exc}") from exc
            effective="cpu"; fallback_reason=f"{type(exc).__name__}: {exc}"; gpu=None
    backend_init_s=time.perf_counter()-backend_init_started
    warmup_s=0.0
    if gpu is not None:
        pop=v10.latin_hypercube_population(config.random_seed,max(5,config.global_popsize*6));warm=ExactBatchResidualJacobianCache(gpu,first["spectrum"],v10.robust_scale(first["spectrum"]));t=time.perf_counter();warm.evaluate(physical_to_solver(pop[0]));gpu.synchronize();warmup_s=time.perf_counter()-t
    process_cuda_startup_s=process_startup_s+cuda_probe_s
    resident_before=None if gpu is None else {"memory":gpu.memory_stats(),"arrays":resident_array_identity(gpu),"tmm_backend_id":id(gpu.tmm_backend),"response_backend_id":id(gpu)}
    cases=[];fatal=None;fit_loop_started=time.perf_counter()
    for index,(path,measurement,sampling) in enumerate(selected):
        seed=config.random_seed+index*1009; case_started=time.perf_counter()
        try:
            cache=None;adapter=None
            if effective=="gpu":
                cache=ExactBatchResidualJacobianCache(gpu,measurement["spectrum"],v10.robust_scale(measurement["spectrum"]));adapter=GpuLeastSquaresAdapter(cache)
                with patched_formal_least_squares(v10,adapter):fit=v10.fit_measurement(measurement,config,seed)
            else:fit=v10.fit_measurement(measurement,config,seed)
            if not fit["success"]:raise RuntimeError("No converged formal V10 local result")
            values=np.asarray(fit["x"],dtype=float);best=fit["attempts"][0];record=matching_adapter_record(best,adapter.calls) if adapter else None;optimizer_stats=cache.snapshot() if cache else {"residual_calls":None,"jacobian_calls":None,"actual_gpu_batch_evaluations":None,"cache_hits":None,"cache_misses":None}
            cpu_t=time.perf_counter();cpu_prediction=cpu_verify.predict(values);cpu_verify_s=time.perf_counter()-cpu_t
            gpu_t=time.perf_counter();gpu_prediction=gpu.predict(values) if gpu is not None else cpu_prediction.copy();
            if gpu is not None:gpu.synchronize()
            gpu_verify_s=time.perf_counter()-gpu_t;delta=gpu_prediction-cpu_prediction;nan_inf=int(delta.size-np.isfinite(delta).sum());closure_rmse=float(np.sqrt(np.mean(delta**2)));closure_max=float(np.max(np.abs(delta)));closure_pass=closure_rmse<=STRICT_RMSE_LIMIT and closure_max<=STRICT_MAX_ABS_LIMIT and nan_inf==0
            scale=v10.robust_scale(measurement["spectrum"]);cpu_exact=float(np.sqrt(np.mean((cpu_prediction-measurement["spectrum"])**2)));gpu_exact=float(np.sqrt(np.mean((gpu_prediction-measurement["spectrum"])**2)))
            if cache is not None:final_eval=cache.evaluate(physical_to_solver(values));rank=rank_diagnostic(final_eval.physical_jacobian)
            else:rank={"numerical_rank":None,"singular_values":[],"smallest_singular_value":None,"condition_number":fit.get("condition_number")}
            truth,noise=v10.load_evaluation_truth(path)
            result={"index":index+1,"filename":path.name,"path":str(path.resolve()),"npz_sha256":sha256_file(path),"seed":seed,"sampling_audit":sampling,"metadata":measurement["metadata"],"global_runtime_s":float(fit["global_runtime_s"]),"total_formal_fit_runtime_s":float(fit["runtime_s"]),"gpu_optimizer":{"success":bool(best["success"]),"status":int(best["status"]),"message":str(best["message"]),"fitted_parameters":{n:float(values[i]) for i,n in enumerate(PARAMETER_NAMES)},"cost":float(best["cost"]),"optimality":float(best["optimality"]),"exact_RMSE":gpu_exact,"nfev":int(best["nfev"]),"njev":None if record is None else record["njev"],"boundary_hits":boundary_hits(values),"rank":rank["numerical_rank"],"rank_diagnostics":rank,"runtime_s":float(sum(r["runtime_s"] for r in adapter.calls)) if adapter else float(fit["local_runtime_s"]),"formal_local_stage_runtime_s":float(fit["local_runtime_s"]),"best_local_attempt_runtime_s":None if record is None else record["runtime_s"],"final_normalized_residual_RMSE":float(np.sqrt(np.mean(((gpu_prediction-measurement["spectrum"])/scale)**2))),"local_attempt_count":len(fit["attempts"])},"gpu_backend":{**optimizer_stats,"response_backend_id":None if gpu is None else id(gpu),"tmm_backend_id":None if gpu is None else id(gpu.tmm_backend)},"final_cpu_strict_verification":{"CPU_strict_RMSE":cpu_exact,"GPU_strict_RMSE":gpu_exact,"CPU_normalized_residual_RMSE":float(np.sqrt(np.mean(((cpu_prediction-measurement["spectrum"])/scale)**2))),"GPU_normalized_residual_RMSE":float(np.sqrt(np.mean(((gpu_prediction-measurement["spectrum"])/scale)**2))),"CPU_GPU_response_RMSE":closure_rmse,"max_abs_difference":closure_max,"nan_inf_count":nan_inf,"CPU_runtime_s":cpu_verify_s,"GPU_runtime_s":gpu_verify_s,"limits":{"rmse":STRICT_RMSE_LIMIT,"max_abs":STRICT_MAX_ABS_LIMIT,"nan_inf":0},"pass":closure_pass},"scientific_errors":truth_errors(values,truth),"truth_loaded_after_fit_and_ranking":True,"noise_audit":noise,"local_attempts":adapter.calls if adapter else [],"case_wall_runtime_s":time.perf_counter()-case_started}
            cases.append(result);print(f"[{index+1}/10] {path.name} success={best['success']} closure={closure_pass} local={fit['local_runtime_s']:.3f}s batches={optimizer_stats.get('actual_gpu_batch_evaluations')}",flush=True)
            if not closure_pass:fatal=f"Same-parameter strict closure failed for {path.name}";break
        except Exception as exc:
            fatal=f"{path.name}: {type(exc).__name__}: {exc}";cases.append({"index":index+1,"filename":path.name,"path":str(path.resolve()),"npz_sha256":sha256_file(path),"seed":seed,"error":fatal});break
    total_s=(time.time_ns()-process_epoch_ns)/1.0e9;resident_after=None if gpu is None else {"memory":gpu.memory_stats(),"arrays":resident_array_identity(gpu),"tmm_backend_id":id(gpu.tmm_backend),"response_backend_id":id(gpu)}
    successful=[c for c in cases if c.get("gpu_optimizer",{}).get("success")];optimizer_times=[c["gpu_optimizer"]["runtime_s"] for c in successful];attempt_times=[a["runtime_s"] for c in successful for a in c.get("local_attempts",[])]
    baseline_path=args.cpu_baseline;baseline=None
    if baseline_path and baseline_path.is_file():baseline=json.loads(baseline_path.read_text(encoding="utf-8"));cpu_baseline=float(baseline["CPU"]["timing"]["total_runtime_s"])
    else:cpu_baseline=67.60879709944129
    mean_attempt=float(np.mean(attempt_times)) if attempt_times else float("inf");throughput=len(cases)/(total_s/60.0) if total_s>0 else 0.0
    fixed_startup_s=process_cuda_startup_s+backend_init_s+warmup_s
    measured_per_npz_s=max(0.0,total_s-fixed_startup_s)/len(cases) if cases else float("inf")
    estimated_401_s=fixed_startup_s+measured_per_npz_s*401 if cases else float("inf")
    amortized_attempt_s=(sum(attempt_times)+fixed_startup_s)/len(attempt_times) if attempt_times else float("inf")
    performance={"process_cuda_startup_s":max(0.0,process_cuda_startup_s),"process_startup_s":process_startup_s,"cuda_probe_s":cuda_probe_s,"input_selection_validation_s":selection_runtime_s,"backend_initialization_s":backend_init_s,"first_warmup_s":warmup_s,"per_npz_optimizer_runtime_s":optimizer_times,"per_npz_final_cpu_verification_runtime_s":[c["final_cpu_strict_verification"]["CPU_runtime_s"] for c in successful],"mean_optimizer_runtime_s":float(np.mean(optimizer_times)) if optimizer_times else float("inf"),"median_optimizer_runtime_s":float(np.median(optimizer_times)) if optimizer_times else float("inf"),"total_gpu_optimizer_runtime_s":float(np.sum(optimizer_times)),"total_10npz_runtime_s":total_s,"effective_throughput_npz_per_min":throughput,"estimated_401_runtime_s":estimated_401_s,"estimated_401_runtime_hours":estimated_401_s/3600.0,"projection_contract":"one measured startup/backend/warmup plus 401 times the measured non-startup per-NPZ runtime","existing_cpu_baseline_optimizer_s":cpu_baseline,"existing_cpu_baseline_source":str(baseline_path) if baseline_path else "Phase 4 job 1482539 recorded CPU strict local optimizer","mean_gpu_runtime_per_local_attempt_s":mean_attempt,"startup_amortized_gpu_runtime_per_local_attempt_s":amortized_attempt_s,"speedup_vs_existing_cpu_local_attempt":cpu_baseline/amortized_attempt_s if amortized_attempt_s>0 else 0.0,"startup_amortized_over_completed_npz":len(cases)}
    closures=sum(bool(c.get("final_cpu_strict_verification",{}).get("pass")) for c in cases);optimizations=len(successful);backend_reused=effective=="gpu" and resident_before["response_backend_id"]==resident_after["response_backend_id"] and resident_before["tmm_backend_id"]==resident_after["tmm_backend_id"] and resident_before["arrays"]==resident_after["arrays"] and all(c.get("gpu_backend",{}).get("response_backend_id")==resident_before["response_backend_id"] for c in successful)
    acceptance={"optimization_success_count":optimizations,"closure_pass_count":closures,"A_all_optimizations_complete":optimizations==10 and fatal is None,"B_all_same_parameter_closures_pass":closures==10,"C_formal_semantics_preserved":source_sha256()==FORMAL_V10_EXPECTED_SHA256,"D_backend_reused":backend_reused,"E_no_per_npz_reinitialization":backend_reused,"F_final_cpu_verification_complete":sum("final_cpu_strict_verification" in c for c in cases)==10,"G_material_speedup":performance["speedup_vs_existing_cpu_local_attempt"]>1.0,"pass":False}
    acceptance["pass"]=all(acceptance[k] for k in ("A_all_optimizations_complete","B_all_same_parameter_closures_pass","C_formal_semantics_preserved","D_backend_reused","E_no_per_npz_reinitialization","F_final_cpu_verification_complete","G_material_speedup"))
    machine=json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None
    report={"phase":"Phase 5 formal V10 GPU local-fit integration","created_utc":datetime.now(timezone.utc).isoformat(),"selection":{"rule":"sorted filename order; first 10 valid files; no cherry-picking","input_dir":str(args.input_dir.resolve()),"pattern":args.pattern,"requested_count":10,"selected_count":len(selected),"filenames":[p.name for p,_,_ in selected],"skipped_invalid":skipped},"backend":{"requested":args.backend,"effective":effective,"allow_cpu_fallback":args.allow_cpu_fallback,"fallback_reason":fallback_reason,"initialization_count":1 if gpu is not None else 0,"resident_before":resident_before,"resident_after":resident_after},"configuration":asdict(config),"optimizer_semantics":{"formal_fit_measurement_used_directly":True,"formal_global_search_and_candidate_selection_used_directly":True,"formal_attempt_ranking_used_directly":True,"formal_truth_isolation_used_directly":True,"least_squares_location":"CPU SciPy","GPU_local_residual":"one exact-key cached B=13 strict response evaluation","GPU_local_jacobian":"same cached B=13 evaluation center difference","CPU_optimizer_endpoint_equality_required":False},"formal_v10_source":str(source_path()),"formal_v10_sha256":source_sha256(),"gpu_backend_source_sha256":backend_source_hash(),"machine":machine,"cpu_baseline_reference":None if baseline is None else {"path":str(baseline_path),"input_npz":baseline.get("input_npz"),"CPU":baseline.get("CPU"),"GPU":baseline.get("GPU")},"cases":cases,"performance":performance,"acceptance":acceptance,"fatal_error":fatal,"stop_condition":"Stopped after deterministic 10-NPZ Phase 5 run; 401-NPZ dataset was not launched."}
    write_outputs(args.output_dir,report);print(json.dumps({"pass":acceptance["pass"],"completed":len(cases),"output":str(args.output_dir),"fatal_error":fatal},indent=2),flush=True)
    if not acceptance["pass"]:raise SystemExit(2)
if __name__=="__main__":main()
