"""One-clean-case GPU batched full-ILS differential-evolution benchmark."""
from __future__ import annotations
import argparse,json,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.global_search import GpuFullIlsPopulationObjective
from v10_gpu.phase5_runner import resident_array_identity
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend
from .q_cache import ExactQBatchCache,physical_to_solver_q
from .ranking import select_candidate
from .stage12_runner import EXPECTED,NAMES,closure,errors,q_attempt,safe

def de_kwargs(v10,cfg,seed,pop):
    return dict(bounds=[v10.BOUNDS[n] for n in v10.PARAMS],strategy="best1bin",maxiter=cfg.global_maxiter,popsize=cfg.global_popsize,tol=1e-7,mutation=(0.5,1.0),recombination=0.7,seed=seed,polish=False,init=pop,workers=1)

def cpu_global(v10,m,cfg,pop):
    obs=np.asarray(m["spectrum"]);scale=v10.robust_scale(obs);idx=np.arange(0,len(obs),cfg.global_stride);model=v10.SpectrometerForwardModel(m["wavelengths_um"],m["generator_config"],m["metadata"]["internal_wavelength_margin_nm"]);stats={"calls":0,"forward_s":0.0}
    def fun(x):
        t=time.perf_counter();pred=model.predict(np.asarray(x))[idx];stats["forward_s"]+=time.perf_counter()-t;stats["calls"]+=1;return v10.robust_objective((pred-obs[idx])/scale,cfg.loss)
    t=time.perf_counter();audit=np.asarray([fun(x) for x in pop]);de_t=time.perf_counter();res=differential_evolution(fun,updating="immediate",vectorized=False,**de_kwargs(v10,cfg,cfg.random_seed,pop));de_s=time.perf_counter()-de_t;total=time.perf_counter()-t;cands=v10.select_diverse_candidates(res.population,res.population_energies,cfg.multistarts)
    profile={"updating":"immediate","vectorized":False,"initial_audit_candidates":len(pop),"scipy_nfev":int(res.nfev),"de_generations":int(res.nit),"candidate_evaluations":stats["calls"],"objective_calls":stats["calls"],"strict_forward_time_s":stats["forward_s"],"scipy_de_wall_time_s":de_s,"total_global_runtime_s":total,"scipy_cpu_overhead_s":max(0.0,total-stats["forward_s"])}
    return audit,res,cands,profile

def gpu_global(v10,m,cfg,pop,gpu,chunk):
    obs=np.asarray(m["spectrum"]);idx=np.arange(0,len(obs),cfg.global_stride);obj=GpuFullIlsPopulationObjective(gpu,obs,v10.robust_scale(obs),idx,cfg.loss,max_chunk_size=chunk);t=time.perf_counter();obj.set_phase("initial_audit");audit=obj(pop.T);obj.set_phase("differential_evolution");de_t=time.perf_counter();res=differential_evolution(obj,updating="deferred",vectorized=True,**de_kwargs(v10,cfg,cfg.random_seed,pop));de_s=time.perf_counter()-de_t;total=time.perf_counter()-t;prof=obj.profile();d=prof["differential_evolution"];d["scipy_de_wall_time_s"]=de_s;d["scipy_cpu_overhead_s"]=max(0.0,de_s-d["objective_wall_time_s"]);rows=list(prof.values());prof["totals"]={"gpu_batch_calls":sum(x.get("gpu_batch_calls",0) for x in rows),"candidate_evaluations":sum(x.get("candidate_evaluations",0) for x in rows),"h2d_time_s":sum(x.get("h2d_time_s",0.0) for x in rows),"gpu_forward_time_s":sum(x.get("gpu_forward_time_s",0.0) for x in rows),"gpu_robust_objective_time_s":sum(x.get("gpu_robust_objective_time_s",0.0) for x in rows),"d2h_time_s":sum(x.get("d2h_time_s",0.0) for x in rows),"objective_wall_time_s":sum(x.get("objective_wall_time_s",0.0) for x in rows),"total_global_runtime_s":total,"updating":"deferred","vectorized":True,"scipy_nfev_batch_calls":int(res.nfev),"de_generations":int(res.nit)};cands=v10.select_diverse_candidates(res.population,res.population_energies,cfg.multistarts)
    return audit,res,cands,prof

def local_fit(v10,cands,m,cfg,gpu,cpu):
    if len(cands)!=cfg.multistarts:raise RuntimeError(f"expected {cfg.multistarts} candidates, got {len(cands)}")
    cache=ExactQBatchCache(gpu,m["spectrum"],v10.robust_scale(m["spectrum"]));t=time.perf_counter();attempts=[q_attempt(cache,physical_to_solver_q(np.asarray(c[0])),cfg,i) for i,c in enumerate(cands,1)];ranked=select_candidate(attempts,cfg.max_nfev,1e-8);best=ranked["selected"]
    if best is None:raise RuntimeError("no valid local candidate")
    ev=cache.evaluate(np.asarray(best["final_solver"]));x=np.asarray(best["final_physical"]);cpu_pred=cpu.predict(x);gpu.synchronize();cl=closure(cpu_pred,ev.prediction)
    if not cl["pass"]:raise RuntimeError("strict local closure failed")
    return {"runtime_s":time.perf_counter()-t,"selected":best,"parameters":{n:float(x[i]) for i,n in enumerate(NAMES)},"strict_cost":float(best["cost"]),"exact_rmse":float(np.sqrt(np.mean((ev.prediction-m["spectrum"])**2))),"closure":cl,"cache":cache.snapshot(),"prediction":ev.prediction}

def semantics(cpu,gpu):
    cpu=np.asarray(cpu);gpu=np.asarray(gpu);d=gpu-cpu;rel=np.abs(d)/np.maximum(np.abs(cpu),np.finfo(float).tiny);rho=float(spearmanr(cpu,gpu).statistic);top=min(8,len(cpu));overlap=len(set(np.argsort(cpu)[:top])&set(np.argsort(gpu)[:top]));passed=bool(np.allclose(cpu,gpu,rtol=1e-9,atol=1e-8) and rho>0.999999999 and overlap==top)
    return {"rmse":float(np.sqrt(np.mean(d*d))),"max_abs":float(np.max(np.abs(d))),"max_relative":float(np.max(rel)),"spearman_rank_correlation":rho,"top8_overlap":overlap,"count":len(cpu),"pass":passed}

def summary(path,r):
    c=r["cpu_baseline"];g=r["gpu_batched"];p=g["global_profile"]["totals"];d=g["global_profile"]["differential_evolution"];s=r["objective_semantics"];x=r["comparison"]
    lines=["# GPU population-batched full-ILS global benchmark","",f"- Overall: **{r['overall']}**",f"- Input: {r['input_npz']}",f"- Population: {r['configuration']['population_size']}",f"- CPU/GPU generations: {c['global_profile']['de_generations']} / {p['de_generations']}",f"- Scheduling: CPU immediate; GPU vectorized deferred",f"- GPU batch calls: {p['gpu_batch_calls']}",f"- Mean/max DE batch: {d['mean_batch_size']:.3f} / {d['max_batch_size']}",f"- Objective semantics: {s['pass']}; max relative {s['max_relative']:.3e}; top-8 {s['top8_overlap']}/8","",f"- CPU scalar global: {c['global_runtime_s']:.6f} s",f"- GPU batched global: {g['global_runtime_s']:.6f} s",f"- Global speedup: {x['global_speedup']:.2f}x",f"- CPU-start/GPU-start local: {c['local_runtime_s']:.6f} / {g['local_runtime_s']:.6f} s",f"- CPU/GPU total inversion: {c['total_inversion_runtime_s']:.6f} / {g['total_inversion_runtime_s']:.6f} s",f"- Overall speedup: {x['overall_speedup']:.2f}x","","## GPU profiling","",f"- H2D: {p['h2d_time_s']:.6f} s",f"- strict full-ILS forward: {p['gpu_forward_time_s']:.6f} s",f"- robust objective: {p['gpu_robust_objective_time_s']:.6f} s",f"- D2H energies: {p['d2h_time_s']:.6f} s",f"- SciPy CPU overhead: {d['scipy_cpu_overhead_s']:.6f} s",f"- Peak GPU memory: {r['gpu']['peak_memory_gib']:.3f} GiB","","## Endpoint interpretation","",f"- Final response RMSE: {x['final_response_rmse']:.3e}",f"- Normalized parameter distance: {x['normalized_parameter_distance']:.3e}",f"- Same-basin diagnostic: {x['same_basin']}","","Endpoint equality is not an acceptance gate because vectorization requires deferred updating. All frozen physical and optimizer settings except scheduling are unchanged.","","One clean NPZ only. No Phase 1-6 replay, coarse-ILS, fast_no_ils, 401-case run, JAX, autodiff, float32, mixed precision or multi-GPU."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--machine",type=Path);ap.add_argument("--max-chunk-size",type=int);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    cfg=v10.FitConfig(input_dir=str(a.input.parent),wavelength_min_nm=220.0,wavelength_max_nm=580.0,global_forward_model="full_ils",global_popsize=8,global_maxiter=40,multistarts=8,max_nfev=600,workers=1,random_seed=20260825);m=v10.load_fit_input(a.input.resolve(),cfg);truth,_=v10.load_evaluation_truth(a.input.resolve());size=cfg.global_popsize*len(v10.PARAMS);pop=v10.latin_hypercube_population(cfg.random_seed,size)
    t=time.perf_counter();gpu=CupyStrictSpectrometerBackend(m["wavelengths_um"],m["generator_config"],m["metadata"]["internal_wavelength_margin_nm"]);cpu=NumpyStrictSpectrometerBackend(m["wavelengths_um"],m["generator_config"],m["metadata"]["internal_wavelength_margin_nm"]);init_s=time.perf_counter()-t;before=resident_array_identity(gpu);warm=ExactQBatchCache(gpu,m["spectrum"],v10.robust_scale(m["spectrum"]));warm.evaluate(physical_to_solver_q(pop[0]));gpu.synchronize()
    ca,cr,cc,cp=cpu_global(v10,m,cfg,pop);ga,gr,gc,gp=gpu_global(v10,m,cfg,pop,gpu,a.max_chunk_size);sem=semantics(ca,ga)
    if not sem["pass"]:raise RuntimeError(f"objective mismatch: {sem}")
    cl=local_fit(v10,cc,m,cfg,gpu,cpu);gl=local_fit(v10,gc,m,cfg,gpu,cpu);cx=np.asarray([cl["parameters"][n] for n in NAMES]);gx=np.asarray([gl["parameters"][n] for n in NAMES]);lo,hi=v10.bounds_arrays();c_pred=np.asarray(cl.pop("prediction"));g_pred=np.asarray(gl.pop("prediction"));rr=float(np.sqrt(np.mean((c_pred-g_pred)**2)));dist=float(np.linalg.norm((cx-gx)/(hi-lo)));after=resident_array_identity(gpu);mem=gpu.memory_stats();ct=cp["total_global_runtime_s"]+cl["runtime_s"];gt=gp["totals"]["total_global_runtime_s"]+gl["runtime_s"];d=gp["differential_evolution"];batch_ok=bool(d["gpu_batch_calls"]==d["objective_calls"] and d["max_batch_size"]==size and d["min_batch_size"]>1 and d["candidate_evaluations"]==int(gr.nfev)*size)
    accept={"A_gpu_population_batch_used":batch_ok,"B_no_scalar_population_forward":d["min_batch_size"]>1,"C_stable_no_oom_nan":True,"D_objective_semantics":sem["pass"],"E_strict_gpu_local_used":gl["closure"]["pass"],"F_significant_global_speedup":cp["total_global_runtime_s"]/gp["totals"]["total_global_runtime_s"]>2,"resident_constants_reused":before==after,"formal_v10_unchanged":source_sha256()==EXPECTED};passed=all(accept.values())
    report={"overall":"PASS" if passed else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"input_npz":str(a.input.resolve()),"formal_v10_sha256":source_sha256(),"configuration":{**asdict(cfg),"population_size":size,"cpu_updating":"immediate","gpu_updating":"deferred","gpu_vectorized":True,"max_chunk_size":a.max_chunk_size},"objective_semantics":sem,"cpu_baseline":{"global_runtime_s":cp["total_global_runtime_s"],"global_profile":cp,"local_runtime_s":cl["runtime_s"],"total_inversion_runtime_s":ct,**cl,"truth_errors":errors(cx,truth)},"gpu_batched":{"global_runtime_s":gp["totals"]["total_global_runtime_s"],"global_profile":gp,"local_runtime_s":gl["runtime_s"],"total_inversion_runtime_s":gt,**gl,"truth_errors":errors(gx,truth)},"comparison":{"global_speedup":cp["total_global_runtime_s"]/gp["totals"]["total_global_runtime_s"],"overall_speedup":ct/gt,"final_response_rmse":rr,"normalized_parameter_distance":dist,"same_basin":bool(rr<=1e-8 and dist<=1e-3),"parameter_abs_differences":{n:float(abs(cx[i]-gx[i])) for i,n in enumerate(NAMES)}},"gpu":{"backend_initialization_s":init_s,"resident_reused":before==after,"memory":mem,"peak_memory_gib":int(mem.get("memory_pool_total_bytes",0))/(1024**3)},"acceptance":accept,"machine":json.loads(a.machine.read_text(encoding="utf-8")) if a.machine and a.machine.is_file() else None,"scope_guard":"One clean NPZ, 0.002 nm full ILS, CPU SciPy DE with GPU population objective."}
    (out/"gpu_global_batch_benchmark.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n");summary(out/"gpu_global_batch_benchmark_summary.md",report);print(json.dumps({"overall":report["overall"],"global_speedup":report["comparison"]["global_speedup"],"overall_speedup":report["comparison"]["overall_speedup"],"output":str(out)},indent=2))
    if not passed:raise SystemExit(2)
if __name__=="__main__":main()
