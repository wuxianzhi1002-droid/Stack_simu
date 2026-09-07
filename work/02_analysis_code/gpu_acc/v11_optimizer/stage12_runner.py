"""V11 Stage 1 ranking audit and Stage 2 q=theta^2 GPU local refit."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import least_squares
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend
from v10_gpu.phase5_runner import resident_array_identity
from .q_cache import ExactQBatchCache,physical_to_solver_q,solver_to_physical_q
from .ranking import select_candidate,robust_cost,boundary_hits

EXPECTED="d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"
NAMES=("Air","HSQ","PSS","SOC","TiO2","Angle")

def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    return v

def errors(x,truth):
    return {"Air_error_nm":1000.0*(float(x[0])-truth["Air"]),"film_MAE_nm":float(np.mean([abs(float(x[i])-truth[n]) for i,n in enumerate(NAMES[1:5],1)])),"angle_abs_error_deg":abs(float(x[5])-abs(float(truth["Angle"]))) }

def closure(cpu,gpu):
    d=np.asarray(gpu)-np.asarray(cpu);n=int(d.size-np.isfinite(d).sum());rmse=float(np.sqrt(np.mean(d*d))) if n==0 else float("inf");m=float(np.max(np.abs(d))) if n==0 else float("inf")
    return {"rmse":rmse,"max_abs":m,"nan_inf_count":n,"pass":rmse<=1e-10 and m<=1e-8 and n==0}

def diagnostics(j):
    j=np.asarray(j,dtype=float);s=np.linalg.svd(j,compute_uv=False);norm=np.linalg.norm(j,axis=0);corr=np.full((6,6),np.nan)
    for a in range(6):
        for b in range(6):
            if norm[a]>0 and norm[b]>0:corr[a,b]=float(np.dot(j[:,a],j[:,b])/(norm[a]*norm[b]))
    tol=max(j.shape)*np.finfo(float).eps*s[0]
    return {"singular_values":s,"condition_number":None if s[-1]==0 else float(s[0]/s[-1]),"smallest_singular_value":float(s[-1]),"numerical_rank":int(np.sum(s>tol)),"column_norms":norm,"correlation_matrix":corr,"rho_air_q":None if not np.isfinite(corr[0,5]) else float(corr[0,5])}

def q_attempt(cache,x0,config,index):
    trace=[]
    def fun(y):
        r=cache.residual(y);trace.append(robust_cost(r,config.loss));return r
    t=time.perf_counter();res=least_squares(fun,x0=x0,jac=cache.jacobian,bounds=(np.zeros(6),np.ones(6)),loss=config.loss,max_nfev=int(config.max_nfev),x_scale=1.0,ftol=1e-8,xtol=1e-8,gtol=float(config.local_gtol));elapsed=time.perf_counter()-t
    x=solver_to_physical_q(res.x)
    return {"call_index":index,"x0_solver":np.asarray(x0),"x0_physical":solver_to_physical_q(x0),"final_solver":np.asarray(res.x),"final_physical":x,"success":bool(res.success),"status":int(res.status),"message":str(res.message),"cost":float(res.cost),"optimality":float(res.optimality),"nfev":int(res.nfev),"njev":None if res.njev is None else int(res.njev),"runtime_s":elapsed,"trace_costs":trace}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase6-json",type=Path,required=True);ap.add_argument("--input-dir",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--machine",type=Path);ap.add_argument("--count",type=int,default=401);args=ap.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    old=json.loads(args.phase6_json.read_text(encoding="utf-8"));cases=old["cases"]
    if len(cases)!=args.count:raise ValueError(f"expected {args.count} Phase 6 cases, got {len(cases)}")
    config=v10.FitConfig(input_dir=str(args.input_dir),global_forward_model="fast_no_ils",global_popsize=8,global_maxiter=40,multistarts=8,max_nfev=600,workers=1,random_seed=20260825)
    stage1=[]
    for case in cases:
        ranked=select_candidate(case.get("local_attempts",[]),600,1e-8);sel=ranked["selected"]
        if sel is None:raise RuntimeError(f"no valid recorded attempt: {case['filename']}")
        stage1.append({"index":case["index"],"filename":case["filename"],"formal_selected_cost":case["gpu_optimizer"]["cost"],"formal_selected_success":case["gpu_optimizer"]["success"],"selected":sel,"ranking_reason":ranked["reason"],"equivalent_candidate_count":ranked["equivalent_candidate_count"]})
    clean=next(x for x in stage1 if "_clean_" in x["filename"]);clean_path=args.input_dir/clean["filename"];truth_clean,_=v10.load_evaluation_truth(clean_path)
    clean_new=np.asarray(clean["selected"]["final_physical"],dtype=float);clean_old=next(c for c in cases if c["filename"]==clean["filename"])["gpu_optimizer"]["fitted_parameters"];old_x=np.asarray([clean_old[n] for n in NAMES])
    stage1_gate=bool(clean["selected"]["cost"]<clean["formal_selected_cost"]/10.0 and errors(clean_new,truth_clean)["film_MAE_nm"]<errors(old_x,truth_clean)["film_MAE_nm"] and len(boundary_hits(clean_new))<len(boundary_hits(old_x)))
    stage1_report={"pass":stage1_gate,"changed_rank_count":sum(abs(float(x["selected"]["cost"])-float(x["formal_selected_cost"]))>1e-15 for x in stage1),"clean":{"formal":{"cost":clean["formal_selected_cost"],"parameters":old_x,"errors":errors(old_x,truth_clean),"boundary_hits":boundary_hits(old_x)},"reselected":{"cost":clean["selected"]["cost"],"quality_class":clean["selected"]["quality_class"],"parameters":clean_new,"errors":errors(clean_new,truth_clean),"boundary_hits":boundary_hits(clean_new)}}}
    (out/"stage1_ranking.json").write_text(json.dumps(safe({"summary":stage1_report,"cases":stage1}),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not stage1_gate:raise RuntimeError("Stage 1 clean ranking gate failed; q stage not started")
    first_path=args.input_dir/cases[0]["filename"];first=v10.load_fit_input(first_path,config);gpu=CupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"]);cpu=NumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"]);resident_before=resident_array_identity(gpu)
    results=[];progress=out/"stage2_q_progress.jsonl"
    for i,(case,signed) in enumerate(zip(cases,stage1),1):
        path=args.input_dir/case["filename"];m=v10.load_fit_input(path,config);scale=v10.robust_scale(m["spectrum"]);cache=ExactQBatchCache(gpu,m["spectrum"],scale);attempts=[];ct=time.perf_counter()
        for k,a in enumerate(case["local_attempts"],1):attempts.append(q_attempt(cache,physical_to_solver_q(np.asarray(a["x0_physical"],dtype=float)),config,k))
        ranked=select_candidate(attempts,600,1e-8);best=ranked["selected"]
        if best is None:raise RuntimeError(f"no valid q candidate: {case['filename']}")
        ev=cache.evaluate(np.asarray(best["final_solver"]));pred_gpu=ev.prediction;pred_cpu=cpu.predict(np.asarray(best["final_physical"]));gpu.synchronize();cl=closure(pred_cpu,pred_gpu)
        if not cl["pass"]:raise RuntimeError(f"q CPU/GPU closure failed: {case['filename']}")
        truth,_=v10.load_evaluation_truth(path);diag=diagnostics(ev.jacobian);row={"index":case["index"],"filename":case["filename"],"seed":case["seed"],"selected":best,"strict_exact_RMSE":float(np.sqrt(np.mean((pred_gpu-m["spectrum"])**2))),"errors":errors(np.asarray(best["final_physical"]),truth),"boundary_hits":boundary_hits(np.asarray(best["final_physical"])),"closure":cl,"diagnostics":diag,"cache":cache.snapshot(),"case_runtime_s":time.perf_counter()-ct,"signed_reselected_cost":signed["selected"]["cost"],"signed_reselected_parameters":signed["selected"]["final_physical"]}
        results.append(row)
        with progress.open("a",encoding="utf-8") as f:f.write(json.dumps(safe(row),ensure_ascii=False)+"\n")
        print(f"[{i}/{len(cases)}] {case['filename']} q_status={best['status']} q_cost={best['cost']:.6g} closure={cl['pass']}",flush=True)
    resident_after=resident_array_identity(gpu);machine=json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else {}
    terminated=sum(bool(r["selected"]["success"]) for r in results);budget=sum(int(r["selected"]["status"])==0 for r in results);total=time.perf_counter()-started
    report={"overall":"PASS","stage1":stage1_report,"stage2":{"processed":len(results),"selected_solver_terminated":terminated,"selected_budget_exhausted":budget,"all_closures_pass":all(r["closure"]["pass"] for r in results),"mean_runtime_s":float(np.mean([r["case_runtime_s"] for r in results])),"total_runtime_s":total,"resident_reused":resident_before==resident_after},"formal_v10_sha256":source_sha256(),"machine":machine,"cases":results}
    (out/"v11_stage12_results.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fields=["index","filename","q_success","q_status","q_cost","q_exact_RMSE","Air_error_nm","film_MAE_nm","angle_abs_error_deg","boundary_hits","closure_rmse","condition_number","smallest_singular_value","rho_air_q","runtime_s"]
    with (out/"v11_stage12_table.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:w.writerow({"index":r["index"],"filename":r["filename"],"q_success":r["selected"]["success"],"q_status":r["selected"]["status"],"q_cost":r["selected"]["cost"],"q_exact_RMSE":r["strict_exact_RMSE"],**r["errors"],"boundary_hits":";".join(r["boundary_hits"]),"closure_rmse":r["closure"]["rmse"],"condition_number":r["diagnostics"]["condition_number"],"smallest_singular_value":r["diagnostics"]["smallest_singular_value"],"rho_air_q":r["diagnostics"]["rho_air_q"],"runtime_s":r["case_runtime_s"]})
    lines=["# V11 Stage 1-2 optimizer experiment","",f"- Overall: **PASS**",f"- Stage 1 clean ranking gate: **{stage1_gate}**",f"- Stage 1 changed rank: {stage1_report['changed_rank_count']}/401",f"- Clean formal/reselected cost: {clean['formal_selected_cost']:.6e} / {clean['selected']['cost']:.6e}",f"- Stage 2 q processed: {len(results)}",f"- Selected SciPy terminated/budget exhausted: {terminated}/{budget}",f"- CPU/GPU closure all pass: {all(r['closure']['pass'] for r in results)}",f"- Runtime: {total:.3f} s",f"- Formal V10 unchanged: {source_sha256()==EXPECTED}","","Stage 1 separates numerical candidate validity from SciPy termination. Stage 2 changes only optimizer Angle coordinates to nonnegative q=theta^2; the frozen V10 strict response and data remain unchanged."]
    (out/"v11_stage12_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"overall":"PASS","processed":len(results),"output":str(out),"runtime_s":total},indent=2),flush=True)

if __name__=="__main__":main()
