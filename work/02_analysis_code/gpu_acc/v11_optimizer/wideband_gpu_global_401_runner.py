"""Production 401-case GPU-global plus GPU-local inversion for a configured wavelength band."""
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
from .gpu_global_batch_benchmark import gpu_global,local_fit
from .stage12_runner import EXPECTED,NAMES,errors,safe
from .wideband_full_runner import aggregate_errors

def compact_selected(item):
    row=dict(item);trace=np.asarray(row.pop("trace_costs",[]),dtype=float);row["trace_count"]=int(trace.size);row["trace_first_cost"]=float(trace[0]) if trace.size else None;row["trace_last_cost"]=float(trace[-1]) if trace.size else None;row["trace_min_cost"]=float(np.min(trace)) if trace.size else None;row["trace_tail_costs"]=trace[-min(20,len(trace)):].tolist();return row

def append(path,row):
    with path.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(safe(row),ensure_ascii=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())

def write_outputs(out,report,output_prefix):
    (out/f"{output_prefix}_results.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    fields=["index","filename","seed","noise_case","quality_class","success","status","cost","exact_RMSE","Air_error_nm","film_MAE_nm","angle_abs_error_deg","boundary_hits","global_runtime_s","local_runtime_s","case_runtime_s","gpu_batch_calls","candidate_evaluations","mean_batch_size","max_batch_size","closure_rmse","closure_max_abs"]
    with (out/f"{output_prefix}_table.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in report["cases"]:
            s=r["selected"];d=r["global_profile"]["differential_evolution"];w.writerow({"index":r["index"],"filename":r["filename"],"seed":r["seed"],"noise_case":r["metadata"]["noise_case"],"quality_class":s["quality_class"],"success":s["success"],"status":s["status"],"cost":s["cost"],"exact_RMSE":r["strict_exact_RMSE"],**r["errors"],"boundary_hits":";".join(s["boundary_hits"]),"global_runtime_s":r["global_runtime_s"],"local_runtime_s":r["local_runtime_s"],"case_runtime_s":r["case_runtime_s"],"gpu_batch_calls":d["gpu_batch_calls"],"candidate_evaluations":d["candidate_evaluations"],"mean_batch_size":d["mean_batch_size"],"max_batch_size":d["max_batch_size"],"closure_rmse":r["closure"]["rmse"],"closure_max_abs":r["closure"]["max_abs"]})
    s=report["summary"];a=report["aggregate_errors"];lo=report["configuration"]["wavelength_min_nm"];hi=report["configuration"]["wavelength_max_nm"];lines=[f"# V11 {lo:g}-{hi:g} nm GPU-global 401-case production result","",f"- Overall: **{report['overall']}**",f"- Processed: {s['processed']}/{report['requested_count']}",f"- Selected terminated/budget/other: {s['terminated']}/{s['budget_exhausted']}/{s['other_finite']}",f"- Strict closures: {s['closure_passed']}/{s['processed']}",f"- Fresh GPU DE: {s['fresh_gpu_de_cases']}/{s['processed']}",f"- Backend initialization: {s['backend_initialization_count']}",f"- Resident constants reused: {s['resident_reused']}",f"- Total/global/local runtime: {s['total_runtime_s']:.3f} / {s['total_global_runtime_s']:.3f} / {s['total_local_runtime_s']:.3f} s",f"- Mean case runtime: {s['mean_case_runtime_s']:.3f} s",f"- Throughput: {s['throughput_cases_per_min']:.3f} cases/min",f"- DE batch min/mean/max: {s['de_min_batch_size']} / {s['de_mean_batch_size']:.3f} / {s['de_max_batch_size']}",f"- Peak GPU memory: {s['peak_gpu_memory_gib']:.3f} GiB",f"- Formal V10 unchanged: {report['formal_v10_sha256']==EXPECTED}","","## Scientific error summary","","| metric | mean | median | p95 | max |","|---|---:|---:|---:|---:|"]
    for k in ("absolute_Air_error_nm","film_MAE_nm","angle_abs_error_deg","exact_RMSE"):
        x=a[k];lines.append(f"| {k} | {x['mean']:.6g} | {x['median']:.6g} | {x['p95']:.6g} | {x['max']:.6g} |")
    lines+=["","## Scope","","Each NPZ generated a fresh Latin-hypercube population and ran SciPy vectorized deferred differential evolution with B=48 GPU strict full-ILS population objectives, followed by eight GPU strict q-local fits. No old starts, fast_no_ils, coarse-ILS, formal V10 edit, JAX, autodiff or multi-GPU."]
    (out/f"{output_prefix}_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input-dir",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--machine",type=Path);ap.add_argument("--count",type=int,default=401);ap.add_argument("--wavelength-min-nm",type=float,default=220.0);ap.add_argument("--wavelength-max-nm",type=float,default=580.0);ap.add_argument("--output-prefix",default="v11_220_580_gpu_global_401");a=ap.parse_args()
    if a.count!=401:raise ValueError("production count must be 401")
    if not (0 < a.wavelength_min_nm < a.wavelength_max_nm):raise ValueError("wavelength bounds must satisfy 0 < min < max")
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    root=a.input_dir.resolve();paths=sorted(root.glob("static_spectrum_*.npz"),key=lambda p:p.name)
    if len(paths)!=401:raise RuntimeError(f"expected 401 NPZ, found {len(paths)}")
    cfg=v10.FitConfig(input_dir=str(root),wavelength_min_nm=a.wavelength_min_nm,wavelength_max_nm=a.wavelength_max_nm,global_forward_model="full_ils",global_popsize=8,global_maxiter=40,multistarts=8,max_nfev=600,workers=1,random_seed=20260825)
    first=v10.load_fit_input(paths[0],cfg);gpu=CupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"]);cpu=NumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"]);contract=exact_backend_contract(first);before=resident_array_identity(gpu);size=cfg.global_popsize*len(v10.PARAMS);progress=out/f"{a.output_prefix}_progress.jsonl";rows=[]
    for pos,path in enumerate(paths,1):
        case_t=time.perf_counter();m=v10.load_fit_input(path,cfg)
        if exact_backend_contract(m)!=contract:raise RuntimeError(f"backend contract changed: {path.name}")
        seed=cfg.random_seed+(pos-1)*1009;pop=v10.latin_hypercube_population(seed,size);case_cfg=v10.FitConfig(**{**asdict(cfg),"random_seed":seed});_,de,cands,profile=gpu_global(v10,m,case_cfg,pop,gpu,None);local=local_fit(v10,cands,m,case_cfg,gpu,cpu);prediction=np.asarray(local.pop("prediction"));truth,_=v10.load_evaluation_truth(path);selected=compact_selected(local["selected"]);d=profile["differential_evolution"]
        if not (d["min_batch_size"]==size and d["max_batch_size"]==size and d["gpu_batch_calls"]==d["objective_calls"] and d["candidate_evaluations"]==int(de.nfev)*size):raise RuntimeError(f"population batching contract failed: {path.name}")
        row={"index":pos,"filename":path.name,"seed":seed,"metadata":m["metadata"],"selected":selected,"fresh_population":True,"population_reused":False,"global_runtime_s":profile["totals"]["total_global_runtime_s"],"global_profile":profile,"global_candidates":[{"physical":np.asarray(c[0]),"population_index":int(c[1]),"energy":float(c[2])} for c in cands],"local_runtime_s":local["runtime_s"],"strict_exact_RMSE":local["exact_rmse"],"errors":errors(np.asarray([local["parameters"][n] for n in NAMES]),truth),"closure":local["closure"],"cache":local["cache"],"case_runtime_s":time.perf_counter()-case_t}
        rows.append(row);append(progress,row);print(f"[{pos}/401] {path.name} global={row['global_runtime_s']:.3f}s local={row['local_runtime_s']:.3f}s B={d['mean_batch_size']:.1f} quality={selected['quality_class']} closure=PASS",flush=True)
    total=time.perf_counter()-started;after=resident_array_identity(gpu);mem=gpu.memory_stats();status=Counter("terminated" if r["selected"]["success"] and r["selected"]["status"]>0 else "budget" if r["selected"]["status"]==0 else "other" for r in rows);batch_sizes=[x for r in rows for x in r["global_profile"]["differential_evolution"]["actual_batch_sizes"]];groups=defaultdict(list)
    for r in rows:groups[r["metadata"]["noise_case"]].append(r)
    s={"processed":len(rows),"terminated":status["terminated"],"budget_exhausted":status["budget"],"other_finite":status["other"],"closure_passed":sum(r["closure"]["pass"] for r in rows),"fresh_gpu_de_cases":sum(r["fresh_population"] and not r["population_reused"] for r in rows),"backend_initialization_count":1,"resident_reused":before==after,"total_runtime_s":total,"total_global_runtime_s":float(sum(r["global_runtime_s"] for r in rows)),"total_local_runtime_s":float(sum(r["local_runtime_s"] for r in rows)),"mean_case_runtime_s":float(np.mean([r["case_runtime_s"] for r in rows])),"throughput_cases_per_min":len(rows)/total*60,"de_min_batch_size":int(min(batch_sizes)),"de_mean_batch_size":float(np.mean(batch_sizes)),"de_max_batch_size":int(max(batch_sizes)),"peak_gpu_memory_bytes":int(mem.get("memory_pool_total_bytes",0)),"peak_gpu_memory_gib":int(mem.get("memory_pool_total_bytes",0))/(1024**3)}
    passed=bool(len(rows)==401 and s["closure_passed"]==401 and s["fresh_gpu_de_cases"]==401 and s["resident_reused"] and s["de_min_batch_size"]==size and source_sha256()==EXPECTED);report={"overall":"PASS" if passed else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"requested_count":401,"configuration":{**asdict(cfg),"population_size":size,"vectorized":True,"updating":"deferred","reuse_old_population":False},"summary":s,"aggregate_errors":aggregate_errors(rows),"noise_case_aggregates":{k:aggregate_errors(v) for k,v in sorted(groups.items())},"formal_v10_sha256":source_sha256(),"input_dir":str(root),"gpu_memory":mem,"machine":json.loads(a.machine.read_text(encoding="utf-8")) if a.machine and a.machine.is_file() else None,"cases":rows,"scope_guard":"401 fresh GPU B=48 full-ILS DE plus GPU strict local; formal V10 unchanged."}
    write_outputs(out,report,a.output_prefix);print(json.dumps({"overall":report["overall"],"processed":len(rows),"runtime_s":total,"output":str(out)},indent=2))
    if not passed:raise SystemExit(2)
if __name__=="__main__":main()
