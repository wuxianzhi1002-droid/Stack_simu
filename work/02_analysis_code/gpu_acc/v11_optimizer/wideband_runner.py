"""V11 matched-bandwidth q-fit and identifiability diagnostics."""
from __future__ import annotations
import argparse,csv,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.jacobian import BatchedCenterDifferenceJacobian,formal_cpu_jacobian
from v10_gpu.benchmark_jacobian import RELATIVE_LIMIT,COSINE_MIN,MAX_ABS_LIMIT
from v10_gpu.spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend
from .q_cache import ExactQBatchCache,physical_to_solver_q
from .ranking import boundary_hits,select_candidate
from .stage12_runner import EXPECTED,NAMES,closure,errors,q_attempt,safe

BANDS=(("450-580",450.0,580.0),("350-580",350.0,580.0),("220-580",220.0,580.0))
DIAGNOSTIC_ANGLE_DEG=0.01

def matrix_diagnostics(jacobian):
    jac=np.asarray(jacobian,dtype=np.float64);singular=np.linalg.svd(jac,compute_uv=False)
    norms=np.linalg.norm(jac,axis=0);correlation=np.full((6,6),np.nan)
    for i in range(6):
        for j in range(6):
            if norms[i]>0 and norms[j]>0:correlation[i,j]=np.dot(jac[:,i],jac[:,j])/(norms[i]*norms[j])
    tolerance=max(jac.shape)*np.finfo(float).eps*singular[0]
    return {"singular_values":singular,"condition_number":None if singular[-1]==0 else float(singular[0]/singular[-1]),"smallest_singular_value":float(singular[-1]),"largest_singular_value":float(singular[0]),"numerical_rank":int(np.sum(singular>tolerance)),"column_norms":{name:float(norms[i]) for i,name in enumerate(NAMES)},"correlation_matrix":correlation,"rho_air_angle":None if not np.isfinite(correlation[0,5]) else float(correlation[0,5])}

def jacobian_closure(cpu,gpu):
    cpu=np.asarray(cpu,dtype=np.float64);gpu=np.asarray(gpu,dtype=np.float64);columns=[];all_pass=True
    for index,name in enumerate(NAMES):
        difference=gpu[:,index]-cpu[:,index];denominator=float(np.linalg.norm(cpu[:,index]));gpu_norm=float(np.linalg.norm(gpu[:,index]))
        relative=None if denominator==0 else float(np.linalg.norm(difference)/denominator)
        maximum=float(np.max(np.abs(difference)));nan_inf=int(difference.size-np.isfinite(difference).sum())
        cosine=None if denominator==0 or gpu_norm==0 else float(np.dot(cpu[:,index],gpu[:,index])/(denominator*gpu_norm))
        passed=nan_inf==0 and maximum<=MAX_ABS_LIMIT and (relative is None or relative<=RELATIVE_LIMIT) and (cosine is None or cosine>=COSINE_MIN);all_pass=all_pass and passed
        columns.append({"parameter":name,"relative_l2_error":relative,"max_abs_difference":maximum,"cosine_similarity":cosine,"nan_inf_count":nan_inf,"pass":passed})
    return {"pass":all_pass,"columns":columns}

def find_clean(path):
    matches=sorted(Path(path).glob("static_spectrum_clean_r0000_seed*.npz"))
    if len(matches)!=1:raise RuntimeError(f"expected exactly one clean NPZ in {path}, found {len(matches)}")
    return matches[0]

def write_summary(path,report):
    lines=["# V11 matched-bandwidth diagnostics","",f"- Overall: **{report['overall']}**",f"- Formal V10 strict source unchanged: **{report['formal_v10_sha256']==EXPECTED}**","- Local coordinate: dimensionless q=theta^2, B=13 GPU strict response/Jacobian.",f"- Physical Air-Angle diagnostic angle: {DIAGNOSTIC_ANGLE_DEG} deg (diagnostic only).","","| band (nm) | points | selected cost | Air error (nm) | film MAE (nm) | angle error (deg) | cond(J raw) | cond(J q-scaled) | smallest sv(q) | rho Air-Angle |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["bands"]:
        raw=row["physical_diagnostics"]["condition_number"];scaled=row["q_scaled_diagnostics"]["condition_number"];rho=row["physical_diagnostics"]["rho_air_angle"]
        lines.append(f"| {row['band_nm']} | {row['reported_point_count']} | {row['fit']['selected']['cost']:.6e} | {row['fit']['errors']['Air_error_nm']:.6g} | {row['fit']['errors']['film_MAE_nm']:.6g} | {row['fit']['errors']['angle_abs_error_deg']:.6g} | {'inf' if raw is None else f'{raw:.3e}'} | {'inf' if scaled is None else f'{scaled:.3e}'} | {row['q_scaled_diagnostics']['smallest_singular_value']:.3e} | {'undefined' if rho is None else f'{rho:.9f}'} |")
    lines+=["","## Interpretation contract","","- raw cond(J) uses unchanged physical parameter units and is scale dependent.","- cond(J q-scaled) uses the V11 dimensionless q solver coordinates and is the primary conditioning comparison.","- normalized Air-Angle correlation is evaluated with the unchanged physical Jacobian at 0.01 deg; it diagnoses structural column collinearity and is invariant to positive single-column scaling.","- Every band uses the same eight physical starting points and the same optimizer tolerances, bounds, residual, robust loss, and strict ILS response.","","No formal V10 source, strict forward, ILS, residual, finite-difference rule, bounds, robust loss, or SciPy convergence setting was modified."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    parser=argparse.ArgumentParser()
    for flag in ("450","350","220"):parser.add_argument(f"--dataset-{flag}",type=Path,required=True)
    parser.add_argument("--phase6-json",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--machine",type=Path)
    args=parser.parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);v10=load_v10_module()
    if source_sha256()!=EXPECTED:raise RuntimeError("formal V10 hash changed")
    phase6=json.loads(args.phase6_json.read_text(encoding="utf-8"));clean_record=next(row for row in phase6["cases"] if "_clean_" in row["filename"])
    starts=[np.asarray(a["x0_physical"],dtype=np.float64) for a in clean_record["local_attempts"]]
    if len(starts)!=8:raise RuntimeError(f"expected eight frozen clean-case starts, got {len(starts)}")
    roots={"450-580":args.dataset_450,"350-580":args.dataset_350,"220-580":args.dataset_220};results=[];started_all=time.perf_counter()
    for label,lower_nm,upper_nm in BANDS:
        dataset=roots[label].resolve();clean_path=find_clean(dataset)
        config=v10.FitConfig(input_dir=str(dataset),wavelength_min_nm=lower_nm,wavelength_max_nm=upper_nm,global_forward_model="fast_no_ils",global_popsize=8,global_maxiter=40,multistarts=8,max_nfev=600,workers=1,random_seed=20260825)
        measurement=v10.load_fit_input(clean_path,config);sampling=v10.validate_sampling(measurement,config);truth,_=v10.load_evaluation_truth(clean_path)
        gpu=CupyStrictSpectrometerBackend(measurement["wavelengths_um"],measurement["generator_config"],measurement["metadata"]["internal_wavelength_margin_nm"])
        cpu=NumpyStrictSpectrometerBackend(measurement["wavelengths_um"],measurement["generator_config"],measurement["metadata"]["internal_wavelength_margin_nm"])
        scale=v10.robust_scale(measurement["spectrum"]);cache=ExactQBatchCache(gpu,measurement["spectrum"],scale);cache.evaluate(physical_to_solver_q(starts[0]));gpu.synchronize()
        attempts=[q_attempt(cache,physical_to_solver_q(start),config,index) for index,start in enumerate(starts,1)]
        ranked=select_candidate(attempts,600,1e-8);selected=ranked["selected"]
        if selected is None:raise RuntimeError(f"no numerically valid q candidate for {label}")
        selected_solver=np.asarray(selected["final_solver"]);selected_physical=np.asarray(selected["final_physical"]);selected_evaluation=cache.evaluate(selected_solver)
        response_consistency=closure(cpu.predict(selected_physical),selected_evaluation.prediction);gpu.synchronize()
        if not response_consistency["pass"]:raise RuntimeError(f"response closure failed for {label}")
        diagnostic_parameters=np.asarray([truth[name] for name in NAMES],dtype=np.float64);diagnostic_parameters[5]=DIAGNOSTIC_ANGLE_DEG
        physical_gpu_engine=BatchedCenterDifferenceJacobian(gpu);gpu.synchronize();physical_gpu=physical_gpu_engine.evaluate(diagnostic_parameters,scale).jacobian;gpu.synchronize()
        physical_cpu=formal_cpu_jacobian(cpu.model,diagnostic_parameters,measurement["spectrum"],scale);physical_consistency=jacobian_closure(physical_cpu,physical_gpu)
        if not physical_consistency["pass"]:raise RuntimeError(f"physical Jacobian closure failed for {label}")
        row={"band_nm":label,"dataset":str(dataset),"clean_npz":str(clean_path),"reported_point_count":int(len(measurement["wavelengths_um"])),"sampling_audit":sampling,"same_start_count":len(starts),"fit":{"selected":selected,"ranking_reason":ranked["reason"],"equivalent_candidate_count":ranked["equivalent_candidate_count"],"parameters":{name:float(selected_physical[i]) for i,name in enumerate(NAMES)},"errors":errors(selected_physical,truth),"boundary_hits":boundary_hits(selected_physical),"exact_rmse":float(np.sqrt(np.mean((selected_evaluation.prediction-measurement["spectrum"])**2))),"attempts":ranked["attempts"]},"response_cpu_gpu_closure":response_consistency,"physical_diagnostic_parameters":{name:float(diagnostic_parameters[i]) for i,name in enumerate(NAMES)},"physical_diagnostics":matrix_diagnostics(physical_gpu),"q_scaled_diagnostics":matrix_diagnostics(selected_evaluation.jacobian),"physical_jacobian_cpu_gpu_closure":physical_consistency,"cache":cache.snapshot()}
        results.append(row);print(f"[PASS] {label}: points={row['reported_point_count']} cost={selected['cost']:.6e} cond_q={row['q_scaled_diagnostics']['condition_number']}",flush=True)
    report={"overall":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"formal_v10_sha256":source_sha256(),"parameter_order":list(NAMES),"diagnostic_angle_deg":DIAGNOSTIC_ANGLE_DEG,"comparison_contract":{"same_clean_case":True,"same_eight_physical_starts":True,"same_optimizer_configuration":True,"q_scaled_is_primary_conditioning_metric":True,"physical_air_angle_correlation_is_scale_invariant":True},"bands":results,"runtime_s":time.perf_counter()-started_all,"machine":json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,"scope_guard":"No formal V10, strict forward, ILS, residual, physical finite-difference rule, bounds, robust loss, or SciPy convergence setting changed."}
    (output/"v11_wideband_diagnostics.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    fields=["band_nm","points","cost","exact_rmse","Air_error_nm","film_MAE_nm","angle_abs_error_deg","raw_condition_number","q_scaled_condition_number","q_smallest_singular_value","rho_air_angle","response_closure_rmse","jacobian_closure_pass"]
    with (output/"v11_wideband_table.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in results:writer.writerow({"band_nm":row["band_nm"],"points":row["reported_point_count"],"cost":row["fit"]["selected"]["cost"],"exact_rmse":row["fit"]["exact_rmse"],**row["fit"]["errors"],"raw_condition_number":row["physical_diagnostics"]["condition_number"],"q_scaled_condition_number":row["q_scaled_diagnostics"]["condition_number"],"q_smallest_singular_value":row["q_scaled_diagnostics"]["smallest_singular_value"],"rho_air_angle":row["physical_diagnostics"]["rho_air_angle"],"response_closure_rmse":row["response_cpu_gpu_closure"]["rmse"],"jacobian_closure_pass":row["physical_jacobian_cpu_gpu_closure"]["pass"]})
    write_summary(output/"v11_wideband_summary.md",report);print(json.dumps({"overall":"PASS","output":str(output),"bands":len(results)},indent=2))
if __name__=="__main__":main()
