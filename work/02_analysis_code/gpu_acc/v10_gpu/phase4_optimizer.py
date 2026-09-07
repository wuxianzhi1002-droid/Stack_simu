"""Phase 4 single-NPZ CPU/GPU strict local least-squares integration."""
from __future__ import annotations
import argparse,hashlib,inspect,json,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .backend.v10_source import load_v10_module,source_sha256
from .jacobian import BatchedCenterDifferenceJacobian,formal_cpu_jacobian
from .jacobian.contract import PARAMETER_NAMES
from .jacobian.metrics import aggregate_case_metrics,column_metrics
from .optimizer_contract import effective_physical_scales,physical_to_solver,solver_jacobian_from_physical,solver_to_physical
from .spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend

RESIDUAL_RMSE_LIMIT=1e-10; RESIDUAL_MAX_ABS_LIMIT=1e-8
JAC_RELATIVE_LIMIT=2e-5; JAC_COSINE_MIN=0.999999999; JAC_MAX_ABS_LIMIT=1e-4
FINAL_RESPONSE_RMSE_LIMIT=1e-8
PARAMETER_EQUIVALENCE_SPAN_FRACTION=1e-6

def json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def robust_cost(residual,loss):
    v10=load_v10_module(); return 0.5*v10.robust_objective(np.asarray(residual),loss)

class TimedResidual:
    def __init__(self,backend,observed,scale,loss):
        self.backend=backend; self.observed=np.asarray(observed); self.scale=float(scale); self.loss=loss; self.calls=0; self.runtime_s=0.0; self.records=[]
    def __call__(self,solver):
        physical=solver_to_physical(solver); started=time.perf_counter(); prediction=self.backend.predict(physical); elapsed=time.perf_counter()-started
        residual=(prediction-self.observed)/self.scale; self.calls+=1; self.runtime_s+=elapsed
        self.records.append({'call_index':self.calls,'solver':np.asarray(solver).tolist(),'physical':physical.tolist(),'cost':robust_cost(residual,self.loss),'residual_l2_norm':float(np.linalg.norm(residual))})
        return residual

class TimedGpuJacobian:
    def __init__(self,backend,scale):
        self.backend=backend; self.physical=BatchedCenterDifferenceJacobian(backend); self.scale=float(scale); self.calls=0; self.runtime_s=0.0; self.records=[]
    def __call__(self,solver):
        physical=solver_to_physical(solver); started=time.perf_counter(); physical_jac=self.physical.jacobian(physical,self.scale); jac=solver_jacobian_from_physical(physical_jac,physical); self.backend.synchronize(); elapsed=time.perf_counter()-started
        self.calls+=1; self.runtime_s+=elapsed; self.records.append({'call_index':self.calls,'solver':np.asarray(solver).tolist(),'physical':physical.tolist(),'runtime_s':elapsed})
        return jac

def trajectory_callback(records):
    def callback(intermediate_result):
        solver=np.asarray(intermediate_result.x,dtype=np.float64); physical=solver_to_physical(solver)
        cost=getattr(intermediate_result,'cost',None); fun=getattr(intermediate_result,'fun',None)
        if cost is None and fun is not None: cost=0.5*float(np.dot(fun,fun))
        records.append({'iteration_index':len(records)+1,'solver':solver.tolist(),'physical':physical.tolist(),'Air':float(physical[0]),'Angle':float(physical[5]),'cost':None if cost is None else float(cost),'nfev':None if getattr(intermediate_result,'nfev',None) is None else int(intermediate_result.nfev)})
    return callback

def run_local(fun,x0_solver,config,jac):
    trajectory=[]; kwargs={'fun':fun,'x0':x0_solver,'jac':jac,'bounds':(np.zeros(6),np.ones(6)),'loss':config.loss,'max_nfev':int(config.max_nfev),'x_scale':1.0,'ftol':1e-8,'xtol':1e-8,'gtol':float(config.local_gtol)}
    callback_supported='callback' in inspect.signature(least_squares).parameters
    if callback_supported: kwargs['callback']=trajectory_callback(trajectory)
    started=time.perf_counter(); result=least_squares(**kwargs); runtime=time.perf_counter()-started
    return result,runtime,trajectory,callback_supported

def boundary_hits(values):
    v10=load_v10_module(); lower,upper=v10.bounds_arrays(); tolerance=1e-5*(upper-lower)
    return [name for i,name in enumerate(PARAMETER_NAMES) if values[i]-lower[i] <= tolerance[i] or upper[i]-values[i] <= tolerance[i]]

def rank_diagnostics(jacobian):
    singular=np.linalg.svd(jacobian,compute_uv=False); largest=float(singular[0]); tolerance=float(max(jacobian.shape)*np.finfo(float).eps*largest)
    return {'rank':int(np.sum(singular>tolerance)),'rank_tolerance':tolerance,'singular_values':singular.tolist(),'smallest_singular_value':float(singular[-1]),'condition_number':None if not np.isfinite(np.linalg.cond(jacobian)) else float(np.linalg.cond(jacobian))}

def truth_errors(values,truth):
    films=[abs(float(values[i])-float(truth[name])) for i,name in enumerate(PARAMETER_NAMES[1:5],start=1)]
    return {'Air_error_nm':1000.0*(float(values[0])-float(truth['Air'])),'Air_abs_error_nm':1000.0*abs(float(values[0])-float(truth['Air'])),'film_MAE_nm':float(np.mean(films)),'Angle_error_deg':float(values[5])-float(truth['Angle']),'Angle_abs_error_deg':abs(float(values[5])-float(truth['Angle']))}

def result_payload(result,values,prediction,observed,timing,trajectory,callback_supported,rank,truth):
    return {'success':bool(result.success) and np.isfinite(result.cost),'status':int(result.status),'message':str(result.message),'fitted_parameters':{name:float(values[i]) for i,name in enumerate(PARAMETER_NAMES)},'fitted_parameter_vector':values.tolist(),'cost':float(result.cost),'optimality':float(result.optimality),'exact_RMSE':float(np.sqrt(np.mean((prediction-observed)**2))),'nfev':int(result.nfev),'njev':None if result.njev is None else int(result.njev),'boundary_hits':boundary_hits(values),'rank':rank,**truth_errors(values,truth),'timing':timing,'trajectory':{'callback_supported':callback_supported,'record_count':len(trajectory),'records':trajectory}}

def first_step(trajectory,x0_solver):
    if not trajectory: return None
    solver=np.asarray(trajectory[0]['solver']); physical=np.asarray(trajectory[0]['physical']); initial_physical=solver_to_physical(x0_solver)
    return {'solver_step':(solver-x0_solver).tolist(),'physical_step':(physical-initial_physical).tolist(),'first_iterate_solver':solver.tolist(),'first_iterate_physical':physical.tolist(),'cost':trajectory[0]['cost']}

def resident_arrays(backend):
    arrays={'electron_weight_device':backend.electron_weight_device,'interpolation_left_device':backend.interpolation_left_device,'interpolation_alpha_device':backend.interpolation_alpha_device,'reference_sampled_device':backend.reference_sampled_device,'wavelengths_device':backend.tmm_backend.wavelengths_device,'n_matrix_device':backend.tmm_backend.n_matrix_device,'k0_device':backend.tmm_backend.k0_device}
    return {name:{'pointer':int(array.data.ptr),'shape':list(array.shape),'dtype':str(array.dtype),'nbytes':int(array.nbytes)} for name,array in arrays.items()}

def main():
    package=Path(__file__).resolve().parent; default_npz=package.parents[2]/'04_results_and_datasets'/'static_stackrt_v10_20260825_232804'/'static_spectrum_clean_r0000_seed20260825.npz'
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,default=default_npz); parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--machine',type=Path); args=parser.parse_args(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    machine_info=json.loads(args.machine.read_text(encoding='utf-8')) if args.machine and args.machine.is_file() else None
    v10=load_v10_module(); config=v10.FitConfig(input_dir=str(args.input.parent),workers=1); measurement=v10.load_fit_input(args.input,config); observed=np.asarray(measurement['spectrum']); wavelengths=np.asarray(measurement['wavelengths_um']); scale=float(v10.robust_scale(observed)); lower,upper=v10.bounds_arrays(); span=upper-lower
    population_size=max(5,int(config.global_popsize)*len(PARAMETER_NAMES)); population=v10.latin_hypercube_population(config.random_seed,population_size); x0_physical=np.asarray(population[0]); x0_solver=physical_to_solver(x0_physical)
    cpu_setup_started=time.perf_counter(); cpu=NumpyStrictSpectrometerBackend(wavelengths,measurement['generator_config'],measurement['metadata']['internal_wavelength_margin_nm']); cpu_setup=time.perf_counter()-cpu_setup_started
    gpu_setup_started=time.perf_counter(); gpu=CupyStrictSpectrometerBackend(wavelengths,measurement['generator_config'],measurement['metadata']['internal_wavelength_margin_nm']); gpu_setup=time.perf_counter()-gpu_setup_started
    resident_before=resident_arrays(gpu); gpu_jac_physical=BatchedCenterDifferenceJacobian(gpu)
    warmup_started=time.perf_counter(); gpu.predict(x0_physical); gpu_jac_physical.jacobian(x0_physical,scale); gpu.synchronize(); gpu_warmup=time.perf_counter()-warmup_started
    first_cpu_res=(cpu.predict(x0_physical)-observed)/scale; first_gpu_res=(gpu.predict(x0_physical)-observed)/scale
    residual_delta=first_gpu_res-first_cpu_res; residual_closure={'rmse':float(np.sqrt(np.mean(residual_delta**2))),'max_abs':float(np.max(np.abs(residual_delta))),'nan_inf_count':int(np.size(first_gpu_res)-np.isfinite(first_gpu_res).sum())}
    started=time.perf_counter(); first_cpu_phys_jac=formal_cpu_jacobian(cpu.model,x0_physical,observed,scale); first_cpu_jac_s=time.perf_counter()-started
    started=time.perf_counter(); first_gpu_phys_jac=gpu_jac_physical.jacobian(x0_physical,scale); gpu.synchronize(); first_gpu_jac_s=time.perf_counter()-started
    first_cpu_solver_jac=solver_jacobian_from_physical(first_cpu_phys_jac,x0_physical); first_gpu_solver_jac=solver_jacobian_from_physical(first_gpu_phys_jac,x0_physical); initial_columns=column_metrics(first_cpu_solver_jac,first_gpu_solver_jac); initial_aggregate=aggregate_case_metrics([initial_columns])
    cpu_fun=TimedResidual(cpu,observed,scale,config.loss); cpu_result,cpu_optimizer_s,cpu_trajectory,cpu_callback=run_local(cpu_fun,x0_solver,config,'2-point'); cpu_values=solver_to_physical(cpu_result.x); cpu_prediction=cpu.predict(cpu_values)
    gpu_fun=TimedResidual(gpu,observed,scale,config.loss); gpu_jac=TimedGpuJacobian(gpu,scale); gpu_result,gpu_optimizer_s,gpu_trajectory,gpu_callback=run_local(gpu_fun,x0_solver,config,gpu_jac); gpu_values=solver_to_physical(gpu_result.x); gpu_prediction=gpu.predict(gpu_values); gpu.synchronize()
    optimizers_completed_utc=datetime.now(timezone.utc).isoformat()
    truth,noise_audit=v10.load_evaluation_truth(args.input)
    started=time.perf_counter(); cpu_final_phys_jac=formal_cpu_jacobian(cpu.model,cpu_values,observed,scale); cpu_final_jac_s=time.perf_counter()-started
    started=time.perf_counter(); gpu_final_phys_jac=gpu_jac_physical.jacobian(gpu_values,scale); gpu.synchronize(); gpu_final_jac_s=time.perf_counter()-started
    cpu_rank=rank_diagnostics(cpu_final_phys_jac); gpu_rank=rank_diagnostics(gpu_final_phys_jac)
    resident_after=resident_arrays(gpu); resident_reused=all(resident_before[name]['pointer']==resident_after[name]['pointer'] for name in resident_before)
    cpu_timing={'total_runtime_s':cpu_optimizer_s,'backend_setup_runtime_s':cpu_setup,'warmup_runtime_s':0.0,'residual_runtime_s':cpu_fun.runtime_s,'jacobian_runtime_s':None,'jacobian_runtime_note':'Formal V10 uses SciPy internal 2-point Jacobian; its time is included in residual callbacks and cannot be separated safely.','optimizer_overhead_s':max(0.0,cpu_optimizer_s-cpu_fun.runtime_s),'residual_callback_calls':cpu_fun.calls,'explicit_jacobian_calls':0,'postfit_rank_jacobian_runtime_s':cpu_final_jac_s}
    gpu_timing={'total_runtime_s':gpu_optimizer_s,'backend_setup_runtime_s':gpu_setup,'warmup_runtime_s':gpu_warmup,'residual_runtime_s':gpu_fun.runtime_s,'jacobian_runtime_s':gpu_jac.runtime_s,'optimizer_overhead_s':max(0.0,gpu_optimizer_s-gpu_fun.runtime_s-gpu_jac.runtime_s),'residual_callback_calls':gpu_fun.calls,'explicit_jacobian_calls':gpu_jac.calls,'postfit_rank_jacobian_runtime_s':gpu_final_jac_s}
    cpu_payload=result_payload(cpu_result,cpu_values,cpu_prediction,observed,cpu_timing,cpu_trajectory,cpu_callback,cpu_rank,truth); gpu_payload=result_payload(gpu_result,gpu_values,gpu_prediction,observed,gpu_timing,gpu_trajectory,gpu_callback,gpu_rank,truth)
    differences=[]
    for i,name in enumerate(PARAMETER_NAMES):
        absolute=abs(float(gpu_values[i]-cpu_values[i])); differences.append({'parameter':name,'CPU':float(cpu_values[i]),'GPU':float(gpu_values[i]),'absolute_difference':absolute,'relative_difference':absolute/max(abs(float(cpu_values[i])),1e-15),'difference_as_bounds_span_fraction':absolute/float(span[i])})
    spectrum_delta=gpu_prediction-cpu_prediction; final_response={'rmse':float(np.sqrt(np.mean(spectrum_delta**2))),'max_abs':float(np.max(np.abs(spectrum_delta))),'exact_RMSE_absolute_difference':abs(gpu_payload['exact_RMSE']-cpu_payload['exact_RMSE']),'cost_absolute_difference':abs(gpu_payload['cost']-cpu_payload['cost'])}
    same_parameters=all(row['difference_as_bounds_span_fraction'] <= PARAMETER_EQUIVALENCE_SPAN_FRACTION for row in differences); equivalent_response=final_response['rmse'] <= FINAL_RESPONSE_RMSE_LIMIT
    closure_pass=residual_closure['rmse'] <= RESIDUAL_RMSE_LIMIT and residual_closure['max_abs'] <= RESIDUAL_MAX_ABS_LIMIT and residual_closure['nan_inf_count']==0
    for row in initial_aggregate:
        closure_pass &= (row['max_relative_l2_error'] is None or row['max_relative_l2_error'] <= JAC_RELATIVE_LIMIT) and (row['min_cosine_similarity'] is None or row['min_cosine_similarity'] >= JAC_COSINE_MIN) and row['max_abs_difference'] <= JAC_MAX_ABS_LIMIT
    acceptance={'A_initial_residual_jacobian_closure':bool(closure_pass),'B_both_optimizers_success':bool(cpu_payload['success'] and gpu_payload['success']),'B_same_parameters':same_parameters,'B_numerically_equivalent_effective_response':equivalent_response,'C_air_angle_identifiability_interpretation_required':bool(not same_parameters and equivalent_response),'D_optimizer_runtime_speedup':cpu_optimizer_s/gpu_optimizer_s,'pass':bool(closure_pass and cpu_payload['success'] and gpu_payload['success'] and (same_parameters or equivalent_response))}
    report={'phase':'Phase 4 single-NPZ GPU local optimizer integration','created_utc':datetime.now(timezone.utc).isoformat(),'formal_v10_sha256':source_sha256(),'input_npz':str(args.input.resolve()),'input_npz_sha256':sha256_file(args.input),'machine':machine_info,'input_metadata':measurement['metadata'],'single_npz_only':True,'seed':int(config.random_seed),'truth_isolation':{'truth_loaded_after_both_optimizers':True,'optimizers_completed_utc':optimizers_completed_utc,'truth_fields_not_used_for_start_or_fit':True},'starting_point_source':'first member of formal V10 LatinHypercube population using the formal seed; no differential evolution and no truth','starting_point_physical':x0_physical.tolist(),'starting_point_solver':x0_solver.tolist(),'population_size_contract':population_size,'optimizer_configuration':{'loss':config.loss,'max_nfev':config.max_nfev,'x_scale':1.0,'ftol':1e-8,'xtol':1e-8,'gtol':config.local_gtol,'solver_bounds':[0.0,1.0],'CPU_jacobian':'SciPy default 2-point as in formal V10','GPU_jacobian':'validated B=13 physical center difference transformed by exact formal solver-coordinate derivative'},'initial_numerical_closure':{'residual':residual_closure,'jacobian_columns':initial_aggregate,'cpu_jacobian_runtime_s':first_cpu_jac_s,'gpu_jacobian_runtime_s':first_gpu_jac_s},'CPU':cpu_payload,'GPU':gpu_payload,'parameter_comparison':differences,'effective_optical_response_comparison':final_response,'first_optimizer_step':{'CPU':first_step(cpu_trajectory,x0_solver),'GPU':first_step(gpu_trajectory,x0_solver)},'resident_gpu_arrays':{'before':resident_before,'after':resident_after,'pointers_unchanged':resident_reused,'contract':'wavelength, material n/k, k0, source/QE/electron weights, ILS interpolation and reference response are uploaded once during backend construction and reused; only B=1/B=13 parameters and SciPy-required outputs cross PCIe per callback.'},'memory':gpu.memory_stats(),'truth':truth,'noise_audit':noise_audit,'acceptance':acceptance,'scope_guard':'Single NPZ strict local fitting only. No formal V10 edits, global GPU search, dataset batch, JAX, autodiff, multi-GPU, parameter, bound, residual, loss, x_scale, tolerance, convergence, rank or truth-isolation changes.'}
    trace=output/'phase4_trace_arrays.npz'; np.savez_compressed(trace,initial_cpu_residual=first_cpu_res,initial_gpu_residual=first_gpu_res,initial_cpu_solver_jacobian=first_cpu_solver_jac,initial_gpu_solver_jacobian=first_gpu_solver_jac,final_cpu_spectrum=cpu_prediction,final_gpu_spectrum=gpu_prediction,cpu_trajectory_solver=np.asarray([row['solver'] for row in cpu_trajectory],dtype=np.float64).reshape((-1,6)),gpu_trajectory_solver=np.asarray([row['solver'] for row in gpu_trajectory],dtype=np.float64).reshape((-1,6)))
    report['trace_archive']={'path':str(trace),'sha256':sha256_file(trace)}
    json_path=output/'phase4_result.json'; json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=json_default)+'\n',encoding='utf-8',newline='\n')
    lines=['# Phase 4 single-NPZ CPU/GPU local optimizer result','',f"- Overall: **{'PASS' if acceptance['pass'] else 'FAIL'}**",f"- Input: `{args.input.name}`",f"- Seed: `{config.random_seed}`",f"- CPU optimizer runtime: `{cpu_optimizer_s:.6f} s`",f"- GPU optimizer runtime: `{gpu_optimizer_s:.6f} s`",f"- Speedup: `{acceptance['D_optimizer_runtime_speedup']:.2f}x`",f"- Final effective-response CPU/GPU RMSE: `{final_response['rmse']:.3e}`",f"- Same parameters: `{same_parameters}`",f"- Numerically equivalent response: `{equivalent_response}`",'', '## Optimizer results','', '| backend | success | status | cost | optimality | exact RMSE | nfev | njev | rank |', '|---|:---:|---:|---:|---:|---:|---:|---:|---:|',f"| CPU | {cpu_payload['success']} | {cpu_payload['status']} | {cpu_payload['cost']:.6e} | {cpu_payload['optimality']:.3e} | {cpu_payload['exact_RMSE']:.3e} | {cpu_payload['nfev']} | {cpu_payload['njev']} | {cpu_rank['rank']} |",f"| GPU | {gpu_payload['success']} | {gpu_payload['status']} | {gpu_payload['cost']:.6e} | {gpu_payload['optimality']:.3e} | {gpu_payload['exact_RMSE']:.3e} | {gpu_payload['nfev']} | {gpu_payload['njev']} | {gpu_rank['rank']} |",'',f"- CPU termination message: `{cpu_payload['message']}`",f"- GPU termination message: `{gpu_payload['message']}`",'', '## Parameter consistency','', '| parameter | CPU | GPU | abs difference | relative difference | span fraction |','|---|---:|---:|---:|---:|---:|']
    for row in differences: lines.append(f"| {row['parameter']} | {row['CPU']:.12g} | {row['GPU']:.12g} | {row['absolute_difference']:.3e} | {row['relative_difference']:.3e} | {row['difference_as_bounds_span_fraction']:.3e} |")
    lines += ['', '## Errors and boundaries','',f"- CPU Air error: `{cpu_payload['Air_error_nm']:.6g} nm`; film MAE: `{cpu_payload['film_MAE_nm']:.6g} nm`; Angle error: `{cpu_payload['Angle_error_deg']:.6g} deg`; boundary hits: `{cpu_payload['boundary_hits']}`",f"- GPU Air error: `{gpu_payload['Air_error_nm']:.6g} nm`; film MAE: `{gpu_payload['film_MAE_nm']:.6g} nm`; Angle error: `{gpu_payload['Angle_error_deg']:.6g} deg`; boundary hits: `{gpu_payload['boundary_hits']}`",'', '## Timing decomposition','',f"- CPU residual callbacks: `{cpu_fun.runtime_s:.6f} s`; CPU internal finite-difference Jacobian time is included and not safely separable.",f"- GPU residual: `{gpu_fun.runtime_s:.6f} s`; GPU Jacobian: `{gpu_jac.runtime_s:.6f} s`; optimizer overhead: `{gpu_timing['optimizer_overhead_s']:.6f} s`.",f"- GPU resident constant-array pointers unchanged: `{resident_reused}`",'', '## Numerical-path interpretation','',f"- Initial residual closure RMSE/max abs: `{residual_closure['rmse']:.3e}` / `{residual_closure['max_abs']:.3e}`.",f"- Initial Jacobian closure: `{'PASS' if closure_pass else 'FAIL'}`.",f"- First-step and termination trajectories are stored in JSON; callback supported CPU/GPU: `{cpu_callback}/{gpu_callback}`."]
    if not same_parameters and equivalent_response: lines.append('- CPU and GPU reached different Air/Angle coordinates but an equivalent optical response; Phase 3 Air-Angle near-collinearity supports a flat-valley explanation rather than a GPU-error attribution.')
    elif same_parameters: lines.append('- CPU and GPU reached the same parameter solution within the declared span-relative tolerance.')
    else: lines.append('- Final responses are not numerically equivalent; inspect the stored first residual, first Jacobian, first step and termination trajectories before attribution.')
    lines += ['', '## Scope','',report['scope_guard']]
    (output/'phase4_result_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pass':acceptance['pass'],'output':str(json_path),'acceptance':acceptance},indent=2))
    if not acceptance['pass']: raise SystemExit(2)
if __name__=='__main__': main()
