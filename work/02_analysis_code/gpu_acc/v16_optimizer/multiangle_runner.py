"""V16 Stage 3 EQ-99X single-angle versus two-angle GPU joint fitting."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution,least_squares
import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import load_v10_module,source_sha256
from v10_gpu.global_search import robust_objective_batch_device
from v10_gpu.phase5_runner import resident_array_identity
from .eq99x_spectrometer import Eq99xCupyStrictSpectrometerBackend,Eq99xNumpyStrictSpectrometerBackend
from .information import whitened_information_metrics
from .noise_covariance import estimate_diagonal_sigma,sha256_file

VERSION='v16_stage3_nonzero_multiangle_joint_gpu';EXPECTED_V10='d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321'
BANDS={'220-580':(220.0,580.0),'200-800':(200.0,800.0)};MODES=('single_primary','joint_two_angle')

def safe(x): return v12.safe(x)
def scalar(d,k,default=None): return np.asarray(d[k]).item() if k in d else default
def sha256(path): return sha256_file(path)
def bundle_hash(paths):
 d=hashlib.sha256();rows=[]
 for p in paths:
  h=sha256(p);rows.append({'filename':p.name,'sha256':h,'bytes':p.stat().st_size});d.update(f'{p.name}\t{h}\t{p.stat().st_size}\n'.encode())
 return d.hexdigest(),rows

def load_group(path,start,stop):
 with np.load(path,allow_pickle=False) as d:
  full_axis=np.asarray(d['reported_wavelengths_nm'],dtype=np.float64);mask=(full_axis>=start)&(full_axis<=stop);idx=np.where(mask)[0]
  spectra=np.asarray(d['spectra_measured'],dtype=np.float64)[:,idx];angles=np.asarray(d['measured_reflector_angles_deg'],dtype=np.float64)
  true_angles=np.asarray(d['true_reflector_angles_deg'],dtype=np.float64);config=json.loads(str(scalar(d,'config_json','{}')))
  if full_axis.shape!=(30001,) or full_axis[0]!=200.0 or full_axis[-1]!=800.0: raise ValueError(f'bad master axis {path.name}')
  if spectra.shape!=(2,len(idx)) or angles.shape!=(2,) or true_angles.shape!=(2,): raise ValueError(f'bad multi-angle shapes {path.name}')
  if str(scalar(d,'generator_version',''))!='main_v16_multiangle_nonzero' or 'stackrt' not in str(scalar(d,'multiangle_provenance','')).lower(): raise ValueError(f'bad provenance {path.name}')
  if config.get('SOURCE_MODEL')!='eq99x_digitized_peak_normalized': raise ValueError(f'not EQ-99X {path.name}')
  if str(scalar(d,'eq99x_source_csv_sha256',''))!=str(config.get('EQ99X_SOURCE_CSV_SHA256')): raise ValueError(f'EQ hash mismatch {path.name}')
  names=[str(x) for x in np.asarray(d['layer_names'])];layers=dict(zip(names,np.asarray(d['layer_thickness_um'],dtype=float)))
  truth={'Air':float(layers['Air']),'HSQ':float(layers['HSQ'])*1000,'PSS':float(layers['PSS'])*1000,'SOC':float(layers['SOC'])*1000,'TiO2':float(layers['TiO2'])*1000}
  metadata={'noise_case':str(scalar(d,'noise_case')),'noise_factor':str(scalar(d,'noise_factor')),'noise_level':str(scalar(d,'noise_level')),'realization_index':int(scalar(d,'realization_index')),'random_seed':int(scalar(d,'random_seed')),'internal_wavelength_margin_nm':float(scalar(d,'internal_wavelength_margin_nm'))}
 return {'wavelengths_um':full_axis[idx]/1000.0,'wavelengths_nm':full_axis[idx],'spectra':spectra,'angles':angles,'true_angles':true_angles,'generator_config':config,'truth':truth,'metadata':metadata}

def physical_batch(free,angles):
 free=np.asarray(free,dtype=np.float64);angles=np.asarray(angles,dtype=np.float64)
 return np.column_stack((np.repeat(free,len(angles),axis=0),np.tile(angles,len(free))))

class JointObjective:
 def __init__(self,backend,observed,scale,angles,config):
  self.backend,self.cp=backend,backend.cp;self.angles=np.asarray(angles);self.k=len(angles);self.scale=float(scale);self.loss=config.loss
  self.margin,self.weight=config.boundary_margin_fraction,config.boundary_penalty_weight;self.lower,self.upper=v12.bounds_arrays();self.span=self.upper-self.lower
  with backend.device:self.observed=self.cp.asarray(observed);self.lower_d=self.cp.asarray(self.lower);self.span_d=self.cp.asarray(self.span)
  self.calls=0;self.candidates=0;self.physical=0;self.batch_sizes=[];self.runtime_s=0.0
 def __call__(self,values):
  t=time.perf_counter();x=np.asarray(values,dtype=np.float64)
  if x.ndim==1:x=x[None,:]
  elif x.ndim==2 and x.shape[0]==5:x=x.T
  if x.ndim!=2 or x.shape[1]!=5:raise ValueError(x.shape)
  self.calls+=1;self.candidates+=len(x);self.physical+=len(x)*self.k;self.batch_sizes.append(len(x));cp=self.cp
  with self.backend.device:
   free=cp.asarray(x);full=cp.concatenate((cp.repeat(free,self.k,axis=0),cp.tile(cp.asarray(self.angles),len(x))[:,None]),axis=1)
   prediction=self.backend.predict_batch_device(full).reshape(len(x),self.k,-1);residual=(prediction-self.observed[None,:,:])/self.scale
   spectrum=robust_objective_batch_device(residual.reshape(len(x),-1),self.loss,cp);normalized=(free-self.lower_d)/self.span_d;distance=cp.minimum(normalized,1-normalized);severity=cp.sum(cp.square(cp.clip((self.margin-distance)/self.margin,0,1)),axis=1);out=cp.asnumpy(spectrum*(1+self.weight*severity))
  self.runtime_s+=time.perf_counter()-t;return out
 def profile(self):return {'objective_calls':self.calls,'logical_candidate_evaluations':self.candidates,'physical_spectrum_evaluations':self.physical,'logical_batch_sizes':self.batch_sizes,'angle_count':self.k,'runtime_s':self.runtime_s}

class JointCache:
 def __init__(self,backend,observed,scale,angles):
  self.backend=backend;self.observed=np.asarray(observed);self.scale=float(scale);self.angles=np.asarray(angles);self.k=len(angles);self.lower,self.upper=v12.bounds_arrays();self.span=self.upper-self.lower;self.cached=None;self.evaluations=self.hits=self.residual_calls=self.jacobian_calls=0;self.runtime_s=0.0
 def to_solver(self,x):return np.clip((np.asarray(x)-self.lower)/self.span,0,1)
 def from_solver(self,y):return self.lower+np.asarray(y)*self.span
 def evaluate(self,solver):
  solver=np.ascontiguousarray(np.asarray(solver,dtype=np.float64));key=solver.tobytes()
  if self.cached is not None and self.cached[0]==key:self.hits+=1;return self.cached[1]
  plus=np.repeat(solver[None,:],5,axis=0);minus=plus.copy();ii=np.arange(5);plus[ii,ii]=np.minimum(1,solver+1e-6);minus[ii,ii]=np.maximum(0,solver-1e-6);den=plus[ii,ii]-minus[ii,ii]
  structural=self.from_solver(np.vstack((solver[None,:],plus,minus)));full=physical_batch(structural,self.angles);t=time.perf_counter();spectra=self.backend.predict_batch(full).reshape(11,self.k,-1);self.runtime_s+=time.perf_counter()-t
  residual=((spectra[0]-self.observed)/self.scale).reshape(-1);signal_jac=((spectra[1:6]-spectra[6:11])/den[:,None,None]);jac=(signal_jac/self.scale).transpose(1,2,0).reshape(-1,5)
  out={'solver':solver,'free':structural[0],'spectra':spectra[0],'residual':residual,'jacobian':jac,'signal_jacobian':signal_jac.transpose(1,2,0).reshape(-1,5),'full':full[:self.k]};self.cached=(key,out);self.evaluations+=1;return out
 def residual(self,x):self.residual_calls+=1;return self.evaluate(x)['residual']
 def jacobian(self,x):self.jacobian_calls+=1;return self.evaluate(x)['jacobian']
 def snapshot(self):return {'structural_batch_size':11,'physical_batch_size':11*self.k,'angle_count':self.k,'gpu_batch_evaluations':self.evaluations,'cache_hits':self.hits,'residual_calls':self.residual_calls,'jacobian_calls':self.jacobian_calls,'runtime_s':self.runtime_s}

def fit(group,mode,config,seed,gpu,cpu,sigma):
 k=1 if mode=='single_primary' else 2;observed=group['spectra'][:k];angles=group['angles'][:k];scale=v12.v10.robust_scale(group['spectra'][0]);pop=v12.latin_hypercube_population(seed,40);objective=JointObjective(gpu,observed,scale,angles,config);tg=time.perf_counter()
 de=differential_evolution(objective,bounds=[v12.v10.BOUNDS[n] for n in v12.FREE_PARAMS],strategy='best1bin',maxiter=config.global_maxiter,popsize=8,tol=1e-7,mutation=(0.5,1.0),recombination=0.7,seed=seed,polish=False,init=pop,workers=1,updating='deferred',vectorized=True);global_runtime=time.perf_counter()-tg
 candidates=v12.select_diverse_candidates(de.population,de.population_energies,config.multistarts)
 if len(candidates)!=config.multistarts:raise RuntimeError('insufficient diverse candidates')
 cache=JointCache(gpu,observed,scale,angles);attempts=[];tl=time.perf_counter()
 for rank,(start,popidx,energy) in enumerate(candidates,1):
  result=least_squares(cache.residual,cache.to_solver(start),jac=cache.jacobian,bounds=(np.zeros(5),np.ones(5)),loss=config.loss,max_nfev=config.max_nfev,x_scale=1.0,ftol=1e-8,xtol=1e-8,gtol=config.local_gtol);ev=cache.evaluate(result.x)
  attempts.append({'call_index':rank,'population_index':popidx,'global_total_objective':energy,'final_solver':result.x,'final_free':ev['free'],'success':bool(result.success),'status':int(result.status),'message':str(result.message),'spectrum_cost':float(result.cost),'optimality':float(result.optimality),'nfev':int(result.nfev),'njev':None if result.njev is None else int(result.njev)})
 local_runtime=time.perf_counter()-tl;ranked=v12.select_local_candidate(attempts,config);selected=ranked['selected']
 if selected is None:raise RuntimeError('no valid candidate')
 ev=cache.evaluate(selected['final_solver']);cpu_pred=cpu.predict_batch(ev['full']);delta=ev['spectra']-cpu_pred;closure={'rmse':float(np.sqrt(np.mean(delta*delta))),'max_abs':float(np.max(np.abs(delta)))};closure['pass']=closure['rmse']<=1e-10 and closure['max_abs']<=1e-8
 if not closure['pass']:raise RuntimeError(f'closure failed {closure}')
 information=whitened_information_metrics(ev['signal_jacobian'],np.tile(sigma,k));raw=information['raw_information'];density=information['per_sample_information_density']
 return {'mode':mode,'angles_deg':angles,'free_parameters':ev['free'],'selected':selected,'ranking':{q:w for q,w in ranked.items() if q not in {'attempts','selected'}},'exact_RMSE':float(np.sqrt(np.mean((ev['spectra']-observed)**2))),'information':information,'sigma_min_Jw':raw['smallest_singular_value'],'condition_number_Jw':raw['condition_number'],'log10_det_JwT_Jw':raw['fisher_log10_determinant'],'density_sigma_min_Jw':density['smallest_singular_value'],'closure':closure,'global_runtime_s':global_runtime,'local_runtime_s':local_runtime,'global_profile':objective.profile(),'cache':cache.snapshot(),'global_nit':int(de.nit),'global_nfev':int(de.nfev)}

def stats(vals):
 a=np.asarray(list(vals),dtype=float);return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max())}
def aggregate(rows):return {m:stats(r[m] for r in rows) for m in ('absolute_Air_error_nm','film_MAE_nm','exact_RMSE','sigma_min_Jw','condition_number_Jw','log10_det_JwT_Jw','density_sigma_min_Jw')}|{'boundary_hit_rate':float(np.mean([bool(r['boundary_hits']) for r in rows]))}

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--noise-calibration-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--machine',type=Path);p.add_argument('--band',choices=list(BANDS),required=True);p.add_argument('--count',type=int,default=100);p.add_argument('--require-production-count',action='store_true');p.add_argument('--global-maxiter',type=int,default=40);p.add_argument('--max-nfev',type=int,default=600);return p.parse_args()

def main():
 args=parse_args()
 if source_sha256()!=EXPECTED_V10:raise RuntimeError('formal V10 changed')
 if args.require_production_count and args.count!=100:raise ValueError('production count must be 100')
 start,stop=BANDS[args.band];root=args.input_dir.resolve();paths=sorted(root.glob('multiangle_*_typical_*.npz'),key=lambda p:p.name)
 if args.require_production_count and len(paths)!=100:raise RuntimeError(f'expected 100 groups, found {len(paths)}')
 paths=paths[:args.count]
 if len(paths)!=args.count:raise RuntimeError(f'requested {args.count}, found {len(paths)}')
 source_bundle,source_rows=bundle_hash(paths);cov=estimate_diagonal_sigma(args.noise_calibration_dir);cov_axis=np.asarray(cov['wavelengths_nm']);mask=(cov_axis>=start)&(cov_axis<=stop);sigma=np.asarray(cov['sigma'])[mask]
 first=load_group(paths[0],start,stop)
 if sigma.shape!=first['wavelengths_nm'].shape or not np.allclose(cov_axis[mask],first['wavelengths_nm'],rtol=0,atol=1e-10):raise RuntimeError('covariance mismatch')
 config=v12.FitConfig(input_dir=str(root),wavelength_min_nm=start,wavelength_max_nm=stop,global_forward_model='full_ils',global_popsize=8,global_maxiter=args.global_maxiter,multistarts=8,max_nfev=args.max_nfev,workers=1,random_seed=20260905)
 gpu=Eq99xCupyStrictSpectrometerBackend(first['wavelengths_um'],first['generator_config'],first['metadata']['internal_wavelength_margin_nm']);cpu=Eq99xNumpyStrictSpectrometerBackend(first['wavelengths_um'],first['generator_config'],first['metadata']['internal_wavelength_margin_nm']);resident_before=resident_array_identity(gpu)
 output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);progress=output/f'v16_stage3_{args.band.replace("-","_")}_progress.jsonl';rows=[];tall=time.perf_counter()
 for index,path in enumerate(paths,1):
  group=load_group(path,start,stop)
  if not np.array_equal(group['wavelengths_um'],first['wavelengths_um']) or not np.array_equal(group['true_angles'],first['true_angles']):raise RuntimeError(f'group contract changed {path.name}')
  seed=config.random_seed+(index-1)*1009
  for mode in MODES:
   fitrow=fit(group,mode,config,seed,gpu,cpu,sigma);free=np.asarray(fitrow['free_parameters']);truth=group['truth'];air=(free[0]-truth['Air'])*1000;film=float(np.mean([abs(free[i]-truth[n]) for i,n in enumerate(v12.FREE_PARAMS[1:],1)]));selected=fitrow.pop('selected');boundary=selected['boundary']
   row={'index':index,'filename':path.name,'seed':seed,'band':args.band,'mode':mode,**group['metadata'],'measured_angles_deg':fitrow['angles_deg'],'true_angles_deg':group['true_angles'],'Air_error_nm':float(air),'absolute_Air_error_nm':float(abs(air)),'film_MAE_nm':film,'boundary_hits':';'.join(boundary['boundary_hits']),'near_boundary':';'.join(boundary['near_boundary']),'success':selected['success'],'status':selected['status'],**fitrow}
   rows.append(row)
   with progress.open('a',encoding='utf-8',newline='\n') as h:h.write(json.dumps(safe(row),ensure_ascii=False,separators=(',',':'))+'\n');h.flush();os.fsync(h.fileno())
  print(f'[{index}/{len(paths)}] {path.name} single_Air={rows[-2]["absolute_Air_error_nm"]:.4g} joint_Air={rows[-1]["absolute_Air_error_nm"]:.4g}',flush=True)
 elapsed=time.perf_counter()-tall;resident_after=resident_array_identity(gpu);groups=defaultdict(list)
 for r in rows:groups[r['mode']].append(r)
 summary={'processed_groups':len(paths),'fit_rows':len(rows),'closure_passed':sum(r['closure']['pass'] for r in rows),'resident_reused':resident_before==resident_after,'runtime_s':elapsed,'throughput_fits_per_min':len(rows)/elapsed*60,'global_runtime_s':sum(r['global_runtime_s'] for r in rows),'local_runtime_s':sum(r['local_runtime_s'] for r in rows),'logical_global_population':40,'single_global_physical_batch':40,'joint_global_physical_batch':80,'single_local_physical_batch':11,'joint_local_physical_batch':22,'fresh_population_fit_count':len(rows),'peak_gpu_memory_gib':int(gpu.memory_stats().get('memory_pool_total_bytes',0))/1024**3}
 passed=len(paths)==args.count and len(rows)==2*args.count and summary['closure_passed']==len(rows) and summary['resident_reused'] and all(r['information']['raw_information']['numerical_rank']==5 for r in rows)
 report={'version':VERSION,'overall':'PASS' if passed else 'FAIL','created_utc':datetime.now(timezone.utc).isoformat(),'configuration':{**asdict(config),'band':args.band,'effective_band_nm':[start,stop],'modes':list(MODES),'shared_parameters':list(v12.FREE_PARAMS),'angles_fixed_per_measurement':True,'angle_MAP':False,'source_model':'digitized EQ-99X','vectorized':True,'updating':'deferred'},'summary':summary,'aggregates':{m:aggregate(groups[m]) for m in MODES},'source_dataset_bundle_sha256':source_bundle,'source_files':source_rows,'input_manifest_sha256':sha256(root/'simulation_manifest.json'),'noise_covariance_audit':cov['audit'],'machine':json.loads(args.machine.read_text(encoding='utf-8')) if args.machine and args.machine.is_file() else None,'cases':rows,'scope_guard':'V16 Stage 3 nonzero-angle optimized design; same fixed two-angle design for both bands; single_primary and joint_two_angle use same group and same fresh initial population seed; five shared structure parameters; strict GPU full ILS.'}
 prefix=f'v16_stage3_{args.band.replace("-","_")}';(output/f'{prefix}_results.json').write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 fields=['index','filename','noise_case','realization_index','band','mode','Air_error_nm','absolute_Air_error_nm','film_MAE_nm','exact_RMSE','boundary_hits','sigma_min_Jw','condition_number_Jw','log10_det_JwT_Jw','density_sigma_min_Jw','success','status','global_runtime_s','local_runtime_s']
 with (output/f'{prefix}_table.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 lines=[f'# V16 Stage 3 {args.band} EQ-99X multi-angle result','',f'- Overall: **{report["overall"]}**',f'- Groups/fits/closure: {len(paths)}/{len(rows)}/{summary["closure_passed"]}',f'- Runtime: {elapsed:.3f} s','', '| mode | Air abs mean (nm) | film MAE mean (nm) | boundary rate | sigma_min(Jw) | kappa(Jw) |','|---|---:|---:|---:|---:|---:|']
 for m in MODES:lines.append(f'| {m} | {report["aggregates"][m]["absolute_Air_error_nm"]["mean"]:.6g} | {report["aggregates"][m]["film_MAE_nm"]["mean"]:.6g} | {report["aggregates"][m]["boundary_hit_rate"]:.4f} | {report["aggregates"][m]["sigma_min_Jw"]["mean"]:.6g} | {report["aggregates"][m]["condition_number_Jw"]["mean"]:.6g} |')
 (output/f'{prefix}_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'overall':report['overall'],'band':args.band,'groups':len(paths),'fits':len(rows),'runtime_s':elapsed},indent=2),flush=True)
 if not passed:raise SystemExit(2)
if __name__=='__main__':main()
