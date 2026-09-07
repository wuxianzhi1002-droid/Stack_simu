"""V16 Stage 1C equal-bandwidth information using V15 Stage 1 fixed measured angles; no inversion."""
from __future__ import annotations
import argparse,csv,json,os,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from .eq99x_spectrometer import Eq99xCupyStrictSpectrometerBackend,Eq99xNumpyStrictSpectrometerBackend
from .information import signal_jacobian_normalized_solver,whitened_information_metrics
from .multiangle_runner import EXPECTED_V10,physical_batch
from .noise_covariance import estimate_diagonal_sigma,sha256_file

VERSION="v16_stage1c_equal_bandwidth_v15_fixed_angles_information_only"
BANDS={"200-400":(200.0,400.0),"300-500":(300.0,500.0),"400-600":(400.0,600.0),"500-700":(500.0,700.0),"600-800":(600.0,800.0)}
ANGLE_CHUNK=4

def scalar(d,k,default=None):return np.asarray(d[k]).item() if k in d else default
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",type=Path,required=True);p.add_argument("--noise-calibration-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--machine",type=Path);return p.parse_args()
def load_case(path):
 with np.load(path,allow_pickle=False) as d:
  axis=np.asarray(d["reported_wavelengths_nm"],dtype=np.float64);angle=float(scalar(d,"measured_reflector_angle_deg",np.nan));config=json.loads(str(scalar(d,"config_json","{}")));names=[str(x) for x in np.asarray(d["layer_names"])];layers=dict(zip(names,np.asarray(d["layer_thickness_um"],dtype=float)))
  truth={"Air":float(layers["Air"]),"HSQ":float(layers["HSQ"])*1000,"PSS":float(layers["PSS"])*1000,"SOC":float(layers["SOC"])*1000,"TiO2":float(layers["TiO2"])*1000}
  meta={"filename":path.name,"noise_case":str(scalar(d,"noise_case")),"noise_factor":str(scalar(d,"noise_factor")),"noise_level":str(scalar(d,"noise_level")),"realization_index":int(scalar(d,"realization_index")),"random_seed":int(scalar(d,"random_seed")),"measured_angle_deg":angle,"true_angle_deg":float(scalar(d,"true_reflector_angle_deg",np.nan)),"internal_wavelength_margin_nm":float(scalar(d,"internal_wavelength_margin_nm",np.nan))}
  if str(scalar(d,"generator_version",""))!="main_v15" or str(scalar(d,"angle_measurement_mode",""))!="fixed_independent_measurement":raise ValueError(path.name)
  if config.get("SOURCE_MODEL")!="eq99x_digitized_peak_normalized" or axis.shape!=(30001,) or axis[0]!=200.0 or axis[-1]!=800.0 or not np.isfinite(angle):raise ValueError(path.name)
 return axis,config,truth,meta

def jacobian_batch(backend,free,angles):
 free=np.asarray(free,dtype=np.float64);angles=np.asarray(angles,dtype=np.float64);lower,upper=v12.bounds_arrays();span=upper-lower;solver=np.clip((free-lower)/span,0,1);step=np.full(5,1e-6);plus=np.repeat(solver[None,:],5,axis=0);minus=plus.copy();ii=np.arange(5);plus[ii,ii]=np.minimum(1,solver+step);minus[ii,ii]=np.maximum(0,solver-step);den=plus[ii,ii]-minus[ii,ii];structural=lower[None,:]+np.vstack((solver[None,:],plus,minus))*span[None,:];full=physical_batch(structural,angles);spectra=np.asarray(backend.predict_batch(full),dtype=np.float64).reshape(11,len(angles),-1);jac=((spectra[1:6]-spectra[6:11])/den[:,None,None]).transpose(1,2,0)
 if jac.shape!=(len(angles),30001,5) or np.any(~np.isfinite(jac)):raise RuntimeError(jac.shape)
 return jac

def stats(values):
 a=np.asarray(list(values),dtype=float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max())}
def main():
 a=parse_args()
 if source_sha256()!=EXPECTED_V10:raise RuntimeError("formal V10 source changed")
 root=a.input_dir.resolve();paths=sorted(root.glob("static_spectrum_*_typical_*.npz"),key=lambda p:p.name)
 if len(paths)!=100:raise RuntimeError(f"expected 100 V15 Stage 1 compact files, found {len(paths)}")
 cases=[load_case(p) for p in paths];axis,config,truth0,meta0=cases[0]
 for x in cases:
  if not np.array_equal(x[0],axis) or x[1]!=config or x[2]!=truth0 or not np.isclose(x[3]["internal_wavelength_margin_nm"],meta0["internal_wavelength_margin_nm"],rtol=0,atol=1e-15):raise RuntimeError("V15 Stage 1 contract changed")
 cov=estimate_diagonal_sigma(a.noise_calibration_dir);caxis=np.asarray(cov["wavelengths_nm"]);csigma=np.asarray(cov["sigma"])
 if not np.array_equal(caxis,axis):raise RuntimeError("noise covariance axis mismatch")
 free=np.asarray([truth0[n] for n in v12.FREE_PARAMS],dtype=np.float64);gpu=Eq99xCupyStrictSpectrometerBackend(axis/1000.0,config,meta0["internal_wavelength_margin_nm"]);cpu=Eq99xNumpyStrictSpectrometerBackend(axis/1000.0,config,meta0["internal_wavelength_margin_nm"]);before=resident_array_identity(gpu)
 gj0,audit=signal_jacobian_normalized_solver(gpu,free,meta0["measured_angle_deg"]);cj0,_=signal_jacobian_normalized_solver(cpu,free,meta0["measured_angle_deg"]);delta=gj0-cj0;closure={"rmse":float(np.sqrt(np.mean(delta*delta))),"max_abs":float(np.max(np.abs(delta))),"reference_rms":float(np.sqrt(np.mean(cj0*cj0))),"reference_max_abs":float(np.max(np.abs(cj0)))};closure["relative_rmse"]=closure["rmse"]/max(closure["reference_rms"],np.finfo(float).tiny);closure["relative_max_abs"]=closure["max_abs"]/max(closure["reference_max_abs"],np.finfo(float).tiny);closure["pass"]=closure["rmse"]<=1e-9 and closure["max_abs"]<=1e-8 and closure["relative_rmse"]<=1e-10
 rows=[];t=time.perf_counter();gpu_calls=0
 for start in range(0,len(cases),ANGLE_CHUNK):
  chunk=cases[start:start+ANGLE_CHUNK];angles=[x[3]["measured_angle_deg"] for x in chunk];jac=jacobian_batch(gpu,free,angles);gpu_calls+=1
  for offset,(_,_,_,meta) in enumerate(chunk):
   for label,(lo,hi) in BANDS.items():
    mask=(axis>=lo)&(axis<=hi);info=whitened_information_metrics(jac[offset,mask],csigma[mask]);raw=info["raw_information"];rows.append({**meta,"band":label,"bandwidth_nm":hi-lo,"wavelength_samples":int(mask.sum()),"sigma_min_Jw":raw["smallest_singular_value"],"log10_det_JwT_Jw":raw["fisher_log10_determinant"],"numerical_rank":raw["numerical_rank"]})
  print(f"[{min(start+len(chunk),len(cases))}/{len(cases)}] angle batch complete",flush=True)
 gpu_s=time.perf_counter()-t;after=resident_array_identity(gpu);grouped=defaultdict(list)
 for r in rows:grouped[r["band"]].append(r)
 aggregate={b:{"cases":len(grouped[b]),"sigma_min_Jw":stats(r["sigma_min_Jw"] for r in grouped[b]),"log10_det_JwT_Jw":stats(r["log10_det_JwT_Jw"] for r in grouped[b])} for b in BANDS};noise_groups=defaultdict(list)
 for r in rows:noise_groups[(r["band"],r["noise_case"])].append(r)
 noise_summary=[{"band":b,"noise_case":n,"realizations":len(g),"mean_sigma_min_Jw":float(np.mean([x["sigma_min_Jw"] for x in g])),"mean_log10_det_JwT_Jw":float(np.mean([x["log10_det_JwT_Jw"] for x in g]))} for (b,n),g in sorted(noise_groups.items())]
 passed=closure["pass"] and before==after and len(rows)==500 and all(r["numerical_rank"]==5 for r in rows) and all(v["cases"]==100 for v in aggregate.values())
 report={"version":VERSION,"overall":"PASS" if passed else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"configuration":{"bands_nm":{k:list(v) for k,v in BANDS.items()},"equal_bandwidth_nm":200.0,"angle_policy":"Use each V15 Stage 1 NPZ measured_reflector_angle_deg, fixed for that case","angle_count":len(cases),"measured_angle_stats_deg":stats(x[3]["measured_angle_deg"] for x in cases),"evaluation_point":"shared true Air/HSQ/PSS/SOC/TiO2 structure","source_model":"digitized EQ-99X","information_definition":"Jw = diag(1/sigma_lambda) @ d(signal)/d(normalized [0,1]^5 parameters)","inversion_or_optimization_performed":False,"angle_chunk_size":ANGLE_CHUNK},"truth":truth0,"jacobian_audit":audit,"gpu_cpu_representative_closure":closure,"gpu_information_runtime_s":gpu_s,"gpu_batch_calls":gpu_calls,"resident_reused":before==after,"peak_gpu_memory_gib":int(gpu.memory_stats().get("memory_pool_total_bytes",0))/1024**3,"input_manifest_sha256":sha256_file(next(q for q in (root/"simulation_manifest.json",root.parent/"compact_manifest.json",root.parent/"simulation_manifest.json") if q.is_file())),"noise_covariance_audit":cov["audit"],"machine":json.loads(a.machine.read_text(encoding="utf-8")) if a.machine and a.machine.is_file() else None,"aggregate":aggregate,"noise_case_means":noise_summary,"cases":rows,"scope_guard":"V15 Stage 1 measured fixed angle per NPZ; five equal 200 nm windows; truth-point information only; no cavity-length inversion."}
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);(out/"v16_stage1c_equal_bandwidth_v15_angles_results.json").write_text(json.dumps(v12.safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 fields=list(rows[0]);
 with (out/"v16_stage1c_equal_bandwidth_v15_angles_cases.csv").open("w",encoding="utf-8-sig",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 with (out/"v16_stage1c_equal_bandwidth_v15_angles_noise_means.csv").open("w",encoding="utf-8-sig",newline="") as h:w=csv.DictWriter(h,fieldnames=list(noise_summary[0]));w.writeheader();w.writerows(noise_summary)
 summary_rows=[{"band":b,"cases":aggregate[b]["cases"],"mean_sigma_min_Jw":aggregate[b]["sigma_min_Jw"]["mean"],"median_sigma_min_Jw":aggregate[b]["sigma_min_Jw"]["median"],"mean_log10_det_JwT_Jw":aggregate[b]["log10_det_JwT_Jw"]["mean"],"median_log10_det_JwT_Jw":aggregate[b]["log10_det_JwT_Jw"]["median"]} for b in BANDS]
 with (out/"v16_stage1c_equal_bandwidth_v15_angles_summary.csv").open("w",encoding="utf-8-sig",newline="") as h:w=csv.DictWriter(h,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
 lines=["# V16 Stage 1C equal-bandwidth information with V15 fixed measured angles","",f"- Overall: **{report['overall']}**",f"- Angle policy: each of 100 V15 Stage 1 measured angles is fixed for its own information calculation; range {report['configuration']['measured_angle_stats_deg']['min']:.6g}-{report['configuration']['measured_angle_stats_deg']['max']:.6g} deg.","- Each window is 200 nm and 10001 samples; no cavity-length inversion.","","| band (nm) | cases | mean sigma_min(Jw) | mean log10 det(Jw^T Jw) |","|---|---:|---:|---:|"]
 for r in summary_rows:lines.append(f"| {r['band']} | {r['cases']} | {r['mean_sigma_min_Jw']:.9g} | {r['mean_log10_det_JwT_Jw']:.9g} |")
 (out/"v16_stage1c_equal_bandwidth_v15_angles_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"overall":report["overall"],"angle_stats":report["configuration"]["measured_angle_stats_deg"],"summary":summary_rows,"gpu_runtime_s":gpu_s,"closure":closure},indent=2),flush=True)
 if not passed:raise SystemExit(2)
if __name__=="__main__":main()
