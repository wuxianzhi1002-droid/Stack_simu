"""V16 Stage 1C equal-bandwidth sliding-window information study without inversion."""
from __future__ import annotations
import argparse,csv,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tmm_joint_inversion_v12 as v12
from v10_gpu.backend.v10_source import source_sha256
from v10_gpu.phase5_runner import resident_array_identity
from .eq99x_spectrometer import Eq99xCupyStrictSpectrometerBackend,Eq99xNumpyStrictSpectrometerBackend
from .information import signal_jacobian_normalized_solver,whitened_information_metrics
from .multiangle_runner import EXPECTED_V10,load_group
from .noise_covariance import estimate_diagonal_sigma,sha256_file

VERSION="v16_stage1c_equal_bandwidth_information_only"
BANDS={"200-400":(200.0,400.0),"300-500":(300.0,500.0),"400-600":(400.0,600.0),"500-700":(500.0,700.0),"600-800":(600.0,800.0)}
FIXED_ANGLE_DEG=0.02

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",type=Path,required=True);p.add_argument("--noise-calibration-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--machine",type=Path);return p.parse_args()

def main():
 a=parse_args()
 if source_sha256()!=EXPECTED_V10:raise RuntimeError("formal V10 source changed")
 root=a.input_dir.resolve();paths=sorted(root.glob("multiangle_*_typical_*.npz"),key=lambda p:p.name)
 if len(paths)!=100:raise RuntimeError(f"expected 100 compact groups, found {len(paths)}")
 first=load_group(paths[0],200.0,800.0)
 if not np.allclose(first["true_angles"],[0.02,0.2],rtol=0,atol=1e-12):raise RuntimeError(f"unexpected V16 angle pair: {first['true_angles']}")
 truth=first["truth"];free=np.asarray([truth[n] for n in v12.FREE_PARAMS],dtype=np.float64)
 cov=estimate_diagonal_sigma(a.noise_calibration_dir);caxis=np.asarray(cov["wavelengths_nm"]);csigma=np.asarray(cov["sigma"]);axis=np.asarray(first["wavelengths_nm"])
 if not np.array_equal(caxis,axis):raise RuntimeError("noise covariance axis mismatch")
 gpu=Eq99xCupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
 cpu=Eq99xNumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
 before=resident_array_identity(gpu);t=time.perf_counter();gj,audit=signal_jacobian_normalized_solver(gpu,free,FIXED_ANGLE_DEG);gpu_s=time.perf_counter()-t
 t=time.perf_counter();cj,_=signal_jacobian_normalized_solver(cpu,free,FIXED_ANGLE_DEG);cpu_s=time.perf_counter()-t;delta=gj-cj
 closure={"rmse":float(np.sqrt(np.mean(delta*delta))),"max_abs":float(np.max(np.abs(delta))),"reference_rms":float(np.sqrt(np.mean(cj*cj))),"reference_max_abs":float(np.max(np.abs(cj)))}
 closure["relative_rmse"]=closure["rmse"]/max(closure["reference_rms"],np.finfo(float).tiny);closure["relative_max_abs"]=closure["max_abs"]/max(closure["reference_max_abs"],np.finfo(float).tiny);closure["pass"]=closure["rmse"]<=1e-9 and closure["max_abs"]<=1e-8 and closure["relative_rmse"]<=1e-10
 rows=[];details={}
 for label,(lo,hi) in BANDS.items():
  mask=(axis>=lo)&(axis<=hi);info=whitened_information_metrics(gj[mask],csigma[mask]);raw=info["raw_information"]
  row={"band":label,"wavelength_min_nm":lo,"wavelength_max_nm":hi,"bandwidth_nm":hi-lo,"wavelength_samples":int(mask.sum()),"sigma_min_Jw":raw["smallest_singular_value"],"log10_det_JwT_Jw":raw["fisher_log10_determinant"]};rows.append(row);details[label]=info
 after=resident_array_identity(gpu);passed=closure["pass"] and before==after and all(details[k]["raw_information"]["numerical_rank"]==5 for k in BANDS) and len({r["wavelength_samples"] for r in rows})==1
 report={"version":VERSION,"overall":"PASS" if passed else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"configuration":{"bands_nm":{k:list(v) for k,v in BANDS.items()},"equal_bandwidth_nm":200.0,"fixed_angle_deg":FIXED_ANGLE_DEG,"evaluation_point":"true Air/HSQ/PSS/SOC/TiO2 structure","source_model":"digitized EQ-99X","information_definition":"Jw = diag(1/sigma_lambda) @ d(signal)/d(normalized [0,1]^5 parameters)","covariance_model":"diagonal single-frame detector covariance","inversion_or_optimization_performed":False,"finite_difference_batch_size":11},"truth":truth,"jacobian_audit":audit,"gpu_cpu_closure":closure,"gpu_runtime_s":gpu_s,"cpu_closure_runtime_s":cpu_s,"resident_reused":before==after,"peak_gpu_memory_gib":int(gpu.memory_stats().get("memory_pool_total_bytes",0))/1024**3,"source_npz":{"filename":paths[0].name,"sha256":sha256_file(paths[0])},"input_manifest_sha256":sha256_file(root/"simulation_manifest.json"),"noise_covariance_audit":cov["audit"],"machine":json.loads(a.machine.read_text(encoding="utf-8")) if a.machine and a.machine.is_file() else None,"bands":details,"rows":rows,"scope_guard":"Equal 200 nm sliding windows; information-only truth-point Jacobian; no DE, no least-squares, no Air/cavity-length inversion."}
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 (out/"v16_stage1c_equal_bandwidth_information_results.json").write_text(json.dumps(v12.safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 with (out/"v16_stage1c_equal_bandwidth_information_table.csv").open("w",encoding="utf-8-sig",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=["# V16 Stage 1C equal-bandwidth information-only result","",f"- Overall: **{report['overall']}**",f"- Fixed angle: {FIXED_ANGLE_DEG:.3f} deg; true structure; each window is 200 nm and {rows[0]['wavelength_samples']} samples.","- No differential evolution, local fitting, or cavity-length inversion.","","| band (nm) | sigma_min(Jw) | log10 det(Jw^T Jw) |","|---|---:|---:|"]
 for r in rows:lines.append(f"| {r['band']} | {r['sigma_min_Jw']:.9g} | {r['log10_det_JwT_Jw']:.9g} |")
 (out/"v16_stage1c_equal_bandwidth_information_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"overall":report["overall"],"rows":rows,"gpu_runtime_s":gpu_s,"closure":closure,"output":str(out)},indent=2),flush=True)
 if not passed:raise SystemExit(2)
if __name__=="__main__":main()
