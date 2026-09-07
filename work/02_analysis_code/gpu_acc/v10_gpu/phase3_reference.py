"""Generate at least 20 formal CPU center-difference Jacobian references."""
from __future__ import annotations
import argparse,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .backend.v10_source import load_v10_module,source_path,source_sha256
from .jacobian import PARAMETER_NAMES,formal_cpu_jacobian
from .reference import array_sha256,sha256_file
from .spectrometer import NumpyStrictSpectrometerBackend

REFERENCE_SEED=20260828
REFERENCE_CASES=20

def validation_parameters(count: int=REFERENCE_CASES,seed: int=REFERENCE_SEED) -> np.ndarray:
 if count < 20: raise ValueError("Phase 3 requires at least 20 parameter cases.")
 v10=load_v10_module(); lower,upper=v10.bounds_arrays(); center=(lower+upper)/2.0
 rng=np.random.default_rng(seed); interior=lower+(.05+.90*rng.random((count-3,6)))*(upper-lower)
 return np.vstack((center,lower,upper,interior)).astype(np.float64,copy=False)

def generate(reference: Path,output: Path,manifest_path: Path) -> dict:
 reference=reference.resolve(); output=output.resolve(); manifest_path=manifest_path.resolve()
 with np.load(reference,allow_pickle=False) as data:
  reported=np.asarray(data["reported_wavelengths_um"],dtype=np.float64); config=json.loads(str(data["generator_config_json"].item())); margin=float(data["internal_margin_nm"].item()); observed=np.asarray(data["spectra"][0],dtype=np.float64)
 v10=load_v10_module(); scale=float(v10.robust_scale(observed)); params=validation_parameters(); backend=NumpyStrictSpectrometerBackend(reported,config,margin)
 started=time.perf_counter(); jacobians=np.stack([formal_cpu_jacobian(backend.model,row,observed,scale) for row in params]); runtime=time.perf_counter()-started
 if not np.all(np.isfinite(jacobians)): raise RuntimeError("CPU Jacobian reference contains NaN/Inf.")
 output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output,reported_wavelengths_um=reported,parameters=params,observed=observed,residual_scale=np.asarray(scale,dtype=np.float64),jacobians=jacobians,parameter_names=np.asarray(PARAMETER_NAMES),generator_config_json=np.asarray(json.dumps(config,sort_keys=True)),internal_margin_nm=np.asarray(margin,dtype=np.float64),source_v10_sha256=np.asarray(source_sha256()),phase2_reference_sha256=np.asarray(sha256_file(reference)),reference_seed=np.asarray(REFERENCE_SEED,dtype=np.int64))
 manifest={"phase":"Phase 3 CPU bounded center-difference Jacobian freeze","created_utc":datetime.now(timezone.utc).isoformat(),"formal_v10_source":str(source_path().resolve()),"formal_v10_sha256":source_sha256(),"phase2_reference":str(reference),"phase2_reference_sha256":sha256_file(reference),"output_npz":str(output),"output_npz_sha256":sha256_file(output),"reference_seed":REFERENCE_SEED,"case_count":int(params.shape[0]),"parameter_names":list(PARAMETER_NAMES),"batch_contract":"B=13 ordered as base + six positive perturbations + six negative perturbations","step_contract":"max(1e-6 * formal parameter span, 1e-8), clipped to formal bounds, actual plus-minus denominator","residual_contract":"(strict_response - observed) / formal robust_scale(observed)","cpu_total_runtime_s":runtime,"arrays":{"parameters_sha256":array_sha256(params),"observed_sha256":array_sha256(observed),"jacobians_sha256":array_sha256(jacobians),"jacobians_shape":list(jacobians.shape),"nan_count":int(np.isnan(jacobians).sum()),"inf_count":int(np.isinf(jacobians).sum())},"environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__},"scope":"Six-parameter center-difference Jacobian only. No least_squares integration, optimizer changes, JAX, autodiff or Phase 4."}
 manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n"); return manifest

def main() -> None:
 package=Path(__file__).resolve().parent; refdir=package/"cpu_reference_phase3"; parser=argparse.ArgumentParser(); parser.add_argument("--phase2-reference",type=Path,default=package/"cpu_reference_phase2"/"spectrometer_reference.npz"); parser.add_argument("--output",type=Path,default=refdir/"jacobian_reference.npz"); parser.add_argument("--manifest",type=Path,default=refdir/"reference_manifest.json"); args=parser.parse_args(); print(json.dumps(generate(args.phase2_reference,args.output,args.manifest),ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
