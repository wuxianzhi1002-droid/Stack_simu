"""Validate the frozen Phase 3 CPU Jacobian artifact and hashes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .backend.v10_source import source_sha256
from .reference import array_sha256,sha256_file

def main() -> None:
 package=Path(__file__).resolve().parent; parser=argparse.ArgumentParser(); parser.add_argument("--reference",type=Path,default=package/"cpu_reference_phase3"/"jacobian_reference.npz"); parser.add_argument("--manifest",type=Path,default=package/"cpu_reference_phase3"/"reference_manifest.json"); args=parser.parse_args(); manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
 with np.load(args.reference,allow_pickle=False) as d:
  params=np.asarray(d["parameters"],dtype=np.float64); observed=np.asarray(d["observed"],dtype=np.float64); jac=np.asarray(d["jacobians"],dtype=np.float64); recorded_source=str(d["source_v10_sha256"].item()); phase2_hash=str(d["phase2_reference_sha256"].item())
 phase2=package/"cpu_reference_phase2"/"spectrometer_reference.npz"
 report={"source_hash_matches":source_sha256()==recorded_source==manifest["formal_v10_sha256"],"phase2_reference_hash_matches":sha256_file(phase2)==phase2_hash==manifest["phase2_reference_sha256"],"parameters_hash_matches":array_sha256(params)==manifest["arrays"]["parameters_sha256"],"observed_hash_matches":array_sha256(observed)==manifest["arrays"]["observed_sha256"],"jacobians_hash_matches":array_sha256(jac)==manifest["arrays"]["jacobians_sha256"],"case_count":int(params.shape[0]),"shape":list(jac.shape),"nan_inf_count":int(jac.size-np.count_nonzero(np.isfinite(jac)))}
 report["pass"]=bool(all(v for k,v in report.items() if k.endswith("_matches")) and report["case_count"]>=20 and jac.shape==(params.shape[0],observed.size,6) and report["nan_inf_count"]==0)
 print(json.dumps(report,ensure_ascii=False,indent=2))
 if not report["pass"]: raise SystemExit(2)
if __name__ == "__main__": main()
