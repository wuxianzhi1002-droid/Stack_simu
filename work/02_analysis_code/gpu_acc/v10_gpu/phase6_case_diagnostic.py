"""Targeted Phase 6 audit for one formal V10 NPZ; no model or optimizer changes."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .backend.v10_source import load_v10_module, source_sha256
from .phase5_cache import ExactBatchResidualJacobianCache
from .phase5_runner import (
    FORMAL_V10_EXPECTED_SHA256,
    GpuLeastSquaresAdapter,
    patched_formal_least_squares,
)
from .spectrometer import CupyStrictSpectrometerBackend


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def attempt_summary(fit):
    attempts=fit.get("attempts",[])
    return {
        "fit_success": bool(fit.get("success")),
        "runtime_s": float(fit.get("runtime_s",0.0)),
        "global_runtime_s": float(fit.get("global_runtime_s",0.0)),
        "local_runtime_s": float(fit.get("local_runtime_s",0.0)),
        "attempt_count": len(attempts),
        "success_count": sum(bool(a.get("success")) for a in attempts),
        "status_counts": {str(status): sum(int(a.get("status",-999))==status for a in attempts) for status in sorted({int(a.get("status",-999)) for a in attempts})},
        "attempts": clean(attempts),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    v10=load_v10_module()
    if source_sha256()!=FORMAL_V10_EXPECTED_SHA256:
        raise RuntimeError("Formal V10 SHA256 changed")
    config=v10.FitConfig(
        input_dir=str(args.input.parent),
        global_forward_model="fast_no_ils",
        global_popsize=8,
        global_maxiter=40,
        multistarts=8,
        max_nfev=600,
        workers=1,
        random_seed=20260825,
    )
    measurement=v10.load_fit_input(args.input,config)
    v10.validate_sampling(measurement,config)
    gpu=CupyStrictSpectrometerBackend(
        measurement["wavelengths_um"],
        measurement["generator_config"],
        measurement["metadata"]["internal_wavelength_margin_nm"],
    )
    cache=ExactBatchResidualJacobianCache(
        gpu,measurement["spectrum"],v10.robust_scale(measurement["spectrum"])
    )
    adapter=GpuLeastSquaresAdapter(cache)
    gpu_started=time.perf_counter()
    with patched_formal_least_squares(v10,adapter):
        gpu_fit=v10.fit_measurement(measurement,config,args.seed)
    gpu.synchronize()
    gpu_wall=time.perf_counter()-gpu_started
    cpu_started=time.perf_counter()
    cpu_fit=v10.fit_measurement(measurement,config,args.seed)
    cpu_wall=time.perf_counter()-cpu_started
    paired=[]
    cpu_attempts=cpu_fit.get("attempts",[])
    for index,ga in enumerate(adapter.calls,start=1):
        matches=[ca for ca in cpu_attempts if np.array_equal(np.asarray(ga["x0_physical"]),np.asarray(ca["x0"]))]
        if len(matches)!=1:
            raise RuntimeError("Could not uniquely pair CPU/GPU attempt by exact physical x0")
        ca=matches[0]
        gx=np.asarray(ga["final_physical"],dtype=float)
        cx=np.asarray(ca["x"],dtype=float)
        paired.append({
            "index":index,
            "same_x0":True,
            "gpu_status":ga["status"],
            "cpu_status":int(ca["status"]),
            "gpu_success":ga["success"],
            "cpu_success":bool(ca["success"]),
            "gpu_cost":ga["cost"],
            "cpu_cost":float(ca["cost"]),
            "parameter_l2_difference":float(np.linalg.norm(gx-cx)),
            "gpu_nfev":ga["nfev"],
            "cpu_nfev":int(ca["nfev"]),
        })
    report={
        "input":str(args.input.resolve()),
        "seed":args.seed,
        "formal_v10_sha256":source_sha256(),
        "configuration":asdict(config),
        "gpu":attempt_summary(gpu_fit),
        "gpu_adapter_calls":clean(adapter.calls),
        "gpu_cache":cache.snapshot(),
        "gpu_wall_s":gpu_wall,
        "cpu":attempt_summary(cpu_fit),
        "cpu_wall_s":cpu_wall,
        "paired_attempts":paired,
        "classification":"GPU-only convergence difference" if bool(gpu_fit.get("success"))!=bool(cpu_fit.get("success")) else "same formal convergence outcome",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({
        "gpu_success":report["gpu"]["fit_success"],
        "gpu_status_counts":report["gpu"]["status_counts"],
        "cpu_success":report["cpu"]["fit_success"],
        "cpu_status_counts":report["cpu"]["status_counts"],
        "classification":report["classification"],
        "output":str(args.output),
    },indent=2))
if __name__=="__main__":
    main()
