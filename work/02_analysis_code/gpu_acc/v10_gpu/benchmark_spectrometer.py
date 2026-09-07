"""Independent Phase 2 CPU/CuPy strict spectrometer benchmark."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backend.base import BackendUnavailableError
from .backend.v10_source import source_sha256
from .reference import REFERENCE_SEED, deterministic_parameters
from .spectrometer import CupyStrictSpectrometerBackend, NumpyStrictSpectrometerBackend

DEFAULT_BATCH_SIZES = (1, 7, 13, 32, 64)
RMSE_LIMIT = 1.0e-10
MAX_ABS_LIMIT = 1.0e-8


def timed_min_sync(function: Callable[[], Any], backend: Any, repeat: int) -> tuple[float, Any]:
    durations: list[float] = []
    result = None
    for _ in range(repeat):
        backend.synchronize()
        start = time.perf_counter()
        result = function()
        backend.synchronize()
        durations.append(time.perf_counter() - start)
    return min(durations), result


def benchmark_backend(backend: Any, params_bank: np.ndarray, batch_sizes: tuple[int, ...], warmup: int, repeat: int) -> dict:
    if warmup:
        for _ in range(warmup):
            backend.predict_batch(params_bank[:1])
    rows=[]
    for batch_size in batch_sizes:
        params=params_bank[:batch_size]
        upload_s, params_device = timed_min_sync(lambda: backend.parameters_to_device(params), backend, repeat)
        kernel_s, result_device = timed_min_sync(lambda: backend.predict_batch_device(params_device), backend, repeat)
        download_s, result_host = timed_min_sync(lambda: backend.to_host(result_device), backend, repeat)
        total_s, total_host = timed_min_sync(lambda: backend.predict_batch(params), backend, repeat)
        finite=np.isfinite(total_host)
        rows.append({
            "batch_size": int(batch_size),
            "upload_s": upload_s,
            "kernel_s": kernel_s,
            "download_s": download_s,
            "component_sum_s": upload_s + kernel_s + download_s,
            "total_s": total_s,
            "candidates_per_s": float(batch_size / total_s),
            "nan_count": int(np.isnan(total_host).sum()),
            "inf_count": int(np.isinf(total_host).sum()),
            "min_spectrum": float(np.min(total_host[finite])),
            "max_spectrum": float(np.max(total_host[finite])),
            "checksum_sum": float(np.sum(total_host)),
        })
    memory=backend.memory_stats()
    report={"status":"ok","backend":backend.backend_name,"timing_contract":"Each timed interval ends with explicit backend.synchronize(); total_s is one contiguous H2D+response+D2H call.","memory":memory,"rows":rows}
    if backend.backend_name == "cuda_cupy_spectrometer":
        report["peak_gpu_memory_bytes"]=int(memory.get("memory_pool_total_bytes",0))
    return report


def compare_backends(cpu: Any, gpu: Any, params_bank: np.ndarray, batch_sizes: tuple[int, ...]) -> list[dict]:
    rows=[]
    for batch_size in batch_sizes:
        cpu_values=cpu.predict_batch(params_bank[:batch_size])
        gpu_values=gpu.predict_batch(params_bank[:batch_size])
        difference=cpu_values-gpu_values
        rmse=float(np.sqrt(np.mean(difference**2)))
        max_abs=float(np.max(np.abs(difference)))
        nan_inf=int(gpu_values.size-np.count_nonzero(np.isfinite(gpu_values)))
        rows.append({"batch_size":batch_size,"rmse":rmse,"max_abs":max_abs,"nan_inf_count":nan_inf,"pass":bool(rmse <= RMSE_LIMIT and max_abs <= MAX_ABS_LIMIT and nan_inf == 0)})
    return rows


def parse_args() -> argparse.Namespace:
    package_dir=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=("cpu_numpy_spectrometer","cuda_cupy_spectrometer","both"),default="both")
    parser.add_argument("--batch-sizes",type=int,nargs="+",default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--warmup",type=int,default=1)
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument("--seed",type=int,default=REFERENCE_SEED)
    parser.add_argument("--reference",type=Path,default=package_dir/"cpu_reference_phase2"/"spectrometer_reference.npz")
    parser.add_argument("--output",type=Path,default=package_dir/"cpu_reference_phase2"/"cpu_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args=parse_args()
    batches=tuple(args.batch_sizes)
    if args.repeat < 1 or args.warmup < 0 or any(b < 1 for b in batches):
        raise ValueError("Invalid repeat, warmup or batch size.")
    with np.load(args.reference,allow_pickle=False) as data:
        reported=np.asarray(data["reported_wavelengths_um"],dtype=np.float64)
        config=json.loads(str(data["generator_config_json"].item()))
        margin=float(data["internal_margin_nm"].item())
    params=deterministic_parameters(max(batches),args.seed)
    report={"phase":"Phase 2 strict spectrometer response","created_utc":datetime.now(timezone.utc).isoformat(),"formal_v10_sha256":source_sha256(),"reference_file":str(args.reference.resolve()),"batch_sizes":list(batches),"warmup":args.warmup,"repeat":args.repeat,"backends":{}}
    cpu=None
    gpu=None
    if args.backend in {"cpu_numpy_spectrometer","both"}:
        cpu=NumpyStrictSpectrometerBackend(reported,config,margin)
        report["backends"][cpu.backend_name]=benchmark_backend(cpu,params,batches,args.warmup,args.repeat)
    if args.backend in {"cuda_cupy_spectrometer","both"}:
        try:
            gpu=CupyStrictSpectrometerBackend(reported,config,margin)
            report["backends"][gpu.backend_name]=benchmark_backend(gpu,params,batches,args.warmup,args.repeat)
        except BackendUnavailableError as exc:
            report["backends"]["cuda_cupy_spectrometer"]={"status":"unavailable","reason":str(exc)}
            if args.backend == "cuda_cupy_spectrometer":
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
                raise SystemExit(2)
    if cpu is not None and gpu is not None:
        report["cpu_gpu_validation"]=compare_backends(cpu,gpu,params,batches)
        report["cpu_gpu_validation_pass"]=all(row["pass"] for row in report["cpu_gpu_validation"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.backend == "both" and not report.get("cpu_gpu_validation_pass",False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
