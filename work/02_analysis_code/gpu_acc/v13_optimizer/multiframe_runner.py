"""GPU comparison of single, averaged, and shared-structure joint frame fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

import tmm_joint_inversion_v12 as v12
from v10_gpu.spectrometer import NumpyStrictSpectrometerBackend, CupyStrictSpectrometerBackend
from v10_gpu.backend.v10_source import source_sha256
from v10_gpu.global_search import robust_objective_batch_device


EXPECTED_V10 = "d0c5076deef971a3cce169220e37072fa8dd61a24f6a21401707f6150474b321"
MODES = ("single", "average", "joint")


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def scalar(data, key, default=None):
    return np.asarray(data[key]).item() if key in data else default


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_group(path: Path, config):
    base = v12.load_fit_input(path, config)
    with np.load(path, allow_pickle=False) as data:
        frames = np.asarray(data["spectra_measured"], dtype=np.float64)
        if frames.shape != (16, 18001):
            raise RuntimeError(f"expected 16 x 18001 frames, got {frames.shape}")
        if str(scalar(data, "multiframe_provenance", "")).find("StackRT") < 0:
            raise RuntimeError("multiframe group lacks StackRT provenance")
        source = str(scalar(data, "multiframe_source_filename", ""))
    base.update({"frames": frames, "source_filename": source})
    return base


def observed_for_mode(group, mode):
    frames = group["frames"]
    if mode == "single": return frames[:1]
    if mode == "average": return np.mean(frames, axis=0, keepdims=True)
    if mode == "joint": return frames
    raise ValueError(mode)


class MultiFrameObjective:
    def __init__(self, backend, observed, scale, angle, config):
        self.backend, self.cp = backend, backend.cp
        self.scale, self.angle, self.loss = float(scale), float(angle), str(config.loss)
        self.margin, self.weight = config.boundary_margin_fraction, config.boundary_penalty_weight
        self.lower, self.upper = v12.bounds_arrays(); self.span = self.upper - self.lower
        with backend.device:
            self.observed = self.cp.asarray(observed, dtype=self.cp.float64)
            self.lower_d = self.cp.asarray(self.lower); self.span_d = self.cp.asarray(self.span)
        self.calls = self.candidates = 0; self.batch_sizes = []; self.runtime_s = 0.0

    def __call__(self, values):
        started = time.perf_counter()
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        elif x.ndim == 2 and x.shape[0] == 5: x = x.T
        if x.ndim != 2 or x.shape[1] != 5: raise ValueError(f"invalid population {x.shape}")
        self.calls += 1; self.candidates += len(x); self.batch_sizes.append(len(x))
        cp = self.cp
        with self.backend.device:
            free = cp.asarray(np.ascontiguousarray(x), dtype=cp.float64)
            full = cp.concatenate((free, cp.full((len(x), 1), self.angle)), axis=1)
            prediction = self.backend.predict_batch_device(full)
            residual = (prediction[:, None, :] - self.observed[None, :, :]) / self.scale
            spectrum = robust_objective_batch_device(residual.reshape(len(x), -1), self.loss, cp)
            normalized = (free - self.lower_d[None, :]) / self.span_d[None, :]
            distance = cp.minimum(normalized, 1.0 - normalized)
            severity = cp.sum(cp.square(cp.clip((self.margin-distance)/self.margin, 0.0, 1.0)), axis=1)
            result = cp.asnumpy(spectrum * (1.0 + self.weight * severity))
        self.runtime_s += time.perf_counter() - started
        return result

    def profile(self):
        return {"objective_calls": self.calls, "candidate_evaluations": self.candidates,
                "actual_batch_sizes": self.batch_sizes, "mean_batch_size": float(np.mean(self.batch_sizes)),
                "runtime_s": self.runtime_s}


class MultiFrameCache:
    def __init__(self, backend, observed, scale, angle):
        self.backend, self.observed = backend, np.asarray(observed)
        self.scale, self.angle = float(scale), float(angle)
        self.lower, self.upper = v12.bounds_arrays(); self.span = self.upper-self.lower
        self.cached = None; self.evaluations = self.hits = self.residual_calls = self.jacobian_calls = 0
        self.runtime_s = 0.0

    def to_solver(self, free): return np.clip((np.asarray(free)-self.lower)/self.span, 0.0, 1.0)
    def from_solver(self, solver): return self.lower + np.asarray(solver)*self.span

    def evaluate(self, solver):
        solver = np.ascontiguousarray(np.asarray(solver, dtype=np.float64)); key = solver.tobytes()
        if self.cached is not None and self.cached[0] == key: self.hits += 1; return self.cached[1]
        plus = np.repeat(solver[None, :], 5, axis=0); minus = plus.copy(); ii=np.arange(5)
        plus[ii,ii]=np.minimum(1.0,solver+1e-6); minus[ii,ii]=np.maximum(0.0,solver-1e-6)
        denominator=plus[ii,ii]-minus[ii,ii]
        batch=self.from_solver(np.vstack((solver[None,:],plus,minus)))
        full=np.c_[batch,np.full(11,self.angle)]
        started=time.perf_counter(); spectra=self.backend.predict_batch(full); self.runtime_s += time.perf_counter()-started
        residual=((spectra[0][None,:]-self.observed)/self.scale).reshape(-1)
        spectral_jac=((spectra[1:6]-spectra[6:11])/denominator[:,None]/self.scale).T
        jacobian=np.tile(spectral_jac,(self.observed.shape[0],1))
        value={"solver":solver,"free":batch[0],"full":full[0],"prediction":spectra[0],
               "residual":residual,"jacobian":jacobian}
        self.cached=(key,value); self.evaluations += 1; return value

    def residual(self, solver): self.residual_calls += 1; return self.evaluate(solver)["residual"]
    def jacobian(self, solver): self.jacobian_calls += 1; return self.evaluate(solver)["jacobian"]
    def snapshot(self): return {"local_batch_size":11,"actual_gpu_batch_evaluations":self.evaluations,
        "cache_hits":self.hits,"residual_calls":self.residual_calls,"jacobian_calls":self.jacobian_calls,
        "gpu_batch_runtime_s":self.runtime_s}


def fit_mode(group, mode, config, seed, gpu, cpu):
    observed = observed_for_mode(group, mode)
    scale = v12.v10.robust_scale(np.mean(group["frames"], axis=0))
    population = v12.latin_hypercube_population(seed, 40)
    objective = MultiFrameObjective(gpu, observed, scale, group["fixed_angle_deg"], config)
    global_started=time.perf_counter()
    de=differential_evolution(objective,bounds=[v12.v10.BOUNDS[n] for n in v12.FREE_PARAMS],
        strategy="best1bin",maxiter=config.global_maxiter,popsize=8,tol=1e-7,mutation=(0.5,1.0),
        recombination=0.7,seed=seed,polish=False,init=population,workers=1,updating="deferred",vectorized=True)
    global_runtime=time.perf_counter()-global_started
    candidates=v12.select_diverse_candidates(de.population,de.population_energies,config.multistarts)
    if len(candidates) != config.multistarts: raise RuntimeError("not enough diverse candidates")
    cache=MultiFrameCache(gpu,observed,scale,group["fixed_angle_deg"]); attempts=[]
    local_started=time.perf_counter()
    for call_index,(start,population_index,energy) in enumerate(candidates,1):
        result=least_squares(cache.residual,x0=cache.to_solver(start),jac=cache.jacobian,
            bounds=(np.zeros(5),np.ones(5)),loss=config.loss,max_nfev=config.max_nfev,
            x_scale=1.0,ftol=1e-8,xtol=1e-8,gtol=config.local_gtol)
        evaluation=cache.evaluate(result.x)
        attempts.append({"call_index":call_index,"population_index":population_index,
            "global_total_objective":energy,"final_solver":result.x,"final_free":evaluation["free"],
            "success":bool(result.success),"status":int(result.status),"message":str(result.message),
            "spectrum_cost":float(result.cost),"optimality":float(result.optimality),"nfev":int(result.nfev),
            "njev":None if result.njev is None else int(result.njev)})
    local_runtime=time.perf_counter()-local_started
    ranked=v12.select_local_candidate(attempts,config); selected=ranked["selected"]
    if selected is None: raise RuntimeError("no valid local candidate")
    evaluation=cache.evaluate(selected["final_solver"]); cpu_prediction=cpu.predict(evaluation["full"])
    delta=evaluation["prediction"]-cpu_prediction
    closure={"rmse":float(np.sqrt(np.mean(delta*delta))),"max_abs":float(np.max(np.abs(delta)))}
    closure["pass"]=closure["rmse"]<=1e-10 and closure["max_abs"]<=1e-8
    if not closure["pass"]: raise RuntimeError(f"CPU/GPU closure failed: {closure}")
    residuals=evaluation["prediction"][None,:]-observed
    singular=np.linalg.svd(evaluation["jacobian"],compute_uv=False)
    return {"mode":mode,"free_parameters":evaluation["free"],"selected":selected,"ranking":ranked,
        "rmse_all_frames":float(np.sqrt(np.mean(residuals**2))),
        "rmse_mean_spectrum":float(np.sqrt(np.mean((evaluation["prediction"]-np.mean(group["frames"],axis=0))**2))),
        "condition_number":None if singular[-1]==0 else float(singular[0]/singular[-1]),
        "smallest_singular_value":float(singular[-1]),
        "closure":closure,"global_runtime_s":global_runtime,"local_runtime_s":local_runtime,
        "global_profile":objective.profile(),"cache":cache.snapshot()}


def stats(values):
    a=np.fromiter(values,dtype=np.float64)
    if a.size == 0:
        raise ValueError("statistics require at least one finite value")
    if not np.all(np.isfinite(a)):
        raise ValueError("statistics received NaN/Inf")
    return {"mean":float(np.mean(a)),"median":float(np.median(a)),
        "p95":float(np.percentile(a,95)),"max":float(np.max(a))}


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--machine",type=Path)
    parser.add_argument("--count",type=int,default=10); parser.add_argument("--global-maxiter",type=int,default=40)
    parser.add_argument("--max-nfev",type=int,default=600); return parser.parse_args()


def main():
    args=parse_args()
    if source_sha256()!=EXPECTED_V10: raise RuntimeError("formal V10 source hash changed")
    paths=sorted(args.input_dir.glob("multiframe_g*.npz"))[:args.count]
    if len(paths)!=args.count: raise RuntimeError(f"requested {args.count}, found {len(paths)}")
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    config=v12.FitConfig(input_dir=str(args.input_dir.resolve()),wavelength_min_nm=220.0,wavelength_max_nm=580.0,
        global_forward_model="full_ils",global_popsize=8,global_maxiter=args.global_maxiter,multistarts=8,
        max_nfev=args.max_nfev,workers=1,random_seed=20260902)
    first=load_group(paths[0],config)
    gpu=CupyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
    cpu=NumpyStrictSpectrometerBackend(first["wavelengths_um"],first["generator_config"],first["metadata"]["internal_wavelength_margin_nm"])
    rows=[]; started=time.perf_counter(); progress=output/"v13_multiframe_progress.jsonl"
    for index,path in enumerate(paths,1):
        group=load_group(path,config); truth,_=v12.load_evaluation_truth(path)
        seed=20260902+index*1009
        for mode in MODES:
            fit=fit_mode(group,mode,config,seed,gpu,cpu); free=np.asarray(fit["free_parameters"])
            row={"group":index,"filename":path.name,"source_filename":group["source_filename"],"mode":mode,
                "Air_error_nm":float((free[0]-truth["Air"])*1000),"absolute_Air_error_nm":float(abs(free[0]-truth["Air"])*1000),
                "film_MAE_nm":float(np.mean([abs(free[i]-truth[name]) for i,name in enumerate(v12.FREE_PARAMS[1:],1)])),
                **fit}
            rows.append(row)
            with progress.open("a",encoding="utf-8") as handle: handle.write(json.dumps(safe(row),ensure_ascii=False)+"\n")
            print(f"[{index}/{len(paths)}] {mode} Air={row['Air_error_nm']:.5g} nm",flush=True)
    aggregates={mode:{metric:stats(row[metric] for row in rows if row["mode"]==mode)
        for metric in ("absolute_Air_error_nm","film_MAE_nm","rmse_mean_spectrum","condition_number","smallest_singular_value")} for mode in MODES}
    report={"overall":"PASS" if all(r["closure"]["pass"] for r in rows) else "FAIL",
        "created_utc":datetime.now(timezone.utc).isoformat(),"configuration":{**asdict(config),"stage":4,
        "wavelength_nm":[220.0,580.0],"sampling_nm":0.02,"wavelength_points":18001,"frame_count":16,
        "modes":list(MODES),"angle_mode":"fixed_independent_measurement","angle_sigma_deg":0.001,
        "same_initial_population_across_modes":True,"shared_scale":"robust_scale(mean_of_16_frames)"},
        "formal_v10_sha256":source_sha256(),"input_manifest_sha256":sha256(args.input_dir/"multiframe_manifest.json"),
        "machine":json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None,
        "runtime_s":time.perf_counter()-started,"aggregates":aggregates,"cases":rows,
        "scope_guard":"V13 Stage 4 only; 220-580 nm; detector_typical StackRT-derived repeated frames; V12/V10 unchanged."}
    (output/"v13_multiframe_results.json").write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fields=["group","filename","source_filename","mode","Air_error_nm","absolute_Air_error_nm","film_MAE_nm",
        "rmse_all_frames","rmse_mean_spectrum","condition_number","smallest_singular_value","global_runtime_s","local_runtime_s"]
    with (output/"v13_multiframe_table.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key:row[key] for key in fields})
    lines=["# V13 multi-frame comparison","","- Band: 220-580 nm; 0.02 nm; 18,001 points","- Frames/group: 16","",
        "| mode | Air abs mean (nm) | film MAE mean (nm) | mean-spectrum RMSE | condition number | sigma_min |","|---|---:|---:|---:|---:|---:|"]
    for mode in MODES: lines.append(f"| {mode} | {aggregates[mode]['absolute_Air_error_nm']['mean']:.6g} | {aggregates[mode]['film_MAE_nm']['mean']:.6g} | {aggregates[mode]['rmse_mean_spectrum']['mean']:.6g} | {aggregates[mode]['condition_number']['mean']:.6g} | {aggregates[mode]['smallest_singular_value']['mean']:.6g} |")
    (output/"v13_multiframe_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"overall":report["overall"],"cases":len(rows),"runtime_s":report["runtime_s"]},indent=2))
    if report["overall"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
