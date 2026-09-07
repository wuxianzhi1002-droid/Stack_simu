"""Exact B=13 cache in dimensionless q=theta^2 solver coordinates."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from v10_gpu.backend.v10_source import load_v10_module

ANGLE_MAX_DEG=0.1

def solver_to_physical_q(solver):
    v10=load_v10_module(); lower,upper=v10.bounds_arrays(); y=np.asarray(solver,dtype=np.float64)
    if y.shape!=(6,) or not np.all(np.isfinite(y)): raise ValueError("six finite solver values required")
    physical=lower+(upper-lower)*y
    physical[5]=ANGLE_MAX_DEG*np.sqrt(max(0.0,float(y[5])))
    return physical

def physical_to_solver_q(physical):
    v10=load_v10_module(); lower,upper=v10.bounds_arrays(); x=np.asarray(physical,dtype=np.float64)
    y=np.clip((x-lower)/(upper-lower),0.0,1.0); y[5]=(abs(float(x[5]))/ANGLE_MAX_DEG)**2
    return y

def build_q_batch(solver):
    y=np.asarray(solver,dtype=np.float64); steps=np.full(6,1.0e-6,dtype=np.float64)
    plus=np.repeat(y[None,:],6,axis=0);minus=np.repeat(y[None,:],6,axis=0);i=np.arange(6)
    plus[i,i]=np.minimum(1.0,y+steps);minus[i,i]=np.maximum(0.0,y-steps);den=plus[i,i]-minus[i,i]
    ys=np.vstack((y[None,:],plus,minus)); physical=np.vstack([solver_to_physical_q(row) for row in ys])
    return physical,den

@dataclass(frozen=True)
class QEvaluation:
    key:bytes; solver:np.ndarray; physical:np.ndarray; prediction:np.ndarray; residual:np.ndarray; jacobian:np.ndarray

class ExactQBatchCache:
    def __init__(self,backend,observed,scale):
        self.backend=backend;self.observed=np.asarray(observed,dtype=np.float64);self.scale=float(scale);self._cached=None
        self.residual_calls=0;self.jacobian_calls=0;self.evaluations=0;self.hits=0;self.gpu_batch_runtime_s=0.0
    def evaluate(self,solver):
        y=np.ascontiguousarray(np.asarray(solver,dtype=np.float64));key=y.tobytes()
        if self._cached is not None and key==self._cached.key:self.hits+=1;return self._cached
        physical_batch,den=build_q_batch(y);t=time.perf_counter();spectra=np.asarray(self.backend.predict_batch(physical_batch),dtype=np.float64);self.backend.synchronize();self.gpu_batch_runtime_s+=time.perf_counter()-t
        residual=(spectra[0]-self.observed)/self.scale;jac=((spectra[1:7]-spectra[7:13])/den[:,None]/self.scale).T
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jac)):raise FloatingPointError("q cache produced NaN/Inf")
        self._cached=QEvaluation(key,y.copy(),physical_batch[0],spectra[0],residual,jac);self.evaluations+=1;return self._cached
    def residual(self,solver):self.residual_calls+=1;return self.evaluate(solver).residual
    def jacobian(self,solver):self.jacobian_calls+=1;return self.evaluate(solver).jacobian
    def snapshot(self):return {"residual_calls":self.residual_calls,"jacobian_calls":self.jacobian_calls,"actual_gpu_batch_evaluations":self.evaluations,"cache_hits":self.hits,"gpu_batch_runtime_s":self.gpu_batch_runtime_s}
