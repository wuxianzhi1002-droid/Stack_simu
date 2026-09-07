"""Exact-key B=13 residual/Jacobian cache for Phase 5."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .jacobian.contract import build_center_difference_batch
from .optimizer_contract import solver_jacobian_from_physical,solver_to_physical

@dataclass(frozen=True)
class CachedEvaluation:
    key: bytes
    solver: np.ndarray
    physical: np.ndarray
    base_prediction: np.ndarray
    residual: np.ndarray
    physical_jacobian: np.ndarray
    solver_jacobian: np.ndarray
    batch_runtime_s: float

class ExactBatchResidualJacobianCache:
    """Cache only the most recent exact float64 solver vector.

    Equality is based on the contiguous six-float byte representation. A changed
    byte invalidates the cache; no tolerance or approximate comparison is used.
    """
    def __init__(self,backend,observed:np.ndarray,scale:float):
        self.backend=backend
        self.observed=np.asarray(observed,dtype=np.float64)
        self.scale=float(scale)
        if self.observed.ndim!=1 or not np.all(np.isfinite(self.observed)):
            raise ValueError("observed must be a finite one-dimensional array")
        if not np.isfinite(self.scale) or self.scale<=0.0:
            raise ValueError("scale must be finite and positive")
        self._cached:CachedEvaluation|None=None
        self.residual_calls=0; self.jacobian_calls=0
        self.actual_batch_evaluations=0; self.cache_hits=0; self.cache_misses=0
        self.batch_runtime_s=0.0; self.residual_callback_runtime_s=0.0; self.jacobian_callback_runtime_s=0.0

    @staticmethod
    def exact_key(solver:np.ndarray)->tuple[bytes,np.ndarray]:
        array=np.ascontiguousarray(np.asarray(solver,dtype=np.float64))
        if array.shape!=(6,) or not np.all(np.isfinite(array)):
            raise ValueError("solver vector must contain six finite float64 values")
        return array.tobytes(order="C"),array

    def _evaluate(self,solver:np.ndarray)->CachedEvaluation:
        key,array=self.exact_key(solver)
        if self._cached is not None and key==self._cached.key:
            self.cache_hits+=1
            return self._cached
        self.cache_misses+=1
        physical=solver_to_physical(array)
        perturbations=build_center_difference_batch(physical)
        started=time.perf_counter()
        spectra=np.asarray(self.backend.predict_batch(perturbations.parameters),dtype=np.float64)
        self.backend.synchronize()
        elapsed=time.perf_counter()-started
        if spectra.shape!=(13,self.observed.size):
            raise RuntimeError(f"Expected B=13 spectra with shape (13,{self.observed.size}), got {spectra.shape}")
        plus=spectra[1:7]; minus=spectra[7:13]
        physical_jac=((plus-minus)/perturbations.denominators[:,None]/self.scale).T
        solver_jac=solver_jacobian_from_physical(physical_jac,physical)
        residual=(spectra[0]-self.observed)/self.scale
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(solver_jac)):
            raise FloatingPointError("B=13 evaluation produced NaN/Inf")
        evaluation=CachedEvaluation(key,array.copy(),physical,spectra[0],residual,physical_jac,solver_jac,elapsed)
        self._cached=evaluation; self.actual_batch_evaluations+=1; self.batch_runtime_s+=elapsed
        return evaluation

    def residual(self,solver:np.ndarray)->np.ndarray:
        self.residual_calls+=1; started=time.perf_counter(); value=self._evaluate(solver).residual
        self.residual_callback_runtime_s+=time.perf_counter()-started
        return value

    def jacobian(self,solver:np.ndarray)->np.ndarray:
        self.jacobian_calls+=1; started=time.perf_counter(); value=self._evaluate(solver).solver_jacobian
        self.jacobian_callback_runtime_s+=time.perf_counter()-started
        return value

    def evaluate(self,solver:np.ndarray)->CachedEvaluation:
        return self._evaluate(solver)

    def snapshot(self)->dict:
        return {"residual_calls":self.residual_calls,"jacobian_calls":self.jacobian_calls,"actual_gpu_batch_evaluations":self.actual_batch_evaluations,"cache_hits":self.cache_hits,"cache_misses":self.cache_misses,"gpu_batch_runtime_s":self.batch_runtime_s,"residual_callback_runtime_s":self.residual_callback_runtime_s,"jacobian_callback_runtime_s":self.jacobian_callback_runtime_s,"exact_key_contract":"contiguous float64 solver vector bytes; one-entry cache; no approximate matching"}
