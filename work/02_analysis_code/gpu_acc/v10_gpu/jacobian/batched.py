"""One-call B=13 GPU batch center-difference Jacobian."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .contract import CenterDifferenceBatch,build_center_difference_batch

@dataclass(frozen=True)
class JacobianEvaluation:
    jacobian: np.ndarray
    base_spectrum: np.ndarray
    base_residual: np.ndarray | None
    perturbation_batch: CenterDifferenceBatch

class BatchedCenterDifferenceJacobian:
    """Wrap a frozen Phase 2 response backend without changing its physics."""
    def __init__(self,response_backend: Any):
        self.response_backend=response_backend

    def evaluate(self,values: np.ndarray,scale: float,observed: np.ndarray | None=None) -> JacobianEvaluation:
        scale=float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Residual scale must be finite and positive.")
        perturbations=build_center_difference_batch(values)
        spectra=self.response_backend.predict_batch(perturbations.parameters)
        if spectra.ndim != 2 or spectra.shape[0] != 13:
            raise RuntimeError("Response backend must return a B=13 spectrum batch.")
        plus=spectra[1:7]
        minus=spectra[7:13]
        jacobian=((plus-minus)/perturbations.denominators[:,None]/scale).T
        base=np.asarray(spectra[0],dtype=np.float64)
        residual=None
        if observed is not None:
            observed=np.asarray(observed,dtype=np.float64)
            if observed.shape != base.shape:
                raise ValueError("observed must match the reported spectrum shape.")
            residual=(base-observed)/scale
        return JacobianEvaluation(jacobian.astype(np.float64,copy=False),base,residual,perturbations)

    def jacobian(self,values: np.ndarray,scale: float) -> np.ndarray:
        return self.evaluate(values,scale).jacobian
