"""Frozen six-parameter bounded center-difference perturbation contract."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..backend.v10_source import load_v10_module

PARAMETER_NAMES=("Air","HSQ","PSS","SOC","TiO2","Angle")

@dataclass(frozen=True)
class CenterDifferenceBatch:
    parameters: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    denominators: np.ndarray
    requested_steps: np.ndarray


def build_center_difference_batch(values: np.ndarray) -> CenterDifferenceBatch:
    v10=load_v10_module()
    lower,upper=v10.bounds_arrays()
    values=np.asarray(values,dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite six-parameter vector.")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("values must remain inside formal V10 bounds.")
    span=upper-lower
    steps=np.maximum(1.0e-6*span,1.0e-8)
    plus=np.repeat(values[None,:],6,axis=0)
    minus=np.repeat(values[None,:],6,axis=0)
    indices=np.arange(6)
    plus[indices,indices]=np.minimum(upper,values+steps)
    minus[indices,indices]=np.maximum(lower,values-steps)
    denominators=plus[indices,indices]-minus[indices,indices]
    if np.any(denominators <= 0.0):
        raise ValueError("Center-difference denominator must remain positive.")
    batch=np.vstack((values[None,:],plus,minus)).astype(np.float64,copy=False)
    if batch.shape != (13,6):
        raise RuntimeError("Phase 3 perturbation batch must have shape (13, 6).")
    return CenterDifferenceBatch(batch,plus,minus,denominators,steps)
