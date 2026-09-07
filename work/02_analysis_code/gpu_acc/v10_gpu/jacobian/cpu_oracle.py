"""Direct adapter to the existing formal V10 CPU center difference."""
from __future__ import annotations
import numpy as np
from ..backend.v10_source import load_v10_module

def formal_cpu_jacobian(model,values: np.ndarray,observed: np.ndarray,scale: float) -> np.ndarray:
    v10=load_v10_module()
    observed=np.asarray(observed,dtype=np.float64)
    scale=float(scale)
    def residual_fn(candidate: np.ndarray) -> np.ndarray:
        return (model.predict(candidate)-observed)/scale
    return np.asarray(v10.approximate_jacobian(residual_fn,np.asarray(values,dtype=np.float64)),dtype=np.float64)
