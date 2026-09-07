"""Frozen adapter for the formal V10 local optimizer coordinates.

The formal source optimizes solver variables in [0, 1]^6 with x_scale=1.0.
The first five physical parameters use a linear bounds mapping. Angle uses the
formal square/square-root map. No optimizer or physical-model behavior is
changed here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .backend.v10_source import load_v10_module
from .jacobian.contract import PARAMETER_NAMES

LEAST_SQUARES_X_SCALE=1.0
ANGLE_INDEX=5

@dataclass(frozen=True)
class EffectiveScales:
    values: np.ndarray
    defined: bool
    reason: str | None

def bounds_and_span():
    lower,upper=load_v10_module().bounds_arrays()
    return lower,upper,upper-lower

def physical_to_solver(values: np.ndarray) -> np.ndarray:
    lower,upper,span=bounds_and_span(); values=np.asarray(values,dtype=np.float64)
    normalized=np.clip((values-lower)/span,0.0,1.0)
    solver=normalized.copy(); solver[ANGLE_INDEX]=normalized[ANGLE_INDEX]**2
    return solver

def solver_to_physical(solver: np.ndarray) -> np.ndarray:
    lower,upper,span=bounds_and_span(); solver=np.asarray(solver,dtype=np.float64)
    transformed=solver.copy(); transformed[ANGLE_INDEX]=np.sqrt(max(0.0,float(transformed[ANGLE_INDEX])))
    return lower+transformed*span

def effective_physical_scales(values: np.ndarray) -> EffectiveScales:
    """Return x_scale*d(physical)/d(solver) for the formal local coordinates."""
    lower,upper,span=bounds_and_span(); values=np.asarray(values,dtype=np.float64)
    normalized=np.clip((values-lower)/span,0.0,1.0)
    scales=span.astype(np.float64)*LEAST_SQUARES_X_SCALE
    if normalized[ANGLE_INDEX] <= 0.0:
        scales[ANGLE_INDEX]=np.nan
        return EffectiveScales(scales,False,"formal Angle sqrt map has an infinite derivative at its lower bound")
    scales[ANGLE_INDEX]=span[ANGLE_INDEX]/(2.0*normalized[ANGLE_INDEX])*LEAST_SQUARES_X_SCALE
    return EffectiveScales(scales,True,None)

def solver_jacobian_from_physical(jacobian: np.ndarray,values: np.ndarray) -> np.ndarray:
    scales=effective_physical_scales(values)
    if not scales.defined:
        raise ValueError(scales.reason)
    return np.asarray(jacobian,dtype=np.float64)*scales.values[None,:]
