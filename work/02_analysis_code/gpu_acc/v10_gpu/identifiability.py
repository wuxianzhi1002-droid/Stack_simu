"""Pure NumPy diagnostics for Phase 3 Jacobian identifiability.

All identifiability quantities are computed from the formal CPU Jacobian.  The
GPU Jacobian is used only for numerical-consistency metrics.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .jacobian.contract import PARAMETER_NAMES
from .jacobian.metrics import ZERO_REFERENCE_NORM_LIMIT,column_metrics

ANGLE_INDEX=5
AIR_INDEX=0

def finite_or_none(value: float) -> float | None:
    value=float(value)
    return value if np.isfinite(value) else None

def normalized_column_correlation(jacobian: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    jacobian=np.asarray(jacobian,dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[1] != 6:
        raise ValueError("jacobian must have shape (samples, 6).")
    norms=np.linalg.norm(jacobian,axis=0)
    correlation=np.full((6,6),np.nan,dtype=np.float64)
    for i in range(6):
        if norms[i] <= ZERO_REFERENCE_NORM_LIMIT:
            continue
        for j in range(6):
            if norms[j] <= ZERO_REFERENCE_NORM_LIMIT:
                continue
            correlation[i,j]=np.clip(
                np.dot(jacobian[:,i],jacobian[:,j])/(norms[i]*norms[j]),-1.0,1.0
            )
    return norms,correlation

def correlation_as_json(correlation: np.ndarray) -> list[list[float | None]]:
    return [[finite_or_none(value) for value in row] for row in correlation]

def analyze_case(
    parameters: np.ndarray,cpu_jacobian: np.ndarray,gpu_jacobian: np.ndarray
) -> dict:
    parameters=np.asarray(parameters,dtype=np.float64)
    cpu=np.asarray(cpu_jacobian,dtype=np.float64)
    gpu=np.asarray(gpu_jacobian,dtype=np.float64)
    if parameters.shape != (6,):
        raise ValueError("parameters must have shape (6,).")
    if cpu.shape != gpu.shape or cpu.ndim != 2 or cpu.shape[1] != 6:
        raise ValueError("CPU/GPU Jacobians must share shape (samples, 6).")
    norms,correlation=normalized_column_correlation(cpu)
    singular_values=np.linalg.svd(cpu,compute_uv=False)
    largest=float(singular_values[0])
    smallest=float(singular_values[-1])
    tolerance=float(max(cpu.shape)*np.finfo(np.float64).eps*largest)
    rank=int(np.sum(singular_values > tolerance))
    condition_j=float(np.linalg.cond(cpu))
    gram=cpu.T@cpu
    condition_jtj=float(np.linalg.cond(gram))
    consistency=column_metrics(cpu,gpu)
    angle=consistency[ANGLE_INDEX]
    return {
        "parameters":{name:float(parameters[index]) for index,name in enumerate(PARAMETER_NAMES)},
        "parameter_vector":parameters.tolist(),
        "reflector_angle_deg":float(parameters[ANGLE_INDEX]),
        "abs_reflector_angle_deg":abs(float(parameters[ANGLE_INDEX])),
        "jacobian_column_norms":{name:float(norms[index]) for index,name in enumerate(PARAMETER_NAMES)},
        "correlation_matrix":correlation_as_json(correlation),
        "correlation_parameter_order":list(PARAMETER_NAMES),
        "rho_air_angle":finite_or_none(correlation[AIR_INDEX,ANGLE_INDEX]),
        "singular_values":singular_values.tolist(),
        "condition_number_J":finite_or_none(condition_j),
        "condition_number_J_is_infinite":not np.isfinite(condition_j),
        "condition_number_JTJ":finite_or_none(condition_jtj),
        "condition_number_JTJ_is_infinite":not np.isfinite(condition_jtj),
        "condition_number_primary":"condition_number_J",
        "numerical_rank":rank,
        "rank_tolerance":tolerance,
        "smallest_singular_value":smallest,
        "largest_singular_value":largest,
        "angle_relative_l2_error":angle["relative_l2_error"],
        "angle_max_abs_difference":angle["max_abs_difference"],
        "angle_cosine_similarity":angle["cosine_similarity"],
        "angle_zero_reference_undefined":not angle["relative_error_defined"],
        "gpu_consistency_columns":consistency,
    }

def angle_norm_power_law(cases: list[dict]) -> dict:
    pairs=[]
    for case in cases:
        angle=float(case["abs_reflector_angle_deg"])
        norm=float(case["jacobian_column_norms"]["Angle"])
        if angle > 0.0 and norm > 0.0 and np.isfinite(angle) and np.isfinite(norm):
            pairs.append((angle,norm))
    if len(pairs) < 2:
        return {"point_count":len(pairs),"slope":None,"intercept":None,"r_squared":None}
    x=np.log(np.asarray([item[0] for item in pairs],dtype=np.float64))
    y=np.log(np.asarray([item[1] for item in pairs],dtype=np.float64))
    slope,intercept=np.polyfit(x,y,1)
    predicted=slope*x+intercept
    total=float(np.sum((y-y.mean())**2))
    residual=float(np.sum((y-predicted)**2))
    r_squared=1.0-residual/total if total > 0.0 else 1.0
    return {"point_count":len(pairs),"slope":float(slope),"intercept":float(intercept),"r_squared":float(r_squared)}

def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()
