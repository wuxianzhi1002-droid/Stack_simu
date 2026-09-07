"""Numerically valid candidate ranking independent of SciPy termination flags."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from v10_gpu.backend.v10_source import load_v10_module

QUALITY = {"terminated": 0, "budget_stable": 1, "budget_descending": 2, "finite_other": 3}

def robust_cost(residual: np.ndarray, loss: str = "soft_l1") -> float:
    z=np.asarray(residual,dtype=float)**2
    if loss=="linear": rho=z
    elif loss=="soft_l1": rho=2.0*(np.sqrt(1.0+z)-1.0)
    elif loss=="huber": rho=np.where(z<=1.0,z,2.0*np.sqrt(z)-1.0)
    elif loss=="cauchy": rho=np.log1p(z)
    elif loss=="arctan": rho=np.arctan(z)
    else: raise ValueError(loss)
    return 0.5*float(np.sum(rho))

def boundary_hits(values) -> list[str]:
    v10=load_v10_module(); lower,upper=v10.bounds_arrays(); span=upper-lower
    values=np.asarray(values,dtype=float); tol=1.0e-5*span
    return [v10.PARAMS[i] for i,x in enumerate(values) if x-lower[i]<=tol[i] or upper[i]-x<=tol[i]]

def enrich_attempt(item: dict[str,Any], max_nfev:int=600, ftol:float=1e-8) -> dict[str,Any]:
    row=dict(item); physical=np.asarray(row.get("final_physical",[]),dtype=float)
    solver=np.asarray(row.get("final_solver",[]),dtype=float)
    cost=float(row.get("cost",float("nan"))); opt=float(row.get("optimality",float("nan")))
    status=int(row.get("status",-999)); nfev=int(row.get("nfev",0)); trace=np.asarray(row.get("trace_costs",[]),dtype=float)
    lower,upper=load_v10_module().bounds_arrays()
    valid=bool(physical.shape==(6,) and solver.shape==(6,) and np.all(np.isfinite(physical)) and np.all(np.isfinite(solver)) and np.isfinite(cost) and cost>=0.0 and np.isfinite(opt) and status>=0 and np.all(physical>=lower-1e-12) and np.all(physical<=upper+1e-12))
    tail=trace[-min(20,len(trace)):] if len(trace) else trace
    tail_drop=0.0
    if len(tail)>=2 and np.all(np.isfinite(tail)):
        tail_drop=max(0.0,float(tail[0]-np.min(tail)))/max(abs(float(tail[0])),np.finfo(float).eps)
    descending_threshold=max(1000.0*ftol,1.0e-8)
    if bool(row.get("success")) and status>0: quality="terminated"
    elif status==0 and nfev>=max_nfev and tail_drop<=descending_threshold and len(tail)>=2: quality="budget_stable"
    elif status==0 and nfev>=max_nfev: quality="budget_descending"
    else: quality="finite_other"
    row.update({"numerically_valid":valid,"quality_class":quality,"quality_grade":QUALITY[quality],"tail_window_count":int(len(tail)),"tail_relative_drop":tail_drop,"tail_still_descending":bool(tail_drop>descending_threshold),"boundary_hits":boundary_hits(physical) if physical.shape==(6,) else []})
    return row

def select_candidate(attempts:list[dict[str,Any]], max_nfev:int=600, ftol:float=1e-8) -> dict[str,Any]:
    enriched=[enrich_attempt(a,max_nfev,ftol) for a in attempts]
    valid=[a for a in enriched if a["numerically_valid"]]
    if not valid: return {"selected":None,"attempts":enriched,"reason":"no numerically valid candidate"}
    min_cost=min(float(a["cost"]) for a in valid); rtol=max(100.0*ftol,1.0e-6); atol=1.0e-12
    equivalent=[a for a in valid if float(a["cost"])<=min_cost+max(abs(min_cost)*rtol,atol)]
    selected=min(equivalent,key=lambda a:(int(a["quality_grade"]),len(a["boundary_hits"]),float(a["optimality"]),float(a["cost"]),int(a.get("call_index",0))))
    return {"selected":selected,"attempts":enriched,"minimum_valid_cost":min_cost,"cost_equivalence_rtol":rtol,"cost_equivalence_atol":atol,"equivalent_candidate_count":len(equivalent),"reason":"minimum strict-cost equivalence set, then termination quality, boundary count, optimality, cost"}
