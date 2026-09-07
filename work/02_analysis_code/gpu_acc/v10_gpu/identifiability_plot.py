"""Render the five requested Phase 3 identifiability figures."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .jacobian.contract import PARAMETER_NAMES


def _series(cases: list[dict],key) -> tuple[np.ndarray,np.ndarray]:
    x=[]; y=[]
    for case in cases:
        value=key(case)
        if value is not None and np.isfinite(value):
            x.append(float(case["abs_reflector_angle_deg"])); y.append(float(value))
    return np.asarray(x),np.asarray(y)

def _finish(path: Path,title: str,xlabel: str,ylabel: str) -> None:
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True,which="both",alpha=0.3)
    plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()

def render_identifiability_figures(cases: list[dict],sweep: list[dict],output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True,exist_ok=True)
    paths=[]

    path=output_dir/"figure1_angle_norm_vs_abs_angle.png"
    x,y=_series(cases,lambda c:c["jacobian_column_norms"]["Angle"])
    sx,sy=_series(sweep,lambda c:c["jacobian_column_norms"]["Angle"])
    plt.figure(figsize=(7.2,4.8)); plt.scatter(x,y,label="Phase 3 cases",zorder=3); plt.plot(sx,sy,".-",label="diagnostic sweep",alpha=0.8)
    plt.xscale("symlog",linthresh=1e-5); plt.yscale("symlog",linthresh=1e-10); plt.legend()
    _finish(path,"Angle local sensitivity","|reflector_angle_deg| (deg)","||J_Angle||2")
    paths.append(path.name)

    path=output_dir/"figure2_angle_relative_error_vs_abs_angle.png"
    x,y=_series(cases,lambda c:c["angle_relative_l2_error"])
    sx,sy=_series(sweep,lambda c:c["angle_relative_l2_error"])
    plt.figure(figsize=(7.2,4.8)); plt.scatter(x,y,label="Phase 3 cases",zorder=3); plt.plot(sx,sy,".-",label="diagnostic sweep",alpha=0.8)
    plt.xscale("symlog",linthresh=1e-5); plt.yscale("log"); plt.legend()
    _finish(path,"Angle CPU/GPU relative L2 error","|reflector_angle_deg| (deg)","relative L2 error")
    paths.append(path.name)

    path=output_dir/"figure3_air_angle_correlation_vs_abs_angle.png"
    x,y=_series(cases,lambda c:abs(c["rho_air_angle"]) if c["rho_air_angle"] is not None else None)
    sx,sy=_series(sweep,lambda c:abs(c["rho_air_angle"]) if c["rho_air_angle"] is not None else None)
    plt.figure(figsize=(7.2,4.8)); plt.scatter(x,y,label="Phase 3 cases",zorder=3); plt.plot(sx,sy,".-",label="diagnostic sweep",alpha=0.8)
    plt.xscale("symlog",linthresh=1e-5); plt.ylim(-0.02,1.02); plt.legend()
    _finish(path,"Air-Angle normalized column correlation","|reflector_angle_deg| (deg)","|rho_Air_Angle|")
    paths.append(path.name)

    path=output_dir/"figure4_condition_number_vs_abs_angle.png"
    x,y=_series(cases,lambda c:c["condition_number_J"])
    sx,sy=_series(sweep,lambda c:c["condition_number_J"])
    plt.figure(figsize=(7.2,4.8)); plt.scatter(x,y,label="Phase 3 cases",zorder=3); plt.plot(sx,sy,".-",label="diagnostic sweep",alpha=0.8)
    plt.xscale("symlog",linthresh=1e-5); plt.yscale("log"); plt.legend()
    _finish(path,"Raw six-parameter Jacobian conditioning","|reflector_angle_deg| (deg)","cond(J)")
    paths.append(path.name)

    path=output_dir/"figure5_column_norm_comparison.png"
    indices=np.arange(len(cases))
    plt.figure(figsize=(9.0,5.2))
    for name in PARAMETER_NAMES:
        values=[case["jacobian_column_norms"][name] for case in cases]
        plt.plot(indices,values,".-",label=name)
    plt.yscale("log"); plt.xticks(indices); plt.legend(ncol=3)
    _finish(path,"Phase 3 Jacobian column norms","Phase 3 case index","column L2 norm (log scale)")
    paths.append(path.name)
    return paths
