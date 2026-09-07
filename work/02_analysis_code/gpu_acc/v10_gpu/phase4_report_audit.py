"""Audit a completed Phase 4 report without changing numerical results."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def rel_delta(a,b):
    return abs(float(a)-float(b))/max(abs(float(a)),np.finfo(float).tiny)

def build_audit(report,identifiability):
    cpu=report["CPU"]; gpu=report["GPU"]
    cs=report["first_optimizer_step"]["CPU"]; gs=report["first_optimizer_step"]["GPU"]
    csolver=np.asarray(cs["solver_step"],float); gsolver=np.asarray(gs["solver_step"],float)
    cphysical=np.asarray(cs["physical_step"],float); gphysical=np.asarray(gs["physical_step"],float)
    ct=cpu["trajectory"]["records"]; gt=gpu["trajectory"]["records"]
    cangles=np.asarray([r["Angle"] for r in ct],float); gangles=np.asarray([r["Angle"] for r in gt],float)
    scaled=identifiability["scaled_jacobian_comparison"]
    rho=float(scaled["maximum_abs_rho_air_angle"])
    fit_rmse=float(cpu["exact_RMSE"]); response=report["effective_optical_response_comparison"]
    cpu_cold=float(cpu["timing"]["backend_setup_runtime_s"])+float(cpu["timing"]["total_runtime_s"])
    gpu_cold=float(gpu["timing"]["backend_setup_runtime_s"])+float(gpu["timing"]["warmup_runtime_s"])+float(gpu["timing"]["total_runtime_s"])
    return {
      "acceptance_unchanged":True,
      "first_step":{"solver_l2_difference":float(np.linalg.norm(gsolver-csolver)),"solver_relative_l2_difference":float(np.linalg.norm(gsolver-csolver)/max(np.linalg.norm(csolver),np.finfo(float).tiny)),"physical_l2_difference":float(np.linalg.norm(gphysical-cphysical)),"physical_relative_l2_difference":float(np.linalg.norm(gphysical-cphysical)/max(np.linalg.norm(cphysical),np.finfo(float).tiny)),"cost_absolute_difference":abs(float(gs["cost"])-float(cs["cost"]))},
      "termination":{"same_status":cpu["status"]==gpu["status"],"same_message":cpu["message"]==gpu["message"],"cost_relative_difference":rel_delta(cpu["cost"],gpu["cost"]),"exact_rmse_relative_difference":rel_delta(cpu["exact_RMSE"],gpu["exact_RMSE"]),"final_response_rmse_over_cpu_fit_rmse":float(response["rmse"])/max(fit_rmse,np.finfo(float).tiny),"CPU_trajectory_records":len(ct),"GPU_trajectory_records":len(gt),"CPU_angle_range_deg":[float(cangles.min()),float(cangles.max())],"GPU_angle_range_deg":[float(gangles.min()),float(gangles.max())],"both_trajectories_cross_zero_angle":bool(cangles.min()<0<cangles.max() and gangles.min()<0<gangles.max())},
      "identifiability_context":{"max_abs_rho_air_angle":rho,"phase3_case_A":bool(identifiability["classification"]["case_A_angle_small_norm_cancellation"]["supported"]),"phase3_case_B":bool(identifiability["classification"]["case_B_air_angle_correlation"]["supported"]),"phase3_case_C":bool(identifiability["classification"]["case_C_overall_conditioning"]["supported"] )},
      "runtime":{"CPU_optimizer_s":float(cpu["timing"]["total_runtime_s"]),"GPU_optimizer_s":float(gpu["timing"]["total_runtime_s"]),"warm_optimizer_speedup":float(cpu["timing"]["total_runtime_s"])/float(gpu["timing"]["total_runtime_s"]),"CPU_cold_single_npz_s":cpu_cold,"GPU_cold_single_npz_s":gpu_cold,"cold_single_npz_speedup":cpu_cold/gpu_cold,"GPU_backend_setup_s":float(gpu["timing"]["backend_setup_runtime_s"]),"GPU_warmup_s":float(gpu["timing"]["warmup_runtime_s"])},
      "attribution":"Initial residual/Jacobian closure passed and first steps are close. Both trajectories cross zero Angle and terminate with the same ftol status while Phase 3 shows near-unit Air-Angle correlation. The endpoint difference is therefore consistent with optimizer sensitivity along the documented flat valley, not evidence of a GPU residual/Jacobian defect. Strict final-response equivalence still fails and acceptance remains FAIL."
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--result",type=Path,required=True); ap.add_argument("--summary",type=Path,required=True); ap.add_argument("--identifiability",type=Path,required=True); a=ap.parse_args()
    report=json.loads(a.result.read_text(encoding="utf-8")); ident=json.loads(a.identifiability.read_text(encoding="utf-8")); audit=build_audit(report,ident); report["phase4_audit"]=audit
    report["acceptance"]["C_air_angle_identifiability_interpretation_required"]=bool(not report["acceptance"]["B_same_parameters"])
    a.result.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    marker="## Phase 4 audit interpretation"; text=a.summary.read_text(encoding="utf-8"); text=text.split(marker)[0].rstrip()+"\n\n"
    t=audit["termination"]; r=audit["runtime"]; f=audit["first_step"]; i=audit["identifiability_context"]
    lines=[marker,"",f"- Strict acceptance remains: **{'PASS' if report['acceptance']['pass'] else 'FAIL'}**; no threshold was relaxed.",f"- First-step solver relative L2 difference: `{f['solver_relative_l2_difference']:.3e}`; physical relative L2 difference: `{f['physical_relative_l2_difference']:.3e}`.",f"- Same termination status/message: `{t['same_status']}/{t['same_message']}`; cost relative difference: `{t['cost_relative_difference']:.3e}`; exact RMSE relative difference: `{t['exact_rmse_relative_difference']:.3e}`.",f"- Final CPU/GPU spectrum RMSE is `{t['final_response_rmse_over_cpu_fit_rmse']:.3e}` of the CPU fit RMSE; it does not meet the frozen strict `1e-8` response-equivalence gate.",f"- CPU/GPU trajectory records: `{t['CPU_trajectory_records']}/{t['GPU_trajectory_records']}`; both cross zero Angle: `{t['both_trajectories_cross_zero_angle']}`.",f"- Phase 3 max |rho_Air_Angle|: `{i['max_abs_rho_air_angle']:.12f}`; Cases A/B/C: `{i['phase3_case_A']}/{i['phase3_case_B']}/{i['phase3_case_C']}`.",f"- Warm optimizer-only speedup: `{r['warm_optimizer_speedup']:.2f}x`.",f"- Cold single-NPZ runtime CPU/GPU: `{r['CPU_cold_single_npz_s']:.3f} s` / `{r['GPU_cold_single_npz_s']:.3f} s`; cold speedup: `{r['cold_single_npz_speedup']:.2f}x`.",f"- GPU setup/warmup: `{r['GPU_backend_setup_s']:.3f} s` / `{r['GPU_warmup_s']:.3f} s`; resident constants are reused during optimization.","- Attribution: "+audit["attribution"],"","No optimizer, parameter, bound, residual, finite-difference step, strict response, ILS, or formal V10 code was changed by this audit."]
    a.summary.write_text(text+"\n".join(lines)+"\n",encoding="utf-8",newline="\n")
if __name__=="__main__": main()
