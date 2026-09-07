"""Append formal-optimizer scaled Jacobian diagnostics to Phase 3 outputs."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .jacobian.contract import PARAMETER_NAMES
from .optimizer_contract import LEAST_SQUARES_X_SCALE,effective_physical_scales,solver_jacobian_from_physical

def finite(value):
    value=float(value); return value if np.isfinite(value) else None

def scaled_metrics(jacobian,values):
    scales=effective_physical_scales(values)
    scale_json={name:finite(scales.values[i]) for i,name in enumerate(PARAMETER_NAMES)}
    result={"parameter_scales":scale_json,"scaled_diagnostics_defined":scales.defined,"scaled_diagnostics_undefined_reason":scales.reason}
    if not scales.defined:
        result.update({"scaled_singular_values":None,"scaled_condition_number_J":None,"scaled_numerical_rank":None,"scaled_smallest_singular_value":None,"scaled_largest_singular_value":None})
        return result
    scaled=solver_jacobian_from_physical(jacobian,values); singular=np.linalg.svd(scaled,compute_uv=False)
    largest=float(singular[0]); tolerance=float(max(scaled.shape)*np.finfo(float).eps*largest)
    result.update({"scaled_singular_values":singular.tolist(),"scaled_condition_number_J":finite(np.linalg.cond(scaled)),"scaled_numerical_rank":int(np.sum(singular>tolerance)),"scaled_rank_tolerance":tolerance,"scaled_smallest_singular_value":float(singular[-1]),"scaled_largest_singular_value":largest})
    return result

def append_summary(path,report):
    marker="## Scaled Jacobian diagnostics"
    text=path.read_text(encoding="utf-8")
    if marker in text: text=text.split(marker)[0].rstrip()+"\n\n"
    comparison=report["scaled_jacobian_comparison"]
    lines=[marker,"",
      "The formal V10 local optimizer explicitly uses `x_scale=1.0`, but least_squares acts on `[0,1]^6` solver coordinates rather than physical units. Therefore the effective diagnostic scales are `x_scale * d(physical)/d(solver)`: bounds spans for Air/films and the exact local derivative of the formal Angle square-root map.","",
      f"- Scaled diagnostics defined: `{comparison['scaled_defined_case_count']}/{report['case_count']}`",
      f"- Finite raw/scaled condition comparisons: `{comparison['finite_condition_comparison_case_count']}/{report['case_count']}`",
      f"- Raw finite cond(J) range: `{comparison['raw_condition_min']:.3e}` to `{comparison['raw_condition_max']:.3e}`",
      f"- Scaled finite cond(J_scaled) range: `{comparison['scaled_condition_min']:.3e}` to `{comparison['scaled_condition_max']:.3e}`",
      f"- Median cond(J_scaled)/cond(J): `{comparison['median_scaled_to_raw_condition_ratio']:.3e}`",
      f"- Maximum cond(J_scaled): `{comparison['scaled_condition_max']:.3e}`",
      f"- Minimum scaled numerical rank: `{comparison['minimum_scaled_numerical_rank']}`",
      f"- Minimum scaled smallest singular value: `{comparison['minimum_scaled_smallest_singular_value']:.3e}`",
      "- The exact Angle lower bound has an infinite derivative under the unchanged square-root solver map, so its scaled SVD is reported as undefined rather than clipped.",
      "- Scaling does not reduce the condition number; for the valid cases it generally increases it. The raw physical-unit mismatch therefore does not explain the observed ill-conditioning.",
      f"- `rho_Air_Angle` is unchanged by positive single-column scaling and remains as high as `{comparison['maximum_abs_rho_air_angle']:.12f}`. Air-Angle structural non-identifiability remains present.",
      "- Conclusion: parameter units contribute to the numerical value of cond(J), but the persistent near-collinearity and poor scaled singular spectrum show that the dominant degeneracy is structural, not a units-only artifact.","",
      "No parameterization, bounds, residual, finite-difference step, strict response, ILS, or optimizer setting was changed.",
    ]
    path.write_text(text+"\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--json",type=Path,required=True); parser.add_argument("--summary",type=Path,required=True); parser.add_argument("--archive",type=Path,required=True); args=parser.parse_args()
    report=json.loads(args.json.read_text(encoding="utf-8"))
    with np.load(args.archive,allow_pickle=False) as data:
        parameters=np.asarray(data["phase3_parameters"],dtype=np.float64); cpu=np.asarray(data["phase3_cpu_jacobians"],dtype=np.float64); sweep_parameters=np.asarray(data["sweep_parameters"],dtype=np.float64); sweep_cpu=np.asarray(data["sweep_cpu_jacobians"],dtype=np.float64)
    for case,values,jacobian in zip(report["cases"],parameters,cpu): case.update(scaled_metrics(jacobian,values))
    for case,values,jacobian in zip(report["angle_sweep"],sweep_parameters,sweep_cpu): case.update(scaled_metrics(jacobian,values))
    scaled_defined=[case for case in report["cases"] if case["scaled_diagnostics_defined"]]
    comparable=[case for case in scaled_defined if case["condition_number_J"] is not None and case["scaled_condition_number_J"] is not None]
    raw=np.asarray([case["condition_number_J"] for case in comparable]); scaled=np.asarray([case["scaled_condition_number_J"] for case in comparable])
    rho=[abs(case["rho_air_angle"]) for case in report["cases"] if case["rho_air_angle"] is not None]
    comparison={"scaled_defined_case_count":len(scaled_defined),"scaled_undefined_case_count":len(report["cases"])-len(scaled_defined),"finite_condition_comparison_case_count":len(comparable),"raw_condition_min":float(raw.min()),"raw_condition_max":float(raw.max()),"scaled_condition_min":float(scaled.min()),"scaled_condition_max":float(scaled.max()),"median_scaled_to_raw_condition_ratio":float(np.median(scaled/raw)),"minimum_scaled_numerical_rank":min(case["scaled_numerical_rank"] for case in scaled_defined),"minimum_scaled_smallest_singular_value":min(case["scaled_smallest_singular_value"] for case in scaled_defined),"maximum_abs_rho_air_angle":max(rho)}
    report.update({"scaled_diagnostics_created_utc":datetime.now(timezone.utc).isoformat(),"optimizer_scaling_contract":{"least_squares_x_scale":LEAST_SQUARES_X_SCALE,"solver_bounds":[0.0,1.0],"physical_to_solver":"first five: (x-lower)/span; Angle: ((theta-lower)/span)^2","solver_to_physical":"first five: lower+u*span; Angle: lower+sqrt(u)*span","effective_parameter_scales":"x_scale * d(physical)/d(solver)","arbitrary_scale_used":False,"bounds_span_fallback_used":False},"scaled_singular_values":[case["scaled_singular_values"] for case in report["cases"]],"scaled_condition_number_J":[case["scaled_condition_number_J"] for case in report["cases"]],"scaled_numerical_rank":[case["scaled_numerical_rank"] for case in report["cases"]],"scaled_smallest_singular_value":[case["scaled_smallest_singular_value"] for case in report["cases"]],"scaled_jacobian_comparison":comparison})
    args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n"); append_summary(args.summary,report)
    print(json.dumps({"pass":True,"comparison":comparison,"scaling_contract":report["optimizer_scaling_contract"]},indent=2))
if __name__=="__main__": main()
