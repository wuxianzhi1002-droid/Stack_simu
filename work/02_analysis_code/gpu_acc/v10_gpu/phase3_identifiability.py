"""Run Phase 3 identifiability diagnostics without changing numerical contracts."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .backend.v10_source import source_sha256
from .identifiability import analyze_case,angle_norm_power_law,sha256_file
from .identifiability_plot import render_identifiability_figures
from .jacobian import BatchedCenterDifferenceJacobian,formal_cpu_jacobian
from .jacobian.contract import PARAMETER_NAMES
from .jacobian.metrics import aggregate_case_metrics
from .spectrometer import CupyStrictSpectrometerBackend,NumpyStrictSpectrometerBackend

ANGLE_SWEEP_DEG=np.asarray([
    0.0,-1e-5,1e-5,-3e-5,3e-5,-1e-4,1e-4,-3e-4,3e-4,
    -1e-3,1e-3,-3e-3,3e-3,-1e-2,1e-2,-3e-2,3e-2,-1e-1,1e-1,
],dtype=np.float64)
STRONG_CORRELATION_LIMIT=0.99
HIGH_CONDITION_LIMIT=1.0e8
LINEAR_SLOPE_RANGE=(0.8,1.2)
ANGLE_COSINE_APPROX_LIMIT=0.99999

def _fmt(value: float | None,format_spec: str=".3e") -> str:
    return "undefined" if value is None else format(float(value),format_spec)

def _finite(values) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(value)]

def _evaluate_cases(parameters,cpu_backend,gpu_jacobian,observed,scale):
    cpu_arrays=[]; gpu_arrays=[]; diagnostics=[]; cpu_total=0.0; gpu_total=0.0
    for index,values in enumerate(parameters):
        started=time.perf_counter(); cpu=formal_cpu_jacobian(cpu_backend.model,values,observed,scale); cpu_total+=time.perf_counter()-started
        gpu_jacobian.response_backend.synchronize(); started=time.perf_counter(); evaluation=gpu_jacobian.evaluate(values,scale); gpu_jacobian.response_backend.synchronize(); gpu_total+=time.perf_counter()-started
        gpu=evaluation.jacobian
        case=analyze_case(values,cpu,gpu); case["case_index"]=index
        case["cpu_jacobian"]={"archive_key":"phase3_cpu_jacobians","index":index,"shape":list(cpu.shape),"dtype":"float64"}
        case["gpu_jacobian"]={"archive_key":"phase3_gpu_jacobians","index":index,"shape":list(gpu.shape),"dtype":"float64"}
        diagnostics.append(case); cpu_arrays.append(cpu); gpu_arrays.append(gpu)
    return np.asarray(cpu_arrays),np.asarray(gpu_arrays),diagnostics,cpu_total,gpu_total

def _classify(cases,sweep,power_law):
    defined_angle=[case for case in cases+sweep if not case["angle_zero_reference_undefined"]]
    zero=[case for case in sweep if case["abs_reflector_angle_deg"] == 0.0][0]
    min_cos=min(_finite(case["angle_cosine_similarity"] for case in defined_angle),default=None)
    slope=power_law["slope"]
    case_a=(zero["jacobian_column_norms"]["Angle"] <= 1e-12 and slope is not None and LINEAR_SLOPE_RANGE[0] <= slope <= LINEAR_SLOPE_RANGE[1] and min_cos is not None and min_cos >= ANGLE_COSINE_APPROX_LIMIT)
    correlations=_finite(abs(case["rho_air_angle"]) if case["rho_air_angle"] is not None else None for case in cases+sweep)
    max_correlation=max(correlations,default=None)
    case_b=max_correlation is not None and max_correlation >= STRONG_CORRELATION_LIMIT
    finite_conditions=_finite(case["condition_number_J"] for case in cases+sweep)
    max_condition=max(finite_conditions,default=None)
    rank_deficient=any(case["numerical_rank"] < 6 for case in cases+sweep)
    case_c=rank_deficient or (max_condition is not None and max_condition >= HIGH_CONDITION_LIMIT)
    return {
        "thresholds":{"zero_angle_norm_max":1e-12,"linear_slope_range":list(LINEAR_SLOPE_RANGE),"minimum_defined_angle_cosine":ANGLE_COSINE_APPROX_LIMIT,"strong_abs_rho_air_angle":STRONG_CORRELATION_LIMIT,"high_condition_number_J":HIGH_CONDITION_LIMIT},
        "case_A_angle_small_norm_cancellation":{"supported":case_a,"zero_angle_norm":zero["jacobian_column_norms"]["Angle"],"angle_norm_log_log_slope":slope,"angle_norm_log_log_r_squared":power_law["r_squared"],"minimum_defined_angle_cosine":min_cos},
        "case_B_air_angle_correlation":{"supported":case_b,"maximum_abs_rho_air_angle":max_correlation},
        "case_C_overall_conditioning":{"supported":case_c,"maximum_finite_condition_number_J":max_condition,"any_numerical_rank_below_6":rank_deficient},
        "case_D_no_obvious_identifiability_issue":{"supported":not (case_a or case_b or case_c)},
    }

def _write_summary(path,report):
    cases=report["cases"]; sweep=report["angle_sweep"]; classification=report["classification"]
    consistency=report["gpu_numerical_consistency"]
    current_norms=[case["jacobian_column_norms"]["Angle"] for case in cases]
    correlations=_finite(abs(case["rho_air_angle"]) if case["rho_air_angle"] is not None else None for case in cases)
    conditions=_finite(case["condition_number_J"] for case in cases)
    smallest=[case["smallest_singular_value"] for case in cases]
    angle=next(row for row in consistency["columns"] if row["parameter"] == "Angle")
    flags=[]
    for label,key in (("Case A","case_A_angle_small_norm_cancellation"),("Case B","case_B_air_angle_correlation"),("Case C","case_C_overall_conditioning"),("Case D","case_D_no_obvious_identifiability_issue")):
        if classification[key]["supported"]: flags.append(label)
    lines=[
        "# Phase 3 Jacobian identifiability diagnostics", "",
        "- Scope: diagnostic only; Phase 3 numerical implementation remains frozen.",
        f"- Phase 3 cases: {len(cases)}",f"- Near-zero angle sweep points: {len(sweep)}",
        f"- Conclusion classification: **{', '.join(flags)}**", "",
        "## (a) GPU numerical consistency", "",
        f"- Angle maximum relative L2 error: `{_fmt(angle['max_relative_l2_error'])}`",
        f"- Angle minimum cosine similarity: `{_fmt(angle['min_cosine_similarity'],'.12f')}`",
        f"- Angle maximum absolute difference: `{_fmt(angle['max_abs_difference'])}`",
        f"- Undefined zero-reference Angle cases: `{angle['undefined_zero_reference_case_count']}`",
        "- These quantities compare the current formal CPU center difference with the unchanged B=13 GPU path.", "",
        "## (b) Angle local sensitivity", "",
        f"- Existing-case ||J_Angle|| range: `{min(current_norms):.3e}` to `{max(current_norms):.3e}`",
        f"- Sweep ||J_Angle|| at 0 deg: `{classification['case_A_angle_small_norm_cancellation']['zero_angle_norm']:.3e}`",
        f"- Sweep log-log slope of ||J_Angle|| versus |theta|: `{_fmt(report['angle_sweep_power_law']['slope'],'.6f')}` (R2 `{_fmt(report['angle_sweep_power_law']['r_squared'],'.6f')}`)",
        f"- Approximate-cosine threshold: `{ANGLE_COSINE_APPROX_LIMIT}`",
        f"- Case A supported: **{classification['case_A_angle_small_norm_cancellation']['supported']}**.", "",
        "## (c) Air-Angle column correlation", "",
        f"- Existing-case maximum |rho_Air_Angle|: `{max(correlations):.12f}`" if correlations else "- Existing-case correlation is undefined.",
        f"- Strong-correlation threshold: `{STRONG_CORRELATION_LIMIT}`",
        f"- Case B supported: **{classification['case_B_air_angle_correlation']['supported']}**.", "",
        "## (d) Overall Jacobian conditioning", "",
        f"- Existing-case cond(J) range: `{min(conditions):.3e}` to `{max(conditions):.3e}`" if conditions else "- Existing-case cond(J) is infinite for all cases.",
        f"- Existing-case smallest singular value range: `{min(smallest):.3e}` to `{max(smallest):.3e}`",
        f"- Existing-case numerical-rank range: `{min(case['numerical_rank'] for case in cases)}` to `{max(case['numerical_rank'] for case in cases)}`",
        f"- Case C supported: **{classification['case_C_overall_conditioning']['supported']}**.",
        "- cond(J) is primary. cond(J^T J) is retained only as a squared, less stable reference.",
        "- The reported conditioning uses the unchanged physical parameter units, so it is coordinate-scale dependent.", "",
        "## Angle sweep", "",
        "| angle (deg) | ||J_Angle||2 | rho_Air_Angle | cond(J) | smallest singular value | relative L2 | cosine | zero reference |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for case in sweep:
        lines.append("| {angle:.1e} | {norm:.3e} | {rho} | {cond} | {small:.3e} | {rel} | {cos} | {zero} |".format(
            angle=case["reflector_angle_deg"],norm=case["jacobian_column_norms"]["Angle"],rho=_fmt(case["rho_air_angle"],".6f"),cond=_fmt(case["condition_number_J"]),small=case["smallest_singular_value"],rel=_fmt(case["angle_relative_l2_error"]),cos=_fmt(case["angle_cosine_similarity"],".12f"),zero=case["angle_zero_reference_undefined"]))
    lines += ["", "## Figures", ""]+[f"- `{name}`" for name in report["figures"]]+[
        "", "## Frozen scope", "",
        "No parameterization, angle definition, finite-difference step, bounds, residual scaling, strict response, ILS, optimizer, scipy least_squares, JAX, autodiff, or Phase 4 changes were made.",
    ]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

def main() -> None:
    package=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,default=package/"cpu_reference_phase3"/"jacobian_reference.npz")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--machine",type=Path)
    args=parser.parse_args(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    with np.load(args.reference,allow_pickle=False) as data:
        reported=np.asarray(data["reported_wavelengths_um"],dtype=np.float64); parameters=np.asarray(data["parameters"],dtype=np.float64)
        observed=np.asarray(data["observed"],dtype=np.float64); scale=float(data["residual_scale"].item())
        config=json.loads(str(data["generator_config_json"].item())); margin=float(data["internal_margin_nm"].item())
    if parameters.shape[0] < 20: raise ValueError("At least 20 Phase 3 cases are required.")
    cpu_backend=NumpyStrictSpectrometerBackend(reported,config,margin); gpu_backend=CupyStrictSpectrometerBackend(reported,config,margin)
    gpu_jacobian=BatchedCenterDifferenceJacobian(gpu_backend); gpu_jacobian.evaluate(parameters[0],scale); gpu_backend.synchronize()
    cpu_arrays,gpu_arrays,cases,cpu_s,gpu_s=_evaluate_cases(parameters,cpu_backend,gpu_jacobian,observed,scale)
    representative=parameters[0].copy(); sweep_parameters=np.repeat(representative[None,:],len(ANGLE_SWEEP_DEG),axis=0); sweep_parameters[:,5]=ANGLE_SWEEP_DEG
    sweep_cpu,sweep_gpu,sweep,cpu_sweep_s,gpu_sweep_s=_evaluate_cases(sweep_parameters,cpu_backend,gpu_jacobian,observed,scale)
    for index,case in enumerate(sweep):
        case["sweep_index"]=index; case["diagnostic_only"]=True
        case["cpu_jacobian"]={"archive_key":"sweep_cpu_jacobians","index":index,"shape":list(sweep_cpu[index].shape),"dtype":"float64"}
        case["gpu_jacobian"]={"archive_key":"sweep_gpu_jacobians","index":index,"shape":list(sweep_gpu[index].shape),"dtype":"float64"}
    archive=output/"phase3_identifiability_jacobians.npz"
    np.savez_compressed(archive,phase3_parameters=parameters,phase3_cpu_jacobians=cpu_arrays,phase3_gpu_jacobians=gpu_arrays,sweep_parameters=sweep_parameters,sweep_cpu_jacobians=sweep_cpu,sweep_gpu_jacobians=sweep_gpu,reported_wavelengths_um=reported,residual_scale=np.asarray(scale))
    figures=render_identifiability_figures(cases,sweep,output/"phase3_identifiability_figures")
    figures=[f"phase3_identifiability_figures/{name}" for name in figures]
    power_law=angle_norm_power_law(sweep); classification=_classify(cases,sweep,power_law)
    metrics=[case["gpu_consistency_columns"] for case in cases]
    machine=json.loads(args.machine.read_text(encoding="utf-8")) if args.machine and args.machine.is_file() else None
    report={
        "phase":"Phase 3 Jacobian identifiability diagnostics","created_utc":datetime.now(timezone.utc).isoformat(),"diagnostic_only":True,
        "formal_v10_sha256":source_sha256(),"phase3_reference":str(args.reference.resolve()),"phase3_reference_sha256":sha256_file(args.reference),
        "case_count":len(cases),"angle_sweep_count":len(sweep),"parameter_order":list(PARAMETER_NAMES),"reflector_angle_parameter":"Angle",
        "parameters":[case["parameters"] for case in cases],"jacobian_column_norms":[case["jacobian_column_norms"] for case in cases],
        "correlation_matrix":[case["correlation_matrix"] for case in cases],"singular_values":[case["singular_values"] for case in cases],
        "condition_number_J":[case["condition_number_J"] for case in cases],"condition_number_JTJ":[case["condition_number_JTJ"] for case in cases],
        "numerical_rank":[case["numerical_rank"] for case in cases],"rho_air_angle":[case["rho_air_angle"] for case in cases],
        "angle_relative_l2_error":[case["angle_relative_l2_error"] for case in cases],"angle_max_abs_difference":[case["angle_max_abs_difference"] for case in cases],
        "cases":cases,"angle_sweep":sweep,"angle_sweep_values_deg":ANGLE_SWEEP_DEG.tolist(),"angle_sweep_representative_parameters":{name:float(representative[i]) for i,name in enumerate(PARAMETER_NAMES)},
        "angle_sweep_power_law":power_law,"classification":classification,
        "gpu_numerical_consistency":{"columns":aggregate_case_metrics(metrics),"nan_inf_count":int(np.isnan(gpu_arrays).sum()+np.isinf(gpu_arrays).sum())},
        "timing":{"phase3_cpu_jacobians_s":cpu_s,"phase3_gpu_jacobians_s":gpu_s,"sweep_cpu_jacobians_s":cpu_sweep_s,"sweep_gpu_jacobians_s":gpu_sweep_s},
        "jacobian_archive":{"path":str(archive),"sha256":sha256_file(archive),"keys":{"phase3_cpu_jacobians":list(cpu_arrays.shape),"phase3_gpu_jacobians":list(gpu_arrays.shape),"sweep_cpu_jacobians":list(sweep_cpu.shape),"sweep_gpu_jacobians":list(sweep_gpu.shape)}},
        "figures":figures,"machine":machine,
        "scope_guard":"No parameterization, angle definition, finite-difference step, bounds, residual scaling, strict response, ILS, optimizer, scipy least_squares, JAX, autodiff, or Phase 4 changes.",
    }
    json_path=output/"phase3_identifiability.json"; json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    _write_summary(output/"phase3_identifiability_summary.md",report)
    print(json.dumps({"pass":True,"json":str(json_path),"archive":str(archive),"summary":str(output/"phase3_identifiability_summary.md"),"classification":classification},ensure_ascii=False,indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
