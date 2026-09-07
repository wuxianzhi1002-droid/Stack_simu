"""V16 Stage 3 robust two-angle experiment design with EQ-99X."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tmm_joint_inversion_v12 as v12

from .band_runner import load_v15_fit_input
from .eq99x_spectrometer import Eq99xNumpyStrictSpectrometerBackend
from .information import signal_jacobian_normalized_solver, whitened_information_metrics
from .noise_covariance import estimate_diagonal_sigma, sha256_file

BANDS = {"220-580": (220.0, 580.0), "200-800": (200.0, 800.0)}
ANGLE_GRID_DEG = np.arange(0.02, 0.2000001, 0.02, dtype=np.float64)
ANGLE_SIGMA_DEG = 0.001


def matrix_metrics(jacobian: np.ndarray, sigma: np.ndarray) -> dict:
    result = whitened_information_metrics(jacobian, sigma)
    raw = result["raw_information"]
    return {
        "sigma_min": raw["smallest_singular_value"],
        "condition_number": raw["condition_number"],
        "log10_det": raw["fisher_log10_determinant"],
        "rank": raw["numerical_rank"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--noise-calibration-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    return p.parse_args()


def main():
    args=parse_args(); root=args.input_dir.resolve(); output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    paths=sorted(root.glob("static_spectrum_*_typical_*.npz"),key=lambda p:p.name)
    if len(paths)!=100: raise RuntimeError(f"expected 100 V15 EQ-99X typical NPZ, found {len(paths)}")
    covariance=estimate_diagonal_sigma(args.noise_calibration_dir)
    cov_axis=np.asarray(covariance["wavelengths_nm"],dtype=np.float64)
    perturbations=(-ANGLE_SIGMA_DEG,0.0,ANGLE_SIGMA_DEG)
    all_rows=[]; selected_by_band={}; single_rows=[]
    for band,(start,stop) in BANDS.items():
        config=v12.FitConfig(input_dir=str(root),wavelength_min_nm=start,wavelength_max_nm=stop)
        measurement=load_v15_fit_input(paths[0],config)
        axis=np.asarray(measurement["wavelengths_um"])*1000.0
        mask=(cov_axis>=start)&(cov_axis<=stop); sigma=np.asarray(covariance["sigma"])[mask]
        if sigma.shape!=axis.shape or not np.allclose(cov_axis[mask],axis,rtol=0,atol=1e-10):
            raise RuntimeError(f"covariance axis mismatch for {band}")
        backend=Eq99xNumpyStrictSpectrometerBackend(
            measurement["wavelengths_um"],measurement["generator_config"],
            measurement["metadata"]["internal_wavelength_margin_nm"])
        truth,_=v12.load_evaluation_truth(paths[0])
        free=np.asarray([truth[n] for n in v12.FREE_PARAMS],dtype=np.float64)
        cache={}
        for theta in ANGLE_GRID_DEG:
            for delta in perturbations:
                actual=float(theta+delta)
                jac,audit=signal_jacobian_normalized_solver(backend,free,actual)
                cache[(float(theta),float(delta))]=jac
            nominal=matrix_metrics(cache[(float(theta),0.0)],sigma)
            single_rows.append({"band":band,"angle_deg":float(theta),**nominal})
            print(f"[{band}] angle={theta:.3f} deg Jacobian ready",flush=True)
        rows=[]
        for theta1,theta2 in itertools.combinations(ANGLE_GRID_DEG,2):
            scenarios=[]
            for d1,d2 in itertools.product(perturbations,repeat=2):
                joint=np.vstack((cache[(float(theta1),float(d1))],cache[(float(theta2),float(d2))]))
                scenarios.append(matrix_metrics(joint,np.tile(sigma,2)))
            nominal=matrix_metrics(np.vstack((cache[(float(theta1),0.0)],cache[(float(theta2),0.0)])),np.tile(sigma,2))
            row={
                "band":band,"theta1_deg":float(theta1),"theta2_deg":float(theta2),
                "separation_deg":float(theta2-theta1),
                "nominal_sigma_min":nominal["sigma_min"],
                "nominal_condition_number":nominal["condition_number"],
                "nominal_log10_det":nominal["log10_det"],
                "worst_sigma_min":min(x["sigma_min"] for x in scenarios),
                "worst_condition_number":max(x["condition_number"] for x in scenarios),
                "worst_log10_det":min(x["log10_det"] for x in scenarios),
                "minimum_rank":min(x["rank"] for x in scenarios),
                "robust_scenarios":len(scenarios),
            }
            rows.append(row); all_rows.append(row)
        valid=[r for r in rows if r["minimum_rank"]==5]
        selected=min(valid,key=lambda r:(-r["worst_sigma_min"],r["worst_condition_number"],-r["worst_log10_det"],r["theta1_deg"],r["theta2_deg"]))
        selected_by_band[band]=selected
    primary=selected_by_band["220-580"]
    report={
        "version":"v16_stage3_nonzero_two_angle_design","overall":"PASS",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "angle_grid_deg":ANGLE_GRID_DEG,"angle_feasibility_constraint":"0.02 <= theta1 < theta2 <= 0.20 deg; zero degrees excluded as experimentally unreliable","angle_measurement_sigma_deg":ANGLE_SIGMA_DEG,
        "angle_error_scenarios_deg":list(perturbations),
        "bands":{k:list(v) for k,v in BANDS.items()},
        "selected_by_band":selected_by_band,
        "production_pair_deg":[primary["theta1_deg"],primary["theta2_deg"]],
        "production_pair_policy":"Optimize robust worst-case sigma_min on 220-580, then reuse the identical pair at 200-800 for an isolated band comparison.",
        "source_model":"digitized EQ-99X source_shape_peak_normalized",
        "source_dataset_manifest_sha256":sha256_file(root/"simulation_manifest.json"),
        "noise_covariance_audit":covariance["audit"],
        "scope_guard":"Two distinct nonzero fixed measured reflector angles; zero degrees excluded; five shared structure parameters; no Angle fit and no Angle MAP.",
    }
    write_csv(output/"v16_stage3_angle_pairs.csv",all_rows)
    write_csv(output/"v16_stage3_single_angles.csv",single_rows)
    (output/"v16_stage3_angle_design.json").write_text(json.dumps(v12.safe(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[
        "# V16 Stage 3 EQ-99X nonzero two-angle design",
        "",
        f"- Production pair: `{primary['theta1_deg']:.3f} deg, {primary['theta2_deg']:.3f} deg`",
        f"- Feasible grid: 0.02 to 0.20 deg, step 0.02 deg; zero excluded; robust perturbation +/-{ANGLE_SIGMA_DEG:.3f} deg.",
        "- Pair selected on 220-580 nm and reused unchanged for 200-800 nm.",
        "",
        "| band | selected pair (deg) | worst sigma_min | worst kappa | worst log10 det |",
        "|---|---:|---:|---:|---:|",
    ]
    for band,row in selected_by_band.items():
        lines.append(f"| {band} | {row['theta1_deg']:.3f}, {row['theta2_deg']:.3f} | {row['worst_sigma_min']:.6g} | {row['worst_condition_number']:.6g} | {row['worst_log10_det']:.6g} |")
    (output/"v16_stage3_angle_design.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"overall":"PASS","production_pair_deg":report["production_pair_deg"],"output":str(output)},indent=2),flush=True)

if __name__=="__main__": main()
