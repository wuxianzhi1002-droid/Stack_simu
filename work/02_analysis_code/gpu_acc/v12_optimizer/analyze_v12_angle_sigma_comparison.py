#!/usr/bin/env python3
"""Paired analysis for V12 fixed-angle runs with sigma(theta)=0.001 vs 0.01 deg."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = {
    "absolute_Air_error_nm": lambda c: c["errors"]["absolute_Air_error_nm"],
    "film_MAE_nm": lambda c: c["errors"]["film_MAE_nm"],
    "angle_abs_error_deg": lambda c: c["errors"]["angle_abs_error_deg"],
    "exact_RMSE": lambda c: c["strict_exact_RMSE"],
    "fixed_condition_number": lambda c: c["fixed_jacobian_diagnostics"]["condition_number"],
}

LEVEL_ORDER = ("typical", "in_spec", "spec_limit", "out_of_spec_stress")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def boundary_hits(case: dict) -> str:
    return ";".join(case["selected"]["boundary"]["boundary_hits"])


def clean_case(cases: list[dict]) -> dict:
    matches = [case for case in cases if case["metadata"]["noise_case"] == "clean"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one clean case, found {len(matches)}")
    return matches[0]


def four_class_row(case: dict) -> dict:
    selected = case["selected"]
    fixed = case["fixed_jacobian_diagnostics"]
    augmented = case["augmented_six_parameter_diagnostics"]
    return {
        "index": case["index"],
        "filename": case["filename"],
        "q_success": int(bool(selected["success"])),
        "q_status": selected["status"],
        "q_cost": selected["spectrum_cost"],
        "q_exact_RMSE": case["strict_exact_RMSE"],
        "Air_error_nm": case["errors"]["Air_error_nm"],
        "film_MAE_nm": case["errors"]["film_MAE_nm"],
        "angle_abs_error_deg": case["errors"]["angle_abs_error_deg"],
        "boundary_hits": boundary_hits(case),
        "closure_rmse": case["closure"]["rmse"],
        "condition_number": fixed["condition_number"],
        "smallest_singular_value": fixed["smallest_singular_value"],
        "rho_air_q": augmented["rho_air_angle"],
        "runtime_s": case["case_runtime_s"],
        "noise_level": case["metadata"]["noise_level"],
        "noise_factor": case["metadata"]["noise_factor"],
    }


def aggregate(cases: list[dict]) -> dict:
    return {name: stats([float(getter(case)) for case in cases]) for name, getter in METRICS.items()}


def compare_group(new_cases: list[dict], old_cases: list[dict]) -> dict:
    old = {case["filename"]: case for case in old_cases}
    paired = [(case, old[case["filename"]]) for case in new_cases]
    result = {"count": len(paired), "metrics": {}}
    for name, getter in METRICS.items():
        new_values = [float(getter(n)) for n, _ in paired]
        old_values = [float(getter(o)) for _, o in paired]
        delta = [n - o for n, o in zip(new_values, old_values)]
        new_mean = statistics.fmean(new_values)
        old_mean = statistics.fmean(old_values)
        result["metrics"][name] = {
            "sigma_0001": stats(new_values),
            "sigma_001": stats(old_values),
            "paired_delta_new_minus_old": stats(delta),
            "mean_change_percent": ((new_mean / old_mean) - 1.0) * 100.0 if old_mean else math.nan,
            "new_lower_count": sum(n < o for n, o in zip(new_values, old_values)),
            "equal_count": sum(n == o for n, o in zip(new_values, old_values)),
            "new_higher_count": sum(n > o for n, o in zip(new_values, old_values)),
        }
    result["boundary"] = {
        "sigma_0001_cases_with_hits": sum(bool(boundary_hits(n)) for n, _ in paired),
        "sigma_001_cases_with_hits": sum(bool(boundary_hits(o)) for _, o in paired),
    }
    return result


def validate_pairing(new: dict, old: dict) -> tuple[list[dict], list[dict]]:
    new_cases = new["cases"]
    old_cases = old["cases"]
    if len(new_cases) != 401 or len(old_cases) != 401:
        raise ValueError(f"Both runs must contain 401 cases: new={len(new_cases)}, old={len(old_cases)}")
    new_names = {case["filename"] for case in new_cases}
    old_names = {case["filename"] for case in old_cases}
    if new_names != old_names:
        raise ValueError("Filename sets differ between paired runs")
    for case in new_cases:
        sigma = float(case["metadata"]["angle_measurement_sigma_deg"])
        if not math.isclose(sigma, 0.001, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"New run contains non-0.001 sigma: {case['filename']} -> {sigma}")
        if not case["metadata"].get("angle_override_applied"):
            raise ValueError(f"New run lacks audited angle override: {case['filename']}")
    for case in old_cases:
        sigma = float(case["metadata"]["angle_measurement_sigma_deg"])
        if not math.isclose(sigma, 0.01, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Old run contains non-0.01 sigma: {case['filename']} -> {sigma}")
    return new_cases, old_cases


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-0001", required=True, type=Path)
    parser.add_argument("--sigma-001", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    new = json.loads(args.sigma_0001.read_text(encoding="utf-8"))
    old = json.loads(args.sigma_001.read_text(encoding="utf-8"))
    new_cases, old_cases = validate_pairing(new, old)
    old_by_name = {case["filename"]: case for case in old_cases}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = []
    for n in new_cases:
        o = old_by_name[n["filename"]]
        row = {
            "index": n["index"],
            "filename": n["filename"],
            "noise_level": n["metadata"]["noise_level"],
            "noise_factor": n["metadata"]["noise_factor"],
            "angle_error_sigma_0001_deg": n["errors"]["angle_measurement_error_deg"],
            "angle_error_sigma_001_deg": o["errors"]["angle_measurement_error_deg"],
            "boundary_hits_sigma_0001": boundary_hits(n),
            "boundary_hits_sigma_001": boundary_hits(o),
        }
        for name, getter in METRICS.items():
            nv = float(getter(n))
            ov = float(getter(o))
            row[f"{name}_sigma_0001"] = nv
            row[f"{name}_sigma_001"] = ov
            row[f"{name}_delta"] = nv - ov
        paired_rows.append(row)

    paired_fields = list(paired_rows[0])
    write_csv(args.output_dir / "v12_sigma0001_vs_sigma001_paired_cases.csv", paired_rows, paired_fields)

    four_rows = [four_class_row(case) for case in new_cases]
    four_fields = [
        "index", "filename", "q_success", "q_status", "q_cost", "q_exact_RMSE",
        "Air_error_nm", "film_MAE_nm", "angle_abs_error_deg", "boundary_hits",
        "closure_rmse", "condition_number", "smallest_singular_value", "rho_air_q",
        "runtime_s", "noise_level", "noise_factor",
    ]
    write_csv(args.output_dir / "v12_sigma0001_four_class_table.csv", four_rows, four_fields)

    workbook_source = args.output_dir / "workbook_source"
    workbook_source.mkdir(parents=True, exist_ok=True)
    new_clean = clean_case(new_cases)
    for level in LEVEL_ORDER:
        level_cases = sorted(
            (case for case in new_cases if case["metadata"]["noise_level"] == level),
            key=lambda case: case["index"],
        )
        if len(level_cases) != 100:
            raise ValueError(f"Expected 100 cases for {level}, got {len(level_cases)}")
        rows_for_sheet = [four_class_row(new_clean)] + [four_class_row(case) for case in level_cases]
        sheet_fields = four_fields[:15]
        if level == "typical":
            sheet_fields = sheet_fields[1:]
        write_csv(workbook_source / f"{level}.csv", rows_for_sheet, sheet_fields)

    comparison = {
        "contract": {
            "design": "strict_paired_single_variable_ablation",
            "fixed_angle_scheme": "A",
            "sigma_0001_result": str(args.sigma_0001),
            "sigma_001_result": str(args.sigma_001),
            "paired_filename_count": 401,
        },
        "overall": compare_group(new_cases, old_cases),
        "by_noise_level": {},
        "by_noise_factor": {},
        "gpu_runtime": {
            "sigma_0001": new["summary"],
            "sigma_001": old["summary"],
        },
    }
    for level in ("clean",) + LEVEL_ORDER:
        selected_new = [c for c in new_cases if c["metadata"]["noise_level"] == level]
        if selected_new:
            comparison["by_noise_level"][level] = compare_group(selected_new, old_cases)
    factors = sorted({c["metadata"]["noise_factor"] for c in new_cases})
    for factor in factors:
        selected_new = [c for c in new_cases if c["metadata"]["noise_factor"] == factor]
        comparison["by_noise_factor"][factor] = compare_group(selected_new, old_cases)

    comparison_path = args.output_dir / "v12_sigma0001_vs_sigma001_comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V12 固定角度方案 A：0.001° 与 0.01° 角度不确定度配对对比",
        "",
        "## 实验口径",
        "",
        "- 401 例按文件名严格配对；光谱、真值和噪声实现保持一致。",
        "- 反演固定使用独立测量角度（方案 A），唯一变化量是测量角度误差及其标准不确定度。",
        "- 正数变化率表示 0.001° 的误差指标更高，负数表示降低。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 0.001° 均值 | 0.01° 均值 | 均值变化 | 逐例更低/相同/更高 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in METRICS:
        item = comparison["overall"]["metrics"][name]
        lines.append(
            f"| {name} | {item['sigma_0001']['mean']:.8g} | {item['sigma_001']['mean']:.8g} | "
            f"{item['mean_change_percent']:+.3f}% | {item['new_lower_count']}/{item['equal_count']}/{item['new_higher_count']} |"
        )
    b = comparison["overall"]["boundary"]
    lines.extend([
        "",
        "## 边界命中与运行",
        "",
        f"- 边界命中案例：0.001° 为 {b['sigma_0001_cases_with_hits']}，0.01° 为 {b['sigma_001_cases_with_hits']}。",
        f"- 0.001° 总运行时间：{new['summary']['total_runtime_s']:.3f} s；吞吐率：{new['summary']['throughput_cases_per_min']:.3f} cases/min。",
        "",
        "详细分噪声等级、分噪声因子统计见同目录 JSON；逐例差值见 CSV。",
    ])
    (args.output_dir / "v12_sigma0001_vs_sigma001_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(comparison_path), "paired_rows": len(paired_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
