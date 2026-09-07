#!/usr/bin/env python3
"""Compare V12 StackRT 220-580 nm and 450-580 nm GPU production results."""

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
    "smallest_singular_value": lambda c: c["fixed_jacobian_diagnostics"]["smallest_singular_value"],
    "absolute_rho_air_angle": lambda c: abs(c["augmented_six_parameter_diagnostics"]["rho_air_angle"]),
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def describe(values: list[float]) -> dict:
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "p95": percentile(values, 0.95),
        "max": max(values),
    }


def boundary_hits(case: dict) -> str:
    return ";".join(case["selected"]["boundary"]["boundary_hits"])


def compare(wide_cases: list[dict], narrow_by_name: dict[str, dict]) -> dict:
    pairs = [(wide, narrow_by_name[wide["filename"]]) for wide in wide_cases]
    result = {"count": len(pairs), "metrics": {}}
    for name, getter in METRICS.items():
        wide_values = [float(getter(w)) for w, _ in pairs]
        narrow_values = [float(getter(n)) for _, n in pairs]
        delta = [w - n for w, n in zip(wide_values, narrow_values)]
        wide_mean, narrow_mean = statistics.fmean(wide_values), statistics.fmean(narrow_values)
        result["metrics"][name] = {
            "wide_220_580": describe(wide_values),
            "narrow_450_580": describe(narrow_values),
            "paired_delta_wide_minus_narrow": describe(delta),
            "mean_change_percent": ((wide_mean / narrow_mean) - 1.0) * 100.0 if narrow_mean else math.nan,
            "wide_lower_count": sum(w < n for w, n in zip(wide_values, narrow_values)),
            "equal_count": sum(w == n for w, n in zip(wide_values, narrow_values)),
            "wide_higher_count": sum(w > n for w, n in zip(wide_values, narrow_values)),
        }
    result["boundary"] = {
        "wide_cases_with_hits": sum(bool(boundary_hits(w)) for w, _ in pairs),
        "narrow_cases_with_hits": sum(bool(boundary_hits(n)) for _, n in pairs),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", required=True, type=Path)
    parser.add_argument("--narrow", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    wide = json.loads(args.wide.read_text(encoding="utf-8"))
    narrow = json.loads(args.narrow.read_text(encoding="utf-8"))
    wide_cases, narrow_cases = wide["cases"], narrow["cases"]
    if len(wide_cases) != 401 or len(narrow_cases) != 401:
        raise ValueError("Both reports must contain 401 cases")
    narrow_by_name = {case["filename"]: case for case in narrow_cases}
    if {case["filename"] for case in wide_cases} != set(narrow_by_name):
        raise ValueError("Filename sets differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    angle_truth_equal = sum(
        float(case["truth_angle_deg"]) == float(narrow_by_name[case["filename"]]["truth_angle_deg"])
        for case in wide_cases
    )
    angle_measurement_equal = sum(
        float(case["measured_angle_deg"]) == float(narrow_by_name[case["filename"]]["measured_angle_deg"])
        for case in wide_cases
    )
    report = {
        "comparison_contract": {
            "type": "same-filename engineering benchmark; not a strict single-variable paired ablation",
            "wide_band_nm": [220, 580], "narrow_band_nm": [450, 580],
            "filename_pair_count": 401,
            "truth_angle_equal_count": angle_truth_equal,
            "measured_angle_equal_count": angle_measurement_equal,
            "warning": "Bandwidth-dependent random-array consumption changed some angle/noise realizations; do not attribute every paired delta only to bandwidth.",
        },
        "overall": compare(wide_cases, narrow_by_name),
        "by_noise_level": {}, "by_noise_factor": {},
        "runtime": {"wide": wide["summary"], "narrow": narrow["summary"]},
    }
    for level in sorted({case["metadata"]["noise_level"] for case in wide_cases}):
        selected = [case for case in wide_cases if case["metadata"]["noise_level"] == level]
        report["by_noise_level"][level] = compare(selected, narrow_by_name)
    for factor in sorted({case["metadata"]["noise_factor"] for case in wide_cases}):
        selected = [case for case in wide_cases if case["metadata"]["noise_factor"] == factor]
        report["by_noise_factor"][factor] = compare(selected, narrow_by_name)

    rows = []
    for wide_case in wide_cases:
        narrow_case = narrow_by_name[wide_case["filename"]]
        row = {
            "index": wide_case["index"], "filename": wide_case["filename"],
            "noise_level": wide_case["metadata"]["noise_level"],
            "noise_factor": wide_case["metadata"]["noise_factor"],
            "truth_angle_equal": wide_case["truth_angle_deg"] == narrow_case["truth_angle_deg"],
            "measured_angle_equal": wide_case["measured_angle_deg"] == narrow_case["measured_angle_deg"],
            "boundary_hits_wide": boundary_hits(wide_case),
            "boundary_hits_narrow": boundary_hits(narrow_case),
        }
        for name, getter in METRICS.items():
            w, n = float(getter(wide_case)), float(getter(narrow_case))
            row[f"{name}_wide"] = w
            row[f"{name}_narrow"] = n
            row[f"{name}_delta"] = w - n
        rows.append(row)
    with (args.output_dir / "v12_220_580_vs_450_580_paired_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "v12_220_580_vs_450_580_comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V12 StackRT：220–580 nm 与 450–580 nm GPU 反演对比", "",
        "## 口径", "",
        "- 两组各 401 例，按同名文件比较。",
        "- 这不是严格单变量配对实验：改变波段后，部分随机角度和噪声实现也发生变化。",
        f"- 真角度完全相同 {angle_truth_equal}/401；测量角完全相同 {angle_measurement_equal}/401。", "",
        "## 总体结果", "",
        "| 指标 | 220–580 均值 | 450–580 均值 | 均值变化 | 宽带更低/相同/更高 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in METRICS:
        item = report["overall"]["metrics"][name]
        lines.append(
            f"| {name} | {item['wide_220_580']['mean']:.8g} | {item['narrow_450_580']['mean']:.8g} | "
            f"{item['mean_change_percent']:+.3f}% | {item['wide_lower_count']}/{item['equal_count']}/{item['wide_higher_count']} |"
        )
    boundary = report["overall"]["boundary"]
    wide_basis = wide["summary"].get("total_runtime_basis", "runner numerical wall time")
    narrow_basis = narrow["summary"].get("total_runtime_basis", "runner numerical wall time")
    lines.extend(["", "## 边界与运行", "",
        f"- 边界命中：宽带 {boundary['wide_cases_with_hits']}/401，窄带 {boundary['narrow_cases_with_hits']}/401。",
        f"- 宽带记录时间：{wide['summary']['total_runtime_s']:.3f} s（{wide_basis}）。",
        f"- 窄带记录时间：{narrow['summary']['total_runtime_s']:.3f} s（{narrow_basis}）。",
        f"- 可比 GPU 分项：宽带 global/local={wide['summary']['total_global_runtime_s']:.3f}/{wide['summary']['total_local_runtime_s']:.3f} s；窄带={narrow['summary']['total_global_runtime_s']:.3f}/{narrow['summary']['total_local_runtime_s']:.3f} s。",
        "- 分噪声等级和噪声因子统计见 JSON，逐例差值见 CSV。", "",
    ])
    md_path = args.output_dir / "v12_220_580_vs_450_580_analysis.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
