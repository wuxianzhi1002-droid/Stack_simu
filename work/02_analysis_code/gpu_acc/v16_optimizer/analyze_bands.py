"""Aggregate and compare the five paired V15 Stage 1a band reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .band_runner import CANDIDATE_BANDS


ERROR_METRICS = ("absolute_Air_error_nm", "film_MAE_nm", "exact_RMSE")
LOWER_IS_BETTER = ERROR_METRICS + ("condition_number_Jw",)
HIGHER_IS_BETTER = (
    "sigma_min_Jw",
    "log10_det_JwT_Jw",
    "density_sigma_min_Jw",
    "density_log10_det_JwT_Jw",
)


def finite(values) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )


def stats(values) -> dict[str, Any]:
    array = finite(values)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("version") != "v15_stage1a_wideband_information":
        raise ValueError(f"not a V15 Stage 1a report: {path}")
    if report.get("overall") != "PASS":
        raise RuntimeError(f"input report did not pass: {path}")
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=5, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [load_report(path.resolve()) for path in args.reports]
    by_band = {report["configuration"]["band_label"]: report for report in reports}
    if set(by_band) != set(CANDIDATE_BANDS):
        raise RuntimeError(f"expected bands {list(CANDIDATE_BANDS)}, got {sorted(by_band)}")
    if len(by_band) != len(reports):
        raise RuntimeError("duplicate band report")

    reference = by_band["220-580"]
    reference_names = [row["filename"] for row in reference["cases"]]
    reference_hash = reference["source_dataset_bundle_sha256"]
    calibration_hash = reference["noise_covariance_audit"]["calibration_bundle_sha256"]
    for label, report in by_band.items():
        if [row["filename"] for row in report["cases"]] != reference_names:
            raise RuntimeError(f"paired filename/order contract failed for {label}")
        if report["source_dataset_bundle_sha256"] != reference_hash:
            raise RuntimeError(f"source dataset hash changed for {label}")
        if report["noise_covariance_audit"]["calibration_bundle_sha256"] != calibration_hash:
            raise RuntimeError(f"noise covariance calibration changed for {label}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    noise_rows = []
    for label in CANDIDATE_BANDS:
        report = by_band[label]
        rows = report["cases"]
        aggregate = report["aggregate"]
        start, stop = CANDIDATE_BANDS[label]
        summary_rows.append(
            {
                "band": label,
                "start_nm": start,
                "stop_nm": stop,
                "width_nm": stop - start,
                "wavelength_samples": rows[0]["wavelength_samples"],
                "mean_abs_Air_error_nm": aggregate["absolute_Air_error_nm"]["mean"],
                "median_abs_Air_error_nm": aggregate["absolute_Air_error_nm"]["median"],
                "mean_film_MAE_nm": aggregate["film_MAE_nm"]["mean"],
                "median_film_MAE_nm": aggregate["film_MAE_nm"]["median"],
                "boundary_hit_rate": aggregate["boundary_hit_rate"],
                "mean_exact_RMSE": aggregate["exact_RMSE"]["mean"],
                "mean_sigma_min_Jw": aggregate["sigma_min_Jw"]["mean"],
                "mean_condition_number_Jw": aggregate["condition_number_Jw"]["mean"],
                "mean_log10_det_JwT_Jw": aggregate["log10_det_JwT_Jw"]["mean"],
                "mean_density_sigma_min_Jw": aggregate["density_sigma_min_Jw"]["mean"],
                "mean_density_log10_det_JwT_Jw": aggregate[
                    "density_log10_det_JwT_Jw"
                ]["mean"],
                "terminated_rate": aggregate["terminated_rate"],
            }
        )
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["noise_case"]].append(row)
        for noise_case, case_rows in sorted(grouped.items()):
            noise_rows.append(
                {
                    "band": label,
                    "noise_case": noise_case,
                    "noise_level": "typical",
                    "realizations": len(case_rows),
                    "mean_abs_Air_error_nm": float(
                        np.mean([row["absolute_Air_error_nm"] for row in case_rows])
                    ),
                    "mean_film_MAE_nm": float(
                        np.mean([row["film_MAE_nm"] for row in case_rows])
                    ),
                    "boundary_hit_rate": float(
                        np.mean([bool(row["boundary_hits"]) for row in case_rows])
                    ),
                    "mean_sigma_min_Jw": float(
                        np.mean([row["sigma_min_Jw"] for row in case_rows])
                    ),
                    "mean_condition_number_Jw": float(
                        np.mean([row["condition_number_Jw"] for row in case_rows])
                    ),
                    "mean_log10_det_JwT_Jw": float(
                        np.mean([row["log10_det_JwT_Jw"] for row in case_rows])
                    ),
                }
            )

    paired_rows = []
    reference_by_name = {row["filename"]: row for row in reference["cases"]}
    for label in list(CANDIDATE_BANDS)[1:]:
        candidate = by_band[label]
        candidate_by_name = {row["filename"]: row for row in candidate["cases"]}
        for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
            differences = []
            wins = 0
            ties = 0
            for name in reference_names:
                baseline_value = float(reference_by_name[name][metric])
                candidate_value = float(candidate_by_name[name][metric])
                differences.append(candidate_value - baseline_value)
                tolerance = 1.0e-12 * max(1.0, abs(baseline_value), abs(candidate_value))
                if abs(candidate_value - baseline_value) <= tolerance:
                    ties += 1
                elif metric in LOWER_IS_BETTER and candidate_value < baseline_value:
                    wins += 1
                elif metric in HIGHER_IS_BETTER and candidate_value > baseline_value:
                    wins += 1
            difference = np.asarray(differences, dtype=np.float64)
            paired_rows.append(
                {
                    "candidate_band": label,
                    "baseline_band": "220-580",
                    "metric": metric,
                    "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                    "mean_candidate_minus_baseline": float(np.mean(difference)),
                    "median_candidate_minus_baseline": float(np.median(difference)),
                    "p05_candidate_minus_baseline": float(np.percentile(difference, 5.0)),
                    "p95_candidate_minus_baseline": float(np.percentile(difference, 95.0)),
                    "candidate_win_rate": wins / len(reference_names),
                    "tie_rate": ties / len(reference_names),
                    "paired_cases": len(reference_names),
                }
            )

    write_csv(output / "v15_stage1_band_summary.csv", summary_rows)
    write_csv(output / "v15_stage1_noise_case_summary.csv", noise_rows)
    write_csv(output / "v15_stage1_paired_vs_220_580.csv", paired_rows)

    best = {
        "lowest_mean_abs_Air_error_nm": min(
            summary_rows, key=lambda row: row["mean_abs_Air_error_nm"]
        )["band"],
        "lowest_mean_film_MAE_nm": min(
            summary_rows, key=lambda row: row["mean_film_MAE_nm"]
        )["band"],
        "lowest_boundary_hit_rate": min(
            summary_rows, key=lambda row: row["boundary_hit_rate"]
        )["band"],
        "highest_mean_sigma_min_Jw": max(
            summary_rows, key=lambda row: row["mean_sigma_min_Jw"]
        )["band"],
        "lowest_mean_condition_number_Jw": min(
            summary_rows, key=lambda row: row["mean_condition_number_Jw"]
        )["band"],
        "highest_mean_log10_det_JwT_Jw": max(
            summary_rows, key=lambda row: row["mean_log10_det_JwT_Jw"]
        )["band"],
        "highest_information_density_sigma_min": max(
            summary_rows, key=lambda row: row["mean_density_sigma_min_Jw"]
        )["band"],
    }
    analysis = {
        "version": "v15_stage1a_paired_band_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": "PASS",
        "candidate_bands_nm": {key: list(value) for key, value in CANDIDATE_BANDS.items()},
        "paired_cases": len(reference_names),
        "noise_scope": "typical only; each noise_case averaged over 10 realizations",
        "source_dataset_bundle_sha256": reference_hash,
        "noise_calibration_bundle_sha256": calibration_hash,
        "best_by_metric": best,
        "summary": summary_rows,
        "paired_comparisons": paired_rows,
        "interpretation_guard": (
            "Raw nested-band Fisher information normally rises when valid rows are added; "
            "use per-sample information density, inversion error, and boundary rate to decide "
            "whether the extra hardware bandwidth is efficient. No arbitrary composite score."
        ),
    }
    (output / "v15_stage1_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# V15 Stage 1a：220–580 nm 基线之外的波段拓宽",
        "",
        "## 审计口径",
        "",
        f"- 配对样本：{len(reference_names)} 个 typical 样本；每种噪声类型 10 个 realization 后取均值。",
        "- 五档波段均包含 220–580 nm，不包含任何更窄候选。",
        "- 同一份 200–800 nm 本地 StackRT 主数据集、同一角度测量与同一噪声 realization。",
        "- 当前源功率/QE/探测器为 V15 仪器链代理，不宣称为实测硬件曲线。",
        "",
        "## 总体结果",
        "",
        "| 波段 (nm) | Air abs mean (nm) | film MAE mean (nm) | boundary rate | sigma_min(Jw) | kappa(Jw) | log10 det | density sigma_min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['band']} | {row['mean_abs_Air_error_nm']:.6g} | "
            f"{row['mean_film_MAE_nm']:.6g} | {row['boundary_hit_rate']:.4f} | "
            f"{row['mean_sigma_min_Jw']:.6g} | {row['mean_condition_number_Jw']:.6g} | "
            f"{row['mean_log10_det_JwT_Jw']:.6g} | {row['mean_density_sigma_min_Jw']:.6g} |"
        )
    lines += ["", "## 单指标最优", ""]
    for key, label in best.items():
        lines.append(f"- {key}: `{label}`")
    lines += [
        "",
        "## 判读说明",
        "",
        "由于候选是嵌套波段，向 `Jw` 增加有效测量行后，原始最小奇异值和 Fisher 行列式通常会自然增加。"
        "因此不能只用原始信息总量宣布最宽波段最优；应同时检查按采样点归一化的信息密度、Air/膜层误差、"
        "边界命中率以及各噪声类型的 10-realization 均值。该报告不使用人为加权的综合分数。",
    ]
    (output / "v15_stage1_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall": "PASS", "output": str(output), "best": best}, indent=2))


if __name__ == "__main__":
    main()
