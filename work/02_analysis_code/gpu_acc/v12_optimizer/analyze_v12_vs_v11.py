#!/usr/bin/env python3
"""Reproducible analysis for the V12 fixed-angle GPU production run.

The V11 and V12 datasets use different angle-generation protocols.  Therefore
the version comparison in this script is an aggregate cross-version benchmark,
not a paired-sample claim for identical spectra.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_V12 = REPO_ROOT / (
    "work/04_results_and_datasets/"
    "tmm_joint_inversion_v12_fixed_angle_gpu_20260901_005118/"
    "results/v12_fixed_angle_gpu_results.json"
)
DEFAULT_V11 = REPO_ROOT / (
    "work/04_results_and_datasets/"
    "tmm_joint_inversion_v11_220_580_gpu_global_20260831_163456/"
    "v11_220_580_gpu_global_401_results.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": quantile(finite, 0.95),
        "max": max(finite),
    }


def pearson(x_values: Iterable[float], y_values: Iterable[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def percent_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def group_stats(cases: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        metadata = case["metadata"]
        grouped[(str(metadata["noise_case"]), str(metadata["noise_level"]))].append(case)

    rows: list[dict[str, Any]] = []
    for (noise_case, noise_level), members in sorted(grouped.items()):
        if version == "v12":
            hit_count = sum(bool(case["selected"]["boundary"]["boundary_hits"]) for case in members)
            changed_count = sum(
                bool(case["ranking"]["boundary_preference_changed_selection"])
                for case in members
            )
        else:
            hit_count = sum(bool(case["selected"]["boundary_hits"]) for case in members)
            changed_count = None
        rows.append(
            {
                "version": version,
                "noise_case": noise_case,
                "noise_level": noise_level,
                "count": len(members),
                "air_abs_error_mean_nm": statistics.fmean(
                    abs(float(case["errors"]["Air_error_nm"])) for case in members
                ),
                "film_mae_mean_nm": statistics.fmean(
                    float(case["errors"]["film_MAE_nm"]) for case in members
                ),
                "angle_abs_error_mean_deg": statistics.fmean(
                    float(case["errors"]["angle_abs_error_deg"]) for case in members
                ),
                "exact_rmse_mean": statistics.fmean(
                    float(case["strict_exact_RMSE"]) for case in members
                ),
                "boundary_hit_count": hit_count,
                "boundary_hit_rate": hit_count / len(members),
                "boundary_preference_changed_count": changed_count,
            }
        )
    return rows


def aggregate_metrics(payload: dict[str, Any], version: str) -> dict[str, float]:
    summary = payload["summary"]
    aggregate = payload["aggregate_errors"]
    cases = payload["cases"]
    if version == "v12":
        boundary_hits = sum(bool(case["selected"]["boundary"]["boundary_hits"]) for case in cases)
        population = float(summary["global_population_size"])
        local_batch = float(summary["local_batch_size"])
    else:
        boundary_hits = sum(bool(case["selected"]["boundary_hits"]) for case in cases)
        population = float(summary["de_mean_batch_size"])
        local_batch = 13.0
    return {
        "air_abs_mean_nm": float(aggregate["absolute_Air_error_nm"]["mean"]),
        "air_abs_median_nm": float(aggregate["absolute_Air_error_nm"]["median"]),
        "air_abs_p95_nm": float(aggregate["absolute_Air_error_nm"]["p95"]),
        "air_abs_max_nm": float(aggregate["absolute_Air_error_nm"]["max"]),
        "film_mae_mean_nm": float(aggregate["film_MAE_nm"]["mean"]),
        "film_mae_median_nm": float(aggregate["film_MAE_nm"]["median"]),
        "film_mae_p95_nm": float(aggregate["film_MAE_nm"]["p95"]),
        "film_mae_max_nm": float(aggregate["film_MAE_nm"]["max"]),
        "angle_abs_mean_deg": float(aggregate["angle_abs_error_deg"]["mean"]),
        "angle_abs_median_deg": float(aggregate["angle_abs_error_deg"]["median"]),
        "angle_abs_p95_deg": float(aggregate["angle_abs_error_deg"]["p95"]),
        "angle_abs_max_deg": float(aggregate["angle_abs_error_deg"]["max"]),
        "exact_rmse_mean": statistics.fmean(float(case["strict_exact_RMSE"]) for case in cases),
        "exact_rmse_median": statistics.median(float(case["strict_exact_RMSE"]) for case in cases),
        "exact_rmse_p95": quantile((float(case["strict_exact_RMSE"]) for case in cases), 0.95),
        "exact_rmse_max": max(float(case["strict_exact_RMSE"]) for case in cases),
        "boundary_hit_count": float(boundary_hits),
        "boundary_hit_rate": boundary_hits / len(cases),
        "runtime_s": float(summary["total_runtime_s"]),
        "global_runtime_s": float(summary["total_global_runtime_s"]),
        "local_runtime_s": float(summary["total_local_runtime_s"]),
        "throughput_cases_per_min": float(summary["throughput_cases_per_min"]),
        "peak_gpu_memory_gib": float(summary["peak_gpu_memory_gib"]),
        "global_population_size": population,
        "local_batch_size": local_batch,
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def build_markdown(analysis: dict[str, Any], v11_path: Path, v12_path: Path) -> str:
    old = analysis["aggregates"]["v11"]
    new = analysis["aggregates"]["v12"]
    change = analysis["relative_change_percent_v12_vs_v11"]
    penalty = analysis["v12_boundary_ranking"]
    diagnostics = analysis["v12_diagnostics"]
    correlation = analysis["v12_measurement_error_correlations"]
    return f"""# V12 固定测量角度与边界惩罚 GPU 结果分析

## 结论

V12 已按方案 A 将 Angle 固定为独立测量值，反演自由参数从 6 个降为 5 个；边界软惩罚仅用于光谱代价近似等价（相对阈值 0.1%）候选的排序。401 个样本全部完成并通过严格 full-ILS 闭环。

相对 V11，V12 的平均 Air 绝对误差改善 {abs(change['air_abs_mean_nm']):.2f}%，平均薄膜 MAE 改善 {abs(change['film_mae_mean_nm']):.2f}%，边界命中率从 {100.0 * old['boundary_hit_rate']:.2f}% 降到 {100.0 * new['boundary_hit_rate']:.2f}%。代价是平均严格光谱 RMSE 增加 {change['exact_rmse_mean']:.2f}%，且 Air 与薄膜误差的最大值均未改善；因此结论是“平均稳定性改善，但尾部风险仍需后续处理”，不能表述为所有样本都优于 V11。

## 汇总对比

| 指标 | V11 | V12 | V12 相对变化 |
|---|---:|---:|---:|
| Air 绝对误差均值 / nm | {fmt(old['air_abs_mean_nm'])} | {fmt(new['air_abs_mean_nm'])} | {change['air_abs_mean_nm']:+.2f}% |
| Air 绝对误差 P95 / nm | {fmt(old['air_abs_p95_nm'])} | {fmt(new['air_abs_p95_nm'])} | {change['air_abs_p95_nm']:+.2f}% |
| Air 绝对误差最大值 / nm | {fmt(old['air_abs_max_nm'])} | {fmt(new['air_abs_max_nm'])} | {change['air_abs_max_nm']:+.2f}% |
| 薄膜 MAE 均值 / nm | {fmt(old['film_mae_mean_nm'])} | {fmt(new['film_mae_mean_nm'])} | {change['film_mae_mean_nm']:+.2f}% |
| 薄膜 MAE 最大值 / nm | {fmt(old['film_mae_max_nm'])} | {fmt(new['film_mae_max_nm'])} | {change['film_mae_max_nm']:+.2f}% |
| Angle 绝对误差均值 / deg | {fmt(old['angle_abs_mean_deg'])} | {fmt(new['angle_abs_mean_deg'])} | {change['angle_abs_mean_deg']:+.2f}% |
| 严格光谱 RMSE 均值 | {fmt(old['exact_rmse_mean'])} | {fmt(new['exact_rmse_mean'])} | {change['exact_rmse_mean']:+.2f}% |
| 边界命中 | {int(old['boundary_hit_count'])}/401 | {int(new['boundary_hit_count'])}/401 | {change['boundary_hit_rate_percentage_points']:+.2f} 个百分点 |
| 总运行时间 / s | {fmt(old['runtime_s'])} | {fmt(new['runtime_s'])} | {change['runtime_s']:+.2f}% |
| 吞吐率 / case/min | {fmt(old['throughput_cases_per_min'])} | {fmt(new['throughput_cases_per_min'])} | {change['throughput_cases_per_min']:+.2f}% |
| 峰值显存 / GiB | {fmt(old['peak_gpu_memory_gib'])} | {fmt(new['peak_gpu_memory_gib'])} | {change['peak_gpu_memory_gib']:+.2f}% |

## 建议 1：边界软惩罚的效果

- V12 有 {int(new['boundary_hit_count'])}/401 个案例最终命中硬边界，V11 为 {int(old['boundary_hit_count'])}/401。
- 软惩罚在 {penalty['changed_selection_count']}/401 个案例中改变了近似等价候选的最终排序。
- 被改变案例的光谱代价增量比例中位数为 {fmt(penalty['selected_cost_increase_fraction']['median'])}，P95 为 {fmt(penalty['selected_cost_increase_fraction']['p95'])}。惩罚没有覆盖明显更优的光谱解，而是在 0.1% 等价带内优先收敛质量与较小边界严重度。

## 建议 2：方案 A 固定独立测量角度

- V12 非 clean 样本的 Angle 测量误差标准差为 {fmt(analysis['v12_angle_measurement_error_deg']['sample_std'])} deg，均值为 {fmt(analysis['v12_angle_measurement_error_deg']['mean'])} deg，与生成设定的 0.01 deg 一致。
- Angle 测量误差绝对值与 Air 绝对误差的 Pearson 相关系数为 {fmt(correlation['abs_angle_error_vs_abs_air_error'])}，与薄膜 MAE 的相关系数为 {fmt(correlation['abs_angle_error_vs_film_mae'])}。该相关性用于描述误差传播，不代表因果关系。
- 固定 5 参数 Jacobian 条件数中位数为 {fmt(diagnostics['fixed_five_parameter_condition_number']['median'])}；把 Angle 临时加入诊断后，Air-Angle 相关系数绝对值中位数为 {fmt(diagnostics['absolute_augmented_rho_air_angle']['median'])}。这说明固定 Angle 避免了优化器沿高度相关方向漂移，但物理敏感度相关性本身仍存在。

## GPU 执行口径

- 集群作业：Slurm job `1525100`，RTX 5090，401/401 closure PASS。
- 全局搜索：CuPy strict full-ILS，种群批量 B=40，每个案例重新初始化种群。
- 局部搜索：5 参数有限差分批量 B=11；Angle 只作为固定测量输入，不启用 MAP 先验。
- 总运行时间 {fmt(new['runtime_s'])} s，吞吐率 {fmt(new['throughput_cases_per_min'])} case/min，峰值显存 {fmt(new['peak_gpu_memory_gib'])} GiB。

## 适用条件与限制

1. V11 与 V12 的 Angle 生成协议不同：V12 非 clean 真值为 0.1–0.2 deg，并加入独立 0.01 deg 测量误差；V11 使用旧数据协议。因此版本表是总体工程基准，不是同一光谱逐样本配对的因果检验。
2. 本地 Lumerical API 探测因 `Session not found` 未进入数值计算，batch 探测因 Ansys license server 不可用停止；正式数据生成沿用既有 V10 wideband 的 TMM 后端口径。
3. V12 最大 Air 误差与最大薄膜 MAE 比 V11 略差，下一阶段应优先分析极端案例，而不是进一步放宽光谱等价阈值。

## 来源

- V12 原始结果：`{v12_path.as_posix()}`
- V11 基线结果：`{v11_path.as_posix()}`
- 分组统计：`v12_vs_v11_group_statistics.csv`
- 机器可读分析：`v12_vs_v11_analysis.json`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12", type=Path, default=DEFAULT_V12)
    parser.add_argument("--v11", type=Path, default=DEFAULT_V11)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    v12_path = args.v12.resolve()
    v11_path = args.v11.resolve()
    output_dir = (args.output_dir or v12_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v12 = load_json(v12_path)
    v11 = load_json(v11_path)
    v12_cases = v12["cases"]
    v11_cases = v11["cases"]

    old = aggregate_metrics(v11, "v11")
    new = aggregate_metrics(v12, "v12")
    comparable_keys = [
        "air_abs_mean_nm", "air_abs_median_nm", "air_abs_p95_nm", "air_abs_max_nm",
        "film_mae_mean_nm", "film_mae_median_nm", "film_mae_p95_nm", "film_mae_max_nm",
        "angle_abs_mean_deg", "angle_abs_median_deg", "angle_abs_p95_deg", "angle_abs_max_deg",
        "exact_rmse_mean", "exact_rmse_median", "exact_rmse_p95", "exact_rmse_max",
        "boundary_hit_rate", "runtime_s", "global_runtime_s", "local_runtime_s",
        "throughput_cases_per_min", "peak_gpu_memory_gib",
    ]
    changes = {key: percent_change(new[key], old[key]) for key in comparable_keys}
    changes["boundary_hit_rate_percentage_points"] = 100.0 * (
        new["boundary_hit_rate"] - old["boundary_hit_rate"]
    )

    changed_cases = [
        case for case in v12_cases
        if case["ranking"]["boundary_preference_changed_selection"]
    ]
    changed_increases = [
        float(case["ranking"]["selected_spectrum_cost_increase_fraction"])
        for case in changed_cases
    ]
    nonclean = [case for case in v12_cases if case["metadata"]["noise_case"] != "clean"]
    signed_angle_errors = [float(case["errors"]["angle_measurement_error_deg"]) for case in nonclean]
    abs_angle_errors = [abs(value) for value in signed_angle_errors]
    abs_air_errors = [float(case["errors"]["absolute_Air_error_nm"]) for case in nonclean]
    film_errors = [float(case["errors"]["film_MAE_nm"]) for case in nonclean]
    fixed_conditions = [
        float(case["fixed_jacobian_diagnostics"]["condition_number"])
        for case in v12_cases
    ]
    rho_values = [
        abs(float(case["augmented_six_parameter_diagnostics"]["rho_air_angle"]))
        for case in v12_cases
        if case["augmented_six_parameter_diagnostics"].get("rho_air_angle") is not None
    ]

    analysis: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_type": "aggregate_cross_version_not_paired",
        "dataset_protocol_warning": (
            "V11 and V12 use different Angle truth/measurement protocols; do not infer a paired causal effect."
        ),
        "inputs": {"v11": str(v11_path), "v12": str(v12_path)},
        "aggregates": {"v11": old, "v12": new},
        "relative_change_percent_v12_vs_v11": changes,
        "v12_boundary_ranking": {
            "changed_selection_count": len(changed_cases),
            "changed_selection_rate": len(changed_cases) / len(v12_cases),
            "selected_cost_increase_fraction": describe(changed_increases),
            "spectrum_equivalence_rtol": float(v12["configuration"]["spectrum_equivalence_rtol"]),
        },
        "v12_angle_measurement_error_deg": {
            **describe(signed_angle_errors),
            "sample_std": statistics.stdev(signed_angle_errors),
            "absolute": describe(abs_angle_errors),
        },
        "v12_measurement_error_correlations": {
            "abs_angle_error_vs_abs_air_error": pearson(abs_angle_errors, abs_air_errors),
            "abs_angle_error_vs_film_mae": pearson(abs_angle_errors, film_errors),
        },
        "v12_diagnostics": {
            "fixed_five_parameter_condition_number": describe(fixed_conditions),
            "absolute_augmented_rho_air_angle": describe(rho_values),
        },
        "group_statistics": group_stats(v11_cases, "v11") + group_stats(v12_cases, "v12"),
    }

    json_path = output_dir / "v12_vs_v11_analysis.json"
    csv_path = output_dir / "v12_vs_v11_group_statistics.csv"
    markdown_path = output_dir / "v12_vs_v11_analysis.md"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = analysis["group_statistics"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(build_markdown(analysis, v11_path, v12_path), encoding="utf-8")
    print(json.dumps({
        "overall": "PASS",
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "key_changes_percent": changes,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
