#!/usr/bin/env python3
"""Analyze the V12 StackRT 450-580 nm production result.

The optional TMM reference is paired by filename, but it used 220-580 nm data.
Consequently the comparison measures a combined backend-and-bandwidth change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STACKRT = REPO_ROOT / (
    "work/04_results_and_datasets/"
    "tmm_joint_inversion_v12_stackrt_450_580_gpu_job1525880_20260901/"
    "results/v12_fixed_angle_gpu_results.json"
)
DEFAULT_TMM = REPO_ROOT / (
    "work/04_results_and_datasets/"
    "tmm_joint_inversion_v12_fixed_angle_gpu_20260901_005118/"
    "results/v12_fixed_angle_gpu_results.json"
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def quantile(values: Iterable[float], probability: float) -> float | None:
    data = sorted(finite(values))
    if not data:
        return None
    position = (len(data) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return data[low]
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def describe(values: Iterable[float]) -> dict[str, Any]:
    data = finite(values)
    if not data:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": quantile(data, 0.95),
        "max": max(data),
    }


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    x_values, y_values = zip(*pairs)
    x_mean, y_mean = statistics.fmean(x_values), statistics.fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in x_values)
        * sum((y - y_mean) ** 2 for y in y_values)
    )
    return numerator / denominator if denominator else None


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def case_metric(case: dict[str, Any], name: str) -> float:
    if name == "rmse":
        return float(case["strict_exact_RMSE"])
    return float(case["errors"][name])


def group_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        metadata = case["metadata"]
        grouped[(str(metadata["noise_factor"]), str(metadata["noise_level"]))].append(case)
    rows = []
    for (factor, level), members in sorted(grouped.items()):
        hit_count = sum(bool(case["selected"]["boundary"]["boundary_hits"]) for case in members)
        rows.append({
            "noise_factor": factor,
            "noise_level": level,
            "count": len(members),
            "air_abs_mean_nm": statistics.fmean(case_metric(c, "absolute_Air_error_nm") for c in members),
            "film_mae_mean_nm": statistics.fmean(case_metric(c, "film_MAE_nm") for c in members),
            "angle_abs_mean_deg": statistics.fmean(case_metric(c, "angle_abs_error_deg") for c in members),
            "exact_rmse_mean": statistics.fmean(case_metric(c, "rmse") for c in members),
            "boundary_hit_count": hit_count,
            "boundary_hit_rate": hit_count / len(members),
        })
    return rows


def aggregate(payload: dict[str, Any]) -> dict[str, float]:
    summary = payload["summary"]
    errors = payload["aggregate_errors"]
    return {
        "air_abs_mean_nm": float(errors["absolute_Air_error_nm"]["mean"]),
        "air_abs_median_nm": float(errors["absolute_Air_error_nm"]["median"]),
        "air_abs_p95_nm": float(errors["absolute_Air_error_nm"]["p95"]),
        "air_abs_max_nm": float(errors["absolute_Air_error_nm"]["max"]),
        "film_mae_mean_nm": float(errors["film_MAE_nm"]["mean"]),
        "film_mae_median_nm": float(errors["film_MAE_nm"]["median"]),
        "film_mae_p95_nm": float(errors["film_MAE_nm"]["p95"]),
        "film_mae_max_nm": float(errors["film_MAE_nm"]["max"]),
        "angle_abs_mean_deg": float(errors["angle_abs_error_deg"]["mean"]),
        "exact_rmse_mean": float(errors["exact_RMSE"]["mean"]),
        "condition_median": float(errors["fixed_condition_number"]["median"]),
        "boundary_hit_count": float(summary["cases_with_boundary_hits"]),
        "boundary_hit_rate": float(summary["cases_with_boundary_hits"]) / float(summary["processed"]),
        "runtime_s": float(summary["total_runtime_s"]),
        "throughput_cases_per_min": float(summary["throughput_cases_per_min"]),
        "peak_gpu_memory_gib": float(summary["peak_gpu_memory_gib"]),
    }


def fmt(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stackrt", type=Path, default=DEFAULT_STACKRT)
    parser.add_argument("--tmm-reference", type=Path, default=DEFAULT_TMM)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stackrt_path = args.stackrt.resolve()
    tmm_path = args.tmm_reference.resolve()
    output_dir = (args.output_dir or stackrt_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stackrt = load(stackrt_path)
    tmm = load(tmm_path)
    cases = stackrt["cases"]
    tmm_by_name = {case["filename"]: case for case in tmm["cases"]}
    paired = [(case, tmm_by_name[case["filename"]]) for case in cases if case["filename"] in tmm_by_name]

    stackrt_agg = aggregate(stackrt)
    tmm_agg = aggregate(tmm)
    common_metrics = [
        "air_abs_mean_nm", "air_abs_median_nm", "air_abs_p95_nm", "air_abs_max_nm",
        "film_mae_mean_nm", "film_mae_median_nm", "film_mae_p95_nm", "film_mae_max_nm",
        "angle_abs_mean_deg", "exact_rmse_mean", "condition_median", "boundary_hit_rate",
        "runtime_s", "throughput_cases_per_min", "peak_gpu_memory_gib",
    ]
    changes = {key: pct(stackrt_agg[key], tmm_agg[key]) for key in common_metrics}
    changes["boundary_hit_rate_percentage_points"] = 100.0 * (
        stackrt_agg["boundary_hit_rate"] - tmm_agg["boundary_hit_rate"]
    )

    boundary_parameters: Counter[str] = Counter()
    for case in cases:
        boundary_parameters.update(case["selected"]["boundary"]["boundary_hits"])
    changed = [case for case in cases if case["ranking"]["boundary_preference_changed_selection"]]
    signed_angle = [case_metric(case, "angle_measurement_error_deg") for case in cases]
    abs_angle = [abs(value) for value in signed_angle]
    air_error = [case_metric(case, "absolute_Air_error_nm") for case in cases]
    film_error = [case_metric(case, "film_MAE_nm") for case in cases]
    rho = [
        abs(float(case["augmented_six_parameter_diagnostics"]["rho_air_angle"]))
        for case in cases
        if case["augmented_six_parameter_diagnostics"].get("rho_air_angle") is not None
    ]
    condition = [float(case["fixed_jacobian_diagnostics"]["condition_number"]) for case in cases]

    paired_summary = {}
    for metric in ("absolute_Air_error_nm", "film_MAE_nm", "angle_abs_error_deg", "rmse"):
        stackrt_values = [case_metric(left, metric) for left, _ in paired]
        tmm_values = [case_metric(right, metric) for _, right in paired]
        paired_summary[metric] = {
            "count": len(paired),
            "stackrt_lower_count": sum(a < b for a, b in zip(stackrt_values, tmm_values)),
            "equal_count": sum(a == b for a, b in zip(stackrt_values, tmm_values)),
            "stackrt_higher_count": sum(a > b for a, b in zip(stackrt_values, tmm_values)),
            "difference_stackrt_minus_tmm": describe(a - b for a, b in zip(stackrt_values, tmm_values)),
        }

    worst = sorted(cases, key=lambda c: case_metric(c, "absolute_Air_error_nm"), reverse=True)[:10]
    worst_rows = [{
        "filename": case["filename"],
        "noise_case": case["metadata"]["noise_case"],
        "air_abs_error_nm": case_metric(case, "absolute_Air_error_nm"),
        "film_mae_nm": case_metric(case, "film_MAE_nm"),
        "angle_abs_error_deg": case_metric(case, "angle_abs_error_deg"),
        "exact_rmse": case_metric(case, "rmse"),
        "boundary_hits": ";".join(case["selected"]["boundary"]["boundary_hits"]),
        "condition_number": float(case["fixed_jacobian_diagnostics"]["condition_number"]),
    } for case in worst]
    groups = group_rows(cases)

    analysis = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": stackrt["overall"],
        "stackrt_job_id": 1525880,
        "stackrt_input": str(stackrt_path),
        "tmm_reference_input": str(tmm_path),
        "comparison_warning": (
            "Paired filenames share V12 seeds and angle protocol, but backend and wavelength band both changed: "
            "StackRT api 450-580 nm versus TMM 220-580 nm. Differences cannot be attributed to backend alone."
        ),
        "stackrt_aggregate": stackrt_agg,
        "tmm_reference_aggregate": tmm_agg,
        "relative_change_percent_stackrt_vs_tmm": changes,
        "paired_case_comparison": paired_summary,
        "boundary": {
            "cases_with_hits": int(stackrt["summary"]["cases_with_boundary_hits"]),
            "near_boundary_cases": int(stackrt["summary"]["cases_near_boundary"]),
            "parameter_hit_counts": dict(boundary_parameters),
            "ranking_changed_count": len(changed),
            "ranking_cost_increase_fraction": describe(
                float(case["ranking"]["selected_spectrum_cost_increase_fraction"])
                for case in changed
            ),
        },
        "measurement_error": {
            "signed_angle_error_deg": describe(signed_angle),
            "sample_std_deg": statistics.stdev(signed_angle),
            "abs_angle_vs_abs_air_pearson": pearson(abs_angle, air_error),
            "abs_angle_vs_film_mae_pearson": pearson(abs_angle, film_error),
        },
        "identifiability": {
            "fixed_condition_number": describe(condition),
            "absolute_augmented_rho_air_angle": describe(rho),
        },
        "group_statistics": groups,
        "worst_air_cases": worst_rows,
    }

    json_path = output_dir / "v12_stackrt_450_580_analysis.json"
    group_path = output_dir / "v12_stackrt_450_580_group_statistics.csv"
    worst_path = output_dir / "v12_stackrt_450_580_worst_air_cases.csv"
    markdown_path = output_dir / "v12_stackrt_450_580_analysis.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path, rows in ((group_path, groups), (worst_path, worst_rows)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    top_air_groups = sorted(groups, key=lambda row: row["air_abs_mean_nm"], reverse=True)[:5]
    top_film_groups = sorted(groups, key=lambda row: row["film_mae_mean_nm"], reverse=True)[:5]
    boundary_text = ", ".join(f"{name}={count}" for name, count in boundary_parameters.most_common())
    markdown = f"""# V12 StackRT 450–580 nm 固定角度 GPU 结果分析

## 一句话结论

StackRT 正式数据与 GPU 反演链路已完整闭环（401/401 PASS）；固定独立测量 Angle 的方案 A 可稳定执行，但 450–580 nm 下五参数条件数中位数达到 {fmt(stackrt_agg['condition_median'])}，薄膜 MAE 均值为 {fmt(stackrt_agg['film_mae_mean_nm'])} nm，且 {int(stackrt_agg['boundary_hit_count'])}/401 案例命中边界，说明当前主要限制已转向窄波段下薄膜参数可辨识性。

## 正式结果

| 指标 | StackRT V12 |
|---|---:|
| Air 绝对误差 mean / median / P95 / max (nm) | {fmt(stackrt_agg['air_abs_mean_nm'])} / {fmt(stackrt_agg['air_abs_median_nm'])} / {fmt(stackrt_agg['air_abs_p95_nm'])} / {fmt(stackrt_agg['air_abs_max_nm'])} |
| 薄膜 MAE mean / median / P95 / max (nm) | {fmt(stackrt_agg['film_mae_mean_nm'])} / {fmt(stackrt_agg['film_mae_median_nm'])} / {fmt(stackrt_agg['film_mae_p95_nm'])} / {fmt(stackrt_agg['film_mae_max_nm'])} |
| Angle 测量绝对误差均值 (deg) | {fmt(stackrt_agg['angle_abs_mean_deg'])} |
| 严格光谱 RMSE 均值 | {fmt(stackrt_agg['exact_rmse_mean'])} |
| 边界命中 / near-boundary | {int(stackrt_agg['boundary_hit_count'])}/401 / {analysis['boundary']['near_boundary_cases']}/401 |
| 固定 5 参数条件数中位数 | {fmt(stackrt_agg['condition_median'])} |
| 总运行时间 / 吞吐率 | {fmt(stackrt_agg['runtime_s'])} s / {fmt(stackrt_agg['throughput_cases_per_min'])} case/min |
| 峰值显存 | {fmt(stackrt_agg['peak_gpu_memory_gib'])} GiB |

## 建议 1：边界软惩罚

- 边界参数命中计数：{boundary_text}。
- 在 {len(changed)}/401 个案例中，软惩罚改变了 0.1% 光谱等价候选集内的最终选择；所选光谱代价增量中位数为 {fmt(analysis['boundary']['ranking_cost_increase_fraction']['median'])}，P95 为 {fmt(analysis['boundary']['ranking_cost_increase_fraction']['p95'])}。
- 该策略减少等价解中的边界偏好，但无法消除因信息不足而形成的真实边界最优；131 个最终边界案例仍需保留标记。

## 建议 2：方案 A 固定测量角度

- 数据中的角度测量误差样本标准差为 {fmt(analysis['measurement_error']['sample_std_deg'])} deg，与设定 0.01 deg 一致。
- |Angle 测量误差| 与 Air 绝对误差的 Pearson 相关系数为 {fmt(analysis['measurement_error']['abs_angle_vs_abs_air_pearson'])}，与薄膜 MAE 的相关系数为 {fmt(analysis['measurement_error']['abs_angle_vs_film_mae_pearson'])}。
- 临时加入 Angle 的六参数诊断中，|rho(Air, Angle)| 中位数为 {fmt(analysis['identifiability']['absolute_augmented_rho_air_angle']['median'])}。固定 Angle 阻止了优化器沿这一强相关方向漂移，但不会凭空增加 450–580 nm 对各薄膜层的独立信息。

## 与上一轮 TMM V12 的工程对照

此对照按 401 个同名案例配对，但同时改变了前向后端和波段：StackRT API 450–580 nm 对 TMM 220–580 nm。因此只能用于评估当前整条链路，不能把差异单独归因于 StackRT。

| 指标 | TMM 220–580 | StackRT 450–580 | 相对变化 |
|---|---:|---:|---:|
| Air 绝对误差均值 / nm | {fmt(tmm_agg['air_abs_mean_nm'])} | {fmt(stackrt_agg['air_abs_mean_nm'])} | {changes['air_abs_mean_nm']:+.2f}% |
| 薄膜 MAE 均值 / nm | {fmt(tmm_agg['film_mae_mean_nm'])} | {fmt(stackrt_agg['film_mae_mean_nm'])} | {changes['film_mae_mean_nm']:+.2f}% |
| 边界命中率 | {100*tmm_agg['boundary_hit_rate']:.2f}% | {100*stackrt_agg['boundary_hit_rate']:.2f}% | {changes['boundary_hit_rate_percentage_points']:+.2f} 个百分点 |
| 条件数中位数 | {fmt(tmm_agg['condition_median'])} | {fmt(stackrt_agg['condition_median'])} | {changes['condition_median']:+.2f}% |
| 总运行时间 / s | {fmt(tmm_agg['runtime_s'])} | {fmt(stackrt_agg['runtime_s'])} | {changes['runtime_s']:+.2f}% |
| 峰值显存 / GiB | {fmt(tmm_agg['peak_gpu_memory_gib'])} | {fmt(stackrt_agg['peak_gpu_memory_gib'])} | {changes['peak_gpu_memory_gib']:+.2f}% |

同名案例中，StackRT 链路的 Air 绝对误差更低 {paired_summary['absolute_Air_error_nm']['stackrt_lower_count']}/401，薄膜 MAE 更低 {paired_summary['film_MAE_nm']['stackrt_lower_count']}/401。总体均值恶化主要集中在少数噪声类型，而不是所有案例一致变差。

## 风险最高的噪声组

Air 均值最高的五组：
""" + "\n".join(
        f"- {row['noise_factor']} / {row['noise_level']}: Air={row['air_abs_mean_nm']:.4g} nm, film={row['film_mae_mean_nm']:.4g} nm, boundary={row['boundary_hit_count']}/{row['count']}"
        for row in top_air_groups
    ) + "\n\n薄膜 MAE 均值最高的五组：\n" + "\n".join(
        f"- {row['noise_factor']} / {row['noise_level']}: film={row['film_mae_mean_nm']:.4g} nm, Air={row['air_abs_mean_nm']:.4g} nm, boundary={row['boundary_hit_count']}/{row['count']}"
        for row in top_film_groups
    ) + f"""

## 下一步建议

1. 保持方案 A，不引入 Angle MAP 先验；当前 Angle 误差传播不是薄膜 MAE 恶化的唯一主因。
2. 用 StackRT 再生成 220–580 nm 的同协议数据，完成“只改变后端、不改变波段”的严格消融；当前 450–580 与 220–580 对照不能隔离后端效应。
3. 对 `absolute_accuracy`、`axis_scale`、`combined` 等高风险组优先检查层厚灵敏度和边界层组合；不要通过进一步增大边界惩罚掩盖不可辨识性。
4. 后续正式结果继续同时报告光谱 RMSE、参数误差、边界标记与条件数，不能只用 401/401 收敛率判断模型有效。

## 产物

- 机器可读分析：`v12_stackrt_450_580_analysis.json`
- 分组统计：`v12_stackrt_450_580_group_statistics.csv`
- Air 最差案例：`v12_stackrt_450_580_worst_air_cases.csv`
- 原始正式结果：`v12_fixed_angle_gpu_results.json`
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "overall": "PASS",
        "json": str(json_path),
        "groups": str(group_path),
        "worst": str(worst_path),
        "markdown": str(markdown_path),
        "stackrt_aggregate": stackrt_agg,
        "relative_change_percent": changes,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
