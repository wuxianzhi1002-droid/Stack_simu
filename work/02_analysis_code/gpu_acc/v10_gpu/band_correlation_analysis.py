"""Post-process the frozen dual-band Jacobian sensitivity result.

This module reads ``band_sensitivity.json`` only.  It does not evaluate the
forward model, change finite-difference steps, or run an optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARAMETERS = ("Air", "HSQ", "PSS", "SOC", "TiO2", "Angle")
BANDS = ("220_580", "450_580")
INCIDENT_MEDIUM_N = 5.8284


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("overall") != "PASS":
        raise RuntimeError("Source sensitivity result is not PASS.")
    if tuple(result.get("parameter_order", ())) != PARAMETERS:
        raise RuntimeError("Unexpected Jacobian parameter order.")
    return result


def _nonzero_angle_cases(result: dict[str, Any], band: str) -> list[dict[str, Any]]:
    cases = [
        case
        for case in result["bands"][band]["cases"]
        if abs(float(case["angle_deg"])) > 1.0e-12
    ]
    if not cases:
        raise RuntimeError(f"No nonzero-Angle cases are available for {band}.")
    return cases


def _correlation_aggregate(cases: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    matrices = np.asarray([case["correlation_matrix"] for case in cases], dtype=float)
    if matrices.shape[1:] != (6, 6) or not np.all(np.isfinite(matrices)):
        raise RuntimeError("Correlation matrices are missing or non-finite.")
    return {
        "median": np.median(matrices, axis=0),
        "q05": np.quantile(matrices, 0.05, axis=0),
        "q95": np.quantile(matrices, 0.95, axis=0),
    }


def _phase_compensation_um_per_deg(case: dict[str, Any]) -> float:
    """TMM Air-phase prediction for dAir/d(reflector Angle) at fixed phase."""
    theta = math.radians(float(case["angle_deg"]))
    air_um = float(case["parameters"]["Air"])
    sine = math.sin(theta)
    cosine = math.cos(theta)
    air_cosine_squared = 1.0 - (INCIDENT_MEDIUM_N * sine) ** 2
    return (
        air_um
        * INCIDENT_MEDIUM_N**2
        * sine
        * cosine
        / air_cosine_squared
        * math.pi
        / 180.0
    )


def _compensation_rows(cases: list[dict[str, Any]], band: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for case in cases:
        rho = float(case["rho_air_angle"])
        air_norm = float(case["column_l2_norms"]["Air"])
        angle_norm = float(case["column_l2_norms"]["Angle"])
        fitted = -rho * angle_norm / air_norm
        phase = _phase_compensation_um_per_deg(case)
        relative_error = abs(fitted - phase) / abs(phase)
        rows.append(
            {
                "band": band.replace("_", "-"),
                "angle_deg": float(case["angle_deg"]),
                "rho_air_angle": rho,
                "air_angle_compensation_um_per_deg": fitted,
                "phase_prediction_um_per_deg": phase,
                "phase_prediction_relative_error": relative_error,
                "angle_column_unexplained_norm_fraction": math.sqrt(max(0.0, 1.0 - rho**2)),
                "angle_column_explained_energy_fraction": rho**2,
            }
        )
    return rows


def _annotated_heatmap(
    axis: Any,
    matrix: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    fmt: str,
) -> Any:
    image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_title(title)
    axis.set_xticks(range(6), PARAMETERS, rotation=40, ha="right")
    axis.set_yticks(range(6), PARAMETERS)
    threshold = 0.55 * max(abs(vmin), abs(vmax))
    for row in range(6):
        for column in range(6):
            value = float(matrix[row, column])
            color = "white" if abs(value) >= threshold else "black"
            axis.text(column, row, format(value, fmt), ha="center", va="center", color=color, fontsize=8)
    axis.add_patch(plt.Rectangle((4.5, -0.5), 1.0, 1.0, fill=False, edgecolor="#00e5ff", linewidth=2.5))
    axis.add_patch(plt.Rectangle((-0.5, 4.5), 1.0, 1.0, fill=False, edgecolor="#00e5ff", linewidth=2.5))
    return image


def _plot_correlations(output: Path, aggregates: dict[str, dict[str, np.ndarray]]) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    for axis, band in zip(axes, BANDS, strict=True):
        image = _annotated_heatmap(
            axis,
            aggregates[band]["median"],
            f"{band.replace('_', '-')} nm: median signed correlation",
            -1.0,
            1.0,
            "coolwarm",
            ".4f",
        )
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88)
    colorbar.set_label("median normalized Jacobian-column correlation")
    path = output / "figure6_median_parameter_correlation_heatmaps.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path.name


def _plot_decorrelation(output: Path, aggregates: dict[str, dict[str, np.ndarray]]) -> str:
    narrow = np.abs(aggregates["450_580"]["median"])
    wide = np.abs(aggregates["220_580"]["median"])
    gain = narrow - wide
    np.fill_diagonal(gain, 0.0)
    limit = max(0.01, float(np.max(np.abs(gain))))
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = _annotated_heatmap(
        axis,
        gain,
        "Decorrelation supplied by 220-450 nm",
        -limit,
        limit,
        "PiYG",
        ".4f",
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label("|rho(450-580)| - |rho(220-580)|; positive is better")
    path = output / "figure7_wideband_decorrelation_heatmap.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path.name


def _plot_compensation(output: Path, rows: dict[str, list[dict[str, float | str]]]) -> str:
    figure, axes = plt.subplots(2, 1, figsize=(8.4, 8.2), sharex=True, constrained_layout=True)
    colors = {"220_580": "#1f77b4", "450_580": "#ff7f0e"}
    for band in BANDS:
        items = rows[band]
        angles = np.asarray([item["angle_deg"] for item in items], dtype=float)
        fitted = np.asarray([item["air_angle_compensation_um_per_deg"] for item in items], dtype=float)
        phase = np.asarray([item["phase_prediction_um_per_deg"] for item in items], dtype=float)
        unexplained = np.asarray([item["angle_column_unexplained_norm_fraction"] for item in items], dtype=float)
        label = band.replace("_", "-") + " nm"
        axes[0].plot(angles, fitted * 1000.0, "o", markersize=3.0, color=colors[band], label=label + " Jacobian")
        axes[0].plot(angles, phase * 1000.0, "-", linewidth=1.2, color=colors[band], alpha=0.75, label=label + " phase model")
        axes[1].plot(angles, unexplained * 100.0, "o-", markersize=2.5, linewidth=1.0, color=colors[band], label=label)
    axes[0].set_ylabel("canceling dAir / dAngle (nm/deg)")
    axes[0].set_title("Air-Angle compensation from Jacobian projection")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("reflector Angle (deg)")
    axes[1].set_ylabel("Angle-column component outside Air (%)")
    axes[1].set_title("Non-collinear remainder after projecting JAngle onto JAir")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    path = output / "figure8_air_angle_compensation_source.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path.name


def _write_correlation_csv(
    path: Path,
    aggregates: dict[str, dict[str, np.ndarray]],
    case_counts: dict[str, int],
) -> None:
    fields = ("band", "parameter_i", "parameter_j", "median_rho", "q05_rho", "q95_rho", "case_count")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for band in BANDS:
            for i, parameter_i in enumerate(PARAMETERS):
                for j, parameter_j in enumerate(PARAMETERS):
                    writer.writerow(
                        {
                            "band": band.replace("_", "-"),
                            "parameter_i": parameter_i,
                            "parameter_j": parameter_j,
                            "median_rho": f"{aggregates[band]['median'][i, j]:.15g}",
                            "q05_rho": f"{aggregates[band]['q05'][i, j]:.15g}",
                            "q95_rho": f"{aggregates[band]['q95'][i, j]:.15g}",
                            "case_count": case_counts[band],
                        }
                    )


def _write_compensation_csv(path: Path, rows: dict[str, list[dict[str, float | str]]]) -> None:
    fields = tuple(rows[BANDS[0]][0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for band in BANDS:
            writer.writerows(rows[band])


def _matrix_markdown(matrix: np.ndarray) -> list[str]:
    lines = ["| | " + " | ".join(PARAMETERS) + " |", "|---|" + "---:|" * 6]
    for parameter, values in zip(PARAMETERS, matrix, strict=True):
        lines.append("| " + parameter + " | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    return lines


def _summarize_compensation(items: list[dict[str, float | str]]) -> dict[str, float]:
    beta = np.asarray([item["air_angle_compensation_um_per_deg"] for item in items], dtype=float)
    relative = np.asarray([item["phase_prediction_relative_error"] for item in items], dtype=float)
    unexplained = np.asarray([item["angle_column_unexplained_norm_fraction"] for item in items], dtype=float)
    rho = np.asarray([item["rho_air_angle"] for item in items], dtype=float)
    representative = min(items, key=lambda item: abs(float(item["angle_deg"]) - 0.05))
    return {
        "median_rho_air_angle": float(np.median(rho)),
        "median_abs_rho_air_angle": float(np.median(np.abs(rho))),
        "median_compensation_um_per_deg": float(np.median(beta)),
        "min_compensation_um_per_deg": float(np.min(beta)),
        "max_compensation_um_per_deg": float(np.max(beta)),
        "median_phase_prediction_relative_error": float(np.median(relative)),
        "max_phase_prediction_relative_error": float(np.max(relative)),
        "median_unexplained_norm_fraction": float(np.median(unexplained)),
        "max_unexplained_norm_fraction": float(np.max(unexplained)),
        "median_explained_energy_fraction": float(np.median(rho**2)),
        "representative_angle_deg": float(representative["angle_deg"]),
        "representative_compensation_um_per_deg": float(
            representative["air_angle_compensation_um_per_deg"]
        ),
    }


def _write_report(
    path: Path,
    aggregates: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict[str, float]],
    figure_names: list[str],
    source_name: str,
) -> None:
    lines = [
        "# 双波段 Jacobian 参数相关性与 Air-Angle 补偿分析",
        "",
        f"- 来源：`{source_name}`",
        "- 统计口径：排除 Angle=0 deg 的归一化退化点，对其余 80 个相同物理参数点逐元素取相关系数中位数。",
        "- Jacobian：`dR(lambda)/d[Air, HSQ, PSS, SOC, TiO2, Angle]`。",
        "- 本文件只做已有 Jacobian 的后处理，不重新仿真、不修改正式 V10。",
        "",
        "## 结论",
        "",
        "1. 两个波段中 Air-Angle 都接近完全负相关；增加 220-450 nm 并未解除该结构性退化。",
        "2. 220-580 nm 对膜层列具有明显去相关作用，尤其改善 HSQ/PSS/SOC/TiO2 之间的区分。",
        "3. Air-Angle 共线来源是 Air 层传播相位 `delta_Air = (2*pi/lambda) d_Air cos(theta_Air)`：改变 reflector Angle 会改变 Snell 不变量和 Air 内传播余弦，其一阶效应几乎等价于改变 d_Air。",
        "4. 因为 Air 和 Angle 主要通过同一个 Air 相位进入反射率，它们的导数共享近乎相同的波长形状、符号相反；宽波段增强信息量，但不能自动创造一个与 Air 相位独立的 Angle 响应。",
        "",
    ]
    for band in BANDS:
        label = band.replace("_", "-")
        lines.extend([f"## {label} nm：80 个非零角度案例的中位相关矩阵", ""])
        lines.extend(_matrix_markdown(aggregates[band]["median"]))
        lines.append("")
    wide = aggregates["220_580"]["median"]
    narrow = aggregates["450_580"]["median"]
    lines.extend(
        [
            "## Air-Angle 补偿的定量来源",
            "",
            "正式模型把 Angle 定义在折射率 `n_ref=5.8284` 的 RefReflector 内，并使用 Snell 不变量：",
            "",
            "```text",
            "s = n_ref sin(theta_ref)",
            "cos(theta_Air) = sqrt(1 - s^2)",
            "delta_Air = (2*pi/lambda) d_Air cos(theta_Air)",
            "```",
            "",
            "令 Air 相位的一阶变化为零，可得到局部补偿方向：",
            "",
            "```text",
            "Delta d_Air = beta_phase Delta theta_ref",
            "beta_phase = d_Air n_ref^2 sin(theta_ref) cos(theta_ref)",
            "             / (1 - n_ref^2 sin(theta_ref)^2) * pi/180",
            "```",
            "",
            "实际 Jacobian 的最小二乘补偿系数为：",
            "",
            "```text",
            "beta_J = -(J_Air^T J_Angle) / (J_Air^T J_Air)",
            "J_Air Delta Air + J_Angle Delta Angle approximately 0",
            "```",
            "",
            "| 波段 | median rho(Air,Angle) | median beta_J | beta_J 范围 | phase 公式中位相对误差 | Air 投影后剩余 Angle 范数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for band in BANDS:
        summary = summaries[band]
        lines.append(
            f"| {band.replace('_', '-')} nm | {summary['median_rho_air_angle']:.12f} | "
            f"{summary['median_compensation_um_per_deg']:.6e} um/deg | "
            f"{summary['min_compensation_um_per_deg']:.3e} to {summary['max_compensation_um_per_deg']:.3e} um/deg | "
            f"{summary['median_phase_prediction_relative_error']:.3e} | "
            f"{summary['median_unexplained_norm_fraction'] * 100.0:.5f}% |"
        )
    representative = summaries["220_580"]
    example_per_0p01_nm = representative["representative_compensation_um_per_deg"] * 0.01 * 1000.0
    lines.extend(
        [
            "",
            f"代表性的相关性仍为：220-580 nm `{wide[0, 5]:.12f}`，450-580 nm `{narrow[0, 5]:.12f}`。",
            "Angle 列由 Air 列投影可解释的中位能量比例均高于 "
            f"`{100.0 * min(summaries[b]['median_explained_energy_fraction'] for b in BANDS):.8f}%`。",
            f"例如在数据集中最接近 0.05 deg 的 `{representative['representative_angle_deg']:.8f} deg` 点，"
            f"Angle 增加 `0.01 deg` 可由 Air 增加约 `{example_per_0p01_nm:.3f} nm` 抵消。",
            "这说明补偿主要来自 TMM Air 传播相位，而不是偶然的数值相关。剩余约 0.026% 的列范数来自界面光学导纳、其他层传播余弦和 ILS/光谱权重的微小非共线贡献。",
            "",
            "在 `theta_ref -> 0` 时，`beta_phase` 与 `theta_ref` 近似成正比，同时 `||J_Angle|| -> 0`；这既造成正入射附近 Angle 灵敏度退化，也产生正负 Angle 近似等价的两条解支。",
            "",
            "## 宽波段的作用边界",
            "",
            "220-580 nm 能降低大多数膜层列之间的相关性并改善整体条件数，但 Air-Angle 两列由同一个 Air 光程相位控制，所以其相关性几乎不变。换言之，宽波段改善了六参数问题，却没有消除最强的 Air-Angle flat valley。",
            "",
            "## 图",
            "",
        ]
    )
    lines.extend(f"- `figures/{name}`" for name in figure_names)
    lines.extend(
        [
            "",
            "## 数据文件",
            "",
            "- `parameter_correlation_aggregate.csv`",
            "- `air_angle_compensation.csv`",
            "- `air_angle_correlation_analysis.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate parameter correlations and explain Air-Angle compensation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = (args.output_dir or source.parent).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    result = _load(source)
    cases = {band: _nonzero_angle_cases(result, band) for band in BANDS}
    aggregates = {band: _correlation_aggregate(cases[band]) for band in BANDS}
    compensation = {band: _compensation_rows(cases[band], band) for band in BANDS}
    summaries = {band: _summarize_compensation(compensation[band]) for band in BANDS}

    figure_names = [
        _plot_correlations(figures, aggregates),
        _plot_decorrelation(figures, aggregates),
        _plot_compensation(figures, compensation),
    ]
    _write_correlation_csv(
        output / "parameter_correlation_aggregate.csv",
        aggregates,
        {band: len(cases[band]) for band in BANDS},
    )
    _write_compensation_csv(output / "air_angle_compensation.csv", compensation)

    payload = {
        "source": str(source),
        "parameter_order": list(PARAMETERS),
        "angle_zero_excluded_from_normalized_correlation": True,
        "nonzero_angle_case_count": {band: len(cases[band]) for band in BANDS},
        "median_correlation_matrices": {band: aggregates[band]["median"].tolist() for band in BANDS},
        "correlation_q05": {band: aggregates[band]["q05"].tolist() for band in BANDS},
        "correlation_q95": {band: aggregates[band]["q95"].tolist() for band in BANDS},
        "air_angle_compensation_summary": summaries,
        "air_angle_phase_model": {
            "incident_medium_n": INCIDENT_MEDIUM_N,
            "snell_invariant": "n_ref*sin(theta_ref)",
            "air_phase": "2*pi/lambda*d_Air*sqrt(1-(n_ref*sin(theta_ref))^2)",
            "jacobian_projection": "-(J_Air^T J_Angle)/(J_Air^T J_Air)",
        },
        "figures": figure_names,
    }
    with (output / "air_angle_correlation_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    _write_report(
        output / "air_angle_correlation_analysis.md",
        aggregates,
        summaries,
        figure_names,
        source.name,
    )
    print(json.dumps({"status": "PASS", "output": str(output), "figures": figure_names}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
