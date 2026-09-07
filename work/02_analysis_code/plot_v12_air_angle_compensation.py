"""Classified Air-angle compensation scatter plot for V12 220-580 nm results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


X_COLUMN = "angle_measurement_error_deg"
Y_COLUMN = "Air_error_nm"
MAIN_GROUPS = ("angle", "source_center_drift", "detector")
COUNTER_GROUPS = ("absolute_accuracy", "axis_offset", "axis_scale")
REFERENCE_SLOPE = 150.0
COLORS = {
    "angle": "#0072B2",
    "source_center_drift": "#009E73",
    "detector": "#D55E00",
    "absolute_accuracy": "#CC79A7",
    "axis_offset": "#E69F00",
    "axis_scale": "#56B4E9",
}
MARKERS = {
    "angle": "o",
    "source_center_drift": "s",
    "detector": "^",
    "absolute_accuracy": "o",
    "axis_offset": "s",
    "axis_scale": "^",
}


def _statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    x = frame[X_COLUMN].to_numpy(dtype=float)
    y = frame[Y_COLUMN].to_numpy(dtype=float)
    through_origin_slope = float(np.dot(x, y) / np.dot(x, x))
    regression = linregress(x, y)
    residual_reference = y - REFERENCE_SLOPE * x
    return {
        "count": int(len(frame)),
        "through_origin_slope_nm_per_deg": through_origin_slope,
        "ols_slope_nm_per_deg": float(regression.slope),
        "ols_intercept_nm": float(regression.intercept),
        "pearson_r": float(regression.rvalue),
        "ols_r_squared": float(regression.rvalue**2),
        "rmse_to_reference_150_nm": float(np.sqrt(np.mean(residual_reference**2))),
    }


def _plot_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    groups: tuple[str, ...],
    title: str,
    statistics: dict[str, dict[str, float | int]],
) -> None:
    for group in groups:
        subset = frame.loc[frame["noise_type"] == group]
        axis.scatter(
            subset[X_COLUMN],
            subset[Y_COLUMN],
            s=52,
            alpha=0.78,
            color=COLORS[group],
            marker=MARKERS[group],
            edgecolor="white",
            linewidth=0.55,
            label=group,
            zorder=3,
        )

    x_values = frame.loc[frame["noise_type"].isin(groups), X_COLUMN].to_numpy(dtype=float)
    x_padding = max(0.08 * np.ptp(x_values), 1.0e-4)
    x_limits = (float(x_values.min() - x_padding), float(x_values.max() + x_padding))
    x_line = np.linspace(*x_limits, 300)
    axis.plot(
        x_line,
        REFERENCE_SLOPE * x_line,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=r"reference: $\Delta Air=150\,\Delta\theta$",
        zorder=2,
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
    axis.axvline(0.0, color="#777777", linewidth=0.8, zorder=1)
    axis.set_xlim(x_limits)
    axis.set_title(title, fontsize=15, pad=10)
    axis.set_xlabel(r"Angle error $\Delta\theta$ (deg)", fontsize=14)
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.tick_params(labelsize=13)
    axis.legend(loc="best", fontsize=11, frameon=True)

    lines = ["group                  slope0       R²"]
    for group in groups:
        item = statistics[group]
        lines.append(
            f"{group:<21} {item['through_origin_slope_nm_per_deg']:>7.1f}  "
            f"{item['ols_r_squared']:>7.3f}"
        )
    axis.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.92},
    )


def _write_markdown(
    path: Path,
    source: Path,
    statistics: dict[str, dict[str, float | int]],
    pooled_statistics: dict[str, dict[str, float | int]],
) -> None:
    rows = []
    for group in MAIN_GROUPS + COUNTER_GROUPS:
        item = statistics[group]
        rows.append(
            f"| {group} | {item['count']} | {item['through_origin_slope_nm_per_deg']:.3f} | "
            f"{item['ols_slope_nm_per_deg']:.3f} | {item['ols_intercept_nm']:.4f} | "
            f"{item['pearson_r']:.4f} | {item['ols_r_squared']:.4f} | "
            f"{item['rmse_to_reference_150_nm']:.4f} |"
        )
    text = f"""# V12 Air-Angle compensation classified scatter summary

- Source: `{source}`
- Band: `220-580 nm`
- Included rows: all 240 rows (40 per noise type; all four noise levels pooled)
- Reference relation: `Delta Air = 150 * Delta theta`
- Pooled compensation groups: slope0 = `{pooled_statistics['main']['through_origin_slope_nm_per_deg']:.3f} nm/deg`, R2 = `{pooled_statistics['main']['ols_r_squared']:.4f}`, RMSE to reference = `{pooled_statistics['main']['rmse_to_reference_150_nm']:.4f} nm`
- Pooled counterexamples: R2 = `{pooled_statistics['counterexamples']['ols_r_squared']:.4f}`, RMSE to reference = `{pooled_statistics['counterexamples']['rmse_to_reference_150_nm']:.4f} nm`

| Noise type | N | slope through origin (nm/deg) | OLS slope | OLS intercept (nm) | Pearson r | R2 | RMSE to reference (nm) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

- `angle`, `source_center_drift`, and `detector` cluster near the reference relation. Their through-origin slopes are close to 150 nm/deg, although detector has visibly more scatter.
- `absolute_accuracy`, `axis_offset`, and `axis_scale` do not show the same one-dimensional relation: their Pearson correlation and R2 are near zero and their deviations from the reference line are much larger.
- The figure supports Air-Angle compensation for the first three perturbation families, but does not imply that every noise source produces that compensation path.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    required = {"noise_type", "noise_level", X_COLUMN, Y_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    expected_groups = set(MAIN_GROUPS + COUNTER_GROUPS)
    if set(frame["noise_type"]) != expected_groups:
        raise ValueError(f"Unexpected noise types: {sorted(set(frame['noise_type']))}")
    counts = frame.groupby("noise_type").size()
    if not (counts == 40).all():
        raise ValueError(f"Expected 40 rows per noise type, got: {counts.to_dict()}")
    if not np.isfinite(frame[[X_COLUMN, Y_COLUMN]].to_numpy(dtype=float)).all():
        raise ValueError("Angle or Air error contains NaN/Inf.")

    statistics = {
        group: _statistics(frame.loc[frame["noise_type"] == group])
        for group in MAIN_GROUPS + COUNTER_GROUPS
    }
    pooled_statistics = {
        "main": _statistics(frame.loc[frame["noise_type"].isin(MAIN_GROUPS)]),
        "counterexamples": _statistics(frame.loc[frame["noise_type"].isin(COUNTER_GROUPS)]),
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=120)
    _plot_panel(axes[0], frame, MAIN_GROUPS, "Air-Angle compensation groups", statistics)
    _plot_panel(axes[1], frame, COUNTER_GROUPS, "Counterexamples", statistics)
    axes[0].set_ylabel(r"Air error $\Delta Air$ (nm)", fontsize=14)
    axes[1].set_ylabel(r"Air error $\Delta Air$ (nm)", fontsize=14)
    figure.suptitle("V12 220-580 nm: classified Air-Angle error relation", fontsize=17, y=0.985)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.88, wspace=0.20)

    image_path = output_dir / "v12_air_angle_compensation_classified_scatter.png"
    figure.savefig(image_path, dpi=120, facecolor="white")
    plt.close(figure)

    json_path = output_dir / "v12_air_angle_compensation_regression_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "band_nm": [220, 580],
                "reference_slope_nm_per_deg": REFERENCE_SLOPE,
                "main_groups": list(MAIN_GROUPS),
                "counterexample_groups": list(COUNTER_GROUPS),
                "statistics": statistics,
                "pooled_statistics": pooled_statistics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "v12_air_angle_compensation_summary.md"
    _write_markdown(markdown_path, source, statistics, pooled_statistics)
    print(
        json.dumps(
            {
                "status": "PASS",
                "image": str(image_path),
                "summary_json": str(json_path),
                "summary_md": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
