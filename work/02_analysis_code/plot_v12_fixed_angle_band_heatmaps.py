"""Create paired-band V12 Air/film heatmaps from the split result workbooks.

The source workbooks are read-only.  Within each band, all realizations are
included in the mean even when ``boundary_hits`` is non-empty.  Both bands are
required to contain the same source filenames/realizations before any
aggregation is performed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


SHEETS = ("typical", "in_spec", "spec_limit", "out_of_spec_stress")
FACTORS = (
    "clean",
    "absolute_accuracy",
    "angle",
    "axis_offset",
    "axis_scale",
    "combined",
    "detector",
    "material",
    "source_center_drift",
    "source_power",
    "thermal_drift",
)
DISPLAY_FACTORS = (
    "clean",
    "abs_accuracy",
    "angle",
    "axis_offset",
    "axis_scale",
    "combined",
    "detector",
    "material",
    "source_center",
    "source_power",
    "thermal_drift",
)
BANDS = ("220-580 nm", "450-580 nm")
BAND_TICK_LABELS = BANDS
FIGURE_SIZE_PX = (1000, 800)
SHEET_TITLES = {
    "typical": "typical",
    "in_spec": "in-spec",
    "spec_limit": "spec-limit",
    "out_of_spec_stress": "out-of-spec stress",
}
REQUIRED_COLUMNS = {
    "noise_case",
    "success",
    "Air_error_nm",
    "film_MAE_nm",
    "boundary_hits",
}


def _realization(filename: str) -> int:
    match = re.search(r"_r(\d{4})_", str(filename))
    if not match:
        if "_clean_" in str(filename):
            return 0
        raise ValueError(f"Cannot parse realization from {filename!r}")
    return int(match.group(1))


def _factor(noise_case: str, sheet: str) -> str:
    if noise_case == "clean":
        return "clean"
    suffix = "_" + sheet
    if not noise_case.endswith(suffix):
        raise ValueError(f"Noise case {noise_case!r} does not match sheet {sheet!r}.")
    return noise_case[: -len(suffix)]


def _load_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}/{sheet} is missing columns: {missing}")
    frame = frame.copy()
    frame["factor"] = [_factor(value, sheet) for value in frame["noise_case"].astype(str)]
    if "filename" in frame.columns:
        frame["realization"] = [_realization(value) for value in frame["filename"].astype(str)]
    else:
        frame["realization"] = frame.groupby("factor", sort=False).cumcount()
    frame["source_key"] = frame["factor"] + ":r" + frame["realization"].astype(str).str.zfill(4)
    if set(frame["factor"]) != set(FACTORS):
        raise ValueError(f"Unexpected noise factors in {path.name}/{sheet}: {sorted(set(frame['factor']))}")
    for factor in FACTORS:
        subset = frame.loc[frame["factor"] == factor]
        expected = 1 if factor == "clean" else 10
        if len(subset) != expected or subset["source_key"].nunique() != expected:
            raise ValueError(f"Expected {expected} unique rows for {factor} in {path.name}/{sheet}.")
        if factor != "clean" and set(subset["realization"]) != set(range(10)):
            raise ValueError(f"Realizations for {factor} are not r0000-r0009 in {path.name}/{sheet}.")
    return frame


def _aggregate(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for factor in FACTORS:
        subset = frame.loc[frame["factor"] == factor]
        boundary_hits = subset["boundary_hits"]
        boundary_hit_count = int(
            (~(boundary_hits.isna() | boundary_hits.astype(str).str.strip().eq(""))).sum()
        )
        output[factor] = {
            "included_realization_count": int(len(subset)),
            "source_realization_count": int((frame["factor"] == factor).sum()),
            "boundary_hit_count_included": boundary_hit_count,
            "mean_abs_Air_error_nm": (
                None if subset.empty else float(subset["Air_error_nm"].abs().mean())
            ),
            "mean_abs_film_MAE_nm": (
                None if subset.empty else float(subset["film_MAE_nm"].abs().mean())
            ),
        }
    return output


def _format_value(value: float) -> str:
    absolute = abs(value)
    if absolute == 0.0:
        return "0"
    if absolute < 1.0e-3:
        return f"{value:.2g}"
    if absolute < 10.0:
        return f"{value:.3g}"
    return f"{value:.3g}"


def _matrix(
    aggregates: dict[str, dict[str, dict[str, float | int | None]]],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                np.nan if aggregates[band][factor][metric] is None else aggregates[band][factor][metric]
                for band in BANDS
            ]
            for factor in FACTORS
        ],
        dtype=float,
    )


def _draw_panel(
    axis: Any,
    values: np.ndarray,
    title: str,
    cmap: str,
    norm: Any,
    show_y: bool,
) -> Any:
    masked = np.ma.masked_invalid(values)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#f0f0f0")
    image = axis.imshow(masked, cmap=colormap, norm=norm, aspect="auto")
    axis.set_title(title, fontsize=14, pad=8)
    axis.set_xticks(range(2), BAND_TICK_LABELS, rotation=0, fontsize=14)
    axis.set_xlabel("Band", fontsize=14, labelpad=8)
    axis.set_yticks(range(len(FACTORS)))
    if show_y:
        axis.set_yticklabels(DISPLAY_FACTORS, fontsize=14)
        axis.set_ylabel("Noise source", fontsize=14, labelpad=8)
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(FACTORS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", labelsize=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                label = "–"
                color = "#666666"
            else:
                label = _format_value(float(value))
                rgba = image.cmap(image.norm(float(value)))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < 0.47 else "black"
            axis.text(column, row, label, ha="center", va="center", fontsize=14, color=color)
    return image


def _plot(
    output_path: Path,
    sheet: str,
    aggregates: dict[str, dict[str, dict[str, float | int | None]]],
) -> None:
    air = _matrix(aggregates, "mean_abs_Air_error_nm")
    film = _matrix(aggregates, "mean_abs_film_MAE_nm")
    air_finite = air[np.isfinite(air)]
    air_max = max(float(np.max(air_finite)), 1.0e-12)
    film_finite = film[np.isfinite(film)]
    film_max = max(float(np.max(film_finite)), 1.0e-12)

    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_SIZE_PX[0] / 100, FIGURE_SIZE_PX[1] / 100),
        dpi=100,
    )
    air_image = _draw_panel(
        axes[0],
        air,
        "Mean |Air error| (nm)",
        "YlGnBu",
        Normalize(vmin=0.0, vmax=air_max),
        True,
    )
    film_image = _draw_panel(
        axes[1],
        film,
        "Mean film MAE (nm)",
        "YlOrRd",
        Normalize(vmin=0.0, vmax=film_max),
        False,
    )
    figure.suptitle(
        f"{SHEET_TITLES[sheet]}: realization mean\n(all boundary-hit cases included)",
        fontsize=14,
        y=0.985,
    )
    air_bar = figure.colorbar(air_image, ax=axes[0], fraction=0.055, pad=0.025)
    air_bar.ax.tick_params(labelsize=14)
    film_bar = figure.colorbar(film_image, ax=axes[1], fraction=0.055, pad=0.025)
    film_bar.ax.tick_params(labelsize=14)
    figure.subplots_adjust(left=0.18, right=0.91, bottom=0.12, top=0.84, wspace=0.36)
    figure.savefig(output_path, dpi=100, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot V12 paired-band Air/film heatmaps.")
    parser.add_argument("--band-220", type=Path, required=True)
    parser.add_argument("--band-450", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    band_paths = {
        BANDS[0]: args.band_220.resolve(),
        BANDS[1]: args.band_450.resolve(),
    }
    for path in band_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "sources": {band: str(path) for band, path in band_paths.items()},
        "contract": {
            "same_noise_factors_and_realization_ids_required_between_bands": True,
            "aggregation": "mean(abs(metric)) within band, noise source, noise level over r0000-r0009",
            "boundary_rule": "include every realization regardless of boundary_hits",
            "clean_realization_count": 1,
            "ordinary_realization_count": 10,
            "image_size_px": list(FIGURE_SIZE_PX),
            "font_size_pt": 14,
        },
        "sheets": {},
        "images": [],
    }
    for sheet in SHEETS:
        frames = {band: _load_sheet(path, sheet) for band, path in band_paths.items()}
        if set(frames[BANDS[0]]["source_key"]) != set(frames[BANDS[1]]["source_key"]):
            raise ValueError(f"The two bands do not have identical factor/realization keys in sheet {sheet}.")
        aggregates = {band: _aggregate(frame) for band, frame in frames.items()}
        image_name = f"v12_fixed_angle_{sheet}_air_film_band_heatmap_1000x800px.png"
        _plot(output_dir / image_name, sheet, aggregates)
        summary["sheets"][sheet] = aggregates
        summary["images"].append(image_name)

    with (output_dir / "v12_fixed_angle_band_heatmap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "images": summary["images"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
