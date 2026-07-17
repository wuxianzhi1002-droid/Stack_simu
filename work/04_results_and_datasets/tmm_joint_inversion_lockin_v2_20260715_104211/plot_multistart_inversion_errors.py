from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import LogLocator

PARAMETERS = ["Air", "HSQ", "PSS", "SOC", "TiO2"]
FILM_PARAMETERS = ["HSQ", "PSS", "SOC", "TiO2"]
TRUTH = {
    "Air": 1000.0,
    "HSQ": 30.0,
    "PSS": 10.0,
    "SOC": 40.0,
    "TiO2": 40.0,
}
UNITS = {"Air": "um", "HSQ": "nm", "PSS": "nm", "SOC": "nm", "TiO2": "nm"}
MODE_TITLES = {
    "I": "I-only",
    "D": "dI/dL-only",
    "joint": "Joint I + dI/dL",
}


def configure_chinese_font() -> None:
    """Use an installed CJK font when available."""
    installed = {item.name for item in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def label_starts(group: pd.DataFrame) -> pd.DataFrame:
    """Identify the nominal/truth start and number all remaining starts by rank."""
    group = group.sort_values("rank").copy()
    truth_mask = np.ones(len(group), dtype=bool)
    for parameter in PARAMETERS:
        column = f"x0_{parameter}_{UNITS[parameter]}"
        truth_mask &= np.isclose(
            group[column].to_numpy(dtype=float),
            TRUTH[parameter],
            rtol=0.0,
            atol=1e-8,
        )

    truth_indices = group.index[truth_mask]
    if len(truth_indices) != 1:
        raise ValueError(
            f"Mode {group['mode'].iloc[0]!r}: expected exactly one truth start, "
            f"found {len(truth_indices)}."
        )

    group["start_label"] = ""
    group["start_order"] = -1
    truth_index = truth_indices[0]
    group.loc[truth_index, ["start_label", "start_order"]] = ["\u771f\u503c\u8d77\u70b9", 0]

    other_number = 1
    for index in group.index:
        if index == truth_index:
            continue
        group.loc[index, ["start_label", "start_order"]] = [f"\u5176\u4ed6\u8d77\u70b9{other_number}", other_number]
        other_number += 1

    return group.sort_values("start_order")


def build_error_table(source: pd.DataFrame) -> pd.DataFrame:
    labeled_groups = []
    for _, group in source.groupby("mode", sort=False):
        labeled_groups.append(label_starts(group))
    labeled = pd.concat(labeled_groups, ignore_index=True)

    records: list[dict[str, object]] = []
    for _, row in labeled.iterrows():
        for parameter in PARAMETERS:
            unit = UNITS[parameter]
            fit = float(row[f"fit_{parameter}_{unit}"])
            initial = float(row[f"x0_{parameter}_{unit}"])
            signed_error = fit - TRUTH[parameter]
            records.append(
                {
                    "mode": row["mode"],
                    "rank": int(row["rank"]),
                    "start_label": row["start_label"],
                    "start_order": int(row["start_order"]),
                    "parameter": parameter,
                    "unit": unit,
                    "truth": TRUTH[parameter],
                    "initial": initial,
                    "fit": fit,
                    "signed_error": signed_error,
                    "absolute_error": abs(signed_error),
                }
            )
    return pd.DataFrame.from_records(records)


def start_color(index: int) -> tuple[float, float, float, float]:
    colors = plt.get_cmap("tab10").colors
    return colors[index % len(colors)]


def grouped_bars(
    ax: plt.Axes,
    mode_data: pd.DataFrame,
    parameters: list[str],
    y_label: str,
    show_legend: bool,
) -> None:
    starts = (
        mode_data[["start_label", "start_order"]]
        .drop_duplicates()
        .sort_values("start_order")
    )
    x = np.arange(len(parameters), dtype=float)
    n_starts = len(starts)
    total_width = 0.84
    width = total_width / n_starts

    for series_index, start in enumerate(starts.itertuples(index=False)):
        subset = mode_data[mode_data["start_label"] == start.start_label]
        values = [
            float(subset.loc[subset["parameter"] == parameter, "absolute_error"].iloc[0])
            for parameter in parameters
        ]
        positions = x - total_width / 2 + width / 2 + series_index * width
        color = start_color(series_index)
        ax.bar(
            positions,
            values,
            width=width * 0.92,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            label=start.start_label,
            zorder=3,
        )

    ax.set_xticks(x, parameters)
    ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center")


def plot_combined(error_table: pd.DataFrame, output_dir: Path) -> list[Path]:
    modes = [mode for mode in ["I", "D", "joint"] if mode in set(error_table["mode"])]
    fig, axes = plt.subplots(
        len(modes),
        2,
        figsize=(15, 4.2 * len(modes)),
        constrained_layout=True,
        squeeze=False,
    )

    legend_handles = None
    legend_labels = None
    for row_index, mode in enumerate(modes):
        mode_data = error_table[error_table["mode"] == mode]
        grouped_bars(
            axes[row_index, 0],
            mode_data,
            ["Air"],
            "Air \u7edd\u5bf9\u53cd\u6f14\u8bef\u5dee (\u03bcm, log)",
            show_legend=False,
        )
        grouped_bars(
            axes[row_index, 1],
            mode_data,
            FILM_PARAMETERS,
            "\u8584\u819c\u7edd\u5bf9\u53cd\u6f14\u8bef\u5dee (nm, log)",
            show_legend=False,
        )
        axes[row_index, 0].set_title(f"{MODE_TITLES.get(mode, mode)}: Air")
        axes[row_index, 1].set_title(f"{MODE_TITLES.get(mode, mode)}: films")
        if legend_handles is None:
            legend_handles, legend_labels = axes[row_index, 1].get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            ncol=4,
            frameon=False,
            loc="outside upper center",
        )

    png_path = output_dir / "multistart_inversion_absolute_errors.png"
    pdf_path = output_dir / "multistart_inversion_absolute_errors.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_each_mode(error_table: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []
    for mode in ["I", "D", "joint"]:
        mode_data = error_table[error_table["mode"] == mode]
        if mode_data.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
        grouped_bars(
            axes[0],
            mode_data,
            ["Air"],
            "Air \u7edd\u5bf9\u53cd\u6f14\u8bef\u5dee (\u03bcm, log)",
            show_legend=False,
        )
        grouped_bars(
            axes[1],
            mode_data,
            FILM_PARAMETERS,
            "\u8584\u819c\u7edd\u5bf9\u53cd\u6f14\u8bef\u5dee (nm, log)",
            show_legend=False,
        )
        axes[0].set_title(f"{MODE_TITLES.get(mode, mode)}: Air")
        axes[1].set_title(f"{MODE_TITLES.get(mode, mode)}: films")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, frameon=False, loc="outside upper center")
        output_path = output_dir / f"multistart_inversion_absolute_errors_{mode}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    default_csv = Path(__file__).resolve().with_name("multistart_results.csv")
    parser = argparse.ArgumentParser(
        description="Plot absolute inversion errors for every multistart and mode."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="Input multistart_results.csv (default: file beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: directory containing the input CSV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_chinese_font()
    source = pd.read_csv(csv_path)
    error_table = build_error_table(source)
    error_csv = output_dir / "multistart_inversion_errors_tidy.csv"
    error_table.to_csv(error_csv, index=False, encoding="utf-8-sig", float_format="%.10g")

    output_paths = [error_csv]
    output_paths.extend(plot_combined(error_table, output_dir))
    output_paths.extend(plot_each_mode(error_table, output_dir))

    print("Generated files:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
