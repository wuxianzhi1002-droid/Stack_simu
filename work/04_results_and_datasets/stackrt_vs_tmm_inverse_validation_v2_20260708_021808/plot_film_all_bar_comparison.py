from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_DIR = Path(__file__).resolve().parent
METRICS_CSV = RESULT_DIR / "metrics_summary.csv"
OUTPUT_PNG = RESULT_DIR / "film_all_tmm_vs_stackrt_mae_bar.png"
OUTPUT_CSV = RESULT_DIR / "film_all_tmm_vs_stackrt_mae_bar_data.csv"

CASE_NAME = "B3_film_all"
METRIC = "MAE_nm"
PARAM_ORDER = ["HSQ", "PSS", "SOC", "TiO2"]
SCENARIO_LABELS = {
    "tmm_generator_nominal": "TMM forward",
    "stackrt_generator": "StackRT forward",
}
SCENARIO_ORDER = ["tmm_generator_nominal", "stackrt_generator"]
COLORS = {
    "tmm_generator_nominal": "#2563EB",
    "stackrt_generator": "#B91C1C",
}


def add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    df = pd.read_csv(METRICS_CSV)
    film = df[df["case"].eq(CASE_NAME)].copy()
    film = film[film["scenario"].isin(SCENARIO_ORDER)]
    if film.empty:
        raise RuntimeError(f"No rows found for case={CASE_NAME!r} in {METRICS_CSV}")

    noise_values = sorted(film["noise_sigma_reflectance"].unique())
    fig, axes = plt.subplots(
        1,
        len(noise_values),
        figsize=(6.4 * len(noise_values), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(noise_values) == 1:
        axes = [axes]

    x = np.arange(len(PARAM_ORDER))
    width = 0.34
    for ax, noise in zip(axes, noise_values):
        sub = film[film["noise_sigma_reflectance"].eq(noise)]
        for idx, scenario in enumerate(SCENARIO_ORDER):
            values = []
            for param in PARAM_ORDER:
                row = sub[sub["scenario"].eq(scenario) & sub["param"].eq(param)]
                values.append(float(row[METRIC].iloc[0]) if not row.empty else np.nan)
            offset = (idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=SCENARIO_LABELS[scenario],
                color=COLORS[scenario],
                alpha=0.9,
            )
            add_bar_labels(ax, bars)

        ax.set_title(f"{CASE_NAME}, noise sigma R = {noise:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_ORDER)
        ax.set_xlabel("Fitted parameter")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("MAE (nm)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Film-all inverse performance: TMM-generated vs StackRT-generated spectra", fontsize=14)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    film.sort_values(["noise_sigma_reflectance", "scenario", "param"]).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    print(f"Saved figure: {OUTPUT_PNG}")
    print(f"Saved plotted data: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
