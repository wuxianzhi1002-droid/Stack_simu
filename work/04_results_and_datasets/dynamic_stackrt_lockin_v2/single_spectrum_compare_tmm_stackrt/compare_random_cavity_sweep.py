from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_single_spectrum import (
    FIXED_THICKNESSES_UM,
    FREQUENCY_AXIS_C_M_S,
    LAYERS,
    material_n,
    stackrt_matched_tmm_rp,
)

LUMERICAL_API_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
LUMERICAL_BIN_PATH = Path(r"D:\Program Files\Lumerical\v241\bin")
REFERENCE_CAVITY_UM = 1000.0


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare Lumerical StackRT and matched TMM spectra at randomly "
            "distributed air-cavity lengths."
        )
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--generator",
        choices=("lumapi", "npz"),
        default="lumapi",
        help=(
            "Use fresh Lumerical StackRT calls or randomly sample existing "
            "dynamic_spectra_*.npz StackRT results."
        ),
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=script_dir.parent,
        help="Directory containing dynamic_spectra_*.npz for --generator npz.",
    )
    parser.add_argument("--cavity-min-um", type=float, default=999.0)
    parser.add_argument("--cavity-max-um", type=float, default=1001.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--wavelength-start-um", type=float, default=0.2)
    parser.add_argument("--wavelength-stop-um", type=float, default=0.6)
    parser.add_argument("--spectral-resolution-nm", type=float, default=0.02)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "random_cavity_sweep_stackrt_tmm",
    )
    parser.add_argument(
        "--show-lumerical-ui",
        action="store_true",
        help="Open the Lumerical UI instead of running the API session hidden.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples < 30:
        raise ValueError("--samples must be at least 30")
    if args.cavity_max_um <= args.cavity_min_um:
        raise ValueError("--cavity-max-um must be greater than --cavity-min-um")
    if args.wavelength_stop_um <= args.wavelength_start_um:
        raise ValueError("Wavelength stop must be greater than wavelength start")
    if args.spectral_resolution_nm <= 0.0:
        raise ValueError("--spectral-resolution-nm must be positive")


def wavelength_axis(args: argparse.Namespace) -> np.ndarray:
    span_nm = (args.wavelength_stop_um - args.wavelength_start_um) * 1000.0
    point_count = int(round(span_nm / args.spectral_resolution_nm)) + 1
    return np.linspace(
        args.wavelength_start_um,
        args.wavelength_stop_um,
        point_count,
        dtype=np.float64,
    )


def random_cavity_lengths(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    lengths = rng.uniform(args.cavity_min_um, args.cavity_max_um, size=args.samples)
    if np.unique(lengths).size != args.samples:
        raise RuntimeError("Random cavity draw unexpectedly contained duplicate values")
    return np.sort(lengths)


def load_random_npz_spectra(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    npz_dir = args.npz_dir.resolve()
    candidates = sorted(npz_dir.glob("dynamic_spectra_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No dynamic_spectra_*.npz found in {npz_dir}")

    records: dict[float, tuple[float, np.ndarray, str]] = {}
    reference_wavelength_um: np.ndarray | None = None
    for path in candidates:
        with np.load(path) as data:
            wavelength_um = np.asarray(data["wavelengths"], dtype=np.float64)
            cavity_um = np.asarray(data["L_t"], dtype=np.float64)
            spectra = np.asarray(data["spectra"], dtype=np.float64)
        if spectra.shape != (cavity_um.size, wavelength_um.size):
            raise ValueError(f"Unexpected spectra shape in {path}")
        if reference_wavelength_um is None:
            reference_wavelength_um = wavelength_um
        elif not np.array_equal(reference_wavelength_um, wavelength_um):
            raise ValueError(f"Wavelength axis differs in {path}")
        for index, value_um in enumerate(cavity_um):
            key = round(float(value_um), 12)
            records[key] = (float(value_um), spectra[index].copy(), path.name)

    if len(records) < args.samples:
        raise ValueError(
            f"Only {len(records)} distinct cavity lengths are available in NPZ files; "
            f"cannot draw {args.samples} without replacement."
        )
    rng = np.random.default_rng(args.seed)
    keys = np.array(sorted(records), dtype=np.float64)
    selected_keys = np.sort(rng.choice(keys, size=args.samples, replace=False))
    cavity_lengths_um = np.array([records[float(key)][0] for key in selected_keys])
    stackrt_matrix = np.vstack([records[float(key)][1] for key in selected_keys])
    source_files = sorted({records[float(key)][2] for key in selected_keys})
    assert reference_wavelength_um is not None
    return reference_wavelength_um, cavity_lengths_um, stackrt_matrix, source_files


def load_lumapi():
    if not LUMERICAL_API_PATH.exists():
        raise FileNotFoundError(f"Lumerical Python API not found: {LUMERICAL_API_PATH}")
    api_path = str(LUMERICAL_API_PATH)
    if api_path not in sys.path:
        sys.path.append(api_path)
    if LUMERICAL_BIN_PATH.exists():
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(LUMERICAL_BIN_PATH)
    try:
        import lumapi
    except ImportError as exc:
        raise RuntimeError("Unable to import lumapi") from exc
    return lumapi


def stackrt_inputs(nominal_wavelength_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency_hz = FREQUENCY_AXIS_C_M_S / (nominal_wavelength_um * 1e-6)
    n_matrix = np.vstack([material_n(name, nominal_wavelength_um) for name in LAYERS])
    thicknesses_m = np.array(
        [
            0.0,
            REFERENCE_CAVITY_UM * 1e-6,
            FIXED_THICKNESSES_UM["HSQ"] * 1e-6,
            FIXED_THICKNESSES_UM["PSS"] * 1e-6,
            FIXED_THICKNESSES_UM["SOC"] * 1e-6,
            FIXED_THICKNESSES_UM["TiO2"] * 1e-6,
            0.0,
        ],
        dtype=np.float64,
    )
    return n_matrix, thicknesses_m, frequency_hz


def matching_metrics(
    stackrt_rp: np.ndarray,
    tmm_rp: np.ndarray,
    nominal_wavelength_um: np.ndarray,
) -> dict[str, float]:
    residual = stackrt_rp - tmm_rp
    abs_error = np.abs(residual)
    max_index = int(np.argmax(abs_error))
    correlation = float(np.corrcoef(stackrt_rp, tmm_rp)[0, 1])
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_signed_error": float(np.mean(residual)),
        "max_abs_error": float(abs_error[max_index]),
        "max_abs_error_wavelength_nm": float(nominal_wavelength_um[max_index] * 1000.0),
        "pearson_correlation": correlation,
        "one_minus_correlation": float(max(0.0, 1.0 - correlation)),
    }


def positive_plot_values(values: np.ndarray) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    return np.maximum(np.asarray(values, dtype=np.float64), tiny)


def save_metric_plot(
    output_dir: Path,
    cavity_lengths_um: np.ndarray,
    rows: list[dict[str, float]],
) -> None:
    cavity_offset_nm = (cavity_lengths_um - REFERENCE_CAVITY_UM) * 1000.0
    sample_ids = np.arange(len(rows))
    metrics = [
        ("mae", "MAE", "Reflectance error", True),
        ("rmse", "RMSE", "Reflectance error", True),
        ("max_abs_error", "Maximum absolute error", "Reflectance error", True),
        ("one_minus_correlation", "1 - Pearson correlation", "1 - correlation", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (key, title, ylabel, use_log_scale) in zip(axes.flat, metrics):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        plot_values = positive_plot_values(values) if use_log_scale else values
        ax.plot(cavity_offset_nm, plot_values, color="#6C757D", lw=0.8, alpha=0.7)
        scatter = ax.scatter(
            cavity_offset_nm,
            plot_values,
            c=sample_ids,
            cmap="viridis",
            s=38,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        if use_log_scale:
            ax.set_yscale("log")
        else:
            upper = max(float(np.max(values)) * 1.15, np.finfo(np.float64).eps)
            ax.set_ylim(-0.03 * upper, upper)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(title)
        ax.set_xlabel("Air-cavity offset from 1000 um (nm)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Sorted random sample index", shrink=0.85)
    fig.suptitle(
        f"StackRT-TMM agreement across {len(rows)} random air-cavity lengths",
        fontsize=14,
    )
    fig.savefig(output_dir / "random_cavity_sweep_matching_metrics.png", dpi=220)
    plt.close(fig)


def save_residual_heatmap(
    output_dir: Path,
    nominal_wavelength_um: np.ndarray,
    cavity_lengths_um: np.ndarray,
    residual_matrix: np.ndarray,
) -> None:
    wavelength_nm = nominal_wavelength_um * 1000.0
    cavity_offset_nm = (cavity_lengths_um - REFERENCE_CAVITY_UM) * 1000.0
    scale = float(np.max(np.abs(residual_matrix)))
    if scale == 0.0:
        scale = 1.0
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = ax.imshow(
        residual_matrix,
        aspect="auto",
        origin="lower",
        extent=[wavelength_nm[0], wavelength_nm[-1], cavity_offset_nm[0], cavity_offset_nm[-1]],
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )
    ax.set_title("Residual heatmap across random cavity lengths")
    ax.set_xlabel("Nominal wavelength (nm)")
    ax.set_ylabel("Air-cavity offset from 1000 um (nm)")
    fig.colorbar(image, ax=ax, label="StackRT Rp - TMM Rp")
    fig.savefig(output_dir / "random_cavity_sweep_residual_heatmap.png", dpi=220)
    plt.close(fig)


def save_worst_case_plot(
    output_dir: Path,
    nominal_wavelength_um: np.ndarray,
    cavity_lengths_um: np.ndarray,
    stackrt_matrix: np.ndarray,
    tmm_matrix: np.ndarray,
    rows: list[dict[str, float]],
) -> int:
    worst_index = int(np.argmax([row["mae"] for row in rows]))
    wavelength_nm = nominal_wavelength_um * 1000.0
    residual = stackrt_matrix[worst_index] - tmm_matrix[worst_index]
    detail_mask = (wavelength_nm >= 499.5) & (wavelength_nm <= 500.5)

    fig, (ax_top, ax_detail, ax_residual) = plt.subplots(
        3,
        1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [2.8, 2.2, 1.2]},
        constrained_layout=True,
    )
    ax_top.plot(wavelength_nm, stackrt_matrix[worst_index], color="#16697A", lw=0.9, label="StackRT")
    ax_top.plot(wavelength_nm, tmm_matrix[worst_index], color="#D1495B", lw=0.7, ls="--", label="TMM")
    ax_top.set_title(
        f"Worst MAE sample: cavity = {cavity_lengths_um[worst_index]:.9f} um, "
        f"MAE = {rows[worst_index]['mae']:.3e}"
    )
    ax_top.set_ylabel("Reflectance Rp")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right")

    ax_detail.plot(
        wavelength_nm[detail_mask],
        stackrt_matrix[worst_index, detail_mask],
        color="#16697A",
        lw=1.3,
        marker="o",
        ms=2.0,
        label="StackRT",
    )
    ax_detail.plot(
        wavelength_nm[detail_mask],
        tmm_matrix[worst_index, detail_mask],
        color="#D1495B",
        lw=1.0,
        ls="--",
        label="TMM",
    )
    ax_detail.set_title("Worst sample central 1 nm detail")
    ax_detail.set_ylabel("Reflectance Rp")
    ax_detail.grid(True, alpha=0.25)

    ax_residual.plot(wavelength_nm, residual, color="#6A4C93", lw=0.7)
    ax_residual.axhline(0.0, color="black", lw=0.7, alpha=0.7)
    ax_residual.set_xlabel("Nominal wavelength (nm)")
    ax_residual.set_ylabel("StackRT - TMM")
    ax_residual.grid(True, alpha=0.25)
    ax_residual.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(output_dir / "random_cavity_sweep_worst_case.png", dpi=220)
    plt.close(fig)
    return worst_index


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files: list[str] = []
    if args.generator == "npz":
        nominal_wavelength_um, cavity_lengths_um, stackrt_matrix, source_files = (
            load_random_npz_spectra(args)
        )
        frequency_hz = FREQUENCY_AXIS_C_M_S / (nominal_wavelength_um * 1e-6)
        print(
            f"Loaded {args.samples} random distinct cavity spectra from: "
            f"{', '.join(source_files)}"
        )
    else:
        nominal_wavelength_um = wavelength_axis(args)
        cavity_lengths_um = random_cavity_lengths(args)
        n_matrix, thicknesses_base_m, frequency_hz = stackrt_inputs(nominal_wavelength_um)
        stackrt_matrix = np.empty(
            (args.samples, nominal_wavelength_um.size), dtype=np.float64
        )
        lumapi = load_lumapi()
        print("Starting Lumerical FDTD API session for StackRT cavity sweep...")
        fdtd = lumapi.FDTD(hide=not args.show_lumerical_ui)
        try:
            for index, cavity_um in enumerate(cavity_lengths_um):
                thicknesses_m = thicknesses_base_m.copy()
                thicknesses_m[1] = cavity_um * 1e-6
                result = fdtd.stackrt(n_matrix, thicknesses_m, frequency_hz)
                stackrt_rp = np.asarray(result["Rp"], dtype=np.float64).reshape(-1)
                if stackrt_rp.size != nominal_wavelength_um.size:
                    raise ValueError(
                        f"Unexpected StackRT spectrum size {stackrt_rp.size}; "
                        f"expected {nominal_wavelength_um.size}"
                    )
                stackrt_matrix[index] = stackrt_rp
        finally:
            fdtd.close()

    tmm_matrix = np.empty_like(stackrt_matrix)
    rows: list[dict[str, float]] = []
    start_time = time.time()
    for index, cavity_um in enumerate(cavity_lengths_um):
        stackrt_rp = stackrt_matrix[index]
        tmm_rp, _, _ = stackrt_matched_tmm_rp(
            nominal_wavelength_um, float(cavity_um)
        )
        row = {
            "sample_index": index,
            "cavity_um": float(cavity_um),
            "cavity_offset_nm": float(
                (cavity_um - REFERENCE_CAVITY_UM) * 1000.0
            ),
            **matching_metrics(stackrt_rp, tmm_rp, nominal_wavelength_um),
        }
        tmm_matrix[index] = tmm_rp
        rows.append(row)
        elapsed = time.time() - start_time
        print(
            f"[{index + 1:02d}/{args.samples}] cavity={cavity_um:.9f} um "
            f"MAE={row['mae']:.3e} RMSE={row['rmse']:.3e} "
            f"max={row['max_abs_error']:.3e} elapsed={elapsed:.1f}s"
        )

    residual_matrix = stackrt_matrix - tmm_matrix
    fieldnames = [
        "sample_index",
        "cavity_um",
        "cavity_offset_nm",
        "mae",
        "rmse",
        "mean_signed_error",
        "max_abs_error",
        "max_abs_error_wavelength_nm",
        "pearson_correlation",
        "one_minus_correlation",
    ]
    with (output_dir / "random_cavity_sweep_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        output_dir / "random_cavity_sweep_spectra.npz",
        nominal_wavelength_um=nominal_wavelength_um,
        frequency_hz=frequency_hz,
        cavity_lengths_um=cavity_lengths_um,
        stackrt_Rp=stackrt_matrix,
        tmm_Rp=tmm_matrix,
        residual_stackrt_minus_tmm=residual_matrix,
    )

    save_metric_plot(output_dir, cavity_lengths_um, rows)
    save_residual_heatmap(output_dir, nominal_wavelength_um, cavity_lengths_um, residual_matrix)
    worst_index = save_worst_case_plot(
        output_dir,
        nominal_wavelength_um,
        cavity_lengths_um,
        stackrt_matrix,
        tmm_matrix,
        rows,
    )

    mae_values = np.array([row["mae"] for row in rows])
    rmse_values = np.array([row["rmse"] for row in rows])
    max_values = np.array([row["max_abs_error"] for row in rows])
    correlation_values = np.array([row["pearson_correlation"] for row in rows])
    summary = {
        "generator": args.generator,
        "source_npz_files": source_files,
        "samples": args.samples,
        "random_seed": args.seed,
        "cavity_min_um": args.cavity_min_um,
        "cavity_max_um": args.cavity_max_um,
        "actual_min_cavity_um": float(cavity_lengths_um.min()),
        "actual_max_cavity_um": float(cavity_lengths_um.max()),
        "wavelength_start_um": args.wavelength_start_um,
        "wavelength_stop_um": args.wavelength_stop_um,
        "spectral_resolution_nm": args.spectral_resolution_nm,
        "wavelength_points": int(nominal_wavelength_um.size),
        "mae_mean": float(mae_values.mean()),
        "mae_max": float(mae_values.max()),
        "rmse_mean": float(rmse_values.mean()),
        "rmse_max": float(rmse_values.max()),
        "max_abs_error_overall": float(max_values.max()),
        "pearson_correlation_min": float(correlation_values.min()),
        "worst_mae_sample_index": worst_index,
        "worst_mae_cavity_um": float(cavity_lengths_um[worst_index]),
        "elapsed_seconds": float(time.time() - start_time),
        "frequency_axis_definition": "f = 3e8 / nominal_wavelength",
        "tmm_phase_wavelength_definition": "lambda_phase = 299792458 / f",
        "complex_index_convention": "n + i*k with -i characteristic-matrix off-diagonal terms",
    }
    with (output_dir / "random_cavity_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    readme = f"""# Random cavity sweep: StackRT vs TMM

- Generator: `{args.generator}`
- Source NPZ files: `{', '.join(source_files) if source_files else 'fresh lumapi StackRT calls'}`
- Samples: `{args.samples}` random cavity lengths
- Random seed: `{args.seed}`
- Requested cavity range for lumapi mode: `{args.cavity_min_um}-{args.cavity_max_um} um`
- Actual sampled range: `{cavity_lengths_um.min():.9f}-{cavity_lengths_um.max():.9f} um`
- Wavelength range: `{args.wavelength_start_um}-{args.wavelength_stop_um} um`
- Spectral resolution: `{args.spectral_resolution_nm} nm`
- Wavelength points: `{nominal_wavelength_um.size}`
- Mean MAE: `{summary['mae_mean']:.12g}`
- Maximum MAE: `{summary['mae_max']:.12g}`
- Maximum absolute error over all spectra: `{summary['max_abs_error_overall']:.12g}`
- Minimum Pearson correlation: `{summary['pearson_correlation_min']:.12g}`

## Files

- `random_cavity_sweep_metrics.csv`: one row of matching metrics per random cavity length
- `random_cavity_sweep_spectra.npz`: cavity lengths, StackRT/TMM spectra, and residual matrix
- `random_cavity_sweep_matching_metrics.png`: MAE, RMSE, maximum error, and correlation comparison
- `random_cavity_sweep_residual_heatmap.png`: residual over wavelength and cavity length
- `random_cavity_sweep_worst_case.png`: worst-MAE spectrum and residual
- `random_cavity_sweep_summary.json`: configuration and aggregate metrics
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
