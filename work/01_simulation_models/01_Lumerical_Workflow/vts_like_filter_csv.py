#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vts_like_filter.py

A practical VTS-like OPD/depth-domain spectral filter.

It implements the public idea of VTS-like filtering:
    spectrum I(lambda)
    -> uniform wavenumber sigma = 1/lambda
    -> FFT to OPD domain
    -> lowpass / highpass / bandpass mask in OPD domain
    -> inverse FFT back to filtered spectrum

This is NOT the proprietary VTS algorithm from Schmidt et al. It is a reproducible
approximation useful for testing whether OPD-domain filtering helps your stackrt / ML data.

Required packages:
    numpy
Optional for diagnostic figures:
    matplotlib

Example 1: keep shallow/top optical information, remove deeper OPD content
    python vts_like_filter.py --input-npz dataset.npz --mode lowpass --opd-cutoff-um 3 --opd-transition-um 0.5

Example 2: keep 1 mm air-cavity peak, OPD≈2L≈2000 um
    python vts_like_filter.py --input-npz dataset.npz --mode bandpass --opd-center-um 2000 --opd-half-width-um 50 --opd-transition-um 20

Input .npz expected keys:
    wavelengths_um / wavelengths / wavelength_um / wavelength_nm / lambda_um / lambda_nm
    spectra / spectra_norm / R / Rp / Rs / intensities

Output directory contains:
    config.json
    summary.json
    failed_cases.json
    spectra_<mode>_filtered.npy
    dataset_vts_like_<mode>.npz
    diagnostic_sample_*.png
    debug_sample_*.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# Fill these defaults if you prefer running the script without command-line paths.
# Command-line --input-csv / --input-npz / --output-dir still overrides these values.
DEFAULT_INPUT_CSV: str | None = None
DEFAULT_INPUT_NPZ: str | None = None
DEFAULT_OUTPUT_DIR: str | None = None


@dataclass
class VTSConfig:
    input_npz: str | None = None
    input_csv: str | None = None
    output_dir: str | None = None
    wavelength_key: str | None = None
    spectra_key: str | None = None
    csv_wavelength_col: str = "wavelength_um"
    csv_reflectance_col: str = "reflectance_Rp"
    wavelength_unit: Literal["auto", "um", "nm"] = "auto"

    mode: Literal["lowpass", "highpass", "bandpass"] = "bandpass"
    opd_cutoff_um: float = 3.0
    opd_center_um: float = 2000.0
    opd_half_width_um: float = 80.0
    opd_transition_um: float = 20.0

    num_sigma_points: int | None = None
    detrend: Literal["poly", "mean", "none"] = "poly"
    poly_order: int = 3
    sigma_taper: Literal["none", "tukey"] = "none"
    tukey_alpha: float = 0.05

    chunk_size: int = 1024
    max_samples: int | None = None
    dtype_out: Literal["float32", "float64"] = "float32"

    diagnostic_indices: str = "0"
    diagnostic_opd_max_um: float | None = None
    save_debug_npz: bool = True


def parse_args() -> VTSConfig:
    parser = argparse.ArgumentParser(description="VTS-like OPD-domain spectral filtering.")
    parser.add_argument("--input-npz", default=None, help="Input .npz dataset path.")
    parser.add_argument("--input-csv", default="01_Lumerical_Workflow\\stackrt_result\\stackrt_reflectance_20260630_195814.csv", help="Input CSV path with wavelength and reflectance columns.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default is created next to input file.")
    parser.add_argument("--wavelength-key", default=None, help="Wavelength key. Auto-detect if omitted.")
    parser.add_argument("--spectra-key", default=None, help="Spectra key. Auto-detect if omitted.")
    parser.add_argument("--csv-wavelength-col", default="wavelength_um", help="CSV wavelength column name.")
    parser.add_argument("--csv-reflectance-col", default="reflectance_Rp", help="CSV reflectance column name.")
    parser.add_argument("--wavelength-unit", choices=["auto", "um", "nm"], default="auto")

    parser.add_argument("--mode", choices=["lowpass", "highpass", "bandpass"], default="bandpass")
    parser.add_argument("--opd-cutoff-um", type=float, default=3.0, help="Low/high-pass cutoff in OPD um.")
    parser.add_argument("--opd-center-um", type=float, default=2000.0, help="Bandpass center in OPD um.")
    parser.add_argument("--opd-half-width-um", type=float, default=80.0, help="Bandpass half width in OPD um.")
    parser.add_argument("--opd-transition-um", type=float, default=20.0, help="Smooth transition width in OPD um.")

    parser.add_argument("--num-sigma-points", type=int, default=None, help="Uniform sigma points. Default equals wavelength length.")
    parser.add_argument("--detrend", choices=["poly", "mean", "none"], default="poly")
    parser.add_argument("--poly-order", type=int, default=3)
    parser.add_argument("--sigma-taper", choices=["none", "tukey"], default="none")
    parser.add_argument("--tukey-alpha", type=float, default=0.05)

    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype-out", choices=["float32", "float64"], default="float32")
    parser.add_argument("--diagnostic-indices", default="0", help='Comma-separated indices, e.g. "0,10,100"')
    parser.add_argument("--diagnostic-opd-max-um", type=float, default=None)
    parser.add_argument("--save-debug-npz", action="store_true", default=True)
    parser.add_argument("--no-save-debug-npz", action="store_false", dest="save_debug_npz")
    args = parser.parse_args()
    args.input_csv = args.input_csv or DEFAULT_INPUT_CSV
    args.input_npz = args.input_npz or DEFAULT_INPUT_NPZ
    args.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if (args.input_npz is None) == (args.input_csv is None):
        parser.error(
            "Specify exactly one of --input-npz or --input-csv, "
            "or set DEFAULT_INPUT_CSV / DEFAULT_INPUT_NPZ near the top of this file."
        )
    return VTSConfig(**vars(args))


def read_numeric_csv_columns(path: Path, wavelength_col: str, reflectance_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read wavelength and reflectance columns from a CSV file."""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        missing = [name for name in [wavelength_col, reflectance_col] if name not in fieldnames]
        if missing:
            raise KeyError(f"CSV missing columns {missing}. Available columns: {fieldnames}")
        wavelengths = []
        reflectance = []
        for row_index, row in enumerate(reader, start=2):
            try:
                wavelengths.append(float(row[wavelength_col]))
                reflectance.append(float(row[reflectance_col]))
            except Exception as exc:
                raise ValueError(f"Invalid numeric value in {path} at row {row_index}: {exc}") from exc

    return np.asarray(wavelengths, dtype=np.float64), np.asarray(reflectance, dtype=np.float64), fieldnames


def choose_key(files: list[str], candidates: list[str], what: str) -> str:
    for key in candidates:
        if key in files:
            return key
    raise KeyError(f"Cannot find {what}. Available keys: {files}")


def infer_wavelength_um(wavelength: np.ndarray, unit: str) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=np.float64).ravel()
    if unit == "um":
        return wavelength
    if unit == "nm":
        return wavelength / 1000.0
    median = float(np.nanmedian(wavelength))
    return wavelength / 1000.0 if median > 20 else wavelength


def parse_index_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def tukey_window(n: int, alpha: float = 0.05) -> np.ndarray:
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        return np.hanning(n)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    left = x < alpha / 2
    right = x > 1 - alpha / 2
    w[left] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[left] - alpha / 2)))
    w[right] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[right] - 1 + alpha / 2)))
    return w


def smooth_lowpass_mask(opd_um: np.ndarray, cutoff_um: float, transition_um: float) -> np.ndarray:
    """Keep |OPD| <= cutoff_um, smoothly decay to zero in transition region."""
    a = np.abs(opd_um)
    mask = np.ones_like(a, dtype=np.float64)
    pass_region = a <= cutoff_um
    stop_region = a >= cutoff_um + transition_um
    trans_region = (~pass_region) & (~stop_region)
    mask[stop_region] = 0.0
    if np.any(trans_region):
        x = (a[trans_region] - cutoff_um) / max(transition_um, 1e-15)
        mask[trans_region] = 0.5 * (1.0 + np.cos(np.pi * x))
    return mask


def smooth_highpass_mask(opd_um: np.ndarray, cutoff_um: float, transition_um: float) -> np.ndarray:
    """Remove |OPD| <= cutoff_um, keep large OPD."""
    return 1.0 - smooth_lowpass_mask(opd_um, cutoff_um, transition_um)


def smooth_bandpass_mask(opd_um: np.ndarray, center_um: float, half_width_um: float, transition_um: float) -> np.ndarray:
    """Keep |OPD| around center_um ± half_width_um. Real-spectrum FFT is symmetric, so abs(OPD) is used."""
    a = np.abs(opd_um)
    dist = np.abs(a - center_um)
    mask = np.zeros_like(a, dtype=np.float64)
    pass_region = dist <= half_width_um
    stop_region = dist >= half_width_um + transition_um
    trans_region = (~pass_region) & (~stop_region)
    mask[pass_region] = 1.0
    if np.any(trans_region):
        x = (dist[trans_region] - half_width_um) / max(transition_um, 1e-15)
        mask[trans_region] = 0.5 * (1.0 + np.cos(np.pi * x))
    mask[stop_region] = 0.0
    return mask


def build_mask(opd_um: np.ndarray, cfg: VTSConfig) -> np.ndarray:
    if cfg.mode == "lowpass":
        return smooth_lowpass_mask(opd_um, cfg.opd_cutoff_um, cfg.opd_transition_um)
    if cfg.mode == "highpass":
        return smooth_highpass_mask(opd_um, cfg.opd_cutoff_um, cfg.opd_transition_um)
    if cfg.mode == "bandpass":
        return smooth_bandpass_mask(opd_um, cfg.opd_center_um, cfg.opd_half_width_um, cfg.opd_transition_um)
    raise ValueError(f"Unknown mode: {cfg.mode}")


@dataclass
class PreparedGrid:
    wavelengths_um: np.ndarray
    sigma_original: np.ndarray
    sigma_order: np.ndarray
    sigma_sorted: np.ndarray
    sigma_uniform: np.ndarray
    d_sigma: float
    opd_um: np.ndarray
    mask: np.ndarray


def prepare_grid(wavelengths_um: np.ndarray, cfg: VTSConfig) -> PreparedGrid:
    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64).ravel()
    if np.any(~np.isfinite(wavelengths_um)) or np.any(wavelengths_um <= 0):
        raise ValueError("Invalid wavelengths_um.")
    sigma_original = 1.0 / wavelengths_um
    order = np.argsort(sigma_original)
    sigma_sorted = sigma_original[order]
    n = cfg.num_sigma_points or len(sigma_sorted)
    if n < 8:
        raise ValueError("num_sigma_points is too small.")
    sigma_uniform = np.linspace(float(sigma_sorted.min()), float(sigma_sorted.max()), n)
    d_sigma = float(sigma_uniform[1] - sigma_uniform[0])
    opd_um = np.fft.fftfreq(n, d=d_sigma)
    mask = build_mask(opd_um, cfg)
    return PreparedGrid(wavelengths_um, sigma_original, order, sigma_sorted, sigma_uniform, d_sigma, opd_um, mask)


def detrend_spectrum(sigma_uniform: np.ndarray, y_uniform: np.ndarray, cfg: VTSConfig) -> tuple[np.ndarray, np.ndarray]:
    if cfg.detrend == "none":
        bg = np.zeros_like(y_uniform, dtype=np.float64)
        return y_uniform.astype(np.float64), bg
    if cfg.detrend == "mean":
        bg = np.full_like(y_uniform, float(np.mean(y_uniform)))
        return y_uniform - bg, bg
    if cfg.detrend == "poly":
        sigma_norm = (sigma_uniform - sigma_uniform.mean()) / (np.ptp(sigma_uniform) + 1e-15)
        coeff = np.polyfit(sigma_norm, y_uniform, deg=cfg.poly_order)
        bg = np.polyval(coeff, sigma_norm)
        return y_uniform - bg, bg
    raise ValueError(f"Unknown detrend: {cfg.detrend}")


def filter_one_spectrum(spectrum: np.ndarray, grid: PreparedGrid, cfg: VTSConfig, return_debug: bool = False):
    y = np.asarray(spectrum, dtype=np.float64).ravel()
    if len(y) != len(grid.wavelengths_um):
        raise ValueError(f"Spectrum length {len(y)} != wavelength length {len(grid.wavelengths_um)}")

    y_sorted = y[grid.sigma_order]
    y_uniform = np.interp(grid.sigma_uniform, grid.sigma_sorted, y_sorted)
    fringe, background = detrend_spectrum(grid.sigma_uniform, y_uniform, cfg)

    if cfg.sigma_taper == "tukey":
        taper = tukey_window(len(fringe), cfg.tukey_alpha)
        fringe_for_fft = fringe * taper
    else:
        taper = np.ones_like(fringe)
        fringe_for_fft = fringe

    fft_data = np.fft.fft(fringe_for_fft)
    fft_filtered = fft_data * grid.mask
    fringe_filtered = np.real(np.fft.ifft(fft_filtered))
    y_filtered_uniform = background + fringe_filtered
    y_filtered_original = np.interp(grid.sigma_original, grid.sigma_uniform, y_filtered_uniform)

    if not return_debug:
        return y_filtered_original
    return y_filtered_original, {
        "y_uniform": y_uniform,
        "background": background,
        "fringe": fringe,
        "taper": taper,
        "fft_data": fft_data,
        "fft_filtered": fft_filtered,
        "y_filtered_uniform": y_filtered_uniform,
    }


def save_diagnostic_plot(out_png: Path, grid: PreparedGrid, debug: dict, original: np.ndarray, filtered: np.ndarray, cfg: VTSConfig):
    import matplotlib.pyplot as plt

    opd = grid.opd_um
    pos = opd >= 0
    opd_pos = opd[pos]
    amp = np.abs(debug["fft_data"])[pos]
    amp_f = np.abs(debug["fft_filtered"])[pos]
    mask = grid.mask[pos]

    if cfg.diagnostic_opd_max_um is not None:
        x_max = cfg.diagnostic_opd_max_um
    elif cfg.mode == "bandpass":
        x_max = cfg.opd_center_um + cfg.opd_half_width_um + 5 * cfg.opd_transition_um
    else:
        x_max = max(10.0, cfg.opd_cutoff_um + 5 * cfg.opd_transition_um)
    m = opd_pos <= x_max

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(grid.wavelengths_um, original, lw=1, label="original")
    ax.plot(grid.wavelengths_um, filtered, lw=1, label="filtered")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Signal")
    ax.set_title("Original vs filtered")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(1.0 / grid.sigma_uniform, debug["y_uniform"], lw=1, label="uniform-sigma original")
    ax.plot(1.0 / grid.sigma_uniform, debug["background"], lw=1, label="background")
    ax.plot(1.0 / grid.sigma_uniform, debug["y_filtered_uniform"], lw=1, label="uniform-sigma filtered")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Signal")
    ax.set_title("Uniform sigma grid")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    norm = np.max(amp[m]) + 1e-15 if np.any(m) else np.max(amp) + 1e-15
    ax.plot(opd_pos[m], amp[m] / norm, lw=1, label="OPD amplitude")
    ax.plot(opd_pos[m], amp_f[m] / norm, lw=1, label="filtered OPD amplitude")
    ax.plot(opd_pos[m], mask[m], "k--", lw=1, label="mask")
    ax.set_xlabel("OPD (um)")
    ax.set_ylabel("Normalized amplitude / mask")
    ax.set_title("OPD-domain filtering")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(grid.wavelengths_um, original - filtered, lw=1)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Original - filtered")
    ax.set_title("Removed component")

    fig.suptitle(f"VTS-like filter: {cfg.mode}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_debug_npz(out_npz: Path, grid: PreparedGrid, debug: dict, filtered: np.ndarray):
    np.savez_compressed(
        out_npz,
        sigma_uniform=grid.sigma_uniform,
        opd_um=grid.opd_um,
        mask=grid.mask,
        y_uniform=debug["y_uniform"],
        background=debug["background"],
        fringe=debug["fringe"],
        fft_data=debug["fft_data"],
        fft_filtered=debug["fft_filtered"],
        y_filtered_uniform=debug["y_filtered_uniform"],
        filtered_spectrum=filtered,
    )


def process_dataset(cfg: VTSConfig) -> dict:
    if cfg.input_csv is not None:
        return process_csv_dataset(cfg)
    return process_npz_dataset(cfg)


def process_csv_dataset(cfg: VTSConfig) -> dict:
    input_path = Path(cfg.input_csv).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else input_path.parent / f"vts_like_{cfg.mode}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wavelengths_raw, spectrum, csv_columns = read_numeric_csv_columns(
        input_path,
        cfg.csv_wavelength_col,
        cfg.csv_reflectance_col,
    )
    wavelengths_um = infer_wavelength_um(wavelengths_raw, cfg.wavelength_unit)
    spectra_2d = spectrum.reshape(1, -1)

    n_samples, n_lambda = spectra_2d.shape
    grid = prepare_grid(wavelengths_um, cfg)
    opd_pos = grid.opd_um[grid.opd_um >= 0]
    opd_max = float(np.max(opd_pos))
    opd_res = float(opd_pos[1] - opd_pos[0]) if len(opd_pos) > 1 else float("nan")

    dtype = np.float32 if cfg.dtype_out == "float32" else np.float64
    out_npy = out_dir / f"spectra_{cfg.mode}_filtered.npy"
    filtered_mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=dtype, shape=(n_samples, n_lambda))

    print(f"Input CSV: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"CSV wavelength column: {cfg.csv_wavelength_col}; reflectance column: {cfg.csv_reflectance_col}")
    print(f"Using spectra shape: {(n_samples, n_lambda)}")
    print(f"sigma range: {grid.sigma_uniform.min():.6g} to {grid.sigma_uniform.max():.6g} um^-1")
    print(f"OPD positive max ~= {opd_max:.3f} um; OPD resolution ~= {opd_res:.6f} um")
    if cfg.mode == "bandpass" and cfg.opd_center_um > opd_max:
        print("WARNING: bandpass center is larger than OPD max. Increase spectral resolution or check units.")

    failed = []
    try:
        filtered_mm[0] = filter_one_spectrum(spectrum, grid, cfg, return_debug=False).astype(dtype)
    except Exception as exc:
        failed.append({"sample_index": 0, "error": repr(exc)})
        filtered_mm[0] = np.nan
    filtered_mm.flush()
    print("Processed 1/1")

    diag_indices = [idx for idx in parse_index_list(cfg.diagnostic_indices) if idx == 0]
    for idx in diag_indices:
        original = np.asarray(spectra_2d[idx], dtype=np.float64)
        filtered, debug = filter_one_spectrum(original, grid, cfg, return_debug=True)
        png = out_dir / f"diagnostic_sample_{idx:06d}.png"
        try:
            save_diagnostic_plot(png, grid, debug, original, filtered, cfg)
        except Exception as exc:
            print(f"WARNING: diagnostic plot failed for sample {idx}: {exc}")
        if cfg.save_debug_npz:
            save_debug_npz(out_dir / f"debug_sample_{idx:06d}.npz", grid, debug, filtered)

    meta_npz = out_dir / f"dataset_vts_like_{cfg.mode}.npz"
    np.savez_compressed(
        meta_npz,
        wavelengths_um=wavelengths_um,
        filtered_spectra_npy_path=str(out_npy),
        source_csv_path=str(input_path),
        source_wavelength_column=cfg.csv_wavelength_col,
        source_reflectance_column=cfg.csv_reflectance_col,
        csv_columns=np.asarray(csv_columns, dtype=object),
        mode=cfg.mode,
        config_json=json.dumps(asdict(cfg), ensure_ascii=False),
        sigma_uniform=grid.sigma_uniform,
        opd_um=grid.opd_um,
        mask=grid.mask,
        opd_positive_max_um=opd_max,
        opd_resolution_um=opd_res,
        failed_count=len(failed),
    )

    filtered_csv = out_dir / f"reflectance_{cfg.mode}_filtered.csv"
    filtered = np.asarray(filtered_mm[0], dtype=np.float64)
    with filtered_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_um", "reflectance_original", "reflectance_filtered"])
        writer.writerows(zip(wavelengths_um, spectrum, filtered))

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "failed_cases.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "input_csv": str(input_path),
        "output_dir": str(out_dir),
        "csv_wavelength_col": cfg.csv_wavelength_col,
        "csv_reflectance_col": cfg.csv_reflectance_col,
        "n_samples": int(n_samples),
        "n_lambda": int(n_lambda),
        "filtered_spectra_npy_path": str(out_npy),
        "filtered_csv_path": str(filtered_csv),
        "metadata_npz_path": str(meta_npz),
        "opd_positive_max_um": opd_max,
        "opd_resolution_um": opd_res,
        "mode": cfg.mode,
        "failed_count": len(failed),
        "diagnostic_indices": diag_indices,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Filtered spectra NPY: {out_npy}")
    print(f"Filtered CSV: {filtered_csv}")
    print(f"Metadata NPZ: {meta_npz}")
    print(f"Summary: {summary_path}")
    return summary


def process_npz_dataset(cfg: VTSConfig) -> dict:
    input_path = Path(cfg.input_npz).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else input_path.parent / f"vts_like_{cfg.mode}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=True, mmap_mode="r") as data:
        files = list(data.files)
        wavelength_key = cfg.wavelength_key or choose_key(
            files,
            ["wavelengths_um", "wavelength_um", "wavelengths", "wavelength", "lambda_um", "lambda_nm", "wavelength_nm"],
            "wavelength key",
        )
        spectra_key = cfg.spectra_key or choose_key(
            files,
            ["spectra", "spectra_norm", "R", "Rp", "Rs", "intensities"],
            "spectra key",
        )

        unit = cfg.wavelength_unit
        if unit == "auto":
            lower = wavelength_key.lower()
            if "nm" in lower:
                unit = "nm"
            elif "um" in lower:
                unit = "um"
        wavelengths_um = infer_wavelength_um(data[wavelength_key], unit)

        spectra = data[spectra_key]
        if spectra.ndim == 1:
            spectra_2d = spectra.reshape(1, -1)
        elif spectra.ndim == 2:
            spectra_2d = spectra
        else:
            raise ValueError(f"{spectra_key} must be 1D or 2D, got shape {spectra.shape}")

        if spectra_2d.shape[1] != len(wavelengths_um):
            if spectra_2d.shape[0] == len(wavelengths_um):
                spectra_2d = spectra_2d.T
            else:
                raise ValueError(f"spectra shape {spectra_2d.shape} does not match wavelength length {len(wavelengths_um)}")

        n_total, n_lambda = spectra_2d.shape
        n_samples = n_total if cfg.max_samples is None else min(n_total, cfg.max_samples)
        grid = prepare_grid(wavelengths_um, cfg)

        opd_pos = grid.opd_um[grid.opd_um >= 0]
        opd_max = float(np.max(opd_pos))
        opd_res = float(opd_pos[1] - opd_pos[0]) if len(opd_pos) > 1 else float("nan")

        dtype = np.float32 if cfg.dtype_out == "float32" else np.float64
        out_npy = out_dir / f"spectra_{cfg.mode}_filtered.npy"
        filtered_mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=dtype, shape=(n_samples, n_lambda))

        print(f"Input: {input_path}")
        print(f"Output dir: {out_dir}")
        print(f"Wavelength key: {wavelength_key}; spectra key: {spectra_key}")
        print(f"Using spectra shape: {(n_samples, n_lambda)}")
        print(f"sigma range: {grid.sigma_uniform.min():.6g} to {grid.sigma_uniform.max():.6g} um^-1")
        print(f"OPD positive max ≈ {opd_max:.3f} um; OPD resolution ≈ {opd_res:.6f} um")
        if cfg.mode == "bandpass" and cfg.opd_center_um > opd_max:
            print("WARNING: bandpass center is larger than OPD max. Increase spectral resolution or check units.")

        failed = []
        chunk = max(1, int(cfg.chunk_size))
        for start in range(0, n_samples, chunk):
            end = min(n_samples, start + chunk)
            block = np.asarray(spectra_2d[start:end], dtype=np.float64)
            for j in range(end - start):
                idx = start + j
                try:
                    filtered_mm[idx] = filter_one_spectrum(block[j], grid, cfg, return_debug=False).astype(dtype)
                except Exception as exc:
                    failed.append({"sample_index": int(idx), "error": repr(exc)})
                    filtered_mm[idx] = np.nan
            filtered_mm.flush()
            print(f"Processed {end:,}/{n_samples:,}")

        diag_indices = [idx for idx in parse_index_list(cfg.diagnostic_indices) if 0 <= idx < n_samples]
        for idx in diag_indices:
            original = np.asarray(spectra_2d[idx], dtype=np.float64)
            filtered, debug = filter_one_spectrum(original, grid, cfg, return_debug=True)
            png = out_dir / f"diagnostic_sample_{idx:06d}.png"
            try:
                save_diagnostic_plot(png, grid, debug, original, filtered, cfg)
            except Exception as exc:
                print(f"WARNING: diagnostic plot failed for sample {idx}: {exc}")
            if cfg.save_debug_npz:
                save_debug_npz(out_dir / f"debug_sample_{idx:06d}.npz", grid, debug, filtered)

        extra_payload = {}
        for key in [
            "sample_id", "process_id", "nominal_stack_id", "split_label", "cavity_true_um",
            "L_fft_um", "delta_L_nm", "H_peak", "film_nominal_nm", "film_true_nm",
            "film_delta_nm", "layer_names", "valid_mask",
        ]:
            if key in files:
                arr = data[key]
                try:
                    extra_payload[key] = arr[:n_samples] if arr.shape[0] == n_total else arr
                except Exception:
                    extra_payload[key] = arr

        meta_npz = out_dir / f"dataset_vts_like_{cfg.mode}.npz"
        np.savez_compressed(
            meta_npz,
            wavelengths_um=wavelengths_um,
            filtered_spectra_npy_path=str(out_npy),
            source_npz_path=str(input_path),
            source_spectra_key=spectra_key,
            mode=cfg.mode,
            config_json=json.dumps(asdict(cfg), ensure_ascii=False),
            sigma_uniform=grid.sigma_uniform,
            opd_um=grid.opd_um,
            mask=grid.mask,
            opd_positive_max_um=opd_max,
            opd_resolution_um=opd_res,
            failed_count=len(failed),
            **extra_payload,
        )

        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "failed_cases.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "input_npz": str(input_path),
            "output_dir": str(out_dir),
            "wavelength_key": wavelength_key,
            "spectra_key": spectra_key,
            "n_samples": int(n_samples),
            "n_lambda": int(n_lambda),
            "filtered_spectra_npy_path": str(out_npy),
            "metadata_npz_path": str(meta_npz),
            "opd_positive_max_um": opd_max,
            "opd_resolution_um": opd_res,
            "mode": cfg.mode,
            "failed_count": len(failed),
            "diagnostic_indices": diag_indices,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Filtered spectra NPY: {out_npy}")
    print(f"Metadata NPZ: {meta_npz}")
    print(f"Summary: {summary_path}")
    return summary


def main():
    cfg = parse_args()
    process_dataset(cfg)


if __name__ == "__main__":
    main()
