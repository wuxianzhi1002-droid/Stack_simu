"""FFT analysis for first-harmonic lock-in spectrum.

Run from this directory:
    python fft_lockin_1f.py

Optional examples:
    python fft_lockin_1f.py --component R
    python fft_lockin_1f.py --component X
    python fft_lockin_1f.py --component complex --npz dynamic_spectra_20260708_112955.npz

Outputs are written to this same directory:
    lockin_1f_fft_<component>_<timestamp>.png
    lockin_1f_fft_peaks_<component>_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_latest_npz(folder: Path) -> Path:
    files = sorted(folder.glob("dynamic_spectra_*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No dynamic_spectra_*.npz found in {folder}")
    return files[-1]


def load_1f_signal(npz_path: Path, component: str):
    with np.load(npz_path) as z:
        wavelengths_um = z["wavelengths"].astype(float)

        if component == "R":
            signal = z["lockin_1f_R"].astype(float)
            label = "lockin_1f_R"
        elif component == "X":
            signal = z["lockin_1f_X"].astype(float)
            label = "lockin_1f_X"
        elif component == "complex":
            signal = z["lockin_1f_X"].astype(float) + 1j * z["lockin_1f_Y"].astype(float)
            label = "lockin_1f_X + i lockin_1f_Y"
        else:
            raise ValueError(f"Unsupported component: {component}")

    return wavelengths_um, signal, label


def fft_in_linear_k(wavelengths_um: np.ndarray, signal: np.ndarray):
    """FFT along linear k=2*pi/lambda axis.

    If the spectral signal contains cos(k * OPD), the FFT conjugate axis after
    the 2*pi conversion is OPD in um. For an air round trip, gap = OPD / 2.
    """
    beta = 2 * np.pi / wavelengths_um  # rad / um
    order = np.argsort(beta)
    beta_sorted = beta[order]
    signal_sorted = signal[order]

    n = len(signal_sorted)
    beta_linear = np.linspace(beta_sorted[0], beta_sorted[-1], n)

    if np.iscomplexobj(signal_sorted):
        real_interp = np.interp(beta_linear, beta_sorted, signal_sorted.real)
        imag_interp = np.interp(beta_linear, beta_sorted, signal_sorted.imag)
        signal_linear = real_interp + 1j * imag_interp
    else:
        signal_linear = np.interp(beta_linear, beta_sorted, signal_sorted)

    signal_ac = signal_linear - np.mean(signal_linear)
    window = np.hanning(n)
    fft_amp = np.abs(np.fft.rfft(signal_ac * window))

    d_beta = float(np.mean(np.diff(beta_linear)))
    cycles_per_beta = np.fft.rfftfreq(n, d=d_beta)
    opd_um = 2 * np.pi * cycles_per_beta

    return beta_linear, signal_linear, opd_um, fft_amp


def top_peaks(opd_um: np.ndarray, fft_amp: np.ndarray, count: int, min_opd_um: float):
    valid = np.where(opd_um >= min_opd_um)[0]
    valid = valid[valid > 0]
    order = valid[np.argsort(fft_amp[valid])[::-1]]

    peaks = []
    used = []
    resolution = float(np.mean(np.diff(opd_um)))
    min_separation = max(3 * resolution, 0.5)

    for idx in order:
        opd = float(opd_um[idx])
        if any(abs(opd - prev) < min_separation for prev in used):
            continue
        used.append(opd)
        peaks.append({
            "rank": len(peaks) + 1,
            "opd_um": opd,
            "air_gap_um_if_roundtrip": opd / 2.0,
            "amplitude": float(fft_amp[idx]),
        })
        if len(peaks) >= count:
            break

    return peaks


def save_peaks_csv(path: Path, peaks):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "opd_um", "air_gap_um_if_roundtrip", "amplitude"],
        )
        writer.writeheader()
        writer.writerows(peaks)


def main():
    parser = argparse.ArgumentParser(description="FFT first-harmonic lock-in spectrum in k-domain.")
    parser.add_argument("--npz", type=str, default=None, help="Input dynamic_spectra_*.npz. Defaults to latest in this folder.")
    parser.add_argument("--component", choices=["R", "X", "complex"], default="R", help="Which 1f signal to FFT.")
    parser.add_argument("--top", type=int, default=12, help="Number of peak rows to save.")
    parser.add_argument("--min-opd-um", type=float, default=1.0, help="Ignore low-OPD envelope peaks below this value.")
    parser.add_argument("--xmax-opd-um", type=float, default=2500.0, help="Plot x-axis limit for OPD spectrum.")
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening the interactive plot window.")
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent
    npz_path = folder / args.npz if args.npz else find_latest_npz(folder)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    wavelengths_um, signal, signal_label = load_1f_signal(npz_path, args.component)
    beta_linear, signal_linear, opd_um, fft_amp = fft_in_linear_k(wavelengths_um, signal)
    peaks = top_peaks(opd_um, fft_amp, args.top, args.min_opd_um)

    timestamp = npz_path.stem.replace("dynamic_spectra_", "")
    out_png = folder / f"lockin_1f_fft_{args.component}_{timestamp}.png"
    out_csv = folder / f"lockin_1f_fft_peaks_{args.component}_{timestamp}.csv"

    save_peaks_csv(out_csv, peaks)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    if np.iscomplexobj(signal):
        axs[0].plot(wavelengths_um * 1000, signal.real, label="X", linewidth=1.0)
        axs[0].plot(wavelengths_um * 1000, signal.imag, label="Y", linewidth=1.0, alpha=0.8)
        axs[0].legend()
    else:
        axs[0].plot(wavelengths_um * 1000, signal, linewidth=1.0)
    axs[0].set_title(f"First-harmonic signal: {signal_label}")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Lock-in amplitude")
    axs[0].grid(True)

    axs[1].plot(opd_um, fft_amp, linewidth=1.0)
    axs[1].set_title("FFT of 1f lock-in spectrum in linear k-domain")
    axs[1].set_xlabel("OPD (um); air round-trip gap = OPD / 2")
    axs[1].set_ylabel("FFT amplitude")
    axs[1].set_xlim(0, args.xmax_opd_um)
    axs[1].grid(True)

    for peak in peaks[:5]:
        opd = peak["opd_um"]
        amp = peak["amplitude"]
        if opd <= args.xmax_opd_um:
            axs[1].axvline(opd, color="tab:red", alpha=0.25, linewidth=1.0)
            axs[1].annotate(
                f"{opd:.1f} um",
                xy=(opd, amp),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                rotation=30,
            )

    fig.savefig(out_png, dpi=200)

    if args.no_show:
        plt.close(fig)
    else:
        print("Opening interactive Matplotlib window. Close the window to finish the script.")
        plt.show()
        plt.close(fig)

    print(f"Input: {npz_path}")
    print(f"Component: {args.component} ({signal_label})")
    print(f"Saved figure: {out_png}")
    print(f"Saved peaks: {out_csv}")
    print("Top peaks:")
    for peak in peaks:
        print(
            f"#{peak['rank']:02d} OPD={peak['opd_um']:.3f} um, "
            f"air_gap={peak['air_gap_um_if_roundtrip']:.3f} um, "
            f"amp={peak['amplitude']:.6g}"
        )


if __name__ == "__main__":
    main()
