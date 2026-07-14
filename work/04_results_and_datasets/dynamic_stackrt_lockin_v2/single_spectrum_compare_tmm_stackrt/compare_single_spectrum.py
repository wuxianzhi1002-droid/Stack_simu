from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C0_M_S = 299_792_458.0
FREQUENCY_AXIS_C_M_S = 3.0e8

LAYERS = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
FIXED_THICKNESSES_UM = {
    "HSQ": 0.040,
    "PSS": 0.005,
    "SOC": 0.050,
    "TiO2": 0.020,
}


def material_n(name: str, nominal_wavelength_um: np.ndarray) -> np.ndarray:
    w = np.asarray(nominal_wavelength_um, dtype=np.float64)
    if name == "RefReflector":
        return np.full_like(w, 5.8284, dtype=np.complex128)
    if name == "Air":
        return np.full_like(w, 1.0, dtype=np.complex128)
    if name == "HSQ":
        return np.full_like(w, 1.41, dtype=np.complex128)
    if name == "PSS":
        return np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    if name == "SOC":
        return (1.55 + 0.005 / w**2).astype(np.complex128)
    if name == "TiO2":
        return (2.4 + 0.02 / w**2).astype(np.complex128)
    if name == "Cu":
        return np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    raise ValueError(f"Unknown material: {name}")


def stackrt_matched_tmm_rp(
    nominal_wavelength_um: np.ndarray,
    cavity_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal-incidence TMM matched to main_dynamic_v2.py's StackRT inputs.

    main_dynamic_v2.py constructs f = 3e8 / lambda_nominal. StackRT propagates
    at those frequencies using the physical speed of light, so the phase
    wavelength is c0/f rather than lambda_nominal. The material arrays remain
    evaluated on lambda_nominal because that is how n_matrix was constructed.
    """
    nominal_wavelength_um = np.asarray(nominal_wavelength_um, dtype=np.float64)
    frequency_hz = FREQUENCY_AXIS_C_M_S / (nominal_wavelength_um * 1e-6)
    phase_wavelength_um = C0_M_S / frequency_hz * 1e6

    n_matrix = np.vstack([material_n(name, nominal_wavelength_um) for name in LAYERS])
    thickness_um = np.array(
        [
            0.0,
            cavity_um,
            FIXED_THICKNESSES_UM["HSQ"],
            FIXED_THICKNESSES_UM["PSS"],
            FIXED_THICKNESSES_UM["SOC"],
            FIXED_THICKNESSES_UM["TiO2"],
            0.0,
        ],
        dtype=np.float64,
    )

    # At normal incidence, s and p power reflectance are identical and q = n.
    # The -i matrix convention is consistent with StackRT's n + i*k inputs.
    q = n_matrix
    k0 = 2.0 * np.pi / phase_wavelength_um
    m11 = np.ones_like(nominal_wavelength_um, dtype=np.complex128)
    m12 = np.zeros_like(nominal_wavelength_um, dtype=np.complex128)
    m21 = np.zeros_like(nominal_wavelength_um, dtype=np.complex128)
    m22 = np.ones_like(nominal_wavelength_um, dtype=np.complex128)

    for layer_idx in range(1, len(LAYERS) - 1):
        delta = k0 * n_matrix[layer_idx] * thickness_um[layer_idx]
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        q_layer = q[layer_idx]
        a11 = c_delta
        a12 = -1j * s_delta / q_layer
        a21 = -1j * q_layer * s_delta
        a22 = c_delta
        new11 = m11 * a11 + m12 * a21
        new12 = m11 * a12 + m12 * a22
        new21 = m21 * a11 + m22 * a21
        new22 = m21 * a12 + m22 * a22
        m11, m12, m21, m22 = new11, new12, new21, new22

    q0 = q[0]
    qs = q[-1]
    numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
    denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
    reflectance = np.abs(numerator / denominator) ** 2
    return reflectance, frequency_hz, phase_wavelength_um


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_source_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Compare the latest dynamic StackRT t=0 spectrum with a matched TMM spectrum."
    )
    parser.add_argument("--source-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--zoom-center-nm",
        type=float,
        default=500.0,
        help="Center wavelength of the zoom plot in nm (default: 500).",
    )
    parser.add_argument(
        "--zoom-span-nm",
        type=float,
        default=40.0,
        help="Wavelength span of the zoom plot in nm (default: 40).",
    )
    parser.set_defaults(source_dir=default_source_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_npz is None:
        candidates = sorted(args.source_dir.glob("dynamic_spectra_*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No dynamic_spectra_*.npz found in {args.source_dir}")
        source_npz = candidates[-1]
    else:
        source_npz = args.source_npz.resolve()

    with np.load(source_npz) as data:
        nominal_wavelength_um = np.asarray(data["wavelengths"], dtype=np.float64)
        t_axis_s = np.asarray(data["t_axis"], dtype=np.float64)
        cavity_axis_um = np.asarray(data["L_t"], dtype=np.float64)
        spectra = np.asarray(data["spectra"], dtype=np.float64)

    if spectra.ndim != 2 or spectra.shape[0] != t_axis_s.size:
        raise ValueError("Unexpected spectra shape in source NPZ")
    if spectra.shape[1] != nominal_wavelength_um.size:
        raise ValueError("Wavelength and spectrum dimensions do not match")

    initial_index = int(np.argmin(t_axis_s))
    stackrt_rp = spectra[initial_index].copy()
    initial_time_s = float(t_axis_s[initial_index])
    cavity_um = float(cavity_axis_um[initial_index])
    tmm_rp, frequency_hz, phase_wavelength_um = stackrt_matched_tmm_rp(
        nominal_wavelength_um,
        cavity_um,
    )

    residual = stackrt_rp - tmm_rp
    abs_error = np.abs(residual)
    max_index = int(np.argmax(abs_error))

    wavelength_nm = nominal_wavelength_um * 1000.0
    if args.zoom_span_nm <= 0.0:
        raise ValueError("--zoom-span-nm must be positive")
    zoom_half_span_nm = args.zoom_span_nm / 2.0
    zoom_start_nm = max(float(wavelength_nm[0]), args.zoom_center_nm - zoom_half_span_nm)
    zoom_stop_nm = min(float(wavelength_nm[-1]), args.zoom_center_nm + zoom_half_span_nm)
    zoom_mask = (wavelength_nm >= zoom_start_nm) & (wavelength_nm <= zoom_stop_nm)
    if np.count_nonzero(zoom_mask) < 2:
        raise ValueError(
            "The requested zoom range does not overlap the simulated wavelength axis "
            "with at least two samples."
        )

    metrics = {
        "source_npz": source_npz.name,
        "selected_time_index": initial_index,
        "selected_time_s": initial_time_s,
        "air_cavity_um": cavity_um,
        "polarization": "p",
        "incidence_angle_deg": 0.0,
        "wavelength_points": int(nominal_wavelength_um.size),
        "nominal_wavelength_start_nm": float(nominal_wavelength_um[0] * 1000.0),
        "nominal_wavelength_stop_nm": float(nominal_wavelength_um[-1] * 1000.0),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_signed_error": float(np.mean(residual)),
        "max_abs_error": float(abs_error[max_index]),
        "max_abs_error_nominal_wavelength_nm": float(nominal_wavelength_um[max_index] * 1000.0),
        "pearson_correlation": float(np.corrcoef(stackrt_rp, tmm_rp)[0, 1]),
        "stackrt_reflectance_min": float(np.min(stackrt_rp)),
        "stackrt_reflectance_max": float(np.max(stackrt_rp)),
        "tmm_reflectance_min": float(np.min(tmm_rp)),
        "tmm_reflectance_max": float(np.max(tmm_rp)),
        "zoom_center_nm": float(args.zoom_center_nm),
        "zoom_requested_span_nm": float(args.zoom_span_nm),
        "zoom_start_nm": float(zoom_start_nm),
        "zoom_stop_nm": float(zoom_stop_nm),
        "zoom_points": int(np.count_nonzero(zoom_mask)),
        "frequency_axis_definition": "f = 3e8 / nominal_wavelength",
        "phase_wavelength_definition": "lambda_phase = 299792458 / f",
        "complex_index_convention": "n + i*k with -i characteristic-matrix off-diagonal terms",
    }

    csv_path = output_dir / "single_spectrum_compare_tmm_stackrt.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nominal_wavelength_nm",
                "stackrt_phase_wavelength_nm",
                "frequency_hz",
                "stackrt_Rp_t0",
                "tmm_Rp",
                "residual_stackrt_minus_tmm",
                "abs_error",
            ]
        )
        writer.writerows(
            zip(
                nominal_wavelength_um * 1000.0,
                phase_wavelength_um * 1000.0,
                frequency_hz,
                stackrt_rp,
                tmm_rp,
                residual,
                abs_error,
            )
        )

    np.savez_compressed(
        output_dir / "single_spectrum_compare_tmm_stackrt.npz",
        nominal_wavelength_um=nominal_wavelength_um,
        stackrt_phase_wavelength_um=phase_wavelength_um,
        frequency_hz=frequency_hz,
        stackrt_Rp_t0=stackrt_rp,
        tmm_Rp=tmm_rp,
        residual_stackrt_minus_tmm=residual,
        abs_error=abs_error,
        selected_time_index=np.array(initial_index),
        selected_time_s=np.array(initial_time_s),
        air_cavity_um=np.array(cavity_um),
        source_npz=np.array(source_npz.name),
    )

    with (output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        constrained_layout=True,
    )
    ax_top.plot(wavelength_nm, stackrt_rp, color="#16697A", lw=1.0, label="StackRT NPZ, t = 0")
    ax_top.plot(wavelength_nm, tmm_rp, color="#D1495B", lw=0.8, ls="--", label="Matched TMM")
    ax_top.set_ylabel("Reflectance Rp")
    ax_top.set_title("Single-spectrum comparison: StackRT vs TMM")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right")
    ax_top.text(
        0.01,
        0.02,
        f"Air cavity = {cavity_um:.6f} um | MAE = {metrics['mae']:.3e} | RMSE = {metrics['rmse']:.3e}",
        transform=ax_top.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    ax_bottom.plot(wavelength_nm, residual, color="#6A4C93", lw=0.8)
    ax_bottom.axhline(0.0, color="black", lw=0.7, alpha=0.7)
    ax_bottom.set_xlabel("Nominal wavelength (nm)")
    ax_bottom.set_ylabel("StackRT - TMM")
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(output_dir / "single_spectrum_compare_tmm_stackrt.png", dpi=220)
    plt.close(fig)

    zoom_wavelength_nm = wavelength_nm[zoom_mask]
    zoom_stackrt_rp = stackrt_rp[zoom_mask]
    zoom_tmm_rp = tmm_rp[zoom_mask]
    zoom_residual = residual[zoom_mask]

    detail_half_span_nm = min(0.5, (zoom_stop_nm - zoom_start_nm) / 2.0)
    detail_start_nm = args.zoom_center_nm - detail_half_span_nm
    detail_stop_nm = args.zoom_center_nm + detail_half_span_nm
    detail_mask = zoom_mask & (wavelength_nm >= detail_start_nm) & (wavelength_nm <= detail_stop_nm)

    zoom_fig, (zoom_ax, detail_ax, zoom_residual_ax) = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2.8, 2.2, 1.2]},
        constrained_layout=True,
    )
    zoom_ax.plot(
        zoom_wavelength_nm,
        zoom_stackrt_rp,
        color="#16697A",
        lw=1.0,
        label="StackRT NPZ, t = 0",
    )
    zoom_ax.plot(
        zoom_wavelength_nm,
        zoom_tmm_rp,
        color="#D1495B",
        lw=0.8,
        ls="--",
        label="Matched TMM",
    )
    zoom_ax.set_xlim(zoom_start_nm, zoom_stop_nm)
    zoom_ax.set_ylabel("Reflectance Rp")
    zoom_ax.set_title(
        f"Zoomed comparison: {zoom_start_nm:.1f}-{zoom_stop_nm:.1f} nm "
        f"({zoom_stop_nm - zoom_start_nm:.1f} nm span)"
    )
    zoom_ax.grid(True, alpha=0.25)
    zoom_ax.legend(loc="upper right")

    if np.count_nonzero(detail_mask) >= 2:
        detail_ax.plot(
            wavelength_nm[detail_mask],
            stackrt_rp[detail_mask],
            color="#16697A",
            lw=1.4,
            marker="o",
            ms=2.2,
            label="StackRT NPZ, t = 0",
        )
        detail_ax.plot(
            wavelength_nm[detail_mask],
            tmm_rp[detail_mask],
            color="#D1495B",
            lw=1.1,
            ls="--",
            label="Matched TMM",
        )
        detail_ax.set_xlim(detail_start_nm, detail_stop_nm)
    detail_ax.set_ylabel("Reflectance Rp")
    detail_ax.set_title(
        f"Central 1 nm detail: {detail_start_nm:.1f}-{detail_stop_nm:.1f} nm"
    )
    detail_ax.grid(True, alpha=0.25)

    zoom_residual_ax.plot(zoom_wavelength_nm, zoom_residual, color="#6A4C93", lw=0.8)
    zoom_residual_ax.axhline(0.0, color="black", lw=0.7, alpha=0.7)
    zoom_residual_ax.set_xlim(zoom_start_nm, zoom_stop_nm)
    zoom_residual_ax.set_xlabel("Nominal wavelength (nm)")
    zoom_residual_ax.set_ylabel("StackRT - TMM")
    zoom_residual_ax.grid(True, alpha=0.25)
    zoom_residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    zoom_fig.savefig(
        output_dir / "single_spectrum_compare_tmm_stackrt_zoom.png",
        dpi=220,
    )
    plt.close(zoom_fig)

    readme = f"""# Single spectrum comparison: TMM vs StackRT

## Selection

- Source NPZ: `{source_npz.name}`
- Time index: `{initial_index}`
- Time: `{initial_time_s:.9g} s`
- Air cavity: `{cavity_um:.9g} um`
- Spectrum: p polarization (`Rp`), normal incidence
- Stack: `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`
- Thicknesses: Air `{cavity_um:.9g} um`, HSQ `40 nm`, PSS `5 nm`, SOC `50 nm`, TiO2 `20 nm`
- First and last layers are treated as semi-infinite media.

## Material model

The TMM uses exactly the refractive-index arrays defined in `main_dynamic_v2.py`:

- RefReflector: `5.8284`
- Air: `1.0`
- HSQ: `1.41`
- PSS: `1.50 + 0.05j`
- SOC: `1.55 + 0.005 / wavelength_um^2`
- TiO2: `2.4 + 0.02 / wavelength_um^2`
- Cu: `1.1 + 2.5j`

## Frequency and sign convention

`main_dynamic_v2.py` constructs the StackRT frequency vector as
`f = 3e8 / nominal_wavelength`. The matched TMM therefore uses the same
frequency and computes propagation phase with
`phase_wavelength = 299792458 / f`. Material arrays remain evaluated at the
nominal wavelength, matching the original construction of `n_matrix`.

For complex index `n + i*k`, the characteristic matrices use `-i` in the
off-diagonal terms. This convention reproduces StackRT attenuation and phase.
Using the opposite sign or directly using nominal wavelength for propagation
produces a large artificial mismatch for the 1 mm cavity.

## Metrics

- MAE: `{metrics['mae']:.12g}`
- RMSE: `{metrics['rmse']:.12g}`
- Maximum absolute error: `{metrics['max_abs_error']:.12g}` at nominal wavelength `{metrics['max_abs_error_nominal_wavelength_nm']:.9g} nm`
- Pearson correlation: `{metrics['pearson_correlation']:.12g}`

## Zoom plot

- Center wavelength: `{args.zoom_center_nm:.6g} nm`
- Requested span: `{args.zoom_span_nm:.6g} nm`
- Actual plotted range: `{zoom_start_nm:.6g}-{zoom_stop_nm:.6g} nm`
- The zoom figure also includes a central 1 nm detail panel so individual fringes remain visible for the 1 mm cavity.

## Files

- `single_spectrum_compare_tmm_stackrt.png`: full-range overlay and residual plot
- `single_spectrum_compare_tmm_stackrt_zoom.png`: configurable zoom, central 1 nm detail, and zoom residual
- `single_spectrum_compare_tmm_stackrt.csv`: per-wavelength comparison table
- `single_spectrum_compare_tmm_stackrt.npz`: compact numerical arrays and metadata
- `comparison_metrics.json`: parameters and scalar metrics
- `compare_single_spectrum.py`: reproducible comparison script
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
