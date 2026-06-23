from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grazing_config import (
    CONFIG,
    DATA_DIR,
    IMG_DIR,
    config_json,
    ensure_output_dirs,
    latest_npz,
    timestamp,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(values: Iterable[Any], **_: Any) -> Iterable[Any]:
        return values


def _json_from_npz(value: Any) -> Dict[str, Any]:
    """从 npz 里的 0 维字符串恢复配置。"""
    text = str(np.asarray(value).item())
    return json.loads(text)


def _as_text(value: Any) -> str:
    """把 npz 里的 0 维字符串转成普通 str。"""
    return str(np.asarray(value).item())


def _finite_clim(values: np.ndarray) -> Tuple[float, float] | None:
    """为热力图生成稳健色阶，避免极端点把图压扁。"""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if np.isclose(lo, hi):
        return None
    return float(lo), float(hi)


def characteristic_admittance(n_values: np.ndarray, cos_values: np.ndarray, pol: str) -> np.ndarray:
    """返回 s/p 偏振的等效光学导纳。"""
    if pol.lower() == "s":
        return n_values * cos_values
    if pol.lower() == "p":
        return cos_values / n_values
    raise ValueError(f"Unsupported polarization: {pol}")


def tmm_reflection_grid(
    n_matrix: np.ndarray,
    thicknesses_m: np.ndarray,
    wavelengths_m: np.ndarray,
    theta_axis_rad: np.ndarray,
    pol: str,
) -> np.ndarray:
    """用特征矩阵法计算任意多层膜的复反射系数。

    第一层和最后一层按半无限入射介质/衬底处理，中间层使用有限厚度。
    """
    n_matrix = np.asarray(n_matrix, dtype=np.complex128)
    thicknesses_m = np.asarray(thicknesses_m, dtype=np.float64)
    wavelengths_m = np.asarray(wavelengths_m, dtype=np.float64)
    theta_axis_rad = np.asarray(theta_axis_rad, dtype=np.float64)

    n_layer, n_lambda = n_matrix.shape
    result = np.full((theta_axis_rad.size, n_lambda), np.nan + 0j, dtype=np.complex64)
    k0 = 2.0 * np.pi / wavelengths_m

    for i_theta, theta in enumerate(theta_axis_rad):
        kx = n_matrix[0, :] * np.sin(theta)
        cos_values = np.sqrt(1.0 - (kx[None, :] / n_matrix) ** 2 + 0j)
        cos_values = np.where(np.real(cos_values) < 0.0, -cos_values, cos_values)
        cos_values = np.where(
            (np.abs(np.real(cos_values)) < 1e-12) & (np.imag(cos_values) < 0.0),
            -cos_values,
            cos_values,
        )

        q_values = characteristic_admittance(n_matrix, cos_values, pol)
        m11 = np.ones(n_lambda, dtype=np.complex128)
        m12 = np.zeros(n_lambda, dtype=np.complex128)
        m21 = np.zeros(n_lambda, dtype=np.complex128)
        m22 = np.ones(n_lambda, dtype=np.complex128)

        for layer_idx in range(1, n_layer - 1):
            thickness = float(thicknesses_m[layer_idx])
            if thickness <= 0.0:
                continue
            delta = k0 * n_matrix[layer_idx, :] * thickness * cos_values[layer_idx, :]
            c_delta = np.cos(delta)
            s_delta = np.sin(delta)
            q_layer = q_values[layer_idx, :]
            a11 = c_delta
            a12 = 1j * s_delta / q_layer
            a21 = 1j * q_layer * s_delta
            a22 = c_delta

            new11 = m11 * a11 + m12 * a21
            new12 = m11 * a12 + m12 * a22
            new21 = m21 * a11 + m22 * a21
            new22 = m21 * a12 + m22 * a22
            m11, m12, m21, m22 = new11, new12, new21, new22

        q0 = q_values[0, :]
        qs = q_values[-1, :]
        numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
        denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
        result[i_theta, :] = (numerator / denominator).astype(np.complex64)

    return result


def asd_from_complex_r(
    r_values: np.ndarray,
    wavelengths_m: np.ndarray,
    theta_axis_rad: np.ndarray,
) -> np.ndarray:
    """由复反射系数相位计算 ASD，返回单位 m。"""
    r_values = np.asarray(r_values)
    theta_axis_rad = np.asarray(theta_axis_rad, dtype=np.float64)
    wavelengths_m = np.asarray(wavelengths_m, dtype=np.float64)
    theta_axis_index = r_values.ndim - 2

    psi = -np.unwrap(np.angle(r_values), axis=theta_axis_index)
    edge_order = 2 if theta_axis_rad.size >= 3 else 1
    dpsi_dtheta = np.gradient(psi, theta_axis_rad, axis=theta_axis_index, edge_order=edge_order)

    factor = wavelengths_m / (4.0 * np.pi)
    sin_theta = np.sin(theta_axis_rad)
    if r_values.ndim == 2:
        scale = factor[None, :] / sin_theta[:, None]
    elif r_values.ndim == 3:
        scale = factor[None, None, :] / sin_theta[None, :, None]
    else:
        raise ValueError(f"Unsupported r_values ndim: {r_values.ndim}")
    return dpsi_dtheta * scale


def broadband_average(asd_nm: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """计算 flat source 的反射率加权宽带 ASD。"""
    weights = np.ones(reflectance.shape[-1], dtype=np.float64)
    numerator = np.nansum(asd_nm * reflectance * weights, axis=-1)
    denominator = np.nansum(reflectance * weights, axis=-1)
    return numerator / np.where(np.abs(denominator) > 1e-15, denominator, np.nan)


def recommend_theta_table(
    theta_axis_deg: np.ndarray,
    asd_nominal_p_nm: np.ndarray,
    asd_nominal_s_nm: np.ndarray,
    asd_error_mc_p_nm: np.ndarray,
    asd_error_mc_s_nm: np.ndarray,
    r_mean_p: np.ndarray,
    r_mean_s: np.ndarray,
) -> List[Dict[str, float | str]]:
    """基于 ASD 稳定性、反射率和几何灵敏度给出推荐角度表。"""
    rows: List[Dict[str, float | str]] = []
    geometry = 2.0 * np.sin(np.deg2rad(theta_axis_deg))

    for pol, asd_nominal, asd_error, r_mean in (
        ("p", asd_nominal_p_nm, asd_error_mc_p_nm, r_mean_p),
        ("s", asd_nominal_s_nm, asd_error_mc_s_nm, r_mean_s),
    ):
        if asd_error.size:
            mc_std = np.nanstd(asd_error, axis=0)
            mc_p95 = np.nanpercentile(np.abs(asd_error), 95.0, axis=0)
            mc_max = np.nanmax(np.abs(asd_error), axis=0)
            mc_mean = np.nanmean(asd_error, axis=0)
        else:
            mc_std = np.zeros_like(theta_axis_deg)
            mc_p95 = np.zeros_like(theta_axis_deg)
            mc_max = np.zeros_like(theta_axis_deg)
            mc_mean = np.zeros_like(theta_axis_deg)

        wl_std = np.nanstd(asd_nominal, axis=1)
        std_norm = (mc_std - np.nanmin(mc_std)) / (np.nanmax(mc_std) - np.nanmin(mc_std) + 1e-12)
        r_norm = (r_mean - np.nanmin(r_mean)) / (np.nanmax(r_mean) - np.nanmin(r_mean) + 1e-12)
        g_norm = (geometry - np.nanmin(geometry)) / (np.nanmax(geometry) - np.nanmin(geometry) + 1e-12)
        score = 0.65 * std_norm + 0.20 * (1.0 - r_norm) + 0.15 * (1.0 - g_norm)

        for idx, theta in enumerate(theta_axis_deg):
            rows.append(
                {
                    "pol": pol,
                    "theta_deg": float(theta),
                    "ASD_mean_nm": float(np.nanmean(asd_nominal[idx, :])),
                    "ASD_wavelength_std_nm": float(wl_std[idx]),
                    "ASD_mc_bias_mean_nm": float(mc_mean[idx]),
                    "ASD_mc_bias_std_nm": float(mc_std[idx]),
                    "ASD_mc_bias_p95_nm": float(mc_p95[idx]),
                    "ASD_mc_bias_max_abs_nm": float(mc_max[idx]),
                    "R_mean": float(r_mean[idx]),
                    "geometric_sensitivity": float(geometry[idx]),
                    "score": float(score[idx]),
                }
            )

    rows.sort(key=lambda item: float(item["score"]))
    return rows


def _plot_asd_heatmap(
    values_nm: np.ndarray,
    wavelengths_m: np.ndarray,
    theta_axis_deg: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """保存 ASD(lambda, theta) 热力图。"""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    mesh = ax.pcolormesh(wavelengths_m * 1e9, theta_axis_deg, values_nm, shading="auto", cmap="coolwarm")
    clim = _finite_clim(values_nm)
    if clim is not None:
        mesh.set_clim(*clim)
    fig.colorbar(mesh, ax=ax, label="ASD (nm)")
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Incident angle from surface normal (deg)")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_lines(theta_axis_deg: np.ndarray, series: Dict[str, np.ndarray], title: str, ylabel: str, save_path: Path) -> None:
    """保存随 theta 变化的折线图。"""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, values in series.items():
        ax.plot(theta_axis_deg, values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Incident angle from surface normal (deg)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_tmm_compare(
    theta_axis_deg: np.ndarray,
    stackrt_p: np.ndarray,
    stackrt_s: np.ndarray,
    tmm_p: np.ndarray,
    tmm_s: np.ndarray,
    save_path: Path,
) -> None:
    """保存 TMM 与 StackRT 的反射率对比图。"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axs[0].plot(theta_axis_deg, np.nanmean(stackrt_p, axis=1), label="StackRT p")
    axs[0].plot(theta_axis_deg, np.nanmean(tmm_p, axis=1), "--", label="TMM p")
    axs[0].plot(theta_axis_deg, np.nanmean(stackrt_s, axis=1), label="StackRT s")
    axs[0].plot(theta_axis_deg, np.nanmean(tmm_s, axis=1), "--", label="TMM s")
    axs[0].set_title("Mean Reflectance Compare")
    axs[0].set_xlabel("Incident angle from surface normal (deg)")
    axs[0].set_ylabel("Mean reflectance")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(theta_axis_deg, np.nanmean(np.abs(stackrt_p - tmm_p), axis=1), label="|p diff|")
    axs[1].plot(theta_axis_deg, np.nanmean(np.abs(stackrt_s - tmm_s), axis=1), label="|s diff|")
    axs[1].set_title("Mean Absolute Difference")
    axs[1].set_xlabel("Incident angle from surface normal (deg)")
    axs[1].set_ylabel("Mean abs diff")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def solve_asd(input_path: Path | None = None, config: Dict[str, Any] | None = None) -> Path:
    """读取 grazing_stackrt npz，计算 ASD 并保存结果。"""
    cfg = CONFIG if config is None else config
    ensure_output_dirs()
    input_path = latest_npz("grazing_stackrt_*.npz") if input_path is None else Path(input_path)

    with np.load(input_path) as data:
        wavelengths_m = np.asarray(data["wavelengths_m"], dtype=np.float64)
        theta_axis_deg = np.asarray(data["theta_axis_deg"], dtype=np.float64)
        theta_axis_rad = np.asarray(data["theta_axis_rad"], dtype=np.float64)
        R_nominal_p = np.asarray(data["R_nominal_p"], dtype=np.float64)
        R_nominal_s = np.asarray(data["R_nominal_s"], dtype=np.float64)
        R_mc_p = np.asarray(data["R_mc_p"], dtype=np.float32)
        R_mc_s = np.asarray(data["R_mc_s"], dtype=np.float32)
        thicknesses_nominal_m = np.asarray(data["thicknesses_nominal_m"], dtype=np.float64)
        thicknesses_mc_m = np.asarray(data["thicknesses_mc_m"], dtype=np.float64)
        n_matrix = np.asarray(data["n_matrix"], dtype=np.complex128)
        layer_names = np.asarray(data["layer_names"], dtype=str)
        stack_config = _json_from_npz(data["config_json"])
        stackrt_result_keys = np.asarray(data["stackrt_result_keys"], dtype=str)
        model_source = _as_text(data["model_source"])

        has_r_p = "r_nominal_p" in data.files and "r_mc_p" in data.files
        has_r_s = "r_nominal_s" in data.files and "r_mc_s" in data.files
        if has_r_p and has_r_s:
            phase_source = "stackrt_complex_r"
            r_nominal_p = np.asarray(data["r_nominal_p"], dtype=np.complex64)
            r_nominal_s = np.asarray(data["r_nominal_s"], dtype=np.complex64)
            r_mc_p_data = np.asarray(data["r_mc_p"], dtype=np.complex64)
            r_mc_s_data = np.asarray(data["r_mc_s"], dtype=np.complex64)
        else:
            phase_source = "internal_tmm"
            r_nominal_p = tmm_reflection_grid(n_matrix, thicknesses_nominal_m, wavelengths_m, theta_axis_rad, "p")
            r_nominal_s = tmm_reflection_grid(n_matrix, thicknesses_nominal_m, wavelengths_m, theta_axis_rad, "s")
            r_mc_p_data = None
            r_mc_s_data = None

    print("=== Grazing ASD Solver ===")
    print(f"Input: {input_path}")
    print(f"Phase source: {phase_source}")

    ASD_nominal_p_m = asd_from_complex_r(r_nominal_p, wavelengths_m, theta_axis_rad)
    ASD_nominal_s_m = asd_from_complex_r(r_nominal_s, wavelengths_m, theta_axis_rad)
    ASD_nominal_p_nm = ASD_nominal_p_m.astype(np.float64) * 1e9
    ASD_nominal_s_nm = ASD_nominal_s_m.astype(np.float64) * 1e9

    ASD_bb_nominal_p_nm = broadband_average(ASD_nominal_p_nm, R_nominal_p)
    ASD_bb_nominal_s_nm = broadband_average(ASD_nominal_s_nm, R_nominal_s)

    n_mc = thicknesses_mc_m.shape[0]
    ASD_mc_p_nm = np.full(R_mc_p.shape, np.nan, dtype=np.float32)
    ASD_mc_s_nm = np.full(R_mc_s.shape, np.nan, dtype=np.float32)
    ASD_bb_mc_p_nm = np.full((n_mc, theta_axis_deg.size), np.nan, dtype=np.float32)
    ASD_bb_mc_s_nm = np.full_like(ASD_bb_mc_p_nm, np.nan)

    if phase_source == "stackrt_complex_r":
        for i_mc in tqdm(range(n_mc), desc="ASD from StackRT complex r"):
            asd_p = asd_from_complex_r(r_mc_p_data[i_mc, :, :], wavelengths_m, theta_axis_rad) * 1e9
            asd_s = asd_from_complex_r(r_mc_s_data[i_mc, :, :], wavelengths_m, theta_axis_rad) * 1e9
            ASD_mc_p_nm[i_mc, :, :] = asd_p.astype(np.float32)
            ASD_mc_s_nm[i_mc, :, :] = asd_s.astype(np.float32)
            ASD_bb_mc_p_nm[i_mc, :] = broadband_average(asd_p, R_mc_p[i_mc, :, :]).astype(np.float32)
            ASD_bb_mc_s_nm[i_mc, :] = broadband_average(asd_s, R_mc_s[i_mc, :, :]).astype(np.float32)
    else:
        for i_mc in tqdm(range(n_mc), desc="ASD from internal TMM"):
            r_p = tmm_reflection_grid(n_matrix, thicknesses_mc_m[i_mc, :], wavelengths_m, theta_axis_rad, "p")
            r_s = tmm_reflection_grid(n_matrix, thicknesses_mc_m[i_mc, :], wavelengths_m, theta_axis_rad, "s")
            asd_p = asd_from_complex_r(r_p, wavelengths_m, theta_axis_rad) * 1e9
            asd_s = asd_from_complex_r(r_s, wavelengths_m, theta_axis_rad) * 1e9
            ASD_mc_p_nm[i_mc, :, :] = asd_p.astype(np.float32)
            ASD_mc_s_nm[i_mc, :, :] = asd_s.astype(np.float32)
            ASD_bb_mc_p_nm[i_mc, :] = broadband_average(asd_p, R_mc_p[i_mc, :, :]).astype(np.float32)
            ASD_bb_mc_s_nm[i_mc, :] = broadband_average(asd_s, R_mc_s[i_mc, :, :]).astype(np.float32)

    ASD_error_mc_p_nm = ASD_bb_mc_p_nm - ASD_bb_nominal_p_nm[None, :]
    ASD_error_mc_s_nm = ASD_bb_mc_s_nm - ASD_bb_nominal_s_nm[None, :]
    R_mean_p = np.nanmean(R_nominal_p, axis=1)
    R_mean_s = np.nanmean(R_nominal_s, axis=1)

    table = recommend_theta_table(
        theta_axis_deg,
        ASD_nominal_p_nm,
        ASD_nominal_s_nm,
        ASD_error_mc_p_nm,
        ASD_error_mc_s_nm,
        R_mean_p,
        R_mean_s,
    )
    best = table[0]

    tmm_R_nominal_p = np.abs(r_nominal_p) ** 2
    tmm_R_nominal_s = np.abs(r_nominal_s) ** 2
    tmm_diff_max_p = float(np.nanmax(np.abs(tmm_R_nominal_p - R_nominal_p)))
    tmm_diff_max_s = float(np.nanmax(np.abs(tmm_R_nominal_s - R_nominal_s)))

    _plot_asd_heatmap(
        ASD_nominal_p_nm,
        wavelengths_m,
        theta_axis_deg,
        "ASD vs Wavelength and Angle (p)",
        IMG_DIR / "ASD_vs_wavelength_theta_p.png",
    )
    _plot_asd_heatmap(
        ASD_nominal_s_nm,
        wavelengths_m,
        theta_axis_deg,
        "ASD vs Wavelength and Angle (s)",
        IMG_DIR / "ASD_vs_wavelength_theta_s.png",
    )
    _plot_lines(
        theta_axis_deg,
        {
            "p nominal": ASD_bb_nominal_p_nm,
            "s nominal": ASD_bb_nominal_s_nm,
            "p MC mean": np.nanmean(ASD_bb_mc_p_nm, axis=0) if n_mc else ASD_bb_nominal_p_nm,
            "s MC mean": np.nanmean(ASD_bb_mc_s_nm, axis=0) if n_mc else ASD_bb_nominal_s_nm,
        },
        "Broadband ASD vs Angle",
        "Broadband ASD (nm)",
        IMG_DIR / "ASD_bb_vs_theta.png",
    )
    _plot_lines(
        theta_axis_deg,
        {
            "p MC std": np.nanstd(ASD_error_mc_p_nm, axis=0) if n_mc else np.zeros_like(theta_axis_deg),
            "s MC std": np.nanstd(ASD_error_mc_s_nm, axis=0) if n_mc else np.zeros_like(theta_axis_deg),
        },
        "ASD Monte Carlo Std vs Angle",
        "ASD bias std (nm)",
        IMG_DIR / "ASD_mc_std_vs_theta.png",
    )
    _plot_lines(
        theta_axis_deg,
        {"p": R_mean_p, "s": R_mean_s},
        "Mean Reflectance vs Angle",
        "Mean reflectance",
        IMG_DIR / "reflectance_vs_theta.png",
    )
    if phase_source == "internal_tmm":
        _plot_tmm_compare(
            theta_axis_deg,
            R_nominal_p,
            R_nominal_s,
            tmm_R_nominal_p,
            tmm_R_nominal_s,
            IMG_DIR / "tmm_stackrt_reflectance_compare.png",
        )

    output_path = DATA_DIR / f"grazing_asd_{timestamp()}.npz"
    np.savez_compressed(
        output_path,
        input_stackrt_npz=np.asarray(str(input_path)),
        wavelengths_m=wavelengths_m,
        theta_axis_deg=theta_axis_deg,
        theta_axis_rad=theta_axis_rad,
        layer_names=layer_names,
        thicknesses_nominal_m=thicknesses_nominal_m,
        ASD_nominal_p_m=ASD_nominal_p_m.astype(np.float32),
        ASD_nominal_s_m=ASD_nominal_s_m.astype(np.float32),
        ASD_nominal_p_nm=ASD_nominal_p_nm.astype(np.float32),
        ASD_nominal_s_nm=ASD_nominal_s_nm.astype(np.float32),
        ASD_mc_p_nm=ASD_mc_p_nm,
        ASD_mc_s_nm=ASD_mc_s_nm,
        ASD_bb_nominal_p_nm=ASD_bb_nominal_p_nm.astype(np.float32),
        ASD_bb_nominal_s_nm=ASD_bb_nominal_s_nm.astype(np.float32),
        ASD_bb_mc_p_nm=ASD_bb_mc_p_nm,
        ASD_bb_mc_s_nm=ASD_bb_mc_s_nm,
        ASD_error_mc_p_nm=ASD_error_mc_p_nm.astype(np.float32),
        ASD_error_mc_s_nm=ASD_error_mc_s_nm.astype(np.float32),
        R_mean_p=R_mean_p.astype(np.float32),
        R_mean_s=R_mean_s.astype(np.float32),
        recommended_theta_table=np.asarray(json.dumps(table, ensure_ascii=False, indent=2)),
        best_theta_deg=np.asarray(float(best["theta_deg"])),
        best_pol=np.asarray(str(best["pol"])),
        phase_source=np.asarray(phase_source),
        tmm_R_nominal_p=tmm_R_nominal_p.astype(np.float32),
        tmm_R_nominal_s=tmm_R_nominal_s.astype(np.float32),
        tmm_stackrt_diff_max_p=np.asarray(tmm_diff_max_p),
        tmm_stackrt_diff_max_s=np.asarray(tmm_diff_max_s),
        stackrt_result_keys=stackrt_result_keys,
        model_source=np.asarray(model_source),
        stack_config_json=np.asarray(json.dumps(stack_config, ensure_ascii=False, indent=2)),
        config_json=np.asarray(config_json(cfg)),
    )
    print(f"Saved ASD data: {output_path}")
    print(f"Best theta/pol by score: {best['theta_deg']:.3f} deg, {best['pol']}")
    if phase_source == "internal_tmm" and max(tmm_diff_max_p, tmm_diff_max_s) > 0.05:
        print(
            "Warning: TMM and StackRT reflectance differ by more than 0.05. "
            "Check layer order, angle convention, or StackRT material conventions."
        )
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Solve grazing-incidence ASD from StackRT npz.")
    parser.add_argument("--input", type=Path, default=None, help="Path to grazing_stackrt_*.npz")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    solve_asd(args.input)


if __name__ == "__main__":
    main()
