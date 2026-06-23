from __future__ import annotations

import argparse
import csv
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
    LINEAR_FIT_DIR,
    build_height_axis,
    config_json,
    ensure_output_dirs,
    latest_npz,
    timestamp,
)


def _npz_text(value: Any) -> str:
    """读取 npz 中保存的 0 维字符串。"""
    return str(np.asarray(value).item())


def _load_recommendation(value: Any) -> List[Dict[str, Any]]:
    """读取 ASD 脚本写入的推荐 theta 表。"""
    return json.loads(_npz_text(value))


def select_polarization(asd_data: Dict[str, Any]) -> str:
    """选择用于三角测量误差评估的偏振。"""
    if "best_pol" in asd_data:
        return _npz_text(asd_data["best_pol"])
    table = _load_recommendation(asd_data["recommended_theta_table"])
    return str(table[0]["pol"])


def summarize_asd_bias(asd_data: Dict[str, Any], pol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每个 theta 下的 ASD 偏置均值、标准差和 RMS。"""
    key = f"ASD_error_mc_{pol}_nm"
    error_mc = np.asarray(asd_data[key], dtype=np.float64)
    if error_mc.size == 0:
        n_theta = np.asarray(asd_data["theta_axis_deg"]).size
        zeros = np.zeros(n_theta, dtype=float)
        return zeros, zeros, zeros
    bias_mean = np.nanmean(error_mc, axis=0)
    bias_std = np.nanstd(error_mc, axis=0)
    bias_rms = np.sqrt(np.nanmean(error_mc**2, axis=0))
    return bias_mean, bias_std, bias_rms


def phase_noise_to_height_std_nm(
    detector_noise_std: float,
    optical_amplitude: np.ndarray,
    grating_pitch_nm: float,
    geometric_gain: np.ndarray,
) -> np.ndarray:
    """四相读数的小噪声近似：相位噪声换算为高度噪声。"""
    sigma_phi = detector_noise_std / (np.sqrt(2.0) * np.maximum(optical_amplitude, 1e-15))
    return sigma_phi * grating_pitch_nm / (2.0 * np.pi * np.maximum(geometric_gain, 1e-15))


def simulate_four_phase_readout(
    height_true_nm: np.ndarray,
    theta_rad: float,
    multipass_n: int,
    asd_bias_mean_nm: float,
    optical_amplitude: float,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """按四相读数模型模拟一次高度恢复曲线。"""
    pitch_nm = float(config["grating_pitch_um"]) * 1000.0
    magnification = float(config["imaging_magnification"])
    detector_noise_std = float(config["detector_noise_std"])
    phi0 = float(config.get("phase_offset_rad", 0.0))
    gain = 2.0 * float(multipass_n) * magnification * np.sin(theta_rad)

    z_meas_nm = height_true_nm + asd_bias_mean_nm
    displacement_nm = gain * z_meas_nm
    phi_g = 2.0 * np.pi * displacement_nm / pitch_nm + phi0

    i_bg = 1.0
    i_amp = max(float(optical_amplitude), 1e-15)
    noise = lambda: rng.normal(0.0, detector_noise_std, size=height_true_nm.shape)
    i0 = i_bg + i_amp * np.cos(phi_g) + noise()
    i90 = i_bg + i_amp * np.sin(phi_g) + noise()
    i180 = i_bg - i_amp * np.cos(phi_g) + noise()
    i270 = i_bg - i_amp * np.sin(phi_g) + noise()

    phi_rec = np.unwrap(np.arctan2(i90 - i270, i0 - i180))
    z_rec_nm = (phi_rec - phi0) * pitch_nm / (2.0 * np.pi * gain)
    return z_rec_nm, z_rec_nm - height_true_nm


def _save_summary_csv(path: Path, rows: List[Dict[str, float | int]]) -> None:
    """保存 theta-N 扫描汇总表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "theta_deg",
        "multipass_N",
        "geometric_gain",
        "optical_power_ratio",
        "random_noise_std_nm",
        "asd_bias_std_nm",
        "asd_bias_rms_nm",
        "total_rms_nm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_best_height_curves(
    height_true_nm: np.ndarray,
    z_rec_nm: np.ndarray,
    z_error_nm: np.ndarray,
    theta_axis_deg: np.ndarray,
    multipass_list: np.ndarray,
    theta_idx: int,
) -> None:
    """保存最佳 theta 下不同多通数的高度恢复曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for n_idx, multipass_n in enumerate(multipass_list):
        ax.plot(height_true_nm, z_rec_nm[theta_idx, n_idx, :], label=f"N={multipass_n}")
    ax.plot(height_true_nm, height_true_nm, "k--", linewidth=1.0, label="ideal")
    ax.set_title(f"Height Reconstruction at {theta_axis_deg[theta_idx]:.2f} deg")
    ax.set_xlabel("True height (nm)")
    ax.set_ylabel("Recovered height (nm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(IMG_DIR / "height_reconstruction_N_compare.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for n_idx, multipass_n in enumerate(multipass_list):
        ax.plot(height_true_nm, z_error_nm[theta_idx, n_idx, :], label=f"N={multipass_n}")
    ax.set_title(f"Height Error at {theta_axis_deg[theta_idx]:.2f} deg")
    ax.set_xlabel("True height (nm)")
    ax.set_ylabel("Height error (nm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(IMG_DIR / "error_vs_height_N_compare.png", dpi=200)
    plt.close(fig)


def _plot_tradeoffs(
    theta_axis_deg: np.ndarray,
    multipass_list: np.ndarray,
    geometric_gain: np.ndarray,
    optical_power_ratio: np.ndarray,
    random_noise_std_nm: np.ndarray,
    asd_bias_std_nm: np.ndarray,
    total_rms_nm: np.ndarray,
    theta_idx: int,
) -> None:
    """保存几何增益、光强和误差权衡图。"""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(multipass_list, geometric_gain[theta_idx, :], marker="o")
    ax.set_title(f"Geometric Gain at {theta_axis_deg[theta_idx]:.2f} deg")
    ax.set_xlabel("Multipass N")
    ax.set_ylabel("Gain = 2 N M sin(theta)")
    ax.grid(True, alpha=0.3)
    fig.savefig(IMG_DIR / "geometric_gain_vs_N.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(multipass_list, optical_power_ratio[theta_idx, :], marker="o")
    ax.set_yscale("log")
    ax.set_title(f"Optical Power Ratio at {theta_axis_deg[theta_idx]:.2f} deg")
    ax.set_xlabel("Multipass N")
    ax.set_ylabel("Power ratio")
    ax.grid(True, alpha=0.3)
    fig.savefig(IMG_DIR / "optical_power_vs_N.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    mesh = ax.pcolormesh(multipass_list, theta_axis_deg, total_rms_nm, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Total RMS (nm)")
    ax.set_title("Total Height Error RMS")
    ax.set_xlabel("Multipass N")
    ax.set_ylabel("Incident angle from surface normal (deg)")
    fig.savefig(IMG_DIR / "total_error_heatmap_theta_N.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(theta_axis_deg, asd_bias_std_nm[:, 0], "k--", label="ASD bias std")
    for n_idx, multipass_n in enumerate(multipass_list):
        ax.plot(theta_axis_deg, random_noise_std_nm[:, n_idx], label=f"Random N={multipass_n}")
    ax.set_title("Random Noise vs ASD Bias")
    ax.set_xlabel("Incident angle from surface normal (deg)")
    ax.set_ylabel("Height error component (nm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(IMG_DIR / "random_vs_asd_error_tradeoff.png", dpi=200)
    plt.close(fig)


def simulate_triangulation(input_path: Path | None = None, config: Dict[str, Any] | None = None) -> Path:
    """读取 ASD 结果并模拟掠入射三角测量高度恢复。"""
    cfg = CONFIG if config is None else config
    ensure_output_dirs()
    input_path = latest_npz("grazing_asd_*.npz") if input_path is None else Path(input_path)

    with np.load(input_path) as data:
        # 只载入三角测量需要的统计量，避免全量 ASD_mc 三维数组在大任务中重复占用内存。
        required_keys = [
            "theta_axis_deg",
            "theta_axis_rad",
            "recommended_theta_table",
            "best_pol",
            "R_mean_p",
            "R_mean_s",
            "ASD_error_mc_p_nm",
            "ASD_error_mc_s_nm",
        ]
        asd_data = {key: data[key].copy() for key in required_keys if key in data.files}

    theta_axis_deg = np.asarray(asd_data["theta_axis_deg"], dtype=np.float64)
    theta_axis_rad = np.asarray(asd_data["theta_axis_rad"], dtype=np.float64)
    multipass_list = np.asarray(cfg["multipass_list"], dtype=int)
    height_true_nm = build_height_axis(cfg)
    selected_pol = select_polarization(asd_data)
    r_mean = np.asarray(asd_data[f"R_mean_{selected_pol}"], dtype=np.float64)
    asd_bias_mean_nm, asd_bias_std_theta_nm, asd_bias_rms_theta_nm = summarize_asd_bias(asd_data, selected_pol)

    n_theta = theta_axis_deg.size
    n_pass = multipass_list.size
    n_height = height_true_nm.size
    pitch_nm = float(cfg["grating_pitch_um"]) * 1000.0
    magnification = float(cfg["imaging_magnification"])
    mirror_reflectivity = float(cfg["mirror_reflectivity"])
    mirror_count = int(cfg["extra_mirror_count_per_wafer_pass"])
    detector_noise_std = float(cfg["detector_noise_std"])
    amplitude_model = str(cfg.get("amplitude_loss_model", "sqrt")).lower()

    geometric_gain = np.zeros((n_theta, n_pass), dtype=np.float64)
    optical_power_ratio = np.zeros_like(geometric_gain)
    optical_amplitude = np.zeros_like(geometric_gain)
    random_noise_std_nm = np.zeros_like(geometric_gain)
    asd_bias_std_nm = np.zeros_like(geometric_gain)
    asd_bias_rms_nm = np.zeros_like(geometric_gain)
    total_rms_nm = np.zeros_like(geometric_gain)
    z_rec_nm = np.zeros((n_theta, n_pass, n_height), dtype=np.float32)
    z_error_nm = np.zeros_like(z_rec_nm)

    rng = np.random.default_rng(int(cfg.get("random_seed", 20260616)) + 17)
    for i_theta, theta_rad in enumerate(theta_axis_rad):
        wafer_reflectance = float(np.clip(r_mean[i_theta], 0.0, 1.0))
        for i_pass, multipass_n in enumerate(multipass_list):
            gain = 2.0 * float(multipass_n) * magnification * np.sin(theta_rad)
            power_ratio = (wafer_reflectance**int(multipass_n)) * (
                mirror_reflectivity ** (mirror_count * int(multipass_n))
            )
            amplitude = np.sqrt(max(power_ratio, 0.0)) if amplitude_model == "sqrt" else max(power_ratio, 0.0)
            z_rec, z_error = simulate_four_phase_readout(
                height_true_nm,
                theta_rad,
                int(multipass_n),
                float(asd_bias_mean_nm[i_theta]),
                amplitude,
                cfg,
                rng,
            )

            geometric_gain[i_theta, i_pass] = gain
            optical_power_ratio[i_theta, i_pass] = power_ratio
            optical_amplitude[i_theta, i_pass] = amplitude
            random_noise_std_nm[i_theta, i_pass] = phase_noise_to_height_std_nm(
                detector_noise_std,
                np.asarray(amplitude),
                pitch_nm,
                np.asarray(gain),
            )
            asd_bias_std_nm[i_theta, i_pass] = asd_bias_std_theta_nm[i_theta]
            asd_bias_rms_nm[i_theta, i_pass] = asd_bias_rms_theta_nm[i_theta]
            total_rms_nm[i_theta, i_pass] = np.sqrt(
                random_noise_std_nm[i_theta, i_pass] ** 2 + asd_bias_rms_nm[i_theta, i_pass] ** 2
            )
            z_rec_nm[i_theta, i_pass, :] = z_rec.astype(np.float32)
            z_error_nm[i_theta, i_pass, :] = z_error.astype(np.float32)

    best_flat_idx = int(np.nanargmin(total_rms_nm))
    best_theta_idx, best_pass_idx = np.unravel_index(best_flat_idx, total_rms_nm.shape)
    best_theta_deg = float(theta_axis_deg[best_theta_idx])
    best_multipass_n = int(multipass_list[best_pass_idx])

    _plot_best_height_curves(height_true_nm, z_rec_nm, z_error_nm, theta_axis_deg, multipass_list, best_theta_idx)
    _plot_tradeoffs(
        theta_axis_deg,
        multipass_list,
        geometric_gain,
        optical_power_ratio,
        random_noise_std_nm,
        asd_bias_std_nm,
        total_rms_nm,
        best_theta_idx,
    )

    rows: List[Dict[str, float | int]] = []
    for i_theta, theta in enumerate(theta_axis_deg):
        for i_pass, multipass_n in enumerate(multipass_list):
            rows.append(
                {
                    "theta_deg": float(theta),
                    "multipass_N": int(multipass_n),
                    "geometric_gain": float(geometric_gain[i_theta, i_pass]),
                    "optical_power_ratio": float(optical_power_ratio[i_theta, i_pass]),
                    "random_noise_std_nm": float(random_noise_std_nm[i_theta, i_pass]),
                    "asd_bias_std_nm": float(asd_bias_std_nm[i_theta, i_pass]),
                    "asd_bias_rms_nm": float(asd_bias_rms_nm[i_theta, i_pass]),
                    "total_rms_nm": float(total_rms_nm[i_theta, i_pass]),
                }
            )
    summary_csv = LINEAR_FIT_DIR / f"triangulation_summary_{timestamp()}.csv"
    _save_summary_csv(summary_csv, rows)

    output_path = DATA_DIR / f"grazing_triangulation_{timestamp()}.npz"
    np.savez_compressed(
        output_path,
        input_asd_npz=np.asarray(str(input_path)),
        height_true_nm=height_true_nm.astype(np.float32),
        theta_axis_deg=theta_axis_deg.astype(np.float32),
        multipass_list=multipass_list.astype(np.int32),
        z_rec_nm=z_rec_nm,
        z_error_nm=z_error_nm,
        random_noise_std_nm=random_noise_std_nm.astype(np.float32),
        asd_bias_std_nm=asd_bias_std_nm.astype(np.float32),
        asd_bias_rms_nm=asd_bias_rms_nm.astype(np.float32),
        total_rms_nm=total_rms_nm.astype(np.float32),
        optical_power_ratio=optical_power_ratio.astype(np.float32),
        optical_amplitude=optical_amplitude.astype(np.float32),
        geometric_gain=geometric_gain.astype(np.float32),
        best_theta_deg=np.asarray(best_theta_deg),
        best_multipass_N=np.asarray(best_multipass_n),
        selected_pol=np.asarray(selected_pol),
        summary_csv=np.asarray(str(summary_csv)),
        config_json=np.asarray(config_json(cfg)),
    )
    print("=== Grazing Triangulation Simulation ===")
    print(f"Input: {input_path}")
    print(f"Selected polarization: {selected_pol}")
    print(f"Best theta/N: {best_theta_deg:.3f} deg, N={best_multipass_n}")
    print(f"Saved triangulation data: {output_path}")
    print(f"Saved summary table: {summary_csv}")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Simulate grazing-incidence triangulation from ASD npz.")
    parser.add_argument("--input", type=Path, default=None, help="Path to grazing_asd_*.npz")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    simulate_triangulation(args.input)


if __name__ == "__main__":
    main()
