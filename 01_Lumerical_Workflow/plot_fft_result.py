import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from solve_npz_fft import FFTSolver, NPZSpectrumLoader


def _读取标量字符串(npz_data, key, default=""):
    if key not in npz_data.files:
        return default

    value = npz_data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def _扫描轴标签(scan_axis_name):
    labels = {
        "angle_deg": "Incident angle (deg)",
        "cavity_um": "Cavity length (um)",
        "cavity_m": "Cavity length (m)",
        "time_or_scan_axis": "Time / scan axis",
        "spectrum_index": "Spectrum index",
    }
    return labels.get(scan_axis_name, scan_axis_name)


def _读取配置(npz_data):
    if "config_json" not in npz_data.files:
        return {
            "FFT_PEAK_HEIGHT_RATIO": 0.2,
            "FFT_IGNORE_DC_BINS": 50,
            "FFT_PEAK_DISTANCE_BINS": 100,
            "ZERO_PAD_FACTOR": 8,
            "SAVE_FFT_MATRIX": False,
        }

    return json.loads(str(npz_data["config_json"].item()))


def 读取_fft_结果(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = {
        "scan_axis",
        "scan_axis_name",
        "peak_count",
        "first_peak_distance_um",
        "dominant_peak_distance_um",
        "all_peak_distances_um",
        "all_peak_heights",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"{npz_path} 缺少必要字段: {missing}")

    return data


def 读取单条原始光谱(fft_data, selected_indices):
    """仅在选择了一条光谱时，从 source_npz 中读取原始光谱。"""
    if selected_indices.size != 1:
        return None

    source_npz = _读取标量字符串(fft_data, "source_npz", "")
    if not source_npz:
        return None

    source = NPZSpectrumLoader.load(source_npz)
    spectrum_idx = int(selected_indices[0])

    if spectrum_idx < 0 or spectrum_idx >= source["spectra"].shape[0]:
        raise IndexError(
            f"selected_spectrum_indices={spectrum_idx} 超出原始光谱数量 "
            f"{source['spectra'].shape[0]}。"
        )

    return {
        "source_npz": source_npz,
        "spectrum_idx": spectrum_idx,
        "wavelengths_um": source["wavelengths"],
        "intensities": source["spectra"][spectrum_idx, :],
    }


def plot_fft_result(
    npz_path,
    output_path=None,
    max_scatter_peaks=None,
    show_interactive=True,
):
    """绘制 solve_npz_fft.py 输出的 FFT 解算结果。

    当输入结果只包含一条光谱时，额外绘制：
    1. 原始光谱：横坐标为 wavelength。
    2. 单条光谱 FFT 幅值：横坐标为 main.py 中的距离轴 Distance。

    当输入结果包含多条或全部光谱时，只绘制随扫描变量变化的解算结果。
    """
    data = 读取_fft_结果(npz_path)

    scan_axis = np.asarray(data["scan_axis"], dtype=float).reshape(-1)
    scan_axis_name = _读取标量字符串(data, "scan_axis_name", "scan_axis")
    selected_indices = (
        np.asarray(data["selected_spectrum_indices"], dtype=int).reshape(-1)
        if "selected_spectrum_indices" in data.files
        else np.arange(scan_axis.size)
    )

    peak_count = np.asarray(data["peak_count"], dtype=int).reshape(-1)
    first_peak_distance_um = np.asarray(data["first_peak_distance_um"], dtype=float).reshape(-1)
    dominant_peak_distance_um = np.asarray(data["dominant_peak_distance_um"], dtype=float).reshape(-1)
    all_peak_distances_um = data["all_peak_distances_um"]
    all_peak_heights = data["all_peak_heights"]

    single_spectrum = 读取单条原始光谱(data, selected_indices)
    has_single_detail = single_spectrum is not None

    x_label = _扫描轴标签(scan_axis_name)
    has_fft_matrix = (
        not has_single_detail
        and "fft_matrix" in data.files
        and "distance_axis_um" in data.files
    )

    n_rows = 4 if has_single_detail else (3 if has_fft_matrix else 2)
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4.0 * n_rows), constrained_layout=True)
    axs = np.asarray(axs).reshape(-1)

    ax_result = axs[0]
    ax_count = axs[1]

    # 图 1：FFT 解算出的距离结果，横坐标由扫描变量决定。
    ax_result.plot(scan_axis, first_peak_distance_um, "o-", ms=3, lw=1, label="First peak")
    ax_result.plot(scan_axis, dominant_peak_distance_um, "s-", ms=3, lw=1, label="Dominant peak")

    scatter_x = []
    scatter_y = []
    scatter_c = []
    for i, x in enumerate(scan_axis):
        distances = np.asarray(all_peak_distances_um[i], dtype=float).reshape(-1)
        heights = np.asarray(all_peak_heights[i], dtype=float).reshape(-1)
        if max_scatter_peaks is not None:
            distances = distances[:max_scatter_peaks]
            heights = heights[:max_scatter_peaks]
        scatter_x.extend([x] * len(distances))
        scatter_y.extend(distances.tolist())
        scatter_c.extend(heights.tolist())

    if scatter_x:
        sc = ax_result.scatter(
            scatter_x,
            scatter_y,
            c=scatter_c,
            s=12,
            cmap="viridis",
            alpha=0.6,
            label="All peaks",
        )
        fig.colorbar(sc, ax=ax_result, label="Peak height")

    ax_result.set_title("FFT Solved Peak Distance")
    ax_result.set_xlabel(x_label)
    ax_result.set_ylabel("Distance (um)")
    ax_result.grid(True)
    ax_result.legend()

    # 图 2：每条光谱检测到的峰数量，横坐标同样由扫描变量决定。
    ax_count.bar(scan_axis, peak_count, width=_柱状图宽度(scan_axis), color="#4c78a8")
    ax_count.set_title("Detected Peak Count")
    ax_count.set_xlabel(x_label)
    ax_count.set_ylabel("Count")
    ax_count.grid(True, axis="y")

    if has_single_detail:
        _绘制单条光谱细节(
            axs[2],
            axs[3],
            single_spectrum,
            _读取配置(data),
        )
    elif has_fft_matrix:
        _绘制_fft_热图(axs[2], data, scan_axis, x_label, fig)

    source_npz = _读取标量字符串(data, "source_npz", "")
    fig.suptitle(
        f"FFT Result: {os.path.basename(npz_path)}\n"
        f"Source: {os.path.basename(source_npz)} | spectra: {selected_indices.size}",
        fontsize=12,
    )

    if output_path is None:
        output_path = _默认输出路径(npz_path)

    fig.savefig(output_path, dpi=220)
    print(f"Saved FFT result figure: {output_path}")

    if show_interactive:
        # 阻塞显示交互窗口，便于缩放、平移和查看坐标。
        plt.show(block=True)
    else:
        plt.close(fig)

    return output_path


def _绘制单条光谱细节(ax_spectrum, ax_fft, single_spectrum, config):
    wavelengths_um = single_spectrum["wavelengths_um"]
    wavelengths_nm = wavelengths_um * 1000
    intensities = single_spectrum["intensities"]
    spectrum_idx = single_spectrum["spectrum_idx"]

    # 原始光谱图：横坐标为波长。
    ax_spectrum.plot(wavelengths_nm, intensities, color="#1f77b4", lw=0.8)
    ax_spectrum.set_title(f"Original Spectrum, index = {spectrum_idx}")
    ax_spectrum.set_xlabel("Wavelength (nm)")
    ax_spectrum.set_ylabel("Reflectance")
    ax_spectrum.grid(True)

    # FFT 幅值图：FFT 后的物理横坐标是距离轴，和 main.py 保持一致。
    fft_res = FFTSolver.solve(wavelengths_um, intensities, config)
    ax_fft.plot(fft_res["distance_axis_um"], fft_res["fft_data"], color="#222222", lw=0.8)

    peak_distances = fft_res["peak_distances_um"]
    peak_heights = fft_res["peak_heights"]
    if len(peak_distances) > 0:
        ax_fft.scatter(peak_distances, peak_heights, color="#d62728", s=28, label="Detected peaks")
        for x, y in zip(peak_distances, peak_heights):
            ax_fft.axvline(x, color="#d62728", ls="--", lw=0.8, alpha=0.35)
            ax_fft.text(x, y, f" {x:.2f} um", rotation=90, fontsize=8, va="bottom")
        ax_fft.legend()

    ax_fft.set_title("FFT Amplitude of Selected Spectrum")
    ax_fft.set_xlabel("Distance (um)")
    ax_fft.set_ylabel("FFT amplitude")
    ax_fft.grid(True)


def _绘制_fft_热图(ax_heat, data, scan_axis, x_label, fig):
    fft_matrix = np.asarray(data["fft_matrix"], dtype=float)
    distance_axis_um = np.asarray(data["distance_axis_um"], dtype=float).reshape(-1)
    vmax = np.nanpercentile(fft_matrix, 99)
    mesh = ax_heat.pcolormesh(
        distance_axis_um,
        scan_axis,
        fft_matrix,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    ax_heat.set_title("FFT Amplitude Map")
    ax_heat.set_xlabel("Distance (um)")
    ax_heat.set_ylabel(x_label)
    fig.colorbar(mesh, ax=ax_heat, label="FFT amplitude")


def _柱状图宽度(x):
    if len(x) < 2:
        return 0.8
    diffs = np.diff(np.sort(np.unique(x)))
    if diffs.size == 0:
        return 0.8
    return float(np.min(diffs) * 0.8)


def _默认输出路径(npz_path):
    folder = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"{stem}_plot_{timestamp}.png")


def main_direct():
    """直接在代码中指定输入参数，不使用命令行。"""
    fft_npz_path = r"./stackrt_result/cavity_spectra_20260609_121020_fft_solved_20260609_221904.npz"
    output_path = None

    # 数据很密时，可以设为 5 之类的小整数，只画每条光谱前几个峰的散点。
    max_scatter_peaks = None

    # True 表示保存图片后弹出 Matplotlib 交互窗口。
    show_interactive = True

    plot_fft_result(
        npz_path=fft_npz_path,
        output_path=output_path,
        max_scatter_peaks=max_scatter_peaks,
        show_interactive=show_interactive,
    )


if __name__ == "__main__":
    main_direct()
