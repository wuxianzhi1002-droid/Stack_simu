import json
import os
from datetime import datetime

import matplotlib
import numpy as np

from solve_npz_fft import FFTSolver, NPZSpectrumLoader


# 优先使用可弹出窗口的 Matplotlib 后端。若本机没有对应 GUI 库，则保留默认后端。
for backend in ("TkAgg", "QtAgg", "WxAgg"):
    try:
        matplotlib.use(backend, force=True)
        break
    except Exception:
        pass

import matplotlib.pyplot as plt


def read_scalar_string(npz_data, key, default=""):
    """从 npz 中读取标量字符串字段。"""
    if key not in npz_data.files:
        return default

    value = npz_data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def scan_axis_label(scan_axis_name):
    """根据 solve_npz_fft.py 保存的扫描轴类型生成坐标轴标签。"""
    labels = {
        "angle_deg": "Incident angle (deg)",
        "cavity_um": "Cavity length (um)",
        "cavity_m": "Cavity length (m)",
        "time_or_scan_axis": "Time / scan axis",
        "spectrum_index": "Spectrum index",
    }
    return labels.get(scan_axis_name, scan_axis_name)


def read_config(npz_data):
    """读取 FFT 解算配置，用于单条光谱时重新计算 FFT 幅值曲线。"""
    if "config_json" not in npz_data.files:
        return {
            "FFT_PEAK_HEIGHT_RATIO": 0.2,
            "FFT_IGNORE_DC_BINS": 50,
            "FFT_PEAK_DISTANCE_BINS": 100,
            "ZERO_PAD_FACTOR": 8,
            "SAVE_FFT_MATRIX": False,
        }

    return json.loads(str(npz_data["config_json"].item()))


def load_fft_result(npz_path):
    """读取 solve_npz_fft.py 输出的 npz 结果。"""
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


def load_selected_original_spectrum(fft_data, selected_indices):
    """仅当 selected_spectrum_indices 数量为 1 时，读取对应原始光谱。"""
    if selected_indices.size != 1:
        return None

    source_npz = read_scalar_string(fft_data, "source_npz", "")
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
    """绘制 FFT 解算结果。

    规则：
    1. selected_spectrum_indices 数量为 1 时，在同一个窗口中额外画原始光谱和该光谱的 FFT 幅值图。
    2. selected_spectrum_indices 数量大于 1 时，只画扫描解算结果。
    3. 解算结果图的横坐标由输入 npz 的扫描变量决定。
    4. 原始光谱图横坐标为 wavelength；FFT 幅值图横坐标为 main.py 算法中的 Distance (um)。
    """
    data = load_fft_result(npz_path)

    scan_axis = np.asarray(data["scan_axis"], dtype=float).reshape(-1)
    scan_axis_name = read_scalar_string(data, "scan_axis_name", "scan_axis")
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

    single_spectrum = load_selected_original_spectrum(data, selected_indices)
    has_single_detail = single_spectrum is not None
    has_fft_matrix = (
        not has_single_detail
        and "fft_matrix" in data.files
        and "distance_axis_um" in data.files
    )

    n_rows = 4 if has_single_detail else (3 if has_fft_matrix else 2)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4.0 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    x_label = scan_axis_label(scan_axis_name)
    plot_solved_peak_distance(
        axes[0],
        fig,
        scan_axis,
        x_label,
        first_peak_distance_um,
        dominant_peak_distance_um,
        all_peak_distances_um,
        all_peak_heights,
        max_scatter_peaks,
    )
    plot_peak_count(axes[1], scan_axis, x_label, peak_count)

    if has_single_detail:
        plot_single_spectrum_detail(axes[2], axes[3], single_spectrum, read_config(data))
    elif has_fft_matrix:
        plot_fft_heatmap(axes[2], fig, data, scan_axis, x_label)

    source_npz = read_scalar_string(data, "source_npz", "")
    fig.suptitle(
        f"FFT Result: {os.path.basename(npz_path)}\n"
        f"Source: {os.path.basename(source_npz)} | selected spectra: {selected_indices.size}",
        fontsize=12,
    )

    if output_path is None:
        output_path = default_output_path(npz_path)

    fig.savefig(output_path, dpi=220)
    print(f"Saved FFT result figure: {output_path}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    if show_interactive:
        # 阻塞显示交互窗口，窗口关闭后脚本才会结束。
        plt.show(block=True)
    else:
        plt.close(fig)

    return output_path


def plot_solved_peak_distance(
    ax,
    fig,
    scan_axis,
    x_label,
    first_peak_distance_um,
    dominant_peak_distance_um,
    all_peak_distances_um,
    all_peak_heights,
    max_scatter_peaks,
):
    """绘制解算出的峰值距离，横坐标为扫描变量。"""
    ax.plot(scan_axis, first_peak_distance_um, "o-", ms=3, lw=1, label="First peak")
    ax.plot(scan_axis, dominant_peak_distance_um, "s-", ms=3, lw=1, label="Dominant peak")

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
        sc = ax.scatter(
            scatter_x,
            scatter_y,
            c=scatter_c,
            s=12,
            cmap="viridis",
            alpha=0.6,
            label="All peaks",
        )
        fig.colorbar(sc, ax=ax, label="Peak height")

    ax.set_title("FFT Solved Peak Distance")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Distance (um)")
    ax.grid(True)
    ax.legend()


def plot_peak_count(ax, scan_axis, x_label, peak_count):
    """绘制每个扫描点检测到的峰数量。"""
    ax.bar(scan_axis, peak_count, width=bar_width(scan_axis), color="#4c78a8")
    ax.set_title("Detected Peak Count")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")


def plot_single_spectrum_detail(ax_spectrum, ax_fft, single_spectrum, config):
    """绘制单条原始光谱和对应 FFT 幅值图。"""
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

    # FFT 后的横坐标为距离轴，和 main.py 的 FFTSolver.solve() 保持一致。
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


def plot_fft_heatmap(ax, fig, data, scan_axis, x_label):
    """当保存了完整 fft_matrix 时，绘制多光谱 FFT 热图。"""
    fft_matrix = np.asarray(data["fft_matrix"], dtype=float)
    distance_axis_um = np.asarray(data["distance_axis_um"], dtype=float).reshape(-1)
    vmax = np.nanpercentile(fft_matrix, 99)
    mesh = ax.pcolormesh(
        distance_axis_um,
        scan_axis,
        fft_matrix,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    ax.set_title("FFT Amplitude Map")
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel(x_label)
    fig.colorbar(mesh, ax=ax, label="FFT amplitude")


def bar_width(x):
    """根据扫描轴间隔估计柱状图宽度。"""
    if len(x) < 2:
        return 0.8
    diffs = np.diff(np.sort(np.unique(x)))
    if diffs.size == 0:
        return 0.8
    return float(np.min(diffs) * 0.8)


def default_output_path(npz_path):
    folder = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"{stem}_plot_{timestamp}.png")


def main_direct():
    """直接在代码中指定输入参数，不使用命令行。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fft_npz_path = r"./stackrt_result/cavity_spectra_20260609_121020_fft_solved_20260609_233306.npz"
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
