import json
import os
from datetime import datetime

import matplotlib
import numpy as np
from scipy.signal import find_peaks


# ==================== 全局配置 ====================
CONFIG = {
    "FFT_PEAK_HEIGHT_RATIO": 0.2,
    "FFT_IGNORE_DC_BINS": 50,
    "FFT_PEAK_DISTANCE_BINS": 100,
    "ZERO_PAD_FACTOR": 8,
    "SAVE_FFT_MATRIX": False,
}


def setup_matplotlib_backend():
    """优先启用可弹出交互窗口的 Matplotlib 后端。"""
    for backend in ("TkAgg", "QtAgg", "WxAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            pass


setup_matplotlib_backend()
import matplotlib.pyplot as plt


# ==================== 单条光谱 FFT 解算 ====================
class FFTSolver:
    @staticmethod
    def solve(wavelengths_um, intensities, config):
        """参考 main.py 的思路，对单条光谱执行 k-space FFT 解算。"""
        wavelengths_um = np.asarray(wavelengths_um, dtype=float).reshape(-1)
        intensities = np.asarray(intensities, dtype=float).reshape(-1)

        if wavelengths_um.size != intensities.size:
            raise ValueError(
                f"wavelengths and intensities must have the same length, got "
                f"{wavelengths_um.size} and {intensities.size}."
            )

        finite_mask = np.isfinite(wavelengths_um) & np.isfinite(intensities)
        wavelengths_um = wavelengths_um[finite_mask]
        intensities = intensities[finite_mask]

        if wavelengths_um.size < 4:
            raise ValueError("At least 4 finite points are required for FFT solving.")

        sort_idx = np.argsort(wavelengths_um)
        wavelengths_um = wavelengths_um[sort_idx]
        intensities = intensities[sort_idx]

        k_raw = 2 * np.pi / wavelengths_um
        k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))

        if k_raw[0] > k_raw[-1]:
            i_linear = np.interp(k_linear, k_raw[::-1], intensities[::-1])
        else:
            i_linear = np.interp(k_linear, k_raw, intensities)

        i_detrend = i_linear - np.mean(i_linear)
        i_windowed = i_detrend * np.hanning(len(i_detrend))

        n_fft = len(i_windowed) * int(config["ZERO_PAD_FACTOR"])
        fft_data = np.abs(np.fft.rfft(i_windowed, n=n_fft))

        dk = abs(k_linear[1] - k_linear[0])
        max_range_um = np.pi / dk
        distance_axis_um = np.linspace(0, max_range_um / 2, len(fft_data))

        ignore = min(int(config["FFT_IGNORE_DC_BINS"]), max(0, len(fft_data) - 1))
        search = fft_data[ignore:]
        if search.size == 0 or np.max(search) <= 0:
            peaks = np.array([], dtype=int)
        else:
            peaks, _ = find_peaks(
                search,
                height=np.max(search) * float(config["FFT_PEAK_HEIGHT_RATIO"]),
                distance=int(config["FFT_PEAK_DISTANCE_BINS"]),
            )
            peaks = peaks + ignore

        return {
            "distance_axis_um": distance_axis_um,
            "fft_data": fft_data,
            "peaks_idx": peaks,
            "peak_distances_um": distance_axis_um[peaks],
            "peak_heights": fft_data[peaks],
            "max_range_um": max_range_um / 2,
        }


# ==================== 原始光谱 npz 读取 ====================
class NPZSpectrumLoader:
    AXIS_PRIORITY = (
        ("angle_axis", "angle_deg"),
        ("theta_axis", "angle_deg"),
        ("cavity_axis_um", "cavity_um"),
        ("cavity_axis_m", "cavity_m"),
        ("L_t", "cavity_um"),
        ("t_axis", "time_or_scan_axis"),
    )

    @staticmethod
    def load(npz_path):
        """读取原始 StackRT 光谱 npz，自动识别光谱矩阵和扫描轴。"""
        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.files)

        if "wavelengths" not in keys:
            raise KeyError(f"{npz_path} does not contain required key 'wavelengths'.")

        spectra_key = NPZSpectrumLoader._find_spectra_key(keys)
        wavelengths = np.asarray(data["wavelengths"], dtype=float).reshape(-1)
        spectra = np.asarray(data[spectra_key], dtype=float)

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        elif spectra.ndim != 2:
            raise ValueError(f"Expected spectra to be 1D or 2D, got shape {spectra.shape}.")

        spectra, wavelengths = NPZSpectrumLoader._orient_spectra(spectra, wavelengths)
        scan_axis, scan_axis_name = NPZSpectrumLoader._find_scan_axis(data, spectra.shape[0])

        return {
            "wavelengths": wavelengths,
            "spectra": spectra,
            "spectra_key": spectra_key,
            "scan_axis": scan_axis,
            "scan_axis_name": scan_axis_name,
            "source_keys": np.array(data.files),
        }

    @staticmethod
    def _find_spectra_key(keys):
        for key in ("spectra", "R", "Rp", "Rs", "intensities"):
            if key in keys:
                return key
        raise KeyError("Could not find a spectra key. Expected spectra, R, Rp, Rs, or intensities.")

    @staticmethod
    def _orient_spectra(spectra, wavelengths):
        if spectra.shape[1] == wavelengths.size:
            return spectra, wavelengths
        if spectra.shape[0] == wavelengths.size:
            return spectra.T, wavelengths
        raise ValueError(
            f"One spectra dimension must match wavelengths length {wavelengths.size}, got {spectra.shape}."
        )

    @staticmethod
    def _find_scan_axis(data, n_spectra):
        for key, axis_name in NPZSpectrumLoader.AXIS_PRIORITY:
            if key in data.files:
                axis = np.asarray(data[key]).reshape(-1)
                if axis.size == n_spectra:
                    return axis.astype(float, copy=False), axis_name

        return np.arange(n_spectra, dtype=float), "spectrum_index"


def normalize_spectrum_selection(selection, n_spectra):
    """将 all、整数、列表、slice 等选择方式统一转换为光谱索引数组。"""
    if selection == "all":
        selected_idx = np.arange(n_spectra, dtype=int)
    elif isinstance(selection, int):
        selected_idx = np.array([selection], dtype=int)
    elif isinstance(selection, slice):
        selected_idx = np.arange(n_spectra, dtype=int)[selection]
    else:
        selected_idx = np.asarray(selection, dtype=int).reshape(-1)

    if selected_idx.size == 0:
        raise ValueError("spectrum_selection selected no spectra.")

    selected_idx = np.where(selected_idx < 0, selected_idx + n_spectra, selected_idx)
    invalid = selected_idx[(selected_idx < 0) | (selected_idx >= n_spectra)]
    if invalid.size > 0:
        raise IndexError(f"spectrum_selection contains invalid indices: {invalid.tolist()}.")

    return selected_idx


# ==================== 批量 FFT 解算和保存 ====================
class BatchFFTSolver:
    @staticmethod
    def solve_npz(npz_path, output_path=None, config=None, spectrum_selection="all"):
        """读取原始光谱 npz，按选择的光谱执行 FFT 解算并保存结果 npz。"""
        loaded = NPZSpectrumLoader.load(npz_path)
        selected_idx = normalize_spectrum_selection(spectrum_selection, loaded["spectra"].shape[0])

        loaded["spectra"] = loaded["spectra"][selected_idx, :]
        loaded["scan_axis"] = loaded["scan_axis"][selected_idx]

        return BatchFFTSolver.solve_loaded_npz(
            loaded=loaded,
            npz_path=npz_path,
            output_path=output_path,
            config=config,
            selected_idx=selected_idx,
        )

    @staticmethod
    def solve_loaded_npz(loaded, npz_path, output_path=None, config=None, selected_idx=None):
        config = dict(CONFIG if config is None else config)

        wavelengths = loaded["wavelengths"]
        spectra = loaded["spectra"]
        scan_axis = loaded["scan_axis"]
        n_spectra = spectra.shape[0]

        if selected_idx is None:
            selected_idx = np.arange(n_spectra, dtype=int)

        first_peak_distance_um = np.full(n_spectra, np.nan)
        first_peak_height = np.full(n_spectra, np.nan)
        dominant_peak_distance_um = np.full(n_spectra, np.nan)
        dominant_peak_height = np.full(n_spectra, np.nan)
        peak_count = np.zeros(n_spectra, dtype=int)
        all_peak_distances_um = np.empty(n_spectra, dtype=object)
        all_peak_heights = np.empty(n_spectra, dtype=object)

        fft_matrix = []
        distance_axis_um = None
        max_range_um = np.nan

        for i in range(n_spectra):
            res = FFTSolver.solve(wavelengths, spectra[i, :], config)

            if distance_axis_um is None:
                distance_axis_um = res["distance_axis_um"]
                max_range_um = res["max_range_um"]

            peak_distances = res["peak_distances_um"]
            peak_heights = res["peak_heights"]
            peak_count[i] = len(peak_distances)
            all_peak_distances_um[i] = peak_distances
            all_peak_heights[i] = peak_heights

            if len(peak_distances) > 0:
                first_peak_distance_um[i] = peak_distances[0]
                first_peak_height[i] = peak_heights[0]
                dominant_idx = int(np.argmax(peak_heights))
                dominant_peak_distance_um[i] = peak_distances[dominant_idx]
                dominant_peak_height[i] = peak_heights[dominant_idx]

            if config["SAVE_FFT_MATRIX"]:
                fft_matrix.append(res["fft_data"])

        if output_path is None:
            output_path = default_fft_output_path(npz_path)

        save_kwargs = {
            "source_npz": np.array(os.path.abspath(npz_path)),
            "source_keys": loaded["source_keys"],
            "spectra_key": np.array(loaded["spectra_key"]),
            "scan_axis_name": np.array(loaded["scan_axis_name"]),
            "scan_axis": scan_axis,
            "selected_spectrum_indices": np.asarray(selected_idx, dtype=int),
            "selected_spectrum_count": np.array(len(selected_idx), dtype=int),
            "is_single_spectrum": np.array(len(selected_idx) == 1, dtype=bool),
            "wavelengths": wavelengths,
            "distance_axis_um": distance_axis_um,
            "max_range_um": np.array(max_range_um),
            "peak_count": peak_count,
            "first_peak_distance_um": first_peak_distance_um,
            "first_peak_height": first_peak_height,
            "dominant_peak_distance_um": dominant_peak_distance_um,
            "dominant_peak_height": dominant_peak_height,
            "all_peak_distances_um": all_peak_distances_um,
            "all_peak_heights": all_peak_heights,
            "config_json": np.array(json.dumps(config, ensure_ascii=False)),
        }

        if config["SAVE_FFT_MATRIX"]:
            save_kwargs["fft_matrix"] = np.asarray(fft_matrix)

        np.savez_compressed(output_path, **save_kwargs)
        return output_path


def default_fft_output_path(npz_path):
    folder = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"{stem}_fft_solved_{timestamp}.npz")


# ==================== FFT 结果读取和绘图 ====================
def read_scalar_string(npz_data, key, default=""):
    if key not in npz_data.files:
        return default
    value = npz_data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def scan_axis_label(scan_axis_name):
    labels = {
        "angle_deg": "Incident angle (deg)",
        "cavity_um": "Cavity length (um)",
        "cavity_m": "Cavity length (m)",
        "time_or_scan_axis": "Time / scan axis",
        "spectrum_index": "Spectrum index",
    }
    return labels.get(scan_axis_name, scan_axis_name)


def read_config(npz_data):
    if "config_json" not in npz_data.files:
        return dict(CONFIG)
    return json.loads(str(npz_data["config_json"].item()))


def load_fft_result(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = {
        "scan_axis",
        "scan_axis_name",
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
    """单条光谱模式下，从 source_npz 读取原始光谱。"""
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


def plot_fft_result(npz_path, output_path=None, max_scatter_peaks=None, show_interactive=True):
    """绘制 FFT 解算结果。

    单条光谱结果会额外绘制原始光谱和该光谱 FFT 幅值图；
    多条或 all 结果只绘制扫描解算结果。
    """
    data = load_fft_result(npz_path)

    scan_axis = np.asarray(data["scan_axis"], dtype=float).reshape(-1)
    scan_axis_name = read_scalar_string(data, "scan_axis_name", "scan_axis")
    selected_indices = (
        np.asarray(data["selected_spectrum_indices"], dtype=int).reshape(-1)
        if "selected_spectrum_indices" in data.files
        else np.arange(scan_axis.size)
    )

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
    plot_cavity_error(
        axes[1],
        scan_axis,
        scan_axis_name,
        x_label,
        first_peak_distance_um,
        dominant_peak_distance_um,
    )

    if has_single_detail:
        plot_single_spectrum_detail(axes[2], axes[3], single_spectrum, read_config(data))
    elif has_fft_matrix:
        plot_fft_heatmap(axes[2], fig, data, scan_axis, x_label)

    if output_path is None:
        output_path = default_plot_output_path(npz_path)

    fig.savefig(output_path, dpi=220)
    print(f"Saved FFT result figure: {output_path}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    if show_interactive:
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
    """绘制解算距离，横坐标为输入 npz 的扫描变量。"""
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


def plot_cavity_error(
    ax,
    scan_axis,
    scan_axis_name,
    x_label,
    first_peak_distance_um,
    dominant_peak_distance_um,
):
    """绘制 error = solved_distance_um - cavity_length_um。"""
    if scan_axis_name == "cavity_um":
        cavity_axis_um = scan_axis
    elif scan_axis_name == "cavity_m":
        cavity_axis_um = scan_axis * 1e6
    else:
        ax.text(
            0.5,
            0.5,
            "Cavity error is only available for cavity length scans.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Error vs Cavity Length")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Error (um)")
        ax.grid(True)
        return

    first_error_um = first_peak_distance_um - cavity_axis_um
    dominant_error_um = dominant_peak_distance_um - cavity_axis_um

    ax.axhline(0, color="#666666", lw=0.9, ls="--")
    ax.plot(scan_axis, first_error_um, "o-", ms=3, lw=1, label="First peak error")
    ax.plot(scan_axis, dominant_error_um, "s-", ms=3, lw=1, label="Dominant peak error")
    ax.set_title("Error vs Cavity Length")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Solved distance - cavity length (um)")
    ax.grid(True)
    ax.legend()


def plot_single_spectrum_detail(ax_spectrum, ax_fft, single_spectrum, config):
    """绘制单条原始光谱和该光谱的 FFT 幅值。"""
    wavelengths_um = single_spectrum["wavelengths_um"]
    wavelengths_nm = wavelengths_um * 1000
    intensities = single_spectrum["intensities"]
    spectrum_idx = single_spectrum["spectrum_idx"]

    ax_spectrum.plot(wavelengths_nm, intensities, color="#1f77b4", lw=0.8)
    ax_spectrum.set_title(f"Original Spectrum, index = {spectrum_idx}")
    ax_spectrum.set_xlabel("Wavelength (nm)")
    ax_spectrum.set_ylabel("Reflectance")
    ax_spectrum.grid(True)

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


def linear_fit_peak_distance(
    npz_path,
    peak_type="dominant",
    output_path=None,
    show_interactive=True,
):
    """对 FFT 解算出的 peak distance 做线性拟合并画图。

    peak_type 可选：
    - "dominant": 使用 dominant_peak_distance_um
    - "first": 使用 first_peak_distance_um
    """
    data = load_fft_result(npz_path)
    scan_axis_name = read_scalar_string(data, "scan_axis_name", "scan_axis")
    x_raw = np.asarray(data["scan_axis"], dtype=float).reshape(-1)

    if peak_type == "dominant":
        y = np.asarray(data["dominant_peak_distance_um"], dtype=float).reshape(-1)
        y_name = "Dominant peak distance"
    elif peak_type == "first":
        y = np.asarray(data["first_peak_distance_um"], dtype=float).reshape(-1)
        y_name = "First peak distance"
    else:
        raise ValueError("peak_type must be 'dominant' or 'first'.")

    # 解算距离单位为 um；若腔长扫描轴单位为 m，则先转换到 um 再拟合。
    if scan_axis_name == "cavity_m":
        x = x_raw * 1e6
        x_label = "Cavity length (um)"
    else:
        x = x_raw
        x_label = scan_axis_label(scan_axis_name)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[finite_mask]
    y_fit = y[finite_mask]
    if x_fit.size < 2:
        raise ValueError("At least 2 finite points are required for linear fitting.")

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    y_pred = slope * x_fit + intercept
    residual = y_fit - y_pred

    r_value = float(np.corrcoef(x_fit, y_fit)[0, 1])
    r_squared = r_value**2
    rmse_um = float(np.sqrt(np.mean(residual**2)))
    mae_um = float(np.mean(np.abs(residual)))
    max_abs_error_um = float(np.max(np.abs(residual)))

    x_line = np.linspace(np.min(x_fit), np.max(x_fit), 500)
    y_line = slope * x_line + intercept

    fig, (ax_fit, ax_error) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_fit.scatter(x_fit, y_fit, s=18, alpha=0.75, label="FFT solved distance")
    ax_fit.plot(
        x_line,
        y_line,
        color="#d62728",
        lw=1.5,
        label=f"Linear fit: y = {slope:.8g} x + {intercept:.8g}",
    )

    metrics_text = (
        f"R = {r_value:.8f}\n"
        f"R^2 = {r_squared:.8f}\n"
        f"RMSE = {rmse_um:.6g} um\n"
        f"MAE = {mae_um:.6g} um\n"
        f"Max |error| = {max_abs_error_um:.6g} um"
    )
    ax_fit.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax_fit.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax_fit.set_title(f"Linear Fit of {y_name}")
    ax_fit.set_xlabel(x_label)
    ax_fit.set_ylabel("Solved peak distance (um)")
    ax_fit.grid(True)
    ax_fit.legend()

    # 误差图：残差 = 解算距离 - 线性拟合距离。
    ax_error.axhline(0, color="#555555", lw=0.9, ls="--", label="Zero error")
    ax_error.axhline(rmse_um, color="#d62728", lw=0.8, ls=":", label="+RMSE")
    ax_error.axhline(-rmse_um, color="#d62728", lw=0.8, ls=":", label="-RMSE")
    ax_error.plot(x_fit, residual, "o-", ms=3, lw=1, color="#1f77b4", label="Fit residual")
    ax_error.set_title("Linear Fit Error")
    ax_error.set_xlabel(x_label)
    ax_error.set_ylabel("Solved - fitted (um)")
    ax_error.grid(True)
    ax_error.legend()

    if output_path is None:
        output_path = default_linear_fit_output_path(npz_path, peak_type)

    fig.savefig(output_path, dpi=220)

    txt_path = os.path.splitext(output_path)[0] + ".txt"
    report_lines = [
        f"Saved linear fit figure: {output_path}",
        f"Linear fit peak type: {peak_type}",
        f"slope = {slope:.12g}",
        f"intercept = {intercept:.12g}",
        f"R = {r_value:.12g}",
        f"R^2 = {r_squared:.12g}",
        f"RMSE = {rmse_um:.12g} um",
        f"MAE = {mae_um:.12g} um",
        f"Max |error| = {max_abs_error_um:.12g} um",
        f"num_fit_points = {int(x_fit.size)}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    for line in report_lines:
        print(line)
    print(f"Saved linear fit report: {txt_path}")

    if show_interactive:
        plt.show(block=True)
    else:
        plt.close(fig)

    return {
        "output_path": output_path,
        "report_path": txt_path,
        "peak_type": peak_type,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": r_value,
        "r_squared": r_squared,
        "rmse_um": rmse_um,
        "mae_um": mae_um,
        "max_abs_error_um": max_abs_error_um,
        "num_fit_points": int(x_fit.size),
    }


def default_plot_output_path(npz_path):
    folder = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"{stem}_plot_{timestamp}.png")


def default_linear_fit_output_path(npz_path, peak_type):
    folder = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"{stem}_{peak_type}_linear_fit_{timestamp}.png")


# ==================== 直接运行入口 ====================
def main_direct():
    """直接在代码中配置运行模式，不使用命令行参数。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # mode 可选：
    # "solve_only"      -> 只从原始光谱 npz 解算并保存 fft_solved npz
    # "plot_only"       -> 只读取已有 fft_solved npz 并绘图
    # "solve_and_plot"  -> 先解算，再绘图
    mode = "solve_only"

    input_npz_path = os.path.join(script_dir, "stackrt_result", "cavity_spectra_20260609_121020.npz")
    fft_result_npz_path = os.path.join(
        script_dir,
        "stackrt_result",
        "cavity_spectra_20260609_121020_fft_solved_20260609_233951.npz",
    )

    # 光谱选择方式："all"、整数、列表或 slice。
    spectrum_selection = "all"

    config = {
        "FFT_PEAK_HEIGHT_RATIO": 0.2,
        "FFT_IGNORE_DC_BINS": 50,
        "FFT_PEAK_DISTANCE_BINS": 100,
        "ZERO_PAD_FACTOR": 8,
        "SAVE_FFT_MATRIX": False,
    }

    max_scatter_peaks = None
    show_interactive = True

    if mode in ("solve_only", "solve_and_plot"):
        fft_result_npz_path = BatchFFTSolver.solve_npz(
            npz_path=input_npz_path,
            output_path=None,
            config=config,
            spectrum_selection=spectrum_selection,
        )
        print(f"Saved FFT solving results: {fft_result_npz_path}")

    if mode in ("plot_only", "solve_and_plot"):
        plot_fft_result(
            npz_path=fft_result_npz_path,
            output_path=None,
            max_scatter_peaks=max_scatter_peaks,
            show_interactive=show_interactive,
        )


def main_direct_v2():
    """直接在代码中配置运行模式，不使用命令行参数。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # mode 可选：
    # "solve_only"          -> 只从原始光谱 npz 解算并保存 fft_solved npz
    # "plot_only"           -> 只读取已有 fft_solved npz 并绘图
    # "fit_only"            -> 只读取已有 fft_solved npz 并做线性拟合
    # "solve_and_plot"      -> 先解算，再绘图
    # "solve_and_fit"       -> 先解算，再做线性拟合
    # "solve_plot_and_fit"  -> 先解算，再绘图并做线性拟合
    mode = "fit_only"

    input_npz_path = os.path.join(script_dir, "stackrt_result", "scan_cavity_length_result","model7-1nm","cavity_spectra_20260616_235630.npz")
    fft_result_npz_path = os.path.join(
        script_dir,
        "stackrt_result", "scan_cavity_length_result","model7-1nm",
        "cavity_spectra_20260616_235630_fft_solved_20260617_110624.npz",
    )

    # 光谱选择方式："all"、整数、列表或 slice。
    spectrum_selection = "all"

    config = {
        "FFT_PEAK_HEIGHT_RATIO": 0.2,
        "FFT_IGNORE_DC_BINS": 50,
        "FFT_PEAK_DISTANCE_BINS": 100,
        "ZERO_PAD_FACTOR": 8,
        "SAVE_FFT_MATRIX": False,
    }

    max_scatter_peaks = None
    show_interactive = True
    fit_peak_type = "dominant"  # 可选 "dominant" 或 "first"

    if mode in ("solve_only", "solve_and_plot", "solve_and_fit", "solve_plot_and_fit"):
        fft_result_npz_path = BatchFFTSolver.solve_npz(
            npz_path=input_npz_path,
            output_path=None,
            config=config,
            spectrum_selection=spectrum_selection,
        )
        print(f"Saved FFT solving results: {fft_result_npz_path}")

    if mode in ("plot_only", "solve_and_plot", "solve_plot_and_fit"):
        plot_fft_result(
            npz_path=fft_result_npz_path,
            output_path=None,
            max_scatter_peaks=max_scatter_peaks,
            show_interactive=show_interactive,
        )

    if mode in ("fit_only", "solve_and_fit", "solve_plot_and_fit"):
        linear_fit_peak_distance(
            npz_path=fft_result_npz_path,
            peak_type=fit_peak_type,
            output_path=None,
            show_interactive=show_interactive,
        )


if __name__ == "__main__":
    main_direct_v2()
