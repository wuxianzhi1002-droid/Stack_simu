import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 优先使用可弹出交互窗口的 Matplotlib 后端。
for backend in ("TkAgg", "QtAgg", "WxAgg"):
    try:
        matplotlib.use(backend, force=True)
        break
    except Exception:
        pass


CONFIG = {
    # 输入文件：这里应填写 solve_npz_fft.py 输出的 fft_solved npz。
    "FFT_SOLVED_NPZ": r"./stackrt_result/scan_cavity_length_result/model5/cavity_spectra_20260610_132100_fft_solved_20260610_132143.npz",

    # 拟合目标："dominant" 使用 dominant_peak_distance_um；"first" 使用 first_peak_distance_um。
    "PEAK_TYPE": "dominant",

    # 输出目录。None 表示保存到输入 npz 同目录。
    "OUTPUT_DIR": None,

    # 是否弹出交互窗口。
    "SHOW_INTERACTIVE": True,

    # 是否保存拟合统计数据 npz。
    "SAVE_RESULT_NPZ": True,

    # 外部线性拟合模型，用于鲁棒性测试。
    # 例如从其他数据集拟合得到 y = slope * x + intercept 后，填入这里评估它在当前数据上的误差。
    "EXTERNAL_MODELS": [
         {"name": "model1", "slope": 1.00069168963, "intercept": 0.266975874558},
    ],

    # 随机训练/测试鲁棒性测试。
    "RUN_RANDOM_SPLIT_TEST": False,
    "RANDOM_SEED": 42,
    "N_SPLITS": 100,
    "TRAIN_RATIO": 0.7,
}


def read_scalar_string(npz_data, key, default=""):
    """从 npz 中读取标量字符串字段。"""
    if key not in npz_data.files:
        return default
    value = npz_data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def scan_axis_label(scan_axis_name):
    """根据扫描轴类型生成坐标轴标签。"""
    labels = {
        "angle_deg": "Incident angle (deg)",
        "cavity_um": "Cavity length (um)",
        "cavity_m": "Cavity length (m)",
        "time_or_scan_axis": "Time / scan axis",
        "spectrum_index": "Spectrum index",
    }
    return labels.get(scan_axis_name, scan_axis_name)


def load_fit_data(npz_path, peak_type):
    """读取 fft_solved npz，并提取线性拟合所需的 x/y 数据。"""
    data = np.load(npz_path, allow_pickle=True)
    scan_axis_name = read_scalar_string(data, "scan_axis_name", "scan_axis")

    required = {"scan_axis", "dominant_peak_distance_um", "first_peak_distance_um"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"{npz_path} 缺少必要字段: {missing}")

    x_raw = np.asarray(data["scan_axis"], dtype=float).reshape(-1)
    if peak_type == "dominant":
        y = np.asarray(data["dominant_peak_distance_um"], dtype=float).reshape(-1)
        y_label = "Dominant peak distance (um)"
    elif peak_type == "first":
        y = np.asarray(data["first_peak_distance_um"], dtype=float).reshape(-1)
        y_label = "First peak distance (um)"
    else:
        raise ValueError("PEAK_TYPE must be 'dominant' or 'first'.")

    # y 的单位是 um。若 x 是 m，则转为 um，避免拟合斜率单位混乱。
    if scan_axis_name == "cavity_m":
        x = x_raw * 1e6
        x_label = "Cavity length (um)"
    else:
        x = x_raw
        x_label = scan_axis_label(scan_axis_name)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size < 2:
        raise ValueError("至少需要 2 个有效点才能做线性拟合。")

    return {
        "x": x,
        "y": y,
        "x_label": x_label,
        "y_label": y_label,
        "scan_axis_name": scan_axis_name,
        "source_npz": read_scalar_string(data, "source_npz", ""),
        "selected_spectrum_indices": (
            np.asarray(data["selected_spectrum_indices"], dtype=int)
            if "selected_spectrum_indices" in data.files
            else np.arange(x.size)
        ),
    }


def evaluate_linear_model(x, y, slope, intercept):
    """评估线性模型 y = slope*x + intercept 的拟合误差。"""
    y_pred = slope * x + intercept
    residual = y - y_pred

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r_value = float(np.corrcoef(y, y_pred)[0, 1]) if np.std(y_pred) > 0 and np.std(y) > 0 else np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": r_value,
        "r_squared": float(r_squared),
        "rmse_um": float(np.sqrt(np.mean(residual**2))),
        "mae_um": float(np.mean(np.abs(residual))),
        "max_abs_error_um": float(np.max(np.abs(residual))),
        "residual": residual,
        "y_pred": y_pred,
    }


def fit_current_data(x, y):
    """对当前数据做最小二乘线性拟合。"""
    slope, intercept = np.polyfit(x, y, 1)
    return evaluate_linear_model(x, y, slope, intercept)


def random_split_test(x, y, n_splits, train_ratio, random_seed):
    """随机训练/测试拆分，用于评估拟合结果对样本选择的鲁棒性。"""
    rng = np.random.default_rng(random_seed)
    n = x.size
    train_size = int(round(n * train_ratio))
    train_size = min(max(train_size, 2), n - 1)

    slopes = []
    intercepts = []
    test_rmse = []
    test_mae = []
    test_max_abs_error = []

    for _ in range(n_splits):
        perm = rng.permutation(n)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        slope, intercept = np.polyfit(x[train_idx], y[train_idx], 1)
        metrics = evaluate_linear_model(x[test_idx], y[test_idx], slope, intercept)

        slopes.append(slope)
        intercepts.append(intercept)
        test_rmse.append(metrics["rmse_um"])
        test_mae.append(metrics["mae_um"])
        test_max_abs_error.append(metrics["max_abs_error_um"])

    return {
        "slopes": np.asarray(slopes),
        "intercepts": np.asarray(intercepts),
        "test_rmse_um": np.asarray(test_rmse),
        "test_mae_um": np.asarray(test_mae),
        "test_max_abs_error_um": np.asarray(test_max_abs_error),
    }


def summarize_random_split(random_result):
    """汇总随机拆分鲁棒性测试结果。"""
    if random_result is None:
        return {}

    return {
        "slope_mean": float(np.mean(random_result["slopes"])),
        "slope_std": float(np.std(random_result["slopes"])),
        "intercept_mean": float(np.mean(random_result["intercepts"])),
        "intercept_std": float(np.std(random_result["intercepts"])),
        "test_rmse_mean_um": float(np.mean(random_result["test_rmse_um"])),
        "test_rmse_std_um": float(np.std(random_result["test_rmse_um"])),
        "test_mae_mean_um": float(np.mean(random_result["test_mae_um"])),
        "test_mae_std_um": float(np.std(random_result["test_mae_um"])),
        "test_max_abs_error_mean_um": float(np.mean(random_result["test_max_abs_error_um"])),
    }


def plot_fit_and_robustness(
    fit_data,
    current_metrics,
    external_metrics,
    random_result,
    output_path,
    show_interactive,
):
    """绘制当前拟合、外部模型残差和随机拆分鲁棒性测试结果。"""
    x = fit_data["x"]
    y = fit_data["y"]
    x_line = np.linspace(np.min(x), np.max(x), 500)

    n_rows = 3 if random_result is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 4.2 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    ax_fit = axes[0]
    ax_residual = axes[1]

    ax_fit.scatter(x, y, s=18, alpha=0.75, label="FFT solved distance")
    ax_fit.plot(
        x_line,
        current_metrics["slope"] * x_line + current_metrics["intercept"],
        color="#d62728",
        lw=1.5,
        label=(
            f"Current fit: y={current_metrics['slope']:.8g}x"
            f"+{current_metrics['intercept']:.8g}"
        ),
    )

    for metrics in external_metrics:
        ax_fit.plot(
            x_line,
            metrics["slope"] * x_line + metrics["intercept"],
            lw=1.0,
            ls="--",
            label=f"External: {metrics['name']}",
        )

    text = (
        f"Current fit\n"
        f"R = {current_metrics['r_value']:.8f}\n"
        f"R^2 = {current_metrics['r_squared']:.8f}\n"
        f"RMSE = {current_metrics['rmse_um']:.6g} um\n"
        f"MAE = {current_metrics['mae_um']:.6g} um\n"
        f"Max |error| = {current_metrics['max_abs_error_um']:.6g} um"
    )
    ax_fit.text(
        0.02,
        0.98,
        text,
        transform=ax_fit.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax_fit.set_title("Linear Fit of FFT Solved Peak Distance")
    ax_fit.set_xlabel(fit_data["x_label"])
    ax_fit.set_ylabel(fit_data["y_label"])
    ax_fit.grid(True)
    ax_fit.legend()

    ax_residual.axhline(0, color="#555555", lw=0.9, ls="--")
    ax_residual.plot(x, current_metrics["residual"], "o-", ms=3, lw=1, label="Current fit residual")
    for metrics in external_metrics:
        ax_residual.plot(x, metrics["residual"], ".-", ms=3, lw=0.9, label=f"{metrics['name']} residual")
    ax_residual.set_title("Fit Error / Residual")
    ax_residual.set_xlabel(fit_data["x_label"])
    ax_residual.set_ylabel("Solved - fitted (um)")
    ax_residual.grid(True)
    ax_residual.legend()

    if random_result is not None:
        ax_random = axes[2]
        ax_random.hist(random_result["test_rmse_um"], bins=24, color="#4c78a8", alpha=0.75)
        ax_random.axvline(
            np.mean(random_result["test_rmse_um"]),
            color="#d62728",
            lw=1.2,
            label="Mean test RMSE",
        )
        ax_random.set_title("Random Split Robustness: Test RMSE Distribution")
        ax_random.set_xlabel("Test RMSE (um)")
        ax_random.set_ylabel("Count")
        ax_random.grid(True, axis="y")
        ax_random.legend()

    fig.savefig(output_path, dpi=220)
    print(f"Saved linear fit robustness figure: {output_path}")

    if show_interactive:
        plt.show(block=True)
    else:
        plt.close(fig)


def save_result_npz(output_path, fit_data, current_metrics, external_metrics, random_result, random_summary):
    """保存线性拟合和鲁棒性测试结果。"""
    save_kwargs = {
        "x": fit_data["x"],
        "y": fit_data["y"],
        "x_label": np.array(fit_data["x_label"]),
        "y_label": np.array(fit_data["y_label"]),
        "scan_axis_name": np.array(fit_data["scan_axis_name"]),
        "current_slope": np.array(current_metrics["slope"]),
        "current_intercept": np.array(current_metrics["intercept"]),
        "current_r_value": np.array(current_metrics["r_value"]),
        "current_r_squared": np.array(current_metrics["r_squared"]),
        "current_rmse_um": np.array(current_metrics["rmse_um"]),
        "current_mae_um": np.array(current_metrics["mae_um"]),
        "current_max_abs_error_um": np.array(current_metrics["max_abs_error_um"]),
        "current_residual_um": current_metrics["residual"],
    }

    if external_metrics:
        save_kwargs["external_names"] = np.array([m["name"] for m in external_metrics])
        save_kwargs["external_slope"] = np.array([m["slope"] for m in external_metrics])
        save_kwargs["external_intercept"] = np.array([m["intercept"] for m in external_metrics])
        save_kwargs["external_rmse_um"] = np.array([m["rmse_um"] for m in external_metrics])
        save_kwargs["external_mae_um"] = np.array([m["mae_um"] for m in external_metrics])
        save_kwargs["external_max_abs_error_um"] = np.array([m["max_abs_error_um"] for m in external_metrics])

    if random_result is not None:
        for key, value in random_result.items():
            save_kwargs[f"random_{key}"] = value
        for key, value in random_summary.items():
            save_kwargs[f"random_summary_{key}"] = np.array(value)

    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved linear fit robustness data: {output_path}")


def build_report_lines(current_metrics, external_metrics, random_summary):
    """生成线性拟合参数和鲁棒性测试报告文本。"""
    lines = [
        "[Current fit]",
        f"slope = {current_metrics['slope']:.12g}",
        f"intercept = {current_metrics['intercept']:.12g}",
        f"R = {current_metrics['r_value']:.12g}",
        f"R^2 = {current_metrics['r_squared']:.12g}",
        f"RMSE = {current_metrics['rmse_um']:.12g} um",
        f"MAE = {current_metrics['mae_um']:.12g} um",
        f"Max |error| = {current_metrics['max_abs_error_um']:.12g} um",
    ]

    for metrics in external_metrics:
        lines.extend(
            [
                "",
                f"[External model: {metrics['name']}]",
                f"slope = {metrics['slope']:.12g}",
                f"intercept = {metrics['intercept']:.12g}",
                f"R = {metrics['r_value']:.12g}",
                f"R^2 = {metrics['r_squared']:.12g}",
                f"RMSE = {metrics['rmse_um']:.12g} um",
                f"MAE = {metrics['mae_um']:.12g} um",
                f"Max |error| = {metrics['max_abs_error_um']:.12g} um",
            ]
        )

    if random_summary:
        lines.extend(["", "[Random split robustness]"])
        for key, value in random_summary.items():
            lines.append(f"{key} = {value:.12g}")

    return lines


def save_report_txt(output_path, report_lines):
    """保存线性拟合参数和鲁棒性测试报告到 txt。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved linear fit robustness report: {output_path}")


def default_output_paths(input_npz_path, output_dir):
    """生成默认输出路径。"""
    folder = output_dir or os.path.dirname(os.path.abspath(input_npz_path))
    stem = os.path.splitext(os.path.basename(input_npz_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = os.path.join(folder, f"{stem}_linear_fit_robustness_{timestamp}.png")
    data_path = os.path.join(folder, f"{stem}_linear_fit_robustness_{timestamp}.npz")
    report_path = os.path.join(folder, f"{stem}_linear_fit_robustness_{timestamp}.txt")
    return figure_path, data_path, report_path


def run_linear_fit_robustness(config):
    """主流程：拟合当前数据，并用外部模型和随机拆分测试鲁棒性。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = config["FFT_SOLVED_NPZ"]
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(script_dir, npz_path)

    output_dir = config["OUTPUT_DIR"]
    if output_dir is not None and not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    fit_data = load_fit_data(npz_path, config["PEAK_TYPE"])
    current_metrics = fit_current_data(fit_data["x"], fit_data["y"])

    external_metrics = []
    for item in config["EXTERNAL_MODELS"]:
        metrics = evaluate_linear_model(
            fit_data["x"],
            fit_data["y"],
            float(item["slope"]),
            float(item["intercept"]),
        )
        metrics["name"] = item.get("name", f"external_{len(external_metrics) + 1}")
        external_metrics.append(metrics)

    random_result = None
    random_summary = {}
    if config["RUN_RANDOM_SPLIT_TEST"]:
        random_result = random_split_test(
            fit_data["x"],
            fit_data["y"],
            int(config["N_SPLITS"]),
            float(config["TRAIN_RATIO"]),
            int(config["RANDOM_SEED"]),
        )
        random_summary = summarize_random_split(random_result)

    figure_path, data_path, report_path = default_output_paths(npz_path, output_dir)
    plot_fit_and_robustness(
        fit_data,
        current_metrics,
        external_metrics,
        random_result,
        figure_path,
        bool(config["SHOW_INTERACTIVE"]),
    )

    if config["SAVE_RESULT_NPZ"]:
        save_result_npz(data_path, fit_data, current_metrics, external_metrics, random_result, random_summary)

    report_lines = build_report_lines(current_metrics, external_metrics, random_summary)
    save_report_txt(report_path, report_lines)

    print("")
    for line in report_lines:
        print(line)

    return {
        "figure_path": figure_path,
        "data_path": data_path if config["SAVE_RESULT_NPZ"] else None,
        "report_path": report_path,
        "current_metrics": current_metrics,
        "external_metrics": external_metrics,
        "random_summary": random_summary,
    }


def main():
    run_linear_fit_robustness(CONFIG)


if __name__ == "__main__":
    main()
