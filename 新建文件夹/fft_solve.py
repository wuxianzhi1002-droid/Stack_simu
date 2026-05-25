import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import find_peaks


def process_sdi_signal_from_csv(csv_file):
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    waves = data[:, 0]
    intensities = data[:, 1]

    # ===== 1. k域 =====
    k_raw = 2 * np.pi / waves
    k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))
    i_linear = np.interp(k_linear, k_raw[::-1], intensities[::-1])

    # ===== 2. 去直流 + 加窗 =====
    i_detrend = i_linear - np.mean(i_linear)
    i_windowed = i_detrend * np.hanning(len(i_detrend))

    # ===== 3. FFT =====
    fft_data = np.abs(np.fft.rfft(i_windowed))

    # 深度分辨率相关的物理长度
    # delta_k 是 k 域的步长
    dk = k_linear[1] - k_linear[0]
    max_range = np.pi / dk
    depth_axis = np.linspace(0, max_range, len(fft_data))

    # ===== 🔥 5. 找多个峰 =====
    ignore = 0  # 去掉DC附近
    peaks, properties = find_peaks(
        fft_data[ignore:],
        height=np.max(fft_data) * 0.05,   # 阈值（可调）
        distance=20                      # 峰间距（防止抖动）
    )

    peaks = peaks + ignore
    peak_depths = depth_axis[peaks]
    peak_heights = properties["peak_heights"]

    # # ===== 🔥 6. 选两个最强峰 =====
    # if len(peaks) < 2:
    #     print("⚠️ 未检测到两个明显峰，请检查数据")
    #     return None
    #
    # # 按幅值排序
    # sorted_idx = np.argsort(peak_heights)[::-1]
    # top2 = sorted_idx[:2]
    #
    # z1, z2 = peak_depths[top2]

    # ===== 🔥 6. 选 z 最小的两个峰 =====
    if len(peaks) < 2:
        print("⚠️ 未检测到两个明显峰，请检查数据")
        return None

    # 按 z 从小到大排序
    sorted_idx = np.argsort(peak_depths)

    # 取最小的两个
    z1 = peak_depths[sorted_idx[0]]
    z2 = peak_depths[sorted_idx[1]]

    # 排序（小的作为 d，大的作为 L）
    d_opd = min(z1, z2)
    L_opd = max(z1, z2)

    # ===== 🔥 7. 转换为实际腔长 =====
    d = d_opd / 2
    L = L_opd / 2

    print("\n===== 识别结果 =====")
    print(f"小腔 d = {d:.6f} um")
    print(f"大腔 L = {L:.6f} um")

    # ===== 8. 绘图 =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(waves * 1000, intensities)
    plt.title("Spectral Interference")

    plt.subplot(1, 2, 2)
    plt.plot(depth_axis, fft_data)
    plt.scatter(depth_axis[peaks], fft_data[peaks], marker='o')
    plt.axvline(d_opd, linestyle='--', label="d")
    plt.axvline(L_opd, linestyle='--', label="L")
    plt.legend()
    plt.title("FFT Peaks")

    plt.tight_layout()
    plt.show()

    return L, d


# ================== MAIN ==================
if __name__ == "__main__":

    # 👉 改成你的实际文件名
    csv_file = "PC2_sdi_final_0413_results_id1.csv"

    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
    else:
        L, d = process_sdi_signal_from_csv(csv_file)
        if L / d > 3:
            print("识别正常：大腔 vs 小腔")
            print(f"\n大腔长 = {L:.6f} um")
            print(f"小腔长 = {d:.6f} um")
        else:
            print("⚠️ 可能识别错误（峰混叠）")
