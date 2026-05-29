import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks


def process_sdi_signal_from_csv(csv_file):
    # 自动识别分隔符（兼容逗号和Tab）
    try:
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    except Exception:
        data = np.loadtxt(csv_file, delimiter="\t", skiprows=1)

    waves = data[:, 0]
    intensities = data[:, 1]

    # ===== 1. k域 线性化 =====
    k_raw = 2 * np.pi / waves
    k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))
    # 确保插网格时x轴单调递增
    if k_raw[0] > k_raw[-1]:
        i_linear = np.interp(k_linear, k_raw[::-1], intensities[::-1])
    else:
        i_linear = np.interp(k_linear, k_raw, intensities)

    # ===== 2. 去直流 + 加窗 =====
    i_detrend = i_linear - np.mean(i_linear)
    i_windowed = i_detrend * np.hanning(len(i_detrend))

    # ===== 3. FFT =====
    fft_data = np.abs(np.fft.rfft(i_windowed))

    # ===== 4. 深度轴计算 (单位: um) =====
    dk = k_linear[1] - k_linear[0]
    max_range = np.pi / dk
    depth_axis = np.linspace(0, max_range, len(fft_data))
    distance_axis = depth_axis / 2  # 物理距离轴 = 光程差 / 2

    # ===== 5. 寻找所有可能的峰 =====
    ignore = 5  # 略微跳过极低频的DC残余分量
    peaks, properties = find_peaks(
        fft_data[ignore:],
        height=np.max(fft_data) * 0.01,  # 门槛调低到 1%，防止漏掉微弱的第二个腔面反射
        distance=10  # 适当缩短间距限制
    )
    peaks = peaks + ignore
    peak_depths = depth_axis[peaks]
    peak_distances = distance_axis[peaks]

    # 提取高度数据（修复原有代码中直接读取字典可能因为未定义高度而报错的问题）
    peak_heights = fft_data[peaks]

    # ===== 6. 健壮的峰值选择逻辑（保证不崩溃、必定绘图） =====
    d, L = 0.0, 0.0
    # d_opd, L_opd = 0.0, 0.0 # 不再直接使用 OPD 标注，改用物理距离 d, L

    if len(peaks) >= 2:
        # 如果找到了至少两个峰，按位置从小到大排序
        sorted_idx = np.argsort(peak_distances)
        d = peak_distances[sorted_idx[0]]
        L = peak_distances[sorted_idx[1]]
        
        print("\n✅ [成功] 识别到两个主要物理腔长:")
        print(f"小腔 d = {d:.6f} um")
        print(f"大腔 L = {L:.6f} um")
    elif len(peaks) == 1:
        d = peak_distances[0]
        print(f"\n⚠️ [警告] 只检测到了 1 个峰：腔长 = {d:.6f} um，另一个腔可能太小或超出了量程。")
    else:
        print("\n❌ [严重警告] 未检测到任何明显的干涉峰！请通过右图检查光谱是否正确或数据是否欠采样。")

    # ===== 7. 强行绘图 (不管结果如何都展示) =====
    plt.figure(figsize=(12, 5))

    # 左图：原始光谱
    plt.subplot(1, 2, 1)
    # 如果波长是 um 级，转成 nm 更好看
    if np.max(waves) < 1e-3:
        plt.plot(waves * 1e9, intensities, color='b')
        plt.xlabel("Wavelength (nm)")
    else:
        plt.plot(waves, intensities, color='b')
        plt.xlabel("Wavelength (um)")
    plt.ylabel("Intensity")
    plt.title("Spectral Interference (Raw Data)")
    plt.grid(True)

    # 右图：FFT 深度域图像（物理距离）
    plt.subplot(1, 2, 2)
    plt.plot(distance_axis, fft_data, color='k', label='FFT Spectrum')
    if len(peaks) > 0:
        plt.scatter(distance_axis[peaks], fft_data[peaks], marker='o', color='r', s=40, label='Detected Peaks')

    if d > 0: plt.axvline(d, color='g', linestyle='--', label=f"d ({d:.1f} um)")
    if L > 0: plt.axvline(L, color='m', linestyle='--', label=f"L ({L:.1f} um)")

    plt.xlabel("Physical Distance / Thickness (um)")
    plt.ylabel("FFT Amplitude")
    plt.title("FFT Spatial Domain (Peaks Search)")
    plt.xlim(0, max_range / 2)  # 锁定显示最大物理距离范围
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return L, d


# ================== MAIN ==================
if __name__ == "__main__":
    csv_file = "../stackrt_result/raw_stack_length1mm.csv"

    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
    else:
        # 即使找不够峰，现在函数也会返回 (0, 0) 而不是 None，这里不会再报错
        L, d = process_sdi_signal_from_csv(csv_file)

        if L > 0 and d > 0:
            if L / d > 3:
                print("\n识别状态：正常 (大腔 vs 小腔 比例明确)")
            else:
                print("\n识别状态：⚠️ 两个峰值靠得太近，可能属于同一个腔的干涉纹理或发生了混叠。")
        else:
            print("\n识别状态：❌ 失败。请参考弹出的 FFT 图形调整找峰参数。")