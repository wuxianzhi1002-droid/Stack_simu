import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing import Tuple, List

# ==========================================
# 1. 路径与配置 (Path & Configuration)
# ==========================================
RESULT_DIR = r"./stackrt_result"
IMG_OUTPUT_DIR = r"./img"
DATA_OUTPUT_DIR = r"./linear_fit"

os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

def get_latest_data():
    if not os.path.exists(RESULT_DIR):
        alt_dir = "./stackrt_result"
        if os.path.exists(alt_dir):
            files = [f for f in os.listdir(alt_dir) if f.startswith("dynamic_spectra") and f.endswith(".npz")]
            if files:
                files.sort()
                return os.path.join(alt_dir, files[-1])
        raise FileNotFoundError(f"未找到结果目录: {RESULT_DIR}")
        
    files = [f for f in os.listdir(RESULT_DIR) if f.startswith("dynamic_spectra") and f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"在 {RESULT_DIR} 中未找到数据文件。")
    files.sort()
    return os.path.join(RESULT_DIR, files[-1])

# ==========================================
# 2. 算法核心：粗寻 + 精拟 (Coarse + Fine Demodulation)
# ==========================================

def coarse_estimate_length(wavelengths: np.ndarray, spectrum: np.ndarray, peak_ratio = 0.2) -> float:
    """
    使用 FFT (快速傅里叶变换) 估算绝对腔长 L。
    原理：干涉条纹在波数空间 (k=1/lambda) 是等间距的。
    """
    # 1. 转换为波数空间 (k)
    k = 1.0 / wavelengths  # um^-1
    
    # 2. 等间距重采样 (FFT 要求 x 轴等间距)
    k_uniform = np.linspace(k.min(), k.max(), len(k))
    f_interp = interp1d(k, spectrum, kind='cubic')
    s_uniform = f_interp(k_uniform)
    
    # 3. 去直流分量并加窗
    s_uniform = (s_uniform - np.mean(s_uniform)) * np.hanning(len(s_uniform))
    
    # 4. FFT
    n_fft = len(s_uniform) * 4 # 补零以提高频率分辨率
    fft_res = np.abs(np.fft.rfft(s_uniform, n=n_fft))
    
    # 5. 频率轴对应到 2*L (光程差)
    # k 的采样间隔 dk
    dk = np.abs(k_uniform[1] - k_uniform[0])
    # FFT 频率轴对应的空间步长
    max_range = np.pi / dk
    distance_axis = np.linspace(0, max_range / 2, len(fft_res))
    
    # 6. 找峰值 (2*L = peak_pos)
    ignore = 50
    peaks, _ = find_peaks(
        fft_res[ignore:],
        height=np.max(fft_res[ignore:]) * peak_ratio,
        distance=100
    )
    peaks = peaks + ignore
    l_coarse = distance_axis[peaks] / 2.0  # 单位: um
    
    # pyrefly: ignore [bad-return]
    return l_coarse * 1e-6 # 返回米

def interference_model(wavelengths: np.ndarray, I_bg: float, I_amp: float, L_total: float) -> np.ndarray:
    """
    干涉光谱理论模型
    """
    phase = 4.0 * np.pi * L_total / wavelengths
    return I_bg + I_amp * np.cos(phase)

# ==========================================
# 3. 主程序
# ==========================================
def run_robust_demodulation():
    try:
        data_path = get_latest_data()
        print(f"📂 加载数据: {data_path}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    data = np.load(data_path)
    t_axis = data['t_axis']
    wavelengths_m = data['wavelengths'] * 1e-6
    spectra = data['spectra']
    L_t_true = data['L_t'] * 1e-6
    
    n_steps = len(t_axis)
    L_recovered = np.zeros(n_steps)
    
    print("\n🔍 正在进行第一帧的粗寻 (FFT Coarse Search)...")
    # 第一帧不使用任何真值信息，完全独立估算
    L_start_coarse = coarse_estimate_length(wavelengths_m * 1e6, spectra[0, :])
    print(f"✨ 粗寻结果: {L_start_coarse*1e6:.4f} um (真值: {L_t_true[0]*1e6:.4f} um)")

    # 初始步精拟合
    try:
        popt, _ = curve_fit(interference_model, wavelengths_m, spectra[0, :],
                           p0=[0.5, 0.3, L_start_coarse],
                           bounds=([0, 0, L_start_coarse-50e-6], [1, 1, L_start_coarse+50e-6]))
        p0 = popt.tolist()
        print(f"🎯 第一帧对准完成: {p0[2]*1e6:.4f} um")
    except Exception as e:
        print(f"❌ 初始对准失败: {e}")
        p0 = [0.5, 0.3, L_start_coarse]

    print(f"\n🚀 开始动态跟踪 (共 {n_steps} 步)...")
    for i in range(n_steps):
        try:
            popt, _ = curve_fit(interference_model, wavelengths_m, spectra[i, :],
                               p0=p0, bounds=([0, 0, p0[2]-5e-6], [1, 1, p0[2]+5e-6]),
                               ftol=1e-8, xtol=1e-8)
            L_recovered[i] = popt[2]
            p0 = popt.tolist() # 连续跟踪
            
            if i % 25 == 0:
                err = (L_recovered[i] - L_t_true[i]) * 1e9
                print(f"进度: {i:3d}/{n_steps} | L: {L_recovered[i]*1e6:9.4f} um | 实时误差: {err:7.2f} nm")
        except Exception as e:
            print(f"⚠️ 步 {i} 跟踪丢失: {e}")
            L_recovered[i] = L_recovered[i-1] if i > 0 else p0[2]

    # --------- 统计与分析 ---------
    errors_nm = (L_recovered - L_t_true) * 1e9
    mean_offset = np.mean(errors_nm)
    rmse_dynamic = np.sqrt(np.mean((errors_nm - mean_offset)**2))
    
    print(f"\n✅ 鲁棒解调完成!")
    print(f"系统偏置: {mean_offset:.4f} nm")
    print(f"动态精度 (Detrended RMSE): {rmse_dynamic:.4f} nm")

    # --------- 可视化 ---------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(t_axis*1000, L_t_true*1e6, 'k-', label='Ground Truth', alpha=0.3)
    ax1.plot(t_axis*1000, L_recovered*1e6, 'r--', label='Robust Recovered', linewidth=1.5)
    ax1.set_title(f"Ground-Truth-Free Dynamic Tracking\nRMSE: {rmse_dynamic:.3f} nm")
    ax1.set_ylabel("Cavity Length (um)")
    ax1.legend()
    ax1.grid(True, ls=':')

    ax2.plot(t_axis*1000, errors_nm - mean_offset, 'g-', label='Relative Tracking Error')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Error (nm)")
    ax2.legend()
    ax2.grid(True, ls=':')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_DIR, "robust_demodulate_dynamic.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    run_robust_demodulation()
