import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List

# ==========================================
# 1. 路径与配置 (Path & Configuration)
# ==========================================
# 自动寻找最新的数据文件
RESULT_DIR = r"./stackrt_result"
IMG_OUTPUT_DIR = r"./img"
DATA_OUTPUT_DIR = r"./linear_fit"

# 确保输出目录存在
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

def get_latest_data():

    files = [f for f in os.listdir(RESULT_DIR) if f.startswith("dynamic_spectra") and f.endswith("155307.npz")]
    if not files:
        raise FileNotFoundError(f"在 {RESULT_DIR} 中未找到动态光谱数据文件。")
    files.sort()
    return os.path.join(RESULT_DIR, files[-1])

try:
    DATA_PATH = get_latest_data()
    print(f"📂 加载数据文件: {DATA_PATH}")
except Exception as e:
    print(f"❌ 错误: {e}")
    DATA_PATH = r"./stackrt_result/dynamic_spectra.npz"

# ==========================================
# 2. 理论模型定义 (Theoretical Model)
# ==========================================
def interference_model(wavelengths: np.ndarray, I_bg: float, I_amp: float, L_total: float) -> np.ndarray:
    """
    干涉光谱理论模型: I(lambda) = I_bg + I_amp * cos(4 * pi * L_total / lambda)
    """
    phase = 4.0 * np.pi * L_total / wavelengths
    return I_bg + I_amp * np.cos(phase)

# ==========================================
# 3. 数据加载与预处理 (Data Loading)
# ==========================================
def load_and_preprocess():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
        
    data = np.load(DATA_PATH)
    t_axis = data['t_axis']
    wavelengths_um = data['wavelengths']
    L_t_um = data['L_t']
    spectra = data['spectra']
    
    wavelengths_m = wavelengths_um * 1e-6
    L_t_m = L_t_um * 1e-6
    
    L0_true = np.mean(L_t_m)
    A_true = (np.max(L_t_m) - np.min(L_t_m)) / 2.0
    
    print(f"--- 数据基本信息 ---")
    print(f"时间步数: {len(t_axis)} | 波长采样点: {len(wavelengths_m)}")
    print(f"真值 L0: {L0_true * 1e6:.4f} um | 振幅 A: {A_true * 1e9:.2f} nm")
    
    return t_axis, wavelengths_m, L_t_m, spectra, L0_true, A_true

# ==========================================
# 4. 执行动态拟合 (Dynamic Fitting)
# ==========================================
def run_dynamic_demodulation():
    t_axis, wavelengths, L_t_true, spectra, L0_true, A_true = load_and_preprocess()
    
    n_steps = len(t_axis)
    L_recovered = np.zeros(n_steps)
    I_bg_list = np.zeros(n_steps)
    I_amp_list = np.zeros(n_steps)
    
    lower_bounds = [0.0, 0.0, L0_true - 10e-6]
    upper_bounds = [1.0, 1.0, L0_true + 10e-6]
    
    print(f"\n🚀 开始动态解调 (共 {n_steps} 步)...")
    
    # 初始步：从真值附近搜索起点
    try:
        y0 = spectra[0, :]
        popt0, _ = curve_fit(interference_model, wavelengths, y0, 
                             p0=[0.6, 0.4, L_t_true[0]],   # 是否应该从真值搜索，理论上不知道这个真值的大小，或者存在1um左右的误差
                             bounds=(lower_bounds, upper_bounds))
        p0 = popt0.tolist()
        print(f"✨ 初始帧对准完成: 提取 L0 = {p0[2]*1e6:.4f} um")
    except Exception as e:
        print(f"⚠️ 初始对准失败: {e}")
        p0 = [0.6, 0.4, L0_true]

    for i in range(n_steps):
        y_data = spectra[i, :]
        try:
            popt, _ = curve_fit(interference_model, wavelengths, y_data, 
                               p0=p0, bounds=(lower_bounds, upper_bounds),
                               ftol=1e-8, xtol=1e-8)
            I_bg_list[i], I_amp_list[i], L_recovered[i] = popt
            p0 = popt.tolist()  # 传递给下一帧
            
            if i % 20 == 0:
                true_L = L_t_true[i] * 1e6
                fit_L = L_recovered[i] * 1e6
                print(f"进度: {i:3d}/{n_steps} | 真值: {true_L:9.4f} um | 提取: {fit_L:9.4f} um | 误差: {(fit_L-true_L)*1e3:7.2f} nm")
        except Exception as e:
            print(f"⚠️ 步 {i} 拟合失败: {e}")
            L_recovered[i] = L_recovered[i-1] if i > 0 else L0_true
            
    # --------- 结果分析 ---------
    errors_nm = (L_recovered - L_t_true) * 1e9
    mean_offset = np.mean(errors_nm)
    detrended_errors = errors_nm - mean_offset
    rmse_dynamic = np.sqrt(np.mean(detrended_errors**2))
    total_rmse = np.sqrt(np.mean(errors_nm**2))
    
    print(f"\n✅ 动态解调完成!")
    print(f"平均偏置 (Systematic Offset): {mean_offset:.4f} nm")
    print(f"动态跟踪误差 (Detrended RMSE): {rmse_dynamic:.4f} nm")
    print(f"总 RMSE: {total_rmse:.4f} nm")
    
    # --------- 保存结果 ---------
    results_file = os.path.join(DATA_OUTPUT_DIR, "dynamic_fitting_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=== 动态非线性最小二乘法解调结果 ===\n")
        f.write(f"平均偏置: {mean_offset:.4f} nm\n")
        f.write(f"动态跟踪误差 (RMSE): {rmse_dynamic:.4f} nm\n")
        f.write(f"总误差 (Total RMSE): {total_rmse:.4f} nm\n")

    # --------- 可视化 ---------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(t_axis * 1000, L_t_true * 1e6, 'k-', label='Ground Truth', linewidth=2)
    ax1.plot(t_axis * 1000, L_recovered * 1e6, 'r--', label='Recovered (LSQ)', linewidth=1.5)
    ax1.set_ylabel("Cavity Length (um)")
    ax1.set_title(f"Dynamic Motion Tracking\nDetrended RMSE: {rmse_dynamic:.3f} nm")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    ax2.plot(t_axis * 1000, detrended_errors, 'g-', label='Tracking Error (Detrended)')
    ax2.axhline(0, color='red', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Error (nm)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(IMG_OUTPUT_DIR, "demodulate_lsq_dynamic.png")
    plt.savefig(plot_path, dpi=200)
    print(f"📊 动态跟踪图已保存: {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_dynamic_demodulation()
