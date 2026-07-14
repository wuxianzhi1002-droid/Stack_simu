import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple

# ==========================================
# 1. 路径与配置 (Path & Configuration)
# ==========================================
# DATA_PATH = r"./stackrt_result/dynamic_spectra.npz"
DATA_PATH = r"./stackrt_result/dynamic_spectra_20260601_155307.npz"
IMG_OUTPUT_DIR = r"./img"
DATA_OUTPUT_DIR = r"./linear_fit"

# 确保输出目录存在
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 理论模型定义 (Theoretical Model)
# ==========================================
def interference_model(wavelengths: np.ndarray, I_bg: float, I_amp: float, L_0: float, delta_L: float) -> np.ndarray:
    """
    干涉光谱理论模型: I(lambda) = I_bg + I_amp * cos(4 * pi * (L_0 + delta_L) / lambda)
    
    参数:
    - wavelengths: 波长数组 (m)
    - I_bg: 背景强度
    - I_amp: 干涉振幅
    - L_0: 待拟合的初始绝对腔长 (m)
    - delta_L: 已知的动态位移项 (m)
    """
    # 这里的 L_0 + delta_L 是当前时刻的总腔长
    phase = 4.0 * np.pi * (L_0 + delta_L) / wavelengths
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
    
    # 转换为标准单位 (米)
    wavelengths_m = wavelengths_um * 1e-6
    L_t_m = L_t_um * 1e-6
    
    # 提取 Ground Truth
    # L_t = L0 + A * sin(2*pi*f*t) -> L0 是均值
    L0_true = np.mean(L_t_m)
    
    # 计算调制振幅 A (峰谷值的一半)
    A_true = (np.max(L_t_m) - np.min(L_t_m)) / 2.0
    
    print(f"--- 数据基本信息 ---")
    print(f"光谱形状: {spectra.shape}")
    print(f"波长范围: {np.min(wavelengths_um):.3f} - {np.max(wavelengths_um):.3f} um")
    print(f"真值 L0: {L0_true * 1e6:.4f} um")
    print(f"调制振幅 A: {A_true * 1e9:.2f} nm")
    
    return t_axis, wavelengths_m, L_t_m, spectra, L0_true, A_true

# ==========================================
# 4. 执行拟合 (Fitting)
# ==========================================
def run_demodulation():
    t_axis, wavelengths, L_t, spectra, L0_true, A_true = load_and_preprocess()
    
    # 选择 t=0 的切片进行拟合
    # t=0 时, delta_L = A * sin(0) = 0
    # 但为了通用性，我们计算 delta_L[0]
    idx = 0
    delta_L_curr = L_t[idx] - L0_true
    y_data = spectra[idx, :]
    
    # 初始猜测 (p0)
    # I_bg: 取光谱均值
    # I_amp: 取一个经验值
    # L_0: 按照任务要求，使用带有小偏差的真值作为 p0
    # 由于余弦函数的多峰性，初始猜测对收敛到正确条纹至关重要
    p0 = [np.mean(y_data), 0.1, L0_true+0.1e-6] # 偏差 0.5 um

    # 设定边界 (bounds)
    # lower_bounds = [0.0, 0.0, L0_true * 0.95]
    # upper_bounds = [1.0, 1.0, L0_true * 1.05]
    lower_bounds = [0.0, 0.0, L0_true-1e6]
    upper_bounds = [1.0, 1.0, L0_true+1e6]
    print(f"\n🚀 正在对 t={t_axis[idx]*1000:.2f} ms 的光谱进行非线性最小二乘拟合...")
    
    # 包装目标函数，固定 delta_L
    def fit_func(w, I_bg, I_amp, L_0):
        return interference_model(w, I_bg, I_amp, L_0, delta_L_curr)
    
    try:
        popt, pcov = curve_fit(fit_func, wavelengths, y_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        I_bg_fit, I_amp_fit, L0_fit = popt
        
        # 误差分析
        error_m = L0_fit - L0_true
        print(f"\n✅ 拟合完成!")
        print(f"提取结果 I_bg: {I_bg_fit:.4f}")
        print(f"提取结果 I_amp: {I_amp_fit:.4f}")
        print(f"提取结果 L0: {L0_fit * 1e6:.6f} um")
        print(f"绝对误差: {error_m * 1e9:.4f} nm")
        
        # --------- 保存结果 ---------
        results_file = os.path.join(DATA_OUTPUT_DIR, "fitting_results.txt")
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("=== 非线性最小二乘法拟合结果 ===\n")
            f.write(f"真值 L0: {L0_true * 1e6:.6f} um\n")
            f.write(f"拟合 L0: {L0_fit * 1e6:.6f} um\n")
            f.write(f"绝对误差: {error_m * 1e9:.4f} nm\n")
            f.write(f"拟合 I_bg: {I_bg_fit:.4f}\n")
            f.write(f"拟合 I_amp: {I_amp_fit:.4f}\n")
        
        # --------- 可视化 ---------
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths * 1e6, y_data, 'b-', label='Original Data (Simulation)', alpha=0.6)
        
        # 生成拟合曲线
        y_fit = fit_func(wavelengths, *popt)
        plt.plot(wavelengths * 1e6, y_fit, 'r--', label='LSQ Fitting Curve', linewidth=2)
        
        plt.title(f"Interference Spectrum Fitting (L0 Extraction)\nError: {error_m * 1e9:.4f} nm")
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plot_path = os.path.join(IMG_OUTPUT_DIR, "demodulate_lsq_fit.png")
        plt.savefig(plot_path, dpi=200)
        print(f"📊 拟合对比图已保存至: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ 拟合失败: {e}")

if __name__ == "__main__":
    run_demodulation()
