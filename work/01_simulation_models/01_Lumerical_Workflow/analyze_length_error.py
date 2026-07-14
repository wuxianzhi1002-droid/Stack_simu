import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple

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
    files = [f for f in os.listdir(RESULT_DIR) if f.startswith("dynamic_spectra") and f.endswith("44.npz")]
    if not files:
        raise FileNotFoundError(f"在 {RESULT_DIR} 中未找到动态光谱数据文件。")
    files.sort()
    return os.path.join(RESULT_DIR, files[-1])

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
# 3. 数据加载与拟合逻辑
# ==========================================
def analyze_error_vs_length():
    try:
        data_path = get_latest_data()
        print(f"📂 加载数据: {data_path}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    data = np.load(data_path)
    t_axis = data['t_axis']
    wavelengths_um = data['wavelengths']
    L_t_um = data['L_t']
    spectra = data['spectra']
    
    wavelengths_m = wavelengths_um * 1e-6
    L_t_m = L_t_um * 1e-6
    
    n_steps = len(t_axis)
    L_true_list = L_t_m
    L_fit_list = np.zeros(n_steps)
    errors_nm = np.zeros(n_steps)
    
    print(f"\n🚀 开始逐点拟合分析 (共 {n_steps} 个长度点)...")
    
    # 为了研究“长度 vs 误差”，我们对每一个点进行独立拟合
    # 使用真值作为 p0，以排除“找错条纹”的干扰，专注于模型本身的系统误差
    for i in range(n_steps):
        y_data = spectra[i, :]
        L_true = L_true_list[i]
        
        # 初始猜测 [I_bg, I_amp, L_total]
        p0 = [0.5, 0.3, L_true]   # 这里从真值开始搜索存在问题
        
        # 设定较窄的搜索范围，确保收敛到当前条纹
        lower_bounds = [0.0, 0.0, L_true - 1e-6]
        upper_bounds = [1.0, 1.0, L_true + 1e-6]
        
        try:
            popt, _ = curve_fit(interference_model, wavelengths_m, y_data, 
                               p0=p0, bounds=(lower_bounds, upper_bounds),
                               ftol=1e-9, xtol=1e-9)
            
            L_fit_list[i] = popt[2]
            errors_nm[i] = (popt[2] - L_true) * 1e9
            
            if i % 40 == 0:
                print(f"进度: {i:3d}/{n_steps} | L: {L_true*1e6:9.4f} um | Error: {errors_nm[i]:7.2f} nm")
        except Exception as e:
            print(f"⚠️ 步 {i} 拟合失败: {e}")
            errors_nm[i] = np.nan

    # --------- 数据处理: 按长度排序 ---------
    # 因为 L_t 是正弦波，长度会重复出现，排序后画线图更清晰
    sort_idx = np.argsort(L_true_list)
    L_sorted_um = L_true_list[sort_idx] * 1e6
    err_sorted_nm = errors_nm[sort_idx]

    # --------- 可视化 ---------
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图，观察分布
    plt.scatter(L_true_list * 1e6, errors_nm, alpha=0.5, s=15, c='blue', label='Data Points')
    
    # 绘制趋势线 (排序后的)
    plt.plot(L_sorted_um, err_sorted_nm, 'r-', linewidth=1, label='Error Trend')
    
    plt.title("Cavity Length vs. Fitting Error Analysis\n(Systematic Error due to Stack Phase)")
    plt.xlabel("Ground Truth Cavity Length (um)")
    plt.ylabel("Fitting Error (nm)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    plot_path = os.path.join(IMG_OUTPUT_DIR, "error_vs_length.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\n📊 误差分析图已保存至: {plot_path}")
    
    # 统计信息
    valid_errors = errors_nm[~np.isnan(errors_nm)]
    print(f"--- 统计结果 ---")
    print(f"平均偏置: {np.mean(valid_errors):.3f} nm")
    print(f"误差峰谷值 (P-V): {np.max(valid_errors) - np.min(valid_errors):.3f} nm")
    print(f"标准差 (STD): {np.std(valid_errors):.3f} nm")

if __name__ == "__main__":
    analyze_error_vs_length()
