import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# 1. 路径与环境配置 (Path & Environment)
# ==========================================
LUMERICAL_PATH = r"D:\Program Files\Lumerical\v241\api\python"
if os.path.exists(LUMERICAL_PATH):
    if LUMERICAL_PATH not in sys.path:
        sys.path.append(LUMERICAL_PATH)
    os.environ['PATH'] += os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    print("错误: 未找到 lumapi。请检查 Lumerical 安装路径及 Python 环境。")
    lumapi = None

# ==========================================
# 2. 全局配置项 (Global Configuration)
# ==========================================
CONFIG = {
    # 模型选择: 'simple' 或 'PSS_TiO2'
    "MODEL_TYPE": "PSS_TiO2",

    # 波长范围 (um)
    "WAVELENGTH_START": 0.2,
    "WAVELENGTH_STOP": 0.6,
    "SPECTRAL_RESOLUTION_NM": 0.01,  # 使用分辨率(nm)作为输入，自动计算点数

    # 统一模型格式: (材料名或固定折射率, 厚度_um)
    # 模型 A: 简单多腔模型 (10um + 1um)
    "SIMPLE_MODEL": {
        "LAYERS": [
            ("RefReflector", 0),  # 参考面 (n=5.8284 对应与空气界面 R=0.5)
            ("Air", 2000.0),  # 10um 空气腔
            (1.6488, 1),  # 中间层 (空气->中间层，等效 R=6%)
            (1.9723, 0)  # 衬底 (中间层->衬底直接接触，等效 R=0.8%)
        ]
    },

    # 模型 B: 真实 PSS-TiO2 堆栈 (带 1mm 空气腔)
    "PSS_TIO2_MODEL": {
        "LAYERS": [
            ("RefReflector", 0),  # 参考面 (n=5.8284)
            ("Air", 1000.0),  # 1mm 腔
            ("HSQ", 0),  # 40nm
            # ("HSQ", 0.040),  # 40nm
            # ("PSS", 0.005),  # 5nm
            # ("SOC", 0.050),  # 50nm
            # ("TiO2", 0.020),  # 20nm
            # ("Cu", 0)  # 衬底
        ]
    },

    # FFT 找峰参数
    "FFT_PEAK_HEIGHT_RATIO": 0.05,
    "FFT_IGNORE_DC_BINS": 50
}


# ==========================================
# 3. 仿真驱动模块 (Simulation Engine)
# ==========================================
class LumericalSimulator:
    def __init__(self, config):
        self.config = config

        # 自动计算采样点数: (终止波长 - 起始波长) * 1000 / 分辨率 + 1
        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000
        num_points = int(span_nm / config["SPECTRAL_RESOLUTION_NM"]) + 1

        print(f"设定分辨率: {config['SPECTRAL_RESOLUTION_NM']} nm")
        print(f"自动计算采样点数: {num_points}")

        self.wavelengths = np.linspace(
            config["WAVELENGTH_START"],
            config["WAVELENGTH_STOP"],
            num_points
        )
        self.freqs = 3e8 / (self.wavelengths * 1e-6)
        self.fdtd = None

    def _get_n_matrix(self, model_key):
        """统一的折射率与厚度生成逻辑"""
        layers = self.config[model_key]["LAYERS"]
        num_layers = len(layers)
        n_matrix = np.zeros((num_layers, len(self.freqs)), dtype=complex)
        thicknesses = []

        w_um = self.wavelengths

        # 预加载 Cu 数据
        cu_n_k = None
        if self.fdtd:
            try:
                cu_n_k = self.fdtd.getindex("Cu (Copper) - Palik", self.freqs).flatten()
            except:
                cu_n_k = (1.1 + 2.5j) * np.ones_like(w_um)

        for i, (mat, thick) in enumerate(layers):
            thicknesses.append(thick * 1e-6)

            # 情况 1: 直接给出了数值折射率
            if isinstance(mat, (int, float, complex)):
                n_matrix[i, :] = mat

            # 情况 2: 材料名字符串逻辑
            elif mat == "RefReflector":
                n_matrix[i, :] = 5.8284
            elif mat == "Air":
                n_matrix[i, :] = 1.0
            elif mat == "HSQ":
                n_matrix[i, :] = 1.41
            elif mat == "PSS":
                n_matrix[i, :] = 1.50 + 0.05j
            elif mat == "SOC":
                n_matrix[i, :] = 1.55 + 0.005 / (w_um ** 2)
            elif mat == "TiO2":
                n_matrix[i, :] = 2.4 + 0.02 / (w_um ** 2)
            elif mat == "Cu":
                n_matrix[i, :] = cu_n_k
            else:
                n_matrix[i, :] = 1.5  # 默认 fallback

        return n_matrix, np.array(thicknesses)

    def run_stackrt(self):
        """执行 stackrt 仿真并返回结果"""
        if not lumapi:
            raise RuntimeError("lumapi 无法加载，请检查配置。")

        model_key = "SIMPLE_MODEL" if self.config["MODEL_TYPE"] == "simple" else "PSS_TIO2_MODEL"

        print(f"正在启动 Lumerical 会话 (模型类型: {self.config['MODEL_TYPE']})...")
        self.fdtd = lumapi.FDTD(hide=True)

        n_matrix, thicknesses = self._get_n_matrix(model_key)

        print(f"执行 stackrt 计算 (层数: {n_matrix.shape[0]}, 采样点: {len(self.freqs)})...")
        res = self.fdtd.stackrt(n_matrix, thicknesses, self.freqs)

        result_data = {
            "wavelengths": self.wavelengths,
            "R": res["Rp"],
            "T": res["Tp"]
        }

        self.fdtd.close()
        return result_data


# ==========================================
# 4. FFT 解算模块 (FFT Solver)
# ==========================================
class FFTSolver:
    @staticmethod
    def solve(wavelengths, intensities, config):
        # 1. k域 线性化
        k_raw = 2 * np.pi / wavelengths
        k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))

        if k_raw[0] > k_raw[-1]:
            i_linear = np.interp(k_linear, k_raw[::-1], intensities[::-1])
        else:
            i_linear = np.interp(k_linear, k_raw, intensities)

        # 2. 去直流 + 加窗
        i_detrend = i_linear - np.mean(i_linear)
        i_windowed = i_detrend * np.hanning(len(i_detrend))

        # ==========================================
        # 【核心修改】：引入 8 倍零填充 (Zero-Padding) 提升谱峰光滑度
        # ==========================================
        pad_factor = 8  # 设为 8 代表数据点数扩大到原先的 8 倍，数值越大越滑
        n_fft = len(i_windowed) * pad_factor

        # 3. FFT
        fft_data = np.abs(np.fft.rfft(i_windowed, n=n_fft))

        # 4. 深度轴计算 (um)
        dk = np.abs(k_linear[1] - k_linear[0])
        max_range = np.pi / dk
        distance_axis = np.linspace(0, max_range / 2, len(fft_data))

        # 5. 找峰
        ignore = config["FFT_IGNORE_DC_BINS"]
        peaks, _ = find_peaks(
            fft_data[ignore:],
            height=np.max(fft_data[ignore:]) * config["FFT_PEAK_HEIGHT_RATIO"],
            distance=100
        )
        peaks = peaks + ignore

        return {
            "distance_axis": distance_axis,
            "fft_data": fft_data,
            "peaks_idx": peaks,
            "peak_distances": distance_axis[peaks],
            "peak_heights": fft_data[peaks],
            "max_range": max_range / 2
        }

    @staticmethod
    def plot_results(wavelengths, intensities, fft_res, model_type):
        print("📊 正在生成具备交互功能的图表...")

        # 【核心修改 1】：改用 2026 年推荐的 Qt/Tk 交互前端（防静默），并激活约束布局防报错
        plt.ion() if hasattr(sys, 'ps1') else plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")

        # ------- 左图：光谱 -------
        ax1.plot(wavelengths * 1000, intensities, 'b', lw=0.5)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Reflectance")
        ax1.set_title(f"Reflection Spectrum ({model_type})")
        ax1.grid(True)

        # 【核心修改 2】：使用 inset_axes 代替全局 plt.axes，从根本上解决 tight_layout 空间冲突警告
        mid = len(wavelengths) // 2
        zoom_range = 100
        # 参数含义: [左下角X比例, 左下角Y比例, 宽度比例, 高度比例]
        ax_zoom = ax1.inset_axes([0.2, 0.55, 0.35, 0.35])
        ax_zoom.plot(wavelengths[mid:mid + zoom_range] * 1000, intensities[mid:mid + zoom_range], 'r')
        ax_zoom.set_title("Zoom Detail", fontsize=9)
        ax_zoom.grid(True)

        # ------- 右图：FFT 空间轴 -------
        dist = fft_res["distance_axis"]
        amp = fft_res["fft_data"]
        peaks_d = fft_res["peak_distances"]

        ax2.plot(dist, amp, 'k')
        ax2.scatter(peaks_d, fft_res["peak_heights"], color='r', marker='x', s=50, label='Detected Peaks', zorder=3)

        # 标注检测到的峰并添加垂直参考线
        max_amp = np.max(amp) if len(amp) > 0 else 1
        for d in peaks_d:
            ax2.axvline(d, color='g', linestyle='--', alpha=0.5)
            ax2.text(d, max_amp * 0.75, f" {d:.2f} $\mu$m", rotation=90, color='g', fontsize=9)

        ax2.set_xlabel("Physical Distance / Thickness ($\mu$m)")
        ax2.set_ylabel("FFT Amplitude")
        ax2.set_title("FFT Spatial Domain Analysis")
        ax2.grid(True)
        ax2.legend()

        if len(peaks_d) > 0:
            ax2.set_xlim(0, max(peaks_d) * 1.5)
        else:
            ax2.set_xlim(0, 20)

        # ------- 保存静态映像文件 -------
        img_dir = "01_Lumerical_Workflow/img"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        save_path = os.path.join(img_dir, "simulation_result.png")
        fig.savefig(save_path, dpi=300)
        print(f"✓ 图像已安全同步至: {save_path}")

        # 【核心修改 3】：打破静默，强行开启 Windows 本地 UI 交互窗口（支持放大、拖拽）
        print("💡 正在拉起 Windows 交互式绘图窗口（可使用放大镜、测量坐标）...")
        plt.show(block=True)  # 阻塞运行，确保没有 GUI 会话时窗口不会“闪退”


# ==========================================
# 5. 主入口 (Main Entry)
# ==========================================
def main():
    print(f"=== Lumerical STACK + FFT 自动化工作流 (统一 LAYERS 架构) ===")

    simulator = LumericalSimulator(CONFIG)

    try:
        sim_data = simulator.run_stackrt()
    except Exception as e:
        print(f"仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("正在解算干涉信号...")
    wavelengths = sim_data["wavelengths"]
    intensities = sim_data["R"].flatten()

    fft_results = FFTSolver.solve(wavelengths, intensities, CONFIG)

    print("\n[解算结果]")
    if len(fft_results["peak_distances"]) > 0:
        for i, d in enumerate(fft_results["peak_distances"]):
            print(f"检测到物理腔长 {i + 1}: {d:.4f} um")
    else:
        print("未检测到明显的干涉峰。")

    FFTSolver.plot_results(wavelengths, intensities, fft_results, CONFIG["MODEL_TYPE"])


if __name__ == "__main__":
    main()