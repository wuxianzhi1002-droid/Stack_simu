import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    # 模型类型: 'PSS_TiO2' 或者 'simple'
    "MODEL_TYPE": "PSS_TiO2",

    # 波长范围 (um)
    "WAVELENGTH_START": 0.3,
    "WAVELENGTH_STOP": 0.9,
    "SPECTRAL_RESOLUTION_NM": 0.1,  # 适度调低分辨率以保障仿真速度 (共计约 6000 点)

    # 动态调制配置
    "MODULATION": {
        "f_Hz": 1000.0,       # 调制频率 f = 1 kHz
        "A_nm": 100.0,        # 调制振幅 A = 100 nm
        "T_s": 0.01,          # 总时长 T = 0.01 s (即 10 毫秒，包含 10 个周期)
        "fs_Hz": 20000.0      # 时间采样率 = 20 kHz (满足 1kHz 振动的高次多普勒采频要求)
    },

    # 模型 A: 简单多腔模型
    "SIMPLE_MODEL": {
        "LAYERS": [
            ("RefReflector", 0), 
            ("Air", 1000.0),      # 1mm 空气腔 (被调制的目标)
            (1.6488, 1), 
            (1.9723, 0)
        ]
    },

    # 模型 B: PSS-TiO2
    "PSS_TIO2_MODEL": {
        "LAYERS": [
            ("RefReflector", 0),  
            ("Air", 1000.0),      # 1mm 腔 (将在这里引入正弦调制)
            ("HSQ", 0.040), 
            ("PSS", 0.005), 
            ("SOC", 0.050), 
            ("TiO2", 0.020), 
            ("Cu", 0) 
        ]
    }
}

# ==========================================
# 3. 动态时间序列仿真核心引擎
# ==========================================
class DynamicSimulator:
    def __init__(self, config):
        self.config = config
        
        # 光谱轴设定
        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000
        num_points = int(span_nm / config["SPECTRAL_RESOLUTION_NM"]) + 1
        self.wavelengths = np.linspace(config["WAVELENGTH_START"], config["WAVELENGTH_STOP"], num_points)
        self.freqs = 3e8 / (self.wavelengths * 1e-6)
        
        # 调制轴设定
        mod = config["MODULATION"]
        self.fs = mod["fs_Hz"]
        self.f = mod["f_Hz"]
        self.T = mod["T_s"]
        self.A_um = mod["A_nm"] / 1000.0
        
        # 时间序列: dt = 1/fs
        self.t_axis = np.arange(0, self.T, 1 / self.fs)
        self.Nt = len(self.t_axis)
        
        print(f"[参数设定] 波长采样点数: N_lambda = {num_points}")
        print(f"[参数设定] 调制频率: f = {self.f} Hz, 振幅 A = {mod['A_nm']} nm")
        print(f"[参数设定] 采样率: fs = {self.fs} Hz, 时间点数: Nt = {self.Nt}")

    def _get_n_matrix(self, model_key):
        layers = self.config[model_key]["LAYERS"]
        num_layers = len(layers)
        n_matrix = np.zeros((num_layers, len(self.freqs)), dtype=complex)
        thicknesses = []

        w_um = self.wavelengths
        cu_n_k = (1.1 + 2.5j) * np.ones_like(w_um)
        air_idx = -1

        for i, (mat, thick) in enumerate(layers):
            thicknesses.append(thick * 1e-6) # 转换为米

            if mat == "Air":
                air_idx = i  # 记录空气腔所在的索引
                
            if isinstance(mat, (int, float, complex)):
                n_matrix[i, :] = mat
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
                # Lumerical 中 Cu 较难不开启 UI 直接获取，此处使用经验近似公式代替 API 获取以加快速度
                n_matrix[i, :] = cu_n_k
            else:
                n_matrix[i, :] = 1.5

        return n_matrix, np.array(thicknesses), air_idx

    def run_dynamic_sequence(self):
        if not lumapi:
            raise RuntimeError("lumapi is not available.")

        model_key = "SIMPLE_MODEL" if self.config["MODEL_TYPE"] == "simple" else "PSS_TIO2_MODEL"
        n_matrix, thicknesses_base, air_idx = self._get_n_matrix(model_key)

        if air_idx == -1:
            raise ValueError("Model does not contain an 'Air' layer to modulate.")

        # L(t) = L0 + A * sin(2*pi*f*t)
        L0_m = thicknesses_base[air_idx]
        L_t_m = L0_m + (self.A_um * 1e-6) * np.sin(2 * np.pi * self.f * self.t_axis)

        # 结果容器 I(lambda, t)
        spectra = np.zeros((self.Nt, len(self.wavelengths)))

        print("🚀 启动 Lumerical FDTD API 会话...")
        fdtd = lumapi.FDTD(hide=True)

        start_time = time.time()
        print(f"🔄 正在循环执行 {self.Nt} 次 StackRT 仿真序列 (矩阵形状: {spectra.shape})...")

        # 核心仿真循环
        for i in range(self.Nt):
            current_thicknesses = thicknesses_base.copy()
            current_thicknesses[air_idx] = L_t_m[i]
            
            res = fdtd.stackrt(n_matrix, current_thicknesses, self.freqs)
            spectra[i, :] = res["Rp"].flatten()
            
            # 每完成25%打印一次进度
            if (i+1) % max(1, (self.Nt // 4)) == 0:
                print(f"   [进度] {i+1}/{self.Nt} ({((i+1)/self.Nt)*100:.0f}%) - 已用时: {time.time() - start_time:.1f}s")

        fdtd.close()
        print(f"✅ 动态仿真序列完成。总耗时: {time.time() - start_time:.2f} s")

        return {
            "t_axis": self.t_axis,
            "wavelengths": self.wavelengths,
            "L_t": L_t_m * 1e6,  # 转回 um 保存
            "spectra": spectra
        }


# ==========================================
# 4. 可视化模块 (Visualization & Analysis)
# ==========================================
class DynamicAnalyzer:
    @staticmethod
    def save_and_plot(data, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # --------- 1. 数据保存为 Npz ---------
        npz_path = os.path.join(save_dir, "dynamic_spectra.npz")
        np.savez_compressed(
            npz_path, 
            t_axis=data["t_axis"], 
            wavelengths=data["wavelengths"], 
            L_t=data["L_t"], 
            spectra=data["spectra"]
        )
        print(f"💾 数据已成功压缩并保存至二维矩阵: {npz_path}")

        # --------- 2. 可视化图表生成 ---------
        t = data["t_axis"] * 1000  # ms
        w = data["wavelengths"] * 1000  # nm
        S = data["spectra"]

        fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

        # (a) 热力图: Time vs Wavelength
        im = axs[0, 0].pcolormesh(w, t, S, shading='auto', cmap='viridis')
        axs[0, 0].set_title("2D Heatmap: I($\lambda$, t)")
        axs[0, 0].set_xlabel("Wavelength (nm)")
        axs[0, 0].set_ylabel("Time (ms)")
        fig.colorbar(im, ax=axs[0, 0], label="Reflectance")

        # (b) 固定时间的单光谱 (t=0 & t=T/4)
        mid_t_idx = len(t) // 4
        axs[0, 1].plot(w, S[0, :], label='t = 0 ms', alpha=0.8)
        axs[0, 1].plot(w, S[mid_t_idx, :], label=f't = {t[mid_t_idx]:.2f} ms', alpha=0.8)
        axs[0, 1].set_title("Spectra at Fixed Timesteps")
        axs[0, 1].set_xlabel("Wavelength (nm)")
        axs[0, 1].set_ylabel("Reflectance")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # (c) 固定波长的时序干涉波形
        mid_w_idx = len(w) // 2
        axs[1, 0].plot(t, S[:, mid_w_idx], color='red')
        axs[1, 0].set_title(f"Time-Series Interference at $\lambda$ = {w[mid_w_idx]:.1f} nm")
        axs[1, 0].set_xlabel("Time (ms)")
        axs[1, 0].set_ylabel("Reflectance")
        axs[1, 0].grid(True)

        # (d) 对 (c) 中的波形进行时间维度 FFT (验证振动频率提取)
        signal = S[:, mid_w_idx] - np.mean(S[:, mid_w_idx])
        signal = signal * np.hanning(len(signal))
        fft_amp = np.abs(np.fft.rfft(signal))
        # 恢复频率轴 freq = (k * fs) / N
        fs = 1.0 / (data["t_axis"][1] - data["t_axis"][0])
        freqs_axis = np.fft.rfftfreq(len(signal), d=1.0/fs)
        
        axs[1, 1].plot(freqs_axis, fft_amp, color='purple')
        axs[1, 1].set_title("FFT of Time-Series (Extracting Harmonic Resonances)")
        axs[1, 1].set_xlabel("Frequency (Hz)")
        axs[1, 1].set_ylabel("FFT Amplitude")
        axs[1, 1].set_xlim(0, 5000)
        axs[1, 1].grid(True)

        # 保存静态图表
        png_path = os.path.join(save_dir, "dynamic_analysis_dashboard.png")
        fig.savefig(png_path, dpi=200)
        print(f"📊 静态分析图表已生成: {png_path}")
        
        # 注意: 避免 Pycharm 执行时卡死，此处将阻塞设置为 False 并在保存动图后关闭
        # 如果需要交互，可以去除这段逻辑
        plt.close(fig)
        
        # --------- 3. 生成光谱动画 (GIF) ---------
        print("🎞️ 正在渲染动态光谱动画 (请耐心等待...)")
        fig_anim, ax_anim = plt.subplots(figsize=(8, 4))
        ax_anim.set_xlim(w[0], w[-1])
        ax_anim.set_ylim(np.min(S), np.max(S) + 0.1)
        ax_anim.set_xlabel("Wavelength (nm)")
        ax_anim.set_ylabel("Reflectance")
        ax_anim.grid(True)
        
        line, = ax_anim.plot([], [], lw=1, color='blue')
        time_text = ax_anim.text(0.02, 0.9, '', transform=ax_anim.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            line.set_data(w, S[i, :])
            time_text.set_text(f"Time: {t[i]:.2f} ms")
            return line, time_text

        # 抽帧生成动画 (最多渲染 100 帧以防止体积过大)
        step = max(1, len(t) // 100)
        frames_idx = list(range(0, len(t), step))
        
        anim = animation.FuncAnimation(fig_anim, animate, init_func=init, frames=frames_idx, interval=50, blit=True)
        gif_path = os.path.join(save_dir, "dynamic_spectrum_animation.gif")
        
        try:
            # 使用 matplotlib 自带的 pillow writer 生成 GIF
            writer = animation.PillowWriter(fps=15)
            anim.save(gif_path, writer=writer)
            print(f"🎬 动态光谱动画已保存: {gif_path}")
        except Exception as e:
            print(f"生成 GIF 失败，可能缺少相关库: {e}")
            
        plt.close(fig_anim)


# ==========================================
# 5. 主入口
# ==========================================
def main():
    print("=== Lumerical 时序动态仿真模式 (Dynamic Simulation) ===")
    
    sim = DynamicSimulator(CONFIG)
    
    try:
        data = sim.run_dynamic_sequence()
    except Exception as e:
        print(f"❌ 仿真发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 将数据保存至结果目录
    output_dir = r"01_Lumerical_Workflow\stackrt_result"
    DynamicAnalyzer.save_and_plot(data, output_dir)
    print("\n✅ 所有任务流执行完毕！数据、图表和动画已生成。")

if __name__ == "__main__":
    main()
