import argparse
import json
import os
import sys
import time
from datetime import datetime
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
GENERATOR_VERSION = "main_dynamic_v4"


# ==========================================
# 2. 全局配置项 (Global Configuration)
# ==========================================
CONFIG = {
    # 模型类型: 'PSS_TiO2' 或者 'simple'
    "MODEL_TYPE": "PSS_TiO2",

    # 波长范围 (um)
    "WAVELENGTH_START": 0.2,
    "WAVELENGTH_STOP": 0.6,
    "SPECTRAL_RESOLUTION_NM": 0.02,  # 适度调低分辨率以保障仿真速度 (共计约 6000 点)

    # 动态调制配置
    "MODULATION": {
        "f_Hz": 1000.0,       # 调制频率 f = 1 kHz
        "A_nm": 5,        # 调制振幅 A = 100 nm，改成5nm更加适合锁相解调
        "T_s": 0.01,          # 总时长 T = 0.01 s (即 10 毫秒，包含 10 个周期)
        "fs_Hz": 40000.0      # 时间采样率 = 40 kHz (满足 1kHz 振动的高次多普勒采频要求)
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
            # ("HSQ", 0.040), 
            # ("PSS", 0.005), 
            # ("SOC", 0.050), 
            # ("TiO2", 0.020), 
            ("HSQ", 0.030), 
            ("PSS", 0.010), 
            ("SOC", 0.040), 
            ("TiO2", 0.040), 
            ("Cu", 0) 
        ]
    }
}

# These levels are an engineering robustness sweep, not hardware-calibrated
# uncertainty specifications. Systematic terms are sampled once per NPZ;
# detector and frame-gain terms vary over the acquired time series.
NOISE_PROFILES = {
    "clean": {
        "n_real_sigma_rel": 0.0,
        "k_sigma_rel": 0.0,
        "angle_sigma_deg": 0.0,
        "wavelength_offset_sigma_nm": 0.0,
        "amplitude_sigma_rel": 0.0,
        "frame_gain_sigma_rel": 0.0,
        "reflectance_sigma_abs": 0.0,
    },
    "low": {
        "n_real_sigma_rel": 5.0e-4,
        "k_sigma_rel": 1.0e-2,
        "angle_sigma_deg": 0.05,
        "wavelength_offset_sigma_nm": 0.005,
        "amplitude_sigma_rel": 1.0e-3,
        "frame_gain_sigma_rel": 2.0e-4,
        "reflectance_sigma_abs": 2.0e-4,
    },
    "medium": {
        "n_real_sigma_rel": 2.0e-3,
        "k_sigma_rel": 5.0e-2,
        "angle_sigma_deg": 0.20,
        "wavelength_offset_sigma_nm": 0.020,
        "amplitude_sigma_rel": 5.0e-3,
        "frame_gain_sigma_rel": 1.0e-3,
        "reflectance_sigma_abs": 1.0e-3,
    },
    "high": {
        "n_real_sigma_rel": 5.0e-3,
        "k_sigma_rel": 1.0e-1,
        "angle_sigma_deg": 0.50,
        "wavelength_offset_sigma_nm": 0.050,
        "amplitude_sigma_rel": 1.0e-2,
        "frame_gain_sigma_rel": 5.0e-3,
        "reflectance_sigma_abs": 5.0e-3,
    },
}

PERTURBED_MATERIALS = ("HSQ", "PSS", "SOC", "TiO2")


def realize_noise(profile_name, rng):
    profile = dict(NOISE_PROFILES[profile_name])
    material_n_delta = {
        name: float(rng.normal(0.0, profile["n_real_sigma_rel"]))
        for name in PERTURBED_MATERIALS
    }
    material_k_delta = {
        name: float(rng.normal(0.0, profile["k_sigma_rel"]))
        for name in PERTURBED_MATERIALS
    }
    return {
        "profile": profile_name,
        "profile_config": profile,
        "material_n_real_rel_delta": material_n_delta,
        "material_k_rel_delta": material_k_delta,
        "angle_error_deg": float(rng.normal(0.0, profile["angle_sigma_deg"])),
        "wavelength_offset_nm": float(rng.normal(0.0, profile["wavelength_offset_sigma_nm"])),
        "amplitude_rel_error": float(rng.normal(0.0, profile["amplitude_sigma_rel"])),
    }

# ==========================================
# 3. 动态时间序列仿真核心引擎
# ==========================================
class DynamicSimulator:
    def __init__(self, config, noise_level, random_seed):
        self.config = config
        self.noise_level = noise_level
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.noise = realize_noise(noise_level, self.rng)

        # 光谱轴设定
        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000
        num_points = int(span_nm / config["SPECTRAL_RESOLUTION_NM"]) + 1
        self.wavelengths = np.linspace(config["WAVELENGTH_START"], config["WAVELENGTH_STOP"], num_points)
        self.physical_wavelengths = self.wavelengths + self.noise["wavelength_offset_nm"] / 1000.0
        self.freqs = 3e8 / (self.physical_wavelengths * 1e-6)
        
        # 调制轴设定
        mod = config["MODULATION"]
        self.fs = mod["fs_Hz"]
        self.f = mod["f_Hz"]
        self.T = mod["T_s"]
        self.nominal_A_um = mod["A_nm"] / 1000.0
        self.A_um = self.nominal_A_um * (1.0 + self.noise["amplitude_rel_error"])
        self.angle_deg = self.noise["angle_error_deg"]
        
        # 时间序列: dt = 1/fs
        self.t_axis = np.arange(0, self.T, 1 / self.fs)
        self.Nt = len(self.t_axis)
        
        print(f"[参数设定] 噪声等级: {self.noise_level}, seed = {self.random_seed}")
        print(f"[参数设定] 波长采样点数: N_lambda = {num_points}")
        print(
            f"[参数设定] 调制频率: f = {self.f} Hz, "
            f"名义振幅 = {mod['A_nm']} nm, 实际振幅 = {self.A_um * 1000.0:.6g} nm"
        )
        print(
            f"[参数设定] 实际入射角 = {self.angle_deg:.6g} deg, "
            f"波长零点偏移 = {self.noise['wavelength_offset_nm']:.6g} nm"
        )
        print(f"[参数设定] 采样率: fs = {self.fs} Hz, 时间点数: Nt = {self.Nt}")

    def _get_n_matrix(self, model_key):
        layers = self.config[model_key]["LAYERS"]
        num_layers = len(layers)
        n_matrix = np.zeros((num_layers, len(self.freqs)), dtype=complex)
        thicknesses = []

        w_um = self.physical_wavelengths
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

            if mat in PERTURBED_MATERIALS:
                dn = self.noise["material_n_real_rel_delta"][mat]
                dk = self.noise["material_k_rel_delta"][mat]
                base_n = n_matrix[i, :]
                n_matrix[i, :] = base_n.real * (1.0 + dn) + 1j * base_n.imag * (1.0 + dk)

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
            
            res = fdtd.stackrt(n_matrix, current_thicknesses, self.freqs, float(self.angle_deg))
            spectra[i, :] = res["Rp"].flatten()
            
            # 每完成25%打印一次进度
            if (i+1) % max(1, (self.Nt // 4)) == 0:
                print(f"   [进度] {i+1}/{self.Nt} ({((i+1)/self.Nt)*100:.0f}%) - 已用时: {time.time() - start_time:.1f}s")

        fdtd.close()
        print(f"✅ 动态仿真序列完成。总耗时: {time.time() - start_time:.2f} s")

        profile = self.noise["profile_config"]
        frame_gain_error = self.rng.normal(0.0, profile["frame_gain_sigma_rel"], size=self.Nt)
        measured_spectra = spectra * (1.0 + frame_gain_error[:, None])
        if profile["reflectance_sigma_abs"] > 0.0:
            measured_spectra = measured_spectra + self.rng.normal(
                0.0,
                profile["reflectance_sigma_abs"],
                size=measured_spectra.shape,
            )
        clip_mask = (measured_spectra < 0.0) | (measured_spectra > 1.0)
        clip_fraction = float(np.mean(clip_mask))
        measured_spectra = np.clip(measured_spectra, 0.0, 1.0)

        layers = self.config[model_key]["LAYERS"]

        return {
            "t_axis": self.t_axis,
            "wavelengths": self.wavelengths,
            "physical_wavelengths": self.physical_wavelengths,
            "L_t": L_t_m * 1e6,  # 转回 um 保存
            "spectra": measured_spectra,
            "spectra_physical": spectra,
            "frame_gain_error": frame_gain_error,
            "reflectance_clip_fraction": clip_fraction,
            "noise_level": self.noise_level,
            "random_seed": self.random_seed,
            "noise_realization": self.noise,
            "nominal_amplitude_nm": self.nominal_A_um * 1000.0,
            "actual_amplitude_nm": self.A_um * 1000.0,
            "actual_angle_deg": self.angle_deg,
            "modulation_frequency_hz": self.f,
            "sampling_rate_hz": self.fs,
            "layer_names": [str(layer[0]) for layer in layers],
            "layer_thickness_um": [float(layer[1]) for layer in layers],
        }


# ==========================================
# 4. 可视化模块 (Visualization & Analysis)
# ==========================================
class DynamicAnalyzer:
    @staticmethod
    def lockin_harmonics(data, f_mod, A_um, harmonics=(1, 2, 3)):
        t_axis = np.asarray(data["t_axis"])
        spectra = np.asarray(data["spectra"])

        if spectra.ndim != 2:
            raise ValueError("spectra must be a 2D array with shape (Nt, N_lambda).")
        if spectra.shape[0] != len(t_axis):
            raise ValueError("spectra.shape[0] must match len(t_axis).")
        if A_um <= 0:
            raise ValueError("A_um must be positive for dIdL_1f normalization.")

        spectra_ac = spectra - np.mean(spectra, axis=0, keepdims=True)
        results = {}

        for harmonic in harmonics:
            key = f"{harmonic}f"
            omega_t = 2 * np.pi * harmonic * f_mod * t_axis
            sin_ref = np.sin(omega_t)
            cos_ref = np.cos(omega_t)

            X = 2.0 * (spectra_ac.T @ sin_ref) / len(t_axis)
            Y = 2.0 * (spectra_ac.T @ cos_ref) / len(t_axis)
            R = np.sqrt(X ** 2 + Y ** 2)
            phase = np.arctan2(Y, X)

            results[key] = {
                "X": X,
                "Y": Y,
                "R": R,
                "phase": phase,
            }

        results["dIdL_1f"] = results["1f"]["R"] / A_um
        return results

    @staticmethod
    def save_and_plot(data, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        profile_name = data["noise_level"]
        file_tag = f"{profile_name}_{timestamp}"
        f_mod = data["modulation_frequency_hz"]
        nominal_A_um = data["nominal_amplitude_nm"] / 1000.0
        actual_A_um = data["actual_amplitude_nm"] / 1000.0
        lockin = DynamicAnalyzer.lockin_harmonics(data, f_mod, nominal_A_um)
        physical_data = {
            "t_axis": data["t_axis"],
            "spectra": data["spectra_physical"],
        }
        physical_lockin = DynamicAnalyzer.lockin_harmonics(physical_data, f_mod, actual_A_um)

        material_names = np.asarray(PERTURBED_MATERIALS, dtype="U16")
        n_deltas = np.asarray(
            [data["noise_realization"]["material_n_real_rel_delta"][name] for name in material_names],
            dtype=float,
        )
        k_deltas = np.asarray(
            [data["noise_realization"]["material_k_rel_delta"][name] for name in material_names],
            dtype=float,
        )
        noise_json = json.dumps(data["noise_realization"], ensure_ascii=False, sort_keys=True)
            
        # --------- 1. 数据保存为 Npz ---------
        npz_path = os.path.join(save_dir, f"dynamic_spectra_{file_tag}.npz")
        np.savez_compressed(
            npz_path, 
            t_axis=data["t_axis"], 
            wavelengths=data["wavelengths"], 
            physical_wavelengths=data["physical_wavelengths"],
            L_t=data["L_t"], 
            spectra=data["spectra"],
            spectra_physical_mean=np.mean(data["spectra_physical"], axis=0),
            lockin_1f_X=lockin["1f"]["X"],
            lockin_1f_Y=lockin["1f"]["Y"],
            lockin_1f_R=lockin["1f"]["R"],
            lockin_1f_phase=lockin["1f"]["phase"],
            lockin_2f_X=lockin["2f"]["X"],
            lockin_2f_Y=lockin["2f"]["Y"],
            lockin_2f_R=lockin["2f"]["R"],
            lockin_2f_phase=lockin["2f"]["phase"],
            lockin_3f_X=lockin["3f"]["X"],
            lockin_3f_Y=lockin["3f"]["Y"],
            lockin_3f_R=lockin["3f"]["R"],
            lockin_3f_phase=lockin["3f"]["phase"],
            dIdL_1f=lockin["dIdL_1f"],
            dIdL_1f_actual_amplitude=lockin["1f"]["R"] / actual_A_um,
            physical_lockin_1f_X=physical_lockin["1f"]["X"],
            physical_lockin_1f_Y=physical_lockin["1f"]["Y"],
            physical_lockin_1f_R=physical_lockin["1f"]["R"],
            physical_dIdL_1f=physical_lockin["dIdL_1f"],
            noise_level=np.asarray(profile_name),
            random_seed=np.asarray(data["random_seed"], dtype=np.int64),
            generator_version=np.asarray(GENERATOR_VERSION),
            noise_config_json=np.asarray(noise_json),
            nominal_amplitude_nm=np.asarray(data["nominal_amplitude_nm"]),
            actual_amplitude_nm=np.asarray(data["actual_amplitude_nm"]),
            modulation_frequency_hz=np.asarray(data["modulation_frequency_hz"]),
            sampling_rate_hz=np.asarray(data["sampling_rate_hz"]),
            nominal_angle_deg=np.asarray(0.0),
            actual_angle_deg=np.asarray(data["actual_angle_deg"]),
            wavelength_offset_nm=np.asarray(data["noise_realization"]["wavelength_offset_nm"]),
            frame_gain_error=data["frame_gain_error"],
            reflectance_clip_fraction=np.asarray(data["reflectance_clip_fraction"]),
            perturbed_material_names=material_names,
            material_n_real_rel_delta=n_deltas,
            material_k_rel_delta=k_deltas,
            layer_names=np.asarray(data["layer_names"], dtype="U32"),
            layer_thickness_um=np.asarray(data["layer_thickness_um"], dtype=float),
        )
        print(f"💾 数据已成功压缩并保存至二维矩阵: {npz_path}")

        metadata_path = os.path.join(save_dir, f"noise_realization_{file_tag}.json")
        metadata = {
            "generator_version": GENERATOR_VERSION,
            "noise_level": profile_name,
            "random_seed": data["random_seed"],
            "noise_realization": data["noise_realization"],
            "nominal_amplitude_nm": data["nominal_amplitude_nm"],
            "actual_amplitude_nm": data["actual_amplitude_nm"],
            "actual_angle_deg": data["actual_angle_deg"],
            "reflectance_clip_fraction": data["reflectance_clip_fraction"],
            "npz_path": npz_path,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # --------- 2. 可视化图表生成 ---------
        t = data["t_axis"] * 1000  # ms
        w = data["wavelengths"] * 1000  # nm
        S = data["spectra"]

        lockin_fig, lockin_axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        lockin_fig.suptitle(f"Lock-in analysis - noise level: {profile_name}")
        lockin_axs[0, 0].plot(w, lockin["1f"]["R"], color='tab:blue')
        lockin_axs[0, 0].set_title("Lock-in 1f R($\\lambda$)")
        lockin_axs[0, 0].set_xlabel("Wavelength (nm)")
        lockin_axs[0, 0].set_ylabel("Amplitude")
        lockin_axs[0, 0].grid(True)

        lockin_axs[0, 1].plot(w, lockin["2f"]["R"], color='tab:orange')
        lockin_axs[0, 1].set_title("Lock-in 2f R($\\lambda$)")
        lockin_axs[0, 1].set_xlabel("Wavelength (nm)")
        lockin_axs[0, 1].set_ylabel("Amplitude")
        lockin_axs[0, 1].grid(True)

        lockin_axs[1, 0].plot(w, lockin["3f"]["R"], color='tab:green')
        lockin_axs[1, 0].set_title("Lock-in 3f R($\\lambda$)")
        lockin_axs[1, 0].set_xlabel("Wavelength (nm)")
        lockin_axs[1, 0].set_ylabel("Amplitude")
        lockin_axs[1, 0].grid(True)

        lockin_axs[1, 1].plot(w, lockin["dIdL_1f"], color='tab:red')
        lockin_axs[1, 1].set_title("dIdL_1f($\\lambda$)")
        lockin_axs[1, 1].set_xlabel("Wavelength (nm)")
        lockin_axs[1, 1].set_ylabel("Reflectance / um")
        lockin_axs[1, 1].grid(True)

        lockin_png_path = os.path.join(save_dir, f"lockin_analysis_{file_tag}.png")
        lockin_fig.savefig(lockin_png_path, dpi=200)
        print(f"📊 数字锁相分析图表已生成: {lockin_png_path}")
        plt.close(lockin_fig)

        fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        fig.suptitle(f"Dynamic StackRT - noise level: {profile_name}")

        # (a) 热力图: Time vs Wavelength
        im = axs[0, 0].pcolormesh(w, t, S, shading='auto', cmap='viridis')
        axs[0, 0].set_title("2D Heatmap: I($\\lambda$, t)")
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
        axs[1, 0].set_title(f"Time-Series Interference at $\\lambda$ = {w[mid_w_idx]:.1f} nm")
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
        png_path = os.path.join(save_dir, f"dynamic_analysis_dashboard_{file_tag}.png")
        fig.savefig(png_path, dpi=200)
        print(f"📊 静态分析图表已生成: {png_path}")
        
        # 注意: 避免 Pycharm 执行时卡死，此处将阻塞设置为 False 并在保存动图后关闭
        # 如果需要交互，可以去除这段逻辑
        plt.close(fig)


# ==========================================
# 5. 主入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clean/low/medium/high noisy dynamic StackRT NPZ datasets."
    )
    parser.add_argument(
        "--noise-level",
        choices=["clean", "low", "medium", "high", "all"],
        default="all",
        help="Noise profile to generate. 'all' generates four comparable NPZ files.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--describe-profiles",
        action="store_true",
        help="Print profile definitions and exit without starting Lumerical.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.describe_profiles:
        print(json.dumps(NOISE_PROFILES, ensure_ascii=False, indent=2))
        return

    print("=== Lumerical 时序动态仿真模式 v4（多档噪声）===")
    profile_names = list(NOISE_PROFILES) if args.noise_level == "all" else [args.noise_level]
    seed_sequences = np.random.SeedSequence(args.seed).spawn(len(profile_names))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(
        script_dir,
        "..",
        "..",
        "04_results_and_datasets",
        "dynamic_stackrt_lockin_v4",
    ))

    failures = []
    for profile_name, seed_sequence in zip(profile_names, seed_sequences):
        profile_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        print(f"\n===== 开始生成 {profile_name} 数据 =====")
        sim = DynamicSimulator(CONFIG, profile_name, profile_seed)
        try:
            data = sim.run_dynamic_sequence()
            DynamicAnalyzer.save_and_plot(data, output_dir)
        except Exception as exc:
            failures.append((profile_name, str(exc)))
            print(f"❌ {profile_name} 仿真发生错误: {exc}")
            import traceback
            traceback.print_exc()
            if "Session not found" in str(exc):
                print("Lumerical 会话无法启动，停止其余噪声档位，避免重复失败。")
                break

    if failures:
        raise RuntimeError(f"Noise profile failures: {failures}")
    print(f"\n✅ 所有噪声任务完成，输出目录: {output_dir}")

if __name__ == "__main__":
    main()
