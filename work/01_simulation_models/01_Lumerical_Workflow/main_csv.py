import os
import sys
import numpy as np
import csv
from datetime import datetime

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
    "SPECTRAL_RESOLUTION_NM": 0.02,  # 使用分辨率(nm)作为输入，自动计算点数

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
            ("HSQ", 0.050),  # 40nm
            ("PSS", 0.005),  # 5nm
            ("SOC", 0.050),  # 50nm
            ("TiO2", 0.020),  # 20nm
            ("Cu", 0)  # 衬底
        ]
    },

    # FFT 找峰参数
    "FFT_PEAK_HEIGHT_RATIO": 0.2,
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
# 4. CSV Exporter
# ==========================================
class CsvExporter:
    @staticmethod
    def save_reflectance_csv(sim_data, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(save_dir, f"stackrt_reflectance_{timestamp}.csv")

        wavelengths = np.asarray(sim_data["wavelengths"], dtype=float).reshape(-1)
        reflectance = np.asarray(sim_data["R"], dtype=float).reshape(-1)
        if wavelengths.shape != reflectance.shape:
            raise ValueError(
                f"wavelengths and reflectance shape mismatch: {wavelengths.shape} vs {reflectance.shape}"
            )

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wavelength_um", "reflectance_Rp"])
            writer.writerows(zip(wavelengths, reflectance))

        print(f"Saved StackRT reflectance CSV: {csv_path}")
        return csv_path


# ==========================================
# 5. 主入口 (Main Entry)
# ==========================================
def main():
    print("=== Lumerical STACK CSV Export Workflow ===")

    simulator = LumericalSimulator(CONFIG)

    try:
        sim_data = simulator.run_stackrt()
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    output_dir = os.path.join(os.path.dirname(__file__), "stackrt_result")
    CsvExporter.save_reflectance_csv(sim_data, output_dir)
    print("All tasks completed.")


if __name__ == "__main__":
    main()