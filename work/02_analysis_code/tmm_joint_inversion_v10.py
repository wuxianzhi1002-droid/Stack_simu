"""V10 固定报告轴光谱的六参数 TMM + 光谱仪响应反演。

拟合参数顺序固定为：
    [Air_um, HSQ_nm, PSS_nm, SOC_nm, TiO2_nm, reflector_angle_deg]

本文件只使用 main_v10.py 输出的固定 reported 波长轴和 spectrum_measured。隐藏的
true_pixel_center_wavelengths、光谱仪真实轴误差、光源漂移、材料扰动、真实膜厚和真实
角度均不得进入前向模型、残差、全局起点或结果排序。

与 V9 的关键区别是反演前向模型显式包含 V10 已知的名义光谱仪响应：
1. 在 NPZ config_json 指定的 0.002 nm 固定绝对网格上计算六参数 TMM 反射率。
2. 使用已知名义光源包络、QE、光子能量转换和 Gaussian ILS。
3. 在固定 reported 像素中心采样，并计算样品/参考的名义归一化光谱。
4. 随机 source drift、轴准确度误差、材料误差和探测器噪声仍作为模型失配，
   用于评估六参数反演鲁棒性，而不是通过读取仿真真值将其消除。

求解继续采用无真值 Latin hypercube、差分进化和多起点有界 least_squares；只有已收敛
结果能够参与 rank 1 排序。拟合和排序结束后才读取结构真值并计算误差。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# 参数定义和单位约定
# ---------------------------------------------------------------------------
# Air 使用 um，四个膜层参数使用 nm，Angle 使用 RefReflector 第一入射介质内的度数。
# BOUNDS 是优化器唯一允许的搜索区域。这里的中心点不能被当作“名义真值起点”强制加入；
# 所有起点均来自与真值无关的 Latin hypercube 和差分进化种群。

VERSION = "tmm_joint_inversion_v10"
C0_M_S = 299_792_458.0
INCIDENT_MEDIUM_N = 5.8284
MAX_WAVELENGTH_ERROR_NM = 0.2
EXPECTED_AXIS_SCENARIO = "fixed_reported_axis_hidden_true_pixel_centers"
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
PARAMS = ["Air", "HSQ", "PSS", "SOC", "TiO2", "Angle"]
FILM_PARAMS = ["HSQ", "PSS", "SOC", "TiO2"]
BOUNDS = {
    "Air": (95.0, 105.0), "HSQ": (20.0, 40.0), "PSS": (1.0, 20.0),
    "SOC": (30.0, 50.0), "TiO2": (30.0, 50.0), "Angle": (-0.1, 0.1),
}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]   
OUTPUT_ROOT = REPO_ROOT / "work" / "04_results_and_datasets"

# ---------------------------------------------------------------------------
# 反演配置
# ---------------------------------------------------------------------------
# stride 作用于局部最小二乘，global_stride 在局部数据基础上再次抽样用于差分进化。
# 当前 100 um 腔默认采用 0.02 nm 局部步长和 0.04 nm 全局残差步长，并由
# validate_sampling 对每个输入文件重新检查，而不是只相信配置中的名义采样间隔。
# full_ils 模式在全局和局部阶段都使用 0.002 nm + ILS；fast_no_ils 仅在差分进化
# 阶段直接计算全局 reported 轴上的 TMM，局部拟合、最终评价和 rank 仍使用严格模型。

@dataclass
class FitConfig:
    input_dir: str
    wavelength_min_nm: float = 450.0
    wavelength_max_nm: float = 580.0
    stride: int = 1
    global_stride: int = 2
    global_forward_model: str = "full_ils"
    global_popsize: int = 8
    global_maxiter: int = 40
    multistarts: int = 8
    max_nfev: int = 600
    local_gtol: float = 1.0e-5
    workers: int = 4
    random_seed: int = 20260825
    loss: str = "soft_l1"

# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------
# scalar 统一读取 NPZ 中的零维数组；parameter_unit 保证 CSV 字段名与实际单位一致。

def scalar(npz, key: str, default):
    return np.asarray(npz[key]).item() if key in npz else default

def parameter_unit(name: str) -> str:
    return "um" if name == "Air" else "deg" if name == "Angle" else "nm"

def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([BOUNDS[name][0] for name in PARAMS], dtype=float),
        np.asarray([BOUNDS[name][1] for name in PARAMS], dtype=float),
    )

# ---------------------------------------------------------------------------
# 名义材料模型与六参数 TMM
# ---------------------------------------------------------------------------
# 反演端故意不读取数据生成时的材料 n/k 扰动。这样 material 噪声案例会表现为真实的
# forward-model mismatch，而不是把仿真真值泄漏给拟合模型。

def material_n(name: str, wavelengths_um: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=float)
    if name == "RefReflector": return np.full_like(w, INCIDENT_MEDIUM_N, dtype=np.complex128)
    if name == "Air": return np.ones_like(w, dtype=np.complex128)
    if name == "HSQ": return np.full_like(w, 1.41, dtype=np.complex128)
    if name == "PSS": return np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    if name == "SOC": return (1.55 + 0.005 / (w**2)).astype(np.complex128)
    if name == "TiO2": return (2.4 + 0.02 / (w**2)).astype(np.complex128)
    if name == "Cu": return np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    raise ValueError(f"Unknown material: {name}")

def propagation_cosines(n_matrix: np.ndarray, reflector_angle_deg: float) -> np.ndarray:
    # reflector_angle_deg 定义在 RefReflector 内。先计算 Snell 不变量 n0*sin(theta0)，
    # 再得到 Air 和各膜层中的复数 cos(theta)，不能把空气角直接传入该函数。
    tangential_index = n_matrix[0] * np.sin(np.deg2rad(float(reflector_angle_deg)))
    cos_values = np.sqrt(1.0 - (tangential_index[None, :] / n_matrix) ** 2)
    cos_values[np.real(cos_values) < 0.0] *= -1.0
    return cos_values

def vector_to_thickness_um(values: np.ndarray) -> tuple[dict[str, float], float]:
    # 优化向量为了数值尺度把薄膜存成 nm，但 TMM 统一使用 um；Air 已经是 um，
    # Angle 不属于厚度字典，作为 reflector 入射角单独返回。
    values = np.asarray(values, dtype=float)
    return {
        "RefReflector": 0.0, "Air": float(values[0]), "HSQ": float(values[1]) / 1000.0,
        "PSS": float(values[2]) / 1000.0, "SOC": float(values[3]) / 1000.0,
        "TiO2": float(values[4]) / 1000.0, "Cu": 0.0,
    }, float(values[5])

def tmm_reflectance(
    wavelengths_um: np.ndarray,
    values: np.ndarray,
    n_matrix: np.ndarray | None = None,
    k0_m_inv: np.ndarray | None = None,
) -> np.ndarray:
    # 核心前向模型：在反演自行构造的固定绝对内部网格上计算 p 偏振反射率。
    # 禁止传入 NPZ 中隐藏的 true_pixel_center/physical 波长轴，否则会把待评估的
    # 光谱仪轴准确度真值泄漏给拟合模型。
    thicknesses_um, reflector_angle_deg = vector_to_thickness_um(values)
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    # n(lambda) 和 k0 只依赖固定内部轴。优化循环可从 SpectrometerForwardModel
    # 传入预计算数组；独立调用本函数时仍会在这里构造，保持原有接口可用。
    if n_matrix is None:
        n_matrix = np.vstack([material_n(name, wavelengths_um) for name in LAYER_NAMES])
    else:
        n_matrix = np.asarray(n_matrix, dtype=np.complex128)
        if n_matrix.shape != (len(LAYER_NAMES), len(wavelengths_um)):
            raise ValueError("Precomputed material matrix has an incompatible shape.")
    cos_values = propagation_cosines(n_matrix, reflector_angle_deg)
    q_values = n_matrix / cos_values
    if k0_m_inv is None:
        k0 = 2.0 * np.pi / (wavelengths_um * 1.0e-6)
    else:
        k0 = np.asarray(k0_m_inv, dtype=float)
        if k0.shape != wavelengths_um.shape:
            raise ValueError("Precomputed vacuum-wave-number array has an incompatible shape.")
    m11 = np.ones(len(wavelengths_um), dtype=complex)
    m12 = np.zeros(len(wavelengths_um), dtype=complex)
    m21 = np.zeros(len(wavelengths_um), dtype=complex)
    m22 = np.ones(len(wavelengths_um), dtype=complex)
    # m11~m22 是当前累计特征矩阵。Air、HSQ、PSS、SOC、TiO2 依次递推；
    # RefReflector 和 Cu 作为半无限边界，只进入入射/出射光学导纳，不乘厚度矩阵。
    for layer_index, name in enumerate(LAYER_NAMES[1:-1], start=1):
        thickness_m = thicknesses_um[name] * 1.0e-6
        if thickness_m <= 0.0: continue
        delta = k0 * n_matrix[layer_index] * cos_values[layer_index] * thickness_m
        c_delta, s_delta, q_layer = np.cos(delta), np.sin(delta), q_values[layer_index]
        a11, a12 = c_delta, -1j * s_delta / q_layer
        a21, a22 = -1j * q_layer * s_delta, c_delta
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21, m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21, m21 * a12 + m22 * a22,
        )
    # 用首末介质光学导纳把累计矩阵转换为复振幅反射系数，最后取 |r|^2。
    q0, qs = q_values[0], q_values[-1]
    numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
    denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
    return np.abs(numerator / denominator) ** 2

def source_power_envelope(
    wavelengths_nm: np.ndarray,
    generator_config: dict,
) -> np.ndarray:
    """重建生成端已知的名义参考光源包络，不读取 realization 的光源漂移真值。"""
    center_nm = float(generator_config["SOURCE_REFERENCE_CENTER_NM"])
    sigma_nm = float(generator_config["SOURCE_ENVELOPE_SIGMA_NM"])
    floor_rel = float(generator_config["SOURCE_ENVELOPE_FLOOR_REL"])
    gaussian = np.exp(
        -0.5 * ((np.asarray(wavelengths_nm, dtype=float) - center_nm) / sigma_nm) ** 2
    )
    envelope = floor_rel + (1.0 - floor_rel) * gaussian
    return envelope / float(np.max(envelope))


def quantum_efficiency(
    wavelengths_nm: np.ndarray,
    generator_config: dict,
) -> np.ndarray:
    """按 main_v10 CONFIG 重建已知名义 QE；后续有产品曲线时只需替换这里。"""
    settings = generator_config["SPECTROMETER"]
    if settings["QE_MODEL"] != "quadratic":
        raise ValueError(f"Unsupported QE model: {settings['QE_MODEL']}")
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    center_nm = float(settings["QE_CENTER_NM"])
    start_nm = float(generator_config["WAVELENGTH_START_UM"]) * 1000.0
    stop_nm = float(generator_config["WAVELENGTH_STOP_UM"]) * 1000.0
    half_span_nm = max(center_nm - start_nm, stop_nm - center_nm)
    normalized = np.clip((wavelengths_nm - center_nm) / half_span_nm, -1.0, 1.0)
    peak = float(settings["QE_PEAK"])
    edge = float(settings["QE_EDGE"])
    return edge + (peak - edge) * (1.0 - normalized**2)


class SpectrometerForwardModel:
    """六参数 TMM 与 V10 名义光谱仪响应的组合前向模型。

    该对象只保存实验已知配置和固定 reported 轴。它不接收 NPZ 中的隐藏真实像素中心、
    总轴误差、光源漂移 realization 或结构真值。模型假设真实像素中心等于 reported 轴，
    因而轴准确度噪声会作为真实模型失配反映到最终反演误差中。
    """

    def __init__(
        self,
        reported_wavelengths_um: np.ndarray,
        generator_config: dict,
        internal_margin_nm: float | None = None,
    ):
        self.reported_um = np.asarray(reported_wavelengths_um, dtype=float)
        self.reported_nm = self.reported_um * 1000.0
        self.generator_config = generator_config
        settings = generator_config["SPECTROMETER"]
        if not bool(settings["ILS_ENABLED"]):
            raise ValueError("V10 inversion expects the generator ILS to be enabled.")
        if settings["ILS_SHAPE"] != "gaussian":
            raise ValueError("Only the V10 Gaussian ILS is currently supported.")
        self.internal_step_nm = float(generator_config["INTERNAL_WAVELENGTH_STEP_NM"])
        self.ils_fwhm_nm = float(settings["ILS_FWHM_NM"])
        self.ils_sigma_nm = self.ils_fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        self.ils_truncate_sigma = float(settings["ILS_TRUNCATE_SIGMA"])
        if self.internal_step_nm <= 0.0:
            raise ValueError("Internal wavelength step must be positive.")
        if self.internal_step_nm > self.ils_fwhm_nm / 5.0:
            raise ValueError("Internal grid is too coarse for the configured ILS.")

        minimum_ils_margin_nm = self.ils_truncate_sigma * self.ils_sigma_nm
        # 生成端保存的 internal margin 是公开的数值网格配置：它只保证边缘像素有足够
        # 积分支撑，不包含某个 realization 的真实轴偏差。复用该值可避免边界卷积差异。
        margin_nm = (
            minimum_ils_margin_nm
            if internal_margin_nm is None
            else max(minimum_ils_margin_nm, float(internal_margin_nm))
        )
        self.internal_margin_nm = margin_nm
        internal_start_nm = float(self.reported_nm[0]) - margin_nm
        internal_stop_nm = float(self.reported_nm[-1]) + margin_nm
        count = int(
            np.ceil((internal_stop_nm - internal_start_nm) / self.internal_step_nm)
        ) + 1
        self.internal_nm = (
            internal_start_nm + np.arange(count) * self.internal_step_nm
        )
        self.internal_um = self.internal_nm / 1000.0
        # 这些数组在一个 NPZ 的全部全局/局部候选之间恒定，预计算可减少重复分配，
        # 但不改变材料公式、角度定义或内部波长采样。
        self.n_matrix = np.vstack([
            material_n(name, self.internal_um) for name in LAYER_NAMES
        ])
        self.k0_m_inv = 2.0 * np.pi / (self.internal_um * 1.0e-6)

        # 样品和参考在名义模型中使用同一光源。随机中心漂移/功率谱变化不读取真值，
        # 因此 source 类案例会自然表现为模型失配。
        source = source_power_envelope(self.internal_nm, generator_config)
        wavelength_m = self.internal_nm * 1.0e-9
        photon_energy_j = 6.626_070_15e-34 * C0_M_S / wavelength_m
        self.electron_weight = (
            source
            * float(settings["OPTICAL_THROUGHPUT"])
            * quantum_efficiency(self.internal_nm, generator_config)
            / photon_energy_j
        )
        self.reference_sampled = self._convolve_and_sample(self.electron_weight)
        if np.any(self.reference_sampled <= 0.0):
            raise ValueError("Nominal reference response must remain positive.")
        reference_exposure_s = float(settings["REFERENCE_EXPOSURE_S"])
        sample_exposure_s = float(settings["SAMPLE_EXPOSURE_S"])
        if reference_exposure_s <= 0.0 or sample_exposure_s <= 0.0:
            raise ValueError("Sample and reference exposure times must be positive.")
        # main_v10 的 measured=(sample-dark)/(reference-dark) 未额外做曝光归一化，
        # 因而已知的样品/参考曝光时间比例必须显式保留在名义前向模型中。
        self.exposure_ratio = sample_exposure_s / reference_exposure_s

    def _convolve_and_sample(self, values: np.ndarray) -> np.ndarray:
        filtered = gaussian_filter1d(
            np.asarray(values, dtype=float),
            sigma=self.ils_sigma_nm / self.internal_step_nm,
            mode="nearest",
            truncate=self.ils_truncate_sigma,
        )
        # 反演看不到隐藏真实中心，只能在固定 reported 像素中心建立名义响应模型。
        return np.interp(self.reported_nm, self.internal_nm, filtered)

    def predict(self, values: np.ndarray) -> np.ndarray:
        reflectance_internal = tmm_reflectance(
            self.internal_um,
            values,
            n_matrix=self.n_matrix,
            k0_m_inv=self.k0_m_inv,
        )
        sample_sampled = self._convolve_and_sample(
            self.electron_weight * reflectance_internal
        )
        return (
            self.exposure_ratio
            * sample_sampled
            / np.maximum(self.reference_sampled, 1.0e-30)
        )


def reflector_to_air_angle_deg(reflector_angle_deg: float) -> float:
    # 该换算仅用于结果报告，不参与 TMM 前向计算。反演参数始终是 reflector 内角度。
    invariant = INCIDENT_MEDIUM_N * math.sin(math.radians(float(reflector_angle_deg)))
    if abs(invariant) > 1.0: raise ValueError("Reflector angle has no Air-layer solution.")
    return math.degrees(math.asin(invariant))

# ---------------------------------------------------------------------------
# V10 输入契约与预处理
# ---------------------------------------------------------------------------
# 实验可见输入只有固定 reported_wavelengths 和 spectrum_measured。NPZ 中可选保存的
# true_pixel_center_wavelengths 及其轴误差曲线属于隐藏仿真真值，本函数既不读取也不返回，
# 因而后续 model 和 residual 无法使用真实像素响应中心修正拟合。
#
# spectrum_measured 已由生成端通过独立 sample/reference/dark 计数执行参考归一化，可能因
# 光源失配、ADC 量化或探测器误差略微超出 [0,1]；这里不裁剪、不平滑、不再次归一化，
# 避免改变原始噪声统计。

def load_fit_input(npz_path: Path, config: FitConfig) -> dict:
    """只加载实验可见量和已知仪器配置，不返回任何隐藏轴或结构真值数组。"""
    with np.load(npz_path, allow_pickle=False) as data:
        if bool(scalar(data, "time_series_enabled", False)):
            raise ValueError("Static data required.")
        if int(scalar(data, "frames_per_realization", 1)) != 1:
            raise ValueError("One frame required.")
        if bool(scalar(data, "modulation_enabled", False)):
            raise ValueError("No modulation accepted.")
        generator_version = str(scalar(data, "generator_version", "unknown"))
        if generator_version != "main_v10":
            raise ValueError(
                f"V10 inversion requires a main_v10 dataset, got {generator_version!r}."
            )

        required = {
            "wavelengths",
            "reported_wavelengths_nm",
            "spectrum_measured",
            "config_json",
            "wavelength_axis_scenario",
            "reported_axis_is_fixed",
            "ils_enabled",
            "ils_fwhm_nm",
            "internal_wavelength_step_nm",
            "internal_wavelength_margin_nm",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"V10 NPZ is missing required fields: {missing}")

        reported_um = np.asarray(data["wavelengths"], dtype=float)
        reported_nm = np.asarray(data["reported_wavelengths_nm"], dtype=float)
        spectrum = np.asarray(data["spectrum_measured"], dtype=float)
        if (
            reported_um.ndim != 1
            or reported_nm.shape != reported_um.shape
            or spectrum.shape != reported_um.shape
        ):
            raise ValueError(
                "V10 reported wavelength and measured-spectrum arrays must be matching 1D arrays."
            )
        if not np.all(np.isfinite(reported_um)) or not np.all(np.isfinite(spectrum)):
            raise ValueError("V10 reported axis and measured spectrum must be finite.")
        if np.any(np.diff(reported_um) <= 0.0):
            raise ValueError("Fixed reported wavelength axis must be monotonic.")
        if not np.allclose(reported_um * 1000.0, reported_nm, rtol=0.0, atol=1.0e-10):
            raise ValueError("NPZ wavelengths and reported_wavelengths_nm disagree.")
        if not bool(scalar(data, "reported_axis_is_fixed", False)):
            raise ValueError("V10 inversion requires reported_axis_is_fixed=True.")
        scenario = str(scalar(data, "wavelength_axis_scenario", "unknown"))
        if scenario != EXPECTED_AXIS_SCENARIO:
            raise ValueError(f"Unexpected wavelength-axis scenario: {scenario!r}.")
        if not bool(scalar(data, "ils_enabled", False)):
            raise ValueError("V10 spectrometer inversion requires ILS-enabled data.")

        try:
            generator_config = json.loads(str(scalar(data, "config_json", "{}")))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid config_json in V10 NPZ.") from exc
        if generator_config.get("WAVELENGTH_AXIS_SCENARIO") != EXPECTED_AXIS_SCENARIO:
            raise ValueError("config_json wavelength-axis scenario disagrees with NPZ fields.")
        settings = generator_config.get("SPECTROMETER", {})
        if settings.get("REPORTED_AXIS_POLICY") != "fixed_factory_calibration":
            raise ValueError("V10 inversion requires the fixed factory-calibration axis policy.")
        if settings.get("ILS_SHAPE") != "gaussian":
            raise ValueError("Only Gaussian V10 ILS data are supported.")
        if not np.isclose(
            float(generator_config["INTERNAL_WAVELENGTH_STEP_NM"]),
            float(scalar(data, "internal_wavelength_step_nm", np.nan)),
        ):
            raise ValueError("Internal wavelength step disagrees with config_json.")
        if not np.isclose(
            float(settings["ILS_FWHM_NM"]),
            float(scalar(data, "ils_fwhm_nm", np.nan)),
        ):
            raise ValueError("ILS FWHM disagrees with config_json.")

        mask = (
            (reported_nm >= config.wavelength_min_nm)
            & (reported_nm <= config.wavelength_max_nm)
        )
        indices = np.where(mask)[0][::max(1, int(config.stride))]
        if len(indices) < 50:
            raise ValueError("Too few samples after mask and stride.")
        metadata = {
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "noise_factor": str(scalar(data, "noise_factor", "unknown")),
            "noise_level": str(scalar(data, "noise_level", "unknown")),
            "realization_index": int(scalar(data, "realization_index", 0)),
            "random_seed": int(scalar(data, "random_seed", 0)),
            "generator_version": generator_version,
            "optical_backend": str(scalar(data, "optical_backend", "unknown")),
            "wavelength_axis_source": "fixed_reported_wavelengths",
            "wavelength_axis_scenario": scenario,
            "reported_axis_is_fixed": True,
            "hidden_true_pixel_centers_used_for_fit": False,
            "hidden_axis_error_used_for_fit": False,
            "ils_enabled": True,
            "ils_fwhm_nm": float(settings["ILS_FWHM_NM"]),
            "internal_wavelength_step_nm": float(
                generator_config["INTERNAL_WAVELENGTH_STEP_NM"]
            ),
            "internal_wavelength_margin_nm": float(
                scalar(data, "internal_wavelength_margin_nm", np.nan)
            ),
        }
    selected_reported_um = reported_um[indices]
    return {
        "path": str(npz_path.resolve()),
        "wavelengths_um": selected_reported_um,
        "spectrum": spectrum[indices],
        "actual_step_nm": float(np.median(np.diff(selected_reported_um))) * 1000.0,
        "generator_config": generator_config,
        "metadata": metadata,
    }


# 真值读取函数与 load_fit_input 分离。主流程只有在 fit_measurement 完成全局搜索、
# 局部收敛筛选和最优结果排序之后才调用它，防止真值影响起点或 rank。

def load_evaluation_truth(npz_path: Path) -> tuple[dict[str, float], dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        names = [str(value) for value in np.asarray(data["layer_names"])]
        layers = dict(zip(names, np.asarray(data["layer_thickness_um"], dtype=float)))
        truth = {
            "Air": float(layers["Air"]), "HSQ": float(layers["HSQ"]) * 1000.0,
            "PSS": float(layers["PSS"]) * 1000.0, "SOC": float(layers["SOC"]) * 1000.0,
            "TiO2": float(layers["TiO2"]) * 1000.0,
            "Angle": float(scalar(data, "true_reflector_angle_deg", 0.0)),
        }
        audit = json.loads(str(scalar(data, "noise_realization_json", "{}")))
    return truth, audit

# ---------------------------------------------------------------------------
# 长腔采样充分性检查
# ---------------------------------------------------------------------------
# Air 腔的最短近似条纹周期出现在最短波长和最大允许腔长：
#   Delta_lambda_min ≈ lambda_min^2 / (2*L_max)
# Nyquist 要求采样间隔不大于该周期的一半。局部和全局两套轴都必须通过检查。

def validate_sampling(measurement: dict, config: FitConfig) -> dict:
    shortest_period_nm = config.wavelength_min_nm**2 / (2.0 * BOUNDS["Air"][1] * 1000.0)
    nyquist_max_step_nm = 0.5 * shortest_period_nm
    local_step_nm = measurement["actual_step_nm"]
    global_step_nm = local_step_nm * max(1, int(config.global_stride))
    if local_step_nm > nyquist_max_step_nm or global_step_nm > nyquist_max_step_nm:
        raise ValueError(
            f"Sampling violates Nyquist: local={local_step_nm:.6g}, global={global_step_nm:.6g}, "
            f"limit={nyquist_max_step_nm:.6g} nm."
        )
    return {"local_step_nm": local_step_nm, "global_step_nm": global_step_nm,
            "shortest_fringe_period_nm": shortest_period_nm,
            "nyquist_max_step_nm": nyquist_max_step_nm}


# ---------------------------------------------------------------------------
# 目标函数与全局搜索辅助函数
# ---------------------------------------------------------------------------
# robust_scale 用中位数绝对偏差估计光谱尺度，避免残差权重随整体反射率幅值改变。
# robust_objective 在差分进化阶段复现 SciPy 各 robust loss 的标量代价。

def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, float(np.std(values)), 1.0e-4)

def robust_objective(residual_values: np.ndarray, loss: str) -> float:
    z = np.asarray(residual_values, dtype=float) ** 2
    if loss == "linear": rho = z
    elif loss == "soft_l1": rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber": rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy": rho = np.log1p(z)
    elif loss == "arctan": rho = np.arctan(z)
    else: raise ValueError(f"Unknown loss: {loss}")
    return float(np.sum(rho))

# Latin hypercube 在每个参数维度均匀覆盖 [0,1]，再映射到物理边界。
# 它只依赖边界和随机 seed，不包含结构真值、名义真值或强制边界中心。

def latin_hypercube_population(seed: int, size: int) -> np.ndarray:
    lower, upper = bounds_arrays()
    sample = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n=size)
    return qmc.scale(sample, lower, upper)

# 从差分进化后的种群按能量升序挑选局部起点，同时要求归一化参数距离至少为 0.03。
# 这样 multistart 不是在同一极小值附近重复启动，而是保留多个有代表性的候选盆地。

def select_diverse_candidates(
    population: np.ndarray,
    energies: np.ndarray,
    count: int,
) -> list[tuple[np.ndarray, int, float]]:
    lower, upper = bounds_arrays()
    span = upper - lower
    selected = []
    for index in np.argsort(energies):
        candidate = np.asarray(population[index], dtype=float)
        scaled = (candidate - lower) / span
        if all(np.linalg.norm(scaled - previous[0]) >= 0.03 for previous in selected):
            selected.append((scaled, int(index), float(energies[index])))
        if len(selected) >= count: break
    return [(lower + scaled * span, index, energy) for scaled, index, energy in selected]

# 最优解处使用有界中心差分近似 Jacobian，并通过奇异值估计条件数。
# 条件数很大说明参数不可辨识或高度相关，即使光谱 RMSE 很小也不能直接认为参数可靠。

def approximate_jacobian(residual_fn, values: np.ndarray) -> np.ndarray:
    lower, upper = bounds_arrays()
    span = upper - lower
    jacobian = np.empty((len(residual_fn(values)), len(values)), dtype=float)
    for index in range(len(values)):
        step = max(1.0e-6 * span[index], 1.0e-8)
        plus, minus = values.copy(), values.copy()
        plus[index] = min(upper[index], plus[index] + step)
        minus[index] = max(lower[index], minus[index] - step)
        denominator = plus[index] - minus[index]
        jacobian[:, index] = (residual_fn(plus) - residual_fn(minus)) / denominator
    return jacobian

def fit_measurement(measurement: dict, config: FitConfig, seed: int) -> dict:
    # measurement 只包含固定 reported 轴、测量光谱和已知仪器 CONFIG，不包含结构真值、
    # 隐藏像素中心、轴误差曲线或噪声 realization。
    wavelengths = measurement["wavelengths_um"]
    observed = measurement["spectrum"]
    scale = robust_scale(observed)
    global_indices = np.arange(
        0, len(wavelengths), max(1, int(config.global_stride))
    )
    global_wavelengths = wavelengths[global_indices]
    global_observed = observed[global_indices]
    lower, upper = bounds_arrays()
    if config.global_forward_model not in {"full_ils", "fast_no_ils"}:
        raise ValueError(
            "global_forward_model must be either 'full_ils' or 'fast_no_ils'."
        )

    # 光谱仪模型只构建一次，内部绝对网格、名义源谱、QE 和参考响应均可复用。
    # 每次优化评估只需重新计算与六个结构参数有关的 TMM 反射率和样品 ILS 卷积。
    instrument_model = SpectrometerForwardModel(
        wavelengths,
        measurement["generator_config"],
        measurement["metadata"]["internal_wavelength_margin_nm"],
    )

    # fast_no_ils 的样品/参考名义源谱与 QE 在同一 reported 波长处相消，因此快速
    # 全局模型等于 TMM 反射率乘已知曝光时间比。材料矩阵和 k0 同样预计算，避免每个
    # 差分进化候选重复分配。该代理模型绝不用于局部 cost、最终 RMSE 或结果排名。
    if config.global_forward_model == "fast_no_ils":
        global_n_matrix = np.vstack([
            material_n(name, global_wavelengths) for name in LAYER_NAMES
        ])
        global_k0_m_inv = 2.0 * np.pi / (global_wavelengths * 1.0e-6)
        settings = measurement["generator_config"]["SPECTROMETER"]
        global_exposure_ratio = (
            float(settings["SAMPLE_EXPOSURE_S"])
            / float(settings["REFERENCE_EXPOSURE_S"])
        )
    else:
        global_n_matrix = None
        global_k0_m_inv = None
        global_exposure_ratio = 1.0

    def model(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        if use_global and config.global_forward_model == "fast_no_ils":
            return global_exposure_ratio * tmm_reflectance(
                global_wavelengths,
                values,
                n_matrix=global_n_matrix,
                k0_m_inv=global_k0_m_inv,
            )
        full_prediction = instrument_model.predict(values)
        return full_prediction[global_indices] if use_global else full_prediction

    def residual(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        target = global_observed if use_global else observed
        return (model(values, use_global) - target) / scale

    started = time.perf_counter()
    global_started = time.perf_counter()
    # 阶段一：生成与真值无关的 Latin hypercube 初始种群，并记录进化前能量，
    # 便于在 global_search_summary.jsonl 中审计全局优化是否真的改善候选解。
    population_size = max(5, int(config.global_popsize) * len(PARAMS))
    initial_population = latin_hypercube_population(seed, population_size)
    initial_energies = np.asarray([
        robust_objective(residual(values, True), config.loss) for values in initial_population
    ])
    # 差分进化在原始物理参数边界内运行；polish=False，避免在候选筛选前隐式做一次
    # 未记录的局部优化。正式局部优化统一在后面的 least_squares 阶段执行。
    global_result = differential_evolution(
        lambda values: robust_objective(residual(values, True), config.loss),
        bounds=[BOUNDS[name] for name in PARAMS],
        strategy="best1bin",
        maxiter=int(config.global_maxiter),
        popsize=int(config.global_popsize),
        tol=1.0e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        init=initial_population,
        updating="immediate",
        workers=1,
    )
    global_runtime_s = time.perf_counter() - global_started
    evolved_population = np.asarray(global_result.population, dtype=float)
    evolved_energies = np.asarray(global_result.population_energies, dtype=float)
    candidates = select_diverse_candidates(
        evolved_population, evolved_energies, int(config.multistarts)
    )
    if not candidates:
        raise RuntimeError("Global search returned no candidate starts.")

    span = upper - lower
    angle_index = PARAMS.index("Angle")

    # 局部优化在 [0,1]^6 中运行，减少 um、nm、deg 数值尺度差异。
    # Angle 维使用平方/平方根构成可逆映射，只改变局部步长分布，不改变物理边界。
    def physical_to_unit(values: np.ndarray) -> np.ndarray:
        unit_values = np.clip((np.asarray(values, dtype=float) - lower) / span, 0.0, 1.0)
        unit_values[angle_index] = unit_values[angle_index] ** 2
        return unit_values

    def unit_to_physical(unit_values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(unit_values, dtype=float).copy()
        transformed[angle_index] = np.sqrt(max(0.0, transformed[angle_index]))
        return lower + transformed * span

    # 阶段二：无论全局阶段使用哪种模式，所有候选都进入 0.002 nm + ILS 严格模型。
    # 最终候选排序只依据这里产生的严格局部 cost。
    local_started = time.perf_counter()
    attempts = []
    for start_rank, (x0, population_index, global_energy) in enumerate(candidates, start=1):
        x0_unit = physical_to_unit(x0)
        local = least_squares(
            lambda unit_values: residual(unit_to_physical(unit_values), False),
            x0=x0_unit,
            bounds=(np.zeros(len(PARAMS)), np.ones(len(PARAMS))),
            loss=config.loss,
            max_nfev=int(config.max_nfev),
            x_scale=1.0,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=float(config.local_gtol),
        )
        fitted_values = unit_to_physical(local.x)
        fitted = model(fitted_values, False)
        attempts.append({
            "start_rank": start_rank,
            "success": bool(local.success) and np.isfinite(local.cost),
            "message": str(local.message),
            "status": int(local.status),
            "optimality": float(local.optimality),
            "cost": float(local.cost),
            "rmse_reflectance": float(np.sqrt(np.mean((fitted - observed) ** 2))),
            "nfev": int(local.nfev),
            "x0": np.asarray(x0, dtype=float),
            "x": fitted_values,
            "global_population_index": int(population_index),
            "global_energy": float(global_energy),
        })
    # 先按 SciPy 收敛状态筛选，再按 cost 排序。达到 max_nfev 但未收敛的结果即使
    # cost 更小也不能成为 rank 1，从实现上修复“未收敛结果被选为最优解”的问题。
    local_runtime_s = time.perf_counter() - local_started
    attempts.sort(key=lambda row: (not row["success"], row["cost"]))
    successful = [row for row in attempts if row["success"]]
    runtime_s = time.perf_counter() - started
    if not successful:
        return {
            "success": False,
            "runtime_s": runtime_s,
            "global_runtime_s": global_runtime_s,
            "local_runtime_s": local_runtime_s,
            "attempts": attempts,
            "global_summary": {
                "global_forward_model": config.global_forward_model,
                "global_runtime_s": global_runtime_s,
                "local_runtime_s": local_runtime_s,
                "global_nit": int(global_result.nit),
                "global_nfev": int(global_result.nfev),
            },
        }
    best = successful[0]
    fitted_values = np.asarray(best["x"], dtype=float)
    # 只对已收敛的最佳结果计算 Jacobian、条件数和边界命中，用于识别假收敛和不可辨识。
    jacobian = approximate_jacobian(lambda values: residual(values, False), fitted_values)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).eps]
    condition_number = float(positive[0] / positive[-1]) if len(positive) else float("inf")
    tolerance = 1.0e-5 * (upper - lower)
    boundary_hits = [
        PARAMS[index] for index, value in enumerate(fitted_values)
        if value - lower[index] <= tolerance[index] or upper[index] - value <= tolerance[index]
    ]
    return {
        "success": True,
        "runtime_s": runtime_s,
        "global_runtime_s": global_runtime_s,
        "local_runtime_s": local_runtime_s,
        "x": fitted_values,
        "cost": float(best["cost"]),
        "rmse_reflectance": float(best["rmse_reflectance"]),
        "nfev": int(best["nfev"]),
        "attempts": attempts,
        "condition_number": condition_number,
        "singular_values": singular_values.tolist(),
        "boundary_hits": boundary_hits,
        "fitted_spectrum": model(fitted_values, False),
        "global_summary": {
            "global_forward_model": config.global_forward_model,
            "global_uses_ils": config.global_forward_model == "full_ils",
            "global_runtime_s": global_runtime_s,
            "local_runtime_s": local_runtime_s,
            "initial_population_source": "LatinHypercube; no truth, nominal truth, or forced center point",
            "internal_wavelength_point_count": int(len(instrument_model.internal_nm)),
            "internal_wavelength_step_nm": float(instrument_model.internal_step_nm),
            "internal_wavelength_margin_nm": float(instrument_model.internal_margin_nm),
            "ils_fwhm_nm": float(instrument_model.ils_fwhm_nm),
            "global_residual_point_count": int(len(global_indices)),
            "local_residual_point_count": int(len(wavelengths)),
            "global_stride_reduces_internal_forward_cost": (
                config.global_forward_model == "fast_no_ils"
            ),
            "initial_population_size": int(len(initial_population)),
            "initial_best_energy": float(np.min(initial_energies)),
            "initial_median_energy": float(np.median(initial_energies)),
            "evolved_population_size": int(len(evolved_population)),
            "evolved_best_energy": float(np.min(evolved_energies)),
            "evolved_median_energy": float(np.median(evolved_energies)),
            "global_success": bool(global_result.success),
            "global_message": str(global_result.message),
            "global_nit": int(global_result.nit),
            "global_nfev": int(global_result.nfev),
        },
    }


# ---------------------------------------------------------------------------
# 结果整理与误差统计
# ---------------------------------------------------------------------------
# result_row 只在拟合结束后把 fit 与 truth 合并。Air 误差从 um 转换为 nm；
# film_mae_nm 是四层膜厚绝对误差的算术平均；空气层角度仅作为派生报告字段。

def result_row(
    measurement: dict,
    fit: dict,
    truth: dict[str, float],
    noise_audit: dict | None = None,
) -> dict:
    metadata = measurement["metadata"]
    row = {
        "input_npz": measurement["path"],
        **metadata,
        "success": bool(fit["success"]),
        "global_forward_model": str(
            fit.get("global_summary", {}).get("global_forward_model", "unknown")
        ),
        "fit_runtime_s": float(fit["runtime_s"]),
        "global_runtime_s": float(fit.get("global_runtime_s", np.nan)),
        "local_runtime_s": float(fit.get("local_runtime_s", np.nan)),
    }
    # 这些噪声真值是在拟合、收敛筛选和 rank 排序结束后才追加到结果表，仅供误差分组。
    audit = noise_audit or {}
    for key in (
        "source_center_drift_nm",
        "source_power_curve_peak_rel",
        "axis_offset_nm",
        "axis_scale_ppm",
        "thermal_drift_nm",
        "calibration_residual_peak_nm",
        "spectrometer_axis_error_max_abs_nm",
        "spectrometer_axis_error_rms_nm",
        "detector_noise_multiplier",
    ):
        if key in audit:
            row[key] = audit[key]
    if not fit["success"]:
        return row
    for index, name in enumerate(PARAMS):
        unit = parameter_unit(name)
        value = float(fit["x"][index])
        row[f"fit_{name}_{unit}"] = value
        row[f"truth_{name}_{unit}"] = float(truth[name])
        row[f"error_{name}_{unit}"] = value - float(truth[name])
    row["cavity_error_nm"] = row["error_Air_um"] * 1000.0
    row["cavity_abs_error_nm"] = abs(row["cavity_error_nm"])
    row["film_mae_nm"] = float(np.mean([abs(row[f"error_{name}_nm"]) for name in FILM_PARAMS]))
    row["angle_abs_error_deg"] = abs(row["error_Angle_deg"])
    row["fit_air_angle_deg"] = reflector_to_air_angle_deg(row["fit_Angle_deg"])
    row["truth_air_angle_deg"] = reflector_to_air_angle_deg(row["truth_Angle_deg"])
    row.update({
        "cost": fit["cost"], "rmse_reflectance": fit["rmse_reflectance"],
        "nfev": fit["nfev"], "condition_number": fit["condition_number"],
        "boundary_hits": ";".join(fit["boundary_hits"]),
        "boundary_hit_count": len(fit["boundary_hits"]),
        "selected_start_rank": fit["attempts"][0]["start_rank"],
    })
    return row

# multistart_results.csv 保留每个局部起点 x0、收敛结果、全局种群来源和 rank。
# benchmark_error 字段仅用于事后评估，不参与 attempts 的排序。

def attempt_rows(measurement: dict, fit: dict, truth: dict[str, float]) -> list[dict]:
    rows = []
    for rank, attempt in enumerate(fit.get("attempts", []), start=1):
        row = {
            "input_npz": measurement["path"],
            "noise_case": measurement["metadata"]["noise_case"],
            "realization_index": measurement["metadata"]["realization_index"],
            "rank": rank, "success": attempt["success"], "status": attempt["status"],
            "message": attempt["message"], "optimality": attempt["optimality"],
            "cost": attempt["cost"], "rmse_reflectance": attempt["rmse_reflectance"],
            "nfev": attempt["nfev"],
            "global_population_index": attempt["global_population_index"],
            "global_energy": attempt["global_energy"],
        }
        for index, name in enumerate(PARAMS):
            unit = parameter_unit(name)
            row[f"x0_{name}_{unit}"] = float(attempt["x0"][index])
            row[f"fit_{name}_{unit}"] = float(attempt["x"][index])
            row[f"benchmark_error_{name}_{unit}"] = float(attempt["x"][index] - truth[name])
        rows.append(row)
    return rows

def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_无可用汇总结果。_"
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 按噪声案例汇总 bias、MAE、RMSE、95% 分位误差、边界命中率和计算时间。
# 汇总前先排除未收敛行，避免失败结果被当作精度样本。

def summarize_results(table: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for case, group in table.groupby("noise_case", sort=False):
        successful = group[group["success"]].copy()
        row = {
            "noise_case": case,
            "noise_factor": str(group.iloc[0]["noise_factor"]),
            "noise_level": str(group.iloc[0]["noise_level"]),
            "runs": int(len(group)),
            "successful_runs": int(len(successful)),
            "fit_success_rate": float(len(successful) / len(group)),
        }
        if not successful.empty:
            for name in PARAMS:
                unit = parameter_unit(name)
                errors = successful[f"error_{name}_{unit}"].to_numpy(dtype=float)
                if name == "Air": errors = errors * 1000.0; output_unit = "nm"
                else: output_unit = unit
                row[f"{name}_bias_{output_unit}"] = float(np.mean(errors))
                row[f"{name}_mae_{output_unit}"] = float(np.mean(np.abs(errors)))
                row[f"{name}_rmse_{output_unit}"] = float(np.sqrt(np.mean(errors**2)))
                row[f"{name}_p95_abs_{output_unit}"] = float(np.percentile(np.abs(errors), 95.0))
            row.update({
                "film_mae_nm_mean": float(successful["film_mae_nm"].mean()),
                "film_mae_nm_p95": float(successful["film_mae_nm"].quantile(0.95)),
                "cavity_max_abs_error_nm": float(successful["cavity_abs_error_nm"].max()),
                "angle_max_abs_error_deg": float(successful["angle_abs_error_deg"].max()),
                "fit_runtime_mean_s": float(successful["fit_runtime_s"].mean()),
                "fit_runtime_median_s": float(successful["fit_runtime_s"].median()),
                "fit_runtime_p95_s": float(successful["fit_runtime_s"].quantile(0.95)),
                "boundary_hit_rate": float((successful["boundary_hit_count"] > 0).mean()),
                "condition_number_median": float(successful["condition_number"].median()),
            })
        summaries.append(row)
    return pd.DataFrame(summaries)

def save_plots(output_dir: Path, table: pd.DataFrame, summary: pd.DataFrame, representatives: dict) -> list[str]:
    paths = []
    successful = table[table["success"]].copy()
    if successful.empty: return paths
    cases = successful["noise_case"].drop_duplicates().tolist()
    values = [successful.loc[successful["noise_case"] == case, "cavity_error_nm"].to_numpy() for case in cases]
    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    ax.boxplot(values, labels=cases, showfliers=True)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Cavity error (nm)")
    ax.set_title("V10 spectrometer-aware six-parameter inversion")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / "cavity_error_boxplot.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    factors = [value for value in summary["noise_factor"].drop_duplicates() if value != "clean"]
    levels = ["typical", "in_spec", "spec_limit", "out_of_spec_stress"]
    metrics = [
        ("Air_mae_nm", "Cavity MAE (nm)"),
        ("film_mae_nm_mean", "Film MAE (nm)"),
        ("Angle_mae_deg", "Reflector angle MAE (deg)"),
        ("fit_runtime_mean_s", "Mean inversion runtime (s)"),
    ]
    if factors:
        fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
        for ax, (column, title) in zip(axes.flat, metrics):
            matrix = np.full((len(factors), len(levels)), np.nan)
            for i, factor in enumerate(factors):
                for j, level in enumerate(levels):
                    match = summary[
                        (summary["noise_factor"] == factor)
                        & (summary["noise_level"] == level)
                    ]
                    if not match.empty and column in match:
                        matrix[i, j] = float(match.iloc[0][column])
            image = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(levels)), labels=levels)
            ax.set_yticks(range(len(factors)), labels=factors)
            ax.set_title(title)
            for i in range(len(factors)):
                for j in range(len(levels)):
                    if np.isfinite(matrix[i, j]):
                        ax.text(
                            j, i, f"{matrix[i, j]:.3g}",
                            ha="center", va="center", color="white",
                        )
            fig.colorbar(image, ax=ax, shrink=0.8)
        path = output_dir / "noise_factor_accuracy_heatmaps.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    if representatives:
        preferred = ["clean"] + [f"combined_{level}" for level in levels]
        selected_names = [name for name in preferred if name in representatives]
        if not selected_names:
            selected_names = list(representatives)[:8]
        fig, axes = plt.subplots(
            len(selected_names), 1,
            figsize=(13, max(5, 2.8 * len(selected_names))),
            squeeze=False, constrained_layout=True,
        )
        for index, name in enumerate(selected_names):
            item = representatives[name]; ax = axes[index, 0]
            ax.plot(item["wavelengths_um"] * 1000.0, item["observed"], lw=0.7, label="measured")
            ax.plot(item["wavelengths_um"] * 1000.0, item["fitted"], lw=0.7, label="fit")
            ax.set_title(name); ax.set_ylabel("Reflectance"); ax.grid(True, alpha=0.3); ax.legend()
        axes[-1, 0].set_xlabel("Fixed reported wavelength (nm)")
        path = output_dir / "representative_fits.png"
        fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths



# 单文件任务函数可在线程池中并行执行。它只完成输入加载、采样审计和拟合；
# 真值由父流程在 outcome 返回之后读取，进一步隔离拟合阶段和评估阶段。

def process_fit_task(payload: tuple[int, str, FitConfig]) -> dict:
    index, npz_path_text, config = payload
    npz_path = Path(npz_path_text)
    try:
        measurement = load_fit_input(npz_path, config)
        sampling_audit = validate_sampling(measurement, config)
        fit = fit_measurement(measurement, config, config.random_seed + index * 1009)
        return {
            "ok": True,
            "index": index,
            "npz_path": npz_path_text,
            "measurement": measurement, # 拟合前的测量数据，包括波长、光谱、元数据等
            "sampling_audit": sampling_audit,
            "fit": fit,
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": index,
            "npz_path": npz_path_text,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# 命令行与批处理入口
# ---------------------------------------------------------------------------
# V10 不设置隐式默认数据目录，正式运行必须通过 --input-dir 或 --inputs 明确指定 NPZ。
# 这样可以避免误把历史 V8/V9 或未保存配置生成的数据送入当前反演。
# workers 只并行不同 NPZ；单个 NPZ 内的差分进化固定 workers=1，保证随机过程可复现。

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10 fixed-reported-axis spectrometer-aware six-parameter TMM inversion."
    )
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\static_stackrt_v10_20260825_232804"))
    parser.add_argument("--pattern", default="static_spectrum_*.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--wavelength-min-nm", type=float, default=450.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--global-stride", type=int, default=2)
    parser.add_argument(
        "--global-forward-model",
        choices=["full_ils", "fast_no_ils"],
        default="full_ils",
        help=(
            "Global differential-evolution model: full_ils uses the strict 0.002 nm "
            "spectrometer model; fast_no_ils uses direct TMM on global reported samples. "
            "Local fitting and final ranking always use full ILS."
        ),
    )
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--local-gtol", type=float, default=1.0e-5)
    parser.add_argument("--workers", type=int, default=2) # 并行处理多个npz文件
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.inputs:
        inputs = [Path(value).resolve() for value in args.inputs]
        input_label = "explicit_inputs"
    else:
        if not args.input_dir: raise ValueError("Provide --input-dir or explicit --inputs.")
        input_dir = Path(args.input_dir).resolve()
        inputs = sorted(input_dir.glob(args.pattern))
        input_label = str(input_dir)
    if args.max_files is not None: inputs = inputs[:args.max_files]
    if not inputs: raise FileNotFoundError("No V10 static NPZ files matched.")
    config = FitConfig(
        input_dir=input_label,
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        stride=args.stride,
        global_stride=args.global_stride,
        global_forward_model=args.global_forward_model,
        global_popsize=args.global_popsize,
        global_maxiter=args.global_maxiter,
        multistarts=args.multistarts,
        max_nfev=args.max_nfev,
        local_gtol=args.local_gtol,
        workers=max(1, int(args.workers)),
        random_seed=args.random_seed,
        loss=args.loss,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else OUTPUT_ROOT / f"{VERSION}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"GLOBAL_FORWARD_MODEL={config.global_forward_model}; "
        "LOCAL_FORWARD_MODEL=full_ils_0.002nm",
        flush=True,
    )
    rows, all_attempts, failures, representatives = [], [], [], {}
    sampling_audits, global_summaries = [], []
    tasks = [(index, str(path), config) for index, path in enumerate(inputs)]
    if config.workers == 1:
        outcomes = map(process_fit_task, tasks)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=config.workers)
        outcomes = executor.map(process_fit_task, tasks)
    try:
        for outcome in outcomes: # outcomes是一个dict，包含每个npz文件的处理结果
            index = int(outcome["index"])
            npz_path = Path(outcome["npz_path"])
            if not outcome["ok"]:
                failures.append({"input": str(npz_path), "error": outcome["error"]})
                print(f"ERROR {npz_path}: {outcome['error']}", flush=True)
                continue
            measurement = outcome["measurement"]
            sampling_audit = outcome["sampling_audit"]
            fit = outcome["fit"]
            # 关键隔离点：父流程在 fit 已完成且 attempts 已排序后才读取结构真值。
            truth, noise_audit = load_evaluation_truth(npz_path)
            row = result_row(measurement, fit, truth, noise_audit)
            rows.append(row)
            all_attempts.extend(attempt_rows(measurement, fit, truth))
            sampling_audits.append({"input_npz": str(npz_path), **sampling_audit})
            global_summaries.append({
                "input_npz": str(npz_path),
                "noise_case": measurement["metadata"]["noise_case"],
                **fit.get("global_summary", {}),
            })
            if fit["success"] and measurement["metadata"]["noise_case"] not in representatives:
                representatives[measurement["metadata"]["noise_case"]] = {
                    "wavelengths_um": measurement["wavelengths_um"],
                    "observed": measurement["spectrum"],
                    "fitted": fit["fitted_spectrum"],
                }
            if fit["success"]:
                print(
                    f"[{index + 1}/{len(inputs)} {measurement['metadata']['noise_case']}] "
                    f"Air_error={row['cavity_error_nm']:.6g} nm, "
                    f"film_MAE={row['film_mae_nm']:.6g} nm, "
                    f"angle_error={row['error_Angle_deg']:.6g} deg, "
                    f"runtime={row['fit_runtime_s']:.3f}s",
                    flush=True,
                )
            else:
                print(
                    f"[{index + 1}/{len(inputs)}] NO CONVERGED LOCAL RESULT: {npz_path}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    # 所有文件处理完成后统一写出逐文件结果、多起点明细、案例汇总、采样审计和报告。
    # 结果目录使用时间戳且 exist_ok=False，避免覆盖先前实验。

    table = pd.DataFrame(rows)
    if table.empty: raise RuntimeError(f"No inversion row was produced: {failures}")
    table.to_csv(output_dir / "fit_results.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(all_attempts).to_csv(
        output_dir / "multistart_results.csv", index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    summary = summarize_results(table)
    summary.to_csv(output_dir / "case_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    successful_table = table[table["success"]].copy()
    if successful_table.empty:
        level_summary = pd.DataFrame(columns=[
            "noise_level", "runs", "cavity_mae_nm", "film_mae_nm",
            "angle_mae_deg", "fit_runtime_mean_s",
        ])
    else:
        level_summary = successful_table.groupby("noise_level", dropna=False).agg(
            runs=("success", "size"),
            cavity_mae_nm=("cavity_abs_error_nm", "mean"),
            film_mae_nm=("film_mae_nm", "mean"),
            angle_mae_deg=("angle_abs_error_deg", "mean"),
            fit_runtime_mean_s=("fit_runtime_s", "mean"),
        ).reset_index()
    level_summary.to_csv(output_dir / "level_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    with (output_dir / "global_search_summary.jsonl").open("w", encoding="utf-8") as handle:
        for item in global_summaries: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    pd.DataFrame(sampling_audits).to_csv(output_dir / "sampling_audit.csv", index=False, encoding="utf-8-sig")
    plot_paths = save_plots(output_dir, table, summary, representatives)
    report = {
        "version": VERSION,
        "input_count": len(inputs),
        "result_count": len(table),
        "successful_count": int(table["success"].sum()),
        "config": asdict(config),
        "parameters": PARAMS,
        "bounds": BOUNDS,
        "truth_usage_policy": (
            "Truth is loaded only after global search, local fits, convergence filtering, and best-result ranking. "
            "No truth or forced bound-center point is used in initialization."
        ),
        "initialization": "Latin-hypercube global population; evolved candidates seed local fits",
        "forward_model": {
            "observable": "single-frame dark-corrected sample/reference spectrum I(lambda)",
            "global_forward_model": config.global_forward_model,
            "global_model_is_only_candidate_generator": True,
            "local_and_final_forward_model": "full_ils_0.002nm",
            "reported_wavelength_axis": "fixed factory-calibration axis from main_v10 NPZ",
            "axis_scenario": EXPECTED_AXIS_SCENARIO,
            "hidden_true_pixel_centers_used_for_fit": False,
            "hidden_axis_error_used_for_fit": False,
            "internal_absolute_grid_step_nm": 0.002,
            "source_model": "known nominal source envelope in both sample and reference branches",
            "unknown_source_realization_used_for_fit": False,
            "qe_model": "known nominal quadratic QE from NPZ config_json",
            "photon_energy_weighting": True,
            "ils_enabled": True,
            "ils_shape": "gaussian",
            "ils_fwhm_nm": 0.02,
            "sample_reference_exposure_ratio_included": True,
            "adc_and_random_detector_noise_in_forward_fit": False,
            "time_series_enabled": False,
            "angle_parameter": "reflector incident-medium angle",
            "air_angle": "derived only for reporting",
            "speed_of_light_m_s": C0_M_S,
        },
        "failures": failures,
        "plots": plot_paths,
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_lines = [
        "# V10 单帧静态光谱六参数反演报告", "", "## 计算口径", "",
        f"- 输入光谱数：{len(inputs)}。",
        f"- 成功收敛数：{int(table['success'].sum())}。",
        "- 同时拟合 Air 腔长、HSQ、PSS、SOC、TiO2 和 reflector 第一入射介质内角度。",
        "- 每个 realization 只包含一帧静态光谱并独立反演，不使用时间序列或调制信息。",
        "- 拟合横轴是光谱仪固定 reported 轴；隐藏 true pixel center 与真实轴误差不进入模型、残差或起点。",
        (
            "- 全局差分进化模式："
            + (
                "直接在全局 reported 采样点计算无 ILS 的快速 TMM，仅用于生成候选。"
                if config.global_forward_model == "fast_no_ils"
                else "使用完整 0.002 nm 内部网格和 Gaussian ILS。"
            )
        ),
        "- 所有局部拟合、最终 cost、RMSE、收敛筛选和 rank 均使用 0.002 nm + 0.02 nm Gaussian ILS 严格模型。",
        "- 模型按生成端定义计算 sample/reference 名义响应；随机 ADC、暗场、探测器噪声及源谱 realization 不读取真值。",
        "- 宽带光源中心漂移与平滑功率谱变化只作为待评估模型失配，不用于修正拟合输入。",
        "- 结构真值和噪声 realization 仅在拟合、收敛筛选与 rank 排序完成后加载，用于误差统计。",
        "- 默认 reported 局部采样间隔为 0.02 nm，全局残差间隔为 0.04 nm；两者均接受逐文件 Nyquist 审计。",
        (
            "- global_stride 在 fast_no_ils 模式下会同时减少全局 TMM 计算点；"
            "在 full_ils 模式下只减少残差比较点，不减少内部 0.002 nm 前向计算。"
        ),
        f"- 局部优化在归一化参数空间使用 gtol={config.local_gtol:.3g}；未触发 SciPy 收敛状态的结果不参与最优解排序。",
        "", "## 分案例汇总", "", dataframe_to_markdown(summary), "",
        "## 解释限制", "",
        "- 当前前向拟合不反演未知轴误差、源谱扰动或探测器参数；这些因素会保留为 model mismatch，并可能形成腔长偏差。",
        "- clean 数据仍包含生成端固定启用的 ADC 量化；反演采用连续名义响应，因此 clean 也不保证数值上完全零残差。",
        "- 静态单光谱中角度在接近 0 度时灵敏度很低，角度误差和条件数必须与腔长、膜厚误差一起判断。",
        "- 噪声案例中的边界命中表示当前六参数模型无法唯一吸收该类模型失配，不应解读为可靠膜厚测量。",
    ]
    (output_dir / "analysis_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8-sig"
    )
    print(f"OUTPUT_DIR={output_dir}")
    if failures: raise RuntimeError(f"Inversion failures: {failures}")

if __name__ == "__main__":
    main()
