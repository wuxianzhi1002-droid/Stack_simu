"""V10 单帧静态光谱仪信号生成器：固定报告轴真实场景。

本版本采用“固定报告波长轴、隐藏真实像素响应中心发生偏差”的测量口径：

1. reported_wavelengths 是光谱仪驱动导出的固定出厂标定轴，不随 realization 改变。
2. true_pixel_center_wavelengths 是每个像素 ILS 的隐藏真实中心，只用于生成计数和审计。
3. StackRT/TMM 始终在 0.002 nm 固定绝对物理波长网格上计算 R(lambda)。
4. 光源中心漂移和功率谱变化只修改 P(lambda)，不修改 StackRT 网格和 ILS 中心。
5. 光谱仪 offset、scale、温漂和高阶校准残差共同形成 reported - true 的轴误差，
   并通过移动隐藏 ILS 中心改变各像素接收的能量。
6. 样品、参考、暗场分别转换为电子数和 ADC 计数，最终通过
   (sample-dark)/(reference-dark) 得到反演使用的单帧光谱。

NPZ 中反演只允许使用固定 reported 轴和 spectrum_measured。隐藏真实像素中心、结构真值、
材料扰动与其他噪声真值仅用于事后误差分析，禁止进入拟合起点、先验或结果排序。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Lumerical Python API 环境
# ---------------------------------------------------------------------------
# api 后端直接调用 lumapi.FDTD.stackrt；batch 后端通过临时文本和 LSF 脚本调用
# fdtd-solutions.exe；tmm 后端用于快速自检。三种后端共享同一层结构、材料和角度口径。

LUMERICAL_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
if LUMERICAL_PATH.exists():
    if str(LUMERICAL_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_PATH))
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    lumapi = None


# ---------------------------------------------------------------------------
# 全局物理常量和数据集口径
# ---------------------------------------------------------------------------
# reflector_angle_deg 始终表示第一入射介质 RefReflector 内的角度，而不是空气层角度。
# WAVELENGTH_ACCURACY_SPEC_NM 约束“固定报告轴 - 隐藏真实像素中心”的总残差。
# 光源中心漂移只改变 P(lambda)，不属于光谱仪波长轴准确度。

GENERATOR_VERSION = "main_v10"
C0_M_S = 299_792_458.0
PLANCK_CONSTANT_J_S = 6.626_070_15e-34
INCIDENT_MEDIUM_N = 5.8284
ANGLE_MAX_DEG = 0.1
WAVELENGTH_ACCURACY_SPEC_NM = 0.05
SOURCE_CENTER_DRIFT_HARD_MAX_NM = 0.2
LAYER_NAMES = ["RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu"]
PERTURBED_MATERIALS = ("HSQ", "PSS", "SOC", "TiO2")
NOISE_FACTORS = (
    "angle",
    "source_center_drift",
    "source_power",
    "axis_offset",
    "axis_scale",
    "thermal_drift",
    "absolute_accuracy",
    "material",
    "detector",
    "combined",
)

CONFIG = {
    "MODEL_TYPE": "PSS_TiO2",
    "WAVELENGTH_AXIS_SCENARIO": "fixed_reported_axis_hidden_true_pixel_centers",
    "WAVELENGTH_START_UM": 0.450,
    "WAVELENGTH_STOP_UM": 0.580,
    # StackRT/TMM 在 0.002 nm 内部网格上计算；该网格不是光谱仪最终输出轴。
    "INTERNAL_WAVELENGTH_STEP_NM": 0.002,
    # 输出采样间隔由 span/(N-1) 决定；当前 450--580 nm、6501 点对应 0.02 nm。
    "OUTPUT_SAMPLE_POINTS": 6501,
    "REFLECTOR_ANGLE_NOMINAL_DEG": 0.0,
    "TIME_SERIES_ENABLED": False,
    "FRAMES_PER_REALIZATION": 1,
    "SOURCE_REFERENCE_CENTER_NM": 515.0,
    "SOURCE_ENVELOPE_SIGMA_NM": 55.0,
    "SOURCE_ENVELOPE_FLOOR_REL": 0.15,
    "SOURCE_POWER_CORRELATION_LENGTH_NM": 12.0,
    # 高阶残差是固定报告轴与隐藏真实像素中心之间无法用 offset/scale 表示的部分。
    "CALIBRATION_RESIDUAL_CORRELATION_LENGTH_NM": 30.0,
    "SPECTROMETER": {
        "MODEL": "generic_array_spectrometer",
        "REPORTED_AXIS_POLICY": "fixed_factory_calibration",
        "AXIS_ERROR_SIGN_CONVENTION": "reported_minus_true",
        "THERMAL_DRIFT_TARGET": "spectrometer_pixel_mapping",
        # ILS 采用归一化 Gaussian；0.02 nm 是 FWHM，不等于输出采样间隔。
        "ILS_ENABLED": True,
        "ILS_SHAPE": "gaussian",
        "ILS_FWHM_NM": 0.02,
        "ILS_TRUNCATE_SIGMA": 4.0,
        "WAVELENGTH_ACCURACY_SPEC_NM": 0.05,
        # QE 采用可替换的经验二次曲线，后续可直接换成产品数据表。
        "QE_MODEL": "quadratic",
        "QE_CENTER_NM": 515.0,
        "QE_PEAK": 0.70,
        "QE_EDGE": 0.45,
        "OPTICAL_THROUGHPUT": 0.25,
        # 样品、参考和暗场分别曝光，暗场由暗电流与读出噪声共同产生。
        "SAMPLE_EXPOSURE_S": 0.010,
        "REFERENCE_EXPOSURE_S": 0.010,
        "DARK_EXPOSURE_S": 0.010,
        "SAMPLE_AVERAGES": 1,
        "REFERENCE_AVERAGES": 1,
        "DARK_AVERAGES": 1,
        # 未知绝对光功率用名义参考谱峰值电子数定标，后续可替换为实测 W/nm。
        "REFERENCE_PEAK_ELECTRONS": 56_000.0,
        "FULL_WELL_ELECTRONS": 80_000.0,
        "READ_NOISE_E_RMS": 5.0,
        "DARK_CURRENT_E_PER_S": 0.1,
        "PIXEL_RESPONSE_NONUNIFORMITY_SIGMA_REL": 0.002,
        "ADC_BITS": 16,
        "ADC_BIAS_COUNTS": 100.0,
        "ADC_NONLINEARITY_REL": 0.001,
        "MIN_REFERENCE_NET_COUNTS": 10.0,
        # clean 和非 detector 单因素案例保留 ILS/采样，但关闭随机探测器噪声。
        "QUANTIZATION_ALWAYS_ENABLED": True,
        "SAVE_INTERNAL_AUDIT_ARRAYS": False,
    },
    "LAYERS": [
        ("RefReflector", 0.0),
        ("Air", 100.0),
        ("HSQ", 0.030),
        ("PSS", 0.010),
        ("SOC", 0.040),
        ("TiO2", 0.040),
        ("Cu", 0.0),
    ],
}

# ---------------------------------------------------------------------------
# 噪声等级定义
# ---------------------------------------------------------------------------
# V10 使用四档工程语义：
# typical             常规运行中的代表性水平；
# in_spec             仍在产品规格内，但明显劣于 typical；
# spec_limit          靠近规格边界；
# out_of_spec_stress  超规格压力测试，不代表正常运行精度。
#
# 每档数值是该档目标上限；实际 realization 在 LEVEL_AMPLITUDE_FRACTION_RANGE
# 指定的窄区间内取值，避免高等级实例随机落入低等级范围。
LEVEL_AMPLITUDE_FRACTION_RANGE = {
    "typical": (0.70, 1.00),
    "in_spec": (0.80, 1.00),
    "spec_limit": (0.90, 1.00),
    "out_of_spec_stress": (0.90, 1.00),
}

NOISE_LEVELS = {
    "typical": {
        "n_real_sigma_rel": 2.0e-4,
        "k_sigma_rel": 5.0e-3,
        "angle_max_deg": 0.002,
        "source_center_drift_max_nm": 0.005,
        "source_power_curve_peak_rel": 1.0e-3,
        "axis_offset_max_nm": 0.001,
        "axis_scale_max_ppm": 5.0,
        "thermal_drift_max_nm": 0.001,
        "calibration_residual_max_nm": 0.0075,
        "detector_noise_multiplier": 1.0,
    },
    "in_spec": {
        "n_real_sigma_rel": 5.0e-4,
        "k_sigma_rel": 1.0e-2,
        "angle_max_deg": 0.010,
        "source_center_drift_max_nm": 0.020,
        "source_power_curve_peak_rel": 3.0e-3,
        "axis_offset_max_nm": 0.003,
        "axis_scale_max_ppm": 10.0,
        "thermal_drift_max_nm": 0.003,
        "calibration_residual_max_nm": 0.018,
        "detector_noise_multiplier": 1.5,
    },
    "spec_limit": {
        "n_real_sigma_rel": 2.0e-3,
        "k_sigma_rel": 5.0e-2,
        "angle_max_deg": 0.050,
        "source_center_drift_max_nm": 0.100,
        "source_power_curve_peak_rel": 1.0e-2,
        "axis_offset_max_nm": 0.006,
        "axis_scale_max_ppm": 30.0,
        "thermal_drift_max_nm": 0.006,
        "calibration_residual_max_nm": 0.035,
        "detector_noise_multiplier": 2.0,
    },
    "out_of_spec_stress": {
        "n_real_sigma_rel": 5.0e-3,
        "k_sigma_rel": 1.0e-1,
        "angle_max_deg": 0.100,
        "source_center_drift_max_nm": 0.200,
        "source_power_curve_peak_rel": 5.0e-2,
        "axis_offset_max_nm": 0.020,
        "axis_scale_max_ppm": 60.0,
        "thermal_drift_max_nm": 0.020,
        "calibration_residual_max_nm": 0.060,
        "detector_noise_multiplier": 3.0,
    },
}


# ---------------------------------------------------------------------------
# 噪声案例选择与幅度抽样
# ---------------------------------------------------------------------------
# 单因素案例只激活对应字段；combined 激活全部字段；clean 全部置零。
def zero_profile() -> dict[str, float]:
    return {name: 0.0 for name in next(iter(NOISE_LEVELS.values()))}


def active_profile(level: str, factor: str) -> dict[str, float]:
    if level == "clean":
        return zero_profile()
    source = NOISE_LEVELS[level]
    if factor == "combined":
        return dict(source)
    keys = {
        "angle": ("angle_max_deg",),
        "source_center_drift": ("source_center_drift_max_nm",),
        "source_power": ("source_power_curve_peak_rel",),
        "axis_offset": ("axis_offset_max_nm",),
        "axis_scale": ("axis_scale_max_ppm",),
        "thermal_drift": ("thermal_drift_max_nm",),
        "absolute_accuracy": ("calibration_residual_max_nm",),
        "material": ("n_real_sigma_rel", "k_sigma_rel"),
        "detector": ("detector_noise_multiplier",),
    }[factor]
    profile = zero_profile()
    for key in keys:
        profile[key] = float(source[key])
    return profile


def all_case_names() -> list[str]:
    return ["clean"] + [
        f"{factor}_{level}" for factor in NOISE_FACTORS for level in NOISE_LEVELS
    ]


def parse_case(case_name: str) -> tuple[str, str]:
    if case_name == "clean":
        return "clean", "clean"
    # 等级名称本身可能含下划线，因此不能使用 rsplit("_", 1)。
    for level in sorted(NOISE_LEVELS, key=len, reverse=True):
        suffix = f"_{level}"
        if case_name.endswith(suffix):
            factor = case_name[:-len(suffix)]
            if factor in NOISE_FACTORS:
                return factor, level
    raise ValueError(f"Unknown case: {case_name}")


def select_cases(value: str) -> list[str]:
    if value == "all":
        return all_case_names()
    if value in NOISE_FACTORS:
        return [f"{value}_{level}" for level in NOISE_LEVELS]
    cases = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(cases) - set(all_case_names()))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    return cases


def amplitude_fraction(level: str, rng: np.random.Generator) -> float:
    if level == "clean":
        return 0.0
    lower, upper = LEVEL_AMPLITUDE_FRACTION_RANGE[level]
    return float(rng.uniform(lower, upper))


def signed_level_value(maximum: float, level: str, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    magnitude = float(maximum) * amplitude_fraction(level, rng)
    return magnitude if rng.integers(0, 2) else -magnitude


def positive_level_value(maximum: float, level: str, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    return float(maximum) * amplitude_fraction(level, rng)


# ---------------------------------------------------------------------------
# 平滑随机曲线
# ---------------------------------------------------------------------------
# 波长标定残差先经过高斯滤波，再去掉常数项和线性项。这样 absolute_accuracy 只描述
# 无法被简单 offset/scale 校正吸收的高阶残差，避免和 axis_offset、axis_scale 重复计数。
# 光源功率谱曲线只去掉均值，保留真实系统中可能出现的谱倾斜和缓慢形状变化。

def remove_constant_and_linear_terms(values: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    x = (wavelengths_nm - np.mean(wavelengths_nm)) / max(float(np.ptp(wavelengths_nm)), 1.0)
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - design @ coefficients


def smoothed_random_base(
    wavelengths_nm: np.ndarray,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """在较稀的辅助网格生成平滑曲线，再插值到目标网格。

    直接在 0.002 nm 网格上使用数千像素宽的 Gaussian 核会非常慢。辅助网格只负责
    生成低频随机形状，不参与 StackRT 或 ILS 积分，因此不会降低物理光谱分辨率。
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    span_nm = float(wavelengths_nm[-1] - wavelengths_nm[0])
    target_step_nm = min(0.25, float(correlation_length_nm) / 20.0)
    control_count = max(33, int(np.ceil(span_nm / target_step_nm)) + 1)
    control_nm = np.linspace(wavelengths_nm[0], wavelengths_nm[-1], control_count)
    control_step_nm = float(np.median(np.diff(control_nm)))
    sigma_pixels = max(float(correlation_length_nm) / control_step_nm, 1.0)
    raw_control = gaussian_filter1d(
        rng.normal(size=control_count), sigma=sigma_pixels, mode="reflect"
    )
    return np.interp(wavelengths_nm, control_nm, raw_control)


def smooth_curve_with_peak(
    wavelengths_nm: np.ndarray,
    target_peak_nm: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_peak_nm <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)
    for _ in range(50):
        raw = smoothed_random_base(wavelengths_nm, correlation_length_nm, rng)
        # absolute_accuracy 只保留无法被简单 offset/scale 吸收的高阶部分。
        curve = remove_constant_and_linear_terms(raw, wavelengths_nm)
        peak = float(np.max(np.abs(curve)))
        if peak > 1.0e-12:
            return curve * (float(target_peak_nm) / peak)
    raise RuntimeError("Could not generate a nonzero smooth wavelength curve.")


def smooth_relative_curve_with_peak(
    wavelengths_nm: np.ndarray,
    target_peak_rel: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_peak_rel <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)
    for _ in range(50):
        raw = smoothed_random_base(wavelengths_nm, correlation_length_nm, rng)
        curve = raw - float(np.mean(raw))
        peak = float(np.max(np.abs(curve)))
        if peak > 1.0e-12:
            return curve * (float(target_peak_rel) / peak)
    raise RuntimeError("Could not generate a nonzero source-power curve.")


# 宽带光源名义功率谱采用“高斯包络 + 非零底座”。非零底座可以避免参考谱边缘接近零时，
# sample/reference 除法放大数值误差。每条包络都按自身最大值归一化，从而让中心漂移主要
# 表示谱形变化；光电转换、散粒噪声和 ADC 由后续光谱仪信号链统一处理。

def source_power_envelope(
    wavelengths_nm: np.ndarray,
    center_nm: float,
) -> np.ndarray:
    sigma_nm = float(CONFIG["SOURCE_ENVELOPE_SIGMA_NM"])
    floor_rel = float(CONFIG["SOURCE_ENVELOPE_FLOOR_REL"])
    gaussian = np.exp(-0.5 * ((np.asarray(wavelengths_nm) - center_nm) / sigma_nm) ** 2)
    envelope = floor_rel + (1.0 - floor_rel) * gaussian
    return envelope / float(np.max(envelope))


# ---------------------------------------------------------------------------
# 材料模型与 TMM 前向模型
# ---------------------------------------------------------------------------
# material_n 返回每个波长点的复折射率 n+i*k。只有 PERTURBED_MATERIALS 中的膜层
# 会在数据生成端施加 n/k 扰动；反演端仍使用名义材料模型，因此材料案例形成模型失配。

def material_n(
    name: str,
    wavelengths_um: np.ndarray,
    n_real_rel_delta: dict[str, float] | None = None,
    k_rel_delta: dict[str, float] | None = None,
) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=float)
    if name == "RefReflector":
        values = np.full_like(w, INCIDENT_MEDIUM_N, dtype=np.complex128)
    elif name == "Air":
        values = np.ones_like(w, dtype=np.complex128)
    elif name == "HSQ":
        values = np.full_like(w, 1.41, dtype=np.complex128)
    elif name == "PSS":
        values = np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    elif name == "SOC":
        values = (1.55 + 0.005 / (w**2)).astype(np.complex128)
    elif name == "TiO2":
        values = (2.4 + 0.02 / (w**2)).astype(np.complex128)
    elif name == "Cu":
        values = np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown material: {name}")
    if name in PERTURBED_MATERIALS:
        dn = 0.0 if n_real_rel_delta is None else float(n_real_rel_delta.get(name, 0.0))
        dk = 0.0 if k_rel_delta is None else float(k_rel_delta.get(name, 0.0))
        values = values.real * (1.0 + dn) + 1j * values.imag * (1.0 + dk)
    return values


def propagation_cosines(n_matrix: np.ndarray, reflector_angle_deg: float) -> np.ndarray:
    # Snell 不变量 n0*sin(theta0) 在各层保持一致，由此计算每层的复数 cos(theta)。
    # 对吸收介质需要保留复数平方根，并选择实部为正的传播分支。
    tangential_index = n_matrix[0] * np.sin(np.deg2rad(float(reflector_angle_deg)))
    cos_values = np.sqrt(1.0 - (tangential_index[None, :] / n_matrix) ** 2)
    cos_values[np.real(cos_values) < 0.0] *= -1.0
    return cos_values


def tmm_reflectance(
    wavelengths_um: np.ndarray,
    thicknesses_um: dict[str, float],
    reflector_angle_deg: float,
    n_real_rel_delta: dict[str, float],
    k_rel_delta: dict[str, float],
) -> np.ndarray:
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    n_matrix = np.vstack([
        material_n(name, wavelengths_um, n_real_rel_delta, k_rel_delta) for name in LAYER_NAMES
    ])
    cos_values = propagation_cosines(n_matrix, reflector_angle_deg)
    q_values = n_matrix / cos_values
    k0 = 2.0 * np.pi / (wavelengths_um * 1.0e-6)
    m11 = np.ones(len(wavelengths_um), dtype=complex)
    m12 = np.zeros(len(wavelengths_um), dtype=complex)
    m21 = np.zeros(len(wavelengths_um), dtype=complex)
    m22 = np.ones(len(wavelengths_um), dtype=complex)
    # 从 Air 到 TiO2 依次左乘各层特征矩阵；首末半无限介质不使用物理厚度。
    for layer_index, name in enumerate(LAYER_NAMES[1:-1], start=1):
        thickness_m = float(thicknesses_um[name]) * 1.0e-6
        if thickness_m <= 0.0:
            continue
        delta = k0 * n_matrix[layer_index] * cos_values[layer_index] * thickness_m
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        q_layer = q_values[layer_index]
        a11 = c_delta
        a12 = -1j * s_delta / q_layer
        a21 = -1j * q_layer * s_delta
        a22 = c_delta
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21,
            m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21,
            m21 * a12 + m22 * a22,
        )
    q0 = q_values[0]
    qs = q_values[-1]
    numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
    denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
    return np.abs(numerator / denominator) ** 2



# ---------------------------------------------------------------------------
# 单个 realization 的噪声实例化
# ---------------------------------------------------------------------------
# 该函数只负责产生噪声参数、波长轴和源谱扰动，不调用光学求解器。返回值分成：
# metadata：适合写入 JSON/CSV 的标量和字典；
# components：与波长等长的数组，供前向计算、NPZ 审计和代表图使用。
#
# 固定报告轴场景的关系为：
#   reported = nominal_calibrated                         （反演可见且固定）
#   axis_error = offset + scale + thermal + high_order   （reported - true）
#   true_pixel_center = reported - axis_error             （隐藏 ILS 中心）
#   internal_absolute_grid = fixed                        （StackRT 与积分网格）
# 光源中心漂移只改变 P(lambda)，不进入上述波长轴关系。

def realize_noise(
    case_name: str,
    reported_wavelengths_nm: np.ndarray,
    internal_absolute_wavelengths_nm: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    """固定一个 Monte Carlo realization 的误差真值与隐藏像素响应中心。

    reported_wavelengths_nm 是光谱仪对外输出的固定标定轴；
    internal_absolute_wavelengths_nm 是 StackRT 和 ILS 积分使用的 0.002 nm 绝对物理网格。
    两条轴都不因光源漂移而移动。只有 true_pixel_center_wavelengths_nm 会因光谱仪
    offset、scale、温漂和高阶残差偏离报告轴。
    """
    factor, level = parse_case(case_name)
    profile = active_profile(level, factor)

    n_delta = {
        name: signed_level_value(profile["n_real_sigma_rel"], level, rng)
        for name in PERTURBED_MATERIALS
    }
    k_delta = {
        name: signed_level_value(profile["k_sigma_rel"], level, rng)
        for name in PERTURBED_MATERIALS
    }

    # 角度噪声围绕名义求解器角生成。反射率对正负角对称，因此前向模型使用绝对值，
    # 同时单独保存带符号误差，避免在 0 度附近通过 clip 产生不真实的零点堆积。
    angle_error_deg = signed_level_value(profile["angle_max_deg"], level, rng)
    angle_deg = abs(float(CONFIG["REFLECTOR_ANGLE_NOMINAL_DEG"]) + angle_error_deg)
    if angle_deg > ANGLE_MAX_DEG + 1.0e-12:
        raise ValueError("Realized reflector angle exceeds 0.1 deg.")

    # 光源误差只修改固定绝对网格上的 P(lambda)。它不会改变 StackRT 的 R(lambda) 网格，
    # 也不会改变 reported 轴或隐藏 ILS 中心。
    source_center_drift_nm = signed_level_value(
        profile["source_center_drift_max_nm"], level, rng
    )
    if abs(source_center_drift_nm) > SOURCE_CENTER_DRIFT_HARD_MAX_NM + 1.0e-12:
        raise ValueError("Source center drift exceeds the 0.2 nm hard limit.")
    source_power_peak = positive_level_value(
        profile["source_power_curve_peak_rel"], level, rng
    )
    source_power_curve_internal = smooth_relative_curve_with_peak(
        internal_absolute_wavelengths_nm,
        source_power_peak,
        CONFIG["SOURCE_POWER_CORRELATION_LENGTH_NM"],
        rng,
    )
    source_power_curve_reported = np.interp(
        reported_wavelengths_nm,
        internal_absolute_wavelengths_nm,
        source_power_curve_internal,
    )

    # 光谱仪轴误差采用 epsilon = reported - true 的符号约定。
    # offset 是常数；scale 关于波段中心呈线性；thermal 当前模拟光谱仪色散映射整体热漂移；
    # high_order 是无法被 offset/scale 吸收的平滑标定残差。
    axis_offset_error_nm = signed_level_value(
        profile["axis_offset_max_nm"], level, rng
    )
    axis_scale_error_ppm = signed_level_value(
        profile["axis_scale_max_ppm"], level, rng
    )
    axis_thermal_error_nm = signed_level_value(
        profile["thermal_drift_max_nm"], level, rng
    )
    center_nm = 0.5 * (reported_wavelengths_nm[0] + reported_wavelengths_nm[-1])
    offset_error_curve = np.full_like(
        reported_wavelengths_nm, axis_offset_error_nm, dtype=float
    )
    scale_error_curve = (
        axis_scale_error_ppm * 1.0e-6 * (reported_wavelengths_nm - center_nm)
    )
    thermal_error_curve = np.full_like(
        reported_wavelengths_nm, axis_thermal_error_nm, dtype=float
    )
    high_order_peak = positive_level_value(
        profile["calibration_residual_max_nm"], level, rng
    )
    high_order_error_curve = smooth_curve_with_peak(
        reported_wavelengths_nm,
        high_order_peak,
        CONFIG["CALIBRATION_RESIDUAL_CORRELATION_LENGTH_NM"],
        rng,
    )
    axis_error_total = (
        offset_error_curve
        + scale_error_curve
        + thermal_error_curve
        + high_order_error_curve
    )
    true_pixel_center_nm = reported_wavelengths_nm - axis_error_total

    total_error_peak = float(np.max(np.abs(axis_error_total)))
    total_error_rms = float(np.sqrt(np.mean(axis_error_total**2)))
    if level != "out_of_spec_stress" and total_error_peak > WAVELENGTH_ACCURACY_SPEC_NM + 1.0e-12:
        raise ValueError(
            "In-spec total spectrometer wavelength error exceeds the configured +/-0.05 nm spec."
        )
    if np.any(np.diff(reported_wavelengths_nm) <= 0.0):
        raise ValueError("Fixed reported wavelength axis is not monotonic.")
    if np.any(np.diff(true_pixel_center_nm) <= 0.0):
        raise ValueError("Hidden true pixel-center mapping is not monotonic.")
    if np.any(np.diff(internal_absolute_wavelengths_nm) <= 0.0):
        raise ValueError("Internal absolute wavelength grid is not monotonic.")

    detector_noise_multiplier = positive_level_value(
        profile["detector_noise_multiplier"], level, rng
    )
    metadata = {
        "case": case_name,
        "factor": factor,
        "level": level,
        "profile": profile,
        "amplitude_fraction_range": (
            [0.0, 0.0] if level == "clean" else list(LEVEL_AMPLITUDE_FRACTION_RANGE[level])
        ),
        "material_n_real_rel_delta": n_delta,
        "material_k_rel_delta": k_delta,
        "reflector_angle_deg": angle_deg,
        "reflector_angle_error_deg": angle_error_deg,
        "source_center_drift_nm": source_center_drift_nm,
        "source_power_curve_peak_rel": float(np.max(np.abs(source_power_curve_internal))),
        "axis_offset_nm": axis_offset_error_nm,
        "axis_scale_ppm": axis_scale_error_ppm,
        "axis_scale_edge_max_nm": float(np.max(np.abs(scale_error_curve))),
        "thermal_drift_nm": axis_thermal_error_nm,
        "calibration_residual_peak_nm": float(np.max(np.abs(high_order_error_curve))),
        "spectrometer_axis_error_max_abs_nm": total_error_peak,
        "spectrometer_axis_error_rms_nm": total_error_rms,
        # 以下旧名称保留给现有汇总脚本；含义已统一为固定报告轴相对隐藏真实中心的误差。
        "physical_axis_shift_max_abs_nm": total_error_peak,
        "axis_estimation_error_max_abs_nm": total_error_peak,
        "axis_estimation_error_rms_nm": total_error_rms,
        "wavelength_accuracy_spec_nm": WAVELENGTH_ACCURACY_SPEC_NM,
        "wavelength_accuracy_within_spec": bool(
            total_error_peak <= WAVELENGTH_ACCURACY_SPEC_NM
        ),
        "reported_axis_is_fixed": True,
        "axis_error_sign_convention": "reported_minus_true",
        "detector_noise_multiplier": detector_noise_multiplier,
    }
    components = {
        "reported_wavelengths_nm": np.asarray(reported_wavelengths_nm, dtype=float),
        "true_pixel_center_wavelengths_nm": true_pixel_center_nm,
        "spectrometer_axis_offset_error_nm": offset_error_curve,
        "spectrometer_axis_scale_error_nm": scale_error_curve,
        "spectrometer_axis_thermal_error_nm": thermal_error_curve,
        "spectrometer_axis_high_order_error_nm": high_order_error_curve,
        "spectrometer_axis_error_total_nm": axis_error_total,
        "source_power_relative_curve": source_power_curve_reported,
        "source_power_relative_curve_internal": source_power_curve_internal,
        "internal_absolute_wavelengths_nm": np.asarray(
            internal_absolute_wavelengths_nm, dtype=float
        ),
        # 兼容旧读取代码的别名。physical_wavelengths 现在明确指隐藏像素真实中心；
        # estimated_calibrated 则等于固定 reported 轴，不再随 realization 改变。
        "physical_wavelengths_nm": true_pixel_center_nm,
        "estimated_calibrated_wavelengths_nm": np.asarray(
            reported_wavelengths_nm, dtype=float
        ),
        "physical_axis_offset_nm": -offset_error_curve,
        "physical_axis_scale_nm": -scale_error_curve,
        "physical_axis_thermal_nm": -thermal_error_curve,
        "physical_axis_shift_total_nm": -axis_error_total,
        "calibration_residual_nm": high_order_error_curve,
        "internal_nominal_wavelengths_nm": np.asarray(
            internal_absolute_wavelengths_nm, dtype=float
        ),
        "internal_physical_wavelengths_nm": np.asarray(
            internal_absolute_wavelengths_nm, dtype=float
        ),
    }
    return metadata, components


# ---------------------------------------------------------------------------
# 光谱仪信号链
# ---------------------------------------------------------------------------

def quantum_efficiency(wavelengths_nm: np.ndarray) -> np.ndarray:
    """返回经验 QE(lambda)；参数全部位于 CONFIG，便于替换为实测曲线。"""
    settings = CONFIG["SPECTROMETER"]
    if settings["QE_MODEL"] != "quadratic":
        raise ValueError(f"Unsupported QE model: {settings['QE_MODEL']}")
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    center_nm = float(settings["QE_CENTER_NM"])
    half_span_nm = max(
        center_nm - CONFIG["WAVELENGTH_START_UM"] * 1000.0,
        CONFIG["WAVELENGTH_STOP_UM"] * 1000.0 - center_nm,
    )
    normalized = np.clip((wavelengths_nm - center_nm) / half_span_nm, -1.0, 1.0)
    peak = float(settings["QE_PEAK"])
    edge = float(settings["QE_EDGE"])
    return edge + (peak - edge) * (1.0 - normalized**2)


def ils_convolve_and_sample(
    internal_absolute_nm: np.ndarray,
    spectral_density: np.ndarray,
    true_pixel_center_nm: np.ndarray,
) -> np.ndarray:
    """在固定绝对网格卷积 Gaussian ILS，再采样到隐藏真实像素中心。

    reported 波长轴不参与卷积中心定位，只用于最终给计数贴标签。这里使用一维
    Gaussian filter 避免构造 6501 x 65001 的稠密 ILS 矩阵。当前模型假设所有像素
    具有相同、平移不变的归一化 Gaussian ILS；获得实测非对称 ILS 后应替换本函数。
    """
    settings = CONFIG["SPECTROMETER"]
    values = np.asarray(spectral_density, dtype=float)
    if settings["ILS_ENABLED"]:
        if settings["ILS_SHAPE"] != "gaussian":
            raise ValueError(f"Unsupported ILS shape: {settings['ILS_SHAPE']}")
        spacing_nm = float(np.median(np.diff(internal_absolute_nm)))
        sigma_nm = float(settings["ILS_FWHM_NM"]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        values = gaussian_filter1d(
            values,
            sigma=sigma_nm / spacing_nm,
            mode="nearest",
            truncate=float(settings["ILS_TRUNCATE_SIGMA"]),
        )
    return np.interp(true_pixel_center_nm, internal_absolute_nm, values)


def expected_photoelectrons(
    internal_absolute_nm: np.ndarray,
    optical_power_density_rel: np.ndarray,
    true_pixel_center_nm: np.ndarray,
    exposure_s: float,
    optical_psd_scale_w_per_nm: float,
) -> np.ndarray:
    """实现真实像素中心处的光谱功率、QE、ILS 与光子能量积分。"""
    settings = CONFIG["SPECTROMETER"]
    wavelength_m = np.asarray(internal_absolute_nm, dtype=float) * 1.0e-9
    photon_energy_j = PLANCK_CONSTANT_J_S * C0_M_S / wavelength_m
    electron_density_per_nm_s = (
        optical_psd_scale_w_per_nm
        * np.asarray(optical_power_density_rel, dtype=float)
        * float(settings["OPTICAL_THROUGHPUT"])
        * quantum_efficiency(internal_absolute_nm)
        / photon_energy_j
    )
    sampled_density = ils_convolve_and_sample(
        internal_absolute_nm,
        electron_density_per_nm_s,
        true_pixel_center_nm,
    )
    # 像素报告轴间隔用于定义通道等效带宽；隐藏中心的小偏差不改变像素数量与标称带宽。
    reported_sampling_nm = (
        (CONFIG["WAVELENGTH_STOP_UM"] - CONFIG["WAVELENGTH_START_UM"])
        * 1000.0
        / (int(CONFIG["OUTPUT_SAMPLE_POINTS"]) - 1)
    )
    return np.maximum(
        sampled_density * reported_sampling_nm * float(exposure_s), 0.0
    )


def adc_counts_from_electrons(
    expected_signal_e: np.ndarray,
    exposure_s: float,
    averages: int,
    detector_noise_multiplier: float,
    pixel_response_error_rel: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """将期望光电子数转换为 ADC 计数，并返回饱和等审计量。"""
    settings = CONFIG["SPECTROMETER"]
    if averages < 1:
        raise ValueError("Averages must be positive.")
    multiplier = float(detector_noise_multiplier)
    photo_mean = np.maximum(
        np.asarray(expected_signal_e, dtype=float) * (1.0 + multiplier * pixel_response_error_rel),
        0.0,
    )
    dark_mean = float(settings["DARK_CURRENT_E_PER_S"]) * float(exposure_s) * multiplier
    if multiplier > 0.0:
        # 多帧平均用 Poisson(sum mean)/N 生成散粒噪声，再叠加缩小后的读出噪声。
        total_mean = np.maximum((photo_mean + dark_mean) * averages, 0.0)
        electrons = rng.poisson(total_mean).astype(float) / averages
        electrons += rng.normal(
            0.0,
            float(settings["READ_NOISE_E_RMS"]) * multiplier / np.sqrt(averages),
            size=photo_mean.shape,
        )
    else:
        electrons = photo_mean

    full_well = float(settings["FULL_WELL_ELECTRONS"])
    saturation_fraction = float(np.mean(electrons >= full_well))
    electrons = np.clip(electrons, 0.0, full_well)
    nonlinearity = float(settings["ADC_NONLINEARITY_REL"]) * multiplier
    electrons_nonlinear = electrons * (1.0 + nonlinearity * (electrons / full_well) ** 2)
    adc_max = float(2 ** int(settings["ADC_BITS"]) - 1)
    usable_counts = adc_max - float(settings["ADC_BIAS_COUNTS"])
    counts = float(settings["ADC_BIAS_COUNTS"]) + electrons_nonlinear * usable_counts / full_well
    if settings["QUANTIZATION_ALWAYS_ENABLED"]:
        counts = np.rint(counts)
    counts = np.clip(counts, 0.0, adc_max)
    return counts.astype(float), {
        "saturation_fraction": saturation_fraction,
        "mean_expected_signal_e": float(np.mean(expected_signal_e)),
        "max_expected_signal_e": float(np.max(expected_signal_e)),
    }


# ---------------------------------------------------------------------------
# 光学求解器统一接口
# ---------------------------------------------------------------------------
class OpticalSolver:
    def __init__(self, backend: str, bridge_dir: Path):
        self.backend = backend
        self.bridge_dir = bridge_dir
        self.fdtd = None

    def __enter__(self):
        if self.backend == "api":
            if lumapi is None:
                raise RuntimeError("lumapi is unavailable; use --backend tmm or batch.")
            self.fdtd = lumapi.FDTD(hide=True, serverArgs={ "use-solve": True})
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    def reflectance(
        self,
        wavelengths_um: np.ndarray,
        thicknesses_um: dict[str, float],
        reflector_angle_deg: float,
        n_real_rel_delta: dict[str, float],
        k_rel_delta: dict[str, float],
    ) -> np.ndarray:
        # 快速后端直接返回本地 TMM；Lumerical 后端需要先构造逐层 n(lambda) 矩阵，
        # 再把波长转换为频率传给 stackrt。
        if self.backend == "tmm":
            return tmm_reflectance(
                wavelengths_um,
                thicknesses_um,
                reflector_angle_deg,
                n_real_rel_delta,
                k_rel_delta,
            )
        n_matrix = np.vstack([
            material_n(name, wavelengths_um, n_real_rel_delta, k_rel_delta)
            for name in LAYER_NAMES
        ])
        thicknesses_m = np.asarray([thicknesses_um[name] for name in LAYER_NAMES]) * 1.0e-6
        frequencies_hz = C0_M_S / (np.asarray(wavelengths_um) * 1.0e-6)
        if self.backend == "api":
            result = self.fdtd.stackrt(n_matrix, thicknesses_m, frequencies_hz, reflector_angle_deg)
            return np.asarray(result["Rp"], dtype=float).reshape(-1)
        return self._batch(n_matrix, thicknesses_m, frequencies_hz, reflector_angle_deg)

    def _batch(
        self,
        n_matrix: np.ndarray,
        thicknesses_m: np.ndarray,
        frequencies_hz: np.ndarray,
        reflector_angle_deg: float,
    ) -> np.ndarray:
        stage = self.bridge_dir
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "appdata" / "Ansys").mkdir(parents=True, exist_ok=True)
        np.savetxt(stage / "n_real.txt", n_matrix.real, fmt="%.17g")
        np.savetxt(stage / "n_imag.txt", n_matrix.imag, fmt="%.17g")
        np.savetxt(stage / "freqs.txt", frequencies_hz[:, None], fmt="%.17g")
        np.savetxt(stage / "thicknesses.txt", thicknesses_m[:, None], fmt="%.17g")
        script = (
            'n_real=readdata("n_real.txt");\n'
            'n_imag=readdata("n_imag.txt");\n'
            'n_matrix=n_real+1i*n_imag;\n'
            'freqs=readdata("freqs.txt");\n'
            'thicknesses=readdata("thicknesses.txt");\n'
            f'result=stackrt(n_matrix,thicknesses,freqs,{reflector_angle_deg:.17g});\n'
            'spectra=transpose(result.Rp);\n'
            'matlabsave("stackrt_static_output",spectra);\n'
            'exit;\n'
        )
        (stage / "run_stackrt_static.lsf").write_text(script, encoding="ascii")
        executable = Path(r"D:\Program Files\Lumerical\v241\bin\fdtd-solutions.exe")
        if not executable.exists():
            raise FileNotFoundError(executable)
        environment = os.environ.copy()
        environment["APPDATA"] = str(stage / "appdata")
        started = time.time()
        completed = subprocess.run(
            [str(executable), "-nw", "-trust-script", "-run", "run_stackrt_static.lsf"],
            cwd=stage,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        (stage / "stackrt_batch.log").write_text(
            f"returncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
            encoding="utf-8",
        )
        output_path = stage / "stackrt_static_output.mat"
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_mtime < started - 1.0:
            raise RuntimeError(f"StackRT batch failed; see {stage / 'stackrt_batch.log'}")
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("The batch backend requires h5py.") from exc
        with h5py.File(output_path, "r") as handle:
            return np.asarray(handle["spectra"], dtype=float).reshape(-1)


# ---------------------------------------------------------------------------
# 单帧静态数据集生成器
# ---------------------------------------------------------------------------
# 每次 generate_one 只产生一帧，不做调制、不做曝光积分，也不在 realization 内平均。
# Monte Carlo 统计通过对同一案例使用多个独立 seed 重复调用 generate_one 实现。

class StaticDatasetGenerator:
    def __init__(self, backend: str, output_dir: Path):
        self.backend = backend
        self.output_dir = output_dir
        start_nm = float(CONFIG["WAVELENGTH_START_UM"]) * 1000.0
        stop_nm = float(CONFIG["WAVELENGTH_STOP_UM"]) * 1000.0
        output_points = int(CONFIG["OUTPUT_SAMPLE_POINTS"])
        if output_points < 2:
            raise ValueError("OUTPUT_SAMPLE_POINTS must be at least 2.")

        # 这条轴模拟光谱仪驱动/文件中永久保存的像素标定轴。每个 realization 都相同。
        self.reported_nm = np.linspace(start_nm, stop_nm, output_points)
        # 保留 nominal_nm 属性，兼容早期汇总代码；在本场景中 nominal == reported。
        self.nominal_nm = self.reported_nm.copy()
        self.output_sampling_nm = float(np.median(np.diff(self.reported_nm)))

        settings = CONFIG["SPECTROMETER"]
        internal_step_nm = float(CONFIG["INTERNAL_WAVELENGTH_STEP_NM"])
        if internal_step_nm <= 0.0:
            raise ValueError("INTERNAL_WAVELENGTH_STEP_NM must be positive.")
        if settings["ILS_ENABLED"] and internal_step_nm > float(settings["ILS_FWHM_NM"]) / 5.0:
            raise ValueError("Internal wavelength grid is too coarse for the configured ILS.")

        sigma_nm = float(settings["ILS_FWHM_NM"]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        ils_margin_nm = (
            float(settings["ILS_TRUNCATE_SIGMA"]) * sigma_nm
            if settings["ILS_ENABLED"] else 0.0
        )
        half_span_nm = 0.5 * (stop_nm - start_nm)
        configured_axis_bound_nm = max(
            float(values["axis_offset_max_nm"])
            + float(values["axis_scale_max_ppm"]) * 1.0e-6 * half_span_nm
            + float(values["thermal_drift_max_nm"])
            + float(values["calibration_residual_max_nm"])
            for values in NOISE_LEVELS.values()
        )
        # 内部绝对网格必须覆盖最差隐藏像素中心偏移及完整 ILS 核，否则边缘像素会外插。
        self.internal_margin_nm = ils_margin_nm + configured_axis_bound_nm
        internal_start_nm = start_nm - self.internal_margin_nm
        internal_stop_nm = stop_nm + self.internal_margin_nm
        internal_count = int(
            np.ceil((internal_stop_nm - internal_start_nm) / internal_step_nm)
        ) + 1
        self.internal_absolute_nm = (
            internal_start_nm + np.arange(internal_count) * internal_step_nm
        )
        # 兼容早期命名；该数组现在是固定绝对物理网格，不再经过 axis mapping。
        self.internal_nominal_nm = self.internal_absolute_nm
        self.layers = dict(CONFIG["LAYERS"])

    def generate_one(
        self,
        solver: OpticalSolver,
        case_name: str,
        realization_index: int,
        seed: int,
    ) -> tuple[dict, dict]:
        total_started = time.perf_counter()
        rng = np.random.default_rng(seed)
        noise, components = realize_noise(
            case_name,
            self.reported_nm,
            self.internal_absolute_nm,
            rng,
        )
        reported_nm = components["reported_wavelengths_nm"]
        true_pixel_center_nm = components["true_pixel_center_wavelengths_nm"]
        internal_absolute_nm = components["internal_absolute_wavelengths_nm"]

        # StackRT/TMM 只接受固定绝对物理波长。光源漂移和光谱仪标定误差都不能修改此网格。
        solver_started = time.perf_counter()
        physical_reflectance_internal = solver.reflectance(
            internal_absolute_nm / 1000.0,
            self.layers,
            noise["reflector_angle_deg"],
            noise["material_n_real_rel_delta"],
            noise["material_k_rel_delta"],
        )
        solver_runtime_s = time.perf_counter() - solver_started

        # 光源中心漂移通过 P(lambda-delta_source) 实现：固定横轴不动，功率谱纵值改变。
        # 样品源还可叠加平滑功率谱扰动；参考源保持名义状态，用于模拟非同步参考测量。
        reference_center_nm = float(CONFIG["SOURCE_REFERENCE_CENTER_NM"])
        source_reference_internal = source_power_envelope(
            internal_absolute_nm, reference_center_nm
        )
        source_sample_base_internal = source_power_envelope(
            internal_absolute_nm,
            reference_center_nm + noise["source_center_drift_nm"],
        )
        source_sample_internal = source_sample_base_internal * (
            1.0 + components["source_power_relative_curve_internal"]
        )
        if np.any(source_sample_internal <= 0.0):
            raise ValueError("Realized source power spectrum must remain positive.")

        settings = CONFIG["SPECTROMETER"]
        sample_power_internal = source_sample_internal * physical_reflectance_internal
        reference_power_internal = source_reference_internal

        # ILS 积分中心使用隐藏 true_pixel_center，而输出标签始终使用固定 reported 轴。
        # 绝对光功率未知：先按 1 W/nm 相对尺度计算，再定标参考峰值电子数。
        reference_unit_e = expected_photoelectrons(
            internal_absolute_nm,
            reference_power_internal,
            true_pixel_center_nm,
            float(settings["REFERENCE_EXPOSURE_S"]),
            1.0,
        )
        detector_multiplier = float(noise["detector_noise_multiplier"])
        photon_budget_divisor = max(detector_multiplier, 1.0) ** 2
        target_reference_peak_e = (
            float(settings["REFERENCE_PEAK_ELECTRONS"]) / photon_budget_divisor
        )
        optical_psd_scale = target_reference_peak_e / max(
            float(np.max(reference_unit_e)), 1.0e-30
        )
        expected_reference_e = expected_photoelectrons(
            internal_absolute_nm,
            reference_power_internal,
            true_pixel_center_nm,
            float(settings["REFERENCE_EXPOSURE_S"]),
            optical_psd_scale,
        )
        expected_sample_e = expected_photoelectrons(
            internal_absolute_nm,
            sample_power_internal,
            true_pixel_center_nm,
            float(settings["SAMPLE_EXPOSURE_S"]),
            optical_psd_scale,
        )

        if detector_multiplier > 0.0:
            pixel_response_error = rng.normal(
                0.0,
                float(settings["PIXEL_RESPONSE_NONUNIFORMITY_SIGMA_REL"]),
                size=len(self.reported_nm),
            )
        else:
            pixel_response_error = np.zeros_like(self.reported_nm)

        sample_counts, sample_stats = adc_counts_from_electrons(
            expected_sample_e,
            float(settings["SAMPLE_EXPOSURE_S"]),
            int(settings["SAMPLE_AVERAGES"]),
            detector_multiplier,
            pixel_response_error,
            rng,
        )
        reference_counts, reference_stats = adc_counts_from_electrons(
            expected_reference_e,
            float(settings["REFERENCE_EXPOSURE_S"]),
            int(settings["REFERENCE_AVERAGES"]),
            detector_multiplier,
            pixel_response_error,
            rng,
        )
        dark_counts, dark_stats = adc_counts_from_electrons(
            np.zeros_like(expected_reference_e),
            float(settings["DARK_EXPOSURE_S"]),
            int(settings["DARK_AVERAGES"]),
            detector_multiplier,
            pixel_response_error,
            rng,
        )

        # 去除 ADC bias，并根据曝光时间把独立暗场缩放到样品/参考曝光。
        adc_bias = float(settings["ADC_BIAS_COUNTS"])
        dark_signal_counts = dark_counts - adc_bias
        dark_exposure_s = float(settings["DARK_EXPOSURE_S"])
        if dark_exposure_s <= 0.0:
            raise ValueError("DARK_EXPOSURE_S must be positive.")
        sample_net_counts = (
            sample_counts - adc_bias
            - dark_signal_counts * float(settings["SAMPLE_EXPOSURE_S"]) / dark_exposure_s
        )
        reference_net_counts = (
            reference_counts - adc_bias
            - dark_signal_counts * float(settings["REFERENCE_EXPOSURE_S"]) / dark_exposure_s
        )
        minimum_reference = float(settings["MIN_REFERENCE_NET_COUNTS"])
        invalid_reference = reference_net_counts < minimum_reference
        safe_reference = np.where(
            invalid_reference, minimum_reference, reference_net_counts
        )
        measured = sample_net_counts / safe_reference
        ideal_post_ils = expected_sample_e / np.maximum(expected_reference_e, 1.0e-30)

        # 以下曲线仅用于审计。它们在隐藏真实中心处取值，最终仍由 reported 轴贴标签。
        physical_reflectance_output = np.interp(
            true_pixel_center_nm,
            internal_absolute_nm,
            physical_reflectance_internal,
        )
        source_reference_output = np.interp(
            true_pixel_center_nm,
            internal_absolute_nm,
            source_reference_internal,
        )
        source_sample_output = np.interp(
            true_pixel_center_nm,
            internal_absolute_nm,
            source_sample_internal,
        )
        qe_output = quantum_efficiency(true_pixel_center_nm)
        out_of_range_mask = (measured < 0.0) | (measured > 1.0)
        total_runtime_s = time.perf_counter() - total_started

        metadata = {
            "noise_case": case_name,
            "noise_factor": noise["factor"],
            "noise_level": noise["level"],
            "realization_index": int(realization_index),
            "random_seed": int(seed),
            "generation_runtime_s": float(total_runtime_s),
            "optical_solver_runtime_s": float(solver_runtime_s),
            "reflectance_out_of_unit_interval_fraction": float(np.mean(out_of_range_mask)),
            "invalid_reference_pixel_fraction": float(np.mean(invalid_reference)),
            "sample_saturation_fraction": sample_stats["saturation_fraction"],
            "reference_saturation_fraction": reference_stats["saturation_fraction"],
            "dark_saturation_fraction": dark_stats["saturation_fraction"],
            "target_reference_peak_electrons": float(target_reference_peak_e),
            "optical_psd_scale_w_per_nm": float(optical_psd_scale),
            "output_sampling_nm": self.output_sampling_nm,
            "internal_wavelength_step_nm": float(CONFIG["INTERNAL_WAVELENGTH_STEP_NM"]),
            "internal_wavelength_point_count": int(len(self.internal_absolute_nm)),
            "internal_wavelength_margin_nm": float(self.internal_margin_nm),
            "output_wavelength_point_count": int(len(self.reported_nm)),
            "backend": self.backend,
            "wavelength_axis_used_by_inversion": "fixed_reported_wavelengths",
            "wavelength_axis_scenario": CONFIG["WAVELENGTH_AXIS_SCENARIO"],
            "source_reference_policy": "separate sample/reference/dark acquisitions",
            "spectrometer_signal_equation": (
                "ADC[sample,reference,dark] -> (sample-dark)/(reference-dark)"
            ),
            **noise,
        }
        data = {
            "reported_wavelengths_nm": np.asarray(reported_nm, dtype=float),
            "true_pixel_center_wavelengths_nm": np.asarray(
                true_pixel_center_nm, dtype=float
            ),
            "physical_reflectance": np.asarray(physical_reflectance_output, dtype=float),
            "physical_reflectance_internal": np.asarray(
                physical_reflectance_internal, dtype=float
            ),
            "source_reference": np.asarray(source_reference_output, dtype=float),
            "source_sample": np.asarray(source_sample_output, dtype=float),
            "source_reference_internal": np.asarray(source_reference_internal, dtype=float),
            "source_sample_internal": np.asarray(source_sample_internal, dtype=float),
            "expected_reference_electrons": np.asarray(expected_reference_e, dtype=float),
            "expected_sample_electrons": np.asarray(expected_sample_e, dtype=float),
            "adc_counts_sample": np.asarray(sample_counts, dtype=float),
            "adc_counts_reference": np.asarray(reference_counts, dtype=float),
            "adc_counts_dark": np.asarray(dark_counts, dtype=float),
            "adc_counts_sample_dark_corrected": np.asarray(sample_net_counts, dtype=float),
            "adc_counts_reference_dark_corrected": np.asarray(
                reference_net_counts, dtype=float
            ),
            "reference_normalized_spectrum": np.asarray(ideal_post_ils, dtype=float),
            "measured_spectrum": np.asarray(measured, dtype=float),
            "quantum_efficiency": np.asarray(qe_output, dtype=float),
            "pixel_response_error_rel": np.asarray(pixel_response_error, dtype=float),
            "components": components,
            "metadata": metadata,
        }
        return data, metadata

    def save_one(self, data: dict) -> Path:
        metadata = data["metadata"]
        path = self.output_dir / (
            f"static_spectrum_{metadata['noise_case']}_r{metadata['realization_index']:04d}_"
            f"seed{metadata['random_seed']}.npz"
        )
        material_names = np.asarray(PERTURBED_MATERIALS, dtype="U16")
        components = data["components"]
        reported_nm = components["reported_wavelengths_nm"]
        true_pixel_center_nm = components["true_pixel_center_wavelengths_nm"]
        total_axis_error = components["spectrometer_axis_error_total_nm"]
        high_order_error = components["spectrometer_axis_high_order_error_nm"]
        payload = {
            # 反演可见量：固定报告轴与暗场/参考校正后的单帧光谱。
            "wavelengths": reported_nm / 1000.0,
            "reported_wavelengths_nm": reported_nm,
            "spectrum_measured": data["measured_spectrum"],
            # 兼容 V9 读取逻辑；estimated_calibrated 在真实场景中就是固定 reported 轴。
            "estimated_calibrated_wavelengths": reported_nm / 1000.0,
            "estimated_calibrated_wavelengths_nm": reported_nm,
            "nominal_wavelengths_nm": self.reported_nm,
            # 隐藏像素真实中心和光谱仪轴误差仅用于审计。
            "true_pixel_center_wavelengths": true_pixel_center_nm / 1000.0,
            "true_pixel_center_wavelengths_nm": true_pixel_center_nm,
            "spectrometer_axis_error_total_nm": total_axis_error,
            "spectrometer_axis_offset_error_nm": components[
                "spectrometer_axis_offset_error_nm"
            ],
            "spectrometer_axis_scale_error_nm": components[
                "spectrometer_axis_scale_error_nm"
            ],
            "spectrometer_axis_thermal_error_nm": components[
                "spectrometer_axis_thermal_error_nm"
            ],
            "spectrometer_axis_high_order_error_nm": high_order_error,
            "axis_error_sign_convention": np.asarray("reported_minus_true"),
            "reported_axis_is_fixed": np.asarray(True),
            # 旧字段别名：physical 表示隐藏真实中心；shift 表示 true-reported。
            "physical_wavelengths": true_pixel_center_nm / 1000.0,
            "physical_wavelengths_nm": true_pixel_center_nm,
            "physical_axis_shift_total_nm": -total_axis_error,
            "physical_axis_offset_nm": -components["spectrometer_axis_offset_error_nm"],
            "physical_axis_scale_nm": -components["spectrometer_axis_scale_error_nm"],
            "physical_axis_thermal_nm": -components["spectrometer_axis_thermal_error_nm"],
            "wavelength_axis_estimation_error_nm": total_axis_error,
            "wavelength_error_total_nm": total_axis_error,
            "wavelength_error_absolute_accuracy_nm": high_order_error,
            "calibration_residual_nm": high_order_error,
            # 光源和探测器链审计量。
            "source_reference_power": data["source_reference"],
            "source_sample_power": data["source_sample"],
            "source_power_relative_curve": components["source_power_relative_curve"],
            "source_center_drift_nm": np.asarray(metadata["source_center_drift_nm"]),
            "quantum_efficiency": data["quantum_efficiency"],
            "pixel_response_error_rel": data["pixel_response_error_rel"],
            "expected_photoelectrons_sample": data["expected_sample_electrons"],
            "expected_photoelectrons_reference": data["expected_reference_electrons"],
            "adc_counts_sample": data["adc_counts_sample"],
            "adc_counts_reference": data["adc_counts_reference"],
            "adc_counts_dark": data["adc_counts_dark"],
            "adc_counts_sample_dark_corrected": data[
                "adc_counts_sample_dark_corrected"
            ],
            "adc_counts_reference_dark_corrected": data[
                "adc_counts_reference_dark_corrected"
            ],
            "spectrum_reference_normalized": data["reference_normalized_spectrum"],
            "spectrum_post_ils_ideal": data["reference_normalized_spectrum"],
            "spectrum_physical": data["physical_reflectance"],
            # 结构与噪声真值只用于误差统计，禁止进入拟合起点、先验或排序。
            "layer_names": np.asarray(LAYER_NAMES, dtype="U32"),
            "layer_thickness_um": np.asarray([
                self.layers[name] for name in LAYER_NAMES
            ]),
            "true_air_um": np.asarray(self.layers["Air"]),
            "true_reflector_angle_deg": np.asarray(metadata["reflector_angle_deg"]),
            "reflector_angle_error_deg": np.asarray(
                metadata["reflector_angle_error_deg"]
            ),
            "perturbed_material_names": material_names,
            "material_n_real_rel_delta": np.asarray([
                metadata["material_n_real_rel_delta"][str(name)]
                for name in material_names
            ]),
            "material_k_rel_delta": np.asarray([
                metadata["material_k_rel_delta"][str(name)]
                for name in material_names
            ]),
            "noise_case": np.asarray(metadata["noise_case"]),
            "noise_factor": np.asarray(metadata["noise_factor"]),
            "noise_level": np.asarray(metadata["noise_level"]),
            "realization_index": np.asarray(metadata["realization_index"]),
            "random_seed": np.asarray(metadata["random_seed"]),
            "detector_noise_multiplier": np.asarray(
                metadata["detector_noise_multiplier"]
            ),
            "physical_axis_shift_max_abs_nm": np.asarray(
                metadata["physical_axis_shift_max_abs_nm"]
            ),
            "axis_estimation_error_max_abs_nm": np.asarray(
                metadata["axis_estimation_error_max_abs_nm"]
            ),
            "axis_estimation_error_rms_nm": np.asarray(
                metadata["axis_estimation_error_rms_nm"]
            ),
            "spectrometer_axis_error_max_abs_nm": np.asarray(
                metadata["spectrometer_axis_error_max_abs_nm"]
            ),
            "spectrometer_axis_error_rms_nm": np.asarray(
                metadata["spectrometer_axis_error_rms_nm"]
            ),
            "source_power_curve_peak_rel": np.asarray(
                metadata["source_power_curve_peak_rel"]
            ),
            "sample_saturation_fraction": np.asarray(
                metadata["sample_saturation_fraction"]
            ),
            "reference_saturation_fraction": np.asarray(
                metadata["reference_saturation_fraction"]
            ),
            "invalid_reference_pixel_fraction": np.asarray(
                metadata["invalid_reference_pixel_fraction"]
            ),
            "internal_wavelength_step_nm": np.asarray(
                CONFIG["INTERNAL_WAVELENGTH_STEP_NM"]
            ),
            "internal_wavelength_margin_nm": np.asarray(self.internal_margin_nm),
            "output_sampling_nm": np.asarray(self.output_sampling_nm),
            "ils_enabled": np.asarray(CONFIG["SPECTROMETER"]["ILS_ENABLED"]),
            "ils_fwhm_nm": np.asarray(CONFIG["SPECTROMETER"]["ILS_FWHM_NM"]),
            "wavelength_accuracy_spec_nm": np.asarray(WAVELENGTH_ACCURACY_SPEC_NM),
            "wavelength_axis_scenario": np.asarray(CONFIG["WAVELENGTH_AXIS_SCENARIO"]),
            "frames_per_realization": np.asarray(1),
            "time_series_enabled": np.asarray(False),
            "modulation_enabled": np.asarray(False),
            "optical_backend": np.asarray(self.backend),
            "generator_version": np.asarray(GENERATOR_VERSION),
            "speed_of_light_m_s": np.asarray(C0_M_S),
            "config_json": np.asarray(
                json.dumps(CONFIG, ensure_ascii=False, sort_keys=True)
            ),
            "noise_realization_json": np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        }
        if CONFIG["SPECTROMETER"]["SAVE_INTERNAL_AUDIT_ARRAYS"]:
            payload.update({
                "internal_absolute_wavelengths_nm": components[
                    "internal_absolute_wavelengths_nm"
                ],
                "internal_nominal_wavelengths_nm": components[
                    "internal_absolute_wavelengths_nm"
                ],
                "internal_physical_wavelengths_nm": components[
                    "internal_absolute_wavelengths_nm"
                ],
                "spectrum_physical_internal": data["physical_reflectance_internal"],
                "source_reference_power_internal": data["source_reference_internal"],
                "source_sample_power_internal": data["source_sample_internal"],
                "source_power_relative_curve_internal": components[
                    "source_power_relative_curve_internal"
                ],
            })
        np.savez_compressed(path, **payload)
        return path


# ---------------------------------------------------------------------------
# 代表图
# ---------------------------------------------------------------------------
def save_representative_plot(output_dir: Path, representatives: dict[str, dict]) -> Path:
    # 全案例可达 41 行，优先展示 clean 与四档 combined；若均未选择则最多取前 8 个。
    preferred = ["clean"] + [f"combined_{level}" for level in NOISE_LEVELS]
    cases = [name for name in preferred if name in representatives]
    if not cases:
        cases = list(representatives)[:8]
    fig, axes = plt.subplots(
        len(cases),
        5,
        figsize=(28, max(5, 3.0 * len(cases))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, case_name in enumerate(cases):
        data = representatives[case_name]
        components = data["components"]
        reported_nm = components["reported_wavelengths_nm"]

        axes[row_index, 0].plot(
            reported_nm,
            data["reference_normalized_spectrum"],
            lw=0.8,
            label="ideal post-ILS",
        )
        axes[row_index, 0].plot(
            reported_nm,
            data["measured_spectrum"],
            lw=0.6,
            alpha=0.85,
            label="ADC normalized",
        )
        axes[row_index, 0].set_title(f"{case_name}: spectrum")
        axes[row_index, 0].set_ylabel("Reflectance")
        axes[row_index, 0].legend(fontsize=7)

        axes[row_index, 1].plot(
            reported_nm, data["adc_counts_sample"], lw=0.6, label="sample"
        )
        axes[row_index, 1].plot(
            reported_nm, data["adc_counts_reference"], lw=0.6, label="reference"
        )
        axes[row_index, 1].plot(
            reported_nm, data["adc_counts_dark"], lw=0.6, label="dark"
        )
        axes[row_index, 1].set_title("Raw ADC counts")
        axes[row_index, 1].set_ylabel("Count")
        axes[row_index, 1].legend(fontsize=7)

        axis_component_plotted = False
        for key, label in (
            ("spectrometer_axis_error_total_nm", "total"),
            ("spectrometer_axis_offset_error_nm", "offset"),
            ("spectrometer_axis_scale_error_nm", "scale"),
            ("spectrometer_axis_thermal_error_nm", "thermal"),
            ("spectrometer_axis_high_order_error_nm", "high-order"),
        ):
            values = components[key]
            if np.any(values):
                axes[row_index, 2].plot(reported_nm, values, lw=0.8, label=label)
                axis_component_plotted = True
        axes[row_index, 2].axhline(0.0, color="black", lw=0.6)
        axes[row_index, 2].set_title("Spectrometer axis error: reported - true")
        axes[row_index, 2].set_ylabel("Axis error (nm)")
        if axis_component_plotted:
            axes[row_index, 2].legend(fontsize=7)

        source_ratio = data["source_sample"] / np.maximum(
            data["source_reference"], 1.0e-12
        )
        axes[row_index, 3].plot(reported_nm, source_ratio, lw=0.8)
        axes[row_index, 3].axhline(1.0, color="black", lw=0.6)
        axes[row_index, 3].set_title("Sample/reference source ratio")
        axes[row_index, 3].set_ylabel("Relative power")

        axes[row_index, 4].plot(
            reported_nm,
            data["physical_reflectance"],
            lw=0.7,
            label="R at hidden true centers",
        )
        axes[row_index, 4].plot(
            reported_nm, data["quantum_efficiency"], lw=0.8, label="QE"
        )
        axes[row_index, 4].set_title("Hidden-center reflectance and QE")
        axes[row_index, 4].set_ylabel("Relative value")
        axes[row_index, 4].legend(fontsize=7)

        for ax in axes[row_index]:
            ax.set_xlabel("Fixed reported wavelength (nm)")
            ax.grid(True, alpha=0.3)
    path = output_dir / "representative_static_spectrometer_realizations.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 配置审计
# ---------------------------------------------------------------------------
def wavelength_budget_audit() -> dict:
    settings = CONFIG["SPECTROMETER"]
    start_nm = float(CONFIG["WAVELENGTH_START_UM"]) * 1000.0
    stop_nm = float(CONFIG["WAVELENGTH_STOP_UM"]) * 1000.0
    span_nm = stop_nm - start_nm
    half_span_nm = 0.5 * span_nm
    output_sampling_nm = span_nm / (int(CONFIG["OUTPUT_SAMPLE_POINTS"]) - 1)
    internal_step_nm = float(CONFIG["INTERNAL_WAVELENGTH_STEP_NM"])
    ils_fwhm_nm = float(settings["ILS_FWHM_NM"])
    if CONFIG["WAVELENGTH_AXIS_SCENARIO"] != "fixed_reported_axis_hidden_true_pixel_centers":
        raise ValueError("Unexpected V10 wavelength-axis scenario.")
    if settings["REPORTED_AXIS_POLICY"] != "fixed_factory_calibration":
        raise ValueError("V10 requires a fixed reported wavelength axis.")
    if not np.isclose(
        float(settings["WAVELENGTH_ACCURACY_SPEC_NM"]),
        WAVELENGTH_ACCURACY_SPEC_NM,
    ):
        raise ValueError("Global and spectrometer wavelength-accuracy settings disagree.")
    if internal_step_nm > ils_fwhm_nm / 5.0:
        raise ValueError("The internal grid must provide at least five points per ILS FWHM.")
    if float(settings["REFERENCE_PEAK_ELECTRONS"]) > float(settings["FULL_WELL_ELECTRONS"]):
        raise ValueError("Reference peak electron target exceeds full well.")
    if not 0.0 <= float(settings["QE_EDGE"]) <= float(settings["QE_PEAK"]) <= 1.0:
        raise ValueError("QE settings must satisfy 0 <= edge <= peak <= 1.")

    level_audit = {}
    for level, values in NOISE_LEVELS.items():
        scale_edge_bound_nm = (
            float(values["axis_scale_max_ppm"]) * 1.0e-6 * half_span_nm
        )
        total_triangle_bound_nm = (
            float(values["axis_offset_max_nm"])
            + scale_edge_bound_nm
            + float(values["thermal_drift_max_nm"])
            + float(values["calibration_residual_max_nm"])
        )
        level_audit[level] = {
            "axis_offset_bound_nm": float(values["axis_offset_max_nm"]),
            "axis_scale_edge_bound_nm": scale_edge_bound_nm,
            "axis_thermal_bound_nm": float(values["thermal_drift_max_nm"]),
            "axis_high_order_bound_nm": float(values["calibration_residual_max_nm"]),
            "total_axis_error_triangle_bound_nm": total_triangle_bound_nm,
            "within_wavelength_accuracy_spec": bool(
                total_triangle_bound_nm <= WAVELENGTH_ACCURACY_SPEC_NM
            ),
            "source_center_drift_bound_nm": float(
                values["source_center_drift_max_nm"]
            ),
            "is_normal_operation_level": level != "out_of_spec_stress",
        }
    normal_levels = [name for name in NOISE_LEVELS if name != "out_of_spec_stress"]
    if not all(
        level_audit[name]["within_wavelength_accuracy_spec"]
        for name in normal_levels
    ):
        raise ValueError(
            "A normal-operation V10 total axis-error budget exceeds the +/-0.05 nm spec."
        )
    if level_audit["out_of_spec_stress"]["within_wavelength_accuracy_spec"]:
        raise ValueError(
            "The stress level should remain explicitly outside the product accuracy spec."
        )

    return {
        "wavelength_axis_scenario": CONFIG["WAVELENGTH_AXIS_SCENARIO"],
        "reported_axis_is_fixed": True,
        "axis_error_sign_convention": "reported_minus_true",
        "wavelength_span_nm": span_nm,
        "output_sample_points": int(CONFIG["OUTPUT_SAMPLE_POINTS"]),
        "output_sampling_nm": output_sampling_nm,
        "internal_wavelength_step_nm": internal_step_nm,
        "ils_fwhm_nm": ils_fwhm_nm,
        "internal_points_per_ils_fwhm": ils_fwhm_nm / internal_step_nm,
        "wavelength_accuracy_spec_nm": WAVELENGTH_ACCURACY_SPEC_NM,
        "levels": level_audit,
    }


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10 single-frame StackRT/TMM spectrometer dataset generator."
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="all、单个噪声因子，或逗号分隔的完整案例名。",
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="每个非 clean 案例的独立 realization 数。"
    )
    parser.add_argument("--clean-repeats", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=["tmm", "api", "batch"],
        default="api",
        help="正式数据使用 api/batch StackRT；tmm 仅用于快速自检。",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--wavelength-start-nm", type=float, default=None,
        help="Override CONFIG wavelength start while preserving the default when omitted.",
    )
    parser.add_argument(
        "--wavelength-stop-nm", type=float, default=None,
        help="Override CONFIG wavelength stop while preserving the default when omitted.",
    )
    parser.add_argument(
        "--output-sampling-nm", type=float, default=None,
        help="Requested output sampling; defaults to the original CONFIG sampling.",
    )
    parser.add_argument("--describe", action="store_true")
    return parser.parse_args()


def apply_wavelength_cli_overrides(args: argparse.Namespace) -> None:
    original_start_nm = float(CONFIG["WAVELENGTH_START_UM"]) * 1000.0
    original_stop_nm = float(CONFIG["WAVELENGTH_STOP_UM"]) * 1000.0
    original_points = int(CONFIG["OUTPUT_SAMPLE_POINTS"])
    original_sampling_nm = (original_stop_nm - original_start_nm) / (original_points - 1)
    start_nm = original_start_nm if args.wavelength_start_nm is None else float(args.wavelength_start_nm)
    stop_nm = original_stop_nm if args.wavelength_stop_nm is None else float(args.wavelength_stop_nm)
    sampling_nm = original_sampling_nm if args.output_sampling_nm is None else float(args.output_sampling_nm)
    if not np.isfinite(start_nm) or not np.isfinite(stop_nm) or not np.isfinite(sampling_nm):
        raise ValueError("Wavelength overrides must be finite.")
    if start_nm <= 0.0 or stop_nm <= start_nm or sampling_nm <= 0.0:
        raise ValueError("Require 0 < wavelength start < stop and positive output sampling.")
    points = int(round((stop_nm - start_nm) / sampling_nm)) + 1
    if points < 2:
        raise ValueError("Wavelength range must contain at least two output samples.")
    actual_sampling_nm = (stop_nm - start_nm) / (points - 1)
    if not math.isclose(actual_sampling_nm, sampling_nm, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"Requested span is not an integer multiple of output sampling: "
            f"requested={sampling_nm}, actual={actual_sampling_nm} nm."
        )
    original_span_nm = original_stop_nm - original_start_nm
    new_span_nm = stop_nm - start_nm
    axis_scale_factor = original_span_nm / new_span_nm
    if not math.isclose(new_span_nm, original_span_nm, rel_tol=0.0, abs_tol=1.0e-12):
        for values in NOISE_LEVELS.values():
            values["axis_scale_max_ppm"] = (
                float(values["axis_scale_max_ppm"]) * axis_scale_factor
            )
    CONFIG["WAVELENGTH_START_UM"] = start_nm / 1000.0
    CONFIG["WAVELENGTH_STOP_UM"] = stop_nm / 1000.0
    CONFIG["OUTPUT_SAMPLE_POINTS"] = points
    CONFIG["WIDEBAND_AXIS_SCALE_POLICY"] = {
        "policy": "preserve_original_absolute_edge_scale_error_bound",
        "reference_start_nm": original_start_nm,
        "reference_stop_nm": original_stop_nm,
        "reference_span_nm": original_span_nm,
        "target_span_nm": new_span_nm,
        "axis_scale_ppm_factor": axis_scale_factor,
    }


def main() -> None:
    args = parse_args()
    apply_wavelength_cli_overrides(args)
    budget = wavelength_budget_audit()
    if args.describe:
        print(json.dumps({
            "version": GENERATOR_VERSION,
            "config": CONFIG,
            "noise_levels": NOISE_LEVELS,
            "noise_factors": NOISE_FACTORS,
            "cases": all_case_names(),
            "wavelength_budget": budget,
            "level_amplitude_fraction_range": LEVEL_AMPLITUDE_FRACTION_RANGE,
        }, indent=2, ensure_ascii=False))
        return
    if args.repeats < 1 or args.clean_repeats < 1:
        raise ValueError("Repeat counts must be positive.")

    cases = select_cases(args.cases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(__file__).resolve().parents[2] / "04_results_and_datasets"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else output_root / f"static_stackrt_v10_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    generator = StaticDatasetGenerator(args.backend, output_dir)
    rows: list[dict] = []
    failures: list[dict] = []
    representatives: dict[str, dict] = {}
    with OpticalSolver(args.backend, output_dir / "_stackrt_batch_bridge") as solver:
        for case_index, case_name in enumerate(cases):
            repeats = args.clean_repeats if case_name == "clean" else args.repeats
            for realization_index in range(repeats):
                seed = int(args.seed + case_index * 1_000_000 + realization_index)
                try:
                    data, metadata = generator.generate_one(
                        solver, case_name, realization_index, seed
                    )
                    npz_path = generator.save_one(data)
                    row = {
                        "noise_case": metadata["noise_case"],
                        "noise_factor": metadata["noise_factor"],
                        "noise_level": metadata["noise_level"],
                        "realization_index": metadata["realization_index"],
                        "random_seed": metadata["random_seed"],
                        "reflector_angle_deg": metadata["reflector_angle_deg"],
                        "reflector_angle_error_deg": metadata["reflector_angle_error_deg"],
                        "reported_axis_is_fixed": metadata["reported_axis_is_fixed"],
                        "axis_offset_nm": metadata["axis_offset_nm"],
                        "axis_scale_ppm": metadata["axis_scale_ppm"],
                        "thermal_drift_nm": metadata["thermal_drift_nm"],
                        "calibration_residual_peak_nm": metadata[
                            "calibration_residual_peak_nm"
                        ],
                        "spectrometer_axis_error_max_abs_nm": metadata[
                            "spectrometer_axis_error_max_abs_nm"
                        ],
                        "spectrometer_axis_error_rms_nm": metadata[
                            "spectrometer_axis_error_rms_nm"
                        ],
                        "physical_axis_shift_max_abs_nm": metadata[
                            "physical_axis_shift_max_abs_nm"
                        ],
                        "axis_estimation_error_max_abs_nm": metadata[
                            "axis_estimation_error_max_abs_nm"
                        ],
                        "axis_estimation_error_rms_nm": metadata[
                            "axis_estimation_error_rms_nm"
                        ],
                        "wavelength_accuracy_within_spec": metadata[
                            "wavelength_accuracy_within_spec"
                        ],
                        "source_center_drift_nm": metadata["source_center_drift_nm"],
                        "source_power_curve_peak_rel": metadata[
                            "source_power_curve_peak_rel"
                        ],
                        "detector_noise_multiplier": metadata[
                            "detector_noise_multiplier"
                        ],
                        "target_reference_peak_electrons": metadata[
                            "target_reference_peak_electrons"
                        ],
                        "sample_saturation_fraction": metadata[
                            "sample_saturation_fraction"
                        ],
                        "reference_saturation_fraction": metadata[
                            "reference_saturation_fraction"
                        ],
                        "invalid_reference_pixel_fraction": metadata[
                            "invalid_reference_pixel_fraction"
                        ],
                        "output_sampling_nm": metadata["output_sampling_nm"],
                        "internal_wavelength_step_nm": metadata[
                            "internal_wavelength_step_nm"
                        ],
                        "optical_solver_runtime_s": metadata[
                            "optical_solver_runtime_s"
                        ],
                        "generation_runtime_s": metadata["generation_runtime_s"],
                        "npz_path": str(npz_path),
                    }
                    rows.append(row)
                    representatives.setdefault(case_name, data)
                    print(
                        f"[{case_name} {realization_index + 1}/{repeats}] "
                        f"axis_error={metadata['axis_estimation_error_max_abs_nm']:.6g} nm, "
                        f"source_center={metadata['source_center_drift_nm']:.6g} nm, "
                        f"detector_x={metadata['detector_noise_multiplier']:.3g}, "
                        f"runtime={metadata['generation_runtime_s']:.4f}s"
                    )
                except Exception as exc:
                    failures.append({
                        "case": case_name,
                        "realization": realization_index,
                        "seed": seed,
                        "error": str(exc),
                    })
                    print(f"ERROR [{case_name} {realization_index + 1}/{repeats}]: {exc}")

    index_path = output_dir / "dataset_index.csv"
    if rows:
        with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_path = save_representative_plot(output_dir, representatives) if representatives else None
    manifest = {
        "version": GENERATOR_VERSION,
        "created": timestamp,
        "backend": args.backend,
        "cases": cases,
        "repeats_non_clean": args.repeats,
        "repeats_clean": args.clean_repeats,
        "random_seed": args.seed,
        "config": CONFIG,
        "noise_levels": NOISE_LEVELS,
        "noise_factors": NOISE_FACTORS,
        "level_amplitude_fraction_range": LEVEL_AMPLITUDE_FRACTION_RANGE,
        "wavelength_axis_scenario": CONFIG["WAVELENGTH_AXIS_SCENARIO"],
        "reported_axis_is_fixed": True,
        "axis_error_sign_convention": "reported_minus_true",
        "wavelength_accuracy_spec_nm": WAVELENGTH_ACCURACY_SPEC_NM,
        "source_center_drift_hard_max_nm": SOURCE_CENTER_DRIFT_HARD_MAX_NM,
        "wavelength_budget_audit": budget,
        "dataset_count": len(rows),
        "dataset_index": str(index_path) if rows else None,
        "representative_plot": str(plot_path) if plot_path else None,
        "failures": failures,
    }
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"OUTPUT_DIR={output_dir}")
    print(f"MANIFEST={manifest_path}")
    if failures:
        raise RuntimeError(f"Simulation failures: {failures}")


if __name__ == "__main__":
    main()
