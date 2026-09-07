"""V9 单帧静态光谱数据生成器。

本文件保持 V8 的“一个 realization 对应一帧静态光谱”结构，但重新划分了波长轴误差
与宽带光源误差，核心数据流程如下：

1. nominal_wavelengths：设计时的名义像素波长网格，只用于定义仿真采样点。
2. physical_wavelengths：光真正经过样品时的物理波长轴，由 offset、scale 和热漂移决定。
3. estimated_calibrated_wavelengths：光谱仪提供给反演程序的估计校正轴，等于物理轴再叠加
   校正残差；反演程序只能看到该轴，不能使用隐藏的物理轴。
4. StackRT/TMM 在 physical_wavelengths 上计算样品真实反射率。
5. 宽带光源中心漂移和功率谱形状变化只修改 S(lambda)，不再错误地移动反射率波长轴。
6. 样品强度除以固定参考光谱，得到实际送入反演程序的参考归一化光谱。

NPZ 同时保存实验可见量和仿真审计量。physical_wavelengths、结构真值和噪声真值只能用于
事后误差分析，不应进入拟合残差、起点生成或最优结果排序。
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
# AXIS_ERROR_HARD_MAX_NM 约束“估计校正轴 - 真实物理轴”的残差；光源中心漂移单独约束，
# 两者属于不同物理量，不能再相加成一条 total_wavelength_error 曲线。

GENERATOR_VERSION = "main_v9"
C0_M_S = 299_792_458.0
INCIDENT_MEDIUM_N = 5.8284
ANGLE_MAX_DEG = 0.1
AXIS_ERROR_HARD_MAX_NM = 0.2
SOURCE_CENTER_DRIFT_HARD_MAX_NM = 0.2
REALIZATION_MIN_FRACTION = 0.8
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
    "WAVELENGTH_START_UM": 0.2,
    "WAVELENGTH_STOP_UM": 0.8,
    "SPECTRAL_SAMPLING_NM": 0.02,
    "REFLECTOR_ANGLE_NOMINAL_DEG": 0.0,
    "ILS_ENABLED": False,
    "TIME_SERIES_ENABLED": False,
    "FRAMES_PER_REALIZATION": 1,
    "SOURCE_REFERENCE_CENTER_NM": 515.0,
    "SOURCE_ENVELOPE_SIGMA_NM": 55.0,
    "SOURCE_ENVELOPE_FLOOR_REL": 0.15,
    "SOURCE_POWER_CORRELATION_LENGTH_NM": 12.0,
    "CALIBRATION_RESIDUAL_CORRELATION_LENGTH_NM": 30.0,
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
# 这里把误差分成三组：
# 1. 几何/材料误差：angle、材料 n/k；
# 2. 波长轴误差：axis_offset、axis_scale、thermal_drift 改变真实像素映射，
#    absolute_accuracy 表示校正后仍未消除的平滑残差；
# 3. 强度域误差：source_center_drift 移动宽带源包络中心，source_power 修改包络形状，
#    detector 再施加整帧增益和反射率偏置。
#
# low/medium/high 给出各变量允许的最大幅度。每个 realization 的实际幅度在最大值的
# 80%~100% 之间抽取，使不同等级保持明显区分，同时通过独立 seed 统计随机方向和曲线形状。
NOISE_LEVELS = {
    "low": {
        "n_real_sigma_rel": 5.0e-4,
        "k_sigma_rel": 1.0e-2,
        "angle_max_deg": 0.01,
        "source_center_drift_max_nm": 0.020,
        "source_power_curve_peak_rel": 2.0e-3,
        "axis_offset_max_nm": 0.0002,
        "axis_scale_max_ppm": 10.0,
        "thermal_drift_max_nm": 0.001,
        "calibration_residual_max_nm": 0.020,
        "frame_gain_sigma_rel": 2.0e-4,
        "reflectance_offset_sigma_abs": 2.0e-4,
    },
    "medium": {
        "n_real_sigma_rel": 2.0e-3,
        "k_sigma_rel": 5.0e-2,
        "angle_max_deg": 0.05,
        "source_center_drift_max_nm": 0.080,
        "source_power_curve_peak_rel": 1.0e-2,
        "axis_offset_max_nm": 0.001,
        "axis_scale_max_ppm": 30.0,
        "thermal_drift_max_nm": 0.005,
        "calibration_residual_max_nm": 0.080,
        "frame_gain_sigma_rel": 1.0e-3,
        "reflectance_offset_sigma_abs": 1.0e-3,
    },
    "high": {
        "n_real_sigma_rel": 5.0e-3,
        "k_sigma_rel": 1.0e-1,
        "angle_max_deg": 0.10,
        "source_center_drift_max_nm": 0.200,
        "source_power_curve_peak_rel": 5.0e-2,
        "axis_offset_max_nm": 0.005,
        "axis_scale_max_ppm": 60.0,
        "thermal_drift_max_nm": 0.020,
        "calibration_residual_max_nm": 0.120,
        "frame_gain_sigma_rel": 5.0e-3,
        "reflectance_offset_sigma_abs": 5.0e-3,
    },
}


# ---------------------------------------------------------------------------
# 噪声案例选择与幅度抽样
# ---------------------------------------------------------------------------
# 单因素案例只激活对应字段；combined 同时激活该等级的所有字段；clean 全部置零。
# signed_level_value 用于允许正负方向的误差，positive_level_value 用于只关心幅值的量。

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
        "detector": ("frame_gain_sigma_rel", "reflectance_offset_sigma_abs"),
    }[factor]
    profile = zero_profile()
    for key in keys:
        profile[key] = float(source[key])
    return profile


def all_case_names() -> list[str]:
    return ["clean"] + [f"{factor}_{level}" for factor in NOISE_FACTORS for level in NOISE_LEVELS]


def parse_case(case_name: str) -> tuple[str, str]:
    if case_name == "clean":
        return "clean", "clean"
    factor, level = case_name.rsplit("_", 1)
    if factor not in NOISE_FACTORS or level not in NOISE_LEVELS:
        raise ValueError(f"Unknown case: {case_name}")
    return factor, level


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


def signed_level_value(maximum: float, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    magnitude = float(maximum) * rng.uniform(REALIZATION_MIN_FRACTION, 1.0)
    return magnitude if rng.integers(0, 2) else -magnitude


def positive_level_value(maximum: float, rng: np.random.Generator) -> float:
    if maximum <= 0.0:
        return 0.0
    return float(maximum) * rng.uniform(REALIZATION_MIN_FRACTION, 1.0)


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


def smooth_curve_with_peak(
    wavelengths_nm: np.ndarray,
    target_peak_nm: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # 如果目标峰值 <= 0，则返回全零曲线。
    # 这表示：该误差分量在当前 realization 中不生效。
    if target_peak_nm <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)

    # 计算波长轴的平均采样间隔，用于把相关长度转换成“像素级 sigma”。
    # 例如：相关长度 30 nm，采样间隔 0.02 nm，那么 sigma_pixels ≈ 1500。
    spacing_nm = float(np.median(np.diff(wavelengths_nm)))
    sigma_pixels = float(correlation_length_nm) / spacing_nm

    # 反复试验，直到生成一条非零、可用的平滑随机曲线。
    # 目的：避免随机噪声在去掉常数和线性项后恰好接近零，导致无法缩放到目标峰值。
    for _ in range(50):
        # 1) 先生成一条标准正态随机噪声
        # 2) 用高斯滤波做平滑，模拟“真实系统中缓慢变化的校准残差”
        raw = gaussian_filter1d(
            rng.normal(size=len(wavelengths_nm)),
            sigma=sigma_pixels,
            mode="reflect",
        )

        # 3) 去掉常数项和线性项：
        #    - 常值偏移：axis_offset 已经单独处理
        #    - 线性漂移：axis_scale/thermal 已经单独处理
        #    - 这样剩下的就是“高阶残差”，对应 absolute_accuracy
        curve = remove_constant_and_linear_terms(raw, wavelengths_nm)

        # 4) 求这条曲线的绝对值最大值
        peak = float(np.max(np.abs(curve)))

        # 5) 如果曲线非零，则按目标峰值做放缩
        #    这样 final_curve 的最大绝对值 = target_peak_nm
        if peak > 1.0e-12:
            return curve * (float(target_peak_nm) / peak)

    # 如果连续 50 次都没得到非零曲线，则报错
    raise RuntimeError("Could not generate a nonzero smooth wavelength curve.")


def smooth_relative_curve_with_peak(
    wavelengths_nm: np.ndarray,
    target_peak_rel: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_peak_rel <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)
    spacing_nm = float(np.median(np.diff(wavelengths_nm)))
    sigma_pixels = float(correlation_length_nm) / spacing_nm
    for _ in range(50):
        raw = gaussian_filter1d(
            rng.normal(size=len(wavelengths_nm)),
            sigma=sigma_pixels,
            mode="reflect",
        )
        curve = raw - float(np.mean(raw))
        peak = float(np.max(np.abs(curve)))
        if peak > 1.0e-12:
            return curve * (float(target_peak_rel) / peak)
    raise RuntimeError("Could not generate a nonzero smooth source-power curve.")


# 宽带光源名义功率谱采用“高斯包络 + 非零底座”。非零底座可以避免参考谱边缘接近零时，
# sample/reference 除法放大数值误差。每条包络都按自身最大值归一化，从而让中心漂移主要
# 表示谱形变化；整帧总功率变化由 detector 的 frame_gain_error_rel 单独描述。

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
# 三个波长轴的关系为：
#   physical = nominal + offset + scale + thermal
#   estimated_calibrated = physical + calibration_residual
# 因此 calibration_residual = estimated_calibrated - physical。

def realize_noise(
    case_name: str,
    nominal_wavelengths_nm: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    # 第一步：解析案例并取得当前等级实际激活的噪声字段。
    factor, level = parse_case(case_name)
    profile = active_profile(level, factor)
    # 第二步：为每个可扰动膜层独立抽取 n、k 相对误差。
    n_delta = {
        name: signed_level_value(profile["n_real_sigma_rel"], rng)
        for name in PERTURBED_MATERIALS
    }
    k_delta = {
        name: signed_level_value(profile["k_sigma_rel"], rng)
        for name in PERTURBED_MATERIALS
    }
    angle_deg = positive_level_value(profile["angle_max_deg"], rng)
    if angle_deg > ANGLE_MAX_DEG + 1.0e-12:
        raise ValueError("Realized reflector angle exceeds 0.1 deg.")

    # 第三步：生成宽带光源误差。中心漂移只移动 S(lambda) 的包络中心；
    # source_power_curve 是乘法谱形扰动，不改变波长坐标。
    source_center_drift_nm = signed_level_value(
        profile["source_center_drift_max_nm"], rng
    )
    if abs(source_center_drift_nm) > SOURCE_CENTER_DRIFT_HARD_MAX_NM + 1.0e-12:
        raise ValueError("Source center drift exceeds the 0.2 nm hard limit.")
    source_power_peak = positive_level_value(
        profile["source_power_curve_peak_rel"], rng
    )
    source_power_curve = smooth_relative_curve_with_peak(
        nominal_wavelengths_nm,
        source_power_peak,
        CONFIG["SOURCE_POWER_CORRELATION_LENGTH_NM"],
        rng,
    )

    # 第四步：构造真实像素到波长的映射变化。
    # offset 是常数，scale 以波段中心为零点形成斜线，thermal 当前按整轴常数漂移处理。
    axis_offset_nm = signed_level_value(profile["axis_offset_max_nm"], rng)
    thermal_drift_nm = signed_level_value(profile["thermal_drift_max_nm"], rng)
    axis_scale_ppm = signed_level_value(profile["axis_scale_max_ppm"], rng)
    center_nm = 0.5 * (nominal_wavelengths_nm[0] + nominal_wavelengths_nm[-1])
    offset_curve = np.full_like(nominal_wavelengths_nm, axis_offset_nm)
    scale_curve = axis_scale_ppm * 1.0e-6 * (nominal_wavelengths_nm - center_nm)
    thermal_curve = np.full_like(nominal_wavelengths_nm, thermal_drift_nm)
    physical_axis_shift = offset_curve + scale_curve + thermal_curve
    physical_wavelengths_nm = nominal_wavelengths_nm + physical_axis_shift

    # 第五步：模拟光谱仪给出的估计校正轴。
    # absolute_accuracy 对应校正后剩余的平滑高阶误差，是反演无法直接知道的轴残差。
    calibration_peak = positive_level_value(
        profile["calibration_residual_max_nm"], rng
    )
    calibration_residual = smooth_curve_with_peak(
        nominal_wavelengths_nm,
        calibration_peak,
        CONFIG["CALIBRATION_RESIDUAL_CORRELATION_LENGTH_NM"],
        rng,
    )
    estimated_calibrated_nm = physical_wavelengths_nm + calibration_residual
    if float(np.max(np.abs(calibration_residual))) > AXIS_ERROR_HARD_MAX_NM + 1.0e-12:
        raise ValueError("Calibrated-axis residual exceeds the 0.2 nm hard limit.")
    if np.any(np.diff(physical_wavelengths_nm) <= 0.0):
        raise ValueError("Physical wavelength mapping is not monotonic.")
    if np.any(np.diff(estimated_calibrated_nm) <= 0.0):
        raise ValueError("Estimated calibrated wavelength mapping is not monotonic.")

    # 第六步：生成参考归一化之后仍可能存在的探测器整帧增益和加性偏置。
    frame_gain_error_rel = signed_level_value(profile["frame_gain_sigma_rel"], rng)
    reflectance_offset_abs = signed_level_value(profile["reflectance_offset_sigma_abs"], rng)
    metadata = {
        "case": case_name,
        "factor": factor,
        "level": level,
        "profile": profile,
        "realization_fraction_range": [REALIZATION_MIN_FRACTION, 1.0],
        "material_n_real_rel_delta": n_delta,
        "material_k_rel_delta": k_delta,
        "reflector_angle_deg": angle_deg,
        "source_center_drift_nm": source_center_drift_nm,
        "source_power_curve_peak_rel": float(np.max(np.abs(source_power_curve))),
        "axis_offset_nm": axis_offset_nm,
        "axis_scale_ppm": axis_scale_ppm,
        "axis_scale_edge_max_nm": float(np.max(np.abs(scale_curve))),
        "thermal_drift_nm": thermal_drift_nm,
        "physical_axis_shift_max_abs_nm": float(np.max(np.abs(physical_axis_shift))),
        "calibration_residual_peak_nm": float(np.max(np.abs(calibration_residual))),
        "axis_estimation_error_max_abs_nm": float(np.max(np.abs(calibration_residual))),
        "axis_estimation_error_rms_nm": float(np.sqrt(np.mean(calibration_residual**2))),
        "frame_gain_error_rel": frame_gain_error_rel,
        "reflectance_offset_abs": reflectance_offset_abs,
    }
    components = {
        "physical_axis_offset_nm": offset_curve,
        "physical_axis_scale_nm": scale_curve,
        "physical_axis_thermal_nm": thermal_curve,
        "physical_axis_shift_total_nm": physical_axis_shift,
        "calibration_residual_nm": calibration_residual,
        "source_power_relative_curve": source_power_curve,
        "physical_wavelengths_nm": physical_wavelengths_nm,
        "estimated_calibrated_wavelengths_nm": estimated_calibrated_nm,
    }
    return metadata, components


# ---------------------------------------------------------------------------
# 光学求解器统一接口
# ---------------------------------------------------------------------------
# reflectance 的输入必须是真实物理波长轴。api/batch 最终调用 Lumerical stackrt，
# tmm 后端调用本文件中的等价前向模型。后端切换不会改变噪声生成和 NPZ 字段定义。

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
        start_nm = CONFIG["WAVELENGTH_START_UM"] * 1000.0
        stop_nm = CONFIG["WAVELENGTH_STOP_UM"] * 1000.0
        step_nm = CONFIG["SPECTRAL_SAMPLING_NM"]
        count = int(round((stop_nm - start_nm) / step_nm)) + 1
        self.nominal_nm = np.linspace(start_nm, stop_nm, count)
        self.layers = dict(CONFIG["LAYERS"])

    def generate_one(
        self,
        solver: OpticalSolver,
        case_name: str,
        realization_index: int,
        seed: int,
    ) -> tuple[dict, dict]:
        rng = np.random.default_rng(seed)
        # 先固定本 realization 的全部误差，再将真实物理轴送入前向求解器。
        noise, components = realize_noise(case_name, self.nominal_nm, rng)
        physical_nm = components["physical_wavelengths_nm"]
        estimated_nm = components["estimated_calibrated_wavelengths_nm"]

        started = time.perf_counter()
        physical_reflectance = solver.reflectance(
            physical_nm / 1000.0,
            self.layers,
            noise["reflector_angle_deg"],
            noise["material_n_real_rel_delta"],
            noise["material_k_rel_delta"],
        )
        generation_runtime_s = time.perf_counter() - started
        if CONFIG["ILS_ENABLED"]:
            raise RuntimeError("V9 requires ILS to be disabled.")

        # 构造参考测量和样品测量的宽带源功率谱：
        # I_ref(lambda)    = S_ref(lambda)
        # I_sample(lambda) = S_sample(lambda) * R_true(lambda)
        # 当前模型把参考样品反射率和探测器名义响应归一化为 1。
        reference_center_nm = float(CONFIG["SOURCE_REFERENCE_CENTER_NM"])
        source_reference = source_power_envelope(physical_nm, reference_center_nm)
        source_sample_base = source_power_envelope(
            physical_nm,
            reference_center_nm + noise["source_center_drift_nm"],
        )
        source_sample = source_sample_base * (
            1.0 + components["source_power_relative_curve"]
        )
        if np.any(source_sample <= 0.0):
            raise ValueError("Realized source power spectrum must remain positive.")

        reference_intensity = source_reference
        sample_intensity = source_sample * physical_reflectance
        # 固定参考谱与当前样品谱不同时，S_sample/S_ref 不会完全抵消，因而中心漂移和
        # 功率谱变化会保留在待拟合光谱中。这对应“参考测量与样品测量不同步”的情形。
        reference_normalized = sample_intensity / np.maximum(reference_intensity, 1.0e-12)
        measured = reference_normalized * (1.0 + noise["frame_gain_error_rel"])
        measured += noise["reflectance_offset_abs"]
        # 参考归一化结果可能因增益、偏置或源谱失配略微超出 [0,1]。这里不裁剪，
        # 否则会把真实的幅度误差变成额外的非线性截断误差；只记录越界比例供审计。
        out_of_range_mask = (measured < 0.0) | (measured > 1.0)

        metadata = {
            "noise_case": case_name,
            "noise_factor": noise["factor"],
            "noise_level": noise["level"],
            "realization_index": int(realization_index),
            "random_seed": int(seed),
            "generation_runtime_s": float(generation_runtime_s),
            "reflectance_out_of_unit_interval_fraction": float(np.mean(out_of_range_mask)),
            "backend": self.backend,
            "wavelength_axis_used_by_inversion": "estimated_calibrated_wavelengths",
            "source_reference_policy": "fixed nominal reference; sample source may drift",
            **noise,
        }
        return {
            "physical_reflectance": np.asarray(physical_reflectance, dtype=float),
            "source_reference": np.asarray(source_reference, dtype=float),
            "source_sample": np.asarray(source_sample, dtype=float),
            "reference_intensity": np.asarray(reference_intensity, dtype=float),
            "sample_intensity": np.asarray(sample_intensity, dtype=float),
            "reference_normalized_spectrum": np.asarray(reference_normalized, dtype=float),
            "measured_spectrum": np.asarray(measured, dtype=float),
            "components": components,
            "metadata": metadata,
        }, metadata

    def save_one(self, data: dict) -> Path:
        # NPZ 字段分为两类：
        # 1. 反演可见量：wavelengths/estimated_calibrated_wavelengths、spectrum_measured；
        # 2. 仿真审计量：physical_wavelengths、结构真值、噪声分量和源谱真值。
        # tmm_joint_inversion_v9.py 必须只把第一类字段送入残差函数。
        metadata = data["metadata"]
        path = self.output_dir / (
            f"static_spectrum_{metadata['noise_case']}_r{metadata['realization_index']:04d}_"
            f"seed{metadata['random_seed']}.npz"
        )
        material_names = np.asarray(PERTURBED_MATERIALS, dtype="U16")
        components = data["components"]
        estimated_nm = components["estimated_calibrated_wavelengths_nm"]
        physical_nm = components["physical_wavelengths_nm"]
        calibration_residual = components["calibration_residual_nm"]
        np.savez_compressed(
            path,
            wavelengths=estimated_nm / 1000.0,
            nominal_wavelengths_nm=self.nominal_nm,
            reported_wavelengths_nm=estimated_nm,
            estimated_calibrated_wavelengths=estimated_nm / 1000.0,
            estimated_calibrated_wavelengths_nm=estimated_nm,
            physical_wavelengths=physical_nm / 1000.0,
            physical_wavelengths_nm=physical_nm,
            physical_axis_shift_total_nm=components["physical_axis_shift_total_nm"],
            physical_axis_offset_nm=components["physical_axis_offset_nm"],
            physical_axis_scale_nm=components["physical_axis_scale_nm"],
            physical_axis_thermal_nm=components["physical_axis_thermal_nm"],
            wavelength_axis_estimation_error_nm=calibration_residual,
            wavelength_error_total_nm=calibration_residual,
            wavelength_error_absolute_accuracy_nm=calibration_residual,
            source_reference_power=data["source_reference"],
            source_sample_power=data["source_sample"],
            source_power_relative_curve=components["source_power_relative_curve"],
            source_center_drift_nm=np.asarray(metadata["source_center_drift_nm"]),
            intensity_reference=data["reference_intensity"],
            intensity_sample=data["sample_intensity"],
            spectrum_reference_normalized=data["reference_normalized_spectrum"],
            spectrum_measured=data["measured_spectrum"],
            spectrum_physical=data["physical_reflectance"],
            layer_names=np.asarray(LAYER_NAMES, dtype="U32"),
            layer_thickness_um=np.asarray([self.layers[name] for name in LAYER_NAMES]),
            true_air_um=np.asarray(self.layers["Air"]),
            true_reflector_angle_deg=np.asarray(metadata["reflector_angle_deg"]),
            perturbed_material_names=material_names,
            material_n_real_rel_delta=np.asarray([
                metadata["material_n_real_rel_delta"][name] for name in material_names
            ]),
            material_k_rel_delta=np.asarray([
                metadata["material_k_rel_delta"][name] for name in material_names
            ]),
            noise_case=np.asarray(metadata["noise_case"]),
            noise_factor=np.asarray(metadata["noise_factor"]),
            noise_level=np.asarray(metadata["noise_level"]),
            realization_index=np.asarray(metadata["realization_index"]),
            random_seed=np.asarray(metadata["random_seed"]),
            frame_gain_error_rel=np.asarray(metadata["frame_gain_error_rel"]),
            reflectance_offset_abs=np.asarray(metadata["reflectance_offset_abs"]),
            physical_axis_shift_max_abs_nm=np.asarray(metadata["physical_axis_shift_max_abs_nm"]),
            axis_estimation_error_max_abs_nm=np.asarray(metadata["axis_estimation_error_max_abs_nm"]),
            axis_estimation_error_rms_nm=np.asarray(metadata["axis_estimation_error_rms_nm"]),
            source_power_curve_peak_rel=np.asarray(metadata["source_power_curve_peak_rel"]),
            ils_enabled=np.asarray(False),
            frames_per_realization=np.asarray(1),
            time_series_enabled=np.asarray(False),
            modulation_enabled=np.asarray(False),
            optical_backend=np.asarray(self.backend),
            generator_version=np.asarray(GENERATOR_VERSION),
            speed_of_light_m_s=np.asarray(C0_M_S),
            config_json=np.asarray(json.dumps(CONFIG, ensure_ascii=False, sort_keys=True)),
            noise_realization_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        return path


# 代表图同时显示待拟合光谱、校正轴剩余误差、样品/参考源谱比值和各轴分量，
# 用于确认横轴误差与纵向功率误差没有在实现中再次混为同一种噪声。

def save_representative_plot(output_dir: Path, representatives: dict[str, dict]) -> Path:
    cases = list(representatives)
    fig, axes = plt.subplots(
        len(cases), 4, figsize=(24, max(5, 3.1 * len(cases))), squeeze=False, constrained_layout=True,
    )
    for row_index, case_name in enumerate(cases):
        data = representatives[case_name]
        components = data["components"]
        estimated_nm = components["estimated_calibrated_wavelengths_nm"]
        axes[row_index, 0].plot(estimated_nm, data["measured_spectrum"], lw=0.7)
        axes[row_index, 0].set_title(f"{case_name}: measured spectrum")
        axes[row_index, 0].set_ylabel("Estimated reflectance")

        axes[row_index, 1].plot(
            estimated_nm,
            components["calibration_residual_nm"],
            lw=0.9,
            label="estimated - physical",
        )
        axes[row_index, 1].axhline(0.0, color="black", lw=0.6)
        axes[row_index, 1].set_title("Calibrated-axis residual")
        axes[row_index, 1].set_ylabel("Axis error (nm)")

        source_ratio = data["source_sample"] / np.maximum(data["source_reference"], 1.0e-12)
        axes[row_index, 2].plot(estimated_nm, source_ratio, lw=0.8)
        axes[row_index, 2].axhline(1.0, color="black", lw=0.6)
        axes[row_index, 2].set_title("Sample/reference source-power ratio")
        axes[row_index, 2].set_ylabel("Relative power")

        for name in (
            "physical_axis_offset_nm",
            "physical_axis_scale_nm",
            "physical_axis_thermal_nm",
            "calibration_residual_nm",
        ):
            values = components[name]
            if np.any(values):
                axes[row_index, 3].plot(estimated_nm, values, lw=0.8, label=name)
        axes[row_index, 3].set_title("Axis components")
        axes[row_index, 3].set_ylabel("Component (nm)")
        if axes[row_index, 3].lines:
            axes[row_index, 3].legend(fontsize=7)
        for ax in axes[row_index]:
            ax.set_xlabel("Estimated calibrated wavelength (nm)")
            ax.grid(True, alpha=0.3)
    path = output_dir / "representative_static_noise_realizations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# 启动前进行确定性的误差预算审计。物理轴变化、校正残差和源中心漂移分别检查，
# 不把不同物理量简单求和。high 等级任一配置超过硬上限时直接终止，避免生成越界数据。

def wavelength_budget_audit() -> dict:
    half_span_nm = 0.5 * (
        CONFIG["WAVELENGTH_STOP_UM"] - CONFIG["WAVELENGTH_START_UM"]
    ) * 1000.0
    audit = {}
    for level, values in NOISE_LEVELS.items():
        scale_edge_nm = values["axis_scale_max_ppm"] * 1.0e-6 * half_span_nm
        physical_mapping_bound_nm = (
            values["axis_offset_max_nm"]
            + scale_edge_nm
            + values["thermal_drift_max_nm"]
        )
        calibration_residual_bound_nm = values["calibration_residual_max_nm"]
        source_center_bound_nm = values["source_center_drift_max_nm"]
        audit[level] = {
            "physical_mapping_triangle_bound_nm": physical_mapping_bound_nm,
            "calibration_residual_max_nm": calibration_residual_bound_nm,
            "source_center_drift_max_nm": source_center_bound_nm,
            "axis_within_0p2_nm": bool(
                physical_mapping_bound_nm <= AXIS_ERROR_HARD_MAX_NM
                and calibration_residual_bound_nm <= AXIS_ERROR_HARD_MAX_NM
            ),
            "source_center_within_0p2_nm": bool(
                source_center_bound_nm <= SOURCE_CENTER_DRIFT_HARD_MAX_NM
            ),
        }
    if not all(
        item["axis_within_0p2_nm"] and item["source_center_within_0p2_nm"]
        for item in audit.values()
    ):
        raise ValueError("Configured V9 wavelength or source-center budget exceeds 0.2 nm.")
    return audit


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
# --backend api 是正式 StackRT 数据生成路径；tmm 适合 smoke test；batch 用于无 GUI 的
# Lumerical 命令行环境。--repeats 控制每个非 clean 案例的独立 Monte Carlo 样本数。

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V9 single-frame StackRT/TMM Monte Carlo dataset generator." # 蒙特卡洛是指采用多个repeat对随机过程采样得到统计分布
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--repeats", type=int, default=10, help="Realizations per non-clean case") # 这个参数是去除随机种子的影响，所以设置30次同噪声等级输入下的结果？
    parser.add_argument("--clean-repeats", type=int, default=1) # 对于clean noise case的重复次数，默认是1次
    parser.add_argument("--backend", choices=["tmm", "api", "batch"], default="api") # 这个是计算光学反射率的求解器后端，默认采用lumapi，不可用时会报错，也可以选择采用tmm模型生成光谱
    parser.add_argument("--seed", type=int, default=20260810) # 这个是随机种子，可以改成当前日期
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--describe", action="store_true") # 如果true，则增加输出配置和噪声等级信息
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = wavelength_budget_audit()
    if args.describe:
        print(json.dumps({
            "version": GENERATOR_VERSION,
            "config": CONFIG,
            "noise_levels": NOISE_LEVELS,
            "noise_factors": NOISE_FACTORS,
            "cases": all_case_names(),
            "wavelength_budget": budget,
            "realization_fraction_range": [REALIZATION_MIN_FRACTION, 1.0],
        }, indent=2, ensure_ascii=False))
        return
    if args.repeats < 1 or args.clean_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    cases = select_cases(args.cases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(__file__).resolve().parents[2] / "04_results_and_datasets" # 默认输出结果路径为父路径下的04_results_and_datasets
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else output_root / f"static_stackrt_v9_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    generator = StaticDatasetGenerator(args.backend, output_dir)
    rows = []
    failures = []
    representatives = {}
    with OpticalSolver(args.backend, output_dir / "_stackrt_batch_bridge") as solver: # 光学求解器封装
        for case_index, case_name in enumerate(cases):
            repeats = args.clean_repeats if case_name == "clean" else args.repeats
            for realization_index in range(repeats):
                seed = int(args.seed + case_index * 1_000_000 + realization_index) #不同的noise case下的realization的随机种子不同
                try:
                    data, metadata = generator.generate_one(solver, case_name, realization_index, seed) # 生成一个噪声case下的光谱数据
                    npz_path = generator.save_one(data) # 保存为npz文件，保留配置信息
                    row = {
                        "noise_case": metadata["noise_case"],
                        "noise_factor": metadata["noise_factor"],
                        "noise_level": metadata["noise_level"],
                        "realization_index": metadata["realization_index"],
                        "random_seed": metadata["random_seed"],
                        "reflector_angle_deg": metadata["reflector_angle_deg"],
                        "physical_axis_shift_max_abs_nm": metadata["physical_axis_shift_max_abs_nm"],
                        "axis_estimation_error_max_abs_nm": metadata["axis_estimation_error_max_abs_nm"],
                        "axis_estimation_error_rms_nm": metadata["axis_estimation_error_rms_nm"],
                        "source_center_drift_nm": metadata["source_center_drift_nm"],
                        "source_power_curve_peak_rel": metadata["source_power_curve_peak_rel"],
                        "frame_gain_error_rel": metadata["frame_gain_error_rel"],
                        "reflectance_offset_abs": metadata["reflectance_offset_abs"],
                        "generation_runtime_s": metadata["generation_runtime_s"],
                        "npz_path": str(npz_path),
                    }
                    rows.append(row)
                    representatives.setdefault(case_name, data)
                    print(
                        f"[{case_name} {realization_index + 1}/{repeats}] "
                        f"axis_residual={metadata['axis_estimation_error_max_abs_nm']:.6g} nm, "
                        f"source_center={metadata['source_center_drift_nm']:.6g} nm, "
                        f"angle={metadata['reflector_angle_deg']:.6g} deg, "
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
        "realization_fraction_range": [REALIZATION_MIN_FRACTION, 1.0],
        "axis_error_hard_max_nm": AXIS_ERROR_HARD_MAX_NM,
        "source_center_drift_hard_max_nm": SOURCE_CENTER_DRIFT_HARD_MAX_NM,
        "wavelength_budget_audit": budget,
        "dataset_count": len(rows),
        "dataset_index": str(index_path) if rows else None,
        "representative_plot": str(plot_path) if plot_path else None,
        "failures": failures,
    }
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"MANIFEST={manifest_path}")
    if failures:
        raise RuntimeError(f"Simulation failures: {failures}")


if __name__ == "__main__":
    main()
