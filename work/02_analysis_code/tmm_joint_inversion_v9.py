"""V9 单帧静态光谱六参数 TMM 反演。

拟合参数顺序固定为：
    [Air_um, HSQ_nm, PSS_nm, SOC_nm, TiO2_nm, reflector_angle_deg]

本文件只读取 main_v9.py 输出的“光谱仪估计校正轴”和参考归一化测量光谱。真实物理
波长轴、真实膜厚、真实角度和噪声 realization 仅用于输入契约审计或拟合完成后的误差
统计，不能进入 model、residual、全局搜索起点或候选结果排序。

求解流程：
1. 检查 NPZ 是 V9 单帧静态数据，并按估计校正轴裁剪波段、执行 stride。
2. 用相同的名义材料和六参数 TMM 构造反射率模型。
3. 差分进化在完整参数边界内进行全局搜索，不注入真值或名义真值起点。
4. 从进化后的种群选择参数空间中相互分散的候选点。
5. 对每个候选点运行有界 least_squares，只在已收敛结果中按目标函数排序。
6. 拟合和排序结束后才读取真值，计算腔长、膜厚和角度误差。
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
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# 参数定义和单位约定
# ---------------------------------------------------------------------------
# Air 使用 um，四个膜层参数使用 nm，Angle 使用 RefReflector 第一入射介质内的度数。
# BOUNDS 是优化器唯一允许的搜索区域。这里的中心点不能被当作“名义真值起点”强制加入；
# 所有起点均来自与真值无关的 Latin hypercube 和差分进化种群。

VERSION = "tmm_joint_inversion_v9"
C0_M_S = 299_792_458.0
INCIDENT_MEDIUM_N = 5.8284
MAX_WAVELENGTH_ERROR_NM = 0.2
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
# 1 mm 腔默认采用 0.02 nm 局部步长和 0.04 nm 全局步长，并由 validate_sampling
# 对每个输入文件重新检查，而不是只相信配置中的名义采样间隔。

@dataclass
class FitConfig:
    input_dir: str
    wavelength_min_nm: float = 450.0
    wavelength_max_nm: float = 580.0
    stride: int = 1
    global_stride: int = 2
    global_popsize: int = 8
    global_maxiter: int = 40
    multistarts: int = 8
    max_nfev: int = 600
    local_gtol: float = 1.0e-5
    workers: int = 4
    random_seed: int = 20260810
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

def tmm_reflectance(wavelengths_um: np.ndarray, values: np.ndarray) -> np.ndarray:
    # 核心前向模型：对每个波长并行计算 p 偏振多层膜反射率。
    # 输入 wavelengths_um 必须是光谱仪估计校正轴；禁止在拟合时替换成 NPZ 中隐藏的
    # physical_wavelengths，否则会人为消除待评估的校正轴剩余误差。
    thicknesses_um, reflector_angle_deg = vector_to_thickness_um(values)
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    n_matrix = np.vstack([material_n(name, wavelengths_um) for name in LAYER_NAMES])
    cos_values = propagation_cosines(n_matrix, reflector_angle_deg)
    q_values = n_matrix / cos_values
    k0 = 2.0 * np.pi / (wavelengths_um * 1.0e-6)
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

def reflector_to_air_angle_deg(reflector_angle_deg: float) -> float:
    # 该换算仅用于结果报告，不参与 TMM 前向计算。反演参数始终是 reflector 内角度。
    invariant = INCIDENT_MEDIUM_N * math.sin(math.radians(float(reflector_angle_deg)))
    if abs(invariant) > 1.0: raise ValueError("Reflector angle has no Air-layer solution.")
    return math.degrees(math.asin(invariant))

# ---------------------------------------------------------------------------
# V9 输入契约与预处理
# ---------------------------------------------------------------------------
# 实验可见输入只有 estimated_calibrated_wavelengths 和 spectrum_measured。
# physical_wavelengths 与 wavelength_axis_estimation_error_nm 在这里仅用于形状/上限审计，
# 不会传入返回字典中的拟合数组，因此后续 model 和 residual 无法访问真实物理轴。
#
# spectrum_measured 已由生成端执行固定参考谱归一化，可能因源谱失配或探测器误差略微
# 超出 [0,1]；这里不再次裁剪或归一化，避免改变噪声统计。

def load_fit_input(npz_path: Path, config: FitConfig) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        if bool(scalar(data, "ils_enabled", False)):
            raise ValueError("ILS must be disabled.")
        if bool(scalar(data, "time_series_enabled", False)):
            raise ValueError("Static data required.")
        if int(scalar(data, "frames_per_realization", 1)) != 1:
            raise ValueError("One frame required.")
        if bool(scalar(data, "modulation_enabled", False)):
            raise ValueError("No modulation accepted.")
        generator_version = str(scalar(data, "generator_version", "unknown"))
        if generator_version != "main_v9":
            raise ValueError(
                f"V9 inversion requires a main_v9 dataset, got {generator_version!r}."
            )

        # 明确要求 V9 关键字段齐全，避免把旧版 NPZ 静默解释成新数据口径。
        required = {
            "estimated_calibrated_wavelengths",
            "physical_wavelengths",
            "wavelength_axis_estimation_error_nm",
            "spectrum_measured",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"V9 NPZ is missing required fields: {missing}")

        # 拟合横轴只取 estimated_calibrated_wavelengths。
        # physical_wavelengths 读取后仅检查 shape，随后不会写入 measurement 返回值。
        wavelengths_um = np.asarray(
            data["estimated_calibrated_wavelengths"], dtype=float
        )
        physical_wavelengths_um = np.asarray(data["physical_wavelengths"], dtype=float)
        spectrum = np.asarray(data["spectrum_measured"], dtype=float)
        axis_error_nm = np.asarray(
            data["wavelength_axis_estimation_error_nm"], dtype=float
        )
        if (
            wavelengths_um.ndim != 1
            or spectrum.shape != wavelengths_um.shape
            or physical_wavelengths_um.shape != wavelengths_um.shape
            or axis_error_nm.shape != wavelengths_um.shape
        ):
            raise ValueError(
                "V9 wavelength, axis-error, and spectrum arrays must be matching 1D arrays."
            )
        if np.any(np.diff(wavelengths_um) <= 0.0):
            raise ValueError("Estimated calibrated wavelength axis must be monotonic.")
        if float(np.max(np.abs(axis_error_nm))) > MAX_WAVELENGTH_ERROR_NM + 1.0e-9:
            raise ValueError("Estimated-axis residual exceeds 0.2 nm.")

        # 先按估计校正轴选择波段，再按 stride 保留等索引样本。
        # 如果校正轴轻微非均匀，actual_step_nm 使用保留点的中位间隔。
        wavelength_nm = wavelengths_um * 1000.0
        mask = (
            (wavelength_nm >= config.wavelength_min_nm)
            & (wavelength_nm <= config.wavelength_max_nm)
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
            "wavelength_axis_source": "estimated_calibrated_wavelengths",
            "axis_estimation_error_max_abs_nm": float(np.max(np.abs(axis_error_nm))),
            "axis_estimation_error_rms_nm": float(np.sqrt(np.mean(axis_error_nm**2))),
            "physical_axis_shift_max_abs_nm": float(
                scalar(data, "physical_axis_shift_max_abs_nm", float("nan"))
            ),
            "source_center_drift_nm": float(
                scalar(data, "source_center_drift_nm", 0.0)
            ),
            "source_power_curve_peak_rel": float(
                scalar(data, "source_power_curve_peak_rel", 0.0)
            ),
        }
    selected_wavelengths = wavelengths_um[indices]
    return {
        "path": str(npz_path.resolve()),
        "wavelengths_um": selected_wavelengths,
        "spectrum": spectrum[indices],
        "actual_step_nm": float(np.median(np.diff(selected_wavelengths))) * 1000.0,
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
    # 核心反演函数。measurement 中不包含任何结构真值，只包含估计校正轴、测量光谱
    # 和不参与目标函数的案例元数据。
    wavelengths = measurement["wavelengths_um"]
    observed = measurement["spectrum"]
    # 所有参数组合都与同一条 observed 比较；scale 只负责无量纲化残差。
    scale = robust_scale(observed)
    # 下面三行是对波长和观测光谱进行下采样，使用全局步长来选择波长和观测值的索引
    # 全局搜索使用较稀疏但已通过 Nyquist 审计的轴，以降低差分进化每次评估成本；
    # 局部最小二乘和最终 RMSE 始终使用完整的局部轴。
    global_indices = np.arange(0, len(wavelengths), max(1, int(config.global_stride)))
    global_wavelengths = wavelengths[global_indices]
    global_observed = observed[global_indices]
    lower, upper = bounds_arrays()

    def model(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        # values 的固定顺序为 Air、四层膜厚和 reflector 角度。
        axis = global_wavelengths if use_global else wavelengths
        return tmm_reflectance(axis, values) # values是一个包含6个参数的数组，分别是Air层厚度、HSQ层厚度、PSS层厚度、SOC层厚度、TiO2层厚度和反射器角度
                                             # axis是波长数组，返回的是一个与axis长度相同的反射率数组
    def residual(values: np.ndarray, use_global: bool = False) -> np.ndarray:
        # residual 返回逐波长归一化残差向量；robust loss 由外层优化器施加。
        target = global_observed if use_global else observed
        return (model(values, use_global) - target) / scale

    started = time.perf_counter()
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

    # 阶段二：对进化种群中相互分散的候选点逐一执行有界局部最小二乘。
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
    attempts.sort(key=lambda row: (not row["success"], row["cost"]))
    successful = [row for row in attempts if row["success"]]
    runtime_s = time.perf_counter() - started
    if not successful:
        return {
            "success": False, "runtime_s": runtime_s, "attempts": attempts,
            "global_summary": {},
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
            "initial_population_source": "LatinHypercube; no truth, nominal truth, or forced center point",
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

def result_row(measurement: dict, fit: dict, truth: dict[str, float]) -> dict:
    metadata = measurement["metadata"]
    row = {
        "input_npz": measurement["path"], **metadata,
        "success": bool(fit["success"]), "fit_runtime_s": float(fit["runtime_s"]),
    }
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
    ax.set_title("V9 six-parameter single-frame inversion")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / "cavity_error_boxplot.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    factors = [value for value in summary["noise_factor"].drop_duplicates() if value != "clean"]
    levels = ["low", "medium", "high"]
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
        selected_names = [name for name in representatives if name == "clean" or name.endswith("_high")]
        fig, axes = plt.subplots(len(selected_names), 1, figsize=(13, max(5, 2.8 * len(selected_names))), squeeze=False, constrained_layout=True)
        for index, name in enumerate(selected_names):
            item = representatives[name]; ax = axes[index, 0]
            ax.plot(item["wavelengths_um"] * 1000.0, item["observed"], lw=0.7, label="measured")
            ax.plot(item["wavelengths_um"] * 1000.0, item["fitted"], lw=0.7, label="fit")
            ax.set_title(name); ax.set_ylabel("Reflectance"); ax.grid(True, alpha=0.3); ax.legend()
        axes[-1, 0].set_xlabel("Estimated calibrated wavelength (nm)")
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
# 默认输入目录是占位符，正式运行前应手动修改或使用 --input-dir/--inputs 指定 V9 NPZ。
# workers 只并行不同 NPZ；单个 NPZ 内的差分进化固定 workers=1，保证随机过程可复现。

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V9 six-parameter static TMM inversion.")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\static_stackrt_v9_20260819_121305"))
    parser.add_argument("--pattern", default="static_spectrum_*.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--wavelength-min-nm", type=float, default=450.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=580.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--global-stride", type=int, default=2)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--local-gtol", type=float, default=1.0e-5)
    parser.add_argument("--workers", type=int, default=2) # 并行处理多个npz文件
    parser.add_argument("--random-seed", type=int, default=20260810)
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
    if not inputs: raise FileNotFoundError("No V9 static NPZ files matched.")
    config = FitConfig(
        input_dir=input_label,
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        stride=args.stride,
        global_stride=args.global_stride,
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
            truth, _noise_audit = load_evaluation_truth(npz_path) # 真正的结构参数
            row = result_row(measurement, fit, truth)
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
            "observable": "reference-normalized single static reflectance spectrum I(lambda)",
            "wavelength_axis": "estimated calibrated axis from main_v9 NPZ",
            "hidden_physical_axis_used_for_fit": False,
            "source_model": "fixed reference envelope; separate center drift and smooth power-spectrum variation",
            "ils_enabled": False,
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
        "# V9 单帧静态光谱六参数反演报告", "", "## 计算口径", "",
        f"- 输入光谱数：{len(inputs)}。",
        f"- 成功收敛数：{int(table['success'].sum())}。",
        "- 同时拟合 Air 腔长、HSQ、PSS、SOC、TiO2 和 reflector 入射角。",
        "- ILS 已关闭；每个 realization 只包含一帧静态光谱并独立反演。",
        "- TMM \u4ec5\u4f7f\u7528\u5149\u8c31\u4eea\u4f30\u8ba1\u6821\u6b63\u8f74\uff1b\u9690\u85cf\u7684\u771f\u5b9e\u7269\u7406\u8f74\u53ea\u7528\u4e8e\u5ba1\u8ba1\uff0c\u4e0d\u8fdb\u5165\u62df\u5408\u3002",
        "- \u5bbd\u5e26\u5149\u6e90\u4e2d\u5fc3\u6ce2\u957f\u6f02\u79fb\u4e0e\u5e73\u6ed1\u529f\u7387\u8c31\u53d8\u5316\u5206\u5f00\u5efa\u6a21\uff0c\u5176\u771f\u503c\u4e0d\u8fdb\u5165\u62df\u5408\u3002",
        "- 真值仅在拟合和结果排序全部完成后加载，用于误差评估。",
        "- 默认局部采样间隔为 0.02 nm，全局搜索采样间隔为 0.04 nm；两者均接受逐文件 Nyquist 审计。",
        f"- 局部优化在归一化参数空间使用 gtol={config.local_gtol:.3g}；未触发 SciPy 收敛状态的结果仍不参与最优解排序。",
        "", "## 分案例汇总", "", dataframe_to_markdown(summary), "",
        "## 解释限制", "",
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
