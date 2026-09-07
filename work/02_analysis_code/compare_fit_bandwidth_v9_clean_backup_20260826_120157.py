"""V9 paired comparison of static-spectrum fitting bandwidths.

The script reads existing main_v9 NPZ files only. Fitting uses the inversion-visible
estimated calibrated wavelength axis and measured spectrum. Hidden physical axes never
enter fitting. Truth is loaded only after each fit for evaluation.
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

VERSION = "tmm_fit_bandwidth_comparison_v9"
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
    wavelength_min_nm: float = 200
    wavelength_max_nm: float = 800
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
# Hidden physical wavelength and axis-error arrays are not read in the fitting path.
# Only the reported calibrated axis and measured spectrum can reach model/residual.
#
# spectrum_measured 已由生成端执行固定参考谱归一化，可能因源谱失配或探测器误差略微
# 超出 [0,1]；这里不再次裁剪或归一化，避免改变噪声统计。


def load_fit_input(npz_path: Path, config: FitConfig) -> dict:
    """Load or evaluate data without exposing truth to fitting."""
    with np.load(npz_path, allow_pickle=False) as data:
        if bool(scalar(data, "ils_enabled", False)):
            raise ValueError("ILS must be disabled for this V9 comparison.")
        if bool(scalar(data, "time_series_enabled", False)):
            raise ValueError("Static data required.")
        if int(scalar(data, "frames_per_realization", 1)) != 1:
            raise ValueError("One frame per realization required.")
        if bool(scalar(data, "modulation_enabled", False)):
            raise ValueError("No modulation accepted.")
        generator_version = str(scalar(data, "generator_version", "unknown"))
        if generator_version != "main_v9":
            raise ValueError(f"Expected main_v9 data, got {generator_version!r}.")
        required = {"estimated_calibrated_wavelengths", "spectrum_measured"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Missing inversion-visible fields: {missing}")

        # Encoding-safe explanatory comment.
        wavelengths_um = np.asarray(data["estimated_calibrated_wavelengths"], dtype=float)
        spectrum = np.asarray(data["spectrum_measured"], dtype=float)
        if wavelengths_um.ndim != 1 or spectrum.shape != wavelengths_um.shape:
            raise ValueError("Visible wavelength axis and spectrum must be matching 1D arrays.")
        if np.any(np.diff(wavelengths_um) <= 0.0):
            raise ValueError("Estimated calibrated wavelength axis must be strictly increasing.")
        wavelength_nm = wavelengths_um * 1000.0
        mask = ((wavelength_nm >= config.wavelength_min_nm)
                & (wavelength_nm <= config.wavelength_max_nm))
        indices = np.where(mask)[0][::max(1, int(config.stride))]
        if len(indices) < 50:
            raise ValueError("Too few visible samples after window mask and stride.")
        metadata = {
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "noise_factor": str(scalar(data, "noise_factor", "unknown")),
            "noise_level": str(scalar(data, "noise_level", "unknown")),
            "realization_index": int(scalar(data, "realization_index", 0)),
            "random_seed": int(scalar(data, "random_seed", 0)),
            "generator_version": generator_version,
            "optical_backend": str(scalar(data, "optical_backend", "unknown")),
            "wavelength_axis_source": "estimated_calibrated_wavelengths",
        }
    selected = wavelengths_um[indices]
    return {
        "path": str(npz_path.resolve()), "wavelengths_um": selected,
        "spectrum": spectrum[indices], "point_count": int(len(indices)),
        "actual_wavelength_min_nm": float(selected[0] * 1000.0),
        "actual_wavelength_max_nm": float(selected[-1] * 1000.0),
        "actual_step_nm": float(np.median(np.diff(selected))) * 1000.0,
        "metadata": metadata,
    }


def load_evaluation_truth(npz_path: Path) -> tuple[dict[str, float], dict]:
    """Load or evaluate data without exposing truth to fitting."""
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


# ---------------------------------------------------------------------------
# Encoding-safe explanatory comment.
# ---------------------------------------------------------------------------

# WINDOWS = [
#     {"width_nm": 20, "min_nm": 505.0, "max_nm": 525.0},
#     {"width_nm": 40, "min_nm": 495.0, "max_nm": 535.0},
#     {"width_nm": 60, "min_nm": 485.0, "max_nm": 545.0},
#     {"width_nm": 80, "min_nm": 475.0, "max_nm": 555.0},
#     {"width_nm": 100, "min_nm": 465.0, "max_nm": 565.0},
#     {"width_nm": 130, "min_nm": 450.0, "max_nm": 580.0},
    
# ]
WINDOWS = [
    {"width_nm": 20,  "min_nm": 505.0, "max_nm": 525.0},
    {"width_nm": 40,  "min_nm": 495.0, "max_nm": 535.0},
    {"width_nm": 60,  "min_nm": 485.0, "max_nm": 545.0},
    {"width_nm": 80,  "min_nm": 475.0, "max_nm": 555.0},
    {"width_nm": 100, "min_nm": 465.0, "max_nm": 565.0},
    {"width_nm": 130, "min_nm": 450.0, "max_nm": 580.0},
    {"width_nm": 160, "min_nm": 435.0, "max_nm": 595.0},
    {"width_nm": 200, "min_nm": 415.0, "max_nm": 615.0},
    {"width_nm": 240, "min_nm": 395.0, "max_nm": 635.0},
    {"width_nm": 280, "min_nm": 375.0, "max_nm": 655.0},
    {"width_nm": 320, "min_nm": 355.0, "max_nm": 675.0},
]
SELECTED_CASES = ("clean", "absolute_accuracy_low")
CASE_ORDER = {name: index for index, name in enumerate(SELECTED_CASES)}


def read_input_identity(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "path": npz_path.resolve(),
            "noise_case": str(scalar(data, "noise_case", npz_path.stem)),
            "realization_index": int(scalar(data, "realization_index", 0)),
            "random_seed": int(scalar(data, "random_seed", 0)),
            "generator_version": str(scalar(data, "generator_version", "unknown")),
        }


def discover_inputs(input_dir: Path, max_realizations_per_case: int | None = None) -> list[dict]:
    identities = []
    for case in SELECTED_CASES:
        items = []
        for path in sorted(input_dir.glob(f"static_spectrum_{case}_r*.npz")):
            identity = read_input_identity(path)
            if identity["noise_case"] != case:
                continue
            if identity["generator_version"] != "main_v9":
                raise ValueError(f"Unexpected generator for {path}: {identity['generator_version']}")
            items.append(identity)
        items.sort(key=lambda item: (
            item["realization_index"], item["random_seed"], item["path"].name
        ))
        if max_realizations_per_case is not None:
            items = items[:max_realizations_per_case]
        if not items:
            raise FileNotFoundError(f"No realization found for required case {case!r}.")
        identities.extend(items)
    identities.sort(key=lambda item: (
        CASE_ORDER[item["noise_case"]], item["realization_index"],
        item["random_seed"], item["path"].name,
    ))
    for pair_index, item in enumerate(identities):
        item["pair_index"] = pair_index
        item["pair_key"] = (
            f"{item['noise_case']}|r{item['realization_index']:04d}|seed{item['random_seed']}"
        )
    return identities


def config_for_window(base: FitConfig, window: dict) -> FitConfig:
    values = asdict(base)
    values["wavelength_min_nm"] = float(window["min_nm"])
    values["wavelength_max_nm"] = float(window["max_nm"])
    return FitConfig(**values)


def process_band_task(payload: tuple[int, dict, dict, FitConfig]) -> dict:
    task_index, identity, window, base_config = payload
    npz_path = Path(identity["path"])
    window_config = config_for_window(base_config, window)
    # Encoding-safe explanatory comment.
    optimizer_seed = int(base_config.random_seed + identity["pair_index"] * 1009)
    try:
        measurement = load_fit_input(npz_path, window_config)
        sampling_audit = validate_sampling(measurement, window_config)
        fit = fit_measurement(measurement, window_config, optimizer_seed)
        return {
            "ok": True, "task_index": task_index, "identity": identity,
            "window": window, "optimizer_seed": optimizer_seed,
            "measurement": measurement, "sampling_audit": sampling_audit, "fit": fit,
        }
    except Exception as exc:
        return {
            "ok": False, "task_index": task_index, "identity": identity,
            "window": window, "optimizer_seed": optimizer_seed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def result_row(outcome: dict, truth: dict[str, float]) -> dict:
    measurement, fit = outcome["measurement"], outcome["fit"]
    identity, window = outcome["identity"], outcome["window"]
    row = {
        "pair_key": identity["pair_key"], "input_npz": measurement["path"],
        **measurement["metadata"],
        "window_width_nm": int(window["width_nm"]),
        "window_min_nm": float(window["min_nm"]),
        "window_max_nm": float(window["max_nm"]),
        "actual_wavelength_min_nm": measurement["actual_wavelength_min_nm"],
        "actual_wavelength_max_nm": measurement["actual_wavelength_max_nm"],
        "spectral_point_count": measurement["point_count"],
        "actual_step_nm": measurement["actual_step_nm"],
        "optimizer_seed": outcome["optimizer_seed"],
        "success": bool(fit["success"]), "fit_runtime_s": float(fit["runtime_s"]),
    }
    if not fit["success"]:
        return row
    for index, name in enumerate(PARAMS):
        unit = parameter_unit(name)
        fitted, true_value = float(fit["x"][index]), float(truth[name])
        row[f"fit_{name}_{unit}"] = fitted
        row[f"truth_{name}_{unit}"] = true_value
        row[f"error_{name}_{unit}"] = fitted - true_value
    row["air_error_nm"] = row["error_Air_um"] * 1000.0
    row["air_abs_error_nm"] = abs(row["air_error_nm"])
    for name in FILM_PARAMS:
        row[f"{name}_abs_error_nm"] = abs(row[f"error_{name}_nm"])
    row["film_mae_nm"] = float(np.mean([
        row[f"{name}_abs_error_nm"] for name in FILM_PARAMS
    ]))
    row["angle_abs_error_deg"] = abs(row["error_Angle_deg"])
    row["fit_air_angle_deg"] = reflector_to_air_angle_deg(row["fit_Angle_deg"])
    row["truth_air_angle_deg"] = reflector_to_air_angle_deg(row["truth_Angle_deg"])
    observed_std = max(float(np.std(measurement["spectrum"])), np.finfo(float).eps)
    row["observed_std_reflectance"] = observed_std
    row["rmse_reflectance"] = float(fit["rmse_reflectance"])
    row["normalized_spectral_rmse"] = row["rmse_reflectance"] / observed_std
    row["optimizer_cost_audit_only"] = float(fit["cost"])
    row["nfev"] = int(fit["nfev"])
    row["condition_number"] = float(fit["condition_number"])
    row["boundary_hits"] = ";".join(fit["boundary_hits"])
    row["boundary_hit_count"] = len(fit["boundary_hits"])
    row["selected_start_rank"] = int(fit["attempts"][0]["start_rank"])
    return row


def attempt_rows(outcome: dict, truth: dict[str, float]) -> list[dict]:
    measurement, fit = outcome["measurement"], outcome["fit"]
    identity, window = outcome["identity"], outcome["window"]
    rows = []
    for rank, attempt in enumerate(fit.get("attempts", []), start=1):
        row = {
            "pair_key": identity["pair_key"], "input_npz": measurement["path"],
            "noise_case": measurement["metadata"]["noise_case"],
            "realization_index": measurement["metadata"]["realization_index"],
            "window_width_nm": int(window["width_nm"]),
            "optimizer_seed": outcome["optimizer_seed"], "rank": rank,
            "success": bool(attempt["success"]), "status": int(attempt["status"]),
            "message": attempt["message"], "optimality": float(attempt["optimality"]),
            "cost": float(attempt["cost"]),
            "rmse_reflectance": float(attempt["rmse_reflectance"]),
            "nfev": int(attempt["nfev"]),
            "global_population_index": int(attempt["global_population_index"]),
            "global_energy": float(attempt["global_energy"]),
        }
        for index, name in enumerate(PARAMS):
            unit = parameter_unit(name)
            row[f"x0_{name}_{unit}"] = float(attempt["x0"][index])
            row[f"fit_{name}_{unit}"] = float(attempt["x"][index])
            row[f"benchmark_error_{name}_{unit}"] = float(attempt["x"][index] - truth[name])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Encoding-safe explanatory comment.
# ---------------------------------------------------------------------------

def add_signed_error_stats(row: dict, errors: np.ndarray, prefix: str, unit: str) -> None:
    errors = np.asarray(errors, dtype=float)
    absolute = np.abs(errors)
    row[f"{prefix}_mean_error_{unit}"] = float(np.mean(errors))
    row[f"{prefix}_mae_{unit}"] = float(np.mean(absolute))
    row[f"{prefix}_rmse_{unit}"] = float(np.sqrt(np.mean(errors**2)))
    row[f"{prefix}_max_abs_{unit}"] = float(np.max(absolute))
    row[f"{prefix}_p95_abs_{unit}"] = float(np.percentile(absolute, 95.0))


def add_positive_stats(row: dict, values: np.ndarray, prefix: str, unit: str) -> None:
    values = np.asarray(values, dtype=float)
    row[f"{prefix}_mean_{unit}"] = float(np.mean(values))
    row[f"{prefix}_rms_{unit}"] = float(np.sqrt(np.mean(values**2)))
    row[f"{prefix}_max_{unit}"] = float(np.max(values))
    row[f"{prefix}_p95_{unit}"] = float(np.percentile(values, 95.0))


def summarize_results(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = table.groupby(["noise_case", "window_width_nm"], sort=False, dropna=False)
    for (case, width), group in grouped:
        successful = group[group["success"]].copy()
        row = {
            "noise_case": case, "window_width_nm": int(width),
            "window_min_nm": float(group.iloc[0]["window_min_nm"]),
            "window_max_nm": float(group.iloc[0]["window_max_nm"]),
            "runs": int(len(group)), "successful_runs": int(len(successful)),
            "fit_success_rate": float(len(successful) / len(group)),
            "spectral_point_count_mean": float(group["spectral_point_count"].mean()),
        }
        if not successful.empty:
            add_signed_error_stats(row, successful["air_error_nm"], "Air", "nm")
            for name in FILM_PARAMS:
                add_signed_error_stats(row, successful[f"error_{name}_nm"], name, "nm")
            add_signed_error_stats(row, successful["error_Angle_deg"], "Angle", "deg")
            add_positive_stats(row, successful["film_mae_nm"], "film_MAE", "nm")
            add_positive_stats(
                row, successful["normalized_spectral_rmse"],
                "normalized_spectral_RMSE", "dimensionless",
            )
            add_positive_stats(
                row, successful["rmse_reflectance"], "spectral_RMSE", "reflectance"
            )
            runtime = successful["fit_runtime_s"].to_numpy(dtype=float)
            row.update({
                "runtime_mean_s": float(np.mean(runtime)),
                "runtime_median_s": float(np.median(runtime)),
                "runtime_rmse_s": float(np.sqrt(np.mean(runtime**2))),
                "runtime_max_s": float(np.max(runtime)),
                "runtime_p95_s": float(np.percentile(runtime, 95.0)),
                "nfev_mean": float(successful["nfev"].mean()),
                "boundary_hit_rate": float((successful["boundary_hit_count"] > 0).mean()),
            })
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_case_order"] = result["noise_case"].map(CASE_ORDER)
    return result.sort_values(["_case_order", "window_width_nm"]).drop(
        columns="_case_order"
    ).reset_index(drop=True)


def paired_vs_full(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    successful = table[table["success"]].copy()
    full = successful[successful["window_width_nm"] == 320].copy()
    metrics = [
        "air_abs_error_nm", "film_mae_nm", "angle_abs_error_deg",
        "normalized_spectral_rmse", "fit_runtime_s",
    ]
    rows = []
    for width in sorted(successful["window_width_nm"].unique()):
        current = successful[successful["window_width_nm"] == width]
        merged = current.merge(
            full[["pair_key", *metrics]], on="pair_key", how="inner",
            suffixes=("", "_full320"),
        )
        for item in merged.itertuples(index=False):
            row = {
                "pair_key": item.pair_key, "noise_case": item.noise_case,
                "realization_index": int(item.realization_index),
                "window_width_nm": int(width),
            }
            for metric in metrics:
                value = float(getattr(item, metric))
                baseline = float(getattr(item, f"{metric}_full320"))
                row[metric] = value
                row[f"{metric}_full320"] = baseline
                row[f"delta_{metric}_vs_full320"] = value - baseline
            rows.append(row)
    detail = pd.DataFrame(rows)
    summaries = []
    if not detail.empty:
        for (case, width), group in detail.groupby(
            ["noise_case", "window_width_nm"], sort=False
        ):
            row = {
                "noise_case": case, "window_width_nm": int(width),
                "paired_runs": int(len(group)),
            }
            for metric in metrics:
                delta = group[f"delta_{metric}_vs_full320"].to_numpy(dtype=float)
                row[f"delta_{metric}_mean"] = float(np.mean(delta))
                row[f"delta_{metric}_median"] = float(np.median(delta))
                row[f"{metric}_improvement_rate"] = float(np.mean(delta < 0.0))
            summaries.append(row)
    return detail, pd.DataFrame(summaries)


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No data available._"
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
            )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Encoding-safe explanatory comment.
# ---------------------------------------------------------------------------

def save_parameter_plot(output_dir: Path, summary: pd.DataFrame) -> Path:
    panels = [
        ("Air_mae_nm", "Air cavity MAE", "nm"),
        ("HSQ_mae_nm", "HSQ thickness MAE", "nm"),
        ("PSS_mae_nm", "PSS thickness MAE", "nm"),
        ("SOC_mae_nm", "SOC thickness MAE", "nm"),
        ("TiO2_mae_nm", "TiO2 thickness MAE", "nm"),
        ("film_MAE_mean_nm", "Mean film MAE", "nm"),
        ("Angle_mae_deg", "Reflector angle MAE", "deg"),
        ("normalized_spectral_RMSE_mean_dimensionless", "Normalized spectral RMSE", "dimensionless"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    colors = {"clean": "#4472C4", "absolute_accuracy_low": "#ED7D31"}
    for ax, (column, title, unit) in zip(axes.flat, panels):
        for case in SELECTED_CASES:
            group = summary[summary["noise_case"] == case].sort_values("window_width_nm")
            if group.empty or column not in group:
                continue
            ax.plot(
                group["window_width_nm"], group[column], marker="o", linewidth=2.2,
                markersize=6, label=case, color=colors[case],
            )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Window width (nm)", fontsize=12)
        ax.set_ylabel(unit, fontsize=12)
        ax.set_xticks([20, 40, 60, 80, 100, 130, 160, 200, 240, 280, 320])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
    axes[0, 0].legend(fontsize=10)
    path = output_dir / "window_width_parameter_errors.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_runtime_plot(output_dir: Path, summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = {"clean": "#4472C4", "absolute_accuracy_low": "#ED7D31"}
    for case in SELECTED_CASES:
        group = summary[summary["noise_case"] == case].sort_values("window_width_nm")
        if group.empty:
            continue
        ax.plot(
            group["window_width_nm"], group["runtime_mean_s"], marker="o",
            linewidth=2.4, markersize=7, label=f"{case}: mean", color=colors[case],
        )
        ax.plot(
            group["window_width_nm"], group["runtime_p95_s"], marker="s",
            linewidth=1.5, markersize=5, linestyle="--", label=f"{case}: p95",
            color=colors[case], alpha=0.8,
        )
    ax.set_title("Window width vs inversion runtime", fontsize=16)
    ax.set_xlabel("Window width (nm)", fontsize=14)
    ax.set_ylabel("Runtime (s)", fontsize=14)
    ax.set_xticks([20, 40, 60, 80, 100, 130, 160, 200, 240, 280, 320])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    path = output_dir / "window_width_runtime.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_report(summary: pd.DataFrame, config: FitConfig, input_count: int) -> str:
    compact_columns = [
        "noise_case", "window_width_nm", "runs", "successful_runs", "fit_success_rate",
        "Air_mae_nm", "film_MAE_mean_nm", "Angle_mae_deg",
        "normalized_spectral_RMSE_mean_dimensionless", "runtime_mean_s",
    ]
    compact = summary[[column for column in compact_columns if column in summary]].copy()
    conclusions = []
    for case in SELECTED_CASES:
        case_rows = summary[summary["noise_case"] == case]
        if case_rows.empty:
            continue
        full = case_rows[case_rows["window_width_nm"] == 320]
        best_air = case_rows.loc[case_rows["Air_mae_nm"].idxmin()]
        best_film = case_rows.loc[case_rows["film_MAE_mean_nm"].idxmin()]
        best_angle = case_rows.loc[case_rows["Angle_mae_deg"].idxmin()]
        statement = (
            f"- `{case}`\uff1aAir MAE \u6700\u5c0f\u7a97\u53e3\u4e3a {int(best_air['window_width_nm'])} nm "
            f"({best_air['Air_mae_nm']:.6g} nm)\uff0cfilm_MAE \u6700\u5c0f\u7a97\u53e3\u4e3a "
            f"{int(best_film['window_width_nm'])} nm ({best_film['film_MAE_mean_nm']:.6g} nm)\uff0c"
            f"\u89d2\u5ea6 MAE \u6700\u5c0f\u7a97\u53e3\u4e3a {int(best_angle['window_width_nm'])} nm "
            f"({best_angle['Angle_mae_deg']:.6g} deg)\u3002"
        )
        if not full.empty:
            full_row = full.iloc[0]
            better = case_rows[
                (case_rows["window_width_nm"] < 320)
                & (case_rows["Air_mae_nm"] < full_row["Air_mae_nm"])
                & (case_rows["film_MAE_mean_nm"] < full_row["film_MAE_mean_nm"])
            ]
            if better.empty:
                statement += "\u6ca1\u6709\u7a84\u7a97\u53e3\u540c\u65f6\u4f18\u4e8e\u5b8c\u6574 320 nm \u6ce2\u6bb5\u7684 Air MAE \u4e0e film_MAE\u3002"
            else:
                widths = ", ".join(str(int(value)) for value in better["window_width_nm"])
                statement += f"\u540c\u65f6\u6539\u5584 Air MAE \u4e0e film_MAE \u7684\u7a84\u7a97\u53e3\u4e3a {widths} nm\u3002"
        conclusions.append(statement)
    noisy = summary[summary["noise_case"] == "absolute_accuracy_low"]
    if not noisy.empty:
        narrow = noisy[noisy["window_width_nm"] == 20].iloc[0]
        full_row = noisy[noisy["window_width_nm"] == 320].iloc[0]
        conclusions.append(
            f"- `absolute_accuracy_low` \u4e2d\uff0c20 nm \u7a97\u53e3\u7684\u6807\u51c6\u5316\u5149\u8c31 RMSE \u4e3a "
            f"{narrow['normalized_spectral_RMSE_mean_dimensionless']:.6g}\uff0c\u4f4e\u4e8e 320 nm \u7684 "
            f"{full_row['normalized_spectral_RMSE_mean_dimensionless']:.6g}\uff1b\u4f46 Air MAE \u7531 "
            f"{full_row['Air_mae_nm']:.6g} nm \u6076\u5316\u4e3a {narrow['Air_mae_nm']:.6g} nm\uff0c"
            f"film_MAE \u7531 {full_row['film_MAE_mean_nm']:.6g} nm \u6076\u5316\u4e3a "
            f"{narrow['film_MAE_mean_nm']:.6g} nm\u3002\u8fd9\u8bf4\u660e\u5c40\u90e8\u5149\u8c31\u62df\u5408\u66f4\u597d\u4e0d\u7b49\u4e8e\u53c2\u6570\u66f4\u51c6\u3002"
        )
        conclusions.append(
            f"- \u8be5\u566a\u58f0\u6848\u4f8b\u5404\u7a97\u53e3\u7684\u8fb9\u754c\u547d\u4e2d\u7387\u4e3a "
            f"{100.0 * noisy['boundary_hit_rate'].min():.0f}%--{100.0 * noisy['boundary_hit_rate'].max():.0f}%\uff0c"
            "\u663e\u793a\u7edd\u5bf9\u6ce2\u957f\u51c6\u786e\u5ea6\u6b8b\u5dee\u5728\u5f53\u524d\u516d\u53c2\u6570\u6a21\u578b\u4e2d\u88ab\u89d2\u5ea6\u4e0e\u819c\u539a\u8fb9\u754c\u8865\u507f\u3002"
        )

    if not summary[summary["noise_case"] == "clean"].empty and int(
        summary[summary["noise_case"] == "clean"]["runs"].max()
    ) == 1:
        conclusions.append(
            "- `clean` \u53ea\u6709\u4e00\u4e2a realization\uff1b\u8be5\u8d8b\u52bf\u662f\u786e\u5b9a\u6027\u914d\u5bf9\u7ed3\u679c\uff0c\u4e0d\u80fd\u4f30\u8ba1\u968f\u673a\u5206\u5e03\u3002"
        )
    conclusions.append(
        "- \u7f29\u5c0f\u6ce2\u6bb5\u53ea\u51cf\u5c11\u7ea6\u675f\u4e0e\u8ba1\u7b97\u70b9\u6570\uff0c\u4e0d\u589e\u52a0\u7269\u7406\u4fe1\u606f\u3002\u53ea\u6709\u914d\u5bf9\u8bef\u5dee\u5728\u591a\u4e2a realization "
        "\u4e0a\u7a33\u5b9a\u4e0b\u964d\uff0c\u4e14\u6210\u529f\u7387\u4e0e\u5176\u4ed6\u53c2\u6570\u4e0d\u6076\u5316\uff0c\u624d\u80fd\u8ba4\u5b9a\u7a84\u6ce2\u6bb5\u771f\u6b63\u63d0\u9ad8\u51c6\u786e\u5ea6\u3002"
    )
    lines = [
        "# V9 \u56fa\u5b9a\u4e2d\u5fc3\u62df\u5408\u6ce2\u6bb5\u5bbd\u5ea6\u5bf9\u6bd4", "", "## \u6570\u636e\u4e0e\u516c\u5e73\u6027\u53e3\u5f84", "",
        f"- \u65e2\u6709 NPZ realization \u6570\uff1a{input_count}\uff1b\u6bcf\u4e2a realization \u4f7f\u7528\u76f8\u540c\u7684\u516d\u4e2a\u7a97\u53e3\u3002",
        "- \u4ec5\u5904\u7406 `clean` \u548c `absolute_accuracy_low`\uff0c\u4e0d\u751f\u6210\u6216\u4fee\u6539\u4efb\u4f55\u539f\u59cb\u6570\u636e\u3002",
        "- \u62df\u5408\u8f74\u4ec5\u4e3a `estimated_calibrated_wavelengths`\uff1b\u62df\u5408\u8def\u5f84\u4e0d\u8bfb\u53d6 `physical_wavelengths`\u3002",
        "- \u6240\u6709\u7a97\u53e3\u4f7f\u7528\u76f8\u540c\u516d\u53c2\u6570 TMM\u3001\u8fb9\u754c\u3001\u5dee\u5206\u8fdb\u5316\u548c\u6709\u754c\u6700\u5c0f\u4e8c\u4e58\u8bbe\u7f6e\u3002",
        "- \u540c\u4e00 realization \u7684\u7a97\u53e3\u5171\u4eab optimizer seed \u548c Latin-hypercube \u521d\u59cb\u603b\u4f53\u3002",
        "- \u771f\u503c\u53ea\u5728\u6bcf\u6b21\u62df\u5408\u7ed3\u675f\u540e\u8bfb\u53d6\uff0c\u4e0d\u53c2\u4e0e\u524d\u5411\u6a21\u578b\u3001\u8d77\u70b9\u3001\u6b8b\u5dee\u6216\u6392\u5e8f\u3002",
        "- \u6807\u51c6\u5316\u5149\u8c31 RMSE = `RMSE(fit-measured) / std(measured)`\uff1b\u539f\u59cb cost \u4ec5\u4f9b\u5ba1\u8ba1\u3002",
        f"- \u4f18\u5316\u8bbe\u7f6e\uff1aglobal_popsize={config.global_popsize}, global_maxiter={config.global_maxiter}, "
        f"multistarts={config.multistarts}, max_nfev={config.max_nfev}, loss={config.loss}, "
        f"stride={config.stride}, global_stride={config.global_stride}, seed={config.random_seed}\u3002",
        "", "## \u4e3b\u8981\u6c47\u603b", "", dataframe_to_markdown(compact), "",
        "## \u914d\u5bf9\u7ed3\u8bba", "", *conclusions, "", "## \u89e3\u91ca\u9650\u5236", "",
        "- `absolute_accuracy_low` \u7684\u9690\u85cf\u7269\u7406\u8f74\u6ca1\u6709\u7528\u4e8e\u7ea0\u6b63\u53cd\u6f14\u53ef\u89c1\u8f74\u3002",
        "- \u63a5\u8fd1\u96f6\u5ea6\u65f6\u89d2\u5ea6\u4e00\u9636\u7075\u654f\u5ea6\u5f88\u5f31\uff0c\u89d2\u5ea6\u3001Air \u548c\u819c\u539a\u4f1a\u4e92\u76f8\u8865\u507f\uff1b\u4f4e\u5149\u8c31 RMSE \u4e0d\u7b49\u4e8e\u53c2\u6570\u552f\u4e00\u3002",
        "- \u5404\u53c2\u6570\u7684 mean error\u3001MAE\u3001RMSE\u3001\u6700\u5927\u7edd\u5bf9\u8bef\u5dee\u548c 95% \u7edd\u5bf9\u8bef\u5dee\u89c1 `window_summary.csv`\u3002",
        "- \u9010 realization \u914d\u5bf9\u5dee\u503c\u89c1 `paired_vs_full320.csv` \u4e0e\u5176\u6c47\u603b\u6587\u4ef6\u3002",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V9 inversion bandwidths on existing NPZ data.")
    parser.add_argument(
        "--input-dir", type=Path,
        default=REPO_ROOT / "work" / "04_results_and_datasets" / "static_stackrt_v9_20260826_111637",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    # parser.add_argument("--window-widths", nargs="*", type=int, default=[20, 40, 60, 80, 100, 130])
    parser.add_argument(
    "--window-widths",
    nargs="*",
    type=int,
    default=[20, 40, 60, 80, 100, 130, 160, 200, 240, 280, 320],
    )
    parser.add_argument("--max-realizations-per-case", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--global-stride", type=int, default=2)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--local-gtol", type=float, default=1.0e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=20260810)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    requested = list(dict.fromkeys(int(value) for value in args.window_widths))
    selected_windows = [window for window in WINDOWS if window["width_nm"] in requested]
    if {window["width_nm"] for window in selected_windows} != set(requested):
        raise ValueError(f"Unsupported window widths: {requested}")

    max_realizations = args.max_realizations_per_case
    if args.smoke:
        max_realizations = 1
        selected_windows = selected_windows[:1]
        args.global_popsize = min(args.global_popsize, 4)
        args.global_maxiter = min(args.global_maxiter, 3)
        args.multistarts = min(args.multistarts, 2)
        args.max_nfev = min(args.max_nfev, 80)
        args.workers = 1
    identities = discover_inputs(input_dir, max_realizations)
    if args.smoke:
        identities = [item for item in identities if item["noise_case"] == "clean"][:1]

    config = FitConfig(
        input_dir=str(input_dir), wavelength_min_nm=450.0, wavelength_max_nm=580.0,
        stride=max(1, int(args.stride)), global_stride=max(1, int(args.global_stride)),
        global_popsize=max(1, int(args.global_popsize)),
        global_maxiter=max(1, int(args.global_maxiter)),
        multistarts=max(1, int(args.multistarts)), max_nfev=max(1, int(args.max_nfev)),
        local_gtol=float(args.local_gtol), workers=max(1, int(args.workers)),
        random_seed=int(args.random_seed), loss=args.loss,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "smoke_tmm_fit_bandwidth_comparison_v9" if args.smoke else VERSION
    output_dir = args.output_dir.resolve() if args.output_dir else OUTPUT_ROOT / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    tasks = []
    for identity in identities:
        for window in selected_windows:
            tasks.append((len(tasks), identity, window, config))
    rows, attempts, failures, sampling_rows, global_rows = [], [], [], [], []
    print(f"Selected {len(identities)} realizations, {len(selected_windows)} windows, {len(tasks)} fits.", flush=True)

    if config.workers == 1:
        outcomes, executor = map(process_band_task, tasks), None
    else:
        executor = ThreadPoolExecutor(max_workers=config.workers)
        outcomes = executor.map(process_band_task, tasks)
    try:
        for completed, outcome in enumerate(outcomes, start=1):
            identity, window = outcome["identity"], outcome["window"]
            if not outcome["ok"]:
                failures.append({
                    "pair_key": identity["pair_key"], "input_npz": str(identity["path"]),
                    "window_width_nm": window["width_nm"], "error": outcome["error"],
                })
                print(f"[{completed}/{len(tasks)}] ERROR {identity['pair_key']} {window['width_nm']}nm: {outcome['error']}", flush=True)
                continue

            # Encoding-safe explanatory comment.
            truth, _noise_audit = load_evaluation_truth(Path(identity["path"]))
            row = result_row(outcome, truth)
            rows.append(row)
            attempts.extend(attempt_rows(outcome, truth))
            sampling_rows.append({
                "pair_key": identity["pair_key"], "window_width_nm": window["width_nm"],
                **outcome["sampling_audit"],
            })
            global_rows.append({
                "pair_key": identity["pair_key"], "window_width_nm": window["width_nm"],
                "optimizer_seed": outcome["optimizer_seed"],
                **outcome["fit"].get("global_summary", {}),
            })
            if row["success"]:
                print(
                    f"[{completed}/{len(tasks)}] {identity['pair_key']} {window['width_nm']}nm "
                    f"Air={row['air_error_nm']:.6g}nm film_MAE={row['film_mae_nm']:.6g}nm "
                    f"angle={row['error_Angle_deg']:.6g}deg NRMSE={row['normalized_spectral_rmse']:.6g} "
                    f"time={row['fit_runtime_s']:.2f}s", flush=True,
                )
            else:
                print(f"[{completed}/{len(tasks)}] NO CONVERGED RESULT {identity['pair_key']} {window['width_nm']}nm", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No fit result produced. Failures: {failures}")
    table["_case_order"] = table["noise_case"].map(CASE_ORDER)
    table = table.sort_values(["_case_order", "realization_index", "window_width_nm"]).drop(columns="_case_order")
    table.to_csv(output_dir / "fit_details.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(attempts).to_csv(output_dir / "multistart_details.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary = summarize_results(table)
    summary.to_csv(output_dir / "window_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    paired_detail, paired_summary = paired_vs_full(table)
    paired_detail.to_csv(output_dir / "paired_vs_full320.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    paired_summary.to_csv(output_dir / "paired_vs_full320_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(sampling_rows).to_csv(output_dir / "sampling_audit.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "global_search_summary.jsonl").open("w", encoding="utf-8") as handle:
        for item in global_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    parameter_plot = save_parameter_plot(output_dir, summary)
    runtime_plot = save_runtime_plot(output_dir, summary)
    (output_dir / "analysis_report.md").write_text(
        build_report(summary, config, len(identities)), encoding="utf-8-sig"
    )
    manifest = {
        "version": VERSION, "source_dataset": str(input_dir),
        "source_files_modified": False, "selected_cases": list(SELECTED_CASES),
        "input_realizations": len(identities), "windows": selected_windows,
        "fit_count_expected": len(tasks), "fit_rows": len(table),
        "successful_fits": int(table["success"].sum()), "config": asdict(config),
        "bounds": BOUNDS, "fit_wavelength_axis": "estimated_calibrated_wavelengths",
        "hidden_physical_wavelengths_used_for_fit": False,
        "normalized_rmse_definition": "RMSE(fit-measured) / std(measured within each window)",
        "paired_seed_policy": "same optimizer seed and Latin-hypercube population across windows for each realization",
        "truth_policy": "loaded only after each fit completes; evaluation only",
        "outputs": [
            "fit_details.csv", "window_summary.csv", "paired_vs_full320.csv",
            "paired_vs_full320_summary.csv", "multistart_details.csv", "sampling_audit.csv",
            parameter_plot.name, runtime_plot.name, "analysis_report.md",
        ],
        "failures": failures,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"OUTPUT_DIR={output_dir}", flush=True)
    if failures:
        raise RuntimeError(f"There were {len(failures)} task failures; see failures.csv")


if __name__ == "__main__":
    main()
