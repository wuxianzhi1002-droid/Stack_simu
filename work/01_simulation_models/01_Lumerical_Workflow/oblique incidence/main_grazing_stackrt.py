from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from grazing_config import (
    CONFIG,
    DATA_DIR,
    WORKFLOW_DIR,
    build_theta_axis,
    build_wavelength_axis,
    config_json,
    ensure_output_dirs,
    timestamp,
)


LUMERICAL_PATH = Path(r"D:\Program Files\Lumerical\v241\api\python")
if LUMERICAL_PATH.exists():
    if str(LUMERICAL_PATH) not in sys.path:
        sys.path.append(str(LUMERICAL_PATH))
    os.environ["PATH"] += os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    lumapi = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(values: Iterable[Any], **_: Any) -> Iterable[Any]:
        return values


SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class StackModel:
    """记录从现有工作流复用的膜层模型。"""

    source_module: str
    source_class: str
    model_key: str
    layer_names: List[str]
    layers: List[Tuple[Any, float]]
    n_matrix: np.ndarray
    thicknesses_m: np.ndarray
    air_idx: int
    original_layer_names: List[str] | None = None


def _prepare_source_config(module: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """把现有 main_angle/main_cavity 的配置补齐成当前掠入射网格。"""
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"{module.__name__} does not define CONFIG")
    source_config = copy.deepcopy(module.CONFIG)
    source_config["WAVELENGTH_START"] = float(config["wavelength_min_um"])
    source_config["WAVELENGTH_STOP"] = float(config["wavelength_max_um"])
    source_config["SPECTRAL_RESOLUTION_NM"] = float(config["wavelength_step_nm"])
    source_config["ANGLE_START_DEG"] = float(config["theta_min_deg"])
    source_config["ANGLE_STOP_DEG"] = float(config["theta_max_deg"])
    source_config["ANGLE_STEP_DEG"] = float(config["theta_step_deg"])
    source_config["ANGLE_DEG"] = float(config["theta_min_deg"])
    source_config["MODEL_TYPE"] = "PSS_TiO2"
    return source_config


def _layer_name(mat: Any) -> str:
    """把数值折射率层和字符串材料层统一转成可记录的层名。"""
    if isinstance(mat, str):
        return mat
    return f"n={mat}"


def load_existing_stack_model(
    wavelengths_um: np.ndarray,
    freqs_hz: np.ndarray,
    config: Dict[str, Any] | None = None,
) -> StackModel:
    """从已有工作流导入 PSS_TIO2_MODEL 和 _get_n_matrix()。

    如果导入失败，会抛出带有候选模块和失败原因的 RuntimeError，便于手动指定模型来源。
    """
    cfg = CONFIG if config is None else config
    if str(WORKFLOW_DIR) not in sys.path:
        sys.path.insert(0, str(WORKFLOW_DIR))

    model_key = str(cfg.get("model_key", "PSS_TIO2_MODEL"))
    errors: List[str] = []
    class_candidates = ("AngleSimulator", "CavitySimulator", "DynamicSimulator", "LumericalSimulator")

    for module_name in cfg.get("model_source_modules", []):
        try:
            module = importlib.import_module(module_name)
            source_config = _prepare_source_config(module, cfg)
            if model_key not in source_config:
                errors.append(f"{module_name}: CONFIG does not contain {model_key}")
                continue

            for class_name in class_candidates:
                simulator_cls = getattr(module, class_name, None)
                if simulator_cls is None or not hasattr(simulator_cls, "_get_n_matrix"):
                    continue
                simulator = simulator_cls(source_config)
                # 强制使用当前脚本构造的精确轴，避免不同 linspace/arange 细节造成维度偏差。
                simulator.wavelengths = wavelengths_um
                simulator.freqs = freqs_hz
                n_matrix, thicknesses_m, air_idx = simulator._get_n_matrix(model_key)
                layers = list(source_config[model_key]["LAYERS"])
                layer_names = [_layer_name(mat) for mat, _ in layers]
                return StackModel(
                    source_module=module_name,
                    source_class=class_name,
                    model_key=model_key,
                    layer_names=layer_names,
                    layers=layers,
                    n_matrix=np.asarray(n_matrix, dtype=complex),
                    thicknesses_m=np.asarray(thicknesses_m, dtype=float),
                    air_idx=int(air_idx),
                    original_layer_names=layer_names,
                )

            errors.append(f"{module_name}: no simulator class with _get_n_matrix() was found")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    detail = "\n".join(f"- {item}" for item in errors) or "- no module candidates were configured"
    raise RuntimeError(
        "无法从现有工作流导入膜层模型和 _get_n_matrix()。请检查 main_angle.py/main_cavity.py，"
        "或在 grazing_config.py 中手动指定 model_source_modules/model_key。\n"
        f"尝试结果：\n{detail}"
    )


def prepare_grazing_incidence_stack(model: StackModel, config: Dict[str, Any] | None = None) -> StackModel:
    """把旧干涉模型裁剪成从 Air 入射到晶圆膜层的 StackRT 层序。

    旧工作流的 PSS_TIO2_MODEL 常包含 RefReflector/Air/HSQ/...，其中 RefReflector 用于
    外腔干涉。掠入射三角测量要评估的是空气侧看向 HSQ 顶面的膜层反射，因此默认从
    Air 层开始裁剪，并把首层 Air 当作半无限入射介质处理。
    """
    cfg = CONFIG if config is None else config
    if not bool(cfg.get("trim_to_air_interface", True)):
        return model

    air_candidates = [idx for idx, name in enumerate(model.layer_names) if name.lower() == "air"]
    if not air_candidates:
        return model

    start_idx = air_candidates[0]
    if start_idx == 0:
        thicknesses = model.thicknesses_m.copy()
        thicknesses[0] = 0.0
        return StackModel(
            source_module=model.source_module,
            source_class=model.source_class,
            model_key=model.model_key,
            layer_names=list(model.layer_names),
            layers=list(model.layers),
            n_matrix=model.n_matrix.copy(),
            thicknesses_m=thicknesses,
            air_idx=0,
            original_layer_names=model.original_layer_names or list(model.layer_names),
        )

    layer_names = list(model.layer_names[start_idx:])
    layers = list(model.layers[start_idx:])
    n_matrix = model.n_matrix[start_idx:, :].copy()
    thicknesses = model.thicknesses_m[start_idx:].copy()
    thicknesses[0] = 0.0
    return StackModel(
        source_module=model.source_module,
        source_class=model.source_class,
        model_key=f"{model.model_key}[Air-interface]",
        layer_names=layer_names,
        layers=layers,
        n_matrix=n_matrix,
        thicknesses_m=thicknesses,
        air_idx=0,
        original_layer_names=model.original_layer_names or list(model.layer_names),
    )


def perturbable_mask(
    layer_names: Sequence[str],
    thicknesses_m: np.ndarray,
    config: Dict[str, Any] | None = None,
) -> np.ndarray:
    """判断哪些有限厚度膜层参与 Monte Carlo 扰动。"""
    cfg = CONFIG if config is None else config
    excluded = [str(item).lower() for item in cfg.get("exclude_perturb_layers_keywords", [])]
    mask = np.ones(len(layer_names), dtype=bool)
    for idx, name in enumerate(layer_names):
        lowered = name.lower()
        if idx == 0 or idx == len(layer_names) - 1:
            mask[idx] = False
        if thicknesses_m[idx] <= 0.0:
            mask[idx] = False
        if any(keyword in lowered for keyword in excluded):
            mask[idx] = False
    return mask


def build_monte_carlo_thicknesses(
    nominal_m: np.ndarray,
    mask: np.ndarray,
    config: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成膜厚扰动，返回厚度矩阵和 nm 扰动量。"""
    cfg = CONFIG if config is None else config
    n_mc = int(cfg["num_monte_carlo"])
    rng = np.random.default_rng(int(cfg.get("random_seed", 20260616)))
    perturb_nm = np.zeros((n_mc, nominal_m.size), dtype=np.float32)
    uncertainty = float(cfg["film_uncertainty_nm"])
    mode = str(cfg.get("film_uncertainty_mode", "uniform")).lower()

    if n_mc == 0:
        return np.zeros((0, nominal_m.size), dtype=np.float64), perturb_nm

    if mode == "normal":
        values = rng.normal(0.0, uncertainty / 3.0, size=(n_mc, int(np.sum(mask))))
        values = np.clip(values, -uncertainty, uncertainty)
    else:
        values = rng.uniform(-uncertainty, uncertainty, size=(n_mc, int(np.sum(mask))))

    perturb_nm[:, mask] = values.astype(np.float32)
    thicknesses_mc = nominal_m[None, :] + perturb_nm.astype(np.float64) * 1e-9
    thicknesses_mc = np.maximum(thicknesses_mc, 0.0)
    return thicknesses_mc, perturb_nm


def stackrt_keys(result: Any) -> List[str]:
    """兼容 lumapi 返回的 dict-like 结果。"""
    if hasattr(result, "keys"):
        return [str(key) for key in result.keys()]
    return []


def _candidate_keys(pol: str, complex_amplitude: bool) -> List[str]:
    prefix = pol.lower()
    if complex_amplitude:
        return [prefix, f"r{prefix}", f"r_{prefix}", f"R{prefix}_amp", f"{prefix}_complex"]
    return [f"R{prefix}", f"R_{prefix}", f"{prefix.upper()}Reflectance", "R"]


def extract_array(result: Any, pol: str, complex_amplitude: bool, expected_size: int) -> np.ndarray | None:
    """从 StackRT 结果中提取指定偏振的强度或复振幅。"""
    keys = stackrt_keys(result)
    key_map = {key.lower(): key for key in keys}
    for candidate in _candidate_keys(pol, complex_amplitude):
        real_key = key_map.get(candidate.lower())
        if real_key is None:
            continue
        values = np.asarray(result[real_key]).reshape(-1)
        if values.size != expected_size:
            values = np.squeeze(values)
            values = np.asarray(values).reshape(-1)
        if values.size == expected_size:
            return values.astype(complex if complex_amplitude else float, copy=False)
    return None


def run_stackrt(config: Dict[str, Any] | None = None) -> Path:
    """运行掠入射 StackRT 扫描并保存 npz 数据。"""
    cfg = CONFIG if config is None else config
    if lumapi is None:
        raise RuntimeError(
            "lumapi is not available. 请检查 Lumerical 安装路径 "
            r"(例如 D:\Program Files\Lumerical\v241\api\python) 和当前 Python 环境。"
        )

    ensure_output_dirs()
    wavelengths_um = build_wavelength_axis(cfg)
    wavelengths_m = wavelengths_um * 1e-6
    freqs_hz = SPEED_OF_LIGHT / wavelengths_m
    theta_axis_deg = build_theta_axis(cfg)
    theta_axis_rad = np.deg2rad(theta_axis_deg)

    imported_model = load_existing_stack_model(wavelengths_um, freqs_hz, cfg)
    model = prepare_grazing_incidence_stack(imported_model, cfg)
    mask = perturbable_mask(model.layer_names, model.thicknesses_m, cfg)
    thicknesses_mc_m, perturbation_nm = build_monte_carlo_thicknesses(model.thicknesses_m, mask, cfg)

    n_theta = theta_axis_deg.size
    n_lambda = wavelengths_um.size
    n_mc = thicknesses_mc_m.shape[0]
    r_nominal: Dict[str, np.ndarray] = {}
    r_mc: Dict[str, np.ndarray] = {}

    R_nominal_p = np.full((n_theta, n_lambda), np.nan, dtype=np.float32)
    R_nominal_s = np.full_like(R_nominal_p, np.nan)
    R_mc_p = np.full((n_mc, n_theta, n_lambda), np.nan, dtype=np.float32)
    R_mc_s = np.full_like(R_mc_p, np.nan)
    first_keys: List[str] = []

    print("=== Grazing StackRT Simulation ===")
    print(f"Model source: {model.source_module}.{model.source_class}.{model.model_key}")
    print(f"Layer stack used: {model.layer_names}")
    print(f"Wavelength points: {n_lambda}, theta points: {n_theta}, Monte Carlo: {n_mc}")
    print(f"Perturbed layers: {[name for name, enabled in zip(model.layer_names, mask) if enabled]}")

    fdtd = lumapi.FDTD(hide=True)
    start_time = time.time()
    try:
        for i_theta, theta_deg in enumerate(tqdm(theta_axis_deg, desc="nominal theta scan")):
            result = fdtd.stackrt(model.n_matrix, model.thicknesses_m, freqs_hz, float(theta_deg))
            if not first_keys:
                first_keys = stackrt_keys(result)
            rp = extract_array(result, "p", False, n_lambda)
            rs = extract_array(result, "s", False, n_lambda)
            if rp is None or rs is None:
                raise KeyError(f"StackRT result lacks Rp/Rs. Available keys: {first_keys}")
            R_nominal_p[i_theta, :] = np.real(rp).astype(np.float32)
            R_nominal_s[i_theta, :] = np.real(rs).astype(np.float32)

            r_p = extract_array(result, "p", True, n_lambda)
            r_s = extract_array(result, "s", True, n_lambda)
            if r_p is not None:
                r_nominal.setdefault("p", np.full((n_theta, n_lambda), np.nan + 0j, dtype=np.complex64))
                r_nominal["p"][i_theta, :] = r_p.astype(np.complex64)
            if r_s is not None:
                r_nominal.setdefault("s", np.full((n_theta, n_lambda), np.nan + 0j, dtype=np.complex64))
                r_nominal["s"][i_theta, :] = r_s.astype(np.complex64)

        for i_mc in tqdm(range(n_mc), desc="film Monte Carlo"):
            thicknesses = thicknesses_mc_m[i_mc, :]
            for i_theta, theta_deg in enumerate(theta_axis_deg):
                result = fdtd.stackrt(model.n_matrix, thicknesses, freqs_hz, float(theta_deg))
                rp = extract_array(result, "p", False, n_lambda)
                rs = extract_array(result, "s", False, n_lambda)
                if rp is None or rs is None:
                    raise KeyError(f"StackRT result lacks Rp/Rs. Available keys: {stackrt_keys(result)}")
                R_mc_p[i_mc, i_theta, :] = np.real(rp).astype(np.float32)
                R_mc_s[i_mc, i_theta, :] = np.real(rs).astype(np.float32)

                r_p = extract_array(result, "p", True, n_lambda)
                r_s = extract_array(result, "s", True, n_lambda)
                if r_p is not None:
                    r_mc.setdefault(
                        "p",
                        np.full((n_mc, n_theta, n_lambda), np.nan + 0j, dtype=np.complex64),
                    )
                    r_mc["p"][i_mc, i_theta, :] = r_p.astype(np.complex64)
                if r_s is not None:
                    r_mc.setdefault(
                        "s",
                        np.full((n_mc, n_theta, n_lambda), np.nan + 0j, dtype=np.complex64),
                    )
                    r_mc["s"][i_mc, i_theta, :] = r_s.astype(np.complex64)
    finally:
        fdtd.close()

    save_path = DATA_DIR / f"grazing_stackrt_{timestamp()}.npz"
    payload: Dict[str, Any] = {
        "wavelengths_m": wavelengths_m.astype(np.float64),
        "wavelengths_um": wavelengths_um.astype(np.float64),
        "freqs_Hz": freqs_hz.astype(np.float64),
        "theta_axis_deg": theta_axis_deg.astype(np.float64),
        "theta_axis_rad": theta_axis_rad.astype(np.float64),
        "polarizations": np.asarray(cfg["polarizations"], dtype=str),
        "layer_names": np.asarray(model.layer_names, dtype=str),
        "original_layer_names": np.asarray(model.original_layer_names or model.layer_names, dtype=str),
        "thicknesses_nominal_m": model.thicknesses_m.astype(np.float64),
        "thicknesses_mc_m": thicknesses_mc_m.astype(np.float64),
        "perturbation_nm": perturbation_nm.astype(np.float32),
        "perturbable_layer_mask": mask.astype(bool),
        "R_nominal_p": R_nominal_p,
        "R_nominal_s": R_nominal_s,
        "R_mc_p": R_mc_p,
        "R_mc_s": R_mc_s,
        "n_matrix": model.n_matrix.astype(np.complex64),
        "air_idx": np.asarray(model.air_idx, dtype=np.int32),
        "stackrt_result_keys": np.asarray(first_keys, dtype=str),
        "model_source": np.asarray(f"{model.source_module}.{model.source_class}.{model.model_key}"),
        "config_json": np.asarray(config_json(cfg)),
    }
    if "p" in r_nominal:
        payload["r_nominal_p"] = r_nominal["p"]
    if "s" in r_nominal:
        payload["r_nominal_s"] = r_nominal["s"]
    if "p" in r_mc:
        payload["r_mc_p"] = r_mc["p"]
    if "s" in r_mc:
        payload["r_mc_s"] = r_mc["s"]

    np.savez_compressed(save_path, **payload)
    elapsed = time.time() - start_time
    print(f"Saved StackRT data: {save_path}")
    print(f"Elapsed: {elapsed:.1f} s")
    if "p" not in r_nominal or "s" not in r_nominal:
        print("Note: lumapi stackrt did not return complex phase; solve_grazing_asd.py will use internal TMM.")
    return save_path


def main() -> None:
    """命令行入口。"""
    try:
        output = run_stackrt(CONFIG)
    except Exception as exc:
        print(f"StackRT simulation failed: {exc}")
        raise
    print(f"Output npz: {output}")


if __name__ == "__main__":
    main()
