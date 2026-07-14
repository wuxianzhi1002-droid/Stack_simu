from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


# theta 是相对于晶圆表面法线的入射角；60-85 deg 属于掠入射区间。
CONFIG: Dict[str, Any] = {
    "wavelength_min_um": 0.2,
    "wavelength_max_um": 0.6,
    "wavelength_step_nm": 0.1,
    "theta_min_deg": 60.0,
    "theta_max_deg": 85.0,
    "theta_step_deg": 0.25,
    "polarizations": ["p", "s"],
    "height_scan_nm": {
        "min": -100.0,
        "max": 100.0,
        "step": 1.0,
    },
    "multipass_list": [1, 2, 4, 6],
    "grating_pitch_um": 20.0,
    "imaging_magnification": 1.0,
    "source_type": "flat",
    "detector_noise_std": 0.002,
    "shot_noise_enable": True,
    "film_uncertainty_nm": 10.0,
    "film_uncertainty_mode": "uniform",
    "num_monte_carlo": 100,
    "exclude_perturb_layers_keywords": [
        "Air",
        "air",
        "Vacuum",
        "vacuum",
        "Substrate",
        "Si substrate",
    ],
    "mirror_reflectivity": 0.98,
    "extra_mirror_count_per_wafer_pass": 2,
    "theta_error_scan_deg": [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
    # 以下是脚本内部使用的可调参数，不影响 prompt 中要求的默认项。
    "model_key": "PSS_TIO2_MODEL",
    "model_source_modules": ["main_angle", "main_cavity", "main_dynamic", "main"],
    "trim_to_air_interface": True,
    "random_seed": 20260616,
    "amplitude_loss_model": "sqrt",
    "phase_offset_rad": 0.0,
}


SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = WORKFLOW_DIR.parent
DATA_DIR = SCRIPT_DIR / "grazing"
IMG_DIR = SCRIPT_DIR / "img" / "grazing"
LINEAR_FIT_DIR = SCRIPT_DIR / "linear_fit" / "grazing"


def timestamp() -> str:
    """返回用于文件名的本地时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dirs() -> None:
    """创建统一输出目录。"""
    for path in (DATA_DIR, IMG_DIR, LINEAR_FIT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def config_json(config: Dict[str, Any] | None = None) -> str:
    """将配置序列化为 npz 里可保存的 JSON 字符串。"""
    payload = CONFIG if config is None else config
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_wavelength_axis(config: Dict[str, Any] | None = None) -> np.ndarray:
    """按 um 返回波长轴，配置中的 step 使用 nm。"""
    cfg = CONFIG if config is None else config
    step_um = float(cfg["wavelength_step_nm"]) * 1e-3
    start = float(cfg["wavelength_min_um"])
    stop = float(cfg["wavelength_max_um"])
    count = int(round((stop - start) / step_um)) + 1
    return np.linspace(start, stop, count, dtype=float)


def build_theta_axis(config: Dict[str, Any] | None = None) -> np.ndarray:
    """按 deg 返回相对晶圆法线的掠入射角轴。"""
    cfg = CONFIG if config is None else config
    start = float(cfg["theta_min_deg"])
    stop = float(cfg["theta_max_deg"])
    step = float(cfg["theta_step_deg"])
    count = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, count, dtype=float)


def build_height_axis(config: Dict[str, Any] | None = None) -> np.ndarray:
    """按 nm 返回三角测量高度扫描轴。"""
    cfg = CONFIG if config is None else config
    scan = cfg["height_scan_nm"]
    start = float(scan["min"])
    stop = float(scan["max"])
    step = float(scan["step"])
    count = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, count, dtype=float)


def latest_npz(pattern: str, directory: Path = DATA_DIR) -> Path:
    """查找指定输出目录下最新的 npz 文件。"""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No npz file matched {pattern!r} in {directory}")
    return matches[-1]


def object_array(values: Iterable[Any]) -> np.ndarray:
    """保存字符串或混合类型列表时使用 object 数组，便于 np.savez_compressed 处理。"""
    return np.asarray(list(values), dtype=object)
