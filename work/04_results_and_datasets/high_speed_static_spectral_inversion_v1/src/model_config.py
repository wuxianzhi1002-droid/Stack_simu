from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

PARAMETER_NAMES = ("Air", "HSQ", "PSS", "SOC", "TiO2")
LAYER_NAMES = ("RefReflector", "Air", "HSQ", "PSS", "SOC", "TiO2", "Cu")
TRUTH_UNITS = ("um", "nm", "nm", "nm", "nm")
NOMINAL_TRUTH = np.array([1000.0, 30.0, 10.0, 40.0, 40.0], dtype=float)
LOWER_BOUNDS = np.array([998.0, 20.0, 1.0, 30.0, 30.0], dtype=float)
UPPER_BOUNDS = np.array([1002.0, 40.0, 20.0, 50.0, 50.0], dtype=float)
C0_M_S = 299_792_458.0
STACKRT_FREQUENCY_C_M_S = 3.0e8


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def wavelength_axis_um(config: dict[str, Any]) -> np.ndarray:
    spectral = config["spectral"]
    start = float(spectral["wavelength_min_nm"])
    stop = float(spectral["wavelength_max_nm"])
    step = float(spectral["step_nm"])
    if not (start > 0.0 and stop > start and step > 0.0):
        raise ValueError("Invalid spectral range or step.")
    count = int(round((stop - start) / step)) + 1
    axis_nm = start + step * np.arange(count, dtype=float)
    if not np.isclose(axis_nm[-1], stop, atol=step * 1e-6):
        raise ValueError("Spectral endpoints are not divisible by step_nm.")
    return axis_nm / 1000.0


def material_n(name: str, wavelength_nominal_um: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelength_nominal_um, dtype=np.float64)
    if name == "RefReflector":
        return np.full_like(w, 5.8284, dtype=np.complex128)
    if name == "Air":
        return np.ones_like(w, dtype=np.complex128)
    if name == "HSQ":
        return np.full_like(w, 1.41, dtype=np.complex128)
    if name == "PSS":
        return np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    if name == "SOC":
        return (1.55 + 0.005 / w**2).astype(np.complex128)
    if name == "TiO2":
        return (2.4 + 0.02 / w**2).astype(np.complex128)
    if name == "Cu":
        return np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    raise ValueError(f"Unknown material: {name}")


def parameters_to_thickness_um(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.shape != (5,):
        raise ValueError(f"Expected five parameters, got shape {x.shape}.")
    return np.array([0.0, x[0], x[1] / 1000.0, x[2] / 1000.0, x[3] / 1000.0, x[4] / 1000.0, 0.0])


def bounds_center() -> np.ndarray:
    return 0.5 * (LOWER_BOUNDS + UPPER_BOUNDS)
