"""Load the unchanged formal V10 module and record its provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def source_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tmm_joint_inversion_v10.py"


def source_sha256() -> str:
    return hashlib.sha256(source_path().read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_v10_module() -> ModuleType:
    path = source_path()
    spec = importlib.util.spec_from_file_location("_frozen_v10_cpu_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load formal V10 source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module
