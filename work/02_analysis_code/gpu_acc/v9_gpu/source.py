"""Load the frozen formal V9 inversion without modifying it."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXPECTED_SHA256 = "80ff6aca073dc81b31a02abadb4f4f47eefa37238f382eff8ab4c3a04e7e29cc"


def source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tmm_joint_inversion_v9.py"


def source_sha256() -> str:
    digest = hashlib.sha256()
    with source_path().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_v9_module() -> ModuleType:
    path = source_path()
    if not path.is_file():
        raise FileNotFoundError(f"Formal V9 source is unavailable: {path}")
    actual = source_sha256()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"Formal V9 SHA256 changed: expected {EXPECTED_SHA256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("_formal_tmm_joint_inversion_v9", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load formal V9 source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module
