"""Validate reference hashes and replay the frozen NumPy backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .backend.numpy_backend import NumpyStrictTMMBackend
from .backend.v10_source import source_sha256


REFERENCE_RMSE_TOL = 1.0e-10
REFERENCE_MAX_ABS_TOL = 1.0e-8


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).view(np.uint8)
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=package_dir / "cpu_reference" / "forward_reference.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=package_dir / "cpu_reference" / "reference_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.reference, allow_pickle=False) as data:
        wavelengths = np.asarray(data["wavelengths_um"], dtype=np.float64)
        params = np.asarray(data["params"], dtype=np.float64)
        expected = np.asarray(data["reflectance"], dtype=np.float64)
        recorded_source_hash = str(data["source_v10_sha256"].item())

    current_source_hash = source_sha256()
    backend = NumpyStrictTMMBackend(wavelengths)
    actual = backend.predict_batch(params)
    difference = actual - expected
    report = {
        "source_hash_matches_npz": current_source_hash == recorded_source_hash,
        "source_hash_matches_manifest": (
            current_source_hash == manifest["formal_v10_sha256"]
        ),
        "wavelength_hash_matches": (
            array_sha256(wavelengths)
            == manifest["arrays"]["wavelengths_um_sha256"]
        ),
        "params_hash_matches": (
            array_sha256(params) == manifest["arrays"]["params_sha256"]
        ),
        "reflectance_hash_matches": (
            array_sha256(expected) == manifest["arrays"]["reflectance_sha256"]
        ),
        "replay_rmse": float(np.sqrt(np.mean(difference**2))),
        "replay_max_abs": float(np.max(np.abs(difference))),
        "nan_inf_count": int(
            np.size(actual) - np.count_nonzero(np.isfinite(actual))
        ),
    }
    report["pass"] = bool(
        all(value for key, value in report.items() if key.endswith("_matches"))
        and report["replay_rmse"] <= REFERENCE_RMSE_TOL
        and report["replay_max_abs"] <= REFERENCE_MAX_ABS_TOL
        and report["nan_inf_count"] == 0
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
