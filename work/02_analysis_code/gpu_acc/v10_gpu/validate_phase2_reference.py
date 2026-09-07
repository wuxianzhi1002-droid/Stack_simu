"""Validate hashes and replay the frozen Phase 2 NumPy response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backend.v10_source import source_sha256
from .reference import array_sha256
from .spectrometer import NumpyStrictSpectrometerBackend

RMSE_LIMIT = 1.0e-10
MAX_ABS_LIMIT = 1.0e-8


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent / "cpu_reference_phase2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=package_dir / "spectrometer_reference.npz")
    parser.add_argument("--manifest", type=Path, default=package_dir / "reference_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.reference, allow_pickle=False) as data:
        reported = np.asarray(data["reported_wavelengths_um"], dtype=np.float64)
        params = np.asarray(data["params"], dtype=np.float64)
        expected = np.asarray(data["spectra"], dtype=np.float64)
        config = json.loads(str(data["generator_config_json"].item()))
        margin = float(data["internal_margin_nm"].item())
        recorded_source_hash = str(data["source_v10_sha256"].item())
    actual = NumpyStrictSpectrometerBackend(reported, config, margin).predict_batch(params)
    difference = actual - expected
    report = {
        "source_hash_matches_npz": source_sha256() == recorded_source_hash,
        "source_hash_matches_manifest": source_sha256() == manifest["formal_v10_sha256"],
        "reported_hash_matches": array_sha256(reported) == manifest["arrays"]["reported_wavelengths_um_sha256"],
        "params_hash_matches": array_sha256(params) == manifest["arrays"]["params_sha256"],
        "spectra_hash_matches": array_sha256(expected) == manifest["arrays"]["spectra_sha256"],
        "replay_rmse": float(np.sqrt(np.mean(difference**2))),
        "replay_max_abs": float(np.max(np.abs(difference))),
        "nan_inf_count": int(actual.size - np.count_nonzero(np.isfinite(actual))),
    }
    report["pass"] = bool(
        all(value for key, value in report.items() if key.endswith("_matches"))
        and report["replay_rmse"] <= RMSE_LIMIT
        and report["replay_max_abs"] <= MAX_ABS_LIMIT
        and report["nan_inf_count"] == 0
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
