"""Diagonal wavelength-noise covariance estimated from V13 detector frames.

The calibration data contain ten independent detector-noise groups and sixteen
frames per group.  Each group is centered independently before pooling its
within-group sum of squares.  This removes the group-specific spectrum and
leaves a single-frame, wavelength-dependent detector-noise estimate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_GROUPS = 10
EXPECTED_FRAMES = 16
DEFAULT_EXPECTED_POINTS = 30_001
DEFAULT_EXPECTED_START_NM = 200.0
DEFAULT_EXPECTED_STOP_NM = 800.0


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _scalar(npz, key: str, default: Any = None) -> Any:
    return np.asarray(npz[key]).item() if key in npz else default


def estimate_diagonal_sigma(
    calibration_dir: Path,
    expected_start_nm: float = DEFAULT_EXPECTED_START_NM,
    expected_stop_nm: float = DEFAULT_EXPECTED_STOP_NM,
    expected_points: int = DEFAULT_EXPECTED_POINTS,
) -> dict[str, Any]:
    """Estimate one-sigma single-frame noise for every reported wavelength."""
    root = calibration_dir.resolve()
    paths = sorted(root.glob("multiframe_g*.npz"), key=lambda path: path.name)
    if len(paths) != EXPECTED_GROUPS:
        raise RuntimeError(
            f"expected {EXPECTED_GROUPS} calibration groups, found {len(paths)}"
        )

    axis = None
    sum_squares = None
    degrees_of_freedom = 0
    file_rows = []
    source_names = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            wavelengths_nm = np.asarray(data["reported_wavelengths_nm"], dtype=np.float64)
            frames = np.asarray(data["spectra_measured"], dtype=np.float64)
            provenance = str(_scalar(data, "multiframe_provenance", ""))
            source_name = str(_scalar(data, "multiframe_source_filename", ""))
            source_sha256 = str(_scalar(data, "multiframe_source_sha256", ""))
            generator = str(_scalar(data, "generator_version", ""))
            sigma_angle = float(_scalar(data, "angle_measurement_sigma_deg", np.nan))
            if frames.shape != (EXPECTED_FRAMES, int(expected_points)):
                raise ValueError(f"unexpected calibration frame shape in {path.name}: {frames.shape}")
            if wavelengths_nm.shape != (int(expected_points),):
                raise ValueError(f"unexpected calibration wavelength shape in {path.name}")
            if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(wavelengths_nm)):
                raise ValueError(f"non-finite calibration data in {path.name}")
            if (
                not np.isclose(wavelengths_nm[0], expected_start_nm, rtol=0.0, atol=1.0e-12)
                or not np.isclose(wavelengths_nm[-1], expected_stop_nm, rtol=0.0, atol=1.0e-12)
            ):
                raise ValueError(
                    f"calibration band is not {expected_start_nm:g}-{expected_stop_nm:g} nm "
                    f"in {path.name}"
                )
            if generator != "main_v15" or "StackRT" not in provenance:
                raise ValueError(f"calibration provenance is not V15 StackRT-derived: {path.name}")
            if not np.isclose(sigma_angle, 0.001, rtol=0.0, atol=1.0e-15):
                raise ValueError(f"calibration angle sigma is not 0.001 deg: {path.name}")
            if axis is None:
                axis = wavelengths_nm.copy()
                sum_squares = np.zeros_like(axis)
            elif not np.array_equal(wavelengths_nm, axis):
                raise ValueError(f"calibration wavelength axis changed: {path.name}")

            centered = frames - np.mean(frames, axis=0, keepdims=True)
            sum_squares += np.sum(centered * centered, axis=0)
            degrees_of_freedom += frames.shape[0] - 1
            source_names.append(source_name)
            file_rows.append(
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "source_filename": source_name,
                    "source_sha256": source_sha256,
                    "frames": int(frames.shape[0]),
                }
            )

    assert axis is not None and sum_squares is not None
    sigma = np.sqrt(sum_squares / float(degrees_of_freedom))
    if sigma.shape != axis.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise FloatingPointError("estimated diagonal noise sigma contains invalid values")

    manifest_path = root / "multiframe_manifest.json"
    manifest_sha256 = sha256_file(manifest_path) if manifest_path.is_file() else None
    digest = hashlib.sha256()
    for row in file_rows:
        digest.update(f"{row['filename']}\t{row['sha256']}\n".encode("utf-8"))
    if manifest_sha256:
        digest.update(f"multiframe_manifest.json\t{manifest_sha256}\n".encode("utf-8"))

    audit = {
        "estimator": "pooled within-group unbiased standard deviation",
        "covariance_model": "diagonal wavelength covariance",
        "noise_target": "single-frame detector acquisition",
        "groups": len(paths),
        "frames_per_group": EXPECTED_FRAMES,
        "degrees_of_freedom": int(degrees_of_freedom),
        "wavelength_points": int(axis.size),
        "wavelength_nm": [float(axis[0]), float(axis[-1])],
        "sigma_min": float(np.min(sigma)),
        "sigma_p01": float(np.percentile(sigma, 1.0)),
        "sigma_median": float(np.median(sigma)),
        "sigma_p99": float(np.percentile(sigma, 99.0)),
        "sigma_max": float(np.max(sigma)),
        "calibration_bundle_sha256": digest.hexdigest(),
        "manifest_sha256": manifest_sha256,
        "source_filenames": source_names,
        "files": file_rows,
        "scope": (
            "V15 digitized EQ-99X source plus synthetic wavelength-dependent detector/photon/read/"
            "dark/ADC behavior; no measured hardware SPD/QE and no off-diagonal covariance"
        ),
    }
    return {"wavelengths_nm": axis, "sigma": sigma, "audit": audit}


def write_covariance_audit(output_path: Path, result: dict[str, Any]) -> None:
    payload = {
        "audit": result["audit"],
        "sigma": {
            "count": int(len(result["sigma"])),
            "units": "reported normalized spectrum",
        },
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
