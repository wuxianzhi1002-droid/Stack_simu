"""Generate a deterministic Phase 0 CPU strict-TMM reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backend.numpy_backend import NumpyStrictTMMBackend
from .backend.v10_source import load_v10_module, source_path, source_sha256

REFERENCE_SEED = 20260826
REFERENCE_BATCH_SIZE = 13
BOUNDS = np.asarray(
    [
        [95.0, 105.0],
        [20.0, 40.0],
        [1.0, 20.0],
        [30.0, 50.0],
        [30.0, 50.0],
        [-0.1, 0.1],
    ],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def deterministic_parameters(
    count: int = REFERENCE_BATCH_SIZE, seed: int = REFERENCE_SEED
) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(seed)
    lower, upper = BOUNDS[:, 0], BOUNDS[:, 1]
    params = rng.uniform(lower, upper, size=(count, 6))
    params[0] = (lower + upper) / 2.0
    return params.astype(np.float64)


def default_input_npz() -> Path:
    repository = source_path().parents[2]
    return (
        repository
        / "work"
        / "04_results_and_datasets"
        / "static_stackrt_v10_20260825_232804"
        / "static_spectrum_clean_r0000_seed20260825.npz"
    )


def internal_grid_from_npz(npz_path: Path) -> tuple[np.ndarray, dict]:
    v10 = load_v10_module()
    with np.load(npz_path, allow_pickle=False) as data:
        config = json.loads(str(np.asarray(data["config_json"]).item()))
        reported_um = np.asarray(data["wavelengths"], dtype=np.float64)
        margin_nm = float(np.asarray(data["internal_wavelength_margin_nm"]).item())
    model = v10.SpectrometerForwardModel(
        reported_um, config, internal_margin_nm=margin_nm
    )
    grid_metadata = {
        "reported_points": int(reported_um.size),
        "internal_points": int(model.internal_um.size),
        "internal_start_nm": float(model.internal_nm[0]),
        "internal_stop_nm": float(model.internal_nm[-1]),
        "internal_step_nm": float(model.internal_step_nm),
        "internal_margin_nm": float(model.internal_margin_nm),
    }
    return model.internal_um.astype(np.float64, copy=False), grid_metadata


def generate_reference(
    input_npz: Path, output_npz: Path, manifest_path: Path
) -> dict:
    input_npz = input_npz.resolve()
    output_npz = output_npz.resolve()
    manifest_path = manifest_path.resolve()
    wavelengths_um, grid_metadata = internal_grid_from_npz(input_npz)
    params = deterministic_parameters()
    backend = NumpyStrictTMMBackend(wavelengths_um)
    reflectance = backend.predict_batch(params)

    if reflectance.dtype != np.float64:
        raise RuntimeError("CPU reference must be float64.")
    if not np.all(np.isfinite(reflectance)):
        raise RuntimeError("CPU reference contains NaN or Inf.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        wavelengths_um=wavelengths_um,
        params=params,
        reflectance=reflectance,
        parameter_names=np.asarray(
            ["Air", "HSQ", "PSS", "SOC", "TiO2", "Angle"]
        ),
        parameter_units=np.asarray(["um", "nm", "nm", "nm", "nm", "deg"]),
        source_v10_sha256=np.asarray(source_sha256()),
        input_npz_sha256=np.asarray(sha256_file(input_npz)),
        reference_seed=np.asarray(REFERENCE_SEED, dtype=np.int64),
    )

    manifest = {
        "phase": "Phase 0 CPU strict-TMM freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_v10_source": str(source_path().resolve()),
        "formal_v10_sha256": source_sha256(),
        "input_npz": str(input_npz),
        "input_npz_sha256": sha256_file(input_npz),
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "reference_seed": REFERENCE_SEED,
        "reference_batch_size": int(params.shape[0]),
        "dtype_real": "float64",
        "dtype_complex": "complex128",
        "wavelength_grid": grid_metadata,
        "arrays": {
            "wavelengths_um_sha256": array_sha256(wavelengths_um),
            "params_sha256": array_sha256(params),
            "reflectance_sha256": array_sha256(reflectance),
            "reflectance_shape": list(reflectance.shape),
            "nan_count": int(np.isnan(reflectance).sum()),
            "inf_count": int(np.isinf(reflectance).sum()),
        },
        "acceptance": {
            "gpu_cpu_rmse_max": 1.0e-10,
            "gpu_cpu_max_abs_max": 1.0e-8,
            "nan_inf_allowed": 0,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "scope": (
            "Strict TMM reflectance only. ILS, Jacobian, least_squares and "
            "global search are intentionally excluded from Phase 1."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    reference_dir = package_dir / "cpu_reference"
    parser = argparse.ArgumentParser(
        description="Freeze formal V10 CPU strict-TMM reference data."
    )
    parser.add_argument("--input-npz", type=Path, default=default_input_npz())
    parser.add_argument(
        "--output",
        type=Path,
        default=reference_dir / "forward_reference.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=reference_dir / "reference_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_reference(args.input_npz, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
