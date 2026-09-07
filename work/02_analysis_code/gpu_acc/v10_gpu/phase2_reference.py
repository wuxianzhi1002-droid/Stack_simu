"""Generate the frozen CPU reference for Phase 2 strict spectrometer response."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backend.v10_source import source_path, source_sha256
from .reference import (
    REFERENCE_BATCH_SIZE,
    REFERENCE_SEED,
    array_sha256,
    default_input_npz,
    deterministic_parameters,
    sha256_file,
)
from .spectrometer import NumpyStrictSpectrometerBackend


def load_contract_input(input_npz: Path) -> tuple[np.ndarray, dict, float]:
    with np.load(input_npz, allow_pickle=False) as data:
        reported_um = np.asarray(data["wavelengths"], dtype=np.float64)
        generator_config = json.loads(str(data["config_json"].item()))
        margin_nm = float(data["internal_wavelength_margin_nm"].item())
    return reported_um, generator_config, margin_nm


def generate_reference(input_npz: Path, output_npz: Path, manifest_path: Path) -> dict:
    input_npz = input_npz.resolve()
    output_npz = output_npz.resolve()
    manifest_path = manifest_path.resolve()
    reported_um, config, margin_nm = load_contract_input(input_npz)
    params = deterministic_parameters()
    backend = NumpyStrictSpectrometerBackend(
        reported_um, config, internal_margin_nm=margin_nm
    )
    spectra = backend.predict_batch(params)
    if spectra.dtype != np.float64 or not np.all(np.isfinite(spectra)):
        raise RuntimeError("Phase 2 CPU reference must be finite float64.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        reported_wavelengths_um=reported_um,
        params=params,
        spectra=spectra,
        generator_config_json=np.asarray(json.dumps(config, sort_keys=True)),
        internal_margin_nm=np.asarray(margin_nm, dtype=np.float64),
        source_v10_sha256=np.asarray(source_sha256()),
        input_npz_sha256=np.asarray(sha256_file(input_npz)),
        reference_seed=np.asarray(REFERENCE_SEED, dtype=np.int64),
    )
    manifest = {
        "phase": "Phase 2 CPU strict spectrometer-response freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_v10_source": str(source_path().resolve()),
        "formal_v10_sha256": source_sha256(),
        "input_npz": str(input_npz),
        "input_npz_sha256": sha256_file(input_npz),
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "reference_seed": REFERENCE_SEED,
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "reported_points": int(reported_um.size),
        "internal_points": int(backend.internal_wavelengths_um.size),
        "internal_margin_nm": margin_nm,
        "arrays": {
            "reported_wavelengths_um_sha256": array_sha256(reported_um),
            "params_sha256": array_sha256(params),
            "spectra_sha256": array_sha256(spectra),
            "spectra_shape": list(spectra.shape),
            "nan_count": int(np.isnan(spectra).sum()),
            "inf_count": int(np.isinf(spectra).sum()),
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
            "Strict TMM plus nominal source, throughput, QE, photon weighting, "
            "Gaussian ILS, fixed reported-axis sampling and sample/reference "
            "exposure normalization. No Jacobian, optimizer, least_squares or JAX."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    reference_dir = package_dir / "cpu_reference_phase2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, default=default_input_npz())
    parser.add_argument("--output", type=Path, default=reference_dir / "spectrometer_reference.npz")
    parser.add_argument("--manifest", type=Path, default=reference_dir / "reference_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(generate_reference(args.input_npz, args.output, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
