"""Create compact, hash-audited V16 multi-angle remote inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = [
    "reported_wavelengths_nm",
    "spectra_measured",
    "measured_reflector_angles_deg",
    "true_reflector_angles_deg",
    "config_json",
    "generator_version",
    "multiangle_provenance",
    "eq99x_source_csv_sha256",
    "layer_names",
    "layer_thickness_um",
    "noise_case",
    "noise_factor",
    "noise_level",
    "realization_index",
    "random_seed",
    "internal_wavelength_margin_nm",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted(source_root.glob("multiangle_*_typical_*.npz"))
    if len(sources) != 100:
        raise RuntimeError(f"expected 100 files, got {len(sources)}")

    rows = []
    for source in sources:
        source_hash = sha256(source)
        with np.load(source, allow_pickle=False) as data:
            missing = [name for name in FIELDS if name not in data]
            if missing:
                raise KeyError(f"{source.name}: {missing}")
            payload = {name: np.asarray(data[name]) for name in FIELDS}
        payload["full_local_source_filename"] = np.asarray(source.name)
        payload["full_local_source_sha256"] = np.asarray(source_hash)
        target = output / source.name
        np.savez_compressed(target, **payload)
        rows.append(
            {
                "filename": target.name,
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
                "full_local_source_sha256": source_hash,
            }
        )

    source_manifest = source_root / "simulation_manifest.json"
    manifest = {
        "version": "v16_compact_nonzero_multiangle_remote_inputs",
        "count": len(rows),
        "source_manifest": json.loads(source_manifest.read_text(encoding="utf-8")),
        "source_manifest_sha256": sha256(source_manifest),
        "fields": FIELDS,
        "files": rows,
        "scope": "Field-lossless for V16 runner; complete StackRT arrays remain in the local source dataset.",
    }
    (output / "simulation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overall": "PASS",
                "count": len(rows),
                "bytes": sum(row["bytes"] for row in rows),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
