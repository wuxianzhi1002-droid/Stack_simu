"""Build field-lossless compact remote inputs for V17 ablation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = [
    "wavelengths", "reported_wavelengths_nm", "spectrum_measured", "config_json",
    "reported_axis_is_fixed", "internal_wavelength_margin_nm", "measured_reflector_angle_deg",
    "angle_measurement_sigma_deg", "angle_measurement_error_deg", "angle_measurement_mode", "reflector_angle_setpoint_deg", "layer_names", "layer_thickness_um",
    "true_reflector_angle_deg", "noise_case", "noise_factor", "noise_level", "realization_index",
    "random_seed", "generator_version", "optical_backend", "ablation_scenario", "reduced_factor",
    "reduced_profile_keys", "reduction_fraction", "baseline_typical_profile_json",
    "scenario_typical_profile_json", "baseline_angle_measurement_sigma_deg", "paired_realization_seed",
    "ablation_pairing_contract", "eq99x_source_csv_sha256",
]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.input_dir.resolve(); output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted(source_root.glob("*/static_spectrum_combined_typical_r*.npz"))
    if len(sources) != 100:
        raise RuntimeError(f"expected 100 inputs, got {len(sources)}")
    rows = []
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            missing = [field for field in FIELDS if field not in data]
            if missing:
                raise KeyError(f"{source}: {missing}")
            payload = {field: np.asarray(data[field]) for field in FIELDS}
        payload["full_local_source_filename"] = np.asarray(str(source.relative_to(source_root)))
        payload["full_local_source_sha256"] = np.asarray(sha256(source))
        scenario = str(payload["ablation_scenario"].item())
        target = output / f"{scenario}__{source.name}"
        np.savez_compressed(target, **payload)
        rows.append({"filename": target.name, "sha256": sha256(target), "bytes": target.stat().st_size, "full_local_source_sha256": str(payload["full_local_source_sha256"].item())})
    source_manifest = source_root / "simulation_manifest.json"
    manifest = {
        "version": "v17_compact_ablation_remote_inputs", "count": len(rows),
        "source_manifest": json.loads(source_manifest.read_text(encoding="utf-8")),
        "source_manifest_sha256": sha256(source_manifest), "fields": FIELDS, "files": rows,
        "scope": "Field-lossless for V17 runner; complete StackRT arrays remain local.",
    }
    (output / "simulation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": "PASS", "count": len(rows), "bytes": sum(row["bytes"] for row in rows), "output": str(output)}, indent=2))

if __name__ == "__main__":
    main()
