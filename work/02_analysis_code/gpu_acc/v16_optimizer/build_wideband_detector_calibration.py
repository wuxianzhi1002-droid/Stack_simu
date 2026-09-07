"""Build V15 Stage 1 detector frames over the 200-800 nm master band.

Optical spectra and expected electron arrays must already come from local
``main_v15.py --backend api`` StackRT generation.  This script only resamples
the saved shot/read/dark detector chain and never substitutes TMM optical data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from v13_optimizer.build_multiframe_detector_dataset import make_frame, scalar, sha256


MASTER_START_NM = 200.0
MASTER_STOP_NM = 800.0
MASTER_SAMPLING_NM = 0.02
MASTER_POINTS = 30_001
DEFAULT_GROUPS = 10
DEFAULT_FRAMES = 16


def validate_source(data, source: Path) -> dict:
    config = json.loads(str(scalar(data, "config_json", "{}")))
    wavelengths_nm = np.asarray(data["reported_wavelengths_nm"], dtype=np.float64)
    if wavelengths_nm.shape != (MASTER_POINTS,):
        raise RuntimeError(f"unexpected master wavelength shape: {source.name}")
    if (
        not np.isclose(wavelengths_nm[0], MASTER_START_NM, rtol=0.0, atol=1.0e-12)
        or not np.isclose(wavelengths_nm[-1], MASTER_STOP_NM, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError(f"source is not 200-800 nm: {source.name}")
    if not np.isclose(
        float(np.median(np.diff(wavelengths_nm))), MASTER_SAMPLING_NM,
        rtol=0.0, atol=1.0e-10,
    ):
        raise RuntimeError(f"source sampling is not 0.02 nm: {source.name}")
    if str(scalar(data, "generator_version", "")) != "main_v15":
        raise RuntimeError(f"source is not main_v15: {source.name}")
    if str(scalar(data, "optical_backend", "")) != "api":
        raise RuntimeError(f"source optical backend is not StackRT API: {source.name}")
    if not bool(scalar(data, "reported_axis_is_fixed", False)):
        raise RuntimeError(f"reported axis is not fixed: {source.name}")
    if not np.isclose(
        float(scalar(data, "angle_measurement_sigma_deg", np.nan)),
        0.001, rtol=0.0, atol=1.0e-15,
    ):
        raise RuntimeError(f"angle sigma is not 0.001 deg: {source.name}")
    if config.get("MODEL_TYPE") != "stackrt_v15":
        raise RuntimeError(f"unexpected model type: {source.name}")
    if config.get("SOURCE_MODEL") != "eq99x_digitized_peak_normalized":
        raise RuntimeError(f"source is not EQ-99X V15: {source.name}")
    if not config.get("EQ99X_SOURCE_CSV_SHA256"):
        raise RuntimeError(f"source lacks EQ-99X hash: {source.name}")
    return config


def build_group(source: Path, output: Path, frame_count: int, group_index: int) -> dict:
    with np.load(source, allow_pickle=False) as data:
        config = validate_source(data, source)
        settings = config["SPECTROMETER"]
        expected_sample = np.asarray(data["expected_photoelectrons_sample"], dtype=np.float64)
        expected_reference = np.asarray(data["expected_photoelectrons_reference"], dtype=np.float64)
        prnu = np.asarray(data["pixel_response_error_rel"], dtype=np.float64)
        multiplier = float(scalar(data, "detector_noise_multiplier", 1.0))
        expected_shape = (MASTER_POINTS,)
        if any(array.shape != expected_shape for array in (expected_sample, expected_reference, prnu)):
            raise RuntimeError(f"detector source array shape changed: {source.name}")
        seeds = np.asarray(
            [202609040000 + group_index * 1000 + index for index in range(frame_count)],
            dtype=np.int64,
        )
        frames, samples, references, darks, audits = [], [], [], [], []
        for seed in seeds:
            frame, sample, reference, dark, audit = make_frame(
                expected_sample,
                expected_reference,
                prnu,
                multiplier,
                settings,
                int(seed),
            )
            frames.append(frame)
            samples.append(sample)
            references.append(reference)
            darks.append(dark)
            audits.append(audit)
        copied = {key: np.asarray(data[key]) for key in data.files}
    frame_array = np.asarray(frames, dtype=np.float64)
    copied.update(
        {
            "spectrum_measured": frame_array[0],
            "spectra_measured": frame_array,
            "spectrum_single": frame_array[0],
            "spectrum_average": np.mean(frame_array, axis=0),
            "multiframe_count": np.asarray(frame_count, dtype=np.int32),
            "multiframe_seeds": seeds,
            "multiframe_adc_counts_sample": np.asarray(samples),
            "multiframe_adc_counts_reference": np.asarray(references),
            "multiframe_adc_counts_dark": np.asarray(darks),
            "multiframe_audit_json": np.asarray(json.dumps(audits, sort_keys=True)),
            "multiframe_provenance": np.asarray(
                "V15 Stage 1 EQ-99X detector frames resampled from main_v15 StackRT "
                "expected-electron arrays"
            ),
            "multiframe_source_filename": np.asarray(source.name),
            "multiframe_source_sha256": np.asarray(sha256(source)),
        }
    )
    target = output / f"multiframe_g{group_index:02d}_{source.name}"
    np.savez_compressed(target, **copied)
    return {
        "group_index": group_index,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": target.name,
        "output_sha256": sha256(target),
        "frame_count": frame_count,
        "seeds": seeds.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.groups != DEFAULT_GROUPS or args.frames != DEFAULT_FRAMES:
        raise ValueError("formal Stage 1 calibration requires 10 groups x 16 frames")
    sources = sorted(
        args.input_dir.resolve().glob("static_spectrum_detector_typical_*.npz"),
        key=lambda path: path.name,
    )
    if len(sources) != DEFAULT_GROUPS:
        raise RuntimeError(f"expected 10 detector_typical sources, found {len(sources)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = [
        build_group(source, output, args.frames, index)
        for index, source in enumerate(sources)
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v15_stage1_detector_calibration",
        "wavelength_nm": [MASTER_START_NM, MASTER_STOP_NM],
        "sampling_nm": MASTER_SAMPLING_NM,
        "wavelength_points": MASTER_POINTS,
        "groups": groups,
        "shared_state": "StackRT optical state and fixed pixel-response nonuniformity",
        "resampled_state": "shot, read, and dark noise",
        "scope_guard": (
            "Derived detector repeats only; optical state is local StackRT API output; "
            "no TMM-generated dataset and no wavelength extrapolation."
        ),
    }
    (output / "multiframe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "groups": len(groups), "frames": args.frames}, indent=2))


if __name__ == "__main__":
    main()
