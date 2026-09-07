"""Build repeated detector frames from saved StackRT expected-electron arrays.

This does not replace the optical forward model.  Every group inherits its
StackRT-generated optical state and resamples only shot/read/dark detector
noise while keeping the saved pixel-response nonuniformity fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def scalar(data, key, default=None):
    return np.asarray(data[key]).item() if key in data else default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adc_counts(expected, exposure_s, averages, multiplier, prnu, settings, rng):
    photo_mean = np.maximum(
        np.asarray(expected, dtype=np.float64) * (1.0 + multiplier * prnu), 0.0
    )
    dark_mean = float(settings["DARK_CURRENT_E_PER_S"]) * exposure_s * multiplier
    if multiplier > 0.0:
        electrons = rng.poisson(np.maximum((photo_mean + dark_mean) * averages, 0.0))
        electrons = electrons.astype(np.float64) / averages
        electrons += rng.normal(
            0.0,
            float(settings["READ_NOISE_E_RMS"]) * multiplier / np.sqrt(averages),
            size=photo_mean.shape,
        )
    else:
        electrons = photo_mean
    full_well = float(settings["FULL_WELL_ELECTRONS"])
    saturation = float(np.mean(electrons >= full_well))
    electrons = np.clip(electrons, 0.0, full_well)
    nonlinearity = float(settings["ADC_NONLINEARITY_REL"]) * multiplier
    electrons = electrons * (1.0 + nonlinearity * (electrons / full_well) ** 2)
    adc_max = float(2 ** int(settings["ADC_BITS"]) - 1)
    usable = adc_max - float(settings["ADC_BIAS_COUNTS"])
    counts = float(settings["ADC_BIAS_COUNTS"]) + electrons * usable / full_well
    if bool(settings["QUANTIZATION_ALWAYS_ENABLED"]):
        counts = np.rint(counts)
    return np.clip(counts, 0.0, adc_max), saturation


def make_frame(expected_sample, expected_reference, prnu, multiplier, settings, seed):
    rng = np.random.default_rng(seed)
    sample, sample_sat = adc_counts(
        expected_sample, float(settings["SAMPLE_EXPOSURE_S"]),
        int(settings["SAMPLE_AVERAGES"]), multiplier, prnu, settings, rng,
    )
    reference, reference_sat = adc_counts(
        expected_reference, float(settings["REFERENCE_EXPOSURE_S"]),
        int(settings["REFERENCE_AVERAGES"]), multiplier, prnu, settings, rng,
    )
    dark, dark_sat = adc_counts(
        np.zeros_like(expected_reference), float(settings["DARK_EXPOSURE_S"]),
        int(settings["DARK_AVERAGES"]), multiplier, prnu, settings, rng,
    )
    bias = float(settings["ADC_BIAS_COUNTS"])
    dark_signal = dark - bias
    dark_exposure = float(settings["DARK_EXPOSURE_S"])
    sample_net = sample - bias - dark_signal * float(settings["SAMPLE_EXPOSURE_S"]) / dark_exposure
    reference_net = reference - bias - dark_signal * float(settings["REFERENCE_EXPOSURE_S"]) / dark_exposure
    minimum = float(settings["MIN_REFERENCE_NET_COUNTS"])
    invalid = reference_net < minimum
    measured = sample_net / np.where(invalid, minimum, reference_net)
    audit = {
        "sample_saturation_fraction": sample_sat,
        "reference_saturation_fraction": reference_sat,
        "dark_saturation_fraction": dark_sat,
        "invalid_reference_pixel_fraction": float(np.mean(invalid)),
    }
    return measured, sample, reference, dark, audit


def build_group(source: Path, output: Path, frame_count: int, group_index: int):
    with np.load(source, allow_pickle=False) as data:
        config = json.loads(str(scalar(data, "config_json", "{}")))
        if float(config["WAVELENGTH_START_UM"]) != 0.22 or float(config["WAVELENGTH_STOP_UM"]) != 0.58:
            raise RuntimeError(f"not a 220-580 nm source: {source.name}")
        if int(config["OUTPUT_SAMPLE_POINTS"]) != 18001:
            raise RuntimeError(f"unexpected wavelength point count: {source.name}")
        if str(scalar(data, "generator_version", "")) != "main_v12":
            raise RuntimeError(f"not a main_v12 StackRT source: {source.name}")
        settings = config["SPECTROMETER"]
        expected_sample = np.asarray(data["expected_photoelectrons_sample"], dtype=np.float64)
        expected_reference = np.asarray(data["expected_photoelectrons_reference"], dtype=np.float64)
        prnu = np.asarray(data["pixel_response_error_rel"], dtype=np.float64)
        multiplier = float(scalar(data, "detector_noise_multiplier", 1.0))
        seeds = np.asarray(
            [202609020000 + group_index * 1000 + i for i in range(frame_count)],
            dtype=np.int64,
        )
        frames, samples, references, darks, audits = [], [], [], [], []
        for seed in seeds:
            frame, sample, reference, dark, audit = make_frame(
                expected_sample, expected_reference, prnu, multiplier, settings, int(seed)
            )
            frames.append(frame)
            samples.append(sample)
            references.append(reference)
            darks.append(dark)
            audits.append(audit)
        copied = {key: np.asarray(data[key]) for key in data.files}
    frames = np.asarray(frames, dtype=np.float64)
    copied.update(
        {
            "spectrum_measured": frames[0],
            "spectra_measured": frames,
            "spectrum_single": frames[0],
            "spectrum_average": np.mean(frames, axis=0),
            "multiframe_count": np.asarray(frame_count, dtype=np.int32),
            "multiframe_seeds": seeds,
            "multiframe_adc_counts_sample": np.asarray(samples),
            "multiframe_adc_counts_reference": np.asarray(references),
            "multiframe_adc_counts_dark": np.asarray(darks),
            "multiframe_audit_json": np.asarray(json.dumps(audits, sort_keys=True)),
            "multiframe_provenance": np.asarray(
                "detector frames resampled from main_v12 StackRT expected-electron arrays"
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--frames", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    sources = sorted(args.input_dir.glob("static_spectrum_detector_typical_*.npz"))
    if len(sources) < args.groups:
        raise RuntimeError(f"requested {args.groups} detector_typical sources, found {len(sources)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = [build_group(source, output, args.frames, i) for i, source in enumerate(sources[: args.groups])]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wavelength_nm": [220.0, 580.0],
        "sampling_nm": 0.02,
        "wavelength_points": 18001,
        "groups": groups,
        "comparison_modes": ["single", "average", "joint"],
        "shared_state": "StackRT optical state and fixed pixel-response nonuniformity",
        "resampled_state": "shot, read, and dark noise",
        "scope_guard": "Derived detector repeats only; no TMM-generated dataset and no optical-state regeneration.",
    }
    (output / "multiframe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "groups": len(groups), "frames": args.frames}, indent=2))


if __name__ == "__main__":
    main()
