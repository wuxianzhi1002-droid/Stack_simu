from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from main_static_stackrt import StaticStackRTCLIGenerator, StaticStackRTGenerator, TMMStaticSmokeGenerator
from model_config import LOWER_BOUNDS, UPPER_BOUNDS, load_config, wavelength_axis_um


def sample_parameters(count: int, rng: np.random.Generator, trajectory: str) -> np.ndarray:
    if count < 1:
        raise ValueError("sample count must be positive.")
    if trajectory == "random":
        return rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(count, 5))
    if trajectory == "tracking":
        values = np.empty((count, 5), dtype=float)
        values[0] = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
        step_sigma = np.array([0.03, 0.03, 0.02, 0.03, 0.03])
        for index in range(1, count):
            values[index] = np.clip(values[index - 1] + rng.normal(0.0, step_sigma), LOWER_BOUNDS, UPPER_BOUNDS)
        return values
    raise ValueError("trajectory must be 'random' or 'tracking'.")


def apply_measurement_effects(
    nominal_wavelengths_um: np.ndarray,
    generator,
    truth: np.ndarray,
    noise_config: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    offset_nm = float(rng.normal(0.0, noise_config["wavelength_offset_sigma_nm"]))
    scale_ppm = float(rng.normal(0.0, noise_config["wavelength_scale_sigma_ppm"]))
    source_scale = float(1.0 + rng.normal(0.0, noise_config["source_scale_sigma"]))
    source_offset = float(rng.normal(0.0, noise_config["source_offset_sigma"]))
    physical_axis = nominal_wavelengths_um * (1.0 + scale_ppm * 1e-6) + offset_nm / 1000.0

    # The generator is rebuilt for drifted axes by the caller-facing factory contract.
    if hasattr(generator, "model"):
        physical_generator = TMMStaticSmokeGenerator(physical_axis)
        clean = physical_generator.spectrum(truth)
    else:
        original_axis = generator.wavelengths_um
        generator.wavelengths_um = physical_axis
        try:
            clean = generator.spectrum(truth)
        finally:
            generator.wavelengths_um = original_axis
    sigma = float(noise_config["noise_sigma"])
    measured = source_scale * clean + source_offset
    if sigma > 0.0:
        measured = measured + rng.normal(0.0, sigma, size=measured.shape)
    return measured, {
        "noise_sigma": sigma,
        "wavelength_offset_nm": offset_nm,
        "wavelength_scale_ppm": scale_ppm,
        "source_scale": source_scale,
        "source_offset": source_offset,
    }


def realize_measurement_nuisance(noise_config: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    return {
        "noise_sigma": float(noise_config["noise_sigma"]),
        "wavelength_offset_nm": float(rng.normal(0.0, noise_config["wavelength_offset_sigma_nm"])),
        "wavelength_scale_ppm": float(rng.normal(0.0, noise_config["wavelength_scale_sigma_ppm"])),
        "source_scale": float(1.0 + rng.normal(0.0, noise_config["source_scale_sigma"])),
        "source_offset": float(rng.normal(0.0, noise_config["source_offset_sigma"])),
    }


def generate_dataset(
    config: dict[str, Any],
    output: Path,
    backend: str,
    noise_level: str,
    sample_count: int | None,
    trajectory: str | None,
) -> Path:
    wavelengths_um = wavelength_axis_um(config)
    profile = str(config["dataset"]["profile"])
    count = int(sample_count or config["dataset"]["sizes"][profile])
    trajectory_name = trajectory or str(config["dataset"]["trajectory"])
    noise_config = config["dataset"]["noise"][noise_level]
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    truth = sample_parameters(count, rng, trajectory_name)
    spectra = np.empty((count, wavelengths_um.size), dtype=float)
    nuisance = {name: np.empty(count, dtype=float) for name in ("noise_sigma", "wavelength_offset_nm", "wavelength_scale_ppm", "source_scale", "source_offset")}

    if backend == "stackrt-cli":
        physical_axes = np.empty_like(spectra)
        for index in range(count):
            realized = realize_measurement_nuisance(noise_config, rng)
            for name, value in realized.items():
                nuisance[name][index] = value
            physical_axes[index] = (
                wavelengths_um * (1.0 + realized["wavelength_scale_ppm"] * 1e-6)
                + realized["wavelength_offset_nm"] / 1000.0
            )
        clean_spectra = StaticStackRTCLIGenerator().spectra(physical_axes, truth)
        for index in range(count):
            spectra[index] = nuisance["source_scale"][index] * clean_spectra[index] + nuisance["source_offset"][index]
            if nuisance["noise_sigma"][index] > 0.0:
                spectra[index] += rng.normal(0.0, nuisance["noise_sigma"][index], size=wavelengths_um.size)
    else:
        generator_class = StaticStackRTGenerator if backend == "stackrt-interop" else TMMStaticSmokeGenerator
        with generator_class(wavelengths_um) as generator:
            for index in range(count):
                spectra[index], realized = apply_measurement_effects(wavelengths_um, generator, truth[index], noise_config, rng)
                for name, value in realized.items():
                    nuisance[name][index] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    generation = {
        "generator": "StackRT_CLI" if backend == "stackrt-cli" else ("StackRT_INTEROP" if backend == "stackrt-interop" else "TMM_SMOKE_TEST_NOT_STACKRT"),
        "backend": backend,
        "noise_level": noise_level,
        "trajectory": trajectory_name,
        "random_seed": seed,
        "sample_count": count,
        "wavelength_count": int(wavelengths_um.size),
        "frequency_convention": "3e8/lambda_nominal",
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": config,
    }
    np.savez_compressed(
        output,
        wavelengths_um=wavelengths_um,
        spectra=spectra,
        air_cavity_um=truth[:, 0],
        film_thicknesses_nm=truth[:, 1:],
        **nuisance,
        generation_parameters_json=np.asarray(json.dumps(generation, ensure_ascii=False)),
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate static StackRT spectra or labelled TMM smoke data.")
    parser.add_argument("--config", default=str(project_root / "config_default.json"))
    parser.add_argument("--backend", choices=("stackrt-cli", "stackrt-interop", "tmm-smoke"), default="stackrt-cli")
    parser.add_argument("--noise-level", choices=("ideal", "noisy"), default="ideal")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--trajectory", choices=("random", "tracking"))
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
    label = args.backend.replace("-", "_") if args.backend != "tmm-smoke" else "tmm_smoke_not_stackrt"
    output = Path(args.output) if args.output else project_root / "datasets" / f"static_{label}_{args.noise_level}.npz"
    result = generate_dataset(config, output, args.backend, args.noise_level, args.sample_count, args.trajectory)
    with np.load(result) as data:
        print(f"Saved {result}")
        print(f"spectra.shape={data['spectra'].shape}, wavelength_count={data['wavelengths_um'].size}")


if __name__ == "__main__":
    main()
