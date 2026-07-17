from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from benchmark_latency import run_benchmark
from generate_static_dataset import sample_parameters
from main_static_stackrt import StaticStackRTGenerator
from model_config import NOMINAL_TRUTH, load_config, wavelength_axis_um
from objective_functions import FitProblem, variable_projection
from spectrum_preprocess import preprocess_spectrum
from tmm_stackrt_matched import StackRTMatchedTMM


def validate_numpy_pipeline(config_path: Path) -> None:
    config = load_config(config_path)
    wavelength = wavelength_axis_um(config)
    assert wavelength.size == 6501, wavelength.size
    model = StackRTMatchedTMM(wavelength)
    spectrum = model.reflectance(NOMINAL_TRUTH)
    assert spectrum.shape == wavelength.shape
    assert np.all(np.isfinite(spectrum)) and np.all(spectrum >= 0.0)
    batch = model.reflectance_batch(np.vstack((NOMINAL_TRUTH, NOMINAL_TRUTH)))
    assert batch.shape == (2, wavelength.size)
    assert np.max(np.abs(batch[0] - spectrum)) < 1e-12
    scale, offset, fitted = variable_projection(spectrum, 1.02 * spectrum - 0.003)
    assert np.isclose(scale, 1.02, atol=1e-10)
    assert np.isclose(offset, -0.003, atol=1e-10)
    assert np.max(np.abs(fitted - (1.02 * spectrum - 0.003))) < 1e-10

    first = sample_parameters(5, np.random.default_rng(1234), "random")
    second = sample_parameters(5, np.random.default_rng(1234), "random")
    assert np.array_equal(first, second)
    measurement = preprocess_spectrum(wavelength, spectrum, 0.0)
    problem = FitProblem(measurement, "linear")
    assert "truth" not in " ".join(problem.__dict__).lower()
    optimizer_source = inspect.getsource(run_benchmark)
    assert optimizer_source.index("result = OPTIMIZERS") < optimizer_source.index("Truth is joined")
    print("PASS: wavelength shape, TMM, variable projection, repeatability, and truth isolation")


def validate_stackrt(config_path: Path, count: int) -> None:
    config = load_config(config_path)
    wavelength = wavelength_axis_um(config)
    tmm = StackRTMatchedTMM(wavelength)
    rng = np.random.default_rng(20260717)
    errors = []
    with StaticStackRTGenerator(wavelength) as stackrt:
        for values in sample_parameters(count, rng, "random"):
            reference = stackrt.spectrum(values)
            prediction = tmm.reflectance(values)
            errors.append(float(np.max(np.abs(reference - prediction))))
    maximum = max(errors)
    print(f"StackRT-TMM max absolute error over {count} random cases: {maximum:.6g}")
    if maximum > 1e-8:
        raise AssertionError(f"StackRT-TMM agreement failed: {maximum:.6g} > 1e-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the independent static inversion project.")
    parser.add_argument("--config", default=str(root / "config_default.json"))
    parser.add_argument("--stackrt", action="store_true", help="Also open Lumerical and run random StackRT closure tests.")
    parser.add_argument("--stackrt-count", type=int, default=3)
    args = parser.parse_args()
    validate_numpy_pipeline(Path(args.config))
    if args.stackrt:
        validate_stackrt(Path(args.config), args.stackrt_count)
    else:
        print("SKIP: real StackRT closure test (run again with --stackrt)")


if __name__ == "__main__":
    main()
