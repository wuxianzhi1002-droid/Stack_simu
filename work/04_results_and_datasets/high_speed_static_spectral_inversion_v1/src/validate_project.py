from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmark_latency import run_benchmark
from generate_static_dataset import sample_parameters
from main_static_stackrt import StaticStackRTCLIGenerator
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
    values_batch = sample_parameters(count, rng, "random")
    axes = np.repeat(wavelength[None, :], count, axis=0)
    references = StaticStackRTCLIGenerator().spectra(axes, values_batch)
    errors = []
    for values, reference in zip(values_batch, references):
        prediction = tmm.reflectance(values)
        errors.append(float(np.max(np.abs(reference - prediction))))
    maximum = max(errors)
    tolerance = 1e-8
    diagnostics_dir = Path(__file__).resolve().parents[1] / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "generator": "StackRT_CLI",
        "sample_count": count,
        "wavelength_count": int(wavelength.size),
        "wavelength_min_nm": float(wavelength.min() * 1000.0),
        "wavelength_max_nm": float(wavelength.max() * 1000.0),
        "max_abs_error": maximum,
        "per_sample_max_abs_error": errors,
        "tolerance": tolerance,
        "passed": maximum <= tolerance,
        "parameters": values_batch.tolist(),
    }
    evidence_path = diagnostics_dir / "stackrt_tmm_closure.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"StackRT-TMM max absolute error over {count} random cases: {maximum:.6g}")
    print(f"Closure evidence: {evidence_path}")
    if maximum > tolerance:
        raise AssertionError(f"StackRT-TMM agreement failed: {maximum:.6g} > {tolerance:.6g}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the independent static inversion project.")
    parser.add_argument("--config", default=str(root / "config_default.json"))
    parser.add_argument("--stackrt", action="store_true", help="Also run FDTD CLI/LSF random StackRT closure tests.")
    parser.add_argument("--stackrt-count", type=int, default=3)
    args = parser.parse_args()
    validate_numpy_pipeline(Path(args.config))
    if args.stackrt:
        validate_stackrt(Path(args.config), args.stackrt_count)
    else:
        print("SKIP: real StackRT closure test (run again with --stackrt)")


if __name__ == "__main__":
    main()
