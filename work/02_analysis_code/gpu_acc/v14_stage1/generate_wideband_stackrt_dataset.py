"""Generate the paired 200-800 nm V14 Stage 1 master dataset with StackRT.

Only the ten ``typical`` noise cases are generated (ten realizations each).
Seeds retain their original position in the complete V12 case matrix.  The
longer wavelength arrays change RNG consumption, so seed/name identity does
not imply exact realization-value identity with the historical 220-580 data.
The optical backend is locked to the local Lumerical ``stackrt`` API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


MASTER_START_NM = 200.0
MASTER_STOP_NM = 800.0
MASTER_SAMPLING_NM = 0.02
MASTER_POINTS = 30_001
CANDIDATE_BANDS = (
    (220.0, 580.0),
    (200.0, 600.0),
    (200.0, 650.0),
    (200.0, 700.0),
    (200.0, 800.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_main_v12():
    source = (
        Path(__file__).resolve().parents[3]
        / "01_simulation_models"
        / "01_Lumerical_Workflow"
        / "main_v12.py"
    )
    module_name = "_v14_stage1_main_v12_generator"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load main_v12 from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats != 10:
        raise ValueError("formal Stage 1 generation requires 10 realizations per noise case")
    main_v12, source_path = load_main_v12()
    override = argparse.Namespace(
        wavelength_start_nm=MASTER_START_NM,
        wavelength_stop_nm=MASTER_STOP_NM,
        output_sampling_nm=MASTER_SAMPLING_NM,
    )
    main_v12.apply_wavelength_cli_overrides(override)
    main_v12.configure_v12()
    kernel = main_v12._kernel
    cases = [f"{factor}_typical" for factor in kernel.NOISE_FACTORS]
    global_case_index = {name: index for index, name in enumerate(kernel.all_case_names())}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generator = main_v12.V12DatasetGenerator("api", output)
    rows = []
    representatives = {}
    failures = []
    with kernel.OpticalSolver("api", output / "_stackrt_batch_bridge") as solver:
        for case_name in cases:
            case_index = global_case_index[case_name]
            for realization_index in range(args.repeats):
                seed = int(args.seed + case_index * 1_000_000 + realization_index)
                try:
                    data, metadata = generator.generate_one(
                        solver, case_name, realization_index, seed
                    )
                    path = generator.save_one(data)
                    rows.append(main_v12.dataset_index_row(metadata, path))
                    representatives.setdefault(case_name, data)
                    print(
                        f"[{len(rows)}/100] {path.name} backend=StackRT/api "
                        f"points={MASTER_POINTS}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "case": case_name,
                            "realization": realization_index,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise
    if len(rows) != 100 or failures:
        raise RuntimeError(f"Stage 1 generation incomplete: rows={len(rows)}, failures={failures}")
    index_path = output / "dataset_index.csv"
    with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = kernel.save_representative_plot(output, representatives)
    kernel_path = source_path.with_name("main_v10.py")
    manifest = {
        "version": "v14_stage1_wideband_stackrt_master",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optical_backend": "Lumerical lumapi.FDTD.stackrt",
        "generator_version_in_npz": "main_v12",
        "master_wavelength_nm": [MASTER_START_NM, MASTER_STOP_NM],
        "output_sampling_nm": MASTER_SAMPLING_NM,
        "output_sample_points": MASTER_POINTS,
        "candidate_bands_nm": [list(item) for item in CANDIDATE_BANDS],
        "candidate_contract": "every candidate contains the 220-580 nm baseline",
        "cases": cases,
        "noise_level": "typical",
        "repeats_per_case": args.repeats,
        "dataset_count": len(rows),
        "seed_base": args.seed,
        "seed_policy": "original complete V12 case-index offset",
        "historical_pairing_guard": (
            "Filenames and seeds align with V12, but realization values are not assumed "
            "identical because the 200-800 nm arrays change RNG consumption. Pairing is "
            "strict only among bands masked from this one master dataset."
        ),
        "angle_mode": "fixed_independent_measurement",
        "angle_measurement_sigma_deg": 0.001,
        "dataset_index": str(index_path),
        "representative_plot": str(plot_path),
        "source_audit": {
            "main_v12": str(source_path),
            "main_v12_sha256": sha256(source_path),
            "main_v10_kernel": str(kernel_path),
            "main_v10_kernel_sha256": sha256(kernel_path),
        },
        "instrument_scope": (
            "V12 synthetic source envelope, quadratic QE, ILS, detector and ADC proxy; "
            "not measured hardware SPD/QE calibration"
        ),
        "failures": failures,
    }
    manifest_path = output / "simulation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "count": 100}, indent=2))


if __name__ == "__main__":
    main()
