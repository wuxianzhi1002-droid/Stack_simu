from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "nn_cavity_scalar_results_20260617_085429"
DEFAULT_NPZ = SCRIPT_DIR / "nn_cavity_scalar_dataset_all_2000.npz"
DEFAULT_SUMMARY = SCRIPT_DIR / "nn_cavity_scalar_dataset_all_2000_summary.json"
DEFAULT_CSV_SHARD_DIR = SCRIPT_DIR / "csv_shards"

PER_SAMPLE_KEYS = [
    "sample_id",
    "process_id",
    "nominal_stack_id",
    "split_id",
    "cavity_true_um",
    "L_fft_um",
    "delta_L_um",
    "delta_L_nm",
    "H_peak",
    "peak_count",
    "film_nominal_nm",
    "film_delta_nm",
    "film_true_nm",
    "valid_mask",
]
OUTPUT_LAYER_ORDER = ["PSS", "HSQ", "SOC", "TiO2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge scalar process checkpoints into one NPZ and CSV shards.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Checkpoint run directory.")
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_NPZ, help="Merged NPZ output path.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Summary JSON output path.")
    parser.add_argument("--csv-shard-dir", type=Path, default=DEFAULT_CSV_SHARD_DIR, help="CSV shard directory.")
    parser.add_argument("--expected-processes", type=int, default=2000, help="Expected process count.")
    parser.add_argument("--csv-processes-per-shard", type=int, default=200, help="Processes per CSV shard.")
    parser.add_argument("--skip-csv", action="store_true", help="Only write merged NPZ and summary.")
    parser.add_argument("--skip-npz", action="store_true", help="Only write CSV shards and summary.")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "00_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def checkpoint_path(run_dir: Path, process_id: int) -> Path:
    return run_dir / f"checkpoint_process_{process_id:04d}.npz"


def verify_checkpoints(run_dir: Path, expected_processes: int) -> list[Path]:
    paths = [checkpoint_path(run_dir, pid) for pid in range(expected_processes)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} checkpoint files. First missing:\n{preview}")
    return paths


def make_axes_from_config(config: dict, first_cavity_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cavity_axis_um = first_cavity_axis.copy()
    if config:
        cavity_start = float(config.get("cavity_start_um", cavity_axis_um[0]))
        cavity_step = float(config.get("cavity_step_um", cavity_axis_um[1] - cavity_axis_um[0]))
        cavity_points = int(config.get("num_cavity_points", len(cavity_axis_um)))
        cavity_axis_um = cavity_start + cavity_step * np.arange(cavity_points, dtype=np.float64)

    if config:
        wavelength_start = float(config.get("wavelength_start_um", 0.2))
        wavelength_stop = float(config.get("wavelength_stop_um", 0.6))
        spectral_resolution_um = float(config.get("spectral_resolution_nm", 0.02)) * 1e-3
        wavelength_points = int(round((wavelength_stop - wavelength_start) / spectral_resolution_um)) + 1
        wavelengths_um = wavelength_start + spectral_resolution_um * np.arange(wavelength_points, dtype=np.float64)
    else:
        wavelengths_um = np.array([], dtype=np.float64)

    return cavity_axis_um, wavelengths_um


def nominal_stack_names(config: dict) -> np.ndarray:
    stacks = config.get("nominal_stacks_nm", []) if config else []
    names = [str(stack.get("name", f"model_{idx:03d}")) for idx, stack in enumerate(stacks)]
    return np.array(names, dtype="<U128")


def allocate_arrays(first_data: np.lib.npyio.NpzFile, total_rows: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    rows_per_checkpoint = int(first_data["process_id"].shape[0])
    for key in PER_SAMPLE_KEYS:
        arr = first_data[key]
        tail_shape = arr.shape[1:]
        arrays[key] = np.empty((total_rows, *tail_shape), dtype=arr.dtype)
    return arrays


def layer_reindex(layer_names: list[str]) -> list[int]:
    layer_index = {name: idx for idx, name in enumerate(layer_names)}
    missing = [name for name in OUTPUT_LAYER_ORDER if name not in layer_index]
    if missing:
        raise ValueError(f"Checkpoint layers are missing {missing}; got {layer_names}")
    return [layer_index[name] for name in OUTPUT_LAYER_ORDER]


def fill_arrays(paths: list[Path], total_rows: int) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(paths[0], allow_pickle=True) as first:
        rows_per_checkpoint = int(first["process_id"].shape[0])
        arrays = allocate_arrays(first, total_rows)
        split_names = np.asarray(first["split_names"])
        source_layer_names = [str(x) for x in first["layer_names"].tolist()]
        reorder_idx = layer_reindex(source_layer_names)

    spectra_saved_values: set[bool] = set()
    source_layer_orders: dict[str, int] = {}
    row_start = 0

    for index, path in enumerate(paths):
        process_id = index
        with np.load(path, allow_pickle=True) as data:
            n = int(data["process_id"].shape[0])
            row_stop = row_start + n
            if row_stop > total_rows:
                raise ValueError(f"Too many rows while reading {path.name}")
            if not np.all(data["process_id"] == process_id):
                raise ValueError(f"Unexpected process_id values in {path.name}")

            current_layer_names = [str(x) for x in data["layer_names"].tolist()]
            source_layer_orders[",".join(current_layer_names)] = source_layer_orders.get(",".join(current_layer_names), 0) + 1
            current_reorder_idx = layer_reindex(current_layer_names)

            for key in PER_SAMPLE_KEYS:
                values = data[key]
                if key.startswith("film_"):
                    values = values[:, current_reorder_idx]
                arrays[key][row_start:row_stop] = values

            spectra_saved_values.add(bool(data["spectra_saved"].item()))
            if index % 100 == 0 or index == len(paths) - 1:
                print(f"Loaded checkpoint {index:04d}/{len(paths) - 1:04d}")
            row_start = row_stop

    if row_start != total_rows:
        raise ValueError(f"Expected {total_rows} rows but loaded {row_start}")

    metadata = {
        "rows_per_checkpoint": rows_per_checkpoint,
        "split_names": split_names,
        "source_layer_names": np.array(source_layer_names, dtype="<U16"),
        "layer_names": np.array(OUTPUT_LAYER_ORDER, dtype="<U16"),
        "source_layer_orders": source_layer_orders,
        "spectra_saved": np.array(any(spectra_saved_values), dtype=bool),
    }
    return arrays, metadata


def split_counts(split_id: np.ndarray, split_names: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for idx, name in enumerate(split_names.tolist()):
        result[str(name)] = int(np.count_nonzero(split_id == idx))
    return result


def write_npz(
    out_npz: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    config: dict,
    run_dir: Path,
) -> None:
    cavity_axis_um, wavelengths_um = make_axes_from_config(config, arrays["cavity_true_um"][: metadata["rows_per_checkpoint"]])
    config_json = json.dumps(config, ensure_ascii=False, indent=2) if config else ""
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing merged NPZ: {out_npz}")
    np.savez_compressed(
        out_npz,
        **arrays,
        split_names=metadata["split_names"],
        layer_names=metadata["layer_names"],
        source_layer_names=metadata["source_layer_names"],
        spectra_saved=metadata["spectra_saved"],
        cavity_axis_um=cavity_axis_um,
        wavelengths_um=wavelengths_um,
        nominal_stack_name_by_id=nominal_stack_names(config),
        source_run_dir=np.array(str(run_dir), dtype="<U512"),
        config_json=np.array(config_json),
        schema_version=np.array("scalar_checkpoint_merge_v1"),
    )


def csv_dataframe_for_slice(arrays: dict[str, np.ndarray], metadata: dict, start: int, stop: int) -> pd.DataFrame:
    split_names = [str(x) for x in metadata["split_names"].tolist()]
    split_id = arrays["split_id"][start:stop].astype(np.int16, copy=False)
    split_name = np.array([split_names[int(x)] for x in split_id], dtype=object)
    row_ids = np.arange(start, stop, dtype=np.int64)

    data: dict[str, np.ndarray] = {
        "row_id": row_ids,
        "sample_id": arrays["sample_id"][start:stop],
        "process_id": arrays["process_id"][start:stop],
        "nominal_stack_id": arrays["nominal_stack_id"][start:stop],
        "split_id": arrays["split_id"][start:stop],
        "split_name": split_name,
        "cavity_true_um": arrays["cavity_true_um"][start:stop],
        "L_fft_um": arrays["L_fft_um"][start:stop],
        "delta_L_um": arrays["delta_L_um"][start:stop],
        "delta_L_nm": arrays["delta_L_nm"][start:stop],
        "H_peak": arrays["H_peak"][start:stop],
        "peak_count": arrays["peak_count"][start:stop],
        "valid_mask": arrays["valid_mask"][start:stop].astype(np.int8),
    }

    for film_key, suffix in [
        ("film_nominal_nm", "nominal_nm"),
        ("film_delta_nm", "delta_nm"),
        ("film_true_nm", "true_nm"),
    ]:
        values = arrays[film_key][start:stop]
        for layer_index, layer_name in enumerate(OUTPUT_LAYER_ORDER):
            data[f"{layer_name}_{suffix}"] = values[:, layer_index]

    return pd.DataFrame(data)


def write_csv_shards(
    csv_shard_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    expected_processes: int,
    processes_per_shard: int,
) -> list[dict]:
    csv_shard_dir.mkdir(parents=True, exist_ok=True)
    rows_per_process = int(metadata["rows_per_checkpoint"])
    shard_infos: list[dict] = []

    for start_process in range(0, expected_processes, processes_per_shard):
        stop_process = min(expected_processes - 1, start_process + processes_per_shard - 1)
        start_row = start_process * rows_per_process
        stop_row = (stop_process + 1) * rows_per_process
        shard_path = csv_shard_dir / f"scalar_results_process_{start_process:04d}_{stop_process:04d}.csv"

        print(f"Writing CSV shard: {shard_path.name}")
        df = csv_dataframe_for_slice(arrays, metadata, start_row, stop_row)
        df.to_csv(shard_path, index=False, encoding="utf-8-sig", float_format="%.9g")
        shard_infos.append(
            {
                "path": str(shard_path),
                "process_start": start_process,
                "process_stop_inclusive": stop_process,
                "rows": int(stop_row - start_row),
                "size_bytes": int(shard_path.stat().st_size),
            }
        )

    return shard_infos


def write_summary(
    summary_path: Path,
    summary: dict,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.time()
    run_dir = args.run_dir.resolve()
    out_npz = args.out_npz.resolve()
    summary_path = args.summary.resolve()
    csv_shard_dir = args.csv_shard_dir.resolve()

    config = load_config(run_dir)
    expected_processes = int(args.expected_processes)
    paths = verify_checkpoints(run_dir, expected_processes)

    with np.load(paths[0], allow_pickle=True) as first:
        rows_per_checkpoint = int(first["process_id"].shape[0])
    total_rows = rows_per_checkpoint * expected_processes

    print(f"Run dir: {run_dir}")
    print(f"Checkpoints: {expected_processes}")
    print(f"Rows per checkpoint: {rows_per_checkpoint}")
    print(f"Total rows: {total_rows:,}")

    arrays, metadata = fill_arrays(paths, total_rows)

    if not args.skip_npz:
        write_npz(out_npz, arrays, metadata, config, run_dir)

    shard_infos: list[dict] = []
    if not args.skip_csv:
        shard_infos = write_csv_shards(
            csv_shard_dir,
            arrays,
            metadata,
            expected_processes,
            int(args.csv_processes_per_shard),
        )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "source_run_dir": str(run_dir),
        "checkpoint_count": expected_processes,
        "process_start": 0,
        "process_stop_inclusive": expected_processes - 1,
        "rows_per_process": rows_per_checkpoint,
        "rows": total_rows,
        "valid_rows": int(np.count_nonzero(arrays["valid_mask"])),
        "invalid_rows": int(total_rows - np.count_nonzero(arrays["valid_mask"])),
        "split_counts": split_counts(arrays["split_id"], metadata["split_names"]),
        "layer_order": OUTPUT_LAYER_ORDER,
        "source_layer_orders": metadata["source_layer_orders"],
        "npz_path": str(out_npz) if not args.skip_npz else None,
        "npz_size_bytes": int(out_npz.stat().st_size) if (not args.skip_npz and out_npz.exists()) else None,
        "csv_shard_dir": str(csv_shard_dir) if not args.skip_csv else None,
        "csv_shards": shard_infos,
        "spectra_saved": bool(metadata["spectra_saved"].item()),
    }
    write_summary(summary_path, summary)
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
