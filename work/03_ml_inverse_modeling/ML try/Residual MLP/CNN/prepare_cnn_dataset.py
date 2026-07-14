"""Prepare a memory-mappable CNN dataset from a final NPZ or checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SPECTRA_SOURCE_FIELD = "spectra_norm_ds"
SPECTRA_OUTPUT_NAME = "spectra_norm.npy"
SCALAR_OUTPUT_NAME = "scalar_fields.npz"
MANIFEST_OUTPUT_NAME = "manifest.json"

ROW_FIELDS = [
    "sample_id",
    "process_id",
    "nominal_stack_id",
    "split_id",
    "valid_mask",
    "L_fft_um",
    "delta_L_nm",
    "cavity_true_um",
    "film_nominal_nm",
    "H_peak",
]

OPTIONAL_METADATA_FIELDS = [
    "split_names",
    "layer_names",
    "wavelengths_spectra_saved_um",
    "spectra_downsample_factor",
    "spectra_downsample_method",
    "spectrum_normalization",
    "spectra_norm_method",
]


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_source() -> Path:
    ml_try_dir = script_dir().parents[1]
    return ml_try_dir / "nn_cavity_spectral_features_20260620_233057"


def ensure_output_is_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_names = {SPECTRA_OUTPUT_NAME, SCALAR_OUTPUT_NAME, MANIFEST_OUTPUT_NAME}
    existing = [output_dir / name for name in managed_names if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Prepared files already exist in {output_dir}: {names}. "
            "Use --overwrite true to replace them."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def validate_row_fields(arrays: Dict[str, np.ndarray], expected_rows: int) -> None:
    missing = [name for name in ROW_FIELDS if name not in arrays]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    for name in ROW_FIELDS:
        if arrays[name].shape[0] != expected_rows:
            raise ValueError(
                f"Field {name!r} has {arrays[name].shape[0]} rows; "
                f"expected {expected_rows}."
            )


def process_split_summary(
    process_id: np.ndarray,
    split_id: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[bool, Dict[str, int], List[int]]:
    valid = np.asarray(valid_mask, dtype=bool)
    process = np.asarray(process_id)[valid]
    split = np.asarray(split_id)[valid]
    split_counts = {
        str(split_value): int(np.count_nonzero(split == split_value))
        for split_value in (0, 1, 2)
    }
    conflicting: List[int] = []
    if len(process):
        order = np.argsort(process, kind="stable")
        process_sorted = process[order]
        split_sorted = split[order]
        boundaries = np.flatnonzero(np.diff(process_sorted)) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, len(process_sorted)]
        conflicting = [
            int(process_sorted[start])
            for start, stop in zip(starts, stops)
            if len(np.unique(split_sorted[start:stop])) != 1
        ]
    return not conflicting, split_counts, conflicting[:20]


def load_scalar_fields_from_npz(
    source: Path,
    expected_rows: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    row_arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, np.ndarray] = {}
    with np.load(source, allow_pickle=True) as data:
        for name in ROW_FIELDS:
            if name not in data.files:
                raise KeyError(f"{source} does not contain required field {name!r}")
            row_arrays[name] = data[name]
        for name in OPTIONAL_METADATA_FIELDS:
            if name in data.files:
                metadata[name] = data[name]
    validate_row_fields(row_arrays, expected_rows)
    return row_arrays, metadata


def extract_spectra_from_final_npz(source: Path, output_path: Path) -> np.memmap:
    member_name = f"{SPECTRA_SOURCE_FIELD}.npy"
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(source, "r") as archive:
        if member_name not in archive.namelist():
            raise KeyError(f"{source} does not contain {member_name}")
        with archive.open(member_name, "r") as src, temp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    temp_path.replace(output_path)
    spectra = np.load(output_path, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D spectra, got shape {spectra.shape}")
    return spectra


def prepare_from_final_npz(
    source: Path,
    output_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    spectra_path = output_dir / SPECTRA_OUTPUT_NAME
    spectra = extract_spectra_from_final_npz(source, spectra_path)
    row_arrays, metadata = load_scalar_fields_from_npz(source, int(spectra.shape[0]))
    source_info = {
        "source_type": "final_npz",
        "source_files_scanned": 1,
        "source_files_used": 1,
        "spectra_shape": list(spectra.shape),
        "spectra_dtype": str(spectra.dtype),
    }
    del spectra
    return row_arrays, metadata, source_info


def checkpoint_files(source: Path) -> List[Path]:
    files = sorted(source.glob("checkpoint_process_*.npz"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_process_*.npz files found in {source}")
    return files


def inspect_checkpoints(files: List[Path]) -> Tuple[int, int, np.dtype]:
    total_rows = 0
    spectra_points: Optional[int] = None
    spectra_dtype: Optional[np.dtype] = None
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=True) as data:
            rows = int(data["sample_id"].shape[0])
            total_rows += rows
            if index == 0:
                spectra = data[SPECTRA_SOURCE_FIELD]
                if spectra.ndim != 2:
                    raise ValueError(f"Expected 2D spectra in {path}, got {spectra.shape}")
                spectra_points = int(spectra.shape[1])
                spectra_dtype = spectra.dtype
    if spectra_points is None or spectra_dtype is None:
        raise RuntimeError("Could not inspect checkpoint spectra")
    return total_rows, spectra_points, spectra_dtype


def prepare_from_checkpoints(
    source: Path,
    output_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    files = checkpoint_files(source)
    total_rows, spectra_points, spectra_dtype = inspect_checkpoints(files)
    spectra_path = output_dir / SPECTRA_OUTPUT_NAME
    spectra = np.lib.format.open_memmap(
        spectra_path,
        mode="w+",
        dtype=spectra_dtype,
        shape=(total_rows, spectra_points),
    )
    row_chunks: Dict[str, List[np.ndarray]] = {name: [] for name in ROW_FIELDS}
    metadata: Dict[str, np.ndarray] = {}
    cursor = 0

    for file_index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=True) as data:
            rows = int(data["sample_id"].shape[0])
            stop = cursor + rows
            chunk = data[SPECTRA_SOURCE_FIELD]
            if chunk.shape != (rows, spectra_points):
                raise ValueError(
                    f"Inconsistent spectra shape in {path}: {chunk.shape}; "
                    f"expected {(rows, spectra_points)}"
                )
            spectra[cursor:stop] = chunk
            for name in ROW_FIELDS:
                if name not in data.files:
                    raise KeyError(f"{path} does not contain required field {name!r}")
                row_chunks[name].append(data[name])
            if not metadata:
                for name in OPTIONAL_METADATA_FIELDS:
                    if name in data.files:
                        metadata[name] = data[name]
            cursor = stop
        if file_index % 100 == 0 or file_index == len(files):
            spectra.flush()
            print(f"prepared checkpoints: {file_index}/{len(files)}, rows={cursor}")

    del spectra
    row_arrays = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in row_chunks.items()
    }
    validate_row_fields(row_arrays, total_rows)
    source_info = {
        "source_type": "checkpoint_directory",
        "source_files_scanned": len(files),
        "source_files_used": len(files),
        "spectra_shape": [total_rows, spectra_points],
        "spectra_dtype": str(spectra_dtype),
    }
    return row_arrays, metadata, source_info


def save_scalar_fields(
    output_dir: Path,
    row_arrays: Dict[str, np.ndarray],
    metadata: Dict[str, np.ndarray],
) -> None:
    output_path = output_dir / SCALAR_OUTPUT_NAME
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    payload = {**row_arrays, **metadata}
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temp_path.replace(output_path)


def prepare(source: Path, output_dir: Path, overwrite: bool) -> Dict[str, object]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    ensure_output_is_available(output_dir, overwrite)

    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Final dataset must be an NPZ file: {source}")
        row_arrays, metadata, source_info = prepare_from_final_npz(source, output_dir)
    elif source.is_dir():
        row_arrays, metadata, source_info = prepare_from_checkpoints(source, output_dir)
    else:
        raise FileNotFoundError(source)

    save_scalar_fields(output_dir, row_arrays, metadata)
    split_by_process, split_counts, conflicting = process_split_summary(
        row_arrays["process_id"],
        row_arrays["split_id"],
        row_arrays["valid_mask"],
    )
    manifest: Dict[str, object] = {
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "output_dir": str(output_dir),
        "spectra_file": SPECTRA_OUTPUT_NAME,
        "scalar_file": SCALAR_OUTPUT_NAME,
        "spectra_source_field": SPECTRA_SOURCE_FIELD,
        "row_count": int(len(row_arrays["sample_id"])),
        "valid_row_count": int(np.count_nonzero(row_arrays["valid_mask"])),
        "split_valid_row_counts": split_counts,
        "split_by_process": split_by_process,
        "conflicting_process_ids": conflicting,
        "row_fields": ROW_FIELDS,
        "metadata_fields": sorted(metadata),
        **source_info,
    }
    (output_dir / MANIFEST_OUTPUT_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source(),
        help="Final NPZ file or directory containing checkpoint_process_*.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir() / "cnn_dataset",
        help="Prepared dataset folder.",
    )
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    manifest = prepare(args.source, args.output, args.overwrite)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
