from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ============================================================================
# Windows CMD examples
# ============================================================================
#
# Step 1: extract train-only PCA features
#
# python "01_Lumerical_Workflow\ML try\Residual MLP\extract_pca_features.py" ^
#   --dataset "01_Lumerical_Workflow\ML try\nn_cavity_spectral_features_20260620_233057\nn_cavity_spectral_features_20260620_233057.npz" ^
#   --n-components 100 ^
#   --max-pca-fit-rows 100000 ^
#   --batch-size 20000
#
# Step 2: train with PCA50
#
# python "01_Lumerical_Workflow\ML try\Residual MLP\train_residual_mlp.py" ^
#   --dataset "01_Lumerical_Workflow\ML try\nn_cavity_spectral_features_20260620_233057\pca_features\nn_cavity_pca_features_100_YYYYMMDD_HHMMSS.npz" ^
#   --feature-groups l_fft_only nominal_thickness pca_scores ^
#   --pca-components 50 ^
#   --epochs 120
#
# Step 3: train with PCA50 plus robust spectral features
#
# python "01_Lumerical_Workflow\ML try\Residual MLP\train_residual_mlp.py" ^
#   --dataset "01_Lumerical_Workflow\ML try\nn_cavity_spectral_features_20260620_233057\pca_features\nn_cavity_pca_features_100_YYYYMMDD_HHMMSS.npz" ^
#   --feature-groups l_fft_only nominal_thickness pca_scores spectral_features_full ^
#   --pca-components 50 ^
#   --spectral-feature-preset robust ^
#   --epochs 120
#
# ============================================================================


COPY_FIELDS = [
    "sample_id",
    "process_id",
    "nominal_stack_id",
    "valid_mask",
    "L_fft_um",
    "H_peak",
    "peak_count",
    "delta_L_nm",
    "delta_L_um",
    "cavity_true_um",
    "L_true_um",
    "film_nominal_nm",
    "film_delta_nm",
    "film_true_nm",
    "layer_names",
    "spectral_features_full",
    "spectral_feature_names",
    "spectral_feature_source",
    "wavelengths_spectra_saved_um",
    "nominal_stack_name_by_id",
    "nominal_stack_values_nm",
    "process_nominal_stack_id",
    "process_film_delta_nm",
    "process_film_true_nm",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PCA on train spectra only and save lightweight PCA features."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Source spectral .npz file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_dir>/pca_features",
    )
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--max-pca-fit-rows", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=20260613)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument(
        "--pca-method",
        choices=["randomized", "full", "auto"],
        default="randomized",
    )
    parser.add_argument(
        "--overwrite",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Allow replacing output files with the same timestamped names.",
    )
    return parser.parse_args()


def require_fields(data: np.lib.npyio.NpzFile, fields: list[str]) -> None:
    missing = [field for field in fields if field not in data.files]
    if missing:
        raise KeyError(f"Source dataset is missing required fields: {missing}")


def build_split_info(
    data: np.lib.npyio.NpzFile,
    num_samples: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    process_id = data["process_id"]
    if process_id.shape != (num_samples,):
        raise ValueError(
            f"process_id shape mismatch: expected {(num_samples,)}, got {process_id.shape}"
        )

    valid_mask = (
        data["valid_mask"].astype(bool)
        if "valid_mask" in data.files
        else np.ones(num_samples, dtype=bool)
    )
    if valid_mask.shape != (num_samples,):
        raise ValueError(
            f"valid_mask shape mismatch: expected {(num_samples,)}, got {valid_mask.shape}"
        )

    if "split_id" in data.files:
        split_id = data["split_id"].astype(np.int8, copy=False)
        if split_id.shape != (num_samples,):
            raise ValueError(
                f"split_id shape mismatch: expected {(num_samples,)}, got {split_id.shape}"
            )
    else:
        require_fields(
            data,
            ["train_process_ids", "val_process_ids", "test_process_ids"],
        )
        split_id = np.full(num_samples, -1, dtype=np.int8)
        split_id[np.isin(process_id, data["train_process_ids"])] = 0
        split_id[np.isin(process_id, data["val_process_ids"])] = 1
        split_id[np.isin(process_id, data["test_process_ids"])] = 2

    split_names = {"train": 0, "val": 1, "test": 2}
    indices = {
        name: np.flatnonzero(valid_mask & (split_id == split_value))
        for name, split_value in split_names.items()
    }
    empty = [name for name, rows in indices.items() if len(rows) == 0]
    if empty:
        raise ValueError(f"No valid rows found for split(s): {empty}")

    process_sets = {
        name: set(np.unique(process_id[rows]).tolist())
        for name, rows in indices.items()
    }
    overlap_train_val = len(process_sets["train"] & process_sets["val"])
    overlap_train_test = len(process_sets["train"] & process_sets["test"])
    overlap_val_test = len(process_sets["val"] & process_sets["test"])
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            "Process leakage detected between splits: "
            f"train/val={overlap_train_val}, "
            f"train/test={overlap_train_test}, "
            f"val/test={overlap_val_test}"
        )

    process_ids = {
        name: np.asarray(sorted(process_sets[name]), dtype=process_id.dtype)
        for name in split_names
    }
    stats = {
        "train_rows": int(len(indices["train"])),
        "val_rows": int(len(indices["val"])),
        "test_rows": int(len(indices["test"])),
        "train_process_count": int(len(process_sets["train"])),
        "val_process_count": int(len(process_sets["val"])),
        "test_process_count": int(len(process_sets["test"])),
        "process_overlap_train_val": overlap_train_val,
        "process_overlap_train_test": overlap_train_test,
        "process_overlap_val_test": overlap_val_test,
    }
    return split_id, process_ids, stats


def choose_fit_indices(
    train_indices: np.ndarray,
    max_rows: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or len(train_indices) <= max_rows:
        return train_indices.copy()
    sampled = rng.choice(train_indices, size=max_rows, replace=False)
    return np.sort(sampled)


def cumulative_ratio_at(
    cumulative_ratio: np.ndarray,
    component_count: int,
) -> float | None:
    if len(cumulative_ratio) < component_count:
        return None
    return float(cumulative_ratio[component_count - 1])


def save_explained_variance_plot(
    path: Path,
    explained_variance_ratio: np.ndarray,
) -> None:
    component_index = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative_ratio = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        component_index,
        explained_variance_ratio,
        linewidth=1.5,
        label="individual explained variance ratio",
    )
    plt.plot(
        component_index,
        cumulative_ratio,
        linewidth=2.0,
        label="cumulative explained variance ratio",
    )
    plt.xlabel("principal component index")
    plt.ylabel("explained variance ratio")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def estimate_payload_size_gb(payload: dict[str, np.ndarray]) -> float:
    total_bytes = 0
    for value in payload.values():
        array = np.asarray(value)
        total_bytes += int(array.nbytes)
    return total_bytes / 1_000_000_000.0


def scalar_text(data: np.lib.npyio.NpzFile, field: str, default: str = "") -> str:
    if field not in data.files:
        return default
    value = data[field]
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return str(value.tolist())


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if args.n_components <= 0:
        raise ValueError("--n-components must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (dataset_path.parent / "pca_features").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_npz_path = output_dir / (
        f"nn_cavity_pca_features_{args.n_components}_{stamp}.npz"
    )
    summary_path = output_dir / f"pca_summary_{stamp}.json"
    plot_path = output_dir / f"pca_explained_variance_{stamp}.png"
    output_paths = [output_npz_path, summary_path, plot_path]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output file(s) already exist and --overwrite is false: "
            + ", ".join(existing)
        )

    score_memmap_path: Path | None = None
    scores: np.memmap | None = None

    print("Loading dataset...")
    with np.load(dataset_path, allow_pickle=True) as data:
        require_fields(data, ["spectra_norm_ds", "process_id", *COPY_FIELDS])
        spectra = data["spectra_norm_ds"]
        if spectra.ndim != 2:
            raise ValueError(f"spectra_norm_ds must be 2-D, got {spectra.shape}")
        if spectra.dtype != np.float32:
            print(
                f"Warning: spectra_norm_ds dtype is {spectra.dtype}; "
                "batches will be converted to float32."
            )

        num_samples, num_spectral_points = spectra.shape
        if args.n_components > min(num_samples, num_spectral_points):
            raise ValueError(
                f"--n-components={args.n_components} exceeds matrix limit "
                f"{min(num_samples, num_spectral_points)}."
            )

        print("Split check...")
        split_id, split_process_ids, split_stats = build_split_info(data, num_samples)
        for split_name in ["train", "val", "test"]:
            print(
                f"  {split_name}: rows={split_stats[f'{split_name}_rows']:,}, "
                f"processes={split_stats[f'{split_name}_process_count']:,}"
            )
        print(
            "  process overlap: "
            f"train/val={split_stats['process_overlap_train_val']}, "
            f"train/test={split_stats['process_overlap_train_test']}, "
            f"val/test={split_stats['process_overlap_val_test']}"
        )

        valid_mask = data["valid_mask"].astype(bool)
        train_indices = np.flatnonzero(valid_mask & (split_id == 0))
        rng = np.random.default_rng(args.random_seed)
        fit_indices = choose_fit_indices(
            train_indices,
            args.max_pca_fit_rows,
            rng,
        )
        if args.n_components > min(len(fit_indices), num_spectral_points):
            raise ValueError(
                f"--n-components={args.n_components} exceeds PCA fit matrix limit "
                f"{min(len(fit_indices), num_spectral_points)}."
            )

        print("Fitting PCA...")
        fit_start = time.perf_counter()
        fit_spectra = np.ascontiguousarray(
            spectra[fit_indices].astype(np.float32, copy=False)
        )
        finite_fit_rows = np.all(np.isfinite(fit_spectra), axis=1)
        if not np.all(finite_fit_rows):
            dropped = int(np.count_nonzero(~finite_fit_rows))
            print(f"  Dropping {dropped:,} non-finite train spectra from PCA fit.")
            fit_indices = fit_indices[finite_fit_rows]
            fit_spectra = fit_spectra[finite_fit_rows]
        if args.n_components > min(fit_spectra.shape):
            raise ValueError(
                f"Only {fit_spectra.shape[0]} finite PCA fit rows remain; "
                f"cannot fit {args.n_components} components."
            )

        pca = PCA(
            n_components=args.n_components,
            svd_solver=args.pca_method,
            random_state=args.random_seed,
        )
        pca.fit(fit_spectra)
        elapsed_fit = time.perf_counter() - fit_start
        del fit_spectra

        temp_file = tempfile.NamedTemporaryFile(
            prefix="pca_scores_",
            suffix=".dat",
            dir=output_dir,
            delete=False,
        )
        score_memmap_path = Path(temp_file.name)
        temp_file.close()
        scores = np.memmap(
            score_memmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, args.n_components),
        )
        scores[:] = np.nan

        print("Transforming PCA scores...")
        transform_start = time.perf_counter()
        num_batches = math.ceil(num_samples / args.batch_size)
        nonfinite_transform_rows = 0
        for batch_index, start in enumerate(
            range(0, num_samples, args.batch_size),
            start=1,
        ):
            end = min(start + args.batch_size, num_samples)
            print(f"Transforming batch {batch_index}/{num_batches}...")
            batch = spectra[start:end].astype(np.float32, copy=False)
            finite_rows = np.all(np.isfinite(batch), axis=1)
            if np.all(finite_rows):
                transformed = pca.transform(batch)
                scores[start:end] = transformed.astype(np.float32, copy=False)
            else:
                nonfinite_transform_rows += int(np.count_nonzero(~finite_rows))
                if np.any(finite_rows):
                    transformed = pca.transform(batch[finite_rows])
                    scores[start:end][finite_rows] = transformed.astype(
                        np.float32,
                        copy=False,
                    )
        scores.flush()
        elapsed_transform = time.perf_counter() - transform_start
        if nonfinite_transform_rows:
            print(
                f"  Warning: {nonfinite_transform_rows:,} rows contained non-finite "
                "spectra; their pca_scores remain NaN."
            )

        pca_components = pca.components_.astype(np.float32, copy=False)
        pca_mean = pca.mean_.astype(np.float32, copy=False)
        explained_variance = pca.explained_variance_.astype(np.float32, copy=False)
        explained_variance_ratio = pca.explained_variance_ratio_.astype(
            np.float32,
            copy=False,
        )
        singular_values = pca.singular_values_.astype(np.float32, copy=False)
        del spectra

        payload: dict[str, np.ndarray] = {
            "pca_scores": scores,
            "pca_components": pca_components,
            "pca_mean": pca_mean,
            "pca_explained_variance": explained_variance,
            "pca_explained_variance_ratio": explained_variance_ratio,
            "pca_singular_values": singular_values,
            "pca_n_components": np.asarray(args.n_components, dtype=np.int32),
            "pca_fit_sample_indices": fit_indices.astype(np.int64, copy=False),
            "pca_fit_policy": np.asarray("fit_on_train_only"),
            "pca_source": np.asarray("spectra_norm_ds"),
            "pca_method": np.asarray(args.pca_method),
            "max_pca_fit_rows": np.asarray(args.max_pca_fit_rows, dtype=np.int64),
            "random_seed": np.asarray(args.random_seed, dtype=np.int64),
            "split_id": split_id,
            "train_process_ids": split_process_ids["train"],
            "val_process_ids": split_process_ids["val"],
            "test_process_ids": split_process_ids["test"],
            "config_json": (
                data["config_json"]
                if "config_json" in data.files
                else np.asarray("")
            ),
            "timestamp_source_dataset": np.asarray(
                scalar_text(data, "timestamp")
            ),
            "source_dataset_path": np.asarray(str(dataset_path)),
        }
        for field in COPY_FIELDS:
            payload[field] = data[field]

        estimated_output_size_gb = estimate_payload_size_gb(payload)

        print("Saving PCA NPZ...")
        try:
            np.savez(output_npz_path, **payload)
        except Exception:
            payload.pop("pca_scores", None)
            del payload
            del scores
            scores = None
            if score_memmap_path is not None and score_memmap_path.exists():
                score_memmap_path.unlink()
            raise
        payload.pop("pca_scores", None)
        del payload

    if scores is not None:
        del scores
    if score_memmap_path is not None and score_memmap_path.exists():
        score_memmap_path.unlink()

    save_explained_variance_plot(plot_path, explained_variance_ratio)
    cumulative_ratio = np.cumsum(explained_variance_ratio, dtype=np.float64)
    elapsed_total = time.perf_counter() - total_start
    summary = {
        "source_dataset_path": str(dataset_path),
        "output_npz_path": str(output_npz_path),
        "num_samples": int(num_samples),
        "num_spectral_points": int(num_spectral_points),
        "n_components": int(args.n_components),
        "pca_scores_shape": [int(num_samples), int(args.n_components)],
        "pca_scores_dtype": "float32",
        **split_stats,
        "max_pca_fit_rows": int(args.max_pca_fit_rows),
        "actual_pca_fit_rows": int(len(fit_indices)),
        "pca_fit_policy": "fit_on_train_only",
        "pca_source": "spectra_norm_ds",
        "pca_method": args.pca_method,
        "random_seed": int(args.random_seed),
        "explained_variance_ratio_first_10": [
            float(value) for value in explained_variance_ratio[:10]
        ],
        "explained_variance_ratio_cumsum_20": cumulative_ratio_at(
            cumulative_ratio,
            20,
        ),
        "explained_variance_ratio_cumsum_50": cumulative_ratio_at(
            cumulative_ratio,
            50,
        ),
        "explained_variance_ratio_cumsum_100": cumulative_ratio_at(
            cumulative_ratio,
            100,
        ),
        "total_explained_variance_ratio": float(cumulative_ratio[-1]),
        "elapsed_seconds_fit": round(elapsed_fit, 3),
        "elapsed_seconds_transform": round(elapsed_transform, 3),
        "elapsed_seconds_total": round(elapsed_total, 3),
        "estimated_output_size_gb": round(estimated_output_size_gb, 6),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Summary saved: {summary_path}")
    print(f"Explained variance plot: {plot_path}")
    print(f"PCA feature NPZ: {output_npz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
