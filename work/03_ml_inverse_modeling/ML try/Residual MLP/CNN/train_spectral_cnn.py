"""Train formal 1D spectral CNN/ResNet models from a prepared CNN dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLIT_ID_TO_NAME = {0: "train", 1: "val", 2: "test"}
SPLIT_NAME_TO_ID = {value: key for key, value in SPLIT_ID_TO_NAME.items()}
PREPARED_SPECTRA_NAME = "spectra_norm.npy"
PREPARED_SCALAR_NAME = "scalar_fields.npz"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_dataset_dir() -> Path:
    return script_dir() / "cnn_dataset"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitArrays:
    spectra_path: Path
    row_indices: np.ndarray
    scalar_raw: np.ndarray
    y_nm: np.ndarray
    l_fft_um: np.ndarray
    cavity_true_um: np.ndarray
    sample_id: np.ndarray
    process_id: np.ndarray
    nominal_stack_id: np.ndarray


@dataclass
class LoadSummary:
    dataset_path: str
    spectra_points: int
    spectra_dtype: str
    total_rows: int
    total_valid_rows: int
    selected_rows: Dict[str, int]
    unique_processes: Dict[str, int]
    split_by_process: bool
    conflicting_process_ids: List[int]
    source_type: str
    use_hpeak: bool
    scalar_feature_names: List[str]
    notes: List[str]


class SpectralDataset(Dataset):
    """Index a read-only NPY memmap without copying the full spectra matrix."""

    def __init__(
        self,
        spectra_path: Path,
        row_indices: np.ndarray,
        scalar_scaled: np.ndarray,
        y_scaled: np.ndarray,
    ) -> None:
        self.spectra_path = Path(spectra_path)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.scalar_scaled = np.ascontiguousarray(
            scalar_scaled.astype(np.float32, copy=False)
        )
        self.y_scaled = np.ascontiguousarray(
            y_scaled.astype(np.float32, copy=False)
        ).reshape(-1, 1)
        self._spectra: Optional[np.ndarray] = None

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        state["_spectra"] = None
        return state

    def _spectra_array(self) -> np.ndarray:
        if self._spectra is None:
            self._spectra = np.load(self.spectra_path, mmap_mode="r")
        return self._spectra

    def __len__(self) -> int:
        return int(len(self.row_indices))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index = int(self.row_indices[index])
        spectrum = np.array(
            self._spectra_array()[source_index],
            dtype=np.float32,
            copy=True,
        )
        return (
            torch.from_numpy(spectrum).unsqueeze(0),
            torch.from_numpy(self.scalar_scaled[index]),
            torch.from_numpy(self.y_scaled[index]),
        )

    def __getitems__(
        self,
        indices: Sequence[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        local_indices = np.asarray(indices, dtype=np.int64)
        source_indices = self.row_indices[local_indices]
        spectra = np.array(
            self._spectra_array()[source_indices],
            dtype=np.float32,
            copy=True,
        )
        return [
            (
                torch.from_numpy(spectra[position]).unsqueeze(0),
                torch.from_numpy(self.scalar_scaled[local_index]),
                torch.from_numpy(self.y_scaled[local_index]),
            )
            for position, local_index in enumerate(local_indices)
        ]


class SpectralPooling(nn.Module):
    def __init__(
        self,
        mode: str,
        channels: int,
        feature_length: int,
        pooling_k: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.pooling_k = pooling_k
        if mode == "gap":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.output_dim = channels
        elif mode == "adaptive_k":
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(pooling_k),
                nn.Flatten(),
            )
            self.output_dim = channels * pooling_k
        elif mode == "flatten":
            self.pool = nn.Flatten()
            self.output_dim = channels * feature_length
        elif mode == "conv_reduce":
            reduced_channels = max(32, channels // 2)
            self.pool = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    groups=channels,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Conv1d(channels, reduced_channels, kernel_size=1),
                nn.BatchNorm1d(reduced_channels),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(pooling_k),
                nn.Flatten(),
            )
            self.output_dim = reduced_channels * pooling_k
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def infer_backbone_shape(
    backbone: nn.Module,
    input_length: int,
) -> Tuple[int, int]:
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        output = backbone(torch.zeros(1, 1, input_length))
    backbone.train(was_training)
    return int(output.shape[1]), int(output.shape[2])


class CNNSmallFusion(nn.Module):
    def __init__(
        self,
        input_length: int,
        pooling: str,
        pooling_k: int,
        scalar_dim: int,
    ) -> None:
        super().__init__()
        self.spec_backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        channels, feature_length = infer_backbone_shape(
            self.spec_backbone,
            input_length,
        )
        self.spec_pool = SpectralPooling(
            pooling,
            channels,
            feature_length,
            pooling_k,
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.spec_pool.output_dim + 32, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_spec: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        z_spec = self.spec_pool(self.spec_backbone(x_spec))
        z_scalar = self.scalar_branch(x_scalar)
        return self.head(torch.cat([z_spec, z_scalar], dim=1))


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class ResNet1DFusion(nn.Module):
    def __init__(
        self,
        input_length: int,
        pooling: str,
        pooling_k: int,
        scalar_dim: int,
    ) -> None:
        super().__init__()
        self.spec_backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResBlock1D(32, 32, stride=1),
            ResBlock1D(32, 32, stride=1),
            ResBlock1D(32, 64, stride=2),
            ResBlock1D(64, 64, stride=1),
            ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 128, stride=1),
            ResBlock1D(128, 256, stride=2),
            ResBlock1D(256, 256, stride=1),
        )
        channels, feature_length = infer_backbone_shape(
            self.spec_backbone,
            input_length,
        )
        self.spec_pool = SpectralPooling(
            pooling,
            channels,
            feature_length,
            pooling_k,
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.spec_pool.output_dim + 64, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_spec: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        z_spec = self.spec_pool(self.spec_backbone(x_spec))
        z_scalar = self.scalar_branch(x_scalar)
        return self.head(torch.cat([z_spec, z_scalar], dim=1))


def resolve_model_and_pooling(
    model_name: str,
    pooling: str,
) -> Tuple[str, str, str]:
    if model_name == "cnn_small_gap":
        return "cnn_small", "gap", "cnn_small_gap"
    if model_name == "resnet1d_gap":
        return "resnet1d", "gap", "resnet1d_gap"
    return model_name, pooling, f"{model_name}_{pooling}"


def build_model(
    model_name: str,
    pooling: str,
    pooling_k: int,
    input_length: int,
    scalar_dim: int,
) -> nn.Module:
    if model_name == "cnn_small":
        return CNNSmallFusion(
            input_length,
            pooling,
            pooling_k,
            scalar_dim,
        )
    if model_name == "resnet1d":
        return ResNet1DFusion(
            input_length,
            pooling,
            pooling_k,
            scalar_dim,
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(
    y_true_nm: np.ndarray,
    y_pred_nm: np.ndarray,
    l_fft_um: np.ndarray,
    cavity_true_um: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true_nm, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred_nm, dtype=np.float64).reshape(-1)
    l_fft = np.asarray(l_fft_um, dtype=np.float64).reshape(-1)
    cavity_true = np.asarray(cavity_true_um, dtype=np.float64).reshape(-1)
    delta_err = y_pred - y_true
    cavity_pred_um = l_fft + y_pred / 1000.0
    cavity_err_nm = (cavity_pred_um - cavity_true) * 1000.0
    return {
        "delta_MAE_nm": float(np.mean(np.abs(delta_err))),
        "delta_RMSE_nm": float(np.sqrt(np.mean(delta_err**2))),
        "delta_MaxAbs_nm": float(np.max(np.abs(delta_err))),
        "delta_P95Abs_nm": float(np.percentile(np.abs(delta_err), 95)),
        "delta_P99Abs_nm": float(np.percentile(np.abs(delta_err), 99)),
        "delta_Bias_nm": float(np.mean(delta_err)),
        "R2_delta": (
            float(r2_score(y_true, y_pred))
            if len(y_true) > 1
            else float("nan")
        ),
        "cavity_MAE_nm": float(np.mean(np.abs(cavity_err_nm))),
        "cavity_RMSE_nm": float(np.sqrt(np.mean(cavity_err_nm**2))),
        "cavity_MaxAbs_nm": float(np.max(np.abs(cavity_err_nm))),
    }


def process_split_consistency(
    process_id: np.ndarray,
    split_id: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[bool, List[int]]:
    valid = np.asarray(valid_mask, dtype=bool)
    process = np.asarray(process_id)[valid]
    split = np.asarray(split_id)[valid]
    if not len(process):
        return True, []
    order = np.argsort(process, kind="stable")
    process = process[order]
    split = split[order]
    boundaries = np.flatnonzero(np.diff(process)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(process)]
    conflicts = [
        int(process[start])
        for start, stop in zip(starts, stops)
        if len(np.unique(split[start:stop])) != 1
    ]
    return not conflicts, conflicts[:20]


def cap_indices(indices: np.ndarray, max_rows: Optional[int]) -> np.ndarray:
    if max_rows is None:
        return indices
    if max_rows <= 0:
        raise ValueError("max rows must be positive when provided")
    return indices[:max_rows]


def load_prepared_dataset(
    dataset_dir: Path,
    max_train_rows: Optional[int],
    max_val_rows: Optional[int],
    max_test_rows: Optional[int],
    use_hpeak: bool,
) -> Tuple[Dict[str, SplitArrays], LoadSummary]:
    spectra_path = dataset_dir / PREPARED_SPECTRA_NAME
    scalar_path = dataset_dir / PREPARED_SCALAR_NAME
    if not spectra_path.is_file() or not scalar_path.is_file():
        raise FileNotFoundError(
            f"{dataset_dir} is not a prepared CNN dataset. Expected "
            f"{PREPARED_SPECTRA_NAME} and {PREPARED_SCALAR_NAME}. "
            "Run prepare_cnn_dataset.py first."
        )

    spectra = np.load(spectra_path, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected spectra shape (N, L), got {spectra.shape}")
    total_rows = int(spectra.shape[0])
    spectra_points = int(spectra.shape[1])
    spectra_dtype = str(spectra.dtype)
    del spectra

    required = [
        "sample_id",
        "process_id",
        "nominal_stack_id",
        "split_id",
        "valid_mask",
        "L_fft_um",
        "delta_L_nm",
        "cavity_true_um",
        "film_nominal_nm",
    ]
    if use_hpeak:
        required.append("H_peak")

    arrays: Dict[str, np.ndarray] = {}
    with np.load(scalar_path, allow_pickle=True) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"Missing prepared scalar fields: {missing}")
        for name in required:
            arrays[name] = data[name]

    for name, array in arrays.items():
        if array.shape[0] != total_rows:
            raise ValueError(
                f"Field {name!r} has {array.shape[0]} rows but spectra has "
                f"{total_rows}."
            )

    film_nominal = np.asarray(arrays["film_nominal_nm"], dtype=np.float32)
    if film_nominal.ndim != 2:
        raise ValueError(
            f"film_nominal_nm must be 2D, got {film_nominal.shape}"
        )
    scalar_feature_names = ["L_fft_um"] + [
        f"film_nominal_nm[{index}]"
        for index in range(film_nominal.shape[1])
    ]
    scalar_columns = [
        np.asarray(arrays["L_fft_um"], dtype=np.float32).reshape(-1, 1),
        film_nominal,
    ]
    if use_hpeak:
        scalar_columns.append(
            np.asarray(arrays["H_peak"], dtype=np.float32).reshape(-1, 1)
        )
        scalar_feature_names.append("H_peak")
    scalar_all = np.column_stack(scalar_columns).astype(np.float32, copy=False)

    valid_mask = np.asarray(arrays["valid_mask"], dtype=bool)
    finite_required = (
        np.isfinite(scalar_all).all(axis=1)
        & np.isfinite(arrays["delta_L_nm"])
        & np.isfinite(arrays["L_fft_um"])
        & np.isfinite(arrays["cavity_true_um"])
    )
    usable_mask = valid_mask & finite_required
    split_id = np.asarray(arrays["split_id"])
    max_rows = {
        "train": max_train_rows,
        "val": max_val_rows,
        "test": max_test_rows,
    }
    splits: Dict[str, SplitArrays] = {}
    for split_name, split_value in SPLIT_NAME_TO_ID.items():
        indices = np.flatnonzero(usable_mask & (split_id == split_value))
        indices = cap_indices(indices, max_rows[split_name])
        if not len(indices):
            raise RuntimeError(f"No usable rows selected for {split_name}")
        splits[split_name] = SplitArrays(
            spectra_path=spectra_path,
            row_indices=indices.astype(np.int64, copy=False),
            scalar_raw=scalar_all[indices],
            y_nm=np.asarray(arrays["delta_L_nm"][indices], dtype=np.float32),
            l_fft_um=np.asarray(arrays["L_fft_um"][indices], dtype=np.float32),
            cavity_true_um=np.asarray(
                arrays["cavity_true_um"][indices],
                dtype=np.float64,
            ),
            sample_id=np.asarray(arrays["sample_id"][indices], dtype=np.int64),
            process_id=np.asarray(
                arrays["process_id"][indices],
                dtype=np.int32,
            ),
            nominal_stack_id=np.asarray(
                arrays["nominal_stack_id"][indices],
                dtype=np.int16,
            ),
        )

    split_by_process, conflicts = process_split_consistency(
        arrays["process_id"],
        split_id,
        valid_mask,
    )
    source_type = "prepared_dataset"
    manifest_path = dataset_dir / "manifest.json"
    notes: List[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_type = str(manifest.get("source_type", source_type))
    dropped_nonfinite = int(np.count_nonzero(valid_mask & ~finite_required))
    if dropped_nonfinite:
        notes.append(
            f"Dropped {dropped_nonfinite} valid_mask rows with non-finite "
            "required model fields."
        )

    summary = LoadSummary(
        dataset_path=str(dataset_dir),
        spectra_points=spectra_points,
        spectra_dtype=spectra_dtype,
        total_rows=total_rows,
        total_valid_rows=int(np.count_nonzero(valid_mask)),
        selected_rows={
            name: int(len(split.row_indices))
            for name, split in splits.items()
        },
        unique_processes={
            name: int(len(np.unique(split.process_id)))
            for name, split in splits.items()
        },
        split_by_process=split_by_process,
        conflicting_process_ids=conflicts,
        source_type=source_type,
        use_hpeak=use_hpeak,
        scalar_feature_names=scalar_feature_names,
        notes=notes,
    )
    return splits, summary


def make_data_loaders(
    splits: Dict[str, SplitArrays],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[
    Dict[str, DataLoader],
    StandardScaler,
    float,
    float,
    Dict[str, np.ndarray],
]:
    scalar_scaler = StandardScaler()
    scalar_scaler.fit(splits["train"].scalar_raw)
    y_mean = float(np.mean(splits["train"].y_nm))
    y_std = float(np.std(splits["train"].y_nm))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    scalar_scaled = {
        name: scalar_scaler.transform(split.scalar_raw).astype(np.float32)
        for name, split in splits.items()
    }
    y_scaled = {
        name: ((split.y_nm - y_mean) / y_std).astype(np.float32)
        for name, split in splits.items()
    }
    datasets = {
        name: SpectralDataset(
            split.spectra_path,
            split.row_indices,
            scalar_scaled[name],
            y_scaled[name],
        )
        for name, split in splits.items()
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, scalar_scaler, y_mean, y_std, scalar_scaled


def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for spectra, scalar, _target in loader:
            prediction = model(
                spectra.to(device, non_blocking=True),
                scalar.to(device, non_blocking=True),
            )
            predictions.append(prediction.detach().cpu().numpy().reshape(-1))
    if not predictions:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(predictions) * y_std + y_mean


def average_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for spectra, scalar, target in loader:
            spectra = spectra.to(device, non_blocking=True)
            scalar = scalar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = criterion(model(spectra, scalar), target)
            total += float(loss.item()) * len(spectra)
            count += len(spectra)
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> Tuple[
    List[Dict[str, float]],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    if args.loss == "smooth_l1":
        criterion: nn.Module = nn.SmoothL1Loss(beta=args.huber_beta)
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.lr_patience),
    )
    best_metric = math.inf
    best_state: Dict[str, torch.Tensor] = {}
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        epoch_start = time.time()
        for spectra, scalar, target in loaders["train"]:
            spectra = spectra.to(device, non_blocking=True)
            scalar = scalar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                prediction = model(spectra, scalar)
                loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.item()) * len(spectra)
            train_count += len(spectra)

        train_loss = train_total / max(train_count, 1)
        val_loss = average_loss(model, loaders["val"], criterion, device)
        val_pred_nm = evaluate_predictions(
            model,
            loaders["val"],
            device,
            y_mean,
            y_std,
        )
        val_dataset = loaders["val"].dataset
        val_true_nm = val_dataset.y_scaled.reshape(-1) * y_std + y_mean
        val_rmse_nm = float(
            np.sqrt(np.mean((val_pred_nm - val_true_nm) ** 2))
        )
        scheduler.step(val_rmse_nm)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_RMSE_nm": val_rmse_nm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_RMSE_nm={val_rmse_nm:.4f} "
            f"lr={row['learning_rate']:.2e}"
        )

        if val_rmse_nm < best_metric - 1e-8:
            best_metric = val_rmse_nm
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            args.early_stopping_patience > 0
            and stale_epochs >= args.early_stopping_patience
        ):
            print(f"early stopping at epoch {epoch}")
            break

    last_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    if best_state:
        model.load_state_dict(best_state)
    return history, best_state, last_state


def save_training_log(
    history: List[Dict[str, float]],
    output_dir: Path,
) -> None:
    if not history:
        return
    with (output_dir / "training_log.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def run_baselines(
    splits: Dict[str, SplitArrays],
    scalar_scaled: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}
    fft_zero = np.zeros_like(splits["test"].y_nm, dtype=np.float32)
    predictions["fft_only"] = fft_zero
    metrics["fft_only"] = compute_metrics(
        splits["test"].y_nm,
        fft_zero,
        splits["test"].l_fft_um,
        splits["test"].cavity_true_um,
    )
    ridge = Ridge(alpha=1.0)
    ridge.fit(scalar_scaled["train"], splits["train"].y_nm)
    ridge_pred = ridge.predict(scalar_scaled["test"]).astype(np.float32)
    predictions["scalar_ridge"] = ridge_pred
    metrics["scalar_ridge"] = compute_metrics(
        splits["test"].y_nm,
        ridge_pred,
        splits["test"].l_fft_um,
        splits["test"].cavity_true_um,
    )
    return metrics, predictions


def save_predictions(
    output_dir: Path,
    test_split: SplitArrays,
    cnn_pred_nm: np.ndarray,
    baseline_preds: Dict[str, np.ndarray],
) -> None:
    data = {
        "sample_id": test_split.sample_id,
        "process_id": test_split.process_id,
        "nominal_stack_id": test_split.nominal_stack_id,
        "L_fft_um": test_split.l_fft_um,
        "cavity_true_um": test_split.cavity_true_um,
        "delta_true_nm": test_split.y_nm,
        "delta_pred_cnn_nm": cnn_pred_nm,
    }
    for name, prediction in baseline_preds.items():
        data[f"delta_pred_{name}_nm"] = prediction
    pd.DataFrame(data).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )


def plot_curves(
    history: List[Dict[str, float]],
    output_dir: Path,
) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(
        epochs,
        [row["train_loss"] for row in history],
        label="train_loss",
    )
    plt.plot(
        epochs,
        [row["val_loss"] for row in history],
        label="val_loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "01_loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        epochs,
        [row["val_RMSE_nm"] for row in history],
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Val RMSE (nm)")
    plt.tight_layout()
    plt.savefig(output_dir / "02_val_rmse_curve.png", dpi=160)
    plt.close()


def plot_test_diagnostics(
    output_dir: Path,
    test_split: SplitArrays,
    cnn_pred_nm: np.ndarray,
    all_metrics: Dict[str, Dict[str, float]],
) -> None:
    true = test_split.y_nm
    error = cnn_pred_nm - true
    plt.figure(figsize=(5, 5))
    plt.scatter(true, cnn_pred_nm, s=10, alpha=0.55)
    lower = float(min(np.min(true), np.min(cnn_pred_nm)))
    upper = float(max(np.max(true), np.max(cnn_pred_nm)))
    plt.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    plt.xlabel("True delta_L (nm)")
    plt.ylabel("Predicted delta_L (nm)")
    plt.tight_layout()
    plt.savefig(output_dir / "03_test_pred_vs_true_delta.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(error, bins=40, alpha=0.85)
    plt.xlabel("CNN delta error (nm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "04_test_error_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(test_split.l_fft_um, error, s=10, alpha=0.55)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("L_fft_um")
    plt.ylabel("CNN delta error (nm)")
    plt.tight_layout()
    plt.savefig(output_dir / "05_test_error_vs_L_fft.png", dpi=160)
    plt.close()

    names = list(all_metrics)
    values = [all_metrics[name]["cavity_RMSE_nm"] for name in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, values)
    plt.ylabel("Test cavity RMSE (nm)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "06_method_comparison_bar.png", dpi=160)
    plt.close()


def pooling_position_description(pooling: str, pooling_k: int) -> str:
    if pooling == "gap":
        return "No; global averaging removes wavelength-region identity."
    if pooling == "adaptive_k":
        return f"Yes; the encoded spectrum is retained as {pooling_k} ordered regions."
    if pooling == "flatten":
        return "Yes; all encoded spectral positions are flattened in order."
    if pooling == "conv_reduce":
        return (
            "Yes; learned local convolutional reduction is followed by "
            f"{pooling_k} ordered regions."
        )
    raise ValueError(pooling)


def save_summary_report(
    output_dir: Path,
    args: argparse.Namespace,
    load_summary: LoadSummary,
    all_metrics: Dict[str, Dict[str, float]],
    device: torch.device,
    effective_model_name: str,
    effective_pooling: str,
) -> None:
    lines = [
        "# Spectral CNN Training Summary",
        "",
        f"- model: `{effective_model_name}`",
        f"- pooling: `{effective_pooling}`",
        f"- pooling k: `{args.pooling_k}`",
        (
            "- spectral position partitions retained: "
            f"{pooling_position_description(effective_pooling, args.pooling_k)}"
        ),
        f"- spectra input length: `{load_summary.spectra_points}`",
        f"- uses H_peak: `{load_summary.use_hpeak}`",
        f"- split is process-level: `{load_summary.split_by_process}`",
        f"- device: `{device}`",
        f"- prepared dataset: `{load_summary.dataset_path}`",
        f"- prepared source type: `{load_summary.source_type}`",
        f"- selected rows: `{load_summary.selected_rows}`",
        f"- unique processes: `{load_summary.unique_processes}`",
        f"- scalar inputs: `{load_summary.scalar_feature_names}`",
        "",
    ]
    if load_summary.notes or load_summary.conflicting_process_ids:
        lines.extend(["## Notes", ""])
        for note in load_summary.notes:
            lines.append(f"- {note}")
        if load_summary.conflicting_process_ids:
            lines.append(
                "- process IDs assigned to multiple splits: "
                f"`{load_summary.conflicting_process_ids}`"
            )
        lines.append("")
    lines.extend(["## Test Metrics", ""])
    for method, metrics in all_metrics.items():
        lines.extend([f"### {method}", ""])
        for key in [
            "delta_MAE_nm",
            "delta_RMSE_nm",
            "delta_MaxAbs_nm",
            "delta_P95Abs_nm",
            "delta_P99Abs_nm",
            "delta_Bias_nm",
            "R2_delta",
            "cavity_MAE_nm",
            "cavity_RMSE_nm",
            "cavity_MaxAbs_nm",
        ]:
            lines.append(f"- `{key}`: {metrics.get(key, float('nan')):.6g}")
        lines.append("")
    (output_dir / "summary_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_config(
    output_dir: Path,
    args: argparse.Namespace,
    load_summary: LoadSummary,
    device: torch.device,
    y_mean: float,
    y_std: float,
    effective_model_name: str,
    effective_pooling: str,
) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "effective_model": effective_model_name,
            "effective_pooling": effective_pooling,
            "spectral_position_partitions_retained": (
                effective_pooling != "gap"
            ),
            "device": str(device),
            "torch_version": torch.__version__,
            "target_mean_train_nm": y_mean,
            "target_std_train_nm": y_std,
            "load_summary": asdict(load_summary),
            "feature_policy": {
                "spectra": "spectra_norm.npy",
                "scalar_inputs": load_summary.scalar_feature_names,
                "target": "delta_L_nm",
                "use_hpeak": load_summary.use_hpeak,
                "scalers_fit_on": "train_only",
            },
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_dir(),
        help="Prepared folder containing spectra_norm.npy and scalar_fields.npz.",
    )
    parser.add_argument("--output-root", type=Path, default=script_dir())
    parser.add_argument(
        "--model",
        choices=[
            "cnn_small",
            "resnet1d",
            "cnn_small_gap",
            "resnet1d_gap",
        ],
        default="cnn_small",
    )
    parser.add_argument(
        "--pooling",
        choices=["gap", "adaptive_k", "flatten", "conv_reduce"],
        default="adaptive_k",
    )
    parser.add_argument("--pooling-k", type=int, default=16)
    parser.add_argument("--use-hpeak", type=str_to_bool, default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        choices=["smooth_l1", "mse"],
        default="smooth_l1",
    )
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-amp", type=str_to_bool, default=True)
    parser.add_argument("--random-seed", type=int, default=20260613)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args(argv)
    if args.pooling_k <= 0:
        parser.error("--pooling-k must be positive")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.random_seed)
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset folder not found: {dataset_path}. "
            "Run prepare_cnn_dataset.py first."
        )

    base_model, effective_pooling, effective_model_name = (
        resolve_model_and_pooling(args.model, args.pooling)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"spectral_cnn_{effective_model_name}_{timestamp}"
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"output_dir={output_dir}")
    print(f"device={device}")
    print(f"dataset={dataset_path}")
    print(f"model={effective_model_name}")

    splits, load_summary = load_prepared_dataset(
        dataset_path,
        args.max_train_rows,
        args.max_val_rows,
        args.max_test_rows,
        args.use_hpeak,
    )
    print(json.dumps(asdict(load_summary), indent=2))
    loaders, scalar_scaler, y_mean, y_std, scalar_scaled = make_data_loaders(
        splits,
        args.batch_size,
        args.num_workers,
        args.random_seed,
    )
    model = build_model(
        base_model,
        effective_pooling,
        args.pooling_k,
        load_summary.spectra_points,
        len(load_summary.scalar_feature_names),
    ).to(device)
    history, best_state, last_state = train_model(
        model,
        loaders,
        args,
        device,
        y_mean,
        y_std,
    )

    cnn_pred_nm = evaluate_predictions(
        model,
        loaders["test"],
        device,
        y_mean,
        y_std,
    )
    cnn_metrics = compute_metrics(
        splits["test"].y_nm,
        cnn_pred_nm,
        splits["test"].l_fft_um,
        splits["test"].cavity_true_um,
    )
    baseline_metrics, baseline_preds = run_baselines(splits, scalar_scaled)
    all_metrics = {
        effective_model_name: cnn_metrics,
        **baseline_metrics,
    }

    save_training_log(history, output_dir)
    save_predictions(
        output_dir,
        splits["test"],
        cnn_pred_nm,
        baseline_preds,
    )
    plot_curves(history, output_dir)
    plot_test_diagnostics(
        output_dir,
        splits["test"],
        cnn_pred_nm,
        all_metrics,
    )
    metrics_payload = {
        "test": all_metrics,
        "best_epoch": (
            min(history, key=lambda row: row["val_RMSE_nm"])
            if history
            else None
        ),
        "load_summary": asdict(load_summary),
        "effective_model": effective_model_name,
        "effective_pooling": effective_pooling,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    save_summary_report(
        output_dir,
        args,
        load_summary,
        all_metrics,
        device,
        effective_model_name,
        effective_pooling,
    )
    save_config(
        output_dir,
        args,
        load_summary,
        device,
        y_mean,
        y_std,
        effective_model_name,
        effective_pooling,
    )

    checkpoint_common = {
        "scalar_scaler": scalar_scaler,
        "target_mean": y_mean,
        "target_std": y_std,
        "config": {
            **vars(args),
            "effective_model": effective_model_name,
            "effective_pooling": effective_pooling,
            "spectra_input_length": load_summary.spectra_points,
        },
        "feature_policy": {
            "scalar_inputs": load_summary.scalar_feature_names,
            "target": "delta_L_nm",
            "use_hpeak": load_summary.use_hpeak,
            "scalers_fit_on": "train_only",
        },
        "metrics": all_metrics,
    }
    torch.save(
        {
            **checkpoint_common,
            "model_state_dict": (
                best_state if best_state else model.state_dict()
            ),
        },
        output_dir / "best_model.pt",
    )
    torch.save(
        {
            **checkpoint_common,
            "model_state_dict": last_state,
        },
        output_dir / "last_model.pt",
    )
    joblib.dump(
        {
            "scalar_scaler": scalar_scaler,
            "target_mean": y_mean,
            "target_std": y_std,
        },
        output_dir / "scalers.joblib",
    )
    print("test metrics:")
    print(json.dumps(all_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
