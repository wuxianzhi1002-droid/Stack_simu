from __future__ import annotations

import argparse
import gc
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Feature configuration
# ============================================================================
# This copy trains exactly two multi-output MLP models:
#   1. base_scalar: BASE_FEATURE_NAMES only.
#   2. more_feature: BASE_FEATURE_NAMES plus MORE_FEATURE_NAMES.
#
# Targets are true cavity length plus bounded true film thickness:
#   L_true_um, PSS_true_nm, HSQ_true_nm, SOC_true_nm, TiO2_true_nm.
# Film predictions are decoded as nominal_nm + 5 * tanh(raw_delta).
#
# To change the second model, edit only MORE_FEATURE_NAMES. Supported names are:
#   - one-dimensional numeric NPZ fields, for example "H_peak" or "peak_count";
#   - names listed in spectral_feature_names;
#   - PCA score names such as "PC1", "PC2", ... when pca_scores exists.
#
# Example command:
# python "01_Lumerical_Workflow\ML try\Residual MLP\train_residual_mlp_simple_multioutput.py" ^
#   --dataset "path\to\training_dataset.npz" ^
#   --epochs 120


SCRIPT_DIR = Path(__file__).resolve().parent
ML_TRY_DIR = SCRIPT_DIR.parent

DEPLOYABLE_LAYER_ORDER = ["PSS", "HSQ", "SOC", "TiO2"]
FILM_DELTA_BOUND_NM = 5.0
LATENT_CLIP_EPS = 1e-6
BASE_FEATURE_NAMES = [
    "L_fft_um",
    "PSS_nominal_nm",
    "HSQ_nominal_nm",
    "SOC_nominal_nm",
    "TiO2_nominal_nm",
]

# Edit this list to change only the more_feature model.
MORE_FEATURE_NAMES = [
    "fft_spectral_centroid_um",
    "fringe_visibility_global",
    "fringe_contrast_std",
]

FORBIDDEN_INPUT_NAMES = {
    "delta_L_nm",
    "delta_L_um",
    "cavity_true_um",
    "L_true_um",
    "film_delta_nm",
    "film_true_nm",
}


@dataclass
class TrainConfig:
    """记录本次训练设置，写入 metrics.json 方便后续复现。"""

    dataset_path: str
    output_dir: str
    split_strategy: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    base_feature_names: list[str]
    more_feature_names: list[str]
    hidden_layers: list[int]
    epochs: int
    batch_size: int
    learning_rate: float
    alpha: float
    random_seed: int
    max_train_rows: int | None
    max_val_rows: int | None
    max_test_rows: int | None
    max_corr_rows: int
    max_plot_points: int
    prediction_preview_rows: int


def discover_default_dataset() -> Path:
    """
    自动寻找默认数据集。

    当前项目的正式合并数据集是 nn_cavity_scalar_dataset_all_2000.npz。
    如果以后换成 nn_cavity_dataset_*/nn_cavity_dataset_*.npz，
    这里也会按修改时间自动选择最新的文件。
    """

    preferred = SCRIPT_DIR / "dataset/nn_cavity_spectral_features_20260620_233057.npz"
    if preferred.exists():
        return preferred

    candidates = list(SCRIPT_DIR.glob("nn_cavity_spectral_features_*.npz"))
    candidates += list(SCRIPT_DIR.glob("nn_cavity_spectral_features_*.npz"))
    if not candidates:
        raise FileNotFoundError(
            "没有找到默认数据集。请使用 --dataset 指定 datset.npz。"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train base_scalar and editable more_feature multi-output Residual MLP models."
    )
    parser.add_argument("--dataset", type=Path, default=None, help="输入 .npz 数据集路径。")
    parser.add_argument("--epochs", type=int, default=120, help="MLP 最大训练轮数。")
    parser.add_argument("--batch-size", type=int, default=4096, help="Adam mini-batch 大小。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam 初始学习率。")
    parser.add_argument("--alpha", type=float, default=1e-4, help="MLP L2 正则化强度。")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[128, 128, 64],
        help="MLP 隐藏层，例如 --hidden-layers 128 128 64。",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="最多使用多少 train 行；快速测试可设为 200000。",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="最多使用多少 val 行；快速测试可设为 50000。",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="最多使用多少 test 行；快速测试可设为 50000。",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["process_within_nominal", "nominal_holdout"],
        default="process_within_nominal",
        help="process_within_nominal 是主评价策略；nominal_holdout 更严格。",
    )
    parser.add_argument("--random-seed", type=int, default=20260613, help="统一随机种子。")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="train process/nominal 比例。")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="val process/nominal 比例。")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="test process/nominal 比例。")
    parser.add_argument(
        "--max-corr-rows",
        type=int,
        default=200000,
        help="相关性矩阵最多使用多少 train 行，避免全量相关性计算太慢。",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20000,
        help="散点图最多绘制多少个 test 点。",
    )
    parser.add_argument(
        "--prediction-preview-rows",
        type=int,
        default=10000,
        help="每个模型保存多少行 test 预测样例。",
    )
    return parser.parse_args()


def make_output_dir() -> Path:
    """每次运行创建一个独立输出目录。"""

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / f"residual_mlp_simple_multioutput_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def layer_indices(layer_names: np.ndarray, expected_layers: list[str]) -> list[int]:
    """根据 layer_names 找到 PSS/HSQ/SOC/TiO2 在矩阵字段中的列号。"""

    names = [str(name) for name in layer_names.tolist()]
    index_by_name = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in expected_layers if name not in index_by_name]
    if missing:
        raise ValueError(f"数据集 layer_names 缺少 {missing}，当前 layer_names={names}")
    return [index_by_name[name] for name in expected_layers]


def read_nominal_thickness(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    读取可部署模型允许使用的名义膜厚。

    优先兼容扁平字段：
      PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm

    如果没有扁平字段，则从：
      film_nominal_nm + layer_names
    中按名称取列。
    """

    flat_fields = [f"{layer}_nominal_nm" for layer in DEPLOYABLE_LAYER_ORDER]
    if all(field in data.files for field in flat_fields):
        return np.column_stack([data[field] for field in flat_fields]).astype(np.float32)

    if "film_nominal_nm" not in data.files:
        raise KeyError("数据集中没有 film_nominal_nm，也没有扁平 nominal 字段。")
    if "layer_names" not in data.files:
        raise KeyError("使用 film_nominal_nm 时必须提供 layer_names。")

    idx = layer_indices(data["layer_names"], DEPLOYABLE_LAYER_ORDER)
    return data["film_nominal_nm"][:, idx].astype(np.float32)


def read_true_targets(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str], list[str], dict[str, str]]:
    """读取多输出标签：真实腔长和各膜层真实厚度。"""

    if "L_true_um" in data.files:
        cavity_target = data["L_true_um"].astype(np.float64)
        cavity_target_name = "L_true_um"
        cavity_source = "L_true_um"
    elif "cavity_true_um" in data.files:
        cavity_target = data["cavity_true_um"].astype(np.float64)
        cavity_target_name = "cavity_true_um"
        cavity_source = "cavity_true_um"
    else:
        raise KeyError("数据集中没有 L_true_um 或 cavity_true_um，无法生成腔长标签。")

    if "film_true_nm" not in data.files:
        raise KeyError("数据集中没有 film_true_nm，无法生成膜层真实厚度标签。")
    if "layer_names" not in data.files:
        raise KeyError("使用 film_true_nm 作为标签时必须提供 layer_names。")

    idx = layer_indices(data["layer_names"], DEPLOYABLE_LAYER_ORDER)
    film_true_nm = data["film_true_nm"][:, idx].astype(np.float64)
    if film_true_nm.ndim != 2 or film_true_nm.shape[1] != len(DEPLOYABLE_LAYER_ORDER):
        raise ValueError(f"film_true_nm 形状异常: {film_true_nm.shape}")

    targets = np.column_stack([cavity_target, film_true_nm]).astype(np.float64, copy=False)
    target_names = [cavity_target_name] + [f"{layer}_true_nm" for layer in DEPLOYABLE_LAYER_ORDER]
    target_units = ["um"] + ["nm"] * len(DEPLOYABLE_LAYER_ORDER)
    target_sources = {
        "cavity_target": cavity_source,
        "film_target": "film_true_nm",
        "film_layer_order": ",".join(DEPLOYABLE_LAYER_ORDER),
    }
    return targets, target_names, target_units, target_sources


def build_valid_mask(
    data: np.lib.npyio.NpzFile,
    base_features: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """基础输入和标签必须有效；附加特征中的 NaN 由 train-only imputer 处理。"""

    valid = np.all(np.isfinite(targets), axis=1)
    valid &= np.all(np.isfinite(base_features), axis=1)
    if "valid_mask" in data.files:
        valid &= data["valid_mask"].astype(bool)
    return valid


def encode_model_targets(targets: np.ndarray, nominal_nm: np.ndarray) -> np.ndarray:
    """
    Encode targets for training.

    Cavity length is learned directly. Film thicknesses are represented as an
    unconstrained latent so inference can decode them into nominal +/- 5 nm.
    """

    film_fraction = (targets[:, 1:] - nominal_nm.astype(np.float64)) / FILM_DELTA_BOUND_NM
    film_fraction = np.clip(film_fraction, -1.0 + LATENT_CLIP_EPS, 1.0 - LATENT_CLIP_EPS)
    film_latent = np.arctanh(film_fraction)
    return np.column_stack([targets[:, 0], film_latent]).astype(np.float64, copy=False)


def decode_model_outputs(model_outputs: np.ndarray, nominal_nm: np.ndarray) -> np.ndarray:
    """Decode model outputs into physical targets with bounded film thickness."""

    decoded = np.empty_like(model_outputs, dtype=np.float64)
    decoded[:, 0] = model_outputs[:, 0]
    decoded[:, 1:] = nominal_nm.astype(np.float64) + FILM_DELTA_BOUND_NM * np.tanh(model_outputs[:, 1:])
    return decoded


def film_prior_violation_summary(targets: np.ndarray, nominal_nm: np.ndarray) -> dict[str, float | int]:
    """Summarize whether true film labels are outside the configured prior range."""

    abs_delta = np.abs(targets[:, 1:] - nominal_nm.astype(np.float64))
    return {
        "bound_nm": FILM_DELTA_BOUND_NM,
        "label_values_outside_bound": int(np.count_nonzero(abs_delta > FILM_DELTA_BOUND_NM)),
        "label_rows_outside_bound": int(np.count_nonzero(np.any(abs_delta > FILM_DELTA_BOUND_NM, axis=1))),
        "label_max_abs_delta_nm": float(np.max(abs_delta)) if abs_delta.size else float("nan"),
    }


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """根据比例把 process 或 nominal group 数量切成 train/val/test。"""

    if total < 3:
        raise ValueError("每组至少需要 3 个元素才能切分 train/val/test。")
    n_train = int(round(total * train_ratio))
    n_val = int(round(total * val_ratio))
    n_train = min(max(n_train, 1), total - 2)
    n_val = min(max(n_val, 1), total - n_train - 1)
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def split_process_ids(
    process_id: np.ndarray,
    nominal_stack_id: np.ndarray,
    valid_mask: np.ndarray,
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    按 process_id 切分数据。

    process_within_nominal：
      每个 nominal group 内部的 20 个 process 按比例切 train/val/test。

    nominal_holdout：
      整个 nominal group 进入同一个 split，测试更严格。
    """

    valid_process = process_id[valid_mask]
    valid_nominal = nominal_stack_id[valid_mask]
    unique_process, first_idx = np.unique(valid_process, return_index=True)
    process_nominal = valid_nominal[first_idx]

    split_pids: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    if split_strategy == "process_within_nominal":
        for nominal in np.sort(np.unique(process_nominal)):
            pids = unique_process[process_nominal == nominal].copy()
            rng.shuffle(pids)
            n_train, n_val, _ = split_counts(len(pids), train_ratio, val_ratio)
            split_pids["train"].extend(pids[:n_train].tolist())
            split_pids["val"].extend(pids[n_train : n_train + n_val].tolist())
            split_pids["test"].extend(pids[n_train + n_val :].tolist())

    elif split_strategy == "nominal_holdout":
        nominals = np.sort(np.unique(process_nominal)).copy()
        rng.shuffle(nominals)
        n_train, n_val, _ = split_counts(len(nominals), train_ratio, val_ratio)
        nominal_splits = {
            "train": set(nominals[:n_train].tolist()),
            "val": set(nominals[n_train : n_train + n_val].tolist()),
            "test": set(nominals[n_train + n_val :].tolist()),
        }
        for split_name, nominal_set in nominal_splits.items():
            mask = np.array([nominal in nominal_set for nominal in process_nominal], dtype=bool)
            split_pids[split_name].extend(unique_process[mask].tolist())
    else:
        raise ValueError(f"未知 split_strategy: {split_strategy}")

    return {name: np.array(sorted(pids), dtype=process_id.dtype) for name, pids in split_pids.items()}


def row_indices_from_process_ids(
    process_id: np.ndarray,
    valid_mask: np.ndarray,
    split_pids: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """把 process_id split 转成样本行索引。"""

    indices: dict[str, np.ndarray] = {}
    for split_name, pids in split_pids.items():
        indices[split_name] = np.flatnonzero(valid_mask & np.isin(process_id, pids))
    return indices


def sample_indices(indices: np.ndarray, max_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    """调试时可从某个 split 中抽样，加快训练。"""

    if max_rows is None or len(indices) <= max_rows:
        return indices
    sampled = rng.choice(indices, size=max_rows, replace=False)
    return np.sort(sampled)


def validate_input_feature_names(feature_names: list[str]) -> None:
    duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"输入特征列表包含重复名称: {duplicates}")

    forbidden = [
        name
        for name in feature_names
        if (
            name in FORBIDDEN_INPUT_NAMES
            or name.endswith("_true_nm")
            or "film_delta" in name
            or name.startswith("delta_L")
        )
    ]
    if forbidden:
        raise ValueError(f"禁止把标签、真实膜厚或膜厚扰动作为模型输入: {forbidden}")


def resolve_feature_matrix(
    data: np.lib.npyio.NpzFile,
    feature_names: list[str],
    nominal_nm: np.ndarray,
) -> np.ndarray:
    """按顶部列表中的名字解析直接字段、光谱标量特征或 PCA 分量。"""

    validate_input_feature_names(feature_names)
    num_samples = nominal_nm.shape[0]
    nominal_columns = {
        f"{layer}_nominal_nm": nominal_nm[:, layer_index]
        for layer_index, layer in enumerate(DEPLOYABLE_LAYER_ORDER)
    }

    spectral_index: dict[str, int] = {}
    if "spectral_feature_names" in data.files:
        spectral_index = {
            str(name): index
            for index, name in enumerate(data["spectral_feature_names"].tolist())
        }
    spectral_matrix: np.ndarray | None = None
    pca_scores: np.ndarray | None = None
    columns: list[np.ndarray] = []

    for name in feature_names:
        if name in nominal_columns:
            column = nominal_columns[name]
        elif name in data.files:
            column = data[name]
            if column.ndim != 1 or len(column) != num_samples:
                raise ValueError(
                    f"特征 {name!r} 必须是长度为 {num_samples} 的一维字段，"
                    f"当前 shape={column.shape}。"
                )
            if not np.issubdtype(column.dtype, np.number):
                raise TypeError(f"特征 {name!r} 不是数值字段，dtype={column.dtype}。")
        elif name in spectral_index:
            if "spectral_features_full" not in data.files:
                raise KeyError(
                    f"特征 {name!r} 位于 spectral_feature_names，"
                    "但数据集缺少 spectral_features_full。"
                )
            if spectral_matrix is None:
                spectral_matrix = data["spectral_features_full"].astype(
                    np.float32,
                    copy=False,
                )
                if spectral_matrix.shape[0] != num_samples:
                    raise ValueError("spectral_features_full 行数与样本数不一致。")
            column = spectral_matrix[:, spectral_index[name]]
        elif name.startswith("PC") and name[2:].isdigit():
            component_index = int(name[2:]) - 1
            if component_index < 0:
                raise ValueError(f"PCA 特征编号必须从 PC1 开始: {name!r}")
            if "pca_scores" not in data.files:
                raise KeyError(f"请求了 {name!r}，但数据集缺少 pca_scores。")
            if pca_scores is None:
                pca_scores = data["pca_scores"].astype(np.float32, copy=False)
                if pca_scores.shape[0] != num_samples:
                    raise ValueError("pca_scores 行数与样本数不一致。")
            if component_index >= pca_scores.shape[1]:
                raise ValueError(
                    f"请求了 {name!r}，但数据集只有 {pca_scores.shape[1]} 个 PCA 分量。"
                )
            column = pca_scores[:, component_index]
        else:
            raise KeyError(
                f"找不到特征 {name!r}。它必须是一维 NPZ 字段、"
                "spectral_feature_names 中的名称或 PC1/PC2 形式的 PCA 分量。"
            )

        numeric_column = np.asarray(column, dtype=np.float32).copy()
        numeric_column[~np.isfinite(numeric_column)] = np.nan
        columns.append(numeric_column)

    return np.column_stack(columns).astype(np.float32, copy=False)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """不额外依赖 sklearn.metrics，直接计算 R2。"""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def metrics_from_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    target_units: list[str],
) -> dict[str, dict]:
    """按输出维度分别评估，并给出跨目标摘要指标。"""

    per_target: dict[str, dict[str, float | str]] = {}
    rmse_values = []
    r2_values = []

    for idx, (name, unit) in enumerate(zip(target_names, target_units)):
        err = y_pred[:, idx] - y_true[:, idx]
        abs_err = np.abs(err)
        rmse = math.sqrt(float(np.mean(err**2)))
        mae = float(np.mean(abs_err))
        max_abs = float(np.max(abs_err))
        r2 = float(r2_score_np(y_true[:, idx], y_pred[:, idx]))
        rmse_values.append(rmse)
        r2_values.append(r2)

        target_metrics: dict[str, float | str] = {
            "unit": unit,
            f"MAE_{unit}": mae,
            f"RMSE_{unit}": rmse,
            f"MaxAbs_{unit}": max_abs,
            f"Bias_{unit}": float(np.mean(err)),
            f"P95Abs_{unit}": float(np.percentile(abs_err, 95)),
            f"P99Abs_{unit}": float(np.percentile(abs_err, 99)),
            "R2": r2,
        }
        if unit == "um":
            target_metrics["MAE_nm_equiv"] = mae * 1000.0
            target_metrics["RMSE_nm_equiv"] = rmse * 1000.0
            target_metrics["MaxAbs_nm_equiv"] = max_abs * 1000.0
        per_target[name] = target_metrics

    cavity_name = target_names[0]
    cavity_metrics = per_target[cavity_name]
    film_names = target_names[1:]
    film_mae_nm = [float(per_target[name]["MAE_nm"]) for name in film_names]
    film_rmse_nm = [float(per_target[name]["RMSE_nm"]) for name in film_names]

    aggregate = {
        "cavity_target_name": cavity_name,
        "cavity_MAE_nm_equiv": float(cavity_metrics.get("MAE_nm_equiv", cavity_metrics.get("MAE_nm", np.nan))),
        "cavity_RMSE_nm_equiv": float(cavity_metrics.get("RMSE_nm_equiv", cavity_metrics.get("RMSE_nm", np.nan))),
        "film_mean_MAE_nm": float(np.mean(film_mae_nm)),
        "film_max_MAE_nm": float(np.max(film_mae_nm)),
        "film_mean_RMSE_nm": float(np.mean(film_rmse_nm)),
        "mean_R2": float(np.nanmean(r2_values)),
        "mean_RMSE_native_units": float(np.mean(rmse_values)),
    }
    return {"aggregate": aggregate, "per_target": per_target}


def evaluate_prediction_by_split(
    y_true_by_split: dict[str, np.ndarray],
    y_pred_by_split: dict[str, np.ndarray],
    target_names: list[str],
    target_units: list[str],
) -> dict[str, dict]:
    """计算 train/val/test 三个 split 的指标。"""

    return {
        split: metrics_from_targets(
            y_true_by_split[split],
            y_pred_by_split[split],
            target_names,
            target_units,
        )
        for split in ["train", "val", "test"]
    }


def artifact_suffix(method_name: str) -> str:
    """模型名可直接作为输出文件后缀。"""

    return method_name


def train_mlp_method(
    method_name: str,
    feature_builder: Callable[[np.ndarray], tuple[np.ndarray, list[str], object | None]],
    indices_by_split: dict[str, np.ndarray],
    targets: np.ndarray,
    nominal_nm: np.ndarray,
    target_names: list[str],
    target_units: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """
    训练一个 MLP 方法。

    重要：StandardScaler 只在 train set 上 fit，val/test 只能 transform。
    """

    print(f"\n========== 训练 {method_name} ==========")
    train_indices = indices_by_split["train"]
    y_train = encode_model_targets(
        targets[train_indices].astype(np.float64),
        nominal_nm[train_indices],
    )
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    if np.any(y_std <= 0):
        bad_targets = [target_names[i] for i in np.flatnonzero(y_std <= 0)]
        raise ValueError(f"{method_name}: train 目标标准差为 0，无法训练: {bad_targets}")

    x_by_split: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    feature_transformer = None

    for split in ["train", "val", "test"]:
        x, names, transformer = feature_builder(indices_by_split[split])
        x_by_split[split] = x
        if feature_names is None:
            feature_names = names
            feature_transformer = transformer

    assert feature_names is not None

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train_imputed = imputer.fit_transform(x_by_split["train"])
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imputed)
    y_train_scaled = (y_train - y_mean) / y_std

    model = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate,
        max_iter=args.epochs,
        shuffle=True,
        random_state=args.random_seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        tol=1e-5,
        verbose=True,
    )

    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train_scaled, y_train_scaled)
    elapsed = time.time() - start

    pred_by_split: dict[str, np.ndarray] = {}
    y_true_by_split: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        y_true_by_split[split] = targets[indices_by_split[split]].astype(np.float64)
        x_imputed = imputer.transform(x_by_split[split])
        y_pred_scaled = model.predict(scaler.transform(x_imputed))
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled[:, None]
        encoded_pred = y_pred_scaled * y_std + y_mean
        pred_by_split[split] = decode_model_outputs(
            encoded_pred,
            nominal_nm[indices_by_split[split]],
        )

    suffix = artifact_suffix(method_name)
    model_path = output_dir / f"residual_mlp_{suffix}.joblib"
    joblib.dump(
        {
            "method_name": method_name,
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "feature_transformer": feature_transformer,
            "feature_names": feature_names,
            "target_names": target_names,
            "target_units": target_units,
            "encoded_target_mean": y_mean,
            "encoded_target_std": y_std,
            "film_delta_bound_nm": FILM_DELTA_BOUND_NM,
            "film_prediction_formula": "film_pred_nm = film_nominal_nm + film_delta_bound_nm * tanh(raw_delta)",
            "uses_true_thickness_as_input": False,
            "prediction_output": "multi-output prediction: direct cavity length plus bounded true film thicknesses",
        },
        model_path,
    )

    feature_json_path = output_dir / f"feature_names_{suffix}.json"
    write_feature_names(feature_json_path, method_name, True, feature_names)

    metrics = evaluate_prediction_by_split(y_true_by_split, pred_by_split, target_names, target_units)
    print(f"{method_name} test metrics:")
    print(json.dumps(metrics["test"], indent=2, ensure_ascii=False))

    # 释放大块训练矩阵，只保留 test 预测用于画图和导出 preview。
    test_pred = pred_by_split["test"].copy()
    del x_by_split, x_train_imputed, x_train_scaled, y_train_scaled, pred_by_split
    gc.collect()

    return {
        "method_name": method_name,
        "metrics": metrics,
        "feature_names": feature_names,
        "model_path": str(model_path),
        "epochs_trained": int(model.n_iter_),
        "training_seconds": round(elapsed, 3),
        "test_pred_targets": test_pred,
    }


def write_feature_names(path: Path, method_name: str, enabled: bool, feature_names: list[str] | None) -> None:
    """保存每个实验版本使用的特征名。"""

    payload = {
        "method_name": method_name,
        "enabled": enabled,
        "feature_count": 0 if feature_names is None else len(feature_names),
        "feature_names": [] if feature_names is None else feature_names,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_correlation_files(
    output_dir: Path,
    x_train: np.ndarray,
    feature_names: list[str],
    rng: np.random.Generator,
    max_corr_rows: int,
) -> dict:
    """
    对 more_feature 模型的最终输入特征做 Pearson 相关性检查。

    如果出现 abs(corr)>0.98 的特征对，报告里会提示特征可能冗余。
    """

    if len(x_train) > max_corr_rows:
        rows = rng.choice(np.arange(len(x_train)), size=max_corr_rows, replace=False)
        x_corr = x_train[np.sort(rows)]
    else:
        x_corr = x_train

    corr_df = pd.DataFrame(x_corr, columns=feature_names).corr(method="pearson")
    corr_path = output_dir / "feature_correlation_matrix.csv"
    corr_df.to_csv(corr_path, encoding="utf-8-sig")

    pairs = []
    names = list(corr_df.columns)
    values = corr_df.to_numpy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = float(values[i, j])
            if abs(corr) > 0.98:
                pairs.append(
                    {
                        "feature_a": names[i],
                        "feature_b": names[j],
                        "corr": corr,
                        "abs_corr": abs(corr),
                    }
                )

    high_corr_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr", "abs_corr"])
    high_corr_path = output_dir / "high_correlation_feature_pairs.csv"
    high_corr_df.to_csv(high_corr_path, index=False, encoding="utf-8-sig")
    high_corr_alias_path = output_dir / "high_correlation_features.csv"
    high_corr_df.to_csv(high_corr_alias_path, index=False, encoding="utf-8-sig")

    return {
        "correlation_matrix_path": str(corr_path),
        "high_correlation_pairs_path": str(high_corr_path),
        "high_correlation_features_alias_path": str(high_corr_alias_path),
        "high_correlation_pair_count": int(len(pairs)),
    }


def save_prediction_preview(
    output_dir: Path,
    method_name: str,
    indices: np.ndarray,
    pred_targets: np.ndarray,
    sample_id: np.ndarray,
    process_id: np.ndarray,
    nominal_stack_id: np.ndarray,
    l_fft_um: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
    target_units: list[str],
    nominal_nm: np.ndarray,
    max_rows: int,
) -> str:
    """保存测试集部分预测结果，方便人工检查。"""

    n = min(max_rows, len(indices))
    rows = indices[:n]
    pred = pred_targets[:n]
    true = targets[rows]

    payload: dict[str, np.ndarray] = {
        "sample_id": sample_id[rows],
        "process_id": process_id[rows],
        "nominal_stack_id": nominal_stack_id[rows],
        "L_fft_um": l_fft_um[rows],
        "PSS_nominal_nm": nominal_nm[rows, 0],
        "HSQ_nominal_nm": nominal_nm[rows, 1],
        "SOC_nominal_nm": nominal_nm[rows, 2],
        "TiO2_nominal_nm": nominal_nm[rows, 3],
    }
    for idx, (name, unit) in enumerate(zip(target_names, target_units)):
        err = pred[:, idx] - true[:, idx]
        payload[f"{name}_true"] = true[:, idx]
        payload[f"{name}_pred"] = pred[:, idx]
        payload[f"{name}_error_{unit}"] = err
        if unit == "um":
            payload[f"{name}_error_nm_equiv"] = err * 1000.0

    df = pd.DataFrame(payload)
    path = output_dir / f"test_predictions_{artifact_suffix(method_name)}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9g")
    return str(path)


def save_plots(
    output_dir: Path,
    results: dict[str, dict],
    y_test: np.ndarray,
    target_names: list[str],
    target_units: list[str],
    test_indices: np.ndarray,
    l_fft_um: np.ndarray,
    nominal_nm: np.ndarray,
    rng: np.random.Generator,
    max_plot_points: int,
) -> dict[str, str]:
    """生成多输出模型的对比图片。"""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 只在本机缺 matplotlib 时触发
        warning_path = output_dir / "plot_warning.txt"
        warning_path.write_text(f"matplotlib 不可用，未生成图片: {exc}", encoding="utf-8")
        return {"plot_warning": str(warning_path)}

    plot_methods = [name for name in ["base_scalar", "more_feature"] if name in results]
    selected_name = "more_feature"
    selected_pred = results[selected_name]["test_pred_targets"]
    cavity_idx = 0
    cavity_unit = target_units[cavity_idx]
    cavity_scale = 1000.0 if cavity_unit == "um" else 1.0
    cavity_error_nm = (selected_pred[:, cavity_idx] - y_test[:, cavity_idx]) * cavity_scale

    if len(y_test) > max_plot_points:
        plot_rows = np.sort(rng.choice(np.arange(len(y_test)), size=max_plot_points, replace=False))
    else:
        plot_rows = np.arange(len(y_test))

    paths: dict[str, str] = {}

    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_test[plot_rows, cavity_idx],
        selected_pred[plot_rows, cavity_idx],
        s=4,
        alpha=0.35,
    )
    lo = float(min(y_test[plot_rows, cavity_idx].min(), selected_pred[plot_rows, cavity_idx].min()))
    hi = float(max(y_test[plot_rows, cavity_idx].max(), selected_pred[plot_rows, cavity_idx].max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel(f"True {target_names[cavity_idx]} ({cavity_unit})")
    plt.ylabel(f"Predicted {target_names[cavity_idx]} ({cavity_unit})")
    plt.title("Test Predicted vs True Cavity Length")
    plt.tight_layout()
    path = output_dir / "01_test_pred_vs_true_cavity.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["pred_vs_true_cavity"] = str(path)

    plt.figure(figsize=(8, 5))
    for method in plot_methods:
        pred = results[method]["test_pred_targets"]
        err = (pred[:, cavity_idx] - y_test[:, cavity_idx]) * cavity_scale
        plt.hist(err, bins=80, alpha=0.38, label=method)
    plt.xlabel(f"{target_names[cavity_idx]} prediction error (nm equiv)")
    plt.ylabel("count")
    plt.title("Test Cavity Error Histogram")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / "02_test_cavity_error_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["cavity_error_hist"] = str(path)

    plt.figure(figsize=(8, 5))
    plt.scatter(l_fft_um[test_indices][plot_rows], cavity_error_nm[plot_rows], s=4, alpha=0.35)
    plt.axhline(0.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("L_fft_um")
    plt.ylabel(f"more_feature {target_names[cavity_idx]} error (nm equiv)")
    plt.title("Test Cavity Error vs L_fft")
    plt.tight_layout()
    path = output_dir / "03_test_error_vs_L_fft.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["error_vs_l_fft"] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for idx, layer in enumerate(DEPLOYABLE_LAYER_ORDER):
        ax = axes[idx // 2, idx % 2]
        target_idx = 1 + idx
        film_error = selected_pred[:, target_idx] - y_test[:, target_idx]
        ax.scatter(nominal_nm[test_indices, idx][plot_rows], film_error[plot_rows], s=4, alpha=0.35)
        ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{layer}_nominal_nm")
        ax.set_ylabel("true film thickness error (nm)")
    fig.suptitle("Test Film Thickness Error vs Nominal Thickness")
    fig.tight_layout()
    path = output_dir / "04_test_film_error_vs_nominal_thickness.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["film_error_vs_nominal_thickness"] = str(path)

    film_mae_by_method = {
        method: [
            results[method]["metrics"]["test"]["per_target"][f"{layer}_true_nm"]["MAE_nm"]
            for layer in DEPLOYABLE_LAYER_ORDER
        ]
        for method in plot_methods
    }
    x_layers = np.arange(len(DEPLOYABLE_LAYER_ORDER))
    width = 0.8 / max(len(plot_methods), 1)
    plt.figure(figsize=(10, 5))
    for method_idx, method in enumerate(plot_methods):
        offset = (method_idx - (len(plot_methods) - 1) / 2) * width
        plt.bar(x_layers + offset, film_mae_by_method[method], width, label=method, alpha=0.85)
    plt.xticks(x_layers, DEPLOYABLE_LAYER_ORDER)
    plt.ylabel("Film Thickness MAE (nm)")
    plt.title("Per-Layer Film Thickness Test MAE")
    plt.legend(fontsize=9)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = output_dir / "05_film_mae_by_layer.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths["film_mae_by_layer"] = str(path)

    labels = plot_methods
    cavity_mae = [
        results[name]["metrics"]["test"]["aggregate"]["cavity_MAE_nm_equiv"]
        for name in labels
    ]
    film_mae = [
        results[name]["metrics"]["test"]["aggregate"]["film_mean_MAE_nm"]
        for name in labels
    ]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, cavity_mae, width, label="Cavity MAE nm equiv", color="#4c78a8", alpha=0.85)
    bars2 = plt.bar(x + width / 2, film_mae, width, label="Mean Film MAE nm", color="#e45756", alpha=0.85)

    def add_value_labels(bars, fmt="{:.1f}", fontsize=9, offset=8):
        """在每个柱子上方标注数值"""
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and np.isfinite(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + offset,
                    fmt.format(height),
                    ha='center',
                    va='bottom',
                    fontsize=fontsize,
                    rotation=0
                )

    add_value_labels(bars1, fmt="{:.1f}")
    add_value_labels(bars2, fmt="{:.1f}")

    display_labels = [
        "base scalar" if label == "base_scalar" else "more feature"
        for label in labels
    ]

    plt.xticks(x, display_labels, rotation=25, ha="right")
    plt.ylabel("Test Error (nm)", fontsize=12)
    plt.title("Method Comparison - Test Set Performance", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = output_dir / "06_method_comparison_bar.png"
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    paths["method_comparison_bar"] = str(path)

    return paths


def fmt_metric(metrics: dict) -> str:
    """报告里统一格式化 test 指标。"""

    agg = metrics["aggregate"]
    return (
        f"cavity_MAE={agg['cavity_MAE_nm_equiv']:.3f} nm equiv, "
        f"film_mean_MAE={agg['film_mean_MAE_nm']:.3f} nm, "
        f"film_max_MAE={agg['film_max_MAE_nm']:.3f} nm, "
        f"mean_R2={agg['mean_R2']:.5f}"
    )


def write_summary_report(
    output_dir: Path,
    metrics_payload: dict,
    high_corr_count: int,
) -> str:
    """生成仅包含 base_scalar 和 more_feature 的训练报告。"""

    results = metrics_payload["results"]
    base_test = results["base_scalar"]["metrics"]["test"]
    more_test = results["more_feature"]["metrics"]["test"]
    more_better = more_test["aggregate"]["mean_R2"] > base_test["aggregate"]["mean_R2"]
    base_names = results["base_scalar"]["feature_names"]
    more_names = results["more_feature"]["feature_names"]
    additional_names = [name for name in more_names if name not in base_names]
    target_names = metrics_payload["dataset"]["target_names"]
    target_units = metrics_payload["dataset"]["target_units"]
    prior = metrics_payload["dataset"]["film_prior_constraint"]

    lines = [
        "# Simple Multi-Output Residual MLP Summary Report",
        "",
        "## Model Definitions",
        "",
        "- `base_scalar`: " + ", ".join(base_names) + ".",
        "- `more_feature`: " + ", ".join(more_names) + ".",
        "- Additional features: " + (", ".join(additional_names) if additional_names else "none") + ".",
        "- No quadratic or interaction features are generated.",
        "- Outputs: " + ", ".join(f"{name} ({unit})" for name, unit in zip(target_names, target_units)) + ".",
        f"- Film output constraint: `film_pred_nm = film_nominal_nm + {prior['bound_nm']:.3g} * tanh(raw_delta)`, so predictions stay within nominal +/- {prior['bound_nm']:.3g} nm.",
        f"- Training labels outside this film prior: rows={prior['label_rows_outside_bound']}, values={prior['label_values_outside_bound']}, max_abs_delta={prior['label_max_abs_delta_nm']:.6g} nm.",
        "",
        "## Method Comparison",
        "",
        "| method | cavity_MAE_nm_equiv | cavity_RMSE_nm_equiv | film_mean_MAE_nm | film_max_MAE_nm | mean_R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method_name, result in results.items():
        test = result["metrics"]["test"]["aggregate"]
        lines.append(
            f"| {method_name} | {test['cavity_MAE_nm_equiv']:.3f} | "
            f"{test['cavity_RMSE_nm_equiv']:.3f} | {test['film_mean_MAE_nm']:.3f} | "
            f"{test['film_max_MAE_nm']:.3f} | {test['mean_R2']:.5f} |"
        )

    lines += [
        "",
        "## Per-Target Test Metrics",
        "",
        "| method | target | unit | MAE | RMSE | MaxAbs | R2 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for method_name, result in results.items():
        for target_name in target_names:
            item = result["metrics"]["test"]["per_target"][target_name]
            unit = item["unit"]
            lines.append(
                f"| {method_name} | {target_name} | {unit} | "
                f"{item[f'MAE_{unit}']:.6g} | {item[f'RMSE_{unit}']:.6g} | "
                f"{item[f'MaxAbs_{unit}']:.6g} | {item['R2']:.5f} |"
            )

    lines += [
        "",
        "## Conclusion",
        "",
        (
            "- `more_feature` has higher mean test R2 than `base_scalar`."
            if more_better
            else "- `more_feature` does not have higher mean test R2 than `base_scalar`."
        ),
        f"- base_scalar test: {fmt_metric(base_test)}",
        f"- more_feature test: {fmt_metric(more_test)}",
        f"- High-correlation feature pairs in more_feature: {high_corr_count}.",
        "- True thickness, film_delta, cavity_true_um, L_true_um, and target fields are not model inputs.",
    ]

    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio 必须等于 1。")

    base_feature_names = BASE_FEATURE_NAMES.copy()
    more_feature_names = base_feature_names + MORE_FEATURE_NAMES
    validate_input_feature_names(base_feature_names)
    validate_input_feature_names(more_feature_names)

    dataset_path = args.dataset.resolve() if args.dataset is not None else discover_default_dataset().resolve()
    output_dir = make_output_dir()
    rng = np.random.default_rng(args.random_seed)

    config = TrainConfig(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        split_strategy=args.split_strategy,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        base_feature_names=base_feature_names,
        more_feature_names=more_feature_names,
        hidden_layers=list(args.hidden_layers),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        alpha=float(args.alpha),
        random_seed=int(args.random_seed),
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
        max_corr_rows=int(args.max_corr_rows),
        max_plot_points=int(args.max_plot_points),
        prediction_preview_rows=int(args.prediction_preview_rows),
    )

    print(f"数据集: {dataset_path}")
    print(f"输出目录: {output_dir}")
    print(f"base_scalar features: {base_feature_names}")
    print(f"more_feature features: {more_feature_names}")
    print("本脚本不生成二阶项或交互项。")

    with np.load(dataset_path, allow_pickle=True) as data:
        nominal_nm = read_nominal_thickness(data)
        targets, target_names, target_units, target_sources = read_true_targets(data)
        base_feature_matrix = resolve_feature_matrix(data, base_feature_names, nominal_nm)
        more_feature_matrix = resolve_feature_matrix(data, more_feature_names, nominal_nm)
        valid_mask = build_valid_mask(data, base_feature_matrix, targets)
        prior_summary = film_prior_violation_summary(targets[valid_mask], nominal_nm[valid_mask])

        l_fft_um = data["L_fft_um"].astype(np.float32)
        process_id = data["process_id"]
        nominal_stack_id = data["nominal_stack_id"]
        sample_id = data["sample_id"] if "sample_id" in data.files else np.arange(len(process_id))

        split_pids = split_process_ids(
            process_id=process_id,
            nominal_stack_id=nominal_stack_id,
            valid_mask=valid_mask,
            split_strategy=args.split_strategy,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        indices_all = row_indices_from_process_ids(process_id, valid_mask, split_pids)
        indices_by_split = {
            "train": sample_indices(indices_all["train"], args.max_train_rows, rng),
            "val": sample_indices(indices_all["val"], args.max_val_rows, rng),
            "test": sample_indices(indices_all["test"], args.max_test_rows, rng),
        }

        print("process split:")
        for split in ["train", "val", "test"]:
            print(
                f"  {split}: processes={len(split_pids[split])}, "
                f"rows_used={len(indices_by_split[split]):,}, rows_all={len(indices_all[split]):,}"
            )

        y_true_by_split = {
            split: targets[indices_by_split[split]].astype(np.float64)
            for split in ["train", "val", "test"]
        }
        results: dict[str, dict] = {}

        def build_base_features(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
            return base_feature_matrix[indices], base_feature_names.copy(), None

        def build_more_features(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
            return more_feature_matrix[indices], more_feature_names.copy(), None

        results["base_scalar"] = train_mlp_method(
            "base_scalar",
            build_base_features,
            indices_by_split,
            targets,
            nominal_nm,
            target_names,
            target_units,
            output_dir,
            args,
        )

        results["more_feature"] = train_mlp_method(
            "more_feature",
            build_more_features,
            indices_by_split,
            targets,
            nominal_nm,
            target_names,
            target_units,
            output_dir,
            args,
        )

        more_train_x = more_feature_matrix[indices_by_split["train"]]
        corr_info = save_correlation_files(
            output_dir,
            more_train_x,
            more_feature_names,
            rng,
            args.max_corr_rows,
        )
        del more_train_x
        gc.collect()

        preview_paths = {}
        for method_name in ["base_scalar", "more_feature"]:
            preview_paths[method_name] = save_prediction_preview(
                output_dir,
                method_name,
                indices_by_split["test"],
                results[method_name]["test_pred_targets"],
                sample_id,
                process_id,
                nominal_stack_id,
                l_fft_um,
                targets,
                target_names,
                target_units,
                nominal_nm,
                args.prediction_preview_rows,
            )

        plot_paths = save_plots(
            output_dir,
            results,
            y_true_by_split["test"],
            target_names,
            target_units,
            indices_by_split["test"],
            l_fft_um,
            nominal_nm,
            rng,
            args.max_plot_points,
        )

        metrics_payload = {
            "config": asdict(config),
            "dataset": {
                "rows_total": int(len(process_id)),
                "valid_rows": int(np.count_nonzero(valid_mask)),
                "process_count_total": int(len(np.unique(process_id[valid_mask]))),
                "nominal_count_total": int(len(np.unique(nominal_stack_id[valid_mask]))),
                "layer_order": DEPLOYABLE_LAYER_ORDER,
                "target": "multi_output_true_geometry",
                "target_names": target_names,
                "target_units": target_units,
                "target_sources": target_sources,
                "film_prior_constraint": prior_summary,
                "input_policy": "no true thickness, film delta, cavity_true_um, L_true_um, target, quadratic, or interaction inputs",
                "base_feature_names": base_feature_names,
                "more_feature_names": more_feature_names,
            },
            "split": {
                "strategy": args.split_strategy,
                "process_ids": {name: pids.tolist() for name, pids in split_pids.items()},
                "process_counts": {name: int(len(pids)) for name, pids in split_pids.items()},
                "rows_all": {name: int(len(indices_all[name])) for name in ["train", "val", "test"]},
                "rows_used": {name: int(len(indices_by_split[name])) for name in ["train", "val", "test"]},
            },
            "correlation": corr_info,
            "results": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key not in {"test_pred_targets"}
                }
                for name, result in results.items()
            },
            "preview_paths": preview_paths,
            "plot_paths": plot_paths,
        }

        report_path = write_summary_report(
            output_dir,
            metrics_payload,
            corr_info["high_correlation_pair_count"],
        )
        metrics_payload["summary_report_path"] = report_path

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n训练流程完成。")
    print(f"metrics.json: {metrics_path}")
    print(f"summary_report.md: {report_path}")


if __name__ == "__main__":
    main()
