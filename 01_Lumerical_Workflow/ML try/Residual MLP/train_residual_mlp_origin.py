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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ML_TRY_DIR = SCRIPT_DIR.parent

DEPLOYABLE_LAYER_ORDER = ["PSS", "HSQ", "SOC", "TiO2"]
BASE_SCALAR_FEATURE_NAMES = [
    "L_fft_um",
    "H_peak",
    "PSS_nominal_nm",
    "HSQ_nominal_nm",
    "SOC_nominal_nm",
    "TiO2_nominal_nm",
]
TRUE_THICKNESS_FEATURE_NAMES = [
    "L_fft_um",
    "H_peak",
    "PSS_true_nm",
    "HSQ_true_nm",
    "SOC_true_nm",
    "TiO2_true_nm",
]


@dataclass
class TrainConfig:
    """记录本次训练设置，写入 metrics.json 方便后续复现。"""

    dataset_path: str
    output_dir: str
    split_strategy: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    enable_all_quadratic: bool
    enable_oracle: bool
    include_hpeak_interactions: bool
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


def str2bool(value: str | bool) -> bool:
    """让命令行同时支持 --flag 和 --flag false 两种写法。"""

    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"不能识别的布尔值: {value}")


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
        description=(
            "Residual MLP + nominal film thickness + optional quadratic interaction features."
        )
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
    parser.add_argument(
        "--enable-all-quadratic",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="开启 scalar_all_quadratic 消融实验，默认 false。",
    )
    parser.add_argument(
        "--enable-oracle",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="开启 true thickness oracle 对照，默认 false，不可部署。",
    )
    parser.add_argument(
        "--include-hpeak-interactions",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="selected_quadratic 中额外加入 H_peak * nominal thickness，默认 false。",
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
    output_dir = SCRIPT_DIR / f"residual_mlp_compare_{stamp}"
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


def read_true_thickness_for_oracle(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    读取真实膜厚，仅用于 scalar_oracle_true_thickness 对照。

    注意：这个函数不在主实验中调用。真实实验里通常拿不到 true thickness，
    所以 oracle 结果不能作为可部署模型的性能。
    """

    flat_fields = [f"{layer}_true_nm" for layer in DEPLOYABLE_LAYER_ORDER]
    if all(field in data.files for field in flat_fields):
        return np.column_stack([data[field] for field in flat_fields]).astype(np.float32)

    if "film_true_nm" not in data.files:
        raise KeyError("数据集中没有 film_true_nm，也没有扁平 true thickness 字段。")
    if "layer_names" not in data.files:
        raise KeyError("使用 film_true_nm 时必须提供 layer_names。")

    idx = layer_indices(data["layer_names"], DEPLOYABLE_LAYER_ORDER)
    return data["film_true_nm"][:, idx].astype(np.float32)


def read_target_delta_nm(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """读取残差标签 delta_L_nm；如果不存在则由 cavity_true_um 和 L_fft_um 计算。"""

    if "delta_L_nm" in data.files:
        return data["delta_L_nm"].astype(np.float64)

    if "cavity_true_um" not in data.files:
        raise KeyError("数据集中没有 delta_L_nm，也没有 cavity_true_um，无法生成标签。")
    return (data["cavity_true_um"].astype(np.float64) - data["L_fft_um"].astype(np.float64)) * 1000.0


def build_valid_mask(
    data: np.lib.npyio.NpzFile,
    nominal_nm: np.ndarray,
    target_delta_nm: np.ndarray,
) -> np.ndarray:
    """
    有效样本筛选。

    任何输入特征或标签出现 nan/inf 都不能进入训练。
    如果数据集自带 valid_mask，也一起取交集。
    """

    valid = np.isfinite(data["L_fft_um"]) & np.isfinite(data["H_peak"]) & np.isfinite(target_delta_nm)
    valid &= np.all(np.isfinite(nominal_nm), axis=1)
    if "valid_mask" in data.files:
        valid &= data["valid_mask"].astype(bool)
    return valid


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


def scalar_baseline_features(
    l_fft_um: np.ndarray,
    h_peak: np.ndarray,
    film_nm: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """版本 1：只使用 6 个标量输入特征。"""

    x = np.column_stack([l_fft_um, h_peak, film_nm]).astype(np.float32)
    return x, BASE_SCALAR_FEATURE_NAMES.copy()


def selected_quadratic_features(
    l_fft_um: np.ndarray,
    h_peak: np.ndarray,
    film_nm: np.ndarray,
    include_hpeak_interactions: bool,
) -> tuple[np.ndarray, list[str]]:
    """
    版本 2：手选二阶交互项。

    默认只加入最有物理意义的：
      L_fft_um * 每层 nominal thickness

    H_peak * thickness 是可选项，默认不加，避免二阶项过多。
    """

    x_base, names = scalar_baseline_features(l_fft_um, h_peak, film_nm)
    features = [x_base]
    feature_names = names.copy()

    for layer_idx, layer in enumerate(DEPLOYABLE_LAYER_ORDER):
        features.append((l_fft_um * film_nm[:, layer_idx]).reshape(-1, 1))
        feature_names.append(f"L_fft_um*{layer}_nominal_nm")

    if include_hpeak_interactions:
        for layer_idx, layer in enumerate(DEPLOYABLE_LAYER_ORDER):
            features.append((h_peak * film_nm[:, layer_idx]).reshape(-1, 1))
            feature_names.append(f"H_peak*{layer}_nominal_nm")

    return np.column_stack(features).astype(np.float32), feature_names


def all_quadratic_features(
    base_x: np.ndarray,
    base_names: list[str],
    poly: PolynomialFeatures | None,
    fit: bool,
) -> tuple[np.ndarray, list[str], PolynomialFeatures]:
    """
    可选消融：自动生成所有二阶项。

    这个模式可能引入冗余和多重共线性，默认不开启。
    """

    if poly is None:
        poly = PolynomialFeatures(degree=2, include_bias=False)
    x = poly.fit_transform(base_x) if fit else poly.transform(base_x)
    names = list(poly.get_feature_names_out(base_names))
    return x.astype(np.float32, copy=False), names, poly


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """不额外依赖 sklearn.metrics，直接计算 R2。"""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def metrics_from_delta(y_true_delta_nm: np.ndarray, y_pred_delta_nm: np.ndarray) -> dict[str, float]:
    """
    输出 delta 和 cavity 两套指标。

    由于 cavity_pred_um = L_fft_um + delta_pred_nm / 1000，
    所以 cavity error 的 nm 数值和 delta error 完全等价。
    这里仍然同时输出两种命名，方便看报告。
    """

    err_nm = y_pred_delta_nm - y_true_delta_nm
    abs_err_nm = np.abs(err_nm)
    rmse_nm = math.sqrt(float(np.mean(err_nm**2)))
    mae_nm = float(np.mean(abs_err_nm))
    max_abs_nm = float(np.max(abs_err_nm))
    return {
        "delta_MAE_nm": mae_nm,
        "delta_RMSE_nm": rmse_nm,
        "delta_MaxAbs_nm": max_abs_nm,
        "cavity_MAE_nm": mae_nm,
        "cavity_RMSE_nm": rmse_nm,
        "cavity_MaxAbs_nm": max_abs_nm,
        "R2_delta": float(r2_score_np(y_true_delta_nm, y_pred_delta_nm)),
        "delta_Bias_nm": float(np.mean(err_nm)),
        "delta_P95Abs_nm": float(np.percentile(abs_err_nm, 95)),
        "delta_P99Abs_nm": float(np.percentile(abs_err_nm, 99)),
    }


def evaluate_prediction_by_split(
    y_true_by_split: dict[str, np.ndarray],
    y_pred_by_split: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """计算 train/val/test 三个 split 的指标。"""

    return {
        split: metrics_from_delta(y_true_by_split[split], y_pred_by_split[split])
        for split in ["train", "val", "test"]
    }


def artifact_suffix(method_name: str) -> str:
    """把内部方法名映射成 prompt.md 要求的输出文件后缀。"""

    mapping = {
        "scalar_baseline": "scalar_baseline",
        "scalar_selected_quadratic": "selected_quadratic",
        "scalar_all_quadratic": "all_quadratic",
        "scalar_oracle_true_thickness": "oracle_true_thickness",
    }
    return mapping.get(method_name, method_name)


def train_mlp_method(
    method_name: str,
    feature_builder: Callable[[np.ndarray], tuple[np.ndarray, list[str], object | None]],
    indices_by_split: dict[str, np.ndarray],
    l_fft_um: np.ndarray,
    h_peak: np.ndarray,
    film_nm: np.ndarray,
    y_delta_nm: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """
    训练一个 MLP 方法。

    重要：StandardScaler 只在 train set 上 fit，val/test 只能 transform。
    """

    print(f"\n========== 训练 {method_name} ==========")
    y_train = y_delta_nm[indices_by_split["train"]].astype(np.float64)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std <= 0:
        raise ValueError(f"{method_name}: train delta_L_nm 标准差为 0，无法训练。")

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

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_by_split["train"])
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
        y_true_by_split[split] = y_delta_nm[indices_by_split[split]].astype(np.float64)
        y_pred_scaled = model.predict(scaler.transform(x_by_split[split]))
        pred_by_split[split] = y_pred_scaled * y_std + y_mean

    suffix = artifact_suffix(method_name)
    model_path = output_dir / f"residual_mlp_{suffix}.joblib"
    joblib.dump(
        {
            "method_name": method_name,
            "model": model,
            "scaler": scaler,
            "feature_transformer": feature_transformer,
            "feature_names": feature_names,
            "target_name": "delta_L_nm",
            "target_mean_nm": y_mean,
            "target_std_nm": y_std,
            "uses_true_thickness": method_name == "scalar_oracle_true_thickness",
            "prediction_formula": "cavity_pred_um = L_fft_um + delta_L_pred_nm / 1000",
        },
        model_path,
    )

    feature_json_path = output_dir / f"feature_names_{suffix}.json"
    write_feature_names(feature_json_path, method_name, True, feature_names)

    metrics = evaluate_prediction_by_split(y_true_by_split, pred_by_split)
    print(f"{method_name} test metrics:")
    print(json.dumps(metrics["test"], indent=2, ensure_ascii=False))

    # 释放大块训练矩阵，只保留 test 预测用于画图和导出 preview。
    test_pred = pred_by_split["test"].copy()
    del x_by_split, x_train_scaled, y_train_scaled, pred_by_split
    gc.collect()

    return {
        "method_name": method_name,
        "metrics": metrics,
        "feature_names": feature_names,
        "model_path": str(model_path),
        "epochs_trained": int(model.n_iter_),
        "training_seconds": round(elapsed, 3),
        "test_pred_delta_nm": test_pred,
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
    对 selected_quadratic 的最终输入特征做 Pearson 相关性检查。

    如果出现 abs(corr)>0.98 的特征对，报告里会提示二阶特征可能冗余。
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
    pred_delta_nm: np.ndarray,
    sample_id: np.ndarray,
    process_id: np.ndarray,
    nominal_stack_id: np.ndarray,
    cavity_true_um: np.ndarray,
    l_fft_um: np.ndarray,
    y_delta_nm: np.ndarray,
    nominal_nm: np.ndarray,
    max_rows: int,
) -> str:
    """保存测试集部分预测结果，方便人工检查。"""

    n = min(max_rows, len(indices))
    rows = indices[:n]
    pred = pred_delta_nm[:n]
    cavity_pred_um = l_fft_um[rows] + pred / 1000.0
    cavity_error_nm = (cavity_pred_um - cavity_true_um[rows]) * 1000.0

    df = pd.DataFrame(
        {
            "sample_id": sample_id[rows],
            "process_id": process_id[rows],
            "nominal_stack_id": nominal_stack_id[rows],
            "cavity_true_um": cavity_true_um[rows],
            "L_fft_um": l_fft_um[rows],
            "delta_true_nm": y_delta_nm[rows],
            "delta_pred_nm": pred,
            "cavity_pred_um": cavity_pred_um,
            "cavity_error_nm": cavity_error_nm,
            "PSS_nominal_nm": nominal_nm[rows, 0],
            "HSQ_nominal_nm": nominal_nm[rows, 1],
            "SOC_nominal_nm": nominal_nm[rows, 2],
            "TiO2_nominal_nm": nominal_nm[rows, 3],
        }
    )
    path = output_dir / f"test_predictions_{artifact_suffix(method_name)}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9g")
    return str(path)


def save_plots(
    output_dir: Path,
    results: dict[str, dict],
    y_test: np.ndarray,
    test_indices: np.ndarray,
    l_fft_um: np.ndarray,
    nominal_nm: np.ndarray,
    rng: np.random.Generator,
    max_plot_points: int,
) -> dict[str, str]:
    """生成 prompt.md 要求的 5 张图片。"""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 只在本机缺 matplotlib 时触发
        warning_path = output_dir / "plot_warning.txt"
        warning_path.write_text(f"matplotlib 不可用，未生成图片: {exc}", encoding="utf-8")
        return {"plot_warning": str(warning_path)}

    plot_methods = [
        name
        for name in [
            "raw_fft_baseline",
            "mean_residual_baseline",
            "scalar_baseline",
            "scalar_selected_quadratic",
            "scalar_all_quadratic",
        ]
        if name in results
    ]
    selected_name = "scalar_selected_quadratic"
    selected_pred = results[selected_name]["test_pred_delta_nm"]
    selected_error = selected_pred - y_test

    if len(y_test) > max_plot_points:
        plot_rows = np.sort(rng.choice(np.arange(len(y_test)), size=max_plot_points, replace=False))
    else:
        plot_rows = np.arange(len(y_test))

    paths: dict[str, str] = {}

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test[plot_rows], selected_pred[plot_rows], s=4, alpha=0.35)
    lo = float(min(y_test[plot_rows].min(), selected_pred[plot_rows].min()))
    hi = float(max(y_test[plot_rows].max(), selected_pred[plot_rows].max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("True delta_L_nm")
    plt.ylabel("Predicted delta_L_nm")
    plt.title("Test Predicted vs True Delta")
    plt.tight_layout()
    path = output_dir / "01_test_pred_vs_true_delta.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["pred_vs_true_delta"] = str(path)

    plt.figure(figsize=(8, 5))
    for method in plot_methods:
        err = results[method]["test_pred_delta_nm"] - y_test
        plt.hist(err, bins=80, alpha=0.38, label=method)
    plt.xlabel("delta prediction error (nm)")
    plt.ylabel("count")
    plt.title("Test Error Histogram")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / "02_test_error_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["error_hist"] = str(path)

    plt.figure(figsize=(8, 5))
    plt.scatter(l_fft_um[test_indices][plot_rows], selected_error[plot_rows], s=4, alpha=0.35)
    plt.axhline(0.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("L_fft_um")
    plt.ylabel("selected_quadratic error (nm)")
    plt.title("Test Error vs L_fft")
    plt.tight_layout()
    path = output_dir / "03_test_error_vs_L_fft.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["error_vs_l_fft"] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for idx, layer in enumerate(DEPLOYABLE_LAYER_ORDER):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(nominal_nm[test_indices, idx][plot_rows], selected_error[plot_rows], s=4, alpha=0.35)
        ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{layer}_nominal_nm")
        ax.set_ylabel("error (nm)")
    fig.suptitle("Test Error vs Nominal Thickness")
    fig.tight_layout()
    path = output_dir / "04_test_error_vs_nominal_thickness.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["error_vs_nominal_thickness"] = str(path)

    labels = plot_methods
    mae = [results[name]["metrics"]["test"]["cavity_MAE_nm"] for name in labels]
    rmse = [results[name]["metrics"]["test"]["cavity_RMSE_nm"] for name in labels]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, mae, width, label="MAE", color="#4c78a8", alpha=0.85)
    bars2 = plt.bar(x + width / 2, rmse, width, label="RMSE", color="#e45756", alpha=0.85)

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

    display_labels = []
    for label in labels:
        if label == "raw_fft_baseline":
            display_labels.append("raw FFT")
        elif label == "mean_residual_baseline":
            display_labels.append("mean residual")
        elif label == "scalar_baseline":
            display_labels.append("scalar baseline")
        elif label == "scalar_selected_quadratic":
            display_labels.append("scalar quadratic")
        elif label == "scalar_all_quadratic":
            display_labels.append("all quadratic")
        else:
            display_labels.append(label)

    plt.xticks(x, display_labels, rotation=25, ha="right")
    plt.ylabel("Test Cavity Error (nm)", fontsize=12)
    plt.title("Method Comparison - Test Set Performance", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = output_dir / "05_method_comparison_bar.png"
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    paths["method_comparison_bar"] = str(path)

    return paths


def fmt_metric(metrics: dict[str, float]) -> str:
    """报告里统一格式化 test 指标。"""

    return (
        f"MAE={metrics['cavity_MAE_nm']:.3f} nm, "
        f"RMSE={metrics['cavity_RMSE_nm']:.3f} nm, "
        f"MaxAbs={metrics['cavity_MaxAbs_nm']:.3f} nm, "
        f"R2={metrics['R2_delta']:.5f}"
    )


def write_summary_report(
    output_dir: Path,
    metrics_payload: dict,
    high_corr_count: int,
    include_all_quadratic: bool,
    include_oracle: bool,
) -> str:
    """生成 prompt.md 要求的 summary_report.md。"""

    results = metrics_payload["results"]
    scalar_test = results["scalar_baseline"]["metrics"]["test"]
    selected_test = results["scalar_selected_quadratic"]["metrics"]["test"]
    selected_better = selected_test["cavity_RMSE_nm"] < scalar_test["cavity_RMSE_nm"]

    lines = [
        "# Residual MLP Summary Report",
        "",
        "## 核心回答",
        "",
        "1. 主模型是否使用了 true film thickness？",
        "",
        "   没有。主模型只使用 nominal thickness；true film thickness 只允许作为可选 oracle 对照。",
        "",
        "2. 版本 1 scalar_baseline 的 test 误差是多少？",
        "",
        f"   {fmt_metric(scalar_test)}",
        "",
        "3. 版本 2 scalar_selected_quadratic 的 test 误差是多少？",
        "",
        f"   {fmt_metric(selected_test)}",
        "",
        "4. 二阶特征是否改善了未见 process 的测试误差？",
        "",
        (
            "   是，selected_quadratic 的 test RMSE 低于 scalar_baseline。"
            if selected_better
            else "   否，selected_quadratic 的 test RMSE 没有低于 scalar_baseline。"
        ),
        "",
        "5. 是否存在高相关特征对？",
        "",
        (
            f"   存在 {high_corr_count} 对 abs(corr)>0.98 的特征。High correlation detected. Quadratic features may be redundant."
            if high_corr_count > 0
            else "   未发现 abs(corr)>0.98 的高相关特征对。"
        ),
        "",
        "6. 如果 all_quadratic 开启，它相比 selected_quadratic 是否真的更好？",
        "",
    ]

    if include_all_quadratic and "scalar_all_quadratic" in results:
        all_test = results["scalar_all_quadratic"]["metrics"]["test"]
        better = all_test["cavity_RMSE_nm"] < selected_test["cavity_RMSE_nm"]
        lines += [
            f"   all_quadratic: {fmt_metric(all_test)}",
            (
                "   all_quadratic 的 test RMSE 低于 selected_quadratic，可以作为后续候选。"
                if better
                else "   all_quadratic 没有明显优于 selected_quadratic，不应默认采用。"
            ),
        ]
    else:
        lines.append("   本次未开启 all_quadratic；它只作为消融实验，默认不启用。")

    selected_mae = selected_test["cavity_MAE_nm"]
    selected_rmse = selected_test["cavity_RMSE_nm"]
    if selected_mae <= 10.0:
        compensation_text = "   当前结果较好，说明仅靠 scalar features 已经能较强地补偿 ±10 nm 膜厚扰动。"
    elif selected_mae <= 20.0:
        compensation_text = (
            "   当前结果说明 scalar features 有一定补偿能力，但还没有非常充分。"
            "如果目标是更低误差，后续需要考虑引入光谱 I(lambda) 或角度/偏振信息。"
        )
    else:
        compensation_text = (
            "   当前结果不足以说明仅靠 scalar features 就能补偿 ±10 nm 膜厚扰动。"
            "后续需要引入光谱 I(lambda) 或角度/偏振信息。"
        )

    lines += [
        "",
        "7. 当前结果是否说明仅靠 scalar features 就能补偿 ±10 nm 膜厚扰动？",
        "",
        compensation_text,
        "",
        "## Method Comparison",
        "",
        "| method | test cavity_MAE_nm | test cavity_RMSE_nm | test cavity_MaxAbs_nm | R2_delta |",
        "|---|---:|---:|---:|---:|",
    ]

    for method_name, result in results.items():
        test = result["metrics"]["test"]
        lines.append(
            f"| {method_name} | {test['cavity_MAE_nm']:.3f} | "
            f"{test['cavity_RMSE_nm']:.3f} | {test['cavity_MaxAbs_nm']:.3f} | "
            f"{test['R2_delta']:.5f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- 主实验输入：L_fft_um, H_peak, PSS/HSQ/SOC/TiO2_nominal_nm。",
        "- 主实验禁止使用 film_true_nm、film_delta_nm、cavity_true_um 作为输入。",
        "- cavity_true_um 只用于生成标签或评价。",
        "- selected_quadratic 默认只加入 L_fft_um 与各层 nominal thickness 的交互项。",
    ]

    if include_oracle:
        lines += [
            "",
            "## Oracle Warning",
            "",
            "scalar_oracle_true_thickness 使用 true film thickness。This is not deployable because true film thickness is unavailable in real measurement.",
        ]

    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio 必须等于 1。")

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
        enable_all_quadratic=bool(args.enable_all_quadratic),
        enable_oracle=bool(args.enable_oracle),
        include_hpeak_interactions=bool(args.include_hpeak_interactions),
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
    print("主模型输入只使用 nominal thickness，不使用 true/delta thickness。")

    with np.load(dataset_path, allow_pickle=True) as data:
        nominal_nm = read_nominal_thickness(data)
        y_delta_nm = read_target_delta_nm(data)
        valid_mask = build_valid_mask(data, nominal_nm, y_delta_nm)

        l_fft_um = data["L_fft_um"].astype(np.float32)
        h_peak = data["H_peak"].astype(np.float32)
        process_id = data["process_id"]
        nominal_stack_id = data["nominal_stack_id"]
        cavity_true_um = data["cavity_true_um"].astype(np.float64)
        sample_id = data["sample_id"] if "sample_id" in data.files else np.arange(len(process_id))

        split_pids = split_process_ids(
            process_id=process_id,
            nominal_stack_id=nominal_stack_id,
            valid_mask=valid_mask,
            split_strategy="nominal_holdout",
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
            split: y_delta_nm[indices_by_split[split]].astype(np.float64)
            for split in ["train", "val", "test"]
        }

        results: dict[str, dict] = {}

        train_mean_delta = float(y_true_by_split["train"].mean())
        for baseline_name, pred_value in [
            ("raw_fft_baseline", 0.0),
            ("mean_residual_baseline", train_mean_delta),
        ]:
            pred_by_split = {
                split: np.full_like(y_true_by_split[split], pred_value, dtype=np.float64)
                for split in ["train", "val", "test"]
            }
            results[baseline_name] = {
                "method_name": baseline_name,
                "metrics": evaluate_prediction_by_split(y_true_by_split, pred_by_split),
                "feature_names": [],
                "model_path": None,
                "epochs_trained": None,
                "training_seconds": 0.0,
                "test_pred_delta_nm": pred_by_split["test"],
            }

        def build_scalar_features_for_indices(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
            x, names = scalar_baseline_features(l_fft_um[indices], h_peak[indices], nominal_nm[indices])
            return x, names, None

        selected_train_feature_cache: tuple[np.ndarray, list[str]] | None = None

        def build_selected_features_for_indices(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
            x, names = selected_quadratic_features(
                l_fft_um[indices],
                h_peak[indices],
                nominal_nm[indices],
                include_hpeak_interactions=bool(args.include_hpeak_interactions),
            )
            return x, names, None

        results["scalar_baseline"] = train_mlp_method(
            "scalar_baseline",
            build_scalar_features_for_indices,
            indices_by_split,
            l_fft_um,
            h_peak,
            nominal_nm,
            y_delta_nm,
            output_dir,
            args,
        )

        results["scalar_selected_quadratic"] = train_mlp_method(
            "scalar_selected_quadratic",
            build_selected_features_for_indices,
            indices_by_split,
            l_fft_um,
            h_peak,
            nominal_nm,
            y_delta_nm,
            output_dir,
            args,
        )

        # 相关性分析只针对主二阶模型 selected_quadratic 的输入特征。
        selected_train_x, selected_feature_names, _ = build_selected_features_for_indices(indices_by_split["train"])
        corr_info = save_correlation_files(
            output_dir,
            selected_train_x,
            selected_feature_names,
            rng,
            args.max_corr_rows,
        )
        del selected_train_feature_cache, selected_train_x
        gc.collect()

        if args.enable_all_quadratic:
            poly_holder: dict[str, PolynomialFeatures | None] = {"poly": None}

            def build_all_quad_features_for_indices(indices: np.ndarray) -> tuple[np.ndarray, list[str], PolynomialFeatures]:
                base_x, base_names = scalar_baseline_features(
                    l_fft_um[indices], h_peak[indices], nominal_nm[indices]
                )
                fit = poly_holder["poly"] is None
                x, names, poly = all_quadratic_features(base_x, base_names, poly_holder["poly"], fit)
                if fit:
                    poly_holder["poly"] = poly
                return x, names, poly_holder["poly"]

            results["scalar_all_quadratic"] = train_mlp_method(
                "scalar_all_quadratic",
                build_all_quad_features_for_indices,
                indices_by_split,
                l_fft_um,
                h_peak,
                nominal_nm,
                y_delta_nm,
                output_dir,
                args,
            )
        else:
            write_feature_names(
                output_dir / "feature_names_all_quadratic.json",
                "scalar_all_quadratic",
                False,
                None,
            )

        if args.enable_oracle:
            true_nm = read_true_thickness_for_oracle(data)
            oracle_valid = np.all(np.isfinite(true_nm), axis=1)
            if not np.all(oracle_valid[indices_by_split["train"]]):
                raise ValueError("oracle true thickness 中存在无效值。")

            def build_oracle_features_for_indices(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
                x_base = np.column_stack(
                    [
                        l_fft_um[indices],
                        h_peak[indices],
                        true_nm[indices, 0],
                        true_nm[indices, 1],
                        true_nm[indices, 2],
                        true_nm[indices, 3],
                    ]
                ).astype(np.float32)
                names = TRUE_THICKNESS_FEATURE_NAMES.copy()
                features = [x_base]
                feature_names = names.copy()
                for layer_idx, layer in enumerate(DEPLOYABLE_LAYER_ORDER):
                    features.append((l_fft_um[indices] * true_nm[indices, layer_idx]).reshape(-1, 1))
                    feature_names.append(f"L_fft_um*{layer}_true_nm")
                return np.column_stack(features).astype(np.float32), feature_names, None

            results["scalar_oracle_true_thickness"] = train_mlp_method(
                "scalar_oracle_true_thickness",
                build_oracle_features_for_indices,
                indices_by_split,
                l_fft_um,
                h_peak,
                nominal_nm,
                y_delta_nm,
                output_dir,
                args,
            )

        preview_paths = {}
        for method_name in [
            "scalar_baseline",
            "scalar_selected_quadratic",
            "scalar_all_quadratic",
            "scalar_oracle_true_thickness",
        ]:
            if method_name not in results:
                continue
            preview_paths[method_name] = save_prediction_preview(
                output_dir,
                method_name,
                indices_by_split["test"],
                results[method_name]["test_pred_delta_nm"],
                sample_id,
                process_id,
                nominal_stack_id,
                cavity_true_um,
                l_fft_um,
                y_delta_nm,
                nominal_nm,
                args.prediction_preview_rows,
            )

        plot_paths = save_plots(
            output_dir,
            results,
            y_true_by_split["test"],
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
                "target": "delta_L_nm",
                "main_input_policy": "nominal thickness only; no true/delta film thickness in deployable models",
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
                    if key not in {"test_pred_delta_nm"}
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
            bool(args.enable_all_quadratic),
            bool(args.enable_oracle),
        )
        metrics_payload["summary_report_path"] = report_path

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n训练流程完成。")
    print(f"metrics.json: {metrics_path}")
    print(f"summary_report.md: {report_path}")


if __name__ == "__main__":
    main()
