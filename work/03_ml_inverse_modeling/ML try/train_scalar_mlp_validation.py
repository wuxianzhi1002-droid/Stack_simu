from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "merged_first_0200_processes_scalar_results.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "scalar_mlp_validation_first_0200"

TARGET_COLUMN = "delta_L_nm"
SPLIT_COLUMN = "split_name"
VALID_COLUMN = "valid_mask"

NOMINAL_FEATURES = [
    "L_fft_um",
    "H_peak",
    "PSS_nominal_nm",
    "HSQ_nominal_nm",
    "SOC_nominal_nm",
    "TiO2_nominal_nm",
]
TRUE_FEATURES = [
    "L_fft_um",
    "H_peak",
    "PSS_true_nm",
    "HSQ_true_nm",
    "SOC_true_nm",
    "TiO2_true_nm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a quick scalar residual model on the first 200 process CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input scalar CSV path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--model",
        choices=["mlp", "ridge"],
        default="mlp",
        help="Model type. Ridge is a fast physics-style baseline.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["nominal", "true", "nominal_poly2", "true_poly2"],
        default="true_poly2",
        help=(
            "Feature set. Use *_poly2 to include second-order interactions such as "
            "L_fft_um x film thickness."
        ),
    )
    parser.add_argument("--epochs", type=int, default=120, help="Maximum MLP training epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Adam mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="MLP hidden layer sizes.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--ridge-alpha", type=float, default=1e-3, help="Ridge regularization strength.")
    return parser.parse_args()


def base_feature_columns(feature_set: str) -> list[str]:
    if feature_set.startswith("nominal"):
        return NOMINAL_FEATURES
    if feature_set.startswith("true"):
        return TRUE_FEATURES
    raise ValueError(f"Unknown feature set: {feature_set}")


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def regression_metrics(y_true_nm: np.ndarray, y_pred_nm: np.ndarray) -> dict[str, float]:
    err_nm = y_pred_nm - y_true_nm
    abs_err_nm = np.abs(err_nm)
    return {
        "mae_nm": float(abs_err_nm.mean()),
        "rmse_nm": float(math.sqrt(np.mean(err_nm**2))),
        "median_abs_nm": float(np.median(abs_err_nm)),
        "p90_abs_nm": float(np.percentile(abs_err_nm, 90)),
        "p95_abs_nm": float(np.percentile(abs_err_nm, 95)),
        "p99_abs_nm": float(np.percentile(abs_err_nm, 99)),
        "max_abs_nm": float(abs_err_nm.max()),
        "bias_nm": float(err_nm.mean()),
    }


def make_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
    poly: PolynomialFeatures | None = None,
    fit_poly: bool = False,
) -> tuple[np.ndarray, PolynomialFeatures | None, list[str]]:
    base_columns = base_feature_columns(feature_set)
    x_base = df[base_columns].to_numpy(dtype=np.float32)

    if feature_set.endswith("_poly2"):
        if poly is None:
            poly = PolynomialFeatures(degree=2, include_bias=False)
        x = poly.fit_transform(x_base) if fit_poly else poly.transform(x_base)
        feature_names = list(poly.get_feature_names_out(base_columns))
        return x.astype(np.float32, copy=False), poly, feature_names

    return x_base, poly, base_columns


def predict_delta_nm(
    model: MLPRegressor | Ridge,
    x_scaler: StandardScaler,
    x: np.ndarray,
    y_mean: float,
    y_std: float,
    model_type: str,
) -> np.ndarray:
    x_scaled = x_scaler.transform(x)
    pred = model.predict(x_scaled)
    if model_type == "mlp":
        return pred * y_std + y_mean
    return pred


def evaluate_split(
    name: str,
    model: MLPRegressor | Ridge,
    x_scaler: StandardScaler,
    y_mean: float,
    y_std: float,
    model_type: str,
    x: np.ndarray,
    df: pd.DataFrame,
) -> dict[str, object]:
    y_true = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_pred = predict_delta_nm(model, x_scaler, x, y_mean, y_std, model_type)

    baseline_zero = np.zeros_like(y_true)
    baseline_mean = np.full_like(y_true, y_mean)

    return {
        "split": name,
        "rows": int(len(df)),
        "delta_metrics": regression_metrics(y_true, y_pred),
        "baseline_no_correction_metrics": regression_metrics(y_true, baseline_zero),
        "baseline_train_mean_delta_metrics": regression_metrics(y_true, baseline_mean),
    }


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_columns = base_feature_columns(args.feature_set)
    needed_columns = base_columns + [TARGET_COLUMN, SPLIT_COLUMN, VALID_COLUMN]
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, usecols=needed_columns, encoding="utf-8-sig")
    require_columns(df, needed_columns)

    df = df[df[VALID_COLUMN].astype(bool)].copy()
    split_counts = df[SPLIT_COLUMN].value_counts().to_dict()
    print(f"Valid rows: {len(df):,}")
    print(f"Split counts: {split_counts}")
    print(f"Model: {args.model}")
    print(f"Feature set: {args.feature_set}")

    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    val_df = df[df[SPLIT_COLUMN] == "val"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Expected non-empty train/val/test splits in split_name column.")

    x_train, poly, expanded_feature_columns = make_feature_matrix(train_df, args.feature_set, fit_poly=True)
    x_val, _, _ = make_feature_matrix(val_df, args.feature_set, poly=poly)
    x_test, _, _ = make_feature_matrix(test_df, args.feature_set, poly=poly)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float64)

    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std <= 0:
        raise ValueError("Target standard deviation is zero; cannot train residual model.")

    if args.model == "mlp":
        y_fit = (y_train - y_mean) / y_std
        model: MLPRegressor | Ridge = MLPRegressor(
            hidden_layer_sizes=tuple(args.hidden),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=args.batch_size,
            learning_rate_init=args.learning_rate,
            max_iter=args.epochs,
            shuffle=True,
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            tol=1e-5,
            verbose=True,
        )
    else:
        y_fit = y_train
        model = Ridge(alpha=args.ridge_alpha)

    print("Training...")
    started_at = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train_scaled, y_fit)
    elapsed = time.time() - started_at

    metrics = {
        "csv_path": str(csv_path),
        "model_type": args.model,
        "feature_set": args.feature_set,
        "base_feature_columns": base_columns,
        "expanded_feature_columns": expanded_feature_columns,
        "target_column": TARGET_COLUMN,
        "target_transform": {"mean_nm": y_mean, "std_nm": y_std},
        "hidden_layer_sizes": list(args.hidden) if args.model == "mlp" else None,
        "epochs_requested": int(args.epochs) if args.model == "mlp" else None,
        "epochs_trained": int(model.n_iter_) if args.model == "mlp" else None,
        "batch_size": int(args.batch_size) if args.model == "mlp" else None,
        "learning_rate": float(args.learning_rate) if args.model == "mlp" else None,
        "ridge_alpha": float(args.ridge_alpha) if args.model == "ridge" else None,
        "training_seconds": round(elapsed, 3),
        "valid_rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "results": {},
    }

    for split_name, split_df, split_x in [
        ("train", train_df, x_train),
        ("val", val_df, x_val),
        ("test", test_df, x_test),
    ]:
        metrics["results"][split_name] = evaluate_split(
            split_name, model, x_scaler, y_mean, y_std, args.model, split_x, split_df
        )

    model_path = out_dir / f"scalar_{args.model}_{args.feature_set}_delta_model.joblib"
    metrics_path = out_dir / f"metrics_{args.model}_{args.feature_set}.json"
    joblib.dump(
        {
            "model": model,
            "model_type": args.model,
            "x_scaler": x_scaler,
            "poly": poly,
            "feature_set": args.feature_set,
            "base_feature_columns": base_columns,
            "expanded_feature_columns": expanded_feature_columns,
            "target_column": TARGET_COLUMN,
            "target_mean_nm": y_mean,
            "target_std_nm": y_std,
        },
        model_path,
    )
    metrics["model_path"] = str(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Validation metrics:")
    print(json.dumps(metrics["results"]["val"]["delta_metrics"], indent=2))
    print("Test metrics:")
    print(json.dumps(metrics["results"]["test"]["delta_metrics"], indent=2))


if __name__ == "__main__":
    main()
