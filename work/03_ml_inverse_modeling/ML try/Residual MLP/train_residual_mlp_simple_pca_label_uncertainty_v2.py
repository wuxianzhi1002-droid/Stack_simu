from __future__ import annotations

import argparse
import copy
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

import train_residual_mlp_simple_pca as base


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class LabelUncertaintyConfig:
    dataset_path: str
    output_dir: str
    split_strategy: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    base_feature_names: list[str]
    more_feature_names: list[str]
    use_pca_features: bool
    pca_components: int
    hidden_layers: list[int]
    epochs: int
    batch_size: int
    learning_rate: float
    alpha: float
    random_seed: int
    noise_seed: int
    scenarios: list[str]
    label_uncertainty_values_nm: list[float]
    max_train_rows: int | None
    max_val_rows: int | None
    max_test_rows: int | None
    max_plot_points: int
    prediction_preview_rows: int
    clean_val_early_stopping: bool
    early_stopping_patience: int
    early_stopping_tol: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the simple PCA residual MLP under multiple artificial label "
            "measurement uncertainties without modifying the source workflow."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128, 64])
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument(
        "--split-strategy",
        choices=["process_within_nominal", "nominal_holdout"],
        default="process_within_nominal",
    )
    parser.add_argument("--random-seed", type=int, default=20260613)
    parser.add_argument("--noise-seed", type=int, default=20260701)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-corr-rows", type=int, default=200000)
    parser.add_argument("--max-plot-points", type=int, default=20000)
    parser.add_argument("--prediction-preview-rows", type=int, default=10000)
    parser.add_argument("--pca-components", type=int, default=base.PCA_COMPONENTS)
    parser.add_argument(
        "--disable-clean-val-early-stopping",
        action="store_true",
        help="Train for all epochs instead of stopping by the external clean validation split.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-tol", type=float, default=1e-5)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["clean", "uniform_pm_1nm", "gaussian_sigma_1nm_clipped"],
        default=["uniform_pm_1nm"],
        help=(
            "clean uses the original labels. uniform_pm_1nm adds U(-u,+u) "
            "noise to the used train labels. gaussian_sigma_1nm_clipped adds "
            "N(0,u) noise clipped to [-u,+u]."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["base_scalar", "more_feature"],
        default=["base_scalar", "more_feature"],
    )
    parser.add_argument(
        "--label-uncertainty-values-nm",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help="Uncertainty half-widths/sigmas to compare, in nm.",
    )
    return parser.parse_args()


def make_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / f"residual_mlp_simple_pca_label_uncertainty_v2_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def uncertainty_tag(uncertainty_nm: float) -> str:
    text = f"{uncertainty_nm:g}".replace(".", "p")
    return f"u{text}nm"


def scenario_label_noise(
    scenario: str,
    size: int,
    uncertainty_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if scenario == "clean":
        return np.zeros(size, dtype=np.float64)
    if scenario == "uniform_pm_1nm":
        return rng.uniform(-uncertainty_nm, uncertainty_nm, size=size).astype(np.float64)
    if scenario == "gaussian_sigma_1nm_clipped":
        noise = rng.normal(0.0, uncertainty_nm, size=size)
        return np.clip(noise, -uncertainty_nm, uncertainty_nm).astype(np.float64)
    raise ValueError(f"Unknown scenario: {scenario}")


def noise_stats(noise_nm: np.ndarray) -> dict[str, float]:
    return {
        "mean_nm": float(np.mean(noise_nm)),
        "std_nm": float(np.std(noise_nm)),
        "min_nm": float(np.min(noise_nm)),
        "max_nm": float(np.max(noise_nm)),
        "rmse_nm": math.sqrt(float(np.mean(noise_nm**2))),
    }


def uncertainty_aware_metrics(
    y_true_delta_nm: np.ndarray,
    y_pred_delta_nm: np.ndarray,
    uncertainty_nm: float,
) -> dict[str, float]:
    err_nm = y_pred_delta_nm - y_true_delta_nm
    abs_err_nm = np.abs(err_nm)
    excess_abs_nm = np.maximum(abs_err_nm - uncertainty_nm, 0.0)
    return {
        "within_label_uncertainty_fraction": float(np.mean(abs_err_nm <= uncertainty_nm)),
        "excess_MAE_after_uncertainty_nm": float(np.mean(excess_abs_nm)),
        "excess_RMSE_after_uncertainty_nm": math.sqrt(float(np.mean(excess_abs_nm**2))),
        "label_uncertainty_nm": float(uncertainty_nm),
    }


def evaluate_by_split_with_uncertainty(
    y_true_by_split: dict[str, np.ndarray],
    y_pred_by_split: dict[str, np.ndarray],
    uncertainty_nm: float,
) -> dict[str, dict[str, float]]:
    metrics = base.evaluate_prediction_by_split(y_true_by_split, y_pred_by_split)
    for split in ["train", "val", "test"]:
        metrics[split].update(
            uncertainty_aware_metrics(
                y_true_by_split[split],
                y_pred_by_split[split],
                uncertainty_nm,
            )
        )
    return metrics


def train_mlp_method(
    method_name: str,
    scenario: str,
    label_uncertainty_nm: float,
    feature_builder: Callable[[np.ndarray], tuple[np.ndarray, list[str], object | None]],
    indices_by_split: dict[str, np.ndarray],
    y_train_delta_nm: np.ndarray,
    y_clean_delta_nm: np.ndarray,
    target_mean_nm: float,
    target_std_nm: float,
    output_dir: Path,
    args: argparse.Namespace,
    train_noise_stats: dict[str, float],
) -> dict:
    artifact_name = f"{uncertainty_tag(label_uncertainty_nm)}__{scenario}__{method_name}"
    print(f"\n========== training {artifact_name} ==========")

    y_train = y_train_delta_nm[indices_by_split["train"]].astype(np.float64)
    y_mean = float(target_mean_nm)
    y_std = float(target_std_nm)
    if y_std <= 0:
        raise ValueError(f"{artifact_name}: clean train target std is zero.")

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
        max_iter=1,
        shuffle=True,
        random_state=args.random_seed,
        early_stopping=False,
        warm_start=True,
        tol=1e-5,
        verbose=False,
    )

    clean_val_scaled = (
        y_clean_delta_nm[indices_by_split["val"]].astype(np.float64) - y_mean
    ) / y_std
    x_val_imputed = imputer.transform(x_by_split["val"])
    x_val_scaled = scaler.transform(x_val_imputed)

    start = time.time()
    best_model = None
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, args.epochs + 1):
            model.fit(x_train_scaled, y_train_scaled)
            val_pred_scaled = model.predict(x_val_scaled)
            val_rmse_scaled = math.sqrt(float(np.mean((val_pred_scaled - clean_val_scaled) ** 2)))
            val_rmse_nm = val_rmse_scaled * y_std
            train_loss = float(model.loss_) if hasattr(model, "loss_") else float("nan")
            print(
                f"Iteration {epoch}, loss = {train_loss:.8f}, "
                f"clean_val_RMSE_nm = {val_rmse_nm:.6f}"
            )
            if val_rmse_nm < best_val_rmse - args.early_stopping_tol:
                best_val_rmse = val_rmse_nm
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model = copy.deepcopy(model)
            else:
                epochs_without_improvement += 1

            if (
                not args.disable_clean_val_early_stopping
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Clean validation early stopping at iteration {epoch}; "
                    f"best iteration = {best_epoch}, best clean val RMSE = {best_val_rmse:.6f} nm"
                )
                break

    if best_model is not None and not args.disable_clean_val_early_stopping:
        model = best_model
    elapsed = time.time() - start

    pred_by_split: dict[str, np.ndarray] = {}
    clean_true_by_split: dict[str, np.ndarray] = {}
    noisy_train_true_by_split: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        rows = indices_by_split[split]
        clean_true_by_split[split] = y_clean_delta_nm[rows].astype(np.float64)
        noisy_train_true_by_split[split] = y_train_delta_nm[rows].astype(np.float64)
        x_imputed = imputer.transform(x_by_split[split])
        y_pred_scaled = model.predict(scaler.transform(x_imputed))
        pred_by_split[split] = y_pred_scaled * y_std + y_mean

    metrics_clean = evaluate_by_split_with_uncertainty(
        clean_true_by_split,
        pred_by_split,
        label_uncertainty_nm,
    )
    metrics_noisy_reference = evaluate_by_split_with_uncertainty(
        noisy_train_true_by_split,
        pred_by_split,
        label_uncertainty_nm,
    )

    model_path = output_dir / f"residual_mlp_{artifact_name}.joblib"
    joblib.dump(
        {
            "method_name": method_name,
            "scenario": scenario,
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "feature_transformer": feature_transformer,
            "feature_names": feature_names,
            "target_name": "delta_L_nm",
            "target_mean_nm": y_mean,
            "target_std_nm": y_std,
            "target_scaling_reference": "clean train labels shared by all scenarios",
            "label_uncertainty_nm": float(label_uncertainty_nm),
            "label_noise_model": scenario,
            "early_stopping_policy": (
                "external clean validation split"
                if not args.disable_clean_val_early_stopping
                else "disabled, fixed epoch count"
            ),
            "best_clean_val_rmse_nm": float(best_val_rmse),
            "best_epoch": int(best_epoch),
            "uses_true_thickness": False,
            "prediction_formula": "cavity_pred_um = L_fft_um + delta_L_pred_nm / 1000",
        },
        model_path,
    )

    feature_json_path = output_dir / f"feature_names_{artifact_name}.json"
    base.write_feature_names(feature_json_path, artifact_name, True, feature_names)

    print(f"{artifact_name} clean-reference test metrics:")
    print(json.dumps(metrics_clean["test"], indent=2, ensure_ascii=False))

    test_pred_path = output_dir / f"test_pred_delta_nm_{artifact_name}.npy"
    np.save(test_pred_path, pred_by_split["test"])

    del x_by_split, x_train_imputed, x_train_scaled, y_train_scaled, pred_by_split
    del x_val_imputed, x_val_scaled, clean_val_scaled
    gc.collect()

    return {
        "method_name": method_name,
        "scenario": scenario,
        "label_uncertainty_nm": float(label_uncertainty_nm),
        "artifact_name": artifact_name,
        "metrics_clean_reference": metrics_clean,
        "metrics_noisy_reference": metrics_noisy_reference,
        "feature_names": feature_names,
        "model_path": str(model_path),
        "epochs_trained": int(best_epoch if not args.disable_clean_val_early_stopping else args.epochs),
        "best_epoch": int(best_epoch),
        "best_clean_val_rmse_nm": float(best_val_rmse),
        "training_seconds": round(elapsed, 3),
        "train_noise_stats": train_noise_stats,
        "test_pred_delta_nm_path": str(test_pred_path),
    }


def save_prediction_preview(
    output_dir: Path,
    artifact_name: str,
    indices: np.ndarray,
    pred_delta_nm: np.ndarray,
    sample_id: np.ndarray,
    process_id: np.ndarray,
    nominal_stack_id: np.ndarray,
    cavity_true_um: np.ndarray,
    l_fft_um: np.ndarray,
    y_clean_delta_nm: np.ndarray,
    y_train_delta_nm: np.ndarray,
    nominal_nm: np.ndarray,
    max_rows: int,
) -> str:
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
            "delta_clean_nm": y_clean_delta_nm[rows],
            "delta_training_reference_nm": y_train_delta_nm[rows],
            "delta_pred_nm": pred,
            "cavity_pred_um": cavity_pred_um,
            "cavity_error_vs_clean_nm": cavity_error_nm,
            "PSS_nominal_nm": nominal_nm[rows, 0],
            "HSQ_nominal_nm": nominal_nm[rows, 1],
            "SOC_nominal_nm": nominal_nm[rows, 2],
            "TiO2_nominal_nm": nominal_nm[rows, 3],
        }
    )
    path = output_dir / f"test_predictions_{artifact_name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9g")
    return str(path)


def save_comparison_plots(
    output_dir: Path,
    results: dict[str, dict],
    y_test_clean: np.ndarray,
    rng: np.random.Generator,
    max_plot_points: int,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning_path = output_dir / "plot_warning.txt"
        warning_path.write_text(f"matplotlib is unavailable: {exc}", encoding="utf-8")
        return {"plot_warning": str(warning_path)}

    rows = np.arange(len(y_test_clean))
    if len(rows) > max_plot_points:
        rows = np.sort(rng.choice(rows, size=max_plot_points, replace=False))

    labels = list(results.keys())
    rmse = [
        results[name]["metrics_clean_reference"]["test"]["cavity_RMSE_nm"]
        for name in labels
    ]
    mae = [
        results[name]["metrics_clean_reference"]["test"]["cavity_MAE_nm"]
        for name in labels
    ]
    within = [
        results[name]["metrics_clean_reference"]["test"][
            "within_label_uncertainty_fraction"
        ]
        * 100.0
        for name in labels
    ]

    paths: dict[str, str] = {}
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(9, len(labels) * 1.7), 5.5))
    plt.bar(x - width / 2, mae, width, label="MAE")
    plt.bar(x + width / 2, rmse, width, label="RMSE")
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("clean-reference test error (nm)")
    plt.title("Label Uncertainty Simulation: Clean Test Error")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "01_clean_reference_mae_rmse.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths["clean_reference_mae_rmse"] = str(path)

    plt.figure(figsize=(max(9, len(labels) * 1.7), 5.5))
    plt.bar(x, within, color="#4c78a8", alpha=0.85)
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("test samples within uncertainty band (%)")
    plt.title("Fraction Inside Label-Uncertainty Band")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "02_within_uncertainty_band_fraction.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths["within_uncertainty_band_fraction"] = str(path)

    more_feature_labels = [name for name in labels if name.endswith("__more_feature")]
    if more_feature_labels:
        plt.figure(figsize=(8, 5))
        for name in more_feature_labels:
            pred = np.load(results[name]["test_pred_delta_nm_path"], mmap_mode="r")
            err = pred[rows] - y_test_clean[rows]
            plt.hist(err, bins=80, alpha=0.35, label=name)
        plt.xlabel("delta error vs clean reference (nm)")
        plt.ylabel("count")
        plt.title("more_feature Test Error Histogram")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = output_dir / "03_more_feature_error_hist.png"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths["more_feature_error_hist"] = str(path)

    return paths


def write_summary_report(
    output_dir: Path,
    metrics_payload: dict,
) -> str:
    rows: list[str] = []
    for name, result in metrics_payload["results"].items():
        test = result["metrics_clean_reference"]["test"]
        noise = result["train_noise_stats"]
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{test['cavity_MAE_nm']:.3f}",
                    f"{test['cavity_RMSE_nm']:.3f}",
                    f"{test['cavity_MaxAbs_nm']:.3f}",
                    f"{test['delta_P95Abs_nm']:.3f}",
                    f"{test['within_label_uncertainty_fraction'] * 100.0:.2f}%",
                    f"{test['excess_RMSE_after_uncertainty_nm']:.3f}",
                    f"{noise['rmse_nm']:.3f}",
                    str(result["epochs_trained"]),
                ]
            )
            + " |"
        )

    comparison_rows = metrics_payload.get("uncertainty_mae_comparison", [])
    conclusion = "See `uncertainty_mae_comparison.csv` for MAE versus label uncertainty."
    if comparison_rows:
        more_feature_rows = [
            row
            for row in comparison_rows
            if row["method_name"] == "more_feature" and row["scenario"] != "clean"
        ]
        if more_feature_rows:
            best = min(more_feature_rows, key=lambda row: row["test_MAE_nm"])
            worst = max(more_feature_rows, key=lambda row: row["test_MAE_nm"])
            conclusion = (
                "For more_feature noisy-label runs, test MAE ranged from "
                f"{best['test_MAE_nm']:.3f} nm at {best['label_uncertainty_nm']:.3g} nm "
                f"to {worst['test_MAE_nm']:.3f} nm at {worst['label_uncertainty_nm']:.3g} nm."
            )

    lines = [
        "# Label Uncertainty Residual MLP Summary",
        "",
        "## Design",
        "",
        "- The source script is not modified.",
        "- Feature policy is inherited from `train_residual_mlp_simple_pca.py`.",
        "- Artificial label noise is applied only to the used train labels.",
        "- Test metrics below are evaluated against the original simulation label as a clean latent reference.",
        "- `within band` and `excess_RMSE_after_uncertainty_nm` report tolerance-band behavior.",
        "",
        "## Method Comparison",
        "",
        "| run | MAE_nm | RMSE_nm | MaxAbs_nm | P95Abs_nm | within band | excess_RMSE_nm | train_noise_RMSE_nm | epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Conclusion",
        "",
        f"- {conclusion}",
    ]
    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def write_uncertainty_mae_comparison(output_dir: Path, results: dict[str, dict]) -> tuple[str, list[dict]]:
    rows: list[dict] = []
    for artifact_name, result in results.items():
        test = result["metrics_clean_reference"]["test"]
        noise = result["train_noise_stats"]
        rows.append(
            {
                "artifact_name": artifact_name,
                "label_uncertainty_nm": float(result["label_uncertainty_nm"]),
                "scenario": result["scenario"],
                "method_name": result["method_name"],
                "test_MAE_nm": float(test["cavity_MAE_nm"]),
                "test_RMSE_nm": float(test["cavity_RMSE_nm"]),
                "test_MaxAbs_nm": float(test["cavity_MaxAbs_nm"]),
                "test_P95Abs_nm": float(test["delta_P95Abs_nm"]),
                "within_label_uncertainty_fraction": float(
                    test["within_label_uncertainty_fraction"]
                ),
                "excess_MAE_after_uncertainty_nm": float(
                    test["excess_MAE_after_uncertainty_nm"]
                ),
                "excess_RMSE_after_uncertainty_nm": float(
                    test["excess_RMSE_after_uncertainty_nm"]
                ),
                "train_noise_RMSE_nm": float(noise["rmse_nm"]),
                "epochs_trained": int(result["epochs_trained"]),
                "best_clean_val_rmse_nm": float(result["best_clean_val_rmse_nm"]),
            }
        )
    rows.sort(key=lambda row: (row["method_name"], row["scenario"], row["label_uncertainty_nm"]))
    path = output_dir / "uncertainty_mae_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9g")
    return str(path), rows


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.")
    if not args.label_uncertainty_values_nm:
        raise ValueError("--label-uncertainty-values-nm must contain at least one value.")
    if any(value < 0 for value in args.label_uncertainty_values_nm):
        raise ValueError("--label-uncertainty-values-nm values must be non-negative.")

    base_feature_names = base.BASE_FEATURE_NAMES.copy()
    pca_names = base.pca_feature_names(args.pca_components) if base.USE_PCA_FEATURES else []
    more_feature_names = base_feature_names + base.MORE_FEATURE_NAMES + pca_names
    base.validate_input_feature_names(base_feature_names)
    base.validate_input_feature_names(more_feature_names)

    dataset_path = args.dataset.resolve() if args.dataset is not None else base.discover_default_dataset().resolve()
    output_dir = make_output_dir()
    rng = np.random.default_rng(args.random_seed)
    noise_rng = np.random.default_rng(args.noise_seed)

    config = LabelUncertaintyConfig(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        split_strategy=args.split_strategy,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        base_feature_names=base_feature_names,
        more_feature_names=more_feature_names,
        use_pca_features=bool(base.USE_PCA_FEATURES),
        pca_components=int(args.pca_components if base.USE_PCA_FEATURES else 0),
        hidden_layers=list(args.hidden_layers),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        alpha=float(args.alpha),
        random_seed=int(args.random_seed),
        noise_seed=int(args.noise_seed),
        scenarios=list(args.scenarios),
        label_uncertainty_values_nm=[float(value) for value in args.label_uncertainty_values_nm],
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
        max_plot_points=int(args.max_plot_points),
        prediction_preview_rows=int(args.prediction_preview_rows),
        clean_val_early_stopping=not bool(args.disable_clean_val_early_stopping),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_tol=float(args.early_stopping_tol),
    )

    print(f"dataset: {dataset_path}")
    print(f"output: {output_dir}")
    print(f"scenarios: {args.scenarios}")
    print(f"methods: {args.methods}")
    print(f"label uncertainties: {args.label_uncertainty_values_nm} nm")
    print(f"PCA components: {args.pca_components}")

    with np.load(dataset_path, allow_pickle=True) as data:
        nominal_nm = base.read_nominal_thickness(data)
        y_clean_delta_nm = base.read_target_delta_nm(data)
        pca_metadata = (
            base.validate_pca_dataset(data, args.pca_components)
            if base.USE_PCA_FEATURES
            else None
        )
        base_feature_matrix = base.resolve_feature_matrix(data, base_feature_names, nominal_nm)
        more_feature_matrix = base.resolve_feature_matrix(data, more_feature_names, nominal_nm)
        valid_mask = base.build_valid_mask(data, base_feature_matrix, y_clean_delta_nm)

        l_fft_um = data["L_fft_um"].astype(np.float32)
        process_id = data["process_id"]
        nominal_stack_id = data["nominal_stack_id"]
        cavity_true_um = data["cavity_true_um"].astype(np.float64)
        sample_id = data["sample_id"] if "sample_id" in data.files else np.arange(len(process_id))

        dataset_split = base.split_from_dataset_ids(data, process_id, valid_mask)
        if dataset_split is not None:
            split_pids, indices_all = dataset_split
            effective_split_strategy = "dataset_split_id"
        else:
            split_pids = base.split_process_ids(
                process_id=process_id,
                nominal_stack_id=nominal_stack_id,
                valid_mask=valid_mask,
                split_strategy=args.split_strategy,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                rng=rng,
            )
            indices_all = base.row_indices_from_process_ids(process_id, valid_mask, split_pids)
            effective_split_strategy = args.split_strategy

        indices_by_split = {
            "train": base.sample_indices(indices_all["train"], args.max_train_rows, rng),
            "val": base.sample_indices(indices_all["val"], args.max_val_rows, rng),
            "test": base.sample_indices(indices_all["test"], args.max_test_rows, rng),
        }

        print("process split:")
        for split in ["train", "val", "test"]:
            print(
                f"  {split}: processes={len(split_pids[split])}, "
                f"rows_used={len(indices_by_split[split]):,}, rows_all={len(indices_all[split]):,}"
            )

        clean_train_delta = y_clean_delta_nm[indices_by_split["train"]].astype(np.float64)
        target_mean_nm = float(clean_train_delta.mean())
        target_std_nm = float(clean_train_delta.std())
        if target_std_nm <= 0:
            raise ValueError("clean train target std is zero.")
        print(
            "target scaling: clean train labels, "
            f"mean={target_mean_nm:.6f} nm, std={target_std_nm:.6f} nm"
        )

        def build_base_features(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
            return base_feature_matrix[indices], base_feature_names.copy(), None

        def build_more_features(indices: np.ndarray) -> tuple[np.ndarray, list[str], None]:
            return more_feature_matrix[indices], more_feature_names.copy(), None

        feature_builders = {
            "base_scalar": build_base_features,
            "more_feature": build_more_features,
        }

        results: dict[str, dict] = {}
        preview_paths: dict[str, str] = {}
        scenario_noise_stats: dict[str, dict] = {}

        for label_uncertainty_nm in args.label_uncertainty_values_nm:
            label_uncertainty_nm = float(label_uncertainty_nm)
            for scenario in args.scenarios:
                train_rows = indices_by_split["train"]
                train_noise = scenario_label_noise(
                    scenario,
                    len(train_rows),
                    label_uncertainty_nm,
                    noise_rng,
                )
                y_training_delta_nm = y_clean_delta_nm.copy()
                y_training_delta_nm[train_rows] = y_training_delta_nm[train_rows] + train_noise
                scenario_key = f"{uncertainty_tag(label_uncertainty_nm)}__{scenario}"
                scenario_noise_stats[scenario_key] = noise_stats(train_noise)

                for method_name in args.methods:
                    result = train_mlp_method(
                        method_name=method_name,
                        scenario=scenario,
                        label_uncertainty_nm=label_uncertainty_nm,
                        feature_builder=feature_builders[method_name],
                        indices_by_split=indices_by_split,
                        y_train_delta_nm=y_training_delta_nm,
                        y_clean_delta_nm=y_clean_delta_nm,
                        target_mean_nm=target_mean_nm,
                        target_std_nm=target_std_nm,
                        output_dir=output_dir,
                        args=args,
                        train_noise_stats=scenario_noise_stats[scenario_key],
                    )
                    artifact_name = result["artifact_name"]
                    results[artifact_name] = result
                    preview_paths[artifact_name] = save_prediction_preview(
                        output_dir,
                        artifact_name,
                        indices_by_split["test"],
                        np.load(result["test_pred_delta_nm_path"], mmap_mode="r"),
                        sample_id,
                        process_id,
                        nominal_stack_id,
                        cavity_true_um,
                        l_fft_um,
                        y_clean_delta_nm,
                        y_training_delta_nm,
                        nominal_nm,
                        args.prediction_preview_rows,
                    )
                del y_training_delta_nm
                gc.collect()

        plot_paths = save_comparison_plots(
            output_dir,
            results,
            y_clean_delta_nm[indices_by_split["test"]].astype(np.float64),
            rng,
            args.max_plot_points,
        )
        comparison_csv_path, comparison_rows = write_uncertainty_mae_comparison(output_dir, results)

        metrics_payload = {
            "config": asdict(config),
            "dataset": {
                "rows_total": int(len(process_id)),
                "valid_rows": int(np.count_nonzero(valid_mask)),
                "process_count_total": int(len(np.unique(process_id[valid_mask]))),
                "nominal_count_total": int(len(np.unique(nominal_stack_id[valid_mask]))),
                "layer_order": base.DEPLOYABLE_LAYER_ORDER,
                "target": "delta_L_nm",
                "input_policy": (
                    "No true thickness, film delta, cavity_true_um, target, "
                    "quadratic, or interaction inputs."
                ),
            },
            "split": {
                "strategy": effective_split_strategy,
                "fallback_strategy": args.split_strategy,
                "process_counts": {name: int(len(pids)) for name, pids in split_pids.items()},
                "rows_all": {name: int(len(indices_all[name])) for name in ["train", "val", "test"]},
                "rows_used": {name: int(len(indices_by_split[name])) for name in ["train", "val", "test"]},
            },
            "label_noise": {
                "policy": "noise is applied only to used train labels",
                "scenario_noise_stats": scenario_noise_stats,
                "uniform_pm_expected_noise_rmse_nm_by_uncertainty": {
                    f"{float(value):g}": float(float(value) / math.sqrt(3.0))
                    for value in args.label_uncertainty_values_nm
                },
            },
            "pca": pca_metadata,
            "uncertainty_mae_comparison_csv": comparison_csv_path,
            "uncertainty_mae_comparison": comparison_rows,
            "results": {
                name: dict(result)
                for name, result in results.items()
            },
            "preview_paths": preview_paths,
            "plot_paths": plot_paths,
        }

        report_path = write_summary_report(output_dir, metrics_payload)
        metrics_payload["summary_report_path"] = report_path
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nlabel uncertainty training complete.")
    print(f"metrics.json: {metrics_path}")
    print(f"summary_report.md: {report_path}")


if __name__ == "__main__":
    main()
