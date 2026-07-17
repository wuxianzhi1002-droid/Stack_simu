from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from benchmark_latency import load_dataset_timed, run_benchmark, write_run_outputs
from benchmark_backends import benchmark_forward_backends
from evaluate_accuracy import save_plots, summarize, write_report
from model_config import load_config


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run static spectral inversion benchmarks.")
    parser.add_argument("--config", default=str(project_root / "config_default.json"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("absolute", "tracking"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--loss", choices=("linear", "soft_l1"))
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    if args.loss:
        config["fit"]["loss"] = args.loss
    algorithms = args.algorithms or list(config["benchmark"]["algorithms"])
    modes = args.modes or list(config["benchmark"]["modes"])
    dataset_path = Path(args.dataset).resolve()
    data, npz_read_ms = load_dataset_timed(dataset_path)
    results, fitted, metadata = run_benchmark(data, config, algorithms, modes, args.max_samples)
    metadata["npz_read_ms"] = npz_read_ms
    metadata["dataset_path"] = str(dataset_path)
    metadata["dataset_generation"] = str(data["generation_parameters_json"].item()) if "generation_parameters_json" in data else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else project_root / "benchmark_runs" / timestamp
    write_run_outputs(run_dir, results, fitted, metadata, config)
    summary = summarize(results)
    summary.to_csv(run_dir / "algorithm_summary.csv", index=False)
    backend_summary = benchmark_forward_backends(data["wavelengths_um"], int(config["random_seed"]))
    backend_summary.to_csv(run_dir / "backend_performance.csv", index=False)
    save_plots(summary, run_dir)
    report = write_report(project_root, run_dir, dataset_path, summary, backend_summary, metadata, config)
    print(f"Run directory: {run_dir}")
    print(f"Report: {report}")
    print(summary[["algorithm", "run_mode", "correct_air_order_rate", "Air_MAE", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
