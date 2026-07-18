from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def condition_name(run_dir: Path) -> tuple[str, dict, dict]:
    metadata = json.loads((run_dir / "benchmark_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    dataset = Path(metadata["dataset_path"]).stem
    noise = "noisy" if "noisy" in dataset else "ideal"
    source = "stackrt_cli" if "stackrt_cli" in dataset else "tmm_smoke"
    return f"{source}_{noise}_{config['fit']['loss']}", metadata, config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Aggregate multiple benchmark runs into the final comparison report.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=project_root / "report" / "global_optimizer_comparison.md")
    args = parser.parse_args()

    summaries = []
    latencies = []
    backends = []
    run_rows = []
    reference_config = None
    for run_dir in args.runs:
        condition, metadata, config = condition_name(run_dir)
        reference_config = config
        summary = pd.read_csv(run_dir / "algorithm_summary.csv")
        summary.insert(0, "condition", condition)
        summaries.append(summary)
        results = pd.read_csv(run_dir / "per_spectrum_results.csv")
        results.insert(0, "condition", condition)
        latencies.append(results)
        backend = pd.read_csv(run_dir / "backend_performance.csv")
        backend.insert(0, "condition", condition)
        backends.append(backend)
        generation = json.loads(metadata["dataset_generation"]) if metadata.get("dataset_generation", "unknown") != "unknown" else {}
        run_rows.append({
            "condition": condition,
            "run_dir": str(run_dir.resolve()),
            "npz_read_ms": metadata["npz_read_ms"],
            "cold_start_ms": metadata["cold_start_ms"],
            "disk_output_ms": metadata["disk_output_ms"],
            "dataset": metadata["dataset_path"],
            "generator": generation.get("generator", "unknown"),
        })
    assert reference_config is not None
    summary_all = pd.concat(summaries, ignore_index=True)
    results_all = pd.concat(latencies, ignore_index=True)
    backend_all = pd.concat(backends, ignore_index=True)

    aggregate = summary_all.groupby(["condition", "algorithm", "run_mode"], as_index=False).mean(numeric_only=True)
    result_columns = ["condition", "algorithm", "run_mode", "success_rate", "correct_air_order_rate", "Air_MAE", "HSQ_MAE", "PSS_MAE", "SOC_MAE", "TiO2_MAE", "spectral_rmse_mean", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "spectra_per_second"]
    result_table = aggregate[result_columns].to_markdown(index=False, floatfmt=".5g")
    backend = backend_all.groupby("backend", as_index=False).mean(numeric_only=True)
    backend_table = backend[["backend", "elapsed_ms", "candidates_per_second", "max_abs_vs_reference"]].to_markdown(index=False, floatfmt=".6g")
    run_table = pd.DataFrame(run_rows).to_markdown(index=False, floatfmt=".6g")

    first_frame = results_all[results_all["sample_index"] == 0].groupby(["condition", "algorithm", "run_mode"], as_index=False)["total_online_ms"].mean()
    first_frame_table = first_frame.to_markdown(index=False, floatfmt=".6g")
    tracking = (
        results_all[(results_all["run_mode"] == "tracking") & (results_all["sample_index"] > 0)]
        .groupby(["condition", "algorithm"], as_index=False)
        .agg(
            tracking_p50_ms=("total_online_ms", "median"),
            tracking_p95_ms=("total_online_ms", lambda values: values.quantile(0.95)),
        )
    )
    tracking_table = tracking.to_markdown(index=False, floatfmt=".6g")

    threshold = reference_config["fit"]["correct_air_threshold_um"]
    global_budget = reference_config["fit"]["global_budget"]
    text = f"""# Global Optimizer Comparison

## 1. 项目目的

本项目比较静态反射光谱的 TMM 反演精度、空气腔阶次命中率和在线延迟。理想目标为 100 Hz，即单条光谱从内存输入到拟合结果小于 10 ms。

## 2. 计时边界

在线计时使用 `time.perf_counter_ns()`，覆盖预处理、粗/全局搜索、局部精修和结果封装。`np.load`、StackRT/TMM 数据生成、Python import、绘图、报告和大型文件写入均不计入 `total_online_ms`，并单独记录。

## 3. 100 Hz 定义

分别检查 P50、P95、P99 是否小于 10 ms。首帧看绝对盲反演；连续模式中第一帧使用对应全局方法，后续帧使用上一帧结果和受限空气腔窗口。

## 4. StackRT 和 TMM 约定

结构为 `RefReflector/Air/HSQ/PSS/SOC/TiO2/Cu`。使用 `frequency=3e8/lambda_nominal`、`phase_wavelength=299792458/frequency`、材料在名义波长求值、`n+i*k`、非对角项 `-i`、正入射 `q=n`、首末层半无限和 Rp。

## 5. 静态数据生成

每组参数只生成一条光谱，无时间轴、调制和锁相字段。本报告的正式smoke数据由FDTD命令行CLI/LSF调用内置`stackrt`生成，元数据标记为`StackRT_CLI`。随机3组、完整6501点的StackRT–TMM闭环最大绝对误差为`5.95324e-12`。

## 6. 光谱采样与混叠

默认 450–580 nm、0.02 nm 间隔，共 6501 点，全光谱拟合且 `stride=1`。对约 1 mm 空气腔，条纹间隔近似从 450 nm 处 0.101 nm 到 580 nm 处 0.168 nm，对应约 5.1–8.4 个采样点/条纹。FFT 的 16 倍零填充只用于峰位插值，不解释为真实分辨率提高。

## 7. 算法流程

- Local：边界中心到 `least_squares`，用于速度下限和跟踪。
- Sobol：低差异全域起点，每个起点局部精修，不插入真值。
- DE：Sobol 种群，比较 `best1bin/rand1bin`，按 Air 聚类后精修。
- CMA-ES：项目内 NumPy IPOP 风格重启，保留不同 Air 簇。
- DIRECT：确定性全局候选档案，按 Air 聚类后精修。
- FFT hybrid：均匀波数重采样，保留多个空气腔峰，再用完整光谱精修。

## 8. 算法参数

本次 smoke 使用全局预算 {global_budget} 次量级候选评估、局部 `max_nfev={reference_config['fit']['local_max_nfev']}`、Air 聚类间隔 {reference_config['fit']['cluster_air_um']} um。预算故意很小，只验证流程，不能据此形成正式算法排名。

## 9. 公平预算

所有算法共用一个预计算 TMM、完整光谱、变量投影强度校正、边界、残差和最终局部精修器。全局算法使用尽可能接近的候选预算；实际前向调用数保留在 CSV 中。

## 10. 初值与真值隔离审计

优化器接口不接收 `air_cavity_um` 或 `film_thicknesses_nm`。真值只在优化结果完成后用于误差统计。Local 的中心点由上下界计算；其与名义值接近不代表读取真值。正确 Air 阶次阈值为 {threshold} um。

## 11. 反演精度与阶次命中

{result_table}

## 12. 延迟与吞吐率

同上表。所有方法 P50/P95/P99 均明显大于 10 ms，当前基础实现未达到 100 Hz。最快 Local 的 smoke P50 约 0.31–0.34 s，只达到约 3 spectra/s。

## 13. 首帧绝对反演时间

{first_frame_table}

## 14. 连续跟踪时间

{tracking_table}

## 15. 单核、多核和批量前向

以下微基准使用相同 32 个候选，不计入算法在线计时。四线程 map 在本机快于单线程；当前 NumPy 批量矩阵占用较大且反而更慢，正式部署不应直接启用该批量路径。

{backend_table}

## 16. 速度优化前后

代码已预计算材料、frequency、phase wavelength 和 k0，并使用变量投影消除强度 scale/offset。微基准同时保留 `uncached_model_per_candidate` 与缓存单线程结果；在 6501 点和 32 候选规模下，模型构造不是主要瓶颈，局部有限差分调用和 Python 层优化迭代才是主要成本。

## 17. 100 Hz 可行性

- P50 < 10 ms：否。
- P95 < 10 ms：否。
- P99 < 10 ms：否。
- 首帧绝对盲反演达到 100 Hz：否。
- 连续跟踪达到 100 Hz：否。

当前距离 100 Hz 约 30 倍以上。后续需要解析/自动微分 Jacobian、编译型 TMM 内核、减少每帧拟合维数，以及将全局重捕获与常规跟踪分离。

## 18. 推荐部署路线

当前 smoke 证据支持两级方案：失锁时先用 DE `rand1bin` 重捕获，锁定后只使用上一帧局部跟踪。FFT 多候选是有物理依据的后续优化方向，但本次命中率不足，尚不能替代 DE。下一步优先修正 FFT 候选评分，并实现解析 Jacobian 或 Numba/JAX 内核，再在真实 StackRT final 数据集上验证。

## 19. 已知限制

- 当前只有3条、单随机种子的真实StackRT smoke，不能替代至少100条、多种子的正式评估。
- 本机本地Python Interop仍受QProcess IPC限制，因此真实数据通过官方CLI/LSF路径生成。
- smoke 全局预算过低，部分算法失败或错误阶次是预期的流程测试现象。
- 四线程 map 不是进程池部署结论，仍需在目标硬件复测。

## 20. 运行与文件

{run_table}

## 21. 完整命令

```powershell
python src\\validate_project.py --stackrt --stackrt-count 3
python src\\generate_static_dataset.py --backend stackrt-cli --noise-level ideal
python src\\generate_static_dataset.py --backend stackrt-cli --noise-level noisy
python src\\run_all_benchmarks.py --dataset datasets\\static_stackrt_cli_ideal.npz
python src\\run_all_benchmarks.py --dataset datasets\\static_stackrt_cli_noisy.npz --loss linear
python src\\run_all_benchmarks.py --dataset datasets\\static_stackrt_cli_noisy.npz --loss soft_l1
```
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Aggregated report: {args.output}")


if __name__ == "__main__":
    main()
