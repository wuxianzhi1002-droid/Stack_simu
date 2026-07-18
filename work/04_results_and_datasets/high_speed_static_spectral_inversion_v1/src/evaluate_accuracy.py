from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_config import PARAMETER_NAMES


def _percentile(series: pd.Series, value: float) -> float:
    return float(np.percentile(series.to_numpy(dtype=float), value))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in results.groupby(["algorithm", "run_mode", "random_seed"], sort=True):
        algorithm, mode, seed = keys
        total = group["total_online_ms"]
        row = {
            "algorithm": algorithm,
            "run_mode": mode,
            "random_seed": seed,
            "samples": len(group),
            "success_rate": float(group["success"].mean()),
            "timeout_rate": float(group["timeout"].mean()),
            "correct_air_order_rate": float(group["correct_air_order"].mean()),
            "spectral_rmse_mean": float(group["spectral_rmse"].mean()),
            "latency_mean_ms": float(total.mean()),
            "latency_std_ms": float(total.std(ddof=0)),
            "latency_min_ms": float(total.min()),
            "latency_max_ms": float(total.max()),
            "latency_p50_ms": _percentile(total, 50),
            "latency_p90_ms": _percentile(total, 90),
            "latency_p95_ms": _percentile(total, 95),
            "latency_p99_ms": _percentile(total, 99),
            "spectra_per_second": float(1000.0 / total.mean()),
            "under_10ms_rate": float((total < 10.0).mean()),
            "forward_evaluations_mean": float(group["n_forward_evaluations"].mean()),
        }
        for name in PARAMETER_NAMES:
            absolute = group[f"error_{name}"].abs()
            row[f"{name}_MAE"] = float(absolute.mean())
            row[f"{name}_RMSE"] = float(np.sqrt(np.mean(group[f"error_{name}"] ** 2)))
            row[f"{name}_P95Abs"] = _percentile(absolute, 95)
            row[f"{name}_MaxAbs"] = float(absolute.max())
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    aggregate = summary.groupby(["algorithm", "run_mode"], as_index=False).mean(numeric_only=True)
    for metric, title, filename in (
        ("latency_p95_ms", "P95 online latency", "latency_p95.png"),
        ("correct_air_order_rate", "Correct air-order rate", "air_order_rate.png"),
        ("Air_MAE", "Air cavity MAE", "air_mae.png"),
    ):
        pivot = aggregate.pivot(index="algorithm", columns="run_mode", values=metric)
        axis = pivot.plot(kind="bar", figsize=(10, 5))
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()


def write_report(project_root: Path, run_dir: Path, dataset_path: Path, summary: pd.DataFrame, backend_summary: pd.DataFrame, metadata: dict, config: dict) -> Path:
    report_path = project_root / "report" / "global_optimizer_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = summary.groupby(["algorithm", "run_mode"], as_index=False).mean(numeric_only=True)
    columns = ["algorithm", "run_mode", "success_rate", "correct_air_order_rate", "Air_MAE", "spectral_rmse_mean", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "spectra_per_second"]
    table = aggregate[columns].to_markdown(index=False, floatfmt=".6g")
    backend_table = backend_summary.groupby("backend", as_index=False).mean(numeric_only=True)[["backend", "elapsed_ms", "candidates_per_second", "max_abs_vs_reference"]].to_markdown(index=False, floatfmt=".6g")
    threshold = config["fit"]["correct_air_threshold_um"]
    text = f"""# Global Optimizer Comparison

## 1. 项目目的

比较静态光谱 TMM 反演的精度、空气腔阶次恢复能力和在线延迟，并评估 100 Hz 可行性。

## 2. 计时边界与 100 Hz 定义

主计时边界为：内存中的光谱数组 -> 预处理 -> 粗/全局搜索 -> 局部精修 -> 结果对象。`np.load`、数据生成、导入、绘图和文件写盘均不在主计时区。100 Hz 要求单帧小于 10 ms，并分别检查 P50/P95/P99。

## 3. StackRT/TMM 约定

结构为 RefReflector/Air/HSQ/PSS/SOC/TiO2/Cu；`frequency=3e8/lambda_nominal`，`phase_wavelength=299792458/frequency`，材料在名义波长求值，使用 `n+i*k`、`-i` 非对角项、正入射 `q=n` 和 Rp。首末层为半无限介质。

## 4. 数据与采样

- 数据集：`{dataset_path}`
- 点数：{metadata['wavelength_count']}
- 完整波段拟合，不使用 stride 抽点。
- 数据生成不计入反演延迟。

## 5. 算法与公平预算

比较单起点局部、Sobol 多起点、DE best1bin、DE rand1bin、IPOP 重启 CMA-ES、DIRECT 和 FFT/匹配滤波混合方法。所有方法共用 TMM、变量投影强度校正、残差、边界和最终局部精修器。当前比较模式：`{config['benchmark']['comparison']}`，全局预算为 {config['fit']['global_budget']} 次量级的候选评估。

## 6. 初值与真值隔离审计

优化器接口只接收波长、光谱、噪声尺度、配置和随机种子。真值仅在结果对象完成后用于计算误差与正确阶次标志。正确空气腔阶次阈值明确设为 `{threshold} um`。

## 7. 结果

{table}

## 8. 100 Hz 判断

请以表中的 P50/P95/P99 对照 10 ms。首帧绝对盲反演看 `absolute`；连续跟踪看 `tracking`。未达到目标的算法不通过降低光谱采样率来伪装加速。

## 9. 单核、多核与优化前后

算法基准固定单核 SciPy 路径，并已预计算材料、频率、相位波长和 k0，使用解析变量投影消除强度 scale/offset。另用相同 32 个候选比较未缓存、单线程缓存、NumPy 批量和四线程 map；该后端微基准不计入在线算法延迟。

{backend_table}

线程 map 是与 SciPy `workers` 类似的 map 语义，用于本机比较调度开销；正式进程池仍需在部署硬件上复测。

## 10. 推荐路线

当前 smoke 证据支持“DE rand1bin 重捕获 + 上一帧 Local 连续跟踪”。FFT 多阶次方法保留为后续重点优化路线，但在通过真实 StackRT 数据的阶次命中验证前不作为当前推荐方案。

## 11. 已知限制

- 若数据集标记为 `TMM_SMOKE_TEST_NOT_STACKRT`，它只验证软件闭环，不能替代真实 StackRT 一致性和反演测试。
- 随机算法需要在 final 数据集和更多种子上正式统计。
- 当前未实现 Numba/JAX 可选后端。

## 12. 运行信息

- 运行目录：`{run_dir}`
- 冷启动模型构建：{metadata['cold_start_ms']:.6g} ms
- 真值隔离：{metadata['truth_isolation']}

## 13. 完整运行命令

```powershell
python src/generate_static_dataset.py --backend stackrt-cli --noise-level ideal
python src/run_all_benchmarks.py --dataset datasets/static_stackrt_cli_ideal.npz
```
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path
