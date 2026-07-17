# High-Speed Static Spectral TMM Inversion v1

独立的静态 StackRT 数据生成、StackRT 匹配 TMM、全局优化器比较和在线延迟基准项目。该目录不导入仓库中的旧反演脚本，复制整个目录后仍可运行。

## 模型与单位

结构：`RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`。

参数顺序与边界：

| 参数 | 单位 | 边界 |
|---|---:|---:|
| Air | um | 998–1002 |
| HSQ | nm | 20–40 |
| PSS | nm | 1–20 |
| SOC | nm | 30–50 |
| TiO2 | nm | 30–50 |

默认波段为 450–580 nm，间隔 0.02 nm，实际数组为 6501 点，所有局部精修均使用完整光谱，不进行 stride 抽点。

TMM 约定与现有 StackRT 闭环保持一致：`frequency=3e8/lambda_nominal`，相位波长使用 `299792458/frequency`，材料在名义波长求值，复折射率为 `n+i*k`，矩阵非对角项为 `-i`，正入射 `q=n`，输出 Rp。

## 安装

```powershell
cd D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1
python -m pip install -r requirements.txt
```

基础 CMA-ES 已由 NumPy 实现，不需要第三方 `cma`。`cma` 仅作为后续可选对照依赖列在 `requirements.txt` 注释中。

## 验证

不启动 Lumerical 的基础验证：

```powershell
python src\validate_project.py
```

启动真实 Lumerical StackRT 并执行随机空气腔/膜厚闭环：

```powershell
python src\validate_project.py --stackrt --stackrt-count 3
```

本机默认 API 路径为 `D:\Program Files\Lumerical\v241\api\python`。若安装位置不同，只需修改新目录中 `src/main_static_stackrt.py` 的两个路径常量。

## 生成数据

真实 StackRT 理想数据：

```powershell
python src\generate_static_dataset.py --backend stackrt --noise-level ideal --trajectory random
```

真实 StackRT 噪声数据：

```powershell
python src\generate_static_dataset.py --backend stackrt --noise-level noisy --trajectory random
```

连续跟踪序列：

```powershell
python src\generate_static_dataset.py --backend stackrt --noise-level noisy --trajectory tracking --output datasets\static_stackrt_noisy_tracking.npz
```

环境无法启动 Lumerical 时，可验证软件闭环：

```powershell
python src\generate_static_dataset.py --config config_smoke_test.json --backend tmm-smoke --noise-level ideal --output datasets\static_tmm_smoke_not_stackrt.npz
```

这种文件的元数据明确写入 `TMM_SMOKE_TEST_NOT_STACKRT`，不得作为真实 StackRT 结果引用。

## 运行基准

完整默认比较：

```powershell
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_ideal.npz
```

噪声数据分别比较 `linear` 和 `soft_l1`：

```powershell
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_noisy.npz --loss linear
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_noisy.npz --loss soft_l1
```

快速全算法 smoke：

```powershell
python src\run_all_benchmarks.py --config config_smoke_test.json --dataset datasets\static_tmm_smoke_not_stackrt.npz
```

单算法调试：

```powershell
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_ideal.npz --algorithms fft_hybrid local --modes absolute tracking --max-samples 10
```

## 算法

- 单起点局部最小二乘：边界中心，不读取真值。
- Sobol 多起点：全边界低差异起点，每个起点局部精修。
- DE：`best1bin` 和 `rand1bin`，Sobol 初始种群，按 Air 聚类后精修。
- Restart CMA-ES：NumPy IPOP 风格重启，按 Air 聚类后精修。
- DIRECT：确定性全局候选档案，按 Air 聚类后精修。
- FFT hybrid：均匀波数重采样，16 倍零填充仅作峰值插值，保留多个空气腔候选，再用完整光谱精修。

所有算法共享同一个 TMM、预处理、变量投影 `I_measured ≈ a*I_TMM+b`、损失函数、边界和局部精修器。优化器对象中不含真值；真值只在结果完成后加入评估表。

## 时间边界

在线延迟从光谱数组已在内存开始，到结果对象和质量标志完成为止。使用 `time.perf_counter_ns()`。`np.load`、StackRT 生成、Python import、绘图、Markdown、CSV/NPZ 写盘不在在线计时中，并在 metadata 中单独记录。

每次运行写入 `benchmark_runs/YYYYMMDD_HHMMSS/`：

- `per_spectrum_results.csv`
- `latency_breakdown.csv`
- `algorithm_summary.csv`
- `backend_performance.csv`
- `fitted_spectra.npz`
- `benchmark_metadata.json`
- `config_used.json`
- `plots/`

汇总报告固定更新到 `report/global_optimizer_comparison.md`。100 Hz 判据为单帧小于 10 ms，报告分别检查 P50、P95 和 P99，并区分绝对盲反演首帧与连续跟踪。

## 已知限制

- 真实 StackRT 结果必须由带有效 Lumerical 许可的环境生成；TMM smoke 不能替代它。
- 默认基准是单核 NumPy/SciPy；多核、Numba/JAX 和真正的候选矩阵批量内核是后续性能阶段。
- `config_smoke_test.json` 只用于流程验证，预算不足以形成正式算法优劣结论。
- 正式结论至少应使用 final 100 条、多个随机种子，并保留所有失败、超时和错误阶次样本。
