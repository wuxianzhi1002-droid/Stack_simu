# Global Optimizer Comparison

## 1. 项目目的

本项目比较静态反射光谱的 TMM 反演精度、空气腔阶次命中率和在线延迟。理想目标为 100 Hz，即单条光谱从内存输入到拟合结果小于 10 ms。

## 2. 计时边界

在线计时使用 `time.perf_counter_ns()`，覆盖预处理、粗/全局搜索、局部精修和结果封装。`np.load`、StackRT/TMM 数据生成、Python import、绘图、报告和大型文件写入均不计入 `total_online_ms`，并单独记录。

## 3. 100 Hz 定义

分别检查 P50、P95、P99 是否小于 10 ms。首帧看绝对盲反演；连续模式中第一帧使用对应全局方法，后续帧使用上一帧结果和受限空气腔窗口。

## 4. StackRT 和 TMM 约定

结构为 `RefReflector/Air/HSQ/PSS/SOC/TiO2/Cu`。使用 `frequency=3e8/lambda_nominal`、`phase_wavelength=299792458/frequency`、材料在名义波长求值、`n+i*k`、非对角项 `-i`、正入射 `q=n`、首末层半无限和 Rp。

## 5. 静态数据生成

每组参数只生成一条光谱，无时间轴、调制和锁相字段。当前三组结果均由 `TMM_SMOKE_TEST_NOT_STACKRT` 数据验证软件闭环；真实 Lumerical API 可导入，但 FDTD 启动返回 `Session not found`，所以本报告不把 smoke 结果称为 StackRT 结果。

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

本次 smoke 使用全局预算 40 次量级候选评估、局部 `max_nfev=15`、Air 聚类间隔 0.1 um。预算故意很小，只验证流程，不能据此形成正式算法排名。

## 9. 公平预算

所有算法共用一个预计算 TMM、完整光谱、变量投影强度校正、边界、残差和最终局部精修器。全局算法使用尽可能接近的候选预算；实际前向调用数保留在 CSV 中。

## 10. 初值与真值隔离审计

优化器接口不接收 `air_cavity_um` 或 `film_thicknesses_nm`。真值只在优化结果完成后用于误差统计。Local 的中心点由上下界计算；其与名义值接近不代表读取真值。正确 Air 阶次阈值为 0.25 um。

## 11. 反演精度与阶次命中

| condition     | algorithm   | run_mode   |   success_rate |   correct_air_order_rate |    Air_MAE |   HSQ_MAE |   PSS_MAE |   SOC_MAE |   TiO2_MAE |   spectral_rmse_mean |   latency_p50_ms |   latency_p95_ms |   latency_p99_ms |   spectra_per_second |
|:--------------|:------------|:-----------|---------------:|-------------------------:|-----------:|----------:|----------:|----------:|-----------:|---------------------:|-----------------:|-----------------:|-----------------:|---------------------:|
| ideal_linear  | cmaes       | absolute   |        0.66667 |                  0       | 0.70173    |  12.488   | 10.683    | 13.326    |  12.351    |           0.14539    |           680.98 |           789.95 |           799.64 |               1.3885 |
| ideal_linear  | cmaes       | tracking   |        0.66667 |                  0       | 0.70173    |  12.488   | 10.683    | 13.326    |  12.351    |           0.14539    |           651.51 |           809.31 |           823.34 |               1.7638 |
| ideal_linear  | de_best1bin | absolute   |        0.33333 |                  0.66667 | 0.12031    |   5.5383  |  3.5681   |  6.6874   |   5.6822   |           0.029275   |           835.54 |           841.15 |           841.65 |               1.2255 |
| ideal_linear  | de_best1bin | tracking   |        0.66667 |                  0.33333 | 0.24056    |  11.057   |  7.1293   | 13.338    |  11.361    |           0.058369   |           764.94 |           842.34 |           849.22 |               1.6726 |
| ideal_linear  | de_rand1bin | absolute   |        0       |                  1       | 7.1231e-05 |   0.16754 |  0.045942 |  0.2057   |   0.028583 |           6.4795e-06 |           856.14 |           876    |           877.77 |               1.1771 |
| ideal_linear  | de_rand1bin | tracking   |        0       |                  1       | 7.1231e-05 |   0.16754 |  0.045942 |  0.2057   |   0.028583 |           6.4795e-06 |           899.33 |           914.56 |           915.92 |               1.1358 |
| ideal_linear  | direct      | absolute   |        0.66667 |                  1       | 0.232      |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.13997    |           787.76 |           789.43 |           789.58 |               1.3019 |
| ideal_linear  | direct      | tracking   |        0.66667 |                  1       | 0.232      |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.13997    |           799.05 |           806.4  |           807.05 |               1.5399 |
| ideal_linear  | fft_hybrid  | absolute   |        0.66667 |                  0       | 1.2234     |   7.8333  |  9.1048   |  6.6719   |  17.031    |           0.26       |           787.29 |           979.73 |           996.83 |               1.3545 |
| ideal_linear  | fft_hybrid  | tracking   |        1       |                  0       | 2.1685     |  16.575   | 10.688    | 19.968    |  17.031    |           0.26351    |           308.57 |           419.18 |           429.02 |               3.1105 |
| ideal_linear  | local       | absolute   |        0.33333 |                  1       | 0.23203    |   3.4255  |  8.3119   |  0.031648 |  17.011    |           0.13997    |           321.18 |           373.02 |           377.63 |               2.9887 |
| ideal_linear  | local       | tracking   |        0.66667 |                  1       | 0.232      |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.13997    |           310.18 |           366.75 |           371.78 |               3.0519 |
| ideal_linear  | sobol       | absolute   |        0       |                  0.33333 | 0.44688    |   7.8325  |  9.1048   |  6.6719   |  15.869    |           0.18915    |           896.08 |          1063.2  |          1078    |               1.1235 |
| ideal_linear  | sobol       | tracking   |        0       |                  0.33333 | 0.44688    |   7.8325  |  9.1048   |  6.6719   |  15.869    |           0.18915    |           921.29 |          1061.4  |          1073.8  |               1.1079 |
| noisy_linear  | cmaes       | absolute   |        0.33333 |                  0.33333 | 0.65823    |   8.0976  |  9.8947   |  6.6837   |  12.351    |           0.16214    |           829.54 |           840.72 |           841.71 |               1.2938 |
| noisy_linear  | cmaes       | tracking   |        0.66667 |                  0       | 0.69994    |  12.469   | 10.684    | 13.319    |  12.351    |           0.14501    |           674.5  |           771.09 |           779.68 |               1.8337 |
| noisy_linear  | de_best1bin | absolute   |        0.33333 |                  0.66667 | 0.12125    |   8.5688  |  6.6783   |  6.6933   |   5.8084   |           0.030444   |           803.53 |           864.64 |           870.07 |               1.262  |
| noisy_linear  | de_best1bin | tracking   |        0.66667 |                  0.33333 | 0.2404     |  12.167   |  8.253    | 13.328    |  11.403    |           0.059276   |           744.62 |           862.8  |           873.3  |               1.6687 |
| noisy_linear  | de_rand1bin | absolute   |        0.33333 |                  0.66667 | 0.12125    |   8.5688  |  6.678    |  6.6933   |   5.8085   |           0.030444   |           872.57 |           896.77 |           898.92 |               1.1442 |
| noisy_linear  | de_rand1bin | tracking   |        0.66667 |                  0.33333 | 0.2404     |  12.167   |  8.2527   | 13.328    |  11.403    |           0.059276   |           782.61 |           894.3  |           904.23 |               1.5748 |
| noisy_linear  | direct      | absolute   |        0       |                  1       | 0.23235    |   3.4255  |  8.3119   |  0.031648 |  17.021    |           0.13987    |           825.75 |           839.64 |           840.87 |               1.2193 |
| noisy_linear  | direct      | tracking   |        0       |                  1       | 0.23235    |   3.4255  |  8.3119   |  0.031648 |  17.021    |           0.13987    |           817.89 |           824.42 |           825    |               1.2332 |
| noisy_linear  | fft_hybrid  | absolute   |        0.33333 |                  0       | 1.2233     |   7.8333  |  9.1048   |  6.6718   |  17.028    |           0.25961    |          1013.5  |          1066.1  |          1070.8  |               1.1611 |
| noisy_linear  | fft_hybrid  | tracking   |        0.33333 |                  0       | 1.223      |   7.8333  |  9.1048   |  6.6717   |  17.03     |           0.25961    |           449.03 |           877.2  |           915.26 |               1.731  |
| noisy_linear  | local       | absolute   |        0       |                  1       | 0.23234    |   3.4255  |  8.3119   |  0.031648 |  17.026    |           0.13987    |           341.59 |           354.66 |           355.82 |               2.9505 |
| noisy_linear  | local       | tracking   |        0       |                  1       | 0.23234    |   3.4255  |  8.3119   |  0.031648 |  17.026    |           0.13987    |           337.32 |           381.15 |           385.05 |               2.8767 |
| noisy_linear  | sobol       | absolute   |        0       |                  0.33333 | 0.4476     |   7.8333  |  9.011    |  6.688    |  16.342    |           0.18855    |           935.76 |          1075.7  |          1088.2  |               1.0974 |
| noisy_linear  | sobol       | tracking   |        0       |                  0.33333 | 0.4476     |   7.8333  |  9.011    |  6.688    |  16.342    |           0.18855    |           899.17 |          1051.9  |          1065.5  |               1.1264 |
| noisy_soft_l1 | cmaes       | absolute   |        0.66667 |                  0.33333 | 0.65869    |   7.2658  |  6.5998   |  8.3505   |  11.558    |           0.16227    |           689.37 |           696.19 |           696.8  |               1.4455 |
| noisy_soft_l1 | cmaes       | tracking   |        0.66667 |                  0       | 0.7        |  11.637   |  7.3888   | 14.985    |  11.558    |           0.14503    |           698.44 |           727.74 |           730.35 |               1.7598 |
| noisy_soft_l1 | de_best1bin | absolute   |        0.33333 |                  0.66667 | 0.12091    |   6.2152  |  7.208    | 10.249    |   6.0336   |           0.031066   |           804.23 |           862.92 |           868.13 |               1.2503 |
| noisy_soft_l1 | de_best1bin | tracking   |        0.66667 |                  0.33333 | 0.24013    |  11.425   |  9.874    | 15.48     |  11.394    |           0.059641   |           818.16 |           865.96 |           870.2  |               1.5988 |
| noisy_soft_l1 | de_rand1bin | absolute   |        0.33333 |                  0.66667 | 0.12957    |   8.5975  |  7.016    | 12.345    |   7.6509   |           0.035319   |           732.85 |           816.87 |           824.34 |               1.3283 |
| noisy_soft_l1 | de_rand1bin | tracking   |        0.66667 |                  0.33333 | 0.24879    |  13.807   |  9.6819   | 17.576    |  13.011    |           0.063895   |           751.96 |           836.3  |           843.79 |               1.6815 |
| noisy_soft_l1 | direct      | absolute   |        0.66667 |                  1       | 0.23366    |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.14018    |           769.35 |           799.53 |           802.21 |               1.3128 |
| noisy_soft_l1 | direct      | tracking   |        0.66667 |                  1       | 0.23366    |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.14018    |           345.14 |           764.49 |           801.77 |               2.0772 |
| noisy_soft_l1 | fft_hybrid  | absolute   |        0.66667 |                  0       | 1.2326     |   7.8333  |  9.1048   |  5.7712   |  15.708    |           0.2597     |           518.91 |           761.83 |           783.42 |               1.6916 |
| noisy_soft_l1 | fft_hybrid  | tracking   |        0.33333 |                  0       | 1.2326     |   7.8333  |  9.1048   |  5.7712   |  15.708    |           0.2597     |           385.99 |           745.52 |           777.48 |               1.9864 |
| noisy_soft_l1 | local       | absolute   |        0.33333 |                  1       | 0.2352     |   3.4254  |  8.2636   |  0.031462 |  16.054    |           0.14035    |           309.78 |           342.07 |           344.94 |               3.1845 |
| noisy_soft_l1 | local       | tracking   |        0.66667 |                  1       | 0.2352     |   3.4254  |  8.2636   |  0.031462 |  16.054    |           0.14035    |           315.55 |           330.18 |           331.48 |               3.1791 |
| noisy_soft_l1 | sobol       | absolute   |        0       |                  0.33333 | 0.44762    |   8.8442  |  7.0018   | 10.301    |  12.354    |           0.19005    |           778.6  |           856.68 |           863.62 |               1.3107 |
| noisy_soft_l1 | sobol       | tracking   |        0       |                  0.33333 | 0.44762    |   8.8442  |  7.0018   | 10.301    |  12.354    |           0.19005    |           780.84 |           899.25 |           909.77 |               1.3115 |

## 12. 延迟与吞吐率

同上表。所有方法 P50/P95/P99 均明显大于 10 ms，当前基础实现未达到 100 Hz。最快 Local 的 smoke P50 约 0.31–0.34 s，只达到约 3 spectra/s。

## 13. 首帧绝对反演时间

| condition     | algorithm   | run_mode   |   total_online_ms |
|:--------------|:------------|:-----------|------------------:|
| ideal_linear  | cmaes       | absolute   |           802.06  |
| ideal_linear  | cmaes       | tracking   |           826.846 |
| ideal_linear  | de_best1bin | absolute   |           835.536 |
| ideal_linear  | de_best1bin | tracking   |           850.944 |
| ideal_linear  | de_rand1bin | absolute   |           856.14  |
| ideal_linear  | de_rand1bin | tracking   |           916.257 |
| ideal_linear  | direct      | absolute   |           789.619 |
| ideal_linear  | direct      | tracking   |           799.046 |
| ideal_linear  | fft_hybrid  | absolute   |           426.47  |
| ideal_linear  | fft_hybrid  | tracking   |           431.474 |
| ideal_linear  | local       | absolute   |           378.783 |
| ideal_linear  | local       | tracking   |           373.038 |
| ideal_linear  | sobol       | absolute   |          1081.76  |
| ideal_linear  | sobol       | tracking   |          1076.92  |
| noisy_linear  | cmaes       | absolute   |           841.963 |
| noisy_linear  | cmaes       | tracking   |           781.822 |
| noisy_linear  | de_best1bin | absolute   |           871.428 |
| noisy_linear  | de_best1bin | tracking   |           875.927 |
| noisy_linear  | de_rand1bin | absolute   |           899.46  |
| noisy_linear  | de_rand1bin | tracking   |           906.714 |
| noisy_linear  | direct      | absolute   |           841.182 |
| noisy_linear  | direct      | tracking   |           817.89  |
| noisy_linear  | fft_hybrid  | absolute   |           498.276 |
| noisy_linear  | fft_hybrid  | tracking   |           449.027 |
| noisy_linear  | local       | absolute   |           341.592 |
| noisy_linear  | local       | tracking   |           337.318 |
| noisy_linear  | sobol       | absolute   |          1091.3   |
| noisy_linear  | sobol       | tracking   |          1068.92  |
| noisy_soft_l1 | cmaes       | absolute   |           689.152 |
| noisy_soft_l1 | cmaes       | tracking   |           731.001 |
| noisy_soft_l1 | de_best1bin | absolute   |           869.436 |
| noisy_soft_l1 | de_best1bin | tracking   |           871.267 |
| noisy_soft_l1 | de_rand1bin | absolute   |           732.851 |
| noisy_soft_l1 | de_rand1bin | tracking   |           751.957 |
| noisy_soft_l1 | direct      | absolute   |           802.88  |
| noisy_soft_l1 | direct      | tracking   |           811.084 |
| noisy_soft_l1 | fft_hybrid  | absolute   |           465.754 |
| noisy_soft_l1 | fft_hybrid  | tracking   |           385.993 |
| noisy_soft_l1 | local       | absolute   |           309.775 |
| noisy_soft_l1 | local       | tracking   |           296.322 |
| noisy_soft_l1 | sobol       | absolute   |           865.359 |
| noisy_soft_l1 | sobol       | tracking   |           912.402 |

## 14. 连续跟踪时间

| condition     | algorithm   |   tracking_p50_ms |   tracking_p95_ms |
|:--------------|:------------|------------------:|------------------:|
| ideal_linear  | cmaes       |           437.028 |           630.064 |
| ideal_linear  | de_best1bin |           471.357 |           735.58  |
| ideal_linear  | de_rand1bin |           862.532 |           895.65  |
| ideal_linear  | direct      |           574.556 |           783.947 |
| ideal_linear  | fft_hybrid  |           266.499 |           304.359 |
| ideal_linear  | local       |           304.986 |           309.656 |
| ideal_linear  | sobol       |           815.449 |           910.71  |
| noisy_linear  | cmaes       |           427.098 |           649.757 |
| noisy_linear  | de_best1bin |           460.922 |           716.25  |
| noisy_linear  | de_rand1bin |           499.141 |           754.261 |
| noisy_linear  | direct      |           807.389 |           823.374 |
| noisy_linear  | fft_hybrid  |           642.025 |           896.501 |
| noisy_linear  | local       |           352.776 |           382.7   |
| noisy_linear  | sobol       |           797.176 |           888.966 |
| noisy_soft_l1 | cmaes       |           486.851 |           677.28  |
| noisy_soft_l1 | de_best1bin |           502.545 |           786.594 |
| noisy_soft_l1 | de_rand1bin |           516.108 |           812.713 |
| noisy_soft_l1 | direct      |           316.589 |           342.286 |
| noisy_soft_l1 | fft_hybrid  |           562.138 |           763.137 |
| noisy_soft_l1 | local       |           323.674 |           330.989 |
| noisy_soft_l1 | sobol       |           687.501 |           771.507 |

## 15. 单核、多核和批量前向

以下微基准使用相同 32 个候选，不计入算法在线计时。四线程 map 在本机快于单线程；当前 NumPy 批量矩阵占用较大且反而更慢，正式部署不应直接启用该批量路径。

| backend                          |   elapsed_ms |   candidates_per_second |   max_abs_vs_reference |
|:---------------------------------|-------------:|------------------------:|-----------------------:|
| numpy_vectorized_batch           |     228.008  |                 141.229 |                      0 |
| scipy_workers_style_thread_map_4 |      45.8954 |                 698.697 |                      0 |
| single_thread_loop               |     105.902  |                 304.265 |                      0 |
| uncached_model_per_candidate     |     108.5    |                 295.485 |                      0 |

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

- 当前只有 3 条、单随机种子的 TMM smoke，不能替代至少 100 条、多种子的正式评估。
- 真实 StackRT 随机闭环因 Lumerical `Session not found` 未完成。
- smoke 全局预算过低，部分算法失败或错误阶次是预期的流程测试现象。
- 四线程 map 不是进程池部署结论，仍需在目标硬件复测。

## 20. 运行与文件

| condition     | run_dir                                                                                                                                                      |   npz_read_ms |   cold_start_ms |   disk_output_ms | dataset                                                                                                                                                                       |
|:--------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|----------------:|-----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ideal_linear  | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\benchmark_runs\20260717_125259 |        3.952  |          0.8179 |          108.671 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\datasets\static_tmm_smoke_not_stackrt.npz       |
| noisy_linear  | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\benchmark_runs\20260717_125332 |        3.7879 |          0.6428 |          104.695 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\datasets\static_tmm_smoke_noisy_not_stackrt.npz |
| noisy_soft_l1 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\benchmark_runs\20260717_125402 |        4.1473 |          0.6426 |          106.734 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\datasets\static_tmm_smoke_noisy_not_stackrt.npz |

## 21. 完整命令

```powershell
python src\validate_project.py --stackrt --stackrt-count 3
python src\generate_static_dataset.py --backend stackrt --noise-level ideal
python src\generate_static_dataset.py --backend stackrt --noise-level noisy
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_ideal.npz
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_noisy.npz --loss linear
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_noisy.npz --loss soft_l1
```
