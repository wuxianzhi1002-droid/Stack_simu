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

本次 smoke 使用全局预算 40 次量级候选评估、局部 `max_nfev=15`、Air 聚类间隔 0.1 um。预算故意很小，只验证流程，不能据此形成正式算法排名。

## 9. 公平预算

所有算法共用一个预计算 TMM、完整光谱、变量投影强度校正、边界、残差和最终局部精修器。全局算法使用尽可能接近的候选预算；实际前向调用数保留在 CSV 中。

## 10. 初值与真值隔离审计

优化器接口不接收 `air_cavity_um` 或 `film_thicknesses_nm`。真值只在优化结果完成后用于误差统计。Local 的中心点由上下界计算；其与名义值接近不代表读取真值。正确 Air 阶次阈值为 0.25 um。

## 11. 反演精度与阶次命中

| condition                 | algorithm   | run_mode   |   success_rate |   correct_air_order_rate |    Air_MAE |   HSQ_MAE |   PSS_MAE |   SOC_MAE |   TiO2_MAE |   spectral_rmse_mean |   latency_p50_ms |   latency_p95_ms |   latency_p99_ms |   spectra_per_second |
|:--------------------------|:------------|:-----------|---------------:|-------------------------:|-----------:|----------:|----------:|----------:|-----------:|---------------------:|-----------------:|-----------------:|-----------------:|---------------------:|
| stackrt_cli_ideal_linear  | cmaes       | absolute   |        0.66667 |                  0       | 0.70173    |  12.488   | 10.683    | 13.326    |  12.351    |           0.14539    |           747.5  |           924.23 |           939.94 |              1.2395  |
| stackrt_cli_ideal_linear  | cmaes       | tracking   |        0.66667 |                  0       | 0.70173    |  12.488   | 10.683    | 13.326    |  12.351    |           0.14539    |           716.77 |           856.94 |           869.4  |              1.6803  |
| stackrt_cli_ideal_linear  | de_best1bin | absolute   |        0.33333 |                  0.66667 | 0.12031    |   5.5385  |  3.5681   |  6.6877   |   5.6822   |           0.029275   |           873.94 |           892.16 |           893.77 |              1.1847  |
| stackrt_cli_ideal_linear  | de_best1bin | tracking   |        0.66667 |                  0.33333 | 0.24056    |  11.057   |  7.1293   | 13.338    |  11.361    |           0.058369   |           791.99 |           902.82 |           912.67 |              1.5779  |
| stackrt_cli_ideal_linear  | de_rand1bin | absolute   |        0       |                  1       | 7.1229e-05 |   0.16754 |  0.045941 |  0.2057   |   0.028582 |           6.4793e-06 |          1173.5  |          1237.3  |          1243    |              0.88323 |
| stackrt_cli_ideal_linear  | de_rand1bin | tracking   |        0       |                  1       | 7.1229e-05 |   0.16754 |  0.045941 |  0.2057   |   0.028582 |           6.4793e-06 |           924.85 |          1029    |          1038.2  |              1.0591  |
| stackrt_cli_ideal_linear  | direct      | absolute   |        0.66667 |                  1       | 0.232      |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.13997    |           839.73 |           934.19 |           942.59 |              1.1639  |
| stackrt_cli_ideal_linear  | direct      | tracking   |        0.66667 |                  1       | 0.232      |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.13997    |           875.25 |           911.84 |           915.1  |              1.4129  |
| stackrt_cli_ideal_linear  | fft_hybrid  | absolute   |        0.33333 |                  0       | 0.92923    |   7.833   |  9.1032   |  6.6715   |  17.021    |           0.25912    |           784.01 |          1040.6  |          1063.4  |              1.2856  |
| stackrt_cli_ideal_linear  | fft_hybrid  | tracking   |        0.33333 |                  0       | 0.92906    |   7.833   |  9.1032   |  6.6715   |  17.03     |           0.25912    |           480.43 |          1002.5  |          1048.9  |              1.5723  |
| stackrt_cli_ideal_linear  | local       | absolute   |        0.33333 |                  1       | 0.23203    |   3.4255  |  8.3119   |  0.031648 |  17.011    |           0.13997    |           354.31 |           373.67 |           375.39 |              2.8125  |
| stackrt_cli_ideal_linear  | local       | tracking   |        0.66667 |                  1       | 0.232      |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.13997    |           356.33 |           392.24 |           395.43 |              2.7464  |
| stackrt_cli_ideal_linear  | sobol       | absolute   |        0       |                  0.33333 | 0.44688    |   7.8325  |  9.1048   |  6.6719   |  15.869    |           0.18915    |           936.43 |          1243    |          1270.3  |              1.024   |
| stackrt_cli_ideal_linear  | sobol       | tracking   |        0       |                  0.33333 | 0.44688    |   7.8325  |  9.1048   |  6.6719   |  15.869    |           0.18915    |           966.17 |          1131.1  |          1145.7  |              1.0513  |
| stackrt_cli_noisy_linear  | cmaes       | absolute   |        0.66667 |                  0.33333 | 0.65885    |   8.0844  |  9.8948   |  6.6838   |  12.351    |           0.16283    |           847.78 |           883.61 |           886.8  |              1.2093  |
| stackrt_cli_noisy_linear  | cmaes       | tracking   |        0.66667 |                  0       | 0.70026    |  12.455   | 10.684    | 13.319    |  12.351    |           0.14555    |           736.38 |           860.36 |           871.38 |              1.6383  |
| stackrt_cli_noisy_linear  | de_best1bin | absolute   |        0.33333 |                  0.66667 | 0.12186    |   9.067   |  7.2027   |  6.6933   |   5.8317   |           0.030394   |           869.49 |           940.4  |           946.7  |              1.1527  |
| stackrt_cli_noisy_linear  | de_best1bin | tracking   |        0.66667 |                  0.33333 | 0.24071    |  12.167   |  8.2469   | 13.328    |  11.405    |           0.059822   |           793.36 |           910.75 |           921.19 |              1.549   |
| stackrt_cli_noisy_linear  | de_rand1bin | absolute   |        0       |                  1       | 0.001646   |   4.6968  |  4.7812   |  0.06483  |   0.20071  |           0.00112    |           976.6  |          1006.1  |          1008.7  |              1.0393  |
| stackrt_cli_noisy_linear  | de_rand1bin | tracking   |        0       |                  1       | 0.001646   |   4.6968  |  4.7812   |  0.06483  |   0.20071  |           0.00112    |          1015.1  |          1035.6  |          1037.5  |              1.0207  |
| stackrt_cli_noisy_linear  | direct      | absolute   |        0.66667 |                  1       | 0.23204    |   3.4255  |  8.3119   |  0.031648 |  17.023    |           0.14111    |           857.45 |           919.71 |           925.25 |              1.1703  |
| stackrt_cli_noisy_linear  | direct      | tracking   |        0.66667 |                  1       | 0.23204    |   3.4255  |  8.3119   |  0.031648 |  17.023    |           0.14111    |           869.5  |           899.6  |           902.27 |              1.4018  |
| stackrt_cli_noisy_linear  | fft_hybrid  | absolute   |        0.33333 |                  0       | 1.2232     |   7.8332  |  9.1045   |  6.6703   |  17.024    |           0.26168    |           832.77 |           953.28 |           964    |              1.3328  |
| stackrt_cli_noisy_linear  | fft_hybrid  | tracking   |        0.33333 |                  0       | 1.2228     |   7.8332  |  9.1045   |  6.6702   |  17.026    |           0.26169    |           451.36 |           902.62 |           942.73 |              1.699   |
| stackrt_cli_noisy_linear  | local       | absolute   |        0.33333 |                  1       | 0.23218    |   3.4255  |  8.3119   |  0.031648 |  16.924    |           0.14112    |           364.73 |           406.52 |           410.23 |              2.7302  |
| stackrt_cli_noisy_linear  | local       | tracking   |        0.66667 |                  1       | 0.23203    |   3.4255  |  8.3119   |  0.031648 |  17.03     |           0.14111    |           332.2  |           359.17 |           361.57 |              2.9822  |
| stackrt_cli_noisy_linear  | sobol       | absolute   |        0       |                  0.33333 | 0.44809    |   7.8331  |  9.1048   |  6.6719   |  16.045    |           0.18985    |          1058.6  |          1190.2  |          1201.9  |              0.98454 |
| stackrt_cli_noisy_linear  | sobol       | tracking   |        0       |                  0.33333 | 0.44809    |   7.8331  |  9.1048   |  6.6719   |  16.045    |           0.18985    |           957.6  |          1183.1  |          1203.2  |              1.0381  |
| stackrt_cli_noisy_soft_l1 | cmaes       | absolute   |        0.66667 |                  0.33333 | 0.65558    |   6.7124  |  7.2939   |  7.7774   |  14.215    |           0.16298    |           745.86 |           797.18 |           801.74 |              1.3166  |
| stackrt_cli_noisy_soft_l1 | cmaes       | tracking   |        0.66667 |                  0       | 0.69658    |  11.083   |  8.083    | 14.412    |  14.215    |           0.14558    |           730.96 |           748.07 |           749.59 |              1.7499  |
| stackrt_cli_noisy_soft_l1 | de_best1bin | absolute   |        0.33333 |                  0.66667 | 0.12272    |   8.6257  |  8.7961   |  9.0913   |   6.0056   |           0.031158   |           866.53 |           869.79 |           870.08 |              1.1851  |
| stackrt_cli_noisy_soft_l1 | de_best1bin | tracking   |        0.66667 |                  0.33333 | 0.24164    |  11.744   |  9.8718   | 15.728    |  11.583    |           0.06059    |           867.53 |           872.65 |           873.11 |              1.5457  |
| stackrt_cli_noisy_soft_l1 | de_rand1bin | absolute   |        0.33333 |                  0.66667 | 0.1302     |  10.695   |  8.6057   | 10.95     |   7.445    |           0.035029   |           851.97 |           915.74 |           921.41 |              1.1442  |
| stackrt_cli_noisy_soft_l1 | de_rand1bin | tracking   |        0.66667 |                  0.33333 | 0.24912    |  13.813   |  9.6814   | 17.587    |  13.023    |           0.064461   |           833.56 |           857.62 |           859.75 |              1.5829  |
| stackrt_cli_noisy_soft_l1 | direct      | absolute   |        0.66667 |                  1       | 0.23337    |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.14143    |           791.82 |           884.3  |           892.52 |              1.2493  |
| stackrt_cli_noisy_soft_l1 | direct      | tracking   |        0.66667 |                  1       | 0.23337    |   3.4255  |  8.3119   |  0.031648 |  17.031    |           0.14143    |           339.11 |           763.33 |           801.04 |              2.0643  |
| stackrt_cli_noisy_soft_l1 | fft_hybrid  | absolute   |        0.66667 |                  0       | 1.2376     |   7.8001  |  8.2841   |  6.6386   |  17.03     |           0.26182    |           617.52 |           715.09 |           723.77 |              1.6096  |
| stackrt_cli_noisy_soft_l1 | fft_hybrid  | tracking   |        0.66667 |                  0       | 1.2376     |   7.8001  |  8.2841   |  6.6386   |  17.03     |           0.26182    |           527.12 |           696.03 |           711.05 |              1.9839  |
| stackrt_cli_noisy_soft_l1 | local       | absolute   |        0       |                  1       | 0.23635    |   3.3052  |  8.298    |  0.026419 |  15.1      |           0.14187    |           373    |           394.24 |           396.13 |              2.6329  |
| stackrt_cli_noisy_soft_l1 | local       | tracking   |        0       |                  1       | 0.23635    |   3.3052  |  8.298    |  0.026419 |  15.1      |           0.14187    |           344.49 |           363.11 |           364.76 |              2.8979  |
| stackrt_cli_noisy_soft_l1 | sobol       | absolute   |        0.66667 |                  0.33333 | 0.44828    |   7.8333  |  9.1048   |  6.6719   |  17.009    |           0.18997    |           940.24 |          1063.8  |          1074.8  |              1.0448  |
| stackrt_cli_noisy_soft_l1 | sobol       | tracking   |        1       |                  0       | 0.61626    |  16.575   | 10.688    | 19.968    |  17.031    |           0.20093    |           218.14 |           882.87 |           941.96 |              2.2963  |

## 12. 延迟与吞吐率

同上表。所有方法 P50/P95/P99 均明显大于 10 ms，当前基础实现未达到 100 Hz。最快 Local 的 smoke P50 约 0.31–0.34 s，只达到约 3 spectra/s。

## 13. 首帧绝对反演时间

| condition                 | algorithm   | run_mode   |   total_online_ms |
|:--------------------------|:------------|:-----------|------------------:|
| stackrt_cli_ideal_linear  | cmaes       | absolute   |           943.862 |
| stackrt_cli_ideal_linear  | cmaes       | tracking   |           872.516 |
| stackrt_cli_ideal_linear  | de_best1bin | absolute   |           894.179 |
| stackrt_cli_ideal_linear  | de_best1bin | tracking   |           915.129 |
| stackrt_cli_ideal_linear  | de_rand1bin | absolute   |          1173.47  |
| stackrt_cli_ideal_linear  | de_rand1bin | tracking   |          1040.55  |
| stackrt_cli_ideal_linear  | direct      | absolute   |           944.685 |
| stackrt_cli_ideal_linear  | direct      | tracking   |           875.253 |
| stackrt_cli_ideal_linear  | fft_hybrid  | absolute   |           480.401 |
| stackrt_cli_ideal_linear  | fft_hybrid  | tracking   |           480.425 |
| stackrt_cli_ideal_linear  | local       | absolute   |           375.822 |
| stackrt_cli_ideal_linear  | local       | tracking   |           356.328 |
| stackrt_cli_ideal_linear  | sobol       | absolute   |          1277.08  |
| stackrt_cli_ideal_linear  | sobol       | tracking   |          1149.4   |
| stackrt_cli_noisy_linear  | cmaes       | absolute   |           887.592 |
| stackrt_cli_noisy_linear  | cmaes       | tracking   |           874.133 |
| stackrt_cli_noisy_linear  | de_best1bin | absolute   |           948.276 |
| stackrt_cli_noisy_linear  | de_best1bin | tracking   |           923.798 |
| stackrt_cli_noisy_linear  | de_rand1bin | absolute   |           976.605 |
| stackrt_cli_noisy_linear  | de_rand1bin | tracking   |          1037.93  |
| stackrt_cli_noisy_linear  | direct      | absolute   |           857.448 |
| stackrt_cli_noisy_linear  | direct      | tracking   |           869.498 |
| stackrt_cli_noisy_linear  | fft_hybrid  | absolute   |           451.404 |
| stackrt_cli_noisy_linear  | fft_hybrid  | tracking   |           451.362 |
| stackrt_cli_noisy_linear  | local       | absolute   |           411.161 |
| stackrt_cli_noisy_linear  | local       | tracking   |           362.171 |
| stackrt_cli_noisy_linear  | sobol       | absolute   |          1204.87  |
| stackrt_cli_noisy_linear  | sobol       | tracking   |          1208.21  |
| stackrt_cli_noisy_soft_l1 | cmaes       | absolute   |           745.86  |
| stackrt_cli_noisy_soft_l1 | cmaes       | tracking   |           749.971 |
| stackrt_cli_noisy_soft_l1 | de_best1bin | absolute   |           866.531 |
| stackrt_cli_noisy_soft_l1 | de_best1bin | tracking   |           873.222 |
| stackrt_cli_noisy_soft_l1 | de_rand1bin | absolute   |           851.968 |
| stackrt_cli_noisy_soft_l1 | de_rand1bin | tracking   |           860.288 |
| stackrt_cli_noisy_soft_l1 | direct      | absolute   |           791.822 |
| stackrt_cli_noisy_soft_l1 | direct      | tracking   |           810.465 |
| stackrt_cli_noisy_soft_l1 | fft_hybrid  | absolute   |           520.404 |
| stackrt_cli_noisy_soft_l1 | fft_hybrid  | tracking   |           527.116 |
| stackrt_cli_noisy_soft_l1 | local       | absolute   |           373.004 |
| stackrt_cli_noisy_soft_l1 | local       | tracking   |           325.57  |
| stackrt_cli_noisy_soft_l1 | sobol       | absolute   |          1077.5   |
| stackrt_cli_noisy_soft_l1 | sobol       | tracking   |           956.733 |

## 14. 连续跟踪时间

| condition                 | algorithm   |   tracking_p50_ms |   tracking_p95_ms |
|:--------------------------|:------------|------------------:|------------------:|
| stackrt_cli_ideal_linear  | cmaes       |           456.463 |           690.744 |
| stackrt_cli_ideal_linear  | de_best1bin |           493.084 |           762.1   |
| stackrt_cli_ideal_linear  | de_rand1bin |           896.051 |           921.973 |
| stackrt_cli_ideal_linear  | direct      |           624.024 |           886.72  |
| stackrt_cli_ideal_linear  | fft_hybrid  |           713.813 |          1025.8   |
| stackrt_cli_ideal_linear  | local       |           368.008 |           393.405 |
| stackrt_cli_ideal_linear  | sobol       |           852.08  |           954.76  |
| stackrt_cli_noisy_linear  | cmaes       |           478.519 |           710.591 |
| stackrt_cli_noisy_linear  | de_best1bin |           506.497 |           764.67  |
| stackrt_cli_noisy_linear  | de_rand1bin |           950.556 |          1008.65  |
| stackrt_cli_noisy_linear  | direct      |           635.312 |           876.179 |
| stackrt_cli_noisy_linear  | fft_hybrid  |           657.19  |           923.203 |
| stackrt_cli_noisy_linear  | local       |           321.907 |           331.17  |
| stackrt_cli_noisy_linear  | sobol       |           840.816 |           945.925 |
| stackrt_cli_noisy_soft_l1 | cmaes       |           482.21  |           706.083 |
| stackrt_cli_noisy_soft_l1 | de_best1bin |           533.847 |           834.164 |
| stackrt_cli_noisy_soft_l1 | de_rand1bin |           517.478 |           801.952 |
| stackrt_cli_noisy_soft_l1 | direct      |           321.389 |           337.339 |
| stackrt_cli_noisy_soft_l1 | fft_hybrid  |           492.52  |           692.572 |
| stackrt_cli_noisy_soft_l1 | local       |           354.835 |           364.142 |
| stackrt_cli_noisy_soft_l1 | sobol       |           174.844 |           213.809 |

## 15. 单核、多核和批量前向

以下微基准使用相同 32 个候选，不计入算法在线计时。四线程 map 在本机快于单线程；当前 NumPy 批量矩阵占用较大且反而更慢，正式部署不应直接启用该批量路径。

| backend                          |   elapsed_ms |   candidates_per_second |   max_abs_vs_reference |
|:---------------------------------|-------------:|------------------------:|-----------------------:|
| numpy_vectorized_batch           |     257.813  |                 124.299 |                      0 |
| scipy_workers_style_thread_map_4 |      58.0597 |                 556.478 |                      0 |
| single_thread_loop               |     117.783  |                 275.432 |                      0 |
| uncached_model_per_candidate     |     124.359  |                 258.956 |                      0 |

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

| condition                 | run_dir                                                                                                                                                      |   npz_read_ms |   cold_start_ms |   disk_output_ms | dataset                                                                                                                                                                   | generator   |
|:--------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|----------------:|-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| stackrt_cli_ideal_linear  | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\benchmark_runs\20260717_180917 |        3.9197 |          1.277  |          115.252 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\datasets\static_stackrt_cli_ideal_smoke.npz | StackRT_CLI |
| stackrt_cli_noisy_linear  | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\benchmark_runs\20260717_180954 |        4.9654 |          0.7596 |          115.754 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\datasets\static_stackrt_cli_noisy_smoke.npz | StackRT_CLI |
| stackrt_cli_noisy_soft_l1 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\benchmark_runs\20260717_181025 |        4.7147 |          1.4263 |          116.815 | D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\high_speed_static_spectral_inversion_v1\datasets\static_stackrt_cli_noisy_smoke.npz | StackRT_CLI |

## 21. 完整命令

```powershell
python src\validate_project.py --stackrt --stackrt-count 3
python src\generate_static_dataset.py --backend stackrt-cli --noise-level ideal
python src\generate_static_dataset.py --backend stackrt-cli --noise-level noisy
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_cli_ideal.npz
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_cli_noisy.npz --loss linear
python src\run_all_benchmarks.py --dataset datasets\static_stackrt_cli_noisy.npz --loss soft_l1
```
