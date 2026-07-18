---
type: experiment
status: reviewed
created: 2026-07-15
updated: 2026-07-15
sources:
  - ../../../work/02_analysis_code/tmm_joint_inversion_lockin_v3.py
  - ../../../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/dynamic_spectra_20260714_161153.npz
  - ../../../work/04_results_and_datasets/tmm_joint_inversion_lockin_v3_20260715_112455/
tags:
  - experiment
  - tmm
  - lock-in
  - joint-inversion
  - global-optimization
  - multistart
---

# TMM 锁相联合反演 v3：无真值起点的全局到局部验证

## 一句话结论

v3 已移除真值起点并默认关闭先验，使用每种模式独立的差分进化全局种群生成 32 个局部起点，同时用有限幅值 TMM 复现时间平均和 1f 测量算子；在当前无噪声同模型数据上，I、D、joint 三种模式均实现 `32/32` 个局部结果收敛到同一真值盆地，且精确真值起点数量为 `0`。

## 为什么需要 v3

v2 的每种模式都执行：

```python
guesses = [initial_vector(), random_guess_1, ...]
```

其中 `initial_vector()` 恰好等于仿真数据真值 `1000/30/10/40/40`，并且 `use_prior=True` 时又把同一向量作为软先验中心。v2 rank 1 因此属于真值辅助局部闭环；排除该起点后，其余 7 个随机结果普遍偏离真值。

v3 的目标不是简单删除第一行，而是同时解决：

1. 真值信息泄漏。
2. 起点数量不足。
3. 1 mm 空气腔导致的大量相位局部极小值。
4. `A=5 nm` 有限幅值观测与静态/小信号模型不一致。
5. 理想闭环下 `soft_l1` cost 下溢为零而导致 rank 无法区分。

## 输入数据与参数

输入：

`../../../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/dynamic_spectra_20260714_161153.npz`

| 参数 | 数值 |
|---|---:|
| Air 真值 | `1000 µm` |
| HSQ/PSS/SOC/TiO2 真值 | `30/10/40/40 nm` |
| 调制幅值 | `5 nm` |
| 拟合波段 | `220-580 nm` |
| 局部拟合 stride | `10` |
| 全局搜索 stride | `50` |
| 先验 | 默认关闭 |

当前 bounds：

| 参数 | 下限 | 上限 |
|---|---:|---:|
| Air | `998 µm` | `1002 µm` |
| HSQ | `20 nm` | `40 nm` |
| PSS | `1 nm` | `20 nm` |
| SOC | `30 nm` | `50 nm` |
| TiO2 | `30 nm` | `50 nm` |

## 真值隔离规则

代码中保留 `EVALUATION_TRUTH` 仅用于拟合全部结束后的误差计算和起点审计。

它不参与：

- 差分进化初始种群。
- 候选筛选。
- least-squares 初值构造。
- residual。
- 默认先验。
- rank 排序。

`postfit_initialization_audit` 在所有模式完成优化后才读取真值，并把审计结果写入 `fit_summary.json`。

## v3 求解流程

### 1. 有限幅值观测模型

对每组候选参数直接模拟：

$$
L(\phi)=L_0+A\sin\phi.
$$

模型光谱和一阶锁相信号为：

$$
I_{\mathrm{model}}(\lambda)=\frac{1}{N_\phi}\sum_\phi I(\lambda,L(\phi)),
$$

$$
D_{\mathrm{model}}(\lambda)=
\frac{2}{A N_\phi}\sum_\phi I(\lambda,L(\phi))\sin\phi.
$$

全局搜索使用 `8` 个相位点，局部精修使用 `16` 个相位点。真值处 `16` 相位模型相对原 40 点/周期 StackRT 数据的误差已达到数值误差量级。

### 2. 每种模式独立全局搜索

I、D、joint 分别执行自己的差分进化，不能共享 joint 搜索结果：

- 初始化：Latin-hypercube population。
- `global_popsize=16`。
- 参数数 `5`，实际 population 为 `80`。
- `global_maxiter=60`。
- 不插入 nominal 或 truth 向量。

全局阶段只负责生成候选池。SciPy 的 differential-evolution 在本次运行中因达到 `60` 轮上限而报告 `success=false`，但这不代表最终反演失败；最终结果只从后续 `success=true` 的局部 least-squares 中选择。

### 3. 增加并精修起点

从每种模式的全局种群中选取 energy 最低的 `32` 个不同候选，分别执行局部 least-squares：

- `multistarts=32`。
- `max_nfev=300`。
- `loss=soft_l1`。
- `use_prior=false`。

### 4. rank 规则

局部结果按以下规则排序：

1. `success=true` 排在失败结果之前。
2. 成功状态相同时，按原始归一化 residual 的平方和排序。

v3 仍保存 SciPy 的 `optimizer_robust_cost`，但不再用它排序。原因是无噪声闭环下 residual 接近机器精度，`soft_l1` cost 会数值下溢为精确的 `0`。

## 最终输出

最终结果目录：

`../../../work/04_results_and_datasets/tmm_joint_inversion_lockin_v3_20260715_112455/`

主要文件：

- `fit_results.csv`
- `multistart_results.csv`
- `fit_summary.json`
- `summary_report.md`
- `best_fit_spectrum.png`
- `best_fit_dIdL.png`
- `joint_residual.png`
- `jacobian_singular_values.png`

## rank 1 结果

| 模式 | Air (`µm`) | HSQ (`nm`) | PSS (`nm`) | SOC (`nm`) | TiO2 (`nm`) | `RMSE_I` | `RMSE_dIdL` |
|---|---:|---:|---:|---:|---:|---:|---:|
| I | 1000.000000 | 30.000000 | 10.000000 | 40.000000 | 40.000000 | `1.41e-12` | `5.13e-10` |
| D | 1000.000000 | 30.000000 | 10.000000 | 40.000000 | 40.000000 | `5.09e-12` | `4.73e-10` |
| joint | 1000.000000 | 30.000000 | 10.000000 | 40.000000 | 40.000000 | `2.81e-12` | `4.69e-10` |

> 原 `best_fit_spectrum.png` 图片引用已失效；对应结果目录当前不存在，保留文字结论，不再嵌入截图。

> 原 `best_fit_dIdL.png` 图片引用已失效；对应结果目录当前不存在，保留文字结论，不再嵌入截图。

## 全部 rank 分布

| 模式 | 成功局部结果 | 精确真值起点 | 边界结果 | 32 个结果最大 Air 误差 | 32 个结果最大膜厚误差 |
|---|---:|---:|---:|---:|---:|
| I | `32/32` | `0` | `0` | 数值精度内 | `6.7e-7 nm` |
| D | `32/32` | `0` | `0` | 数值精度内 | `5.5e-7 nm` |
| joint | `32/32` | `0` | `0` | 数值精度内 | `3.72e-7 nm` |

拟合前候选到真值的最小 bounds 归一化距离：

| 模式 | 最小距离 | 最大距离 |
|---|---:|---:|
| I | `0.00695` | `0.09136` |
| D | `0.06125` | `0.36735` |
| joint | `0.02672` | `0.17265` |

这证明 32 个局部优化并不是从精确真值开始。全局搜索先利用观测把候选种群推进到正确盆地，随后局部优化从不同候选收敛到同一解。

## 与 v2 的差异

| 项目 | v2 | v3 |
|---|---|---|
| 首个起点 | 精确仿真真值 | 不插入任何 nominal/truth 起点 |
| 默认先验 | 开启且中心等于真值 | 关闭 |
| 起点数 | 8 | 32 个局部起点，来自 80 个体全局种群 |
| 全局搜索 | 无 | 每种模式独立差分进化 |
| I 模型 | 静态 `I(L0)` | 有限幅值时间平均 |
| D 模型 | 中心差分 | 同一正弦序列的数字 1f |
| rank cost | SciPy robust cost | 稳定的原始 residual 平方和 |
| rank 2 以后 | 多数远离真值 | 32/32 收敛到同一真值盆地 |

## 事实、推断与限制

### 已确认事实

- v3 的优化前和优化中不读取仿真真值。
- 默认运行不使用软先验。
- 三种模式各自独立生成候选池。
- 32 个起点中没有精确真值起点。
- 所有 96 个局部拟合均 `success=true`，没有边界解。
- 有限幅值 forward model 消除了 v2 在 `A=5 nm` 下的观测算子偏差。

### 当前可以得出的结论

在当前 bounds、无噪声、材料模型完全一致的理想闭环中，v3 已能从非真值候选池稳定找到正确参数盆地，v2 的起点泄漏和局部极小值问题得到修正。

### 仍不能得出的结论

- 不能据此宣称真实实验可达到 `1e-6 nm` 精度；这些极小误差只是同模型无噪声闭环的数值结果。
- 不能宣称 joint 已优于 I-only，因为当前三种模式全部精确恢复，测试没有形成可区分的难度。
- 不能证明 bounds 扩大、真值偏离 bounds 中心、加入噪声和材料失配后仍有 `32/32` 收敛率。
- 全局种群已经利用观测向正确区域演化，因此“所有局部 rank 接近”证明的是全局到局部流程有效，不代表任意均匀随机初值都可直接收敛。

## 下一步验证

1. 生成多组真值，不让真值固定在 bounds 中心或工艺名义值上。
2. 加入光谱噪声、锁相噪声、波长漂移和材料 `n/k` 偏差。
3. 扩大 Air 和膜厚 bounds，统计全局阶段找到正确盆地的比例。
4. 在相同计算预算下比较 I、D、joint 的成功率、参数误差、置信区间和条件数。
5. 对不同随机种子重复运行，不能只依赖单个 `20260715` 种子。
6. 将调制幅值、频率和膜栈元数据写入 NPZ，避免手动同步。

### 2026-07-16 噪声数据生成准备

已创建 `../../../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v4.py`，提供 `clean/low/medium/high` 四档材料 `n/k`、角度、波长零点、调制幅值、帧增益和反射率读出噪声。NPZ 会保存名义值、实际抽样值和干净物理 1f 参考。

当前仅完成编译、CLI 和 mock StackRT 保存链路验证。真实运行在启动 `lumapi.FDTD(hide=True)` 时出现 `Session not found`，因此 `dynamic_stackrt_lockin_v4` 正式 NPZ 尚未生成，后续恢复 Lumerical 会话后再进入噪声反演对照。

## 来源路径

- v3 代码：`../../../work/02_analysis_code/tmm_joint_inversion_lockin_v3.py`
- 输入 NPZ：`../../../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/dynamic_spectra_20260714_161153.npz`
- 最终输出：`../../../work/04_results_and_datasets/tmm_joint_inversion_lockin_v3_20260715_112455/`
- v2 问题与历史：[[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]]
- 方法页：[[03_Methods/Signal_Processing/height_modulated_lockin_observability]]
