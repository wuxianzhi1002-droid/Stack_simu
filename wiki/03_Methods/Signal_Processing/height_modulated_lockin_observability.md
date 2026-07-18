---
type: method
status: reviewed
created: 2026-07-08
updated: 2026-07-15
sources:
  - ../../../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v2.py
  - ../../../work/02_analysis_code/tmm_joint_inversion_lockin_v2.py
  - [[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]]
  - [[03_Methods/multilayer_white_light_interferometry_theory]]
  - [[01_Projects/先验约束与调制增强研究方案]]
tags:
  - lock-in
  - height-modulation
  - spectral-reflectometry
  - multilayer-film
  - inverse-problem
  - observability
---

# 高度调制锁相增强多层膜反演可观测性

## 一句话结论

在普通光谱 `I(lambda)` 之外，对参考空气腔长度施加已知正弦调制并提取带符号的一阶锁相通道，可以构造 `(I(lambda), dI/dL(lambda))` 联合观测；它的目标是增加参数敏感度约束，而不是提高光谱仪本身的分辨率。当前仿真已完成全流程闭环，但 `A=5 nm` 下必须用有限幅值锁相 forward model，不能继续把局部中心差分当作无偏替代。

## 适用场景

- 测量结构中存在参考反射面或外部空气腔，腔长变化会改变可观测干涉相位。
- 待估参数包括空气腔长、HSQ/PSS/SOC/Hard Mask 等膜厚，后续可扩展少量材料参数。
- 正向模型使用与数据生成一致的 TMM/StackRT，而不是只依赖单一余弦玩具模型。
- 采样能够覆盖整数个调制周期，并已知调制幅值、频率和参考相位。

如果系统只是无参考臂的普通反射率测量，整体样品刚体平移不会改变反射率强度，此时不能把顶面绝对高度直接写成可辨识参数。

## 为什么 toy model 不足以证明可辨识性

抽象模型

$$
I(\lambda)=0.5+0.4\cos\left(\frac{4\pi(nd+z)}{\lambda}\right)
$$

适合演示锁相数学，但只依赖组合量 `nd+z`。`z` 和 `d` 的 Jacobian 方向天然共线，加入 `dI/dz` 也不能自动把两者分离。

真实模型应写为：

$$
I(\lambda)=F\left(\lambda;L_{\mathrm{air}},d_1,d_2,\ldots,n_i(\lambda),k_i(\lambda),c_{\mathrm{sys}}\right),
$$

其中空气腔相位、各膜层内部传播相位、Fresnel 系数和吸收项通过不同路径进入模型。只有不同参数的 Jacobian 不完全共线时，额外锁相通道才可能改善可观测性。

## 数字锁相定义

调制空气腔长度：

$$
L(t)=L_0+A\sin(2\pi f_{\mathrm{mod}}t).
$$

先对每个波长去直流，再计算第 `h` 次谐波：

$$
X_h(\lambda)=\frac{2}{N_t}\sum_t I_{\mathrm{AC}}(t,\lambda)
\sin(2\pi h f_{\mathrm{mod}}t),
$$

$$
Y_h(\lambda)=\frac{2}{N_t}\sum_t I_{\mathrm{AC}}(t,\lambda)
\cos(2\pi h f_{\mathrm{mod}}t).
$$

幅值和相位为：

$$
R_h=\sqrt{X_h^2+Y_h^2},\qquad
\phi_h=\operatorname{atan2}(Y_h,X_h).
$$

小信号极限下：

$$
X_{1f}\approx A\frac{\partial I}{\partial L}.
$$

因此联合反演优先使用：

$$
D_{\mathrm{meas}}(\lambda)=\frac{X_{1f}(\lambda)}{A}.
$$

`R_1f/A` 只表示响应幅值并且恒为非负，适合显示信号强弱，不适合代替有符号导数。若实验存在机械或电子相位延迟，应先由参考标定旋转 `X/Y` 坐标，再确定物理同相通道。

## 小信号与有限幅值

Taylor 展开为：

$$
I(L_0+A\sin\omega t)
=I(L_0)+A I'(L_0)\sin\omega t
+\frac{A^2}{2}I''(L_0)\sin^2\omega t+\cdots.
$$

当 `A` 足够小时，`X_1f/A` 接近局部一阶导数。随着 `A` 增大：

- 时间平均光谱不再等于 `I(L0)`。
- 1f 不再严格等于 `A*I'(L0)`。
- 2f、3f 开始携带可测的非线性响应。

因此有两种正确建模方式：

1. 小幅值方案：先验证 `A` 足够小，再用中心差分近似导数。
2. 有限幅值方案：对每个候选参数模拟完整 `L(t)`，用与数据完全相同的锁相算法生成模型观测。

当前 `A=5 nm` 数据应采用第二种方式。

## 联合反演残差

设参数向量为：

$$
\theta=[L_{\mathrm{air}},d_{\mathrm{HSQ}},d_{\mathrm{PSS}},d_{\mathrm{SOC}},d_{\mathrm{TiO2}}].
$$

联合残差建议写为：

$$
r(\theta)=
\begin{bmatrix}
(I_{\mathrm{model}}-I_{\mathrm{meas}})/\sigma_I\\
(D_{\mathrm{model}}-D_{\mathrm{meas}})/\sigma_D\\
(\theta-\theta_0)/\sigma_\theta
\end{bmatrix}.
$$

其中 `sigma_I` 和 `sigma_D` 应来自实验噪声、重复测量或明确的 robust scale。两个通道量纲不同，不能未经归一化直接拼接。

有限幅值模型应使用：

$$
I_{\mathrm{model}}(\lambda)=\langle I(\lambda,L(t);\theta)\rangle_t,
$$

$$
D_{\mathrm{model}}(\lambda)=\frac{X_{1f,\mathrm{model}}(\lambda)}{A}.
$$

## 推荐对照组

| 对照 | 输入 | 目的 |
|---|---|---|
| I-only | `I(lambda)` | 光谱基线 |
| D-only | `X_1f(lambda)/A` | 判断调制通道独立信息量 |
| joint | `I + X_1f/A` | 联合观测目标方法 |
| joint + prior | 联合观测与工艺先验 | 当前最可能落地的方案 |

所有对照必须使用相同 bounds、相同真值分布和可比的多起点策略。还应分别报告有先验和无先验结果，避免把先验中心等于真值时的高精度误判为观测本身的能力。

## 评价指标

- 各参数偏差、MAE、RMSE 和失败率。
- I 与 D 两个物理通道各自的未归一化残差。
- Jacobian 奇异值、条件数和参数相关系数。
- 多起点落入不同局部极小值的比例。
- 真值偏离名义参数后能否仍恢复正确解。
- 调制幅值扫描下的 1f/2f/3f 比例和非线性偏差。
- 加入噪声、波长漂移、线展宽、材料色散误差后的鲁棒性。

## 当前仿真状态

详细证据、参数、三组对照和问题定位见：

[[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]]

截至 2026-07-14：

- StackRT 动态光谱与 1f/2f/3f 锁相保存已完成。
- StackRT 与独立 TMM 单时刻光谱可在浮点误差级闭环。
- 联合反演 v2 已采用匹配的 TMM 约定，并修正未收敛尝试被选为最优解的问题。
- v2 的 `A=1 nm` 和 `A=5 nm` rank 1 都使用了与仿真真值相同的首个初值，并启用了真值中心软先验；这些结果只能作为局部数值闭环，不能作为盲反演精度。
- 排除真值起点后，其余随机起点普遍进入错误局部极小值，表明 8 个随机起点和当前一次性 least-squares 流程不足。
- `A=5 nm` 还存在有限幅值时间平均/Fourier 观测与静态光谱/局部中心差分模型不一致的问题。
- v3 必须隔离真值、取消真值起点、增加起点数，并采用能够跨越 1 mm 腔密集相位局部极小值的分阶段初始化或全局搜索。

v3 已于 2026-07-15 完成上述修正：每种模式使用独立差分进化生成候选池，再对 32 个非真值起点做局部精修；当前理想闭环中三种模式均达到 `32/32` 收敛。详见：

[[TMM_Joint_Inversion_Lockin_v3_20260715]]

## 参数与数据规范

- 调制幅值必须写入数据文件；若未保存，可由 `0.5*ptp(L_t)` 反算并交叉检查。
- 频率、采样率、相位零点和窗口长度必须随数据保存。
- 膜层名称、厚度、材料模型和频率轴约定必须进入元数据。
- 反演使用手动指定 NPZ 时，仍应检查命令行幅值是否与 NPZ 元数据或 `L_t` 一致。
- `spectra[i]` 必须与同索引 `L_t[i]` 配对，不能把任意时间点当作静态中心位置。

## 已知风险

- 多层膜逆问题仍可能有多解；额外通道增加约束，但不保证唯一解。
- 材料 `n,k` 自由度过高时，会吸收膜厚或空气腔误差。
- 1f 的符号依赖参考相位；实验系统需要相位标定。
- 非整数周期、采样抖动、机械传递函数和光谱仪积分时间会改变锁相比例。
- 当前闭环无实验噪声，不能直接当作真实精度指标。

## 来源路径

- `../../../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v2.py`
- `../../../work/02_analysis_code/tmm_joint_inversion_lockin_v2.py`
- `../../../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/dynamic_spectra_20260714_161153.npz`
- `../../../work/04_results_and_datasets/tmm_joint_inversion_lockin_v2_20260714_165223/`
- [[04_Experiments/Simulation_Reports/Dynamic_StackRT_TMM_Single_Spectrum_20260713]]
- [[01_Projects/先验约束与调制增强研究方案]]

## 待验证问题

- 完整有限幅值 forward model 是否能让 `A=5 nm` 的 joint 回到真值。
- 最佳调制幅值及 2f/3f 的实际利用方式。
- 在真值偏离先验中心、加入噪声后，联合观测是否降低多解率和参数相关性。
