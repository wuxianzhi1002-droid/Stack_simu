---
type: method
status: draft
created: 2026-07-08
updated: 2026-07-08
sources:
  - 本次对话：2026-07-08 高度调制锁相信号反演思路
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

在普通宽谱反射/干涉观测 `I(lambda)` 之外，同步调制样品顶面高度 `z(t)=z0+A sin(omega t)` 并锁相提取 `dI/dz`，可以把单一光谱观测扩展为 `(I(lambda), dI/dz(lambda))` 的联合观测；对 HSQ/SOC/Hard Mask/Si 这类多层膜反演，它的研究价值在于增加对顶面高度、膜厚和材料参数的局部敏感度约束，而不是单纯提高光谱分辨率。

## 适用场景

本方法面向当前研究中的多层薄膜反演问题：

- 样品结构：空气 / HSQ / SOC / Hard Mask / Si，或其仿真替代材料栈。
- 待反演参数：顶面高度 `z_surface`、膜厚 `d_HSQ`、`d_SOC`、`d_HM`，以及必要时的材料折射率/消光系数参数。
- 正向模型：优先使用 TMM/StackRT 或经 Lumerical/FDTD 校准过的等效 forward model，而不是只用单一余弦玩具模型。
- 观测数据：普通光谱 `I(lambda)` 和高度调制后的锁相信号 `S_lockin(lambda)`。

## 必要修正

用户给出的抽象模型：

$$
I(\lambda)=0.5+0.4\cos\left(rac{4\pi(n d+z)}{\lambda}
ight)
$$

适合说明锁相提取导数的数学流程，但不适合作为证明 `z` 和 `d` 可同时唯一反演的物理模型。原因是该模型只依赖组合量 `n d+z`，所以 `z` 和 `d` 在这个简化模型中天然耦合；即使额外加入 `dI/dz`，导数仍然只依赖同一个相位组合量，不能真正打破退化。

在真实研究中，需要把 forward model 改成更接近测量系统的形式：

$$
I(\lambda)=F(\lambda; z_{surface}, d_{HSQ}, d_{SOC}, d_{HM}, n_i(\lambda), k_i(\lambda), c_{sys})
$$

其中 `z_surface` 应通过参考臂、外部空气腔或白光干涉测量几何进入相位；膜厚则通过各层内部相位、界面 Fresnel 系数和吸收项进入 TMM。只有当 `z` 和各层 `d_i` 对光谱的 Jacobian 方向不完全共线时，联合观测才会提高可观测性。

## 信号模型

高度调制为：

$$
z(t)=z_0+A\sin(\omega t)
$$

当 `A` 足够小，且系统在一个调制周期内近似线性时：

$$
I(\lambda,t) pprox I(\lambda,z_0)+Arac{\partial I}{\partial z}(\lambda,z_0)\sin(\omega t)
$$

用同频参考 `r(t)=sin(omega t)` 做数字锁相：

$$
S(\lambda)=rac{1}{N}\sum_t I(\lambda,t)r(t)
$$

若参考信号与调制同相、采样覆盖整数周期，则：

$$
S(\lambda) pprox rac{A}{2}rac{\partial I}{\partial z}
$$

因此：

$$
rac{\partial I}{\partial z}(\lambda) pprox rac{2}{A}S(\lambda)
$$

工程实现时建议同时计算正交通道：

$$
X=\langle I(t)\sin\omega t
angle,\quad Y=\langle I(t)\cos\omega t
angle
$$

这样可以检查相位延迟、触发不同步和机械响应滞后；如果存在相位差，需要用 `sqrt(X^2+Y^2)` 或相位校正后的 `X`，不能直接把单一 sin 通道当作绝对导数。

## 反演流程

1. 采集或仿真普通光谱：`I_meas(lambda)`。
2. 对顶面高度施加小幅正弦调制，采集时间序列：`I(lambda,t)`。
3. 对每个波长点做数字锁相，得到 `S_lockin(lambda)`。
4. 用标定后的调制幅值 `A` 把锁相信号换算成 `dIdz_meas(lambda)`。
5. 用同一个 forward model 计算：
   - `I_model(lambda; theta)`
   - `dIdz_model(lambda; theta)`，可用中心差分、自动微分或解析导数。
6. 联合拟合参数 `theta=[z_surface,d_HSQ,d_SOC,d_HM,n_i...]`。
7. 对比 `I` 单独反演和 `(I,dI/dz)` 联合反演的误差、退化方向和置信区间。

## 代码骨架

```python
import numpy as np
from scipy.optimize import least_squares


def forward_tmm(wavelength_nm, params):
    """Return modeled spectrum for the current multilayer stack.

    params should include z_surface, layer thicknesses, and compact
    material parameters. In production, call StackRT/TMM here.
    """
    raise NotImplementedError


def lockin_demodulate(I_time, t, fm_hz, amplitude_nm):
    ref_x = np.sin(2 * np.pi * fm_hz * t)
    ref_y = np.cos(2 * np.pi * fm_hz * t)

    x = I_time.T @ ref_x / len(t)
    y = I_time.T @ ref_y / len(t)

    dIdz_x = 2 * x / amplitude_nm
    dIdz_y = 2 * y / amplitude_nm
    return dIdz_x, dIdz_y


def numerical_dIdz(wavelength_nm, params, dz_nm=0.05):
    p_plus = dict(params)
    p_minus = dict(params)
    p_plus["z_surface"] += dz_nm
    p_minus["z_surface"] -= dz_nm
    return (
        forward_tmm(wavelength_nm, p_plus)
        - forward_tmm(wavelength_nm, p_minus)
    ) / (2 * dz_nm)


def residual_vector(x, wavelength_nm, I_meas, dIdz_meas, sigma_I, sigma_dz):
    params = unpack_params(x)
    I_model = forward_tmm(wavelength_nm, params)
    dIdz_model = numerical_dIdz(wavelength_nm, params)

    return np.concatenate([
        (I_model - I_meas) / sigma_I,
        (dIdz_model - dIdz_meas) / sigma_dz,
    ])


result = least_squares(
    residual_vector,
    x0=x0,
    bounds=(lower_bounds, upper_bounds),
    args=(wavelength_nm, I_meas, dIdz_meas, sigma_I, sigma_dz),
)
```

注意：`sigma_I` 和 `sigma_dz` 必须按实验噪声或仿真噪声标定。否则两个 residual 的量纲和幅值不同，优化器可能只拟合其中一个观测量。

## 与当前 HSQ/SOC/Hard Mask/Si 研究的对接

推荐的第一版参数化：

```python
params = {
    "z_surface": z_nm,
    "d_HSQ": d_hsq_nm,
    "d_SOC": d_soc_nm,
    "d_HM": d_hm_nm,
    "mat_HSQ": compact_material_params_hsq,
    "mat_SOC": compact_material_params_soc,
    "mat_HM": compact_material_params_hm,
}
```

材料参数不建议直接把每个波长点的 `n(lambda)`、`k(lambda)` 都作为自由参数。更稳妥的做法是：

- 第一阶段：固定材料库，只反演 `z_surface` 和膜厚。
- 第二阶段：给每种材料增加少量低维参数，例如 `n_offset`、`n_slope`、`k_scale`。
- 第三阶段：把材料参数的先验范围写入 bounds 或 Bayesian prior，避免把光谱噪声解释成材料色散。

## 验证指标

为了证明“调制增强可观测性”确实有效，建议至少做四组对照：

| 对照 | 输入 | 目标 |
|---|---|---|
| baseline | `I(lambda)` | 复现现有光谱反演性能 |
| lockin only | `dI/dz(lambda)` | 检查导数通道本身携带的信息 |
| joint | `I(lambda)+dI/dz(lambda)` | 目标方法 |
| joint + prior | `I(lambda)+dI/dz(lambda)+工艺先验` | 当前研究最可能落地的方案 |

核心评价不只看 MAE，还要看：

- Jacobian/Fisher 矩阵的条件数是否下降。
- `z_surface` 与 `d_i` 的参数相关性是否降低。
- 多解样本的候选解数量是否减少。
- 在 `z` 先验误差 `<=10 nm`、膜厚工艺扰动和材料参数扰动下是否仍稳定。

## 实验参数口径

- 调制幅值 `A`：应小于光谱相位非线性显著变化的尺度，先从 `5-20 nm` 扫描；过大会引入二阶项，过小会被噪声吞没。
- 调制频率 `fm`：应低于光谱采集 Nyquist 频率，并避开机械共振、环境振动和电源干扰频率。
- 采样时间：覆盖整数个调制周期，减少锁相泄漏。
- 光谱预处理：暗场扣除、白板/参考归一化、坏点掩膜、波长标定必须在锁相前后保持一致。
- 差分步长 `dz_nm`：数值求导时需做步长收敛测试，例如 `0.01, 0.05, 0.1 nm`。

## 已知风险

- 如果测量系统只是普通反射率，没有参考臂或外部相位参考，整体样品高度平移不会改变强度反射率；此时 `z_surface` 不会自然出现在 `I(lambda)` 中。
- 单余弦 toy model 中 `n d+z` 完全耦合，不能用来证明多参数可辨识，只能用来演示锁相数学。
- 锁相信号的比例系数依赖调制幅值、参考相位、采样窗口和归一化约定，必须实验标定。
- 多层膜逆问题仍可能有多解；`dI/dz` 是增加约束，不是保证唯一解。
- 若材料 `n,k` 自由度过高，联合观测可能被材料参数吸收，反而降低膜厚反演稳定性。

## 下一步实现建议

1. 在当前 TMM/StackRT forward model 上增加一个 `z_surface` 或参考腔长度参数，明确它对应真实光路中的哪一段 OPD。
2. 生成仿真数据集：同一组膜厚和材料参数下，输出 `I(lambda)`、中心差分 `dI/dz(lambda)`、以及不同 `A` 和噪声水平下的模拟锁相信号。
3. 用 `least_squares` 做小规模可辨识性验证，再扩展到 Residual MLP/CNN 输入通道。
4. 报告中把研究贡献表述为“通过主动高度调制改变测量可观测性”，而不是“用更复杂网络拟合原始光谱”。

## 来源路径

- 本页直接来源：2026-07-08 对“高度调制 + 数字锁相 + 联合反演”的方法讨论。
- 理论连接：[[03_Methods/multilayer_white_light_interferometry_theory]]。
- 研究方案连接：[[01_Projects/先验约束与调制增强研究方案]]。
