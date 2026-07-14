---
type: method
status: draft
created: 2026-07-07
updated: 2026-07-07
sources:
  - https://arxiv.org/abs/1603.02720
  - https://arxiv.org/abs/2111.13667
  - https://arxiv.org/abs/2511.21383
  - https://arxiv.org/abs/2005.04552
  - https://arxiv.org/abs/2409.17199
  - https://en.wikipedia.org/wiki/Coherence_scanning_interferometry
tags:
  - multilayer-film
  - white-light-interferometry
  - tmm
  - spectral-reflectometry
  - inverse-problem
---

# 多层膜白光干涉测厚理论推导

## 一句话结论

多层膜白光测厚通常不是逐层直接读峰，而是用宽谱反射/白光干涉信号作为观测量，用多层膜 TMM 正向模型和材料库约束反演层厚。`0.5 nm` 多层膜精度只有在层序、材料色散、待拟合参数数量、光谱范围、信噪比和校准条件受控时才有物理意义。

## 参考文献定位

| 文献                                                                            | 本文采用的理论作用                        |
| ----------------------------------------------------------------------------- | -------------------------------- |
| Byrnes, `Multilayer optical calculations`                                     | 多层膜 TMM 正向模型、相干/非相干厚层处理、吸收介质注意事项 |
| Luce et al., `TMM-Fast`                                                       | TMM 的批量化、梯度化和优化问题表述              |
| Ziapkoff et al., `White light interferometry analysis...`                     | 白光谱到薄膜厚度的自动推断、谱范围/分辨率/折射率已知性限制   |
| Suja et al., `Hyperspectral imaging for dynamic thin film interferometry`     | 每个像素采集完整光谱并用谱形恢复厚度，避免依赖绝对强度      |
| Ma, Ma, Guo, `Optical Multilayer Thin Film Structure Inverse Design...`       | 多层膜逆问题的非唯一性、传统优化与深度学习框架          |
| Harasaki, Schmit, Wyant, `Improved vertical-scanning interferometry` 及 CSI 文献 | 扫描白光干涉的包络/相位模型，说明其与谱反射测厚的关系和差异   |

## 1. 单界面 Fresnel 反射

先考虑第 `j` 层和第 `j+1` 层的平面界面。复折射率写作：

$$
N_j(\lambda)=n_j(\lambda)+i\,k_j(\lambda)
$$

入射角满足 Snell 定律：

$$
N_j\sin\theta_j=N_{j+1}\sin\theta_{j+1}
$$

定义光学导纳。常用形式为：

$$
\begin{aligned}
q_j^{s} &= N_j\cos\theta_j \\
q_j^{p} &= \frac{N_j}{\cos\theta_j}
\end{aligned}
$$

这里的 `q` 采用薄膜特征矩阵中的归一化光学导纳约定。若使用 Byrnes/tmm 常见的 `gamma_s=N cos(theta)`、`gamma_p=cos(theta)/N`，反射率仍可正确计算，但那是另一套符号约定，不能与本文的导纳矩阵公式混用。`r_p` 的振幅符号也会随约定变化，强度反射率 `R=|r|^2` 不受影响。`r` 可写成统一形式：

$$
r_{j,j+1}=\frac{q_j-q_{j+1}}{q_j+q_{j+1}}
$$

垂直入射时 `cos(theta)=1`，s/p 等价：

$$
r_{j,j+1}=\frac{N_j-N_{j+1}}{N_j+N_{j+1}}
$$

单界面强度反射率为：

$$
R=|r|^2
$$

如果介质有吸收，`N` 为复数，不能只用实数折射率；PSS、Cr、Cu 这类材料的 `k(lambda)` 会直接改变反射率幅度和条纹可见度。

## 2. 单层膜的干涉周期

空气 / 薄膜 / 基底的最简单结构中，反射光主要来自上表面和下表面。单次往返相位为：

$$
\delta=\frac{2\pi}{\lambda}N_f d\cos\theta_f
$$

两束反射光的相位差近似为：

$$
\begin{aligned}
\Phi(\lambda) &= 2\delta+\phi_r \\
&= \frac{4\pi N_f d\cos\theta_f}{\lambda}+\phi_r
\end{aligned}
$$

其中 `phi_r` 是界面反射相位。反射极值近似满足：

$$
\Phi(\lambda_m)=2\pi m
$$

若用波数：

$$
\sigma=\frac{1}{\lambda}
$$

则相位为线性形式：

$$
\Phi(\sigma)=4\pi N_f d\cos\theta_f\,\sigma+\phi_r
$$

相邻极值在波数域的间隔【是否可以理解为光谱周期】：

$$
\Delta\sigma\approx\frac{1}{2d\,\operatorname{Re}\{N_f\cos\theta_f\}}
$$

因此单层膜厚可由条纹周期估计：

$$
d\approx\frac{1}{2\operatorname{Re}\{N_f\cos\theta_f\}\Delta\sigma}
$$

上述周期估计默认使用相位传播常数。透明或弱吸收材料中，`operatorname{Re}{N_f cos(theta_f)}` 可近似为 `n_f cos(theta_f)`。在波长域，局部近似为：

$$
\Delta\lambda\approx\frac{\lambda^2}{2d\,\operatorname{Re}\{N_f\cos\theta_f\}}
$$

这解释了两个现象：

- `30-100 nm` 薄膜在 400-800 nm 窗口内只出现宽谷/宽峰，条纹周期通常大于观测窗口；
- `1 mm` 空气腔在 600 nm 附近周期约 `0.18 nm`，需要 `0.01-0.05 nm` 级采样才可靠。

## 3. 多层膜 TMM 正向模型

多层膜不能简单用两束光叠加，因为每个界面都产生反射和透射，形成无限多条光路。TMM 将这些多次反射合并为矩阵乘法。

第 `j` 层厚度为 `d_j`，相位厚度：

$$
\delta_j=\frac{2\pi}{\lambda}N_j d_j\cos\theta_j
$$

特征矩阵：

$$
M_j=\begin{bmatrix}
\cos\delta_j & \dfrac{i\sin\delta_j}{q_j} \\
iq_j\sin\delta_j & \cos\delta_j
\end{bmatrix}
$$

整个膜栈的系统矩阵为：

$$
M=M_1M_2\cdots M_L=\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
$$

设入射介质导纳为 `q_0`，基底导纳为 `q_s`。从膜栈入口看进去的等效导纳可由场边界条件直接推出。

在任意位置定义切向场向量：

$$
\mathbf{v}=\begin{bmatrix}E\\H\end{bmatrix}
$$

其中 `H/E` 用归一化光学导纳表示。整个膜栈矩阵满足：

$$
\begin{bmatrix}E_0\\H_0\end{bmatrix}
=M
\begin{bmatrix}E_s\\H_s\end{bmatrix}
=
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}E_s\\H_s\end{bmatrix}
$$

基底是半无限介质，内部只有向前传播波，没有从无穷远返回的反射波。因此基底侧满足：

$$
H_s=q_sE_s
$$

代入系统矩阵：

$$
\begin{aligned}
E_0 &= AE_s+BH_s \\
    &= (A+Bq_s)E_s \\
H_0 &= CE_s+DH_s \\
    &= (C+Dq_s)E_s
\end{aligned}
$$

膜栈入口等效导纳定义为入口处总切向磁场与总切向电场之比：

$$
Y\equiv\frac{H_0}{E_0}
$$

所以：

$$
Y=\frac{C+Dq_s}{A+Bq_s}
$$

物理含义是：整个多层膜加半无限基底可以从入射侧等效成一个导纳为 `Y` 的单一负载。之后只需计算入射介质 `q_0` 与等效负载 `Y` 的界面反射。

总反射振幅：

$$
r(\lambda)=\frac{q_0-Y}{q_0+Y}
$$

总反射率：

$$
R(\lambda)=|r(\lambda)|^2
$$

这就是反射式膜厚仪、谱反射计、多层膜计算器的核心正向模型。对吸收层和金属层，`N_j` 必须使用复数；对很厚层，需判断相干还是非相干。如果厚度远大于光源相干长度或样品厚度非均匀导致相位平均，干涉项应被部分或完全平均。

## 4. 白光谱反射测厚模型

实际仪器测到的不是纯 `R(lambda)`，而是：

$$
I_{\mathrm{meas}}(\lambda)=S(\lambda)E(\lambda)R(\lambda;\mathbf{p})+I_{\mathrm{dark}}(\lambda)+\varepsilon(\lambda)
$$

其中：

- `S(lambda)`：光源谱；
- `E(lambda)`：光路、光栅、探测器响应；
- `p`：待拟合参数，如层厚、粗糙度、折射率模型参数；
- `epsilon`：噪声。

通常要用参考谱和暗电流归一化：【这里参考谱如何得到？是否可以使用一个已知膜层参数做标定】

$$
\hat{R}(\lambda)=\frac{I_{\mathrm{sample}}(\lambda)-I_{\mathrm{dark}}(\lambda)}{I_{\mathrm{ref}}(\lambda)-I_{\mathrm{dark}}(\lambda)}\,R_{\mathrm{ref}}(\lambda)
$$

反演目标函数：

$$
\begin{aligned}
\min_{\mathbf{p}}\quad &\chi^2(\mathbf{p}) \\
\chi^2(\mathbf{p}) &= \sum_i w_i\left[\hat{R}(\lambda_i)-R_{\mathrm{TMM}}(\lambda_i;\mathbf{p})\right]^2
\end{aligned}
$$

若绝对强度不可靠，可引入尺度和偏置：

$$
\min_{\mathbf{p},a,b}\sum_i\left[\hat{R}_i-aR_{\mathrm{TMM}}(\lambda_i;\mathbf{p})-b\right]^2
$$

也可以只拟合谱形，例如减均值/归一化后比较：

$$
\tilde{R}=\frac{R-\operatorname{mean}(R)}{\operatorname{std}(R)}
$$

这与高光谱薄膜干涉工作中的“避免依赖绝对强度”思想一致：厚度信息主要来自谱形、峰谷位置和相位演化，而不仅是强度幅值。

## 5. 扫描白光干涉 CSI 与谱反射测厚的关系

扫描白光干涉或 CSI 的典型信号是沿 `z` 扫描时的强度：

$$
I(z)=\int S(k)\left[I_r(k)+I_s(k)+2\sqrt{I_r(k)I_s(k)}\,\operatorname{Re}\left\{\gamma(k)e^{i2kz}\right\}\right]dk
$$

其中 `k = 2 pi / lambda`，`z` 是参考臂与样品臂的光程差变量。若样品是简单表面，包络峰位置对应表面高度。若样品是透明薄膜或多层膜，样品反射不是单一界面，而是复反射系数：

$$
r_s(k;\mathbf{p})=r_{\mathrm{TMM}}(k;\mathbf{p})
$$

扫描信号变为：

$$
I(z;\mathbf{p})=I_0+2\operatorname{Re}\left\{\int S(k)r_{\mathrm{ref}}^{*}(k)r_{\mathrm{TMM}}(k;\mathbf{p})e^{i2kz}\,dk\right\}
$$

这说明：

- CSI 的 `z` 域信号本质上是**光谱复反射的傅里叶型投影**；
- 多层膜会产生多个界面贡献和重叠包络；
- 若各界面 OPD 分离大于相干长度，可以看到多个包络峰；
- 若膜层很薄，各界面包络重叠，必须用模型拟合，不能直接用峰位置读厚度。

因此“白光干涉仪测多层膜”有两种实现路线：

1. 频域/谱域：测 `R(lambda)` 或干涉谱，TMM 拟合层厚；
2. 扫描域/CSI：测 `I(z)`，用 TMM 生成的复反射谱预测扫描干涉图，再拟合。

厂商如果宣称多层膜 `0.5 nm` 精度，核心通常不是传统单峰包络定位，而是模型反演。

## 6. 多层膜反演的可辨识性

设离散测量向量为：

$$
\mathbf{y}=F(\mathbf{p})+\boldsymbol{\varepsilon}
$$

其中：

$$
\mathbf{p}=\left[d_1,d_2,\ldots,d_L,\text{material\_params},\text{roughness},\ldots\right]^T
$$

局部线性化：

$$
\mathbf{y}\approx F(\mathbf{p}_0)+J\Delta\mathbf{p}+\boldsymbol{\varepsilon}
$$

Jacobian：

$$
J_{i,j}=\frac{\partial F_i}{\partial p_j}
$$

最小二乘近似解：

$$
\Delta\hat{\mathbf{p}}=(J^TWJ)^{-1}J^TW\left[\mathbf{y}-F(\mathbf{p}_0)\right]
$$

参数协方差近似为：

$$
\operatorname{Cov}(\hat{\mathbf{p}})\approx\sigma^2(J^TWJ)^{-1}
$$

因此多层膜能否达到 `0.5 nm` 取决于 `J^T W J` 的条件数。如果两层对谱的影响相似，Jacobian 两列高度相关：

$$
\operatorname{corr}(J_{:a},J_{:b})\to 1
$$

则厚度会互相补偿，产生多解或大置信区间。实际工程中必须降低自由度：

- 固定层序；
- 固定大部分材料 `n,k(lambda)`；
- 固定或约束部分层厚；
- 使用工艺先验范围；
- 使用更宽光谱范围、多角度或偏振数据；
- 对拟合结果输出残差、置信区间和参数相关矩阵。

## 7. 优化、梯度和深度学习

TMM-Fast 类工作将正向模型批量化：

$$
R=F(\lambda_{\mathrm{grid}},\theta_{\mathrm{grid}},\mathrm{polarization},\mathbf{p})
$$

对多组 `lambda/theta/p` 并行计算，可用于：

- 全局优化；
- 遗传算法；
- 梯度优化；
- 训练数据生成；
- 神经网络 surrogate model。

若 TMM 由可微框架实现，可直接计算：

$$
\frac{\partial\chi^2}{\partial\mathbf{p}}
$$

用于局部优化：

$$
\mathbf{p}_{t+1}=\mathbf{p}_t-\eta\nabla_{\mathbf{p}}\chi^2
$$

但多层膜逆问题通常非唯一。深度学习逆设计文献强调的问题是：

$$
\mathbf{p}\to R(\lambda)
$$

是多对一映射。不同膜系可能产生接近相同的光谱。因此对测厚反演而言，神经网络不能消除物理不可辨识性，只能在训练分布和先验约束内给出一个可能解或候选解分布。

更稳妥的形式是概率反演：

$$
P(\mathbf{p}\mid\mathbf{y})\propto P(\mathbf{y}\mid\mathbf{p})P(\mathbf{p})
$$

其中 `P(p)` 来自工艺先验，`P(y|p)` 来自测量噪声模型。若后验分布很宽或多峰，就不应报告单一 `0.5 nm` 精度。

## 8. 对 NS-20 类设备声明的解释框架

若设备用宽谱白光、材料库和 TMM 反演，多层膜 `0.5 nm` 精度在受控样品上并非违反物理原理。但它应被理解为条件性指标：

$$
\operatorname{accuracy}=f\!\left(\begin{array}{l}
\text{known layer order},\ \text{known }n,k\text{ database},\\
\text{spectral range},\ \text{spectral resolution},\ \mathrm{SNR},\\
\text{calibration},\ \text{number of free parameters},\\
\text{layer sensitivity},\ \text{parameter correlation}
\end{array}\right)
$$

适合相信的场景：

- 单层或少数关键层自由拟合；
- 其余层厚和材料已知；
- 材料透明或 `n,k` 数据可靠；
- 光谱覆盖 UV/VIS/NIR，能提供足够相位信息；
- 有标准片、椭偏或截面测量交叉验证。

需要质疑的场景：

- 50-100 层全部未知且同时自由拟合；
- 金属/吸收层 `k(lambda)` 不准；
- 粗糙界面和梯度折射率未建模；
- 层厚强相关但没有报告置信区间；
- 只给单一精度数字，不说明样品、材料、层数和校准方法。

## 9. 与当前 STACK_simu 项目的建模建议

当前项目的 film stack 可按两级模型处理：

1. 快速趋势模型：
   - 使用 TMM；
   - 固定材料近似 `n,k`；
   - 扫 400-800 nm；
   - 观察宽谷、宽峰和敏感层。

2. 定量反演模型：
   - 使用复折射率 `n,k(lambda)`；
   - 对 PSS、Cr、Cu 等吸收层单独建库；
   - 引入外部参考面时必须显式加入参考反射；
   - 对 `1 mm` 空气腔使用 `0.01-0.05 nm` 或等效波数均匀采样；
   - 输出参数协方差、相关矩阵和多解检查。

对厂商设备评估时，建议索取：

- 多层膜 `0.5 nm` 精度对应的测试样品结构；
- 每层材料和厚度范围；
- 是否固定 `n,k`；
- 是否同时拟合所有层；
- 标准片或第三方校准报告；
- 残差、置信区间、参数相关矩阵；
- 对金属层、吸收层、粗糙界面的模型说明。

## 参考链接

- Steven J. Byrnes, `Multilayer optical calculations`: https://arxiv.org/abs/1603.02720
- Alexander Luce et al., `TMM-Fast`: https://arxiv.org/abs/2111.13667
- Victor Ziapkoff et al., `White light interferometry analysis for measuring thin film thickness down to few nanometers`: https://arxiv.org/abs/2511.21383
- Vineeth Chandran Suja et al., `Hyperspectral imaging for dynamic thin film interferometry`: https://arxiv.org/abs/2005.04552
- Taigao Ma, Mingqian Ma, L. Jay Guo, `Optical Multilayer Thin Film Structure Inverse Design`: https://arxiv.org/abs/2409.17199
- Coherence scanning interferometry reference entry including Harasaki/Schmit/Wyant: https://en.wikipedia.org/wiki/Coherence_scanning_interferometry




