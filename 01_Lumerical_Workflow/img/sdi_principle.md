# 谱域干涉 (SDI) 与 FFT 腔长解算原理

在 `fft_solve.py` 中，我们利用光谱仪获取的干涉光谱来提取光学腔的物理厚度。以下是详细的数学推导与转换过程。

## 1. 干涉信号模型 (Wavelength Domain)

对于一个厚度为 $d$、折射率为 $n$ 的法布里-珀罗 (FP) 腔，其反射光谱的强度 $I(\lambda)$ 可以简化表示为：

$I(\lambda) = A(\lambda) + B(\lambda) \cos\left( \frac{2\pi}{\lambda} \cdot \text{OPD} \right)$

其中：

- $\text{OPD} = 2nd$ 是双程光程差 (Optical Path Difference)。
- $A(\lambda)$ 为直流分量（背景强度）。
- $B(\lambda)$ 为干涉条纹的对比度（调制度）。
- $\lambda$ 是真空波长。

## 2. 波长域到波数域的转换 (Wavelength to Wavenumber)

光谱仪在波长域 $\lambda$ 是等间距采样的，但干涉信号的相位 $\phi = k \cdot \text{OPD}$ 在波长域是非线性的。为了使用 FFT，必须将其转换到线性相位域——波数域 $k$：

$k = \frac{2\pi}{\lambda}$

转换后的信号变为：
$I(k) = A(k) + B(k) \cos(k \cdot \text{OPD})$

此时，干涉信号在 $k$ 域表现为一个频率为 $\text{OPD}$ 的正弦振荡。

## 3. 重采样与线性化 (Resampling & Linearization)

由于 $k = 2\pi / \lambda$ 是非线性映射，直接转换后的 $k$ 点不是等间距的。代码中执行了以下步骤：

1. 计算 $k$ 的范围：$[k_{\min}, k_{\max}] = [2\pi/\lambda_{\max}, 2\pi/\lambda_{\min}]$。
2. 在该范围内创建一个等间距的向量 $k_{linear}$。
3. 使用 `np.interp` 将原始强度值插值到 $k_{linear}$ 上。

## 4. 傅里叶变换 (FFT Processing)

对去直流并加窗（如 Hanning 窗）后的信号 $I(k_{linear})$ 进行快速傅里叶变换 (FFT)：

$\mathcal{F}\{I(k)\} \rightarrow \text{Peak at } \xi$

在这里，由于自变量是波数 $k$（单位：$\mu m^{-1}$），共轭变量 $\xi$ 的单位就是长度 ($\mu m$)，其物理意义正是 **光程差 OPD**。

## 5. 腔长解算 (Distance Calculation)

### 5.1 深度轴计算

FFT 结果的频率分辨率由 $k$ 域的采样间隔 $\Delta k$ 决定。最大的解算范围（奈奎斯特极限）为：

$\text{OPD}_{\max} = \frac{\pi}{\delta k}
$  其中$\delta k$是 k 域的采样步长。

### 5.2 物理距离转换

由于 OPD 代表光在腔内往返的总路程，实际的物理腔长（或膜层厚度）$d$ 为：

$$d = \frac{\text{OPD}}{2n}$$

在代码中，假设空气腔 $n \approx 1$，因此：
$$\text{distance\_axis} = \frac{\text{depth\_axis (OPD)}}{2}$$

## 6. 总结流程

1. **输入**: $I(\lambda)$ (光谱仪数据)
2. **映射**: $\lambda \rightarrow k$ (波数域)
3. **线性化**: 等间距插值 $I(k)$
4. **变换**: FFT 提取振荡频率
5. **输出**: 峰值位置对应的 $\text{OPD}/2 = d$ (物理厚度)
