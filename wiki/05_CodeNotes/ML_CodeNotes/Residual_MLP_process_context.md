# 当前仿真与机器学习路线总结

## 1. 当前仿真目标

当前核心问题是：

**多层膜厚度存在约 ±10 nm 工艺不确定性时，如何从反射光谱中高精度解算空气腔长 \(L\)。**

目前使用 Lumerical `stackrt` 生成多层膜结构的反射光谱。已有 `main_cavity.py` 中定义了 `PSS_TIO2_MODEL`，基本层结构为：

```text
RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu
```

其中：

- `Air` 是待扫描的空气腔长层；
- `HSQ`、`PSS`、`SOC`、`TiO2` 是存在工艺厚度扰动的膜层；
- 腔长单位主要使用 `um`；
- 膜层厚度在训练数据中统一转为 `nm`；
- 光谱由 `lumapi.FDTD(hide=True)` 和 `fdtd.stackrt(...)` 计算得到。

仿真输出的基本量为：

```text
cavity_axis_um
cavity_axis_m
wavelengths
spectra
```

---

## 2. 已发现的问题：固定线性标定不鲁棒

当前 FFT 粗解流程为：

```text
I(lambda) -> I(k) -> FFT -> L_fft
```

然后用名义工艺做线性拟合：

\[
L_\mathrm{true}=aL_\mathrm{FFT}+b
\]

但你已经发现：**改变膜层厚度后，线性拟合的斜率和截距都会变化。**

因此真实关系更接近：

\[
L_\mathrm{FFT}=a(\mathbf d)L_\mathrm{true}+b(\mathbf d)
\]

其中：

\[
\mathbf d=[d_\mathrm{HSQ}, d_\mathrm{PSS}, d_\mathrm{SOC}, d_\mathrm{TiO2}]
\]

这说明：

> 单一固定线性标定参数无法适配不同工艺膜厚扰动。

---

## 3. 已讨论的物理方法：带膜厚先验的联合拟合

为处理膜厚 ±10 nm 的不确定性，讨论过带先验约束的联合拟合方案。

对于同一个 process 下的一组腔长扫描，拟合变量为：

\[
x=[L_1,L_2,\cdots,L_N,\Delta d_1,\Delta d_2,\cdots,\Delta d_M]
\]

其中：

- 每个腔长扫描点有自己的 \(L_i\)；
- 同一个 process 下的膜厚扰动 \(\Delta d_i\) 是共享参数；
- 膜厚扰动有先验约束。

目标函数包括光谱残差和膜厚先验残差：

\[
r=[
r_\mathrm{spec},
\lambda_\mathrm{prior}r_\mathrm{prior}
]
\]

膜厚先验项为：

\[
r_{\mathrm{prior},i}=\frac{\Delta d_i}{\sigma_{d_i}}
\]

若 ±10 nm 被视作约 \(3\sigma\)，则：

\[
\sigma_d=\frac{10}{3}\ \mathrm{nm}
\]

这个方案的意义是：

> 同一工艺 process 下，膜厚扰动应该作为共享隐变量，而不是每条光谱独立拟合膜厚。

但真实使用中，如果每次 least-squares 都调用 Lumerical，速度会很慢。因此它更适合作为物理 baseline，而不是最终快速部署模型。

---

## 4. 当前机器学习数据集设计

当前已经转向构建机器学习数据集，用残差模型学习：

\[
[I(\lambda),L_\mathrm{FFT},H_\mathrm{peak},\mathbf d_\mathrm{nom}]
\rightarrow
\Delta L
\]

其中：

\[
\Delta L_\mathrm{nm}=(L_\mathrm{true}-L_\mathrm{FFT})\times1000
\]

最终腔长预测为：

\[
L_\mathrm{pred}
=
L_\mathrm{FFT}
+
\frac{\Delta L_\mathrm{pred,nm}}{1000}
\]

数据集中建议包含：

```text
wavelengths_um
spectra
spectra_norm
L_true_um
L_fft_um
delta_L_um
delta_L_nm
H_peak
film_nominal_nm
film_true_nm
film_delta_nm
process_id
nominal_stack_id
sample_id
layer_names
split_label
```

---

## 5. 哪些字段可以作为模型输入？

关键原则：

> 最终可部署模型只能使用真实实验中可获得的输入。

| 字段 | 是否作为最终模型输入 | 说明 |
|---|---:|---|
| `spectra_norm` | 是 | 完整归一化光谱，包含膜厚扰动相关信息 |
| `L_fft_um` | 是 | FFT 物理粗解 |
| `H_peak` | 可选 | FFT 峰强度特征 |
| `film_nominal_nm` | 是 | 工艺名义膜厚，真实实验可知 |
| `film_true_nm` | 否 | 真实测量中通常无法知道 |
| `film_delta_nm` | 否 | 真实扰动未知 |
| `cavity_true_um` | 否 | 标签 |
| `delta_L_nm` | 标签 | 神经网络预测目标 |

因此主模型输入应为：

\[
[I(\lambda),L_\mathrm{FFT},H_\mathrm{peak},\mathbf d_\mathrm{nom}]
\]

而不是：

\[
[I(\lambda),L_\mathrm{FFT},H_\mathrm{peak},\mathbf d_\mathrm{true}]
\]

`film_true_nm` 和 `film_delta_nm` 可以保存，但只能用于：

1. 数据生成记录；
2. 分组误差分析；
3. oracle 理论上限对照；
4. 判断膜厚扰动对误差的影响。

---

## 6. 当前数据集结构

目前数据集结构是：

```text
每 20 个 process 对应一个 nominal thickness group
```

即：

\[
d_\mathrm{true}=d_\mathrm{nom}+\Delta d
\]

其中：

\[
\Delta d_i\in[-10,10]\ \mathrm{nm}
\]

这很好地模拟了真实情况：

- 名义膜厚 \(d_\mathrm{nom}\) 可知；
- 实际膜厚 \(d_\mathrm{true}\) 不可知；
- 膜厚扰动 \(\Delta d\) 不可知。

因此最终训练时不能把 `film_true_nm` 作为可部署模型输入。

---

## 7. 当前 Residual MLP 模型

当前 `train_residual_mlp.py` 的主模型策略已经改为：

```text
主模型输入只使用 nominal thickness，不使用 true/delta thickness
```

当前主要模型包括以下几类。

### 7.1 `scalar_baseline`

输入：

```text
L_fft_um
H_peak
PSS_nominal_nm
HSQ_nominal_nm
SOC_nominal_nm
TiO2_nominal_nm
```

输出：

```text
delta_L_nm
```

这是当前主 baseline。

### 7.2 `scalar_selected_quadratic`

输入：

```text
L_fft_um
H_peak
nominal thickness
L_fft_um * each nominal thickness
```

可选加入：

```text
H_peak * each nominal thickness
```

它是少量物理启发的二阶交互项。

其中比较有物理意义的是：

\[
L_\mathrm{FFT}\cdot d_i
\]

因为它表示某一层膜厚对残差的影响可能随腔长位置变化。

### 7.3 `scalar_all_quadratic`

自动生成全部二阶特征，仅作为消融实验。

如果基础输入之间本身强相关，自动二阶特征可能导致：

1. 冗余；
2. 多重共线性；
3. 过拟合；
4. 解释困难。

因此它不能作为默认最终方案，必须看 test set 是否真正提升。

### 7.4 `scalar_oracle_true_thickness`

oracle 对照模型，输入真实膜厚：

```text
L_fft_um
H_peak
PSS_true_nm
HSQ_true_nm
SOC_true_nm
TiO2_true_nm
```

以及：

```text
L_fft_um * each true thickness
```

这个模型不可部署，只能作为理论上限。

当前 `metrics.json` 中已经出现 `scalar_oracle_true_thickness`，说明 oracle 确实被训练和评估了。

你贴出的 oracle test 指标为：

```text
test MAE  = 5.03 nm
test RMSE = 5.99 nm
test Max  = 27.73 nm
test P95  = 10.35 nm
```

这说明：

1. oracle 确实跑了；
2. 真实膜厚作为输入后可以达到约 5 nm MAE；
3. 但需要和 nominal 模型对比，才能判断真实膜厚提供了多少额外信息。

---

## 9. 数据划分策略

后续应明确区分两类测试。

### 9.1 `process_within_nominal`

同一个名义膜厚组合可以同时出现在 train/val/test 中，但具体 process 不重叠。

测试的是：

\[
\text{已见名义工艺下，对未见膜厚扰动 process 的泛化}
\]

这对应最直接的实际问题：

> 工艺名义值已知，但每片实际膜厚存在 ±10 nm 随机波动。

### 9.2 `nominal_holdout`

整个 nominal group 进入同一个 split。

测试的是：

\[
\text{对未见过名义膜厚组合的泛化}
\]

这个更严格，也更难。

建议修复 split bug 后，分别跑：

```bash
python train_residual_mlp.py --enable-oracle true --split-strategy process_within_nominal --epochs 120
```

和：

```bash
python train_residual_mlp.py --enable-oracle true --split-strategy nominal_holdout --epochs 120
```

---

## 10. 下一步重点：加入完整光谱特征

目前标量 MLP 的输入主要是：

\[
[L_\mathrm{FFT},H_\mathrm{peak},\mathbf d_\mathrm{nom}]
\]

但在同一个 nominal thickness 下，不同真实扰动：

\[
\Delta \mathbf d
\]

在输入中的 \(\mathbf d_\mathrm{nom}\) 完全相同。

因此，如果不输入完整光谱，模型最多只能学习“平均补偿”，很难区分具体膜厚扰动。

真正包含膜厚扰动信息的是完整光谱：

\[
I(\lambda)
\]

所以后续最重要的方向是：

\[
[I(\lambda),L_\mathrm{FFT},H_\mathrm{peak},\mathbf d_\mathrm{nom}]
\rightarrow
\Delta L
\]

---

## 11. 后续光谱特征输入方案

### 11.1 方案 A：PCA 光谱特征 + MLP

输入：

```text
PCA(spectra_norm)
L_fft_um
H_peak
film_nominal_nm
```

输出：

```text
delta_L_nm
```

流程：

1. 使用 `spectra_norm`；
2. PCA 只在 train set 上 fit；
3. val/test 只能 transform；
4. PCA 分量数可做消融，例如 20、50、100；
5. 将 PCA 特征与标量特征拼接后训练 MLP。

模型形式：

\[
X=
[
PC_1,\cdots,PC_k,
L_\mathrm{FFT},
H_\mathrm{peak},
\mathbf d_\mathrm{nom}
]
\]

\[
y=\Delta L_\mathrm{nm}
\]

这是最推荐的下一步，因为它实现简单、训练稳定，也能判断完整光谱是否提供了额外信息。

### 11.2 方案 B：手工光谱特征 + MLP

从光谱中提取更具物理意义的特征。

#### 光谱统计特征

```text
mean
std
min
max
skewness
kurtosis
```

#### 频域特征

对 \(I(k)\) 做 FFT，提取：

```text
dominant peak position
dominant peak height
peak width
second peak position
second peak height
peak height ratio
local noise floor
spectral centroid
band energy
```

当前 `H_peak` 只是其中一个特征，后续可以扩展成一组 FFT 特征。

#### 相位/包络特征

如果可行，可以提取：

```text
instantaneous phase slope
unwrapped phase residual
Hilbert envelope
fringe contrast
spectral visibility
```

这些特征可能比单个 `H_peak` 更能反映膜层扰动。

### 11.3 方案 C：1D CNN + 标量融合

输入：

```text
spectra_norm
```

经过 1D CNN 提取谱形特征：

\[
\mathrm{CNN}(I(\lambda))\rightarrow z_\mathrm{spec}
\]

再与标量特征拼接：

```text
L_fft_um
H_peak
film_nominal_nm
```

输出：

```text
delta_L_nm
```

即：

\[
[z_\mathrm{spec},L_\mathrm{FFT},H_\mathrm{peak},\mathbf d_\mathrm{nom}]
\rightarrow
\Delta L
\]

这是后续主力方向，但建议在 PCA 方案证明有效后再做。

---

## 12. 推荐的模型对照表

| 模型 | 输入 | 目的 |
|---|---|---|
| raw FFT | `L_fft_um` | 无补偿 baseline |
| mean residual | train set 平均残差 | 最弱统计 baseline |
| scalar baseline | `L_fft_um`, `H_peak`, `d_nom` | 当前主模型 |
| selected quadratic | scalar + 手选二阶项 | 检查物理交互是否有帮助 |
| all quadratic | scalar + 全二阶项 | 消融实验，不默认采用 |
| oracle true | `L_fft_um`, `H_peak`, `d_true` | 理论上限，不可部署 |
| spectrum PCA nominal | `PCA(I(lambda)) + scalar` | 下一步重点 |
| 1D CNN nominal | `I(lambda) + scalar` | 后续主力 |

---

## 13. 当前最建议做的三件事

### 第一优先级：修复训练代码

1. 将：

```python
split_strategy="nominal_holdout"
```

改成：

```python
split_strategy=args.split_strategy
```

2. 将 `scalar_oracle_true_thickness` 加入方法对比柱状图。

3. 在 oracle 分支打印：

```text
max |film_true_nm - film_nominal_nm|
mean |film_true_nm - film_nominal_nm|
```

### 第二优先级：分别跑两种 split

```bash
python train_residual_mlp.py --enable-oracle true --split-strategy process_within_nominal --epochs 120
```

```bash
python train_residual_mlp.py --enable-oracle true --split-strategy nominal_holdout --epochs 120
```

### 第三优先级：增加 `spectrum_pca_nominal`

新增模型：

```text
spectrum_pca_nominal
```

输入：

```text
PCA(spectra_norm, n_components=50)
L_fft_um
H_peak
film_nominal_nm
```

输出：

```text
delta_L_nm
```

如果 `spectrum_pca_nominal` 明显优于 `scalar_baseline` / `selected_quadratic`，说明完整光谱确实携带了膜厚扰动信息，是后续应该重点发展的方向。

---

## 14. 一句话总结

当前路线已经从：

```text
固定线性标定
```

推进到：

```text
多工艺数据集 + residual MLP 补偿
```

主模型输入应该只使用真实可获得的：

\[
[I(\lambda),L_\mathrm{FFT},H_\mathrm{peak},d_\mathrm{nom}]
\]

不能使用：

\[
d_\mathrm{true},\Delta d
\]

作为可部署输入。

接下来最重要的不是继续堆二阶标量特征，而是把完整光谱 `spectra_norm` 通过 PCA 或 1D CNN 加入模型，让网络从谱形中识别 ±10 nm 膜厚扰动带来的系统误差。
