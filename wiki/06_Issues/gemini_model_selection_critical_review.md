# 针对 Gemini 神经网络选择建议的批判性分析

## 1. 总体判断

Gemini 给出的建议在通用机器学习框架上是合理的：

- 表格型多参数输入适合 MLP；
- 连续一维信号适合 1D-CNN；
- 强时间依赖序列适合 LSTM / GRU / Transformer；
- 光谱序列 + 标量参数的组合适合双分支混合网络。

但是，对于当前这个 StackRT 光谱残差补偿任务来说，它的建议仍然偏泛化，没有充分考虑以下关键约束：

1. 当前任务是 **residual learning**，不是直接预测腔长；
2. 模型输入必须区分 **真实实验可获得输入** 和 **仿真中才知道的 oracle 输入**；
3. 真实实验中存在光源功率、探测器增益和耦合效率漂移，因此不应过度依赖绝对强度特征；
4. 数据划分必须按 `process_id`，而不是按 sample 随机划分；
5. PCA 必须只在 train set 上 fit，val/test 只能 transform；
6. 1D-CNN / ResNet-1D 应该在 PCA 证明完整光谱有效之后再上。

因此，Gemini 的建议可以作为模型架构方向参考，但不能直接作为当前实验路线。

---

## 2. 当前任务的数据形态

当前数据集是典型的混合输入结构：

### 2.1 序列输入

```text
spectra_norm_ds
shape = (800000, 2000)
```

这是降采样后的归一化反射光谱，属于一维连续信号。

### 2.2 标量输入

```text
L_fft_um
film_nominal_nm
peak_count
robust spectral features
```

其中：

- `L_fft_um` 是 FFT 物理粗解；
- `film_nominal_nm` 是可部署模型中可获得的名义膜厚；
- `peak_count` 和部分 robust spectral features 描述光谱形状或 FFT 质量。

### 2.3 标签

模型不直接预测 `cavity_true_um`，而是预测：

```text
delta_L_nm = (L_true_um - L_fft_um) * 1000
```

最终腔长预测为：

```text
L_pred_um = L_fft_um + delta_L_pred_nm / 1000
```

这意味着当前任务本质上是：

```text
[I(lambda), L_fft_um, film_nominal_nm] -> delta_L_nm
```

而不是：

```text
I(lambda) -> L_true_um
```

这是和普通光谱回归任务的重要区别。

---

## 3. 对 Gemini 建议的逐条评估

## 3.1 MLP 用于表格特征：正确，但不能停在这里

Gemini 认为相互独立的表格型特征适合 MLP，这一点是正确的。

当前以下模型都可以使用 MLP：

```text
scalar_no_intensity:
    L_fft_um + film_nominal_nm

scalar_with_hpeak:
    L_fft_um + H_peak + film_nominal_nm

selected_quadratic:
    scalar + L_fft_um * film_nominal_nm

robust_scalar_features:
    L_fft_um + film_nominal_nm + robust spectral features
```

但是纯 scalar 模型有一个根本限制：

同一个 nominal group 下，不同 process 的 `film_nominal_nm` 完全相同，而真实膜厚扰动不同。仅靠 `L_fft_um + film_nominal_nm`，模型最多只能学习平均补偿，很难区分具体 process 的膜厚扰动。

因此，MLP scalar 是必要 baseline，但不应作为最终主路线。

---

## 3.2 1D-CNN 用于光谱序列：方向正确，但优先级不应最高

Gemini 认为 1D-CNN 擅长提取波峰、波谷、干涉条纹等局部特征，这一点对光谱数据是合理的。

后续如果使用 1D-CNN，合理结构应为：

```text
CNN(spectra_norm_ds) -> z_spec
[z_spec, L_fft_um, film_nominal_nm, robust_features] -> delta_L_nm
```

而不是：

```text
CNN(spectra_norm_ds) -> L_true_um
```

也就是说，CNN 应该服务于 residual correction，而不是从零学习腔长。

但是当前不建议直接上 CNN，原因是：

1. CNN 训练和调参更复杂；
2. 解释性弱于 PCA；
3. 当前首先需要验证归一化光谱是否真的提供了 scalar features 之外的信息；
4. PCA + MLP 可以更快回答这个问题。

因此合理优先级应为：

```text
PCA + MLP 先行
1D-CNN / ResNet-1D 后续
```

---

## 3.3 LSTM / GRU / Transformer：当前不推荐

Gemini 将强时间依赖序列对应到 LSTM / GRU / Transformer，这对真正的时间序列预测是合理的。

但当前的 `spectra_norm_ds` 是按波长排列的反射光谱：

```text
I(lambda)
```

它不是时间序列，也不存在“前一时刻决定后一时刻”的动态记忆问题。

相邻波长点之间确实有局部相关性，但这种结构更适合：

```text
PCA
1D-CNN
ResNet-1D
```

而不是：

```text
LSTM
GRU
Transformer
```

Transformer 不是不能做，但对于当前 2000 点光谱、80 万样本的数据来说，它会显著增加复杂度和算力需求，而且解释性更弱。当前阶段不建议使用。

---

## 3.4 混合网络：方向正确，但属于后续阶段

Gemini 提到的双分支/多模态网络与当前数据形态匹配：

```text
光谱序列分支：1D-CNN 或 ResNet-1D
标量参数分支：L_fft_um, film_nominal_nm, robust features
融合后输出：delta_L_nm
```

这可以作为后续主力模型，但应建立在 PCA 结果已经证明光谱有效的基础上。

当前不建议直接跳到混合 CNN，因为否则很难判断性能提升来自哪里：

- 是光谱本身提供了新信息？
- 是模型容量更大？
- 是过拟合了某些强度特征？
- 是否对未见 process 真正泛化？

---

## 4. Gemini 建议中缺失的关键约束

## 4.1 可部署输入约束

当前主模型不能使用以下字段作为输入：

```text
film_true_nm
film_delta_nm
process_film_true_nm
process_film_delta_nm
cavity_true_um
L_true_um
```

其中：

- `film_true_nm` 和 `film_delta_nm` 只在仿真中已知；
- 真实实验中通常只能知道 `film_nominal_nm`；
- `cavity_true_um` 和 `L_true_um` 只能作为标签或评价依据。

因此，即使 oracle 模型效果很好，也不能作为可部署模型。

可部署主模型只能使用：

```text
L_fft_um
film_nominal_nm
spectra_norm_ds 或其 PCA 特征
robust spectral features
```

可选但需谨慎使用：

```text
H_peak
强度相关 spectral features
```

---

## 4.2 强度漂移约束

真实实验中可能存在：

```text
光源功率漂移
探测器增益变化
光路耦合效率变化
背景光或暗电流漂移
```

因此，以下绝对强度相关特征不适合作为主模型输入：

```text
H_peak
spec_mean
spec_std
spec_min
spec_max
spec_ptp
spec_q05
spec_q25
spec_q50
spec_q75
spec_q95
fft_peak_height_1
fft_peak_prominence_1
fft_noise_floor
fft_band_energy_low
fft_band_energy_mid
fft_band_energy_high
```

它们可以作为消融实验，但不适合作为主模型依赖。

更适合作为主模型输入的是：

```text
spectra_norm_ds 的 PCA 特征
L_fft_um
film_nominal_nm
peak_count
spec_skew
spec_kurtosis
fft_peak_width_1
fft_num_peaks
fft_spectral_centroid_um
fft_snr_1
fringe_visibility_global
fringe_contrast_std
```

其中 `fft_snr_1`、`fringe_visibility_global`、`fringe_contrast_std` 仍然可能受加性背景影响，但比绝对强度特征稳健。

---

## 4.3 Split 约束

当前每个 process 下有多个 cavity samples，且同一个 process 内共享同一组真实膜厚扰动。

因此不能按 sample 随机划分 train/val/test，否则训练集和测试集可能共享同一个真实膜厚扰动，测试结果会过于乐观。

正确策略是：

```text
按 process_id 划分 train / val / test
```

当前数据检查结果显示：

```text
train processes = 1400
val processes   = 300
test processes  = 300
train ∩ val     = 0
train ∩ test    = 0
val ∩ test      = 0
```

这是正确的。

---

## 4.4 PCA 数据泄漏约束

PCA 不能对全体数据 fit。

错误做法：

```text
PCA.fit(all spectra_norm_ds)
```

正确做法：

```text
PCA.fit(train spectra_norm_ds)
PCA.transform(train spectra_norm_ds)
PCA.transform(val spectra_norm_ds)
PCA.transform(test spectra_norm_ds)
```

否则，PCA 主成分会提前看到 val/test 数据分布，造成数据泄漏。

---

## 5. 对当前训练路线的推荐修正版

## 5.1 第一阶段：scalar baseline

必须先建立基础对照：

```text
raw_fft_baseline:
    delta_pred_nm = 0

mean_residual_baseline:
    delta_pred_nm = mean(delta_L_nm on train)

scalar_no_intensity:
    L_fft_um + film_nominal_nm

scalar_with_hpeak:
    L_fft_um + H_peak + film_nominal_nm

selected_quadratic:
    L_fft_um + film_nominal_nm + L_fft_um * film_nominal_nm
```

其中 `scalar_with_hpeak` 和含 `H_peak` 的二阶项只作为消融实验，不应默认作为最终可部署主模型。

---

## 5.2 第二阶段：PCA 光谱主线

从大数据集：

```text
nn_cavity_spectral_features_*.npz
```

中读取：

```text
spectra_norm_ds
```

然后预提取：

```text
pca_scores
```

生成轻量 PCA 数据集：

```text
nn_cavity_pca_features_100_*.npz
```

训练时只读取 PCA 小数据集。

推荐比较：

```text
pca20_no_intensity:
    L_fft_um + film_nominal_nm + PC1~PC20

pca50_no_intensity:
    L_fft_um + film_nominal_nm + PC1~PC50

pca100_no_intensity:
    L_fft_um + film_nominal_nm + PC1~PC100
```

这一步的核心问题是：

```text
归一化光谱是否提供了 scalar features 之外的膜厚扰动信息？
```

---

## 5.3 第三阶段：PCA + robust hand-crafted features

如果 PCA 明显优于 scalar baseline，可以进一步加入稳健手工特征：

```text
spec_skew
spec_kurtosis
fft_peak_width_1
fft_num_peaks
fft_spectral_centroid_um
fft_snr_1
fringe_visibility_global
fringe_contrast_std
```

模型：

```text
pca50_robust_features:
    L_fft_um
    film_nominal_nm
    PC1~PC50
    robust spectral features
```

需要明确排除强度敏感特征。

---

## 5.4 第四阶段：1D-CNN / ResNet-1D 混合网络

只有当 PCA 证明光谱有效后，再考虑：

```text
CNN(spectra_norm_ds) + scalar fusion -> delta_L_nm
```

推荐结构：

```text
spectra_norm_ds -> 1D-CNN / ResNet-1D -> z_spec
[L_fft_um, film_nominal_nm, robust_features, z_spec] -> MLP head -> delta_L_nm
```

这时 Gemini 的 1D-CNN / ResNet-1D 建议才进入主线。

---

## 6. 推荐模型对照表

| 阶段 | 模型 | 输入 | 作用 |
|---|---|---|---|
| Baseline | raw_fft_baseline | `L_fft_um` | 无补偿物理基线 |
| Baseline | mean_residual_baseline | train 平均残差 | 最弱统计补偿 |
| Scalar | scalar_no_intensity | `L_fft_um + film_nominal_nm` | 最保守可部署模型 |
| Scalar | scalar_with_hpeak | `L_fft_um + H_peak + film_nominal_nm` | 检查强度特征是否有用 |
| Scalar | selected_quadratic | scalar + `L_fft_um * film_nominal_nm` | 检查二阶交互项 |
| PCA | pca20_no_intensity | `L_fft_um + film_nominal_nm + PC1~20` | 低维光谱信息 |
| PCA | pca50_no_intensity | `L_fft_um + film_nominal_nm + PC1~50` | 推荐主线模型之一 |
| PCA | pca100_no_intensity | `L_fft_um + film_nominal_nm + PC1~100` | 检查更多 PCA 分量收益 |
| PCA + robust | pca50_robust_features | PCA50 + robust features | 非 CNN 强模型 |
| CNN | cnn_scalar_fusion | `spectra_norm_ds + scalar` | 后续深度模型 |
| Oracle | scalar_oracle_true_thickness | `L_fft_um + film_true_nm` | 理论上限，不可部署 |

---

## 7. 对损失函数的建议

当前 sklearn `MLPRegressor` 使用默认 squared error，可以先作为 baseline。

后续如果切到 PyTorch，建议比较：

```text
MSELoss
L1Loss
HuberLoss / SmoothL1Loss
```

对于当前 nm 级残差补偿任务，`HuberLoss` 可能更稳，因为它兼顾：

- MSE 的平滑收敛；
- MAE 对异常值的鲁棒性。

评价指标不应只看 RMSE，还应同时看：

```text
MAE
RMSE
MaxAbs
P95Abs
P99Abs
Bias
R2_delta
```

---

## 8. 训练脚本设计建议

## 8.1 PCA 应提前提取

不建议每次训练都从 5–7 GB 的大 NPZ 中读取 `spectra_norm_ds` 并重新 PCA。

更推荐两步流程：

```text
Step 1:
extract_pca_features.py
大 NPZ -> PCA 小 NPZ

Step 2:
train_residual_mlp.py
PCA 小 NPZ -> MLP 训练
```

PCA 小 NPZ 应是 self-contained training dataset，应包含：

```text
pca_scores
sample_id
process_id
nominal_stack_id
split_id
valid_mask
L_fft_um
H_peak
peak_count
delta_L_nm
cavity_true_um
film_nominal_nm
film_delta_nm
film_true_nm
layer_names
spectral_features_full
spectral_feature_names
train_process_ids
val_process_ids
test_process_ids
```

不要复制：

```text
spectra_norm_ds
spectra
spectra_norm
```

这样后续训练只需要读取一个 PCA 小数据集，不需要同时调用原始大 NPZ。

---

## 8.2 训练脚本应支持的 feature groups

建议 `train_residual_mlp.py` 支持：

```text
l_fft_only
fft_scalar
nominal_thickness
selected_quadratic
pca_scores
spectral_features_full
```

其中：

```text
l_fft_only = L_fft_um
fft_scalar = L_fft_um + H_peak
```

并支持：

```text
--pca-components 20 / 50 / 100
--spectral-feature-preset robust
```

---

## 9. 最终批判结论

Gemini 的建议可以总结为：

```text
表格特征 -> MLP
一维光谱 -> 1D-CNN
时序 -> LSTM / GRU / Transformer
混合输入 -> 多分支网络
```

这个框架是正确的，但对于当前任务并不够具体。

针对当前 StackRT 光谱残差补偿任务，更合理的路线是：

```text
1. scalar baseline
2. scalar no-intensity
3. PCA20/50/100 no-intensity
4. PCA + robust spectral features
5. 1D-CNN / ResNet-1D hybrid
```

不要直接跳到 CNN，也不要使用 LSTM / GRU / Transformer。更不能忽略可部署输入限制、强度漂移风险、process-level split 和 PCA train-only fit 的要求。

当前最优先的工作不是换更复杂的神经网络，而是验证：

```text
spectra_norm_ds 的 PCA 特征是否能显著降低未见 process 的 test error。
```

如果 PCA 已经明显优于 scalar baseline，再进入 CNN / ResNet-1D 阶段才更有依据。
