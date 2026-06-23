# 光谱 1D-CNN / ResNet-1D 训练详细计划（prompt.md）

## 0. 项目目标

在已经证明 `PCA(spectra_norm_ds)` 能显著提升 test 指标的前提下，进一步训练基于原始光谱序列的深度模型：

- 光谱序列分支：`1D-CNN` 或 `ResNet-1D`
- 标量参数分支：`L_fft_um` 与 `film_nominal_nm`
- 融合后输出：`delta_L_nm`

最终预测仍采用残差补偿形式：

\[
L_{pred,um} = L_{FFT,um} + \frac{\Delta L_{pred,nm}}{1000}
\]

主任务是学习 **膜厚扰动导致的 FFT 粗解系统偏差**，而不是直接从头预测 `cavity_true_um`。

---

## 1. 数据与输入定义

### 1.1 单个样本的光谱长度

当前原始光谱范围为：

- wavelength start = `200 nm`
- wavelength stop  = `600 nm`
- spectral resolution = `0.02 nm`

你当前要求中，单个 sample 按 **20000 点** 处理。

> 注：从数学上讲，如果两端点都包含，则 `(600-200)/0.02 + 1 = 20001`。因此代码实现中必须显式统一：
>
> - 若采用 `20000` 点：需说明是左闭右开、去掉最后一点，或经过裁剪/trim；
> - 若采用 `20001` 点：则在模型和文档中统一写明。
>
> 本计划文件中，为与你当前要求一致，统一按 **20000 点输入** 描述。请在实现时确保数据维度和文档一致。

### 1.2 模型输入

#### 光谱输入（序列分支）

- 字段：`spectra_norm_ds` 或未降采样的 `spectra_norm_full`
- 当前目标：使用单个样本长度 `20000`
- 形状：
  - 原始数组：`(N, 20000)`
  - 送入 PyTorch：`(B, 1, 20000)`

说明：光谱已经做 per-spectrum normalization，因此 CNN 分支直接吃归一化后的谱形序列。

#### 标量输入（scalar branch）

只使用以下 5 个可部署输入：

- `L_fft_um`
- `film_nominal_nm[:, 0]`
- `film_nominal_nm[:, 1]`
- `film_nominal_nm[:, 2]`
- `film_nominal_nm[:, 3]`

也就是：

```text
[L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm]
```

> 当前阶段不默认加入 `H_peak`，因为它更容易受光源功率、探测器增益和耦合效率漂移影响。

### 1.3 标签

训练标签：

- `delta_L_nm`

不是：

- `cavity_true_um`

### 1.4 不能进入主模型输入的字段

以下字段只用于标签、评价、分析或 oracle 对照，**不能作为可部署模型输入**：

- `cavity_true_um`
- `L_true_um`
- `film_true_nm`
- `film_delta_nm`
- `process_film_true_nm`
- `process_film_delta_nm`

---

## 2. 为什么这里要使用 PyTorch，而不是 sklearn

### 2.1 sklearn 的优势与局限

`sklearn` 非常适合当前已经在做的：

- `MLPRegressor`
- `PCA + MLP`
- 手工特征 + MLP
- 快速 baseline 对比

但是对于本阶段的光谱深度模型，它的限制很明显：

#### 限制 1：sklearn 不适合卷积网络 / ResNet

`sklearn.neural_network.MLPRegressor` 只适合全连接网络，不支持：

- `Conv1d`
- `BatchNorm1d`
- 残差连接（Residual Block）
- 自定义双分支网络
- 灵活的 forward 融合逻辑

因此：

- `1D-CNN + scalar fusion`
- `ResNet-1D + scalar fusion`

这类结构在 sklearn 中几乎不适合实现。

#### 限制 2：sklearn 不适合 GPU 训练

当前数据量较大：

- 样本数：`800000`
- 每个光谱长度：`20000`

对于一维卷积模型，CPU 训练会非常慢，而 `sklearn` 基本不提供 GPU 深度学习训练路径。

PyTorch 可以：

- 直接使用 GPU
- 混合精度训练（AMP）
- 更高效地处理大 batch 和长序列

#### 限制 3：sklearn 不适合自定义 Dataset / DataLoader

CNN 训练通常需要：

- 按 batch 读取光谱
- 多进程加载数据
- memmap / `.npy` 形式读取大数组
- 动态拼接 scalar branch 与 spectrum branch

PyTorch 的：

- `Dataset`
- `DataLoader`
- `num_workers`
- `pin_memory`

都更适合这个任务。

#### 限制 4：sklearn 不适合训练细节控制

当前深度模型需要灵活控制：

- 自定义 loss（如 `SmoothL1Loss` / `HuberLoss`）
- optimizer（如 `AdamW`）
- learning rate scheduler
- early stopping
- best checkpoint 保存
- train / val 曲线记录

这些在 PyTorch 中都很自然；在 sklearn 中则不方便或不支持。

### 2.2 为什么 PyTorch 更合适

PyTorch 适合本任务的原因：

1. **天然支持 1D-CNN / ResNet-1D**
2. **支持双分支网络与任意融合结构**
3. **支持 GPU 和 AMP**，更适合长光谱序列训练
4. **支持 Dataset / DataLoader**，更适合 80 万样本的大数据集
5. **支持自定义 loss 与训练循环**
6. **后续扩展方便**：CNN、ResNet、注意力模块、噪声增强都能自然追加

### 2.3 实际定位建议

建议这样分工：

- `sklearn`：继续用于 `scalar baseline`、`selected quadratic`、`PCA + MLP`
- `PyTorch`：用于 `1D-CNN / ResNet-1D + scalar fusion`

也就是说：

- baseline 与中间验证用 `sklearn`
- 真正的光谱深度模型用 `PyTorch`

---

## 3. 数据预处理与数据文件组织

### 3.1 推荐新增脚本

建议新增：

```text
prepare_cnn_dataset.py
```

目标：把当前大 NPZ 转成更适合 PyTorch 训练的格式。

### 3.2 推荐输出结构

```text
cnn_dataset_YYYYMMDD_HHMMSS/
  spectra_norm.npy
  scalar_fields.npz
  dataset_summary.json
```

其中：

#### `spectra_norm.npy`

保存：

- shape=`(800000, 20000)`
- dtype=`float32`

尽量使用可 `mmap_mode='r'` 读取的 `.npy`。

#### `scalar_fields.npz`

保存：

- `sample_id`
- `process_id`
- `nominal_stack_id`
- `split_id`
- `valid_mask`
- `L_fft_um`
- `H_peak`
- `peak_count`
- `delta_L_nm`
- `cavity_true_um`
- `film_nominal_nm`
- `film_true_nm`
- `film_delta_nm`
- `layer_names`
- `spectral_features_full`
- `spectral_feature_names`
- `train_process_ids`
- `val_process_ids`
- `test_process_ids`

> 训练时只用一个“CNN 数据文件夹”即可，不应要求用户同时再手动传第二个数据集。

### 3.3 split 使用规则

必须使用已有 `split_id`：

- `split_id == 0` → train
- `split_id == 1` → val
- `split_id == 2` → test

并与 `valid_mask == True` 取交集。

不能重新按 sample 随机切分。

原因：同一个 process 下的 400 个 cavity samples 共享同一组膜厚扰动，因此必须整体落在同一个 split。

---

## 4. 模型一：CNN-small + Scalar Fusion

### 4.1 设计思想

这是第一版光谱深度模型，目标是：

- 结构尽量简单
- 先验证 1D-CNN 是否优于 PCA50 / PCA100
- 训练稳定、参数量适中

### 4.2 光谱分支

输入：

- `x_spec.shape = (B, 1, 20000)`

结构建议：

```text
Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
BatchNorm1d(32)
GELU

Conv1d(32, 64, kernel_size=9, stride=2, padding=4)
BatchNorm1d(64)
GELU

Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
BatchNorm1d(128)
GELU

Conv1d(128, 128, kernel_size=5, stride=2, padding=2)
BatchNorm1d(128)
GELU

AdaptiveAvgPool1d(1)
Flatten
```

输出：

- `z_spec.shape = (B, 128)`

### 4.3 标量分支

输入：

- `x_scalar.shape = (B, 5)`

结构建议：

```text
Linear(5, 32)
GELU
Linear(32, 32)
GELU
```

输出：

- `z_scalar.shape = (B, 32)`

### 4.4 融合头

拼接：

```text
concat([z_spec, z_scalar])
shape = (B, 160)
```

输出头：

```text
Linear(160, 128)
GELU
Dropout(0.1)

Linear(128, 64)
GELU

Linear(64, 1)
```

输出：

- `delta_L_pred_nm`

---

## 5. 模型二：ResNet-1D + Scalar Fusion

### 5.1 设计思想

如果 `CNN-small` 已经优于 PCA，则进一步使用残差结构：

- 增强光谱局部模式提取能力
- 提高训练稳定性
- 适合更长的序列输入

### 5.2 ResBlock 定义

每个残差块：

```text
Conv1d(C_in, C_out, kernel_size=7, stride=s, padding=3)
BatchNorm1d(C_out)
GELU

Conv1d(C_out, C_out, kernel_size=7, stride=1, padding=3)
BatchNorm1d(C_out)

Skip:
  if stride != 1 or C_in != C_out:
      Conv1d(1x1, stride=s)

Output = GELU(main + skip)
```

### 5.3 光谱分支结构

输入：

- `x_spec.shape = (B, 1, 20000)`

结构建议：

```text
Stem:
Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
BatchNorm1d(32)
GELU

Stage 1:
2 x ResBlock(32, 32, stride=1)

Stage 2:
ResBlock(32, 64, stride=2)
ResBlock(64, 64, stride=1)

Stage 3:
ResBlock(64, 128, stride=2)
ResBlock(128, 128, stride=1)

Stage 4:
ResBlock(128, 256, stride=2)
ResBlock(256, 256, stride=1)

AdaptiveAvgPool1d(1)
Flatten
```

输出：

- `z_spec.shape = (B, 256)`

### 5.4 标量分支

输入：

- `x_scalar.shape = (B, 5)`

结构建议：

```text
Linear(5, 64)
GELU
Linear(64, 64)
GELU
```

输出：

- `z_scalar.shape = (B, 64)`

### 5.5 融合头

拼接：

```text
concat([z_spec, z_scalar])
shape = (B, 320)
```

输出头：

```text
Linear(320, 256)
GELU
Dropout(0.1)

Linear(256, 128)
GELU

Linear(128, 1)
```

输出：

- `delta_L_pred_nm`

---

## 6. 训练设置

### 6.1 训练框架

必须使用 `PyTorch`。

建议新增脚本：

```text
train_spectral_cnn.py
```

放置位置：

```text
01_Lumerical_Workflow/ML try/Residual MLP/train_spectral_cnn.py
```

> 虽然目录名叫 `Residual MLP`，但暂时可以继续放这里；后续可重构为 `Residual Models/`。

### 6.2 loss 函数

第一版建议用：

```text
SmoothL1Loss(beta=1.0)
```

理由：

- 比 MSE 更不容易被少量极端样本主导
- 比纯 MAE 更平滑，更利于优化
- 对当前关注的 `P95 / MaxAbs` 更友好

可做消融：

- `MSELoss`
- `SmoothL1Loss(beta=1.0)`
- `SmoothL1Loss(beta=2.0)`

### 6.3 optimizer

建议：

```text
AdamW
learning_rate = 1e-3
weight_decay = 1e-4
```

### 6.4 scheduler

建议：

```text
ReduceLROnPlateau
```

监控：

- `val_RMSE_nm`

### 6.5 batch size

推荐：

- GPU 正式训练：`512`
- 显存较小时：`256`
- 快速调试：`128` 或 `256`

### 6.6 epoch

推荐：

- quick test：`10`
- 正式训练：`50 ~ 100`
- early stopping patience：`10`

### 6.7 mixed precision

如果有 CUDA，建议：

```text
use_amp = true
```

提升训练速度并减少显存占用。

---

## 7. 标准化策略

### 7.1 光谱输入

当前光谱已经是 per-spectrum normalized，因此：

- `spectra_norm` 直接输入 CNN
- 不再对全体样本额外做一次全局 StandardScaler

### 7.2 标量输入

标量输入必须：

- 只在 train split 上 `fit StandardScaler`
- val/test 只能 `transform`

标量为：

- `L_fft_um`
- `film_nominal_nm`

### 7.3 标签

建议对 `delta_L_nm` 也做标准化：

\[
y_{scaled} = \frac{y - \mu_{train}}{\sigma_{train}}
\]

模型输出 scaled residual，推理后再还原到 nm。

---

## 8. 推荐比较的模型版本

### 8.1 baseline

#### A. scalar_no_intensity

输入：

```text
L_fft_um + film_nominal_nm
```

#### B. pca50_no_intensity

输入：

```text
L_fft_um + film_nominal_nm + PC1~PC50
```

#### C. pca100_no_intensity

输入：

```text
L_fft_um + film_nominal_nm + PC1~PC100
```

### 8.2 深度模型

#### D. cnn_small_no_intensity

输入：

```text
CNN-small(spectra_norm) + L_fft_um + film_nominal_nm
```

#### E. resnet1d_no_intensity

输入：

```text
ResNet1D(spectra_norm) + L_fft_um + film_nominal_nm
```

### 8.3 可选扩展

#### F. cnn_small_robust_features

输入：

```text
CNN-small(spectra_norm) + L_fft_um + film_nominal_nm + robust spectral features
```

其中 `robust spectral features` 可选：

- `spec_skew`
- `spec_kurtosis`
- `fft_peak_width_1`
- `fft_num_peaks`
- `fft_spectral_centroid_um`
- `fft_snr_1`
- `fringe_visibility_global`
- `fringe_contrast_std`

> 第一轮不建议默认加入 robust features，先验证纯光谱序列 + scalar 融合是否已经优于 PCA。

---

## 9. 评价指标

每个模型都必须输出：

- `delta_MAE_nm`
- `delta_RMSE_nm`
- `delta_MaxAbs_nm`
- `delta_P95Abs_nm`
- `delta_P99Abs_nm`
- `delta_Bias_nm`
- `R2_delta`
- `cavity_MAE_nm`
- `cavity_RMSE_nm`
- `cavity_MaxAbs_nm`

重点看 test：

- `cavity_RMSE_nm`
- `cavity_MAE_nm`
- `delta_P95Abs_nm`
- `cavity_MaxAbs_nm`

判断准则：

1. `CNN-small` 若优于 `PCA50/PCA100`，说明 CNN 学到 PCA 未充分提取的局部谱形信息；
2. `ResNet-1D` 若进一步优于 `CNN-small`，则可作为更强候选；
3. 若 train 大幅下降但 val/test 不降，则说明过拟合，不应盲目加深网络。

---

## 10. 输出文件要求

每次训练创建：

```text
spectral_cnn_compare_YYYYMMDD_HHMMSS/
```

输出：

- `config.json`
- `metrics.json`
- `summary_report.md`
- `best_model.pt`
- `last_model.pt`
- `scalers.joblib`
- `training_log.csv`
- `test_predictions.csv`
- `01_loss_curve.png`
- `02_val_rmse_curve.png`
- `03_test_pred_vs_true_delta.png`
- `04_test_error_hist.png`
- `05_test_error_vs_L_fft.png`
- `06_method_comparison_bar.png`

`best_model.pt` 至少应保存：

- `model_state_dict`
- `scalar_scaler`
- `target_mean`
- `target_std`
- `config`
- `feature_policy`

---

## 11. 命令行接口建议

`train_spectral_cnn.py` 建议支持以下参数：

```text
--dataset path/to/cnn_dataset_folder_or_npz
--model cnn_small / resnet1d
--epochs 80
--batch-size 512
--learning-rate 1e-3
--weight-decay 1e-4
--loss smooth_l1
--huber-beta 1.0
--num-workers 4
--use-amp true
--random-seed 20260613

--max-train-rows
--max-val-rows
--max-test-rows

--use-robust-features false
--use-hpeak false
```

默认：

- `model = cnn_small`
- `use_hpeak = false`
- `use_robust_features = false`

---

## 12. Windows CMD 命令示例

### 12.1 快速测试：CNN-small

```bat
python "01_Lumerical_Workflow\ML try\Residual MLP\train_spectral_cnn.py" ^
  --dataset "01_Lumerical_Workflow\ML try\nn_cavity_spectral_features_20260620_233057\nn_cavity_spectral_features_20260620_233057.npz" ^
  --model cnn_small ^
  --max-train-rows 100000 ^
  --max-val-rows 20000 ^
  --max-test-rows 20000 ^
  --epochs 10 ^
  --batch-size 512 ^
  --use-amp true
```

### 12.2 正式训练：CNN-small

```bat
python "01_Lumerical_Workflow\ML try\Residual MLP\train_spectral_cnn.py" ^
  --dataset "01_Lumerical_Workflow\ML try\nn_cavity_spectral_features_20260620_233057\nn_cavity_spectral_features_20260620_233057.npz" ^
  --model cnn_small ^
  --epochs 80 ^
  --batch-size 512 ^
  --use-amp true
```

### 12.3 正式训练：ResNet-1D

```bat
python "01_Lumerical_Workflow\ML try\Residual MLP\train_spectral_cnn.py" ^
  --dataset "01_Lumerical_Workflow\ML try\nn_cavity_spectral_features_20260620_233057\nn_cavity_spectral_features_20260620_233057.npz" ^
  --model resnet1d ^
  --epochs 100 ^
  --batch-size 512 ^
  --use-amp true
```

---

## 13. 进一步实施顺序（推荐）

推荐按下面顺序推进：

### Step 1
先实现并跑：

- `cnn_small_no_intensity`

输入：

```text
spectra_norm + L_fft_um + film_nominal_nm
```

### Step 2
与以下模型对比：

- `scalar_no_intensity`
- `pca50_no_intensity`
- `pca100_no_intensity`

### Step 3
如果 `CNN-small` 明显优于 PCA：

- 继续训练 `resnet1d_no_intensity`

### Step 4
如果 `ResNet-1D` 进一步优于 `CNN-small`：

- 再尝试加入少量 `robust spectral features`

### Step 5
最后才考虑：

- 噪声增强
- 波长轴微小平移增强
- 更复杂结构（attention / transformer 等）

---

## 14. 最终一句话总结

如果 PCA 已经证明完整光谱确实有用，那么下一步最合理的深度模型方案是：

- 使用 `PyTorch`
- 光谱序列走 `1D-CNN` 或 `ResNet-1D`
- 标量分支只使用 `L_fft_um + film_nominal_nm`
- 融合后输出 `delta_L_nm`
- 继续坚持 **process-level split**、**train-only scaler**、**不使用 true film thickness** 的原则

并且执行顺序应为：

```text
CNN-small → 对比 PCA → ResNet-1D → 再考虑 robust features 与增强
```
