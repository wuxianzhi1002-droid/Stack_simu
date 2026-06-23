# Residual MLP 训练代码

本文件夹用于训练和比较：

```text
Residual MLP + nominal film thickness + optional quadratic interaction features
```

核心原则：

```text
主模型只使用 nominal thickness。
true film thickness 只能作为 oracle 对照，默认不开启。
film_delta_nm 不能作为主模型输入。
cavity_true_um 只能用于生成标签或评价，不能作为输入。
```

默认数据集：

```text
优先自动选择最新的：
../nn_cavity_spectral_features_*/nn_cavity_spectral_features_*.npz

如果版本 2 数据集不存在，则回退到：
../nn_cavity_scalar_dataset_all_2000.npz
```

## 选择进入模型的特征

新增命令行入口：

```bash
--feature-groups
```

可组合选择：

```text
fft_scalar               -> L_fft_um, H_peak
peak_count               -> FFT peak 数量
nominal_thickness        -> PSS/HSQ/SOC/TiO2_nominal_nm
selected_quadratic       -> L_fft_um * 各层 nominal thickness
spectral_features_full   -> 版本 2 数据集中的完整光谱标量特征
```

查看当前数据集支持的字段和全部光谱特征名：

```bash
python train_residual_mlp.py --list-available-features
```

使用全部 full-spectrum 特征：

```bash
python train_residual_mlp.py ^
  --feature-groups fft_scalar peak_count nominal_thickness selected_quadratic spectral_features_full ^
  --spectral-feature-names all
```

只选择部分 full-spectrum 特征：

```bash
python train_residual_mlp.py ^
  --feature-groups fft_scalar nominal_thickness spectral_features_full ^
  --spectral-feature-names spec_mean spec_std spec_ptp fft_snr_1 fringe_visibility_global
```

指定 `--feature-groups` 后，脚本会在原有两个基线之外额外训练：

```text
custom_selected_features
```

其模型文件为：

```text
residual_mlp_custom_selected_features.joblib
feature_names_custom_selected_features.json
test_predictions_custom_selected_features.csv
```

模型包会保存实际特征名、train-only median imputer 和 StandardScaler。光谱特征中的 NaN 不会使用全数据统计量填补。

## 训练目标

模型不直接预测 `cavity_true_um`，而是预测 FFT 粗解残差：

```text
delta_L_nm = (cavity_true_um - L_fft_um) * 1000
```

最终腔长还原：

```text
cavity_pred_um = L_fft_um + delta_L_pred_nm / 1000
```

## 两个主实验

### 版本 1：scalar_baseline

输入特征：

```text
L_fft_um
H_peak
PSS_nominal_nm
HSQ_nominal_nm
SOC_nominal_nm
TiO2_nominal_nm
```

模型：

```text
StandardScaler(train only)
MLPRegressor(128, 128, 64)
```

### 版本 2：scalar_selected_quadratic

输入包括版本 1 的全部特征，并额外加入手选二阶交互项：

```text
L_fft_um * PSS_nominal_nm
L_fft_um * HSQ_nominal_nm
L_fft_um * SOC_nominal_nm
L_fft_um * TiO2_nominal_nm
```

默认不加入膜厚之间的两两乘积，例如 `HSQ_nominal_nm * SOC_nominal_nm`。

如需额外加入：

```text
H_peak * PSS/HSQ/SOC/TiO2_nominal_nm
```

可以使用：

```bash
--include-hpeak-interactions true
```

## 可选消融实验

### scalar_all_quadratic

默认不开启。开启方式：

```bash
--enable-all-quadratic
```

该模式会使用 `PolynomialFeatures(degree=2, include_bias=False)` 自动生成所有二阶项，仅用于 ablation。

### scalar_oracle_true_thickness

默认不开启。开启方式：

```bash
--enable-oracle true
```

该模式使用真实膜厚 `film_true_nm`，不可部署，只用于判断如果有独立膜厚计量时理论上能改善多少。

## Split 策略

默认：

```bash
--split-strategy process_within_nominal
```

含义：每个 nominal thickness group 下的 20 个 process 按 process_id 划分 train/val/test。这样同一个 process 不会同时出现在 train/val/test。

更严格的可选策略：

```bash
--split-strategy nominal_holdout
```

含义：整个 nominal thickness group 进入同一个 split，用来测试完全没见过的名义膜厚组合。

## 快速测试

建议先在 PyCharm 中跑小样本：

```bash
python train_residual_mlp.py --max-train-rows 200000 --max-val-rows 50000 --max-test-rows 50000 --epochs 60
```

## 完整训练

```bash
python train_residual_mlp.py --epochs 120
```

如需开启 all quadratic 消融：

```bash
python train_residual_mlp.py --epochs 120 --enable-all-quadratic
```

## 输出目录

每次运行会创建：

```text
residual_mlp_compare_YYYYMMDD_HHMMSS/
```

主要输出：

```text
metrics.json
summary_report.md
feature_names_scalar_baseline.json
feature_names_selected_quadratic.json
feature_names_all_quadratic.json
feature_correlation_matrix.csv
high_correlation_feature_pairs.csv
test_predictions_scalar_baseline.csv
test_predictions_selected_quadratic.csv
residual_mlp_scalar_baseline.joblib
residual_mlp_selected_quadratic.joblib
residual_mlp_custom_selected_features.joblib  # 使用 --feature-groups 时
01_test_pred_vs_true_delta.png
02_test_error_hist.png
03_test_error_vs_L_fft.png
04_test_error_vs_nominal_thickness.png
05_method_comparison_bar.png
```

如果开启 `--enable-all-quadratic`，还会输出：

```text
test_predictions_all_quadratic.csv
residual_mlp_all_quadratic.joblib
```

## 判断重点

不要只看 train 指标。重点比较：

```text
test cavity_MAE_nm
test cavity_RMSE_nm
test cavity_MaxAbs_nm
```

如果 `scalar_selected_quadratic` 的 test RMSE 明显低于 `scalar_baseline`，并且 val/test 差距没有显著扩大，才认为手选二阶特征有效。
