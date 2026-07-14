# 大规模 StackRT 标量 + 光谱特征数据集生成要求

更新时间：2026-06-20

本文件用于记录当前项目中 `work/03_ml_inverse_modeling/ML try/` 目录下神经网络 / 残差模型数据集的最新仿真范围、保存格式、光谱特征保存策略和续跑要求。

当前版本的核心变化是：

1. 仿真仍然使用 `lumapi + stackrt` 逐条生成完整反射光谱。
2. 每条完整光谱首先在内存中用于 FFT 粗解。
3. 每条完整光谱在被丢弃前，额外提取一组 full-spectrum scalar features。
4. 不默认保存完整原始光谱矩阵 `spectra`。
5. 默认保存归一化后的降采样光谱矩阵 `spectra_norm_ds`，用于后续 PCA / CNN / 光谱模型。
6. 标量结果、膜厚标签、腔长标签、降采样光谱、完整光谱提取特征共同保存。

这样可以避免完整光谱数据集达到数十 GB 到上百 GB，同时保留后续 PCA、CNN 和手工光谱特征建模所需的信息。

---

## 一、路径要求

所有本次相关脚本和输出文件必须放在：

```text
work/03_ml_inverse_modeling/ML try/
```

不要输出到：

```text
work/01_simulation_models/01_Lumerical_Workflow/stackrt_result/
```

必须使用 `pathlib.Path` 管理路径，因为 `ML try` 文件夹名包含空格。

当前主要脚本：

```text
work/03_ml_inverse_modeling/ML try/build_nn_cavity_dataset.py
work/03_ml_inverse_modeling/ML try/resume_nn_cavity_scalar_results.py
```

建议新的输出目录命名为：

```text
work/03_ml_inverse_modeling/ML try/nn_cavity_spectral_features_YYYYMMDD_HHMMSS/
```

---

## 二、当前数据集类型

当前数据集类型改为：

```text
scalar_plus_optional_spectra_and_full_features
```

也就是说，标量结果一定保存，光谱结果可选保存，完整光谱特征默认保存。

### 2.1 一定保存的标量结果

```text
cavity_true_um
L_true_um
L_fft_um
delta_L_um
delta_L_nm
H_peak
peak_count
film_nominal_nm
film_delta_nm
film_true_nm
process_id
nominal_stack_id
nominal_stack_name_by_id
split_id
split_names
valid_mask
wavelengths_um
cavity_axis_um
```

### 2.2 默认保存的降采样光谱

默认保存：

```text
spectra_norm_ds
wavelengths_spectra_saved_um
```

其中：

```text
spectra_norm_ds
```

表示每条完整光谱经过 per-spectrum normalization 后，再按指定 factor 降采样得到的归一化光谱。

推荐默认：

```text
save_spectra = True
spectra_save_mode = "norm_downsampled"
spectra_dtype = "float32"
spectra_downsample_factor = 10
spectra_downsample_method = "mean"
spectrum_normalization = "per_spectrum_zscore"
```

### 2.3 默认保存的 full-spectrum scalar features

每条完整光谱在内存中被丢弃前，需要提取并保存：

```text
spectral_features_full
spectral_feature_names
```

这些特征必须基于完整原始光谱或完整 k-space FFT 结果提取，而不是基于降采样后的 `spectra_norm_ds` 提取。

推荐字段：

```text
spectral_feature_source = "full_spectrum_before_downsampling"
```

---

## 三、为什么不默认保存完整原始光谱

当前波长设置为：

```text
wavelength_start_um = 0.2
wavelength_stop_um = 0.6
spectral_resolution_nm = 0.02
```

因此：

```text
num_wavelengths = 20001
```

当前仿真规模为：

```text
100 nominal models
* 20 process / nominal model
* 400 cavity points / process
= 800,000 samples
```

如果保存完整 `float64` 光谱矩阵：

```text
800000 × 20001 × 8 bytes ≈ 128 GB
```

如果保存完整 `float32` 光谱矩阵：

```text
800000 × 20001 × 4 bytes ≈ 64 GB
```

这还没有计算 `spectra_norm`、标量字段、膜厚字段和索引文件的额外开销。

因此默认不保存完整原始光谱，而是保存：

```text
spectra_norm_ds
spectral_features_full
```

这样既可以控制存储压力，又可以保留后续建模需要的信息。

---

## 四、仿真光谱范围

沿用 `main_cavity.py` 的波长设置：

```text
wavelength_start_um = 0.2
wavelength_stop_um = 0.6
spectral_resolution_nm = 0.02
```

因此：

```text
num_wavelengths = 20001
```

每次 StackRT 返回的完整光谱长度应为 `20001`。

完整波长轴必须保存：

```text
wavelengths_um
```

如果保存降采样光谱，还必须保存对应的降采样波长轴：

```text
wavelengths_spectra_saved_um
```

---

## 五、腔长扫描范围

当前大规模仿真使用 1 nm 腔长步长，扫描 400 个点：

```text
cavity_start_um = 1000.000
cavity_step_um = 0.001
num_cavity_points = 400
```

实际腔长范围为：

```text
cavity_true_um = 1000.000 到 1000.399 um
```

注意：当前不再是每个 process 1000 个 cavity points，而是每个 process 400 个 cavity points。

---

## 六、nominal model 范围

当前要求使用 `100` 个 nominal models。

每个膜层厚度必须是 `10 nm` 的倍数，并从以下范围内选取：

```text
PSS_nm:  10, 20, 30, 40
HSQ_nm:  20, 30, 40, 50, 60
SOC_nm:  40, 50, 60, 70, 80
TiO2_nm: 30, 40, 50, 60, 70, 80
```

完整网格数量为：

```text
4 * 5 * 5 * 6 = 600
```

但当前仿真只取其中 `100` 个 nominal models。

抽样要求：

1. 使用固定随机种子，保证可复现。
2. 厚度全部来自上述 10 nm 网格。
3. 覆盖每个膜层的上下边界。
4. 保存 `nominal_stack_name_by_id` 和 `nominal_stack_values_nm`。

当前默认随机种子：

```text
random_seed = 20260620
```

---

## 七、process 设置

每个 nominal model 生成：

```text
num_process_per_nominal = 20
```

其中：

```text
process_idx = 0
```

表示无扰动 nominal process：

```text
film_delta_nm = [0, 0, 0, 0]
```

其余 process：

```text
process_idx = 1..19
```

对每层膜厚加入随机扰动：

```text
delta_d_i ~ Uniform(-5 nm, +5 nm)
```

---

## 八、总仿真规模

当前大规模仿真总量为：

```text
100 nominal models
* 20 process / nominal model
* 400 cavity points / process
= 800,000 samples
```

总 process 数：

```text
num_processes = 100 * 20 = 2000
```

每个 process 包含：

```text
400 samples
```

因此：

```text
checkpoint_process_0000.npz -> process_id = 0
checkpoint_process_0099.npz -> process_id = 99，对应前 100 个 process 完成
checkpoint_process_1999.npz -> process_id = 1999，对应全部 2000 个 process 完成
```

---

## 九、层结构和材料模型

必须沿用 `main_cavity.py` 中的 StackRT 层结构和折射率模型。

层结构：

```python
[
    ("RefReflector", 0),
    ("Air", cavity_um),
    ("HSQ", hsq_um),
    ("PSS", pss_um),
    ("SOC", soc_um),
    ("TiO2", tio2_um),
    ("Cu", 0),
]
```

单位要求：

```text
Air 层厚度使用 um，即 cavity_um
其他膜层 nominal / true / delta 保存为 nm
传给 StackRT 的膜层厚度需要从 nm 转成 um，再在 thicknesses 中转成 m
```

折射率模型：

```text
RefReflector: 5.8284
Air:          1.0
HSQ:          1.41
PSS:          1.50 + 0.05j
SOC:          1.55 + 0.005 / (w_um**2)
TiO2:         2.4 + 0.02 / (w_um**2)
Cu:           1.1 + 2.5j
```

StackRT 调用方式：

```python
fdtd = lumapi.FDTD(hide=True)
res = fdtd.stackrt(n_matrix, thicknesses, freqs, angle_deg)
spectrum = np.real(np.asarray(res[result_key]).flatten())
```

其中：

```text
polarization = "p" -> result_key = "Rp"
polarization = "s" -> result_key = "Rs"
```

当前默认：

```text
angle_deg = 0.0
polarization = "p"
```

---

## 十、FFT 粗解要求

每条 StackRT 光谱生成后立即执行 FFT 粗解。

FFT 必须使用完整原始光谱，而不是降采样光谱。

FFT 逻辑沿用 `solve_npz_fft.py` 中 `FFTSolver.solve` 的 k-space FFT 思路：

1. wavelength 排序。
2. `k_raw = 2*pi / wavelengths_um`。
3. 插值到均匀 `k_linear`。
4. 去均值。
5. Hann window。
6. zero padding。
7. `np.fft.rfft`。
8. `find_peaks`。
9. dominant peak 取 peak height 最大者。
10. `L_fft_um = dominant_peak_distance_um`。
11. `H_peak = dominant_peak_height`。

当前 FFT 配置：

```text
fft_peak_height_ratio = 0.2
fft_ignore_dc_bins = 50
fft_peak_distance_bins = 100
zero_pad_factor = 8
```

如果没有检测到 peak：

```text
L_fft_um = nan
H_peak = nan
valid_mask = False
```

并记录到 failed FFT 日志。

---

## 十一、光谱保存策略

### 11.1 保存模式

`spectra_save_mode` 支持：

```text
none
norm_downsampled
norm_full
raw_downsampled
raw_and_norm_downsampled
```

含义如下。

#### 1. none

不保存任何光谱矩阵。

仅保存标量结果和 full-spectrum scalar features。

#### 2. norm_downsampled

保存归一化后的降采样光谱：

```text
spectra_norm_ds
```

同时保存对应波长轴：

```text
wavelengths_spectra_saved_um
```

这是推荐默认模式，用于 PCA / CNN / 光谱特征开发。

#### 3. norm_full

保存完整归一化光谱：

```text
spectra_norm
```

用于小规模对照实验或最终验证，不建议默认开启。

#### 4. raw_downsampled

保存原始反射率降采样光谱：

```text
spectra_ds
```

一般不推荐作为默认模式。

#### 5. raw_and_norm_downsampled

同时保存原始降采样光谱和归一化降采样光谱：

```text
spectra_ds
spectra_norm_ds
```

用于检查归一化方式是否影响模型表现。

### 11.2 默认设置

推荐默认：

```text
save_spectra = True
spectra_save_mode = "norm_downsampled"
spectra_dtype = "float32"
spectra_downsample_factor = 10
spectra_downsample_method = "mean"
spectrum_normalization = "per_spectrum_zscore"
```

### 11.3 重要要求

FFT 粗解与 full-spectrum scalar features 必须基于完整原始光谱计算。

降采样光谱只用于后续 PCA / CNN / 光谱模型输入，不用于生成当前数据集中的 `L_fft_um`、`H_peak`、`peak_count`。

---

## 十二、光谱归一化要求

每条光谱保存前进行 per-spectrum normalization。

推荐默认：

```text
spectrum_norm = (spectrum - mean(spectrum)) / std(spectrum)
```

其中：

```text
mean/std 只对当前单条 spectrum 计算
```

如果 `std` 非有限或小于阈值 `eps`，则：

```text
spectrum_norm = spectrum - mean(spectrum)
```

并记录异常样本。

保存：

```text
spectra_norm_method = "per_spectrum_zscore"
```

注意：

1. 这个 per-spectrum normalization 只使用单条光谱自身的信息，不涉及 train / val / test 泄漏。
2. 后续 PCA、StandardScaler 仍然只能在 train set 上 fit，val / test 只能 transform。

---

## 十三、光谱降采样要求

支持两种降采样方法。

### 13.1 slice

```python
spectrum_saved = spectrum[::factor]
wavelengths_saved_um = wavelengths_um[::factor]
```

### 13.2 mean

将光谱按 `factor` 分组求平均，波长轴也按相同分组求平均。

如果长度不能被 `factor` 整除，则丢弃末尾不足一个 bin 的点。

推荐默认：

```text
spectra_downsample_method = "mean"
```

原因：mean 降采样比 slice 更不容易受到局部采样点偶然波动影响。

---

## 十四、full-spectrum scalar features 要求

本节是当前版本新增的重点。

每条完整光谱在内存中被丢弃前，必须提取一组基于完整光谱的标量特征。

保存字段：

```text
spectral_features_full
spectral_feature_names
spectral_feature_source
```

其中：

```text
spectral_features_full.shape = [num_samples, num_spectral_features]
spectral_feature_names.shape = [num_spectral_features]
spectral_feature_source = "full_spectrum_before_downsampling"
```

### 14.1 光谱统计特征

基于完整原始 spectrum 提取：

```text
spec_mean
spec_std
spec_min
spec_max
spec_ptp
spec_skew
spec_kurtosis
spec_q05
spec_q25
spec_q50
spec_q75
spec_q95
```

说明：

```text
spec_ptp = spec_max - spec_min
```

### 14.2 FFT 扩展特征

基于完整光谱插值到均匀 k-space 后的 FFT 结果提取。

建议保存：

```text
fft_peak_pos_1_um
fft_peak_height_1
fft_peak_width_1
fft_peak_prominence_1

fft_peak_pos_2_um
fft_peak_height_2
fft_peak_width_2
fft_peak_prominence_2

fft_peak_height_ratio_21
fft_peak_distance_21_um
fft_num_peaks
fft_noise_floor
fft_snr_1
fft_spectral_centroid_um
fft_band_energy_low
fft_band_energy_mid
fft_band_energy_high
```

其中：

```text
fft_peak_pos_1_um
```

应与当前 dominant peak 对应，原则上与 `L_fft_um` 一致或高度一致。

如果 peak 数不足两个，则第二峰相关特征填 `nan`，并在后续 `valid_mask` 或训练阶段处理。

### 14.3 条纹 / 可见度特征

基于完整原始 spectrum 提取：

```text
fringe_visibility_global
fringe_contrast_std
```

推荐定义：

```text
fringe_visibility_global = (spec_max - spec_min) / (spec_max + spec_min + eps)
fringe_contrast_std = spec_std / (abs(spec_mean) + eps)
```

### 14.4 可选 envelope 特征

如果当前环境中可以稳定使用 `scipy.signal.hilbert`，可以额外保存：

```text
envelope_mean
envelope_std
envelope_ptp
```

如果暂时不想引入 Hilbert，可先不实现 envelope 特征，不影响主流程。

---

## 十五、PCA 特征是否在生成阶段保存

不建议在数据集生成阶段直接保存 PCA 特征。

原因：PCA 是数据驱动的降维方法，必须满足：

```text
PCA.fit(train spectra_norm_ds)
PCA.transform(train / val / test spectra_norm_ds)
```

如果在数据集生成阶段对全部样本统一 fit PCA，会造成数据泄漏。

因此当前生成阶段只保存：

```text
spectra_norm_ds
```

后续训练脚本中再生成：

```text
PCA(spectra_norm_ds)
```

---

## 十六、数据划分

必须按 `process_id` 划分 train / val / test，而不是按 sample 随机划分。

原因：同一 `process_id` 下 400 个 cavity 点共享同一组真实膜厚扰动，必须落在同一个 split 中。

当前比例：

```text
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
```

保存：

```text
train_process_ids
val_process_ids
test_process_ids
split_id
split_names = ["train", "val", "test"]
```

---

## 十七、checkpoint 和续跑机制

大规模仿真必须以 process 为单位保存 checkpoint：

```text
checkpoint_process_0000.npz
checkpoint_process_0001.npz
...
checkpoint_process_1999.npz
```

每个 checkpoint 包含该 process 下的 400 个 cavity 点结果。

如果中途停止或关机，已经写出的 checkpoint 不应丢失。

续跑要求：

1. 读取已有 checkpoint。
2. 找到最大 `checkpoint_process_XXXX.npz`。
3. 从 `process_id = XXXX + 1` 继续。
4. 不覆盖已有 checkpoint。
5. 继续写入同一个 run directory。
6. 续跑时必须读取原 run directory 中的 `00_config.json`，保证光谱保存模式和特征配置一致。

当前续跑脚本：

```text
work/03_ml_inverse_modeling/ML try/resume_nn_cavity_scalar_results.py
```

建议将其能力扩展为支持 spectral features，但可以保留原文件名。

续跑状态记录：

```text
resume_manifest.json
resume_failed_cases_YYYYMMDD_HHMMSS.json
```

---

## 十八、每个 checkpoint 字段

每个 `checkpoint_process_XXXX.npz` 应包含该 process 下 400 个 cavity 点的结果。

基础字段：

```text
sample_id
process_id
nominal_stack_id
split_id
valid_mask

cavity_true_um
L_true_um
L_fft_um
delta_L_um
delta_L_nm
H_peak
peak_count

film_nominal_nm
film_delta_nm
film_true_nm
layer_names
```

新增 full-spectrum 特征字段：

```text
spectral_features_full
spectral_feature_names
spectral_feature_source
```

可选光谱字段，根据 `spectra_save_mode` 保存：

如果 `norm_downsampled`：

```text
spectra_norm_ds
wavelengths_spectra_saved_um
```

如果 `norm_full`：

```text
spectra_norm
wavelengths_spectra_saved_um
```

如果 `raw_downsampled`：

```text
spectra_ds
wavelengths_spectra_saved_um
```

如果 `raw_and_norm_downsampled`：

```text
spectra_ds
spectra_norm_ds
wavelengths_spectra_saved_um
```

元数据字段：

```text
spectra_saved
spectra_save_mode
spectra_dtype
spectra_downsample_factor
spectra_downsample_method
spectrum_normalization
```

---

## 十九、输出文件要求

完整运行结束后，run directory 中应包含：

```text
00_config.json
build_nn_cavity_dataset.py
resume_nn_cavity_scalar_results.py
checkpoint_process_0000.npz
...
checkpoint_process_1999.npz
nn_cavity_spectral_features_YYYYMMDD_HHMMSS.npz
nn_cavity_spectral_features_index_YYYYMMDD_HHMMSS.csv
summary_YYYYMMDD_HHMMSS.json
failed_cases_YYYYMMDD_HHMMSS.json
failed_fft_cases_YYYYMMDD_HHMMSS.json
07_valid_mask_summary.json
```

检查图可以保存为：

```text
01_fft_vs_true_cavity.png
02_delta_L_hist.png
03_h_peak_hist.png
04_process_split.png
05_film_delta_distribution.png
06_nominal_model_coverage.png
07_example_spectra_norm_ds.png
08_spectral_feature_hist.png
```

如果 `save_spectra = False`，则不需要生成 `07_example_spectra_norm_ds.png`。

---

## 二十、最终 NPZ 字段

最终 NPZ 应包含：

```text
wavelengths_um
wavelengths_spectra_saved_um

sample_id
process_id
nominal_stack_id
nominal_stack_name_by_id
split_id
split_names

cavity_true_um
L_true_um
L_fft_um
delta_L_um
delta_L_nm
H_peak
peak_count

film_nominal_nm
film_delta_nm
film_true_nm
layer_names

spectral_features_full
spectral_feature_names
spectral_feature_source

cavity_axis_um
train_process_ids
val_process_ids
test_process_ids
valid_mask

nominal_stack_values_nm
process_nominal_stack_id
process_film_delta_nm
process_film_true_nm

num_samples_planned
num_processes_planned

spectra_saved
spectra_save_mode
spectra_dtype
spectra_downsample_factor
spectra_downsample_method
spectrum_normalization
spectra_norm_method

config_json
timestamp
```

根据保存模式，额外包含以下之一或多个：

```text
spectra_norm_ds
spectra_ds
spectra_norm
spectra
```

推荐默认只包含：

```text
spectra_norm_ds
```

默认不要包含：

```text
spectra
spectra_norm
```

---

## 二十一、CSV 索引字段

CSV 每行一个 sample，字段包括：

```text
sample_id
process_id
nominal_stack_id
nominal_stack_name
split_label
cavity_true_um
L_fft_um
delta_L_nm
H_peak
peak_count

film_HSQ_nominal_nm
film_PSS_nominal_nm
film_SOC_nominal_nm
film_TiO2_nominal_nm

film_HSQ_delta_nm
film_PSS_delta_nm
film_SOC_delta_nm
film_TiO2_delta_nm

film_HSQ_true_nm
film_PSS_true_nm
film_SOC_true_nm
film_TiO2_true_nm
```

CSV 可以额外保存部分关键 full-spectrum 特征，方便快速检查：

```text
spec_mean
spec_std
spec_ptp
fft_peak_pos_1_um
fft_peak_height_1
fft_peak_pos_2_um
fft_peak_height_ratio_21
fft_noise_floor
fft_snr_1
fringe_visibility_global
fringe_contrast_std
```

不建议把全部 `spectra_norm_ds` 写入 CSV。

---

## 二十二、summary JSON 字段

summary 应包含：

```text
output_mode
spectra_saved
spectra_save_mode
spectra_dtype
spectra_downsample_factor
spectra_downsample_method
spectrum_normalization
num_wavelengths_full
num_wavelengths_saved
estimated_spectra_storage_gb

num_samples_planned
num_samples_total
num_samples_valid
num_failed_simulation
num_failed_fft
num_nominal_stacks
num_processes
num_train_processes
num_val_processes
num_test_processes

wavelength_start_um
wavelength_stop_um
num_wavelengths
cavity_start_um
cavity_stop_um
cavity_step_um
num_cavity_points

film_layer_names
film_uncertainty_nm
nominal_ranges_nm
nominal_step_nm

L_fft_nan_count
delta_L_nm_mean
delta_L_nm_std
delta_L_nm_min
delta_L_nm_max

num_spectral_features
spectral_feature_names
spectral_feature_source
```

其中：

```text
output_mode = "scalar_plus_optional_spectra_and_full_features"
```

---

## 二十三、命令行参数

请支持以下参数：

```text
--save-spectra true/false
--spectra-save-mode none/norm_downsampled/norm_full/raw_downsampled/raw_and_norm_downsampled
--spectra-dtype float32/float16
--spectra-downsample-factor 10
--spectra-downsample-method mean/slice
--spectrum-normalization per_spectrum_zscore
--extract-full-spectral-features true/false
```

默认：

```text
--save-spectra true
--spectra-save-mode norm_downsampled
--spectra-dtype float32
--spectra-downsample-factor 10
--spectra-downsample-method mean
--spectrum-normalization per_spectrum_zscore
--extract-full-spectral-features true
```

推荐运行命令：

```bash
python build_nn_cavity_dataset.py ^
  --num-cavity-points 400 ^
  --save-spectra true ^
  --spectra-save-mode norm_downsampled ^
  --spectra-dtype float32 ^
  --spectra-downsample-factor 10 ^
  --spectra-downsample-method mean ^
  --extract-full-spectral-features true
```

快速 scalar-only 调试命令：

```bash
python build_nn_cavity_dataset.py ^
  --num-cavity-points 400 ^
  --save-spectra false ^
  --spectra-save-mode none ^
  --extract-full-spectral-features true
```

小规模完整光谱对照实验命令：

```bash
python build_nn_cavity_dataset.py ^
  --num-nominal-models 5 ^
  --num-process-per-nominal 5 ^
  --num-cavity-points 100 ^
  --save-spectra true ^
  --spectra-save-mode norm_full ^
  --spectra-dtype float32 ^
  --spectra-downsample-factor 1 ^
  --extract-full-spectral-features true
```

---

## 二十四、训练阶段使用建议

后续训练模型时，推荐按以下顺序使用数据字段。

### 24.1 scalar baseline

输入：

```text
L_fft_um
H_peak
film_nominal_nm
```

输出：

```text
delta_L_nm
```

### 24.2 scalar + selected quadratic

输入：

```text
L_fft_um
H_peak
film_nominal_nm
L_fft_um * film_nominal_nm
```

输出：

```text
delta_L_nm
```

### 24.3 scalar + full-spectrum features

输入：

```text
L_fft_um
H_peak
film_nominal_nm
spectral_features_full
```

输出：

```text
delta_L_nm
```

### 24.4 PCA spectrum nominal

输入：

```text
PCA(spectra_norm_ds)
L_fft_um
H_peak
film_nominal_nm
```

输出：

```text
delta_L_nm
```

注意：

```text
PCA 只能在 train set 上 fit，val/test 只能 transform。
```

### 24.5 PCA + full-spectrum features

输入：

```text
PCA(spectra_norm_ds)
L_fft_um
H_peak
film_nominal_nm
spectral_features_full
```

输出：

```text
delta_L_nm
```

这个版本用于判断：

```text
降采样光谱 PCA 特征 + 完整光谱手工特征
```

是否比单独 scalar 或单独 PCA 更好。

---

## 二十五、禁止事项

主实验中禁止使用以下字段作为输入：

```text
film_true_nm
film_delta_nm
PSS_true_nm
HSQ_true_nm
SOC_true_nm
TiO2_true_nm
PSS_delta_nm
HSQ_delta_nm
SOC_delta_nm
TiO2_delta_nm
cavity_true_um
L_true_um
```

其中：

```text
cavity_true_um
L_true_um
```

只能用于生成标签或评价，不能作为模型输入。

可以额外做 oracle 对照，但必须单独命名：

```text
scalar_oracle_true_thickness
```

并在报告里明确说明：

```text
This is not deployable because true film thickness is unavailable in real measurement.
```

默认不要开启 oracle。

---

## 二十六、运行结束后终端输出要求

完整运行结束时应打印：

```text
Dataset saved to: ...
NPZ path: ...
CSV path: ...
Summary path: ...

Output mode: scalar_plus_optional_spectra_and_full_features
Raw spectra saved: False by default
Normalized spectra saved: True by default
Spectra save mode: norm_downsampled
Full wavelengths: 20001
Saved wavelengths: depends on downsample factor
Spectra dtype: float32
Downsample factor: 10
Downsample method: mean
Estimated spectra storage: ... GB

Total planned samples:
Saved samples:
Valid samples:
Failed simulations:
Failed FFT:
Train/Val/Test processes:
delta_L_nm mean/std/min/max:

Full-spectrum features extracted: True
Number of full-spectrum features:
Spectral feature source: full_spectrum_before_downsampling
```

并确认：

```text
len(sample_id) == len(cavity_true_um) == len(L_fft_um)
film_nominal_nm.shape[0] == len(sample_id)
film_nominal_nm.shape[1] == len(layer_names)
spectral_features_full.shape[0] == len(sample_id)
len(spectral_feature_names) == spectral_features_full.shape[1]

if spectra_save_mode == "norm_downsampled":
    "spectra_norm_ds" in npz
    spectra_norm_ds.shape[0] == len(sample_id)
    spectra_norm_ds.shape[1] == len(wavelengths_spectra_saved_um)

"spectra" not in npz by default
"spectra_norm" not in npz by default
```

---

## 二十七、核心结论

当前数据集生成流程应从：

```text
scalar_only_no_raw_spectra
```

升级为：

```text
scalar_plus_optional_spectra_and_full_features
```

也就是：

```text
完整光谱用于 FFT 粗解和 full-spectrum 特征提取；
降采样归一化光谱用于后续 PCA / CNN；
完整原始光谱默认不落盘；
真实膜厚只用于仿真记录和 oracle 对照；
名义膜厚才是可部署模型输入。
```
