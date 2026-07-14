# 大规模 StackRT 标量结果数据集生成要求

更新时间：2026-06-17

本文件用于记录当前项目中 `work/03_ml_inverse_modeling/ML try/` 目录下神经网络/残差模型数据集的最新仿真范围、保存格式和续跑要求。

当前版本的核心变化是：

1. 仿真仍然使用 `lumapi + stackrt` 逐条生成光谱。
2. 每条光谱只在内存中临时用于 FFT 粗解。
3. 不再保存原始光谱矩阵 `spectra`。
4. 不再保存归一化光谱矩阵 `spectra_norm`。
5. 最终只保存标量结果和膜厚/腔长标签。

这样可以避免完整光谱数据集达到数百 GB 的存储压力。

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

当前长仿真输出目录：

```text
work/03_ml_inverse_modeling/ML try/nn_cavity_scalar_results_20260617_085429/
```

---

## 二、当前数据集类型

当前数据集类型为：

```text
scalar_only_no_raw_spectra
```

也就是说：

保存：

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

不保存：

```text
spectra
spectra_norm
raw StackRT reflectance matrix
normalized spectrum matrix
```

注意：由于不保存光谱，后续训练不能直接使用 `X_spectrum = spectra_norm`。当前数据更适合训练或分析：

```text
X_scalar = [L_fft_um, H_peak]
X_film = film_nominal_nm
y = delta_L_nm
```

最终预测形式仍建议为：

```text
L_pred_um = L_fft_um + delta_L_pred_nm / 1000
```

---

## 三、仿真光谱范围

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

每次 StackRT 返回的光谱长度应为 `20001`。

虽然最终不保存 `spectra`，仍需要保存：

```text
wavelengths_um
```

用于记录 FFT 解算所基于的波长轴。

---

## 四、腔长扫描范围

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

---

## 五、nominal model 范围

当前要求使用 `100` 个 nominal models。

每个膜层厚度必须是 `10 nm` 的倍数，并从以下范围内选取：

```text
PSS_nm:  10，20，30,40
HSQ_nm:  20, 30, 40, 50, 60
SOC_nm:  40, 50, 60, 70, 80
TiO2_nm: 30, 40, 50, 60, 70，80
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

当前实现中的默认随机种子：

```text
random_seed = 20260620
```

---

## 六、process 设置

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

## 七、总仿真规模

当前大规模仿真总量为：

```text
100 nominal models
* 20 process / nominal model
* 1000 cavity points / process
= 2,000,000 samples
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
checkpoint_process_1999.npz -> process_id = 1999，对应全部 process 完成
```

---

## 八、层结构和材料模型

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
其他膜层 nominal/true/delta 保存为 nm
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

## 九、FFT 粗解要求

每条 StackRT 光谱生成后立即执行 FFT 粗解，然后丢弃原始光谱。

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

## 十、数据划分

必须按 `process_id` 划分 train / val / test，而不是按 sample 随机划分。

原因：同一 `process_id` 下 1000 个 cavity 点共享同一组真实膜厚扰动，必须落在同一个 split 中。

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

## 十一、checkpoint 和续跑机制

大规模仿真必须以 process 为单位保存 checkpoint：

```text
checkpoint_process_0000.npz
checkpoint_process_0001.npz
...
checkpoint_process_1999.npz
```

每个 checkpoint 包含该 process 下的 1000 个 cavity 点结果。

如果中途停止或关机，已经写出的 checkpoint 不应丢失。

续跑要求：

1. 读取已有 checkpoint。
2. 找到最大 `checkpoint_process_XXXX.npz`。
3. 从 `process_id = XXXX + 1` 继续。
4. 不覆盖已有 checkpoint。
5. 继续写入同一个 run directory。

当前续跑脚本：

```text
work/03_ml_inverse_modeling/ML try/resume_nn_cavity_scalar_results.py
```

当前续跑目录：

```text
work/03_ml_inverse_modeling/ML try/nn_cavity_scalar_results_20260617_085429/
```

续跑状态记录：

```text
resume_manifest.json
resume_failed_cases_YYYYMMDD_HHMMSS.json
```

---

## 十二、输出文件要求

完整运行结束后，run directory 中应包含：

```text
00_config.json
build_nn_cavity_dataset.py
resume_nn_cavity_scalar_results.py
checkpoint_process_0000.npz
...
checkpoint_process_1999.npz
nn_cavity_scalar_results_YYYYMMDD_HHMMSS.npz
nn_cavity_scalar_results_index_YYYYMMDD_HHMMSS.csv
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
```

注意：当前版本不需要 `01_example_spectra.png`，因为不保存原始光谱。

---

## 十三、最终 NPZ 字段

最终 scalar-only NPZ 应包含：

```text
wavelengths_um
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
spectra_norm_saved
config_json
timestamp
```

其中：

```text
spectra_saved = False
spectra_norm_saved = False
```

不得包含：

```text
spectra
spectra_norm
```

---

## 十四、CSV 索引字段

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

---

## 十五、summary JSON 字段

summary 应包含：

```text
output_mode
spectra_saved
spectra_norm_saved
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
```

---

## 十六、当前已运行状态记录

当前长仿真目录：

```text
work/03_ml_inverse_modeling/ML try/nn_cavity_scalar_results_20260617_085429/
```

曾在 `process_id = 12` 后暂停，并保存：

```text
partial_scalar_results_through_process_0012.npz
partial_scalar_results_through_process_0012.csv
partial_summary_through_process_0012.json
resume_manifest.json
```

之后已通过 `resume_nn_cavity_scalar_results.py` 继续仿真。

判断进度的方法：

```text
checkpoint_process_0099.npz 存在 -> 前 100 个 process 已完成
checkpoint_process_0199.npz 存在 -> 前 200 个 process 已完成
checkpoint_process_1999.npz 存在 -> 全部 2000 个 process 已完成
```

---

## 十七、运行结束后终端输出要求

完整运行结束时应打印：

```text
Dataset saved to: ...
NPZ path: ...
CSV path: ...
Summary path: ...
Output mode: scalar_only_no_raw_spectra
Raw spectra saved: False
Total planned samples:
Saved samples:
Valid samples:
Failed simulations:
Failed FFT:
Train/Val/Test processes:
delta_L_nm mean/std/min/max:
```

并确认：

```text
len(sample_id) == len(cavity_true_um) == len(L_fft_um)
film_nominal_nm.shape[0] == len(sample_id)
film_nominal_nm.shape[1] == len(layer_names)
"spectra" not in npz
"spectra_norm" not in npz
```
