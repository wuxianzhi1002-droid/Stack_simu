# Spectral CNN Training Summary

- model: `cnn_small_adaptive_k`
- pooling: `adaptive_k`
- pooling k: `16`
- spectral position partitions retained: Yes; the encoded spectrum is retained as 16 ordered regions.
- spectra input length: `2000`
- uses H_peak: `False`
- split is process-level: `True`
- device: `cpu`
- prepared dataset: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\03_ml_inverse_modeling\ML try\Residual MLP\CNN\cnn_dataset`
- prepared source type: `checkpoint_directory`
- selected rows: `{'train': 560000, 'val': 120000, 'test': 120000}`
- unique processes: `{'train': 1400, 'val': 300, 'test': 300}`
- scalar inputs: `['L_fft_um', 'film_nominal_nm[0]', 'film_nominal_nm[1]', 'film_nominal_nm[2]', 'film_nominal_nm[3]']`

## Test Metrics

### cnn_small_adaptive_k

- `delta_MAE_nm`: 5.87345
- `delta_RMSE_nm`: 7.19569
- `delta_MaxAbs_nm`: 28.2266
- `delta_P95Abs_nm`: 13.6603
- `delta_P99Abs_nm`: 17.2466
- `delta_Bias_nm`: -0.00969117
- `R2_delta`: 0.994167
- `cavity_MAE_nm`: 5.87347
- `cavity_RMSE_nm`: 7.19572
- `cavity_MaxAbs_nm`: 28.2333

### fft_only

- `delta_MAE_nm`: 1064.02
- `delta_RMSE_nm`: 1068.18
- `delta_MaxAbs_nm`: 1267.17
- `delta_P95Abs_nm`: 1213.18
- `delta_P99Abs_nm`: 1240.94
- `delta_Bias_nm`: 1064.02
- `R2_delta`: -127.551
- `cavity_MAE_nm`: 1064.02
- `cavity_RMSE_nm`: 1068.18
- `cavity_MaxAbs_nm`: 1267.16

### scalar_ridge

- `delta_MAE_nm`: 25.3619
- `delta_RMSE_nm`: 31.0261
- `delta_MaxAbs_nm`: 100.667
- `delta_P95Abs_nm`: 58.9584
- `delta_P99Abs_nm`: 72.8358
- `delta_Bias_nm`: -0.898176
- `R2_delta`: 0.891549
- `cavity_MAE_nm`: 25.362
- `cavity_RMSE_nm`: 31.0263
- `cavity_MaxAbs_nm`: 100.662
