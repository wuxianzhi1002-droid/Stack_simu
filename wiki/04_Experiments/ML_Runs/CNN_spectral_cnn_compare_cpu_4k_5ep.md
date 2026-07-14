# Spectral CNN Validation Summary

- model: `cnn_small`
- device: `cpu`
- dataset: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\03_ml_inverse_modeling\ML try\nn_cavity_spectral_features_20260620_233057`
- spectra points: `2000`
- selected rows: `{'train': 4000, 'val': 1000, 'test': 1000}`
- unique processes: `{'train': 10, 'val': 3, 'test': 3}`
- checkpoint files scanned/used: `19` / `16`

## Notes

- Dataset provides spectra_norm_ds with 2000 points; plan text mentions 20000 points.

## Test Metrics

### cnn_small

- `delta_MAE_nm`: 5.36923
- `delta_RMSE_nm`: 6.4626
- `delta_MaxAbs_nm`: 15.1967
- `delta_P95Abs_nm`: 11.9233
- `delta_P99Abs_nm`: 13.6627
- `delta_Bias_nm`: 1.37246
- `R2_delta`: 0.178351
- `cavity_MAE_nm`: 5.37056
- `cavity_RMSE_nm`: 6.46432
- `cavity_MaxAbs_nm`: 15.2031

### fft_only

- `delta_MAE_nm`: 917.526
- `delta_RMSE_nm`: 917.554
- `delta_MaxAbs_nm`: 933.442
- `delta_P95Abs_nm`: 930.191
- `delta_P99Abs_nm`: 932.446
- `delta_Bias_nm`: 917.526
- `R2_delta`: -16561.9
- `cavity_MAE_nm`: 917.527
- `cavity_RMSE_nm`: 917.555
- `cavity_MaxAbs_nm`: 933.439

### scalar_ridge

- `delta_MAE_nm`: 10.1836
- `delta_RMSE_nm`: 11.8987
- `delta_MaxAbs_nm`: 25.606
- `delta_P95Abs_nm`: 20.9826
- `delta_P99Abs_nm`: 23.3566
- `delta_Bias_nm`: -9.57344
- `R2_delta`: -1.78528
- `cavity_MAE_nm`: 10.1827
- `cavity_RMSE_nm`: 11.8977
- `cavity_MaxAbs_nm`: 25.5835
