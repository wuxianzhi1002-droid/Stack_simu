# Spectral CNN Validation Summary

- model: `cnn_small`
- device: `cpu`
- dataset: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\03_ml_inverse_modeling\ML try\nn_cavity_spectral_features_20260620_233057`
- spectra points: `2000`
- selected rows: `{'train': 1600, 'val': 800, 'test': 800}`
- unique processes: `{'train': 4, 'val': 2, 'test': 2}`
- checkpoint files scanned/used: `12` / `8`

## Notes

- Dataset provides spectra_norm_ds with 2000 points; plan text mentions 20000 points.

## Test Metrics

### cnn_small

- `delta_MAE_nm`: 12.8445
- `delta_RMSE_nm`: 15.1089
- `delta_MaxAbs_nm`: 29.7398
- `delta_P95Abs_nm`: 26.4634
- `delta_P99Abs_nm`: 28.6658
- `delta_Bias_nm`: -12.5701
- `R2_delta`: -3.23198
- `cavity_MAE_nm`: 12.8438
- `cavity_RMSE_nm`: 15.1084
- `cavity_MaxAbs_nm`: 29.7173

### fft_only

- `delta_MAE_nm`: 918.202
- `delta_RMSE_nm`: 918.231
- `delta_MaxAbs_nm`: 933.442
- `delta_P95Abs_nm`: 930.685
- `delta_P99Abs_nm`: 932.455
- `delta_Bias_nm`: 918.202
- `R2_delta`: -15629.8
- `cavity_MAE_nm`: 918.203
- `cavity_RMSE_nm`: 918.232
- `cavity_MaxAbs_nm`: 933.439

### scalar_ridge

- `delta_MAE_nm`: 11.6287
- `delta_RMSE_nm`: 13.5249
- `delta_MaxAbs_nm`: 27.5724
- `delta_P95Abs_nm`: 23.651
- `delta_P99Abs_nm`: 25.6488
- `delta_Bias_nm`: -11.3668
- `R2_delta`: -2.39116
- `cavity_MAE_nm`: 11.6278
- `cavity_RMSE_nm`: 13.5242
- `cavity_MaxAbs_nm`: 27.5499
