# Simple Residual MLP Limit Summary Report

## Model Definitions

- `base_scalar`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm.
- `more_feature`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- `more_feature_true_thickness`: L_fft_um, PSS_true_nm, HSQ_true_nm, SOC_true_nm, TiO2_true_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- Additional features: fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- No quadratic or interaction features are generated.

## Method Comparison

| method | cavity_MAE_nm | cavity_RMSE_nm | cavity_MaxAbs_nm | delta_P95Abs_nm | R2_delta |
|---|---:|---:|---:|---:|---:|
| base_scalar | 46.188 | 55.441 | 127.536 | 97.554 | 0.62428 |
| more_feature | 43.532 | 51.352 | 117.099 | 94.284 | 0.67767 |
| more_feature_true_thickness | 43.403 | 51.062 | 116.114 | 89.865 | 0.68130 |

## Conclusion

- `more_feature` has lower test RMSE than `base_scalar`.
- `more_feature_true_thickness` has lower test RMSE than `more_feature`.
- base_scalar test: MAE=46.188 nm, RMSE=55.441 nm, MaxAbs=127.536 nm, R2=0.62428
- more_feature test: MAE=43.532 nm, RMSE=51.352 nm, MaxAbs=117.099 nm, R2=0.67767
- more_feature_true_thickness test: MAE=43.403 nm, RMSE=51.062 nm, MaxAbs=116.114 nm, R2=0.68130
- High-correlation feature pairs in more_feature: 0.
- Only `more_feature_true_thickness` uses true film thickness. Target fields, film_delta, and cavity_true_um are never model inputs.
