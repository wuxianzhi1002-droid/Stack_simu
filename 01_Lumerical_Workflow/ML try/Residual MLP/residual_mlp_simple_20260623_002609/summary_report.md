# Simple Residual MLP Summary Report

## Model Definitions

- `base_scalar`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm.
- `more_feature`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- Additional features: fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- No quadratic or interaction features are generated.

## Method Comparison

| method | cavity_MAE_nm | cavity_RMSE_nm | cavity_MaxAbs_nm | delta_P95Abs_nm | R2_delta |
|---|---:|---:|---:|---:|---:|
| base_scalar | 11.894 | 15.451 | 60.655 | 31.352 | 0.97267 |
| more_feature | 8.661 | 11.551 | 71.291 | 23.536 | 0.98473 |

## Conclusion

- `more_feature` has lower test RMSE than `base_scalar`.
- base_scalar test: MAE=11.894 nm, RMSE=15.451 nm, MaxAbs=60.655 nm, R2=0.97267
- more_feature test: MAE=8.661 nm, RMSE=11.551 nm, MaxAbs=71.291 nm, R2=0.98473
- High-correlation feature pairs in more_feature: 0.
- True thickness, film_delta, cavity_true_um, and target fields are not model inputs.
