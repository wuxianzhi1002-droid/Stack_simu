# Simple PCA Residual MLP Summary Report

## Model Definitions

- `base_scalar`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm.
- `more_feature`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15, PC16, PC17, PC18, PC19, PC20, PC21, PC22, PC23, PC24, PC25, PC26, PC27, PC28, PC29, PC30, PC31, PC32, PC33, PC34, PC35, PC36, PC37, PC38, PC39, PC40, PC41, PC42, PC43, PC44, PC45, PC46, PC47, PC48, PC49, PC50.
- Additional features: fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15, PC16, PC17, PC18, PC19, PC20, PC21, PC22, PC23, PC24, PC25, PC26, PC27, PC28, PC29, PC30, PC31, PC32, PC33, PC34, PC35, PC36, PC37, PC38, PC39, PC40, PC41, PC42, PC43, PC44, PC45, PC46, PC47, PC48, PC49, PC50.
- No quadratic or interaction features are generated.

## PCA Audit

- PCA enabled for `more_feature`: yes.
- PCA fit policy: `fit_on_train_only`.
- PCA components used: 50 of 100.
- Cumulative explained variance ratio: 0.98551904.
- PCA source: `spectra_norm_ds`; method: `randomized`.
- PCA fit rows: 100000.

## Method Comparison

| method | cavity_MAE_nm | cavity_RMSE_nm | cavity_MaxAbs_nm | delta_P95Abs_nm | R2_delta |
|---|---:|---:|---:|---:|---:|
| base_scalar | 11.890 | 15.385 | 60.500 | 31.825 | 0.97333 |
| more_feature | 1.322 | 1.728 | 19.833 | 3.366 | 0.99966 |

## Conclusion

- `more_feature` has lower test RMSE than `base_scalar`.
- base_scalar test: MAE=11.890 nm, RMSE=15.385 nm, MaxAbs=60.500 nm, R2=0.97333
- more_feature test: MAE=1.322 nm, RMSE=1.728 nm, MaxAbs=19.833 nm, R2=0.99966
- High-correlation feature pairs in more_feature: 0.
- True thickness, film_delta, cavity_true_um, and target fields are not model inputs.
