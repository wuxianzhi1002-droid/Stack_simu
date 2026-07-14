# PCA50 Residual MLP Limit Summary Report

## Model Definitions

- `more_feature_pca50_nominal_thickness`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15, PC16, PC17, PC18, PC19, PC20, PC21, PC22, PC23, PC24, PC25, PC26, PC27, PC28, PC29, PC30, PC31, PC32, PC33, PC34, PC35, PC36, PC37, PC38, PC39, PC40, PC41, PC42, PC43, PC44, PC45, PC46, PC47, PC48, PC49, PC50.
- `more_feature_pca50_true_thickness`: L_fft_um, PSS_true_nm, HSQ_true_nm, SOC_true_nm, TiO2_true_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15, PC16, PC17, PC18, PC19, PC20, PC21, PC22, PC23, PC24, PC25, PC26, PC27, PC28, PC29, PC30, PC31, PC32, PC33, PC34, PC35, PC36, PC37, PC38, PC39, PC40, PC41, PC42, PC43, PC44, PC45, PC46, PC47, PC48, PC49, PC50.
- Both models use MORE_FEATURE_NAMES plus PC1~PC50; only the thickness inputs differ.
- No quadratic or interaction features are generated.

## PCA Audit

- PCA enabled: yes.
- PCA fit policy: `fit_on_train_only`.
- PCA components used: 50 of 100.
- Cumulative explained variance ratio: 0.98551904.
- PCA source: `spectra_norm_ds`; method: `randomized`.
- PCA fit rows: 100000.

## Method Comparison

| method | cavity_MAE_nm | cavity_RMSE_nm | cavity_MaxAbs_nm | delta_P95Abs_nm | R2_delta |
|---|---:|---:|---:|---:|---:|
| more_feature_pca50_nominal_thickness | 52.867 | 62.256 | 196.337 | 104.224 | 0.54102 |
| more_feature_pca50_true_thickness | 53.232 | 62.588 | 196.542 | 102.520 | 0.53611 |

## Conclusion

- `more_feature_pca50_true_thickness` does not have lower test RMSE than `more_feature_pca50_nominal_thickness`.
- nominal/fuzzy thickness test: MAE=52.867 nm, RMSE=62.256 nm, MaxAbs=196.337 nm, R2=0.54102
- true thickness test: MAE=53.232 nm, RMSE=62.588 nm, MaxAbs=196.542 nm, R2=0.53611
- High-correlation feature pairs in nominal/fuzzy model inputs: 0.
- Only the true-thickness model uses film_true_nm. Target fields, film_delta, and cavity_true_um are never model inputs.
