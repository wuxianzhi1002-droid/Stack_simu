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
| more_feature_pca50_nominal_thickness | 1.390 | 1.815 | 22.782 | 3.553 | 0.99963 |
| more_feature_pca50_true_thickness | 1.208 | 1.571 | 20.652 | 3.078 | 0.99972 |

## Conclusion

- `more_feature_pca50_true_thickness` has lower test RMSE than `more_feature_pca50_nominal_thickness`.
- nominal/fuzzy thickness test: MAE=1.390 nm, RMSE=1.815 nm, MaxAbs=22.782 nm, R2=0.99963
- true thickness test: MAE=1.208 nm, RMSE=1.571 nm, MaxAbs=20.652 nm, R2=0.99972
- High-correlation feature pairs in nominal/fuzzy model inputs: 0.
- Only the true-thickness model uses film_true_nm. Target fields, film_delta, and cavity_true_um are never model inputs.
