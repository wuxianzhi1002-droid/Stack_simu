# Simple Multi-Output Residual MLP Summary Report

## Model Definitions

- `base_scalar`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm.
- `more_feature`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- Additional features: fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- No quadratic or interaction features are generated.
- Outputs: L_true_um (um), PSS_true_nm (nm), HSQ_true_nm (nm), SOC_true_nm (nm), TiO2_true_nm (nm).

## Method Comparison

| method | cavity_MAE_nm_equiv | cavity_RMSE_nm_equiv | film_mean_MAE_nm | film_max_MAE_nm | mean_R2 |
|---|---:|---:|---:|---:|---:|
| base_scalar | 126.583 | 145.613 | 14.655 | 21.655 | -0.27717 |
| more_feature | 137.220 | 163.555 | 16.524 | 18.553 | -0.62949 |

## Per-Target Test Metrics

| method | target | unit | MAE | RMSE | MaxAbs | R2 |
|---|---|---|---:|---:|---:|---:|
| base_scalar | L_true_um | um | 0.126583 | 0.145613 | 0.261462 | -0.59450 |
| base_scalar | PSS_true_nm | nm | 13.7765 | 16.3953 | 36.6721 | -0.59954 |
| base_scalar | HSQ_true_nm | nm | 13.1911 | 15.607 | 45.5124 | 0.13819 |
| base_scalar | SOC_true_nm | nm | 9.99833 | 12.3882 | 32.4686 | 0.40358 |
| base_scalar | TiO2_true_nm | nm | 21.6545 | 25.5887 | 61.6253 | -0.73357 |
| more_feature | L_true_um | um | 0.13722 | 0.163555 | 0.316542 | -1.01166 |
| more_feature | PSS_true_nm | nm | 15.5324 | 19.0168 | 43.7787 | -1.15195 |
| more_feature | HSQ_true_nm | nm | 15.644 | 19.181 | 49.3222 | -0.30170 |
| more_feature | SOC_true_nm | nm | 18.5528 | 20.8327 | 44.7613 | -0.68665 |
| more_feature | TiO2_true_nm | nm | 16.3652 | 19.3909 | 48.0691 | 0.00451 |

## Conclusion

- `more_feature` does not have higher mean test R2 than `base_scalar`.
- base_scalar test: cavity_MAE=126.583 nm equiv, film_mean_MAE=14.655 nm, film_max_MAE=21.655 nm, mean_R2=-0.27717
- more_feature test: cavity_MAE=137.220 nm equiv, film_mean_MAE=16.524 nm, film_max_MAE=18.553 nm, mean_R2=-0.62949
- High-correlation feature pairs in more_feature: 0.
- True thickness, film_delta, cavity_true_um, L_true_um, and target fields are not model inputs.
