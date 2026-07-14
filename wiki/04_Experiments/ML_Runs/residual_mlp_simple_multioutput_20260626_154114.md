# Simple Multi-Output Residual MLP Summary Report

## Model Definitions

- `base_scalar`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm.
- `more_feature`: L_fft_um, PSS_nominal_nm, HSQ_nominal_nm, SOC_nominal_nm, TiO2_nominal_nm, fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- Additional features: fft_spectral_centroid_um, fringe_visibility_global, fringe_contrast_std.
- No quadratic or interaction features are generated.
- Outputs: L_true_um (um), PSS_true_nm (nm), HSQ_true_nm (nm), SOC_true_nm (nm), TiO2_true_nm (nm).
- Film output constraint: `film_pred_nm = film_nominal_nm + 5 * tanh(raw_delta)`, so predictions stay within nominal +/- 5 nm.
- Training labels outside this film prior: rows=0, values=0, max_abs_delta=4.99917 nm.

## Method Comparison

| method | cavity_MAE_nm_equiv | cavity_RMSE_nm_equiv | film_mean_MAE_nm | film_max_MAE_nm | mean_R2 |
|---|---:|---:|---:|---:|---:|
| base_scalar | 127.257 | 146.266 | 3.294 | 3.727 | 0.62369 |
| more_feature | 137.122 | 163.302 | 3.016 | 3.313 | 0.55200 |

## Per-Target Test Metrics

| method | target | unit | MAE | RMSE | MaxAbs | R2 |
|---|---|---|---:|---:|---:|---:|
| base_scalar | L_true_um | um | 0.127257 | 0.146266 | 0.263243 | -0.60884 |
| base_scalar | PSS_true_nm | nm | 3.64836 | 4.4118 | 9.25337 | 0.88418 |
| base_scalar | HSQ_true_nm | nm | 2.92257 | 3.48435 | 9.19756 | 0.95705 |
| base_scalar | SOC_true_nm | nm | 3.72748 | 4.56917 | 9.71158 | 0.91887 |
| base_scalar | TiO2_true_nm | nm | 2.87724 | 3.51959 | 7.75065 | 0.96720 |
| more_feature | L_true_um | um | 0.137122 | 0.163302 | 0.314554 | -1.00544 |
| more_feature | PSS_true_nm | nm | 3.31313 | 4.09196 | 8.65834 | 0.90036 |
| more_feature | HSQ_true_nm | nm | 3.07886 | 3.9251 | 9.14537 | 0.94549 |
| more_feature | SOC_true_nm | nm | 3.02434 | 3.55889 | 8.71002 | 0.95078 |
| more_feature | TiO2_true_nm | nm | 2.6461 | 3.43383 | 8.67912 | 0.96878 |

## Conclusion

- `more_feature` does not have higher mean test R2 than `base_scalar`.
- base_scalar test: cavity_MAE=127.257 nm equiv, film_mean_MAE=3.294 nm, film_max_MAE=3.727 nm, mean_R2=0.62369
- more_feature test: cavity_MAE=137.122 nm equiv, film_mean_MAE=3.016 nm, film_max_MAE=3.313 nm, mean_R2=0.55200
- High-correlation feature pairs in more_feature: 0.
- True thickness, film_delta, cavity_true_um, L_true_um, and target fields are not model inputs.
