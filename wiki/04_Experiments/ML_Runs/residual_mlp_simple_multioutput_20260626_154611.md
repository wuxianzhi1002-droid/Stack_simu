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
| base_scalar | 12.123 | 15.709 | 2.489 | 2.579 | 0.96405 |
| more_feature | 10.580 | 13.631 | 2.214 | 2.534 | 0.96577 |

## Per-Target Test Metrics

| method | target | unit | MAE | RMSE | MaxAbs | R2 |
|---|---|---|---:|---:|---:|---:|
| base_scalar | L_true_um | um | 0.0121234 | 0.0157092 | 0.0644823 | 0.98149 |
| base_scalar | PSS_true_nm | nm | 2.41809 | 2.92279 | 7.98237 | 0.94160 |
| base_scalar | HSQ_true_nm | nm | 2.48305 | 2.91838 | 8.06802 | 0.96506 |
| base_scalar | SOC_true_nm | nm | 2.57903 | 3.11088 | 8.14141 | 0.95490 |
| base_scalar | TiO2_true_nm | nm | 2.47484 | 2.91648 | 6.96001 | 0.97720 |
| more_feature | L_true_um | um | 0.0105797 | 0.0136306 | 0.0699558 | 0.98607 |
| more_feature | PSS_true_nm | nm | 2.41407 | 3.0118 | 9.24158 | 0.93799 |
| more_feature | HSQ_true_nm | nm | 2.52791 | 3.11424 | 9.39974 | 0.96021 |
| more_feature | SOC_true_nm | nm | 2.53373 | 3.15509 | 9.15396 | 0.95361 |
| more_feature | TiO2_true_nm | nm | 1.38016 | 1.83739 | 8.76415 | 0.99095 |

## Conclusion

- `more_feature` has higher mean test R2 than `base_scalar`.
- base_scalar test: cavity_MAE=12.123 nm equiv, film_mean_MAE=2.489 nm, film_max_MAE=2.579 nm, mean_R2=0.96405
- more_feature test: cavity_MAE=10.580 nm equiv, film_mean_MAE=2.214 nm, film_max_MAE=2.534 nm, mean_R2=0.96577
- High-correlation feature pairs in more_feature: 0.
- True thickness, film_delta, cavity_true_um, L_true_um, and target fields are not model inputs.
