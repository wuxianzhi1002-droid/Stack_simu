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
| base_scalar | 11.980 | 15.556 | 2.412 | 2.469 | 0.96636 |
| more_feature | 8.911 | 11.707 | 2.109 | 2.410 | 0.97053 |

## Per-Target Test Metrics

| method | target | unit | MAE | RMSE | MaxAbs | R2 |
|---|---|---|---:|---:|---:|---:|
| base_scalar | L_true_um | um | 0.0119795 | 0.0155558 | 0.0619686 | 0.98185 |
| base_scalar | PSS_true_nm | nm | 2.37806 | 2.84442 | 7.08838 | 0.94469 |
| base_scalar | HSQ_true_nm | nm | 2.42073 | 2.83236 | 6.95972 | 0.96709 |
| base_scalar | SOC_true_nm | nm | 2.46905 | 2.95158 | 7.35405 | 0.95940 |
| base_scalar | TiO2_true_nm | nm | 2.38077 | 2.81483 | 6.69975 | 0.97876 |
| more_feature | L_true_um | um | 0.00891147 | 0.0117068 | 0.0695352 | 0.98972 |
| more_feature | PSS_true_nm | nm | 2.28719 | 2.82119 | 9.34964 | 0.94559 |
| more_feature | HSQ_true_nm | nm | 2.38178 | 2.86534 | 9.59222 | 0.96632 |
| more_feature | SOC_true_nm | nm | 2.41027 | 2.94232 | 9.21723 | 0.95966 |
| more_feature | TiO2_true_nm | nm | 1.35731 | 1.79269 | 7.98375 | 0.99139 |

## Conclusion

- `more_feature` has higher mean test R2 than `base_scalar`.
- base_scalar test: cavity_MAE=11.980 nm equiv, film_mean_MAE=2.412 nm, film_max_MAE=2.469 nm, mean_R2=0.96636
- more_feature test: cavity_MAE=8.911 nm equiv, film_mean_MAE=2.109 nm, film_max_MAE=2.410 nm, mean_R2=0.97053
- High-correlation feature pairs in more_feature: 0.
- True thickness, film_delta, cavity_true_um, L_true_um, and target fields are not model inputs.
