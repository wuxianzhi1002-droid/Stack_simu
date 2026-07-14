# Robust TMM Inverse Validation

## What changed from the first baseline

- Synthetic spectra are generated on a drifted true wavelength axis, but fitted on the recorded wavelength axis.
- Truth generation can perturb material n/k and fixed layer thicknesses; inverse fitting still uses the nominal model.
- Each spectrum is fitted from multiple random initial guesses inside the bounds.
- Near-equivalent solutions are counted to expose multi-solution risk.

## Configuration

```json
{
  "wavelength_start_um": 0.2,
  "wavelength_stop_um": 0.6,
  "spectral_resolution_nm": 0.2,
  "random_seed": 20260707,
  "noise_sigmas_reflectance": [
    0.0,
    0.005
  ],
  "mc_trials_per_noise": 2,
  "truth_samples_per_case": 2,
  "multistarts": 6,
  "max_nfev": 80,
  "fit_affine_intensity": true,
  "wavelength_offset_sigma_nm": 0.03,
  "wavelength_scale_sigma_ppm": 80.0,
  "material_real_sigma_fraction": 0.003,
  "material_imag_sigma_fraction": 0.05,
  "fixed_layer_sigma_nm": 0.5,
  "near_solution_cost_rel_tol": 0.001,
  "near_solution_cost_abs_tol": 1e-10,
  "source_scale_sigma": 0.02,
  "source_offset_sigma": 0.002
}
```

## Metrics

| scenario   | case             | noise | param |   n | success |  MAE_nm | P95Abs_nm | MaxAbs_nm | equiv_count | max_equiv_spread_nm |
| ---------- | ---------------- | ----: | ----- | --: | ------: | ------: | --------: | --------: | ----------: | ------------------: |
| tmm_robust | A0_cavity_air    |     0 | Air   |   2 |    1.00 |   209.4 |     308.2 |     319.2 |        1.00 |                   0 |
| tmm_robust | A0_cavity_air    | 0.005 | Air   |   4 |    1.00 |     557 |      1179 |      1300 |        1.00 |                   0 |
| tmm_robust | B1_film_tio2     |     0 | TiO2  |   2 |    1.00 |  0.1876 |    0.3318 |    0.3478 |        3.00 |            3.78e-08 |
| tmm_robust | B1_film_tio2     | 0.005 | TiO2  |   4 |    1.00 |  0.2234 |    0.3841 |    0.4123 |        3.25 |            2.52e-07 |
| tmm_robust | B2_film_soc_tio2 |     0 | SOC   |   2 |    1.00 |   1.295 |     1.388 |     1.399 |        2.50 |           2.458e-07 |
| tmm_robust | B2_film_soc_tio2 | 0.005 | SOC   |   4 |    1.00 |   1.159 |     1.282 |     1.297 |        3.50 |           1.008e-06 |
| tmm_robust | B2_film_soc_tio2 |     0 | TiO2  |   2 |    1.00 |  0.4404 |    0.6138 |    0.6331 |        2.50 |           2.661e-09 |
| tmm_robust | B2_film_soc_tio2 | 0.005 | TiO2  |   4 |    1.00 |  0.5125 |    0.8121 |    0.8508 |        3.50 |            1.67e-07 |
| tmm_robust | B3_film_all      |     0 | HSQ   |   2 |    1.00 |  0.4733 |    0.7567 |    0.7882 |        1.50 |           3.287e-07 |
| tmm_robust | B3_film_all      | 0.005 | HSQ   |   4 |    1.00 |   1.578 |     2.028 |     2.087 |        3.00 |           1.709e-05 |
| tmm_robust | B3_film_all      |     0 | PSS   |   2 |    1.00 |   1.544 |     2.863 |     3.009 |        1.50 |           1.371e-06 |
| tmm_robust | B3_film_all      | 0.005 | PSS   |   4 |    1.00 |    2.25 |     4.479 |     4.894 |        3.00 |           3.748e-05 |
| tmm_robust | B3_film_all      |     0 | SOC   |   2 |    1.00 |    1.14 |     1.978 |     2.071 |        1.50 |            1.15e-06 |
| tmm_robust | B3_film_all      | 0.005 | SOC   |   4 |    1.00 |   2.871 |     6.315 |     7.012 |        3.00 |            5.44e-05 |
| tmm_robust | B3_film_all      |     0 | TiO2  |   2 |    1.00 | 0.08375 |    0.1035 |    0.1057 |        1.50 |           6.958e-08 |
| tmm_robust | B3_film_all      | 0.005 | TiO2  |   4 |    1.00 |   0.242 |    0.4722 |    0.5199 |        3.00 |           4.534e-06 |

## Output files

- metrics: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_195417\metrics_summary.csv`
- fits: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_195417\fit_results.csv`
- plot: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_195417\robust_error_vs_noise_mae.png`

## Interpretation

This is a stricter synthetic validation, not a direct experimental claim. A large equivalent-solution spread means the spectrum admits several parameter sets with nearly identical residuals.
