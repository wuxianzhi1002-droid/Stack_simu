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
  "spectral_resolution_nm": 0.1,
  "random_seed": 20260707,
  "noise_sigmas_reflectance": [
    0.0,
    0.002,
    0.005
  ],
  "mc_trials_per_noise": 3,
  "truth_samples_per_case": 3,
  "multistarts": 12,
  "max_nfev": 120,
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

| scenario | case | noise | param | n | success | MAE_nm | P95Abs_nm | MaxAbs_nm | equiv_count | max_equiv_spread_nm |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| tmm_robust | A0_cavity_air | 0 | Air | 3 | 0.67 | 138.9 | 279.3 | 303.9 | 1.00 | 0 |
| tmm_robust | A0_cavity_air | 0.002 | Air | 9 | 1.00 | 224.7 | 626.9 | 849.2 | 1.33 | 0.04272 |
| tmm_robust | A0_cavity_air | 0.005 | Air | 9 | 1.00 | 162.8 | 333.2 | 343.5 | 1.33 | 0.003407 |
| tmm_robust | B1_film_tio2 | 0 | TiO2 | 3 | 1.00 | 0.4414 | 0.6127 | 0.6233 | 7.00 | 7.43e-08 |
| tmm_robust | B1_film_tio2 | 0.002 | TiO2 | 9 | 1.00 | 0.3548 | 0.7382 | 0.8379 | 5.67 | 1.158e-06 |
| tmm_robust | B1_film_tio2 | 0.005 | TiO2 | 9 | 1.00 | 0.3501 | 0.8143 | 1.101 | 4.78 | 2.715e-07 |
| tmm_robust | B2_film_soc_tio2 | 0 | SOC | 3 | 1.00 | 0.5467 | 0.8509 | 0.8819 | 7.33 | 1.373e-08 |
| tmm_robust | B2_film_soc_tio2 | 0.002 | SOC | 9 | 1.00 | 0.3305 | 0.6244 | 0.6412 | 5.11 | 2.754e-07 |
| tmm_robust | B2_film_soc_tio2 | 0.005 | SOC | 9 | 1.00 | 0.3037 | 0.6926 | 0.8005 | 6.89 | 2.091e-06 |
| tmm_robust | B2_film_soc_tio2 | 0 | TiO2 | 3 | 1.00 | 0.2783 | 0.4052 | 0.4055 | 7.33 | 2.624e-09 |
| tmm_robust | B2_film_soc_tio2 | 0.002 | TiO2 | 9 | 1.00 | 0.2909 | 0.7036 | 0.8235 | 5.11 | 2.954e-08 |
| tmm_robust | B2_film_soc_tio2 | 0.005 | TiO2 | 9 | 1.00 | 0.4009 | 0.7371 | 0.8315 | 6.89 | 2.051e-07 |
| tmm_robust | B3_film_all | 0 | HSQ | 3 | 1.00 | 1.64 | 3.003 | 3.17 | 5.67 | 4.281e-05 |
| tmm_robust | B3_film_all | 0.002 | HSQ | 9 | 1.00 | 1.322 | 3.082 | 3.536 | 3.89 | 0.0004562 |
| tmm_robust | B3_film_all | 0.005 | HSQ | 9 | 1.00 | 1.372 | 2.55 | 2.934 | 5.11 | 0.001897 |
| tmm_robust | B3_film_all | 0 | PSS | 3 | 1.00 | 1.613 | 2.539 | 2.647 | 5.67 | 6.042e-06 |
| tmm_robust | B3_film_all | 0.002 | PSS | 9 | 1.00 | 0.6752 | 1.674 | 1.983 | 3.89 | 0.000219 |
| tmm_robust | B3_film_all | 0.005 | PSS | 9 | 1.00 | 1.65 | 3.673 | 4.191 | 5.11 | 0.0003693 |
| tmm_robust | B3_film_all | 0 | SOC | 3 | 1.00 | 1.386 | 2.144 | 2.244 | 5.67 | 4.411e-05 |
| tmm_robust | B3_film_all | 0.002 | SOC | 9 | 1.00 | 1.012 | 2.102 | 2.47 | 3.89 | 0.0003837 |
| tmm_robust | B3_film_all | 0.005 | SOC | 9 | 1.00 | 0.5114 | 1.188 | 1.347 | 5.11 | 0.001903 |
| tmm_robust | B3_film_all | 0 | TiO2 | 3 | 1.00 | 0.426 | 0.7037 | 0.746 | 5.67 | 1.989e-06 |
| tmm_robust | B3_film_all | 0.002 | TiO2 | 9 | 1.00 | 0.3626 | 0.66 | 0.7327 | 3.89 | 1.619e-05 |
| tmm_robust | B3_film_all | 0.005 | TiO2 | 9 | 1.00 | 0.3013 | 0.4969 | 0.5196 | 5.11 | 9.99e-05 |

## Output files

- metrics: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_203937\metrics_summary.csv`
- fits: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_203937\fit_results.csv`
- plot: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_203937\robust_error_vs_noise_mae.png`

## Interpretation

This is a stricter synthetic validation, not a direct experimental claim. A large equivalent-solution spread means the spectrum admits several parameter sets with nearly identical residuals.
