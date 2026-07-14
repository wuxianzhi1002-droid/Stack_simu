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
  "spectral_resolution_nm": 2.0,
  "random_seed": 20260707,
  "noise_sigmas_reflectance": [
    0.005
  ],
  "mc_trials_per_noise": 1,
  "truth_samples_per_case": 1,
  "multistarts": 2,
  "max_nfev": 20,
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
| tmm_robust | B1_film_tio2 | 0.005 | TiO2 | 1 | 1.00 | 0.6673 | 0.6673 | 0.6673 | 2.00 | 2.317e-07 |

## Output files

- metrics: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_192652\metrics_summary.csv`
- fits: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_192652\fit_results.csv`
- plot: `D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\work\04_results_and_datasets\tmm_inverse_validation_robust_20260707_192652\robust_error_vs_noise_mae.png`

## Interpretation

This is a stricter synthetic validation, not a direct experimental claim. A large equivalent-solution spread means the spectrum admits several parameter sets with nearly identical residuals.
