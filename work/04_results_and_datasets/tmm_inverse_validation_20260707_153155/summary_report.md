# TMM Inverse Validation Summary

## Scope

- Forward model: coherent TMM with the simplified material model from `main_cavity.py`.
- Branch A keeps `RefReflector / Air / films / Cu` and fits the 1 mm air cavity.
- Branch B removes the air cavity and fits film thickness from `Air / films / Cu` reflectance.
- Optional affine intensity scale/offset is fitted analytically inside each spectral residual.

## Configuration

```json
{
  "wavelength_start_um": 0.2,
  "wavelength_stop_um": 0.6,
  "spectral_resolution_nm": 0.02,
  "random_seed": 20260707,
  "noise_sigmas_reflectance": [
    0.0,
    0.005
  ],
  "mc_trials_per_noise": 1,
  "truth_samples_per_case": 2,
  "max_nfev": 30,
  "fit_affine_intensity": true,
  "objective": "reflectance_with_optional_affine_scale_offset",
  "source_model": "main_cavity.py simplified material model"
}
```

## Metrics

| case | noise_sigma_R | param | n | success | MAE_nm | RMSE_nm | P95Abs_nm | MaxAbs_nm | bias_nm |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A0_cavity_air | 0 | Air | 2 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| A0_cavity_air | 0.005 | Air | 2 | 1.00 | 0.0006818 | 0.0008324 | 0.001112 | 0.001159 | -0.0004775 |
| B1_film_tio2 | 0 | TiO2 | 2 | 1.00 | 5.919e-10 | 8.37e-10 | 1.125e-09 | 1.184e-09 | 5.919e-10 |
| B1_film_tio2 | 0.005 | TiO2 | 2 | 1.00 | 0.006597 | 0.006939 | 0.008532 | 0.008748 | -0.006597 |
| B2_film_soc_tio2 | 0 | SOC | 2 | 1.00 | 3.553e-15 | 5.024e-15 | 6.75e-15 | 7.105e-15 | 3.553e-15 |
| B2_film_soc_tio2 | 0.005 | SOC | 2 | 1.00 | 0.01113 | 0.01436 | 0.01929 | 0.0202 | 0.01113 |
| B2_film_soc_tio2 | 0 | TiO2 | 2 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| B2_film_soc_tio2 | 0.005 | TiO2 | 2 | 1.00 | 0.005572 | 0.006377 | 0.008364 | 0.008674 | -0.003102 |
| B3_film_all | 0 | HSQ | 2 | 1.00 | 9.489e-12 | 1.342e-11 | 1.803e-11 | 1.898e-11 | -9.489e-12 |
| B3_film_all | 0.005 | HSQ | 2 | 1.00 | 0.1408 | 0.1413 | 0.1516 | 0.1529 | 0.1408 |
| B3_film_all | 0 | PSS | 2 | 1.00 | 9.617e-12 | 1.36e-11 | 1.827e-11 | 1.923e-11 | 9.617e-12 |
| B3_film_all | 0.005 | PSS | 2 | 1.00 | 0.1176 | 0.1177 | 0.123 | 0.1236 | 0.006011 |
| B3_film_all | 0 | SOC | 2 | 1.00 | 7.283e-13 | 1.03e-12 | 1.384e-12 | 1.457e-12 | 7.283e-13 |
| B3_film_all | 0.005 | SOC | 2 | 1.00 | 0.1432 | 0.1894 | 0.2547 | 0.2671 | -0.1432 |
| B3_film_all | 0 | TiO2 | 2 | 1.00 | 2.327e-13 | 3.291e-13 | 4.421e-13 | 4.654e-13 | -2.327e-13 |
| B3_film_all | 0.005 | TiO2 | 2 | 1.00 | 0.01348 | 0.01766 | 0.02375 | 0.02489 | -0.01142 |

## Identifiability Diagnostics

- `A0_cavity_air` params=['Air'], condition_number=1.000e+00
- `B1_film_tio2` params=['TiO2'], condition_number=1.000e+00
- `B2_film_soc_tio2` params=['SOC', 'TiO2'], condition_number=2.981e+00
- `B3_film_all` params=['HSQ', 'PSS', 'SOC', 'TiO2'], condition_number=1.878e+01

## Interpretation Notes

- Use MAE/P95Abs for practical accuracy; MaxAbs is mainly an outlier-risk indicator.
- High condition number or near +/-1 parameter correlations indicate non-identifiability.
- Film-only results are the main reference for thin-film metrology; the cavity case is a bridge to the existing long-cavity model.
