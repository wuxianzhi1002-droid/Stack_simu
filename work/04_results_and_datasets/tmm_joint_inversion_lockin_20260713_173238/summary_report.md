# TMM Joint Inversion With Lock-in dI/dL

## Method

- Input channels: `I(lambda)` from mean dynamic spectra and `dI/dL(lambda)` from `lockin_1f_X / A_um`.
- Forward model: coherent TMM with `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`.
- Fitted parameters: Air in um; HSQ/PSS/SOC/TiO2 in nm.
- Comparison modes: I-only, dI/dL-only, joint.

## Best Fit

| mode | Air um | HSQ nm | PSS nm | SOC nm | TiO2 nm | RMSE I | RMSE dI/dL | cond |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 1000.01 | 40.0224 | 1 | 55.1929 | 5 | 0.345062 | 17.6894 | 8353.52 |
| D | 1000.27 | 60 | 19.9998 | 65.3777 | 38.6331 | 0.349841 | 15.1447 | 11546.3 |
| joint | 999.65 | 41.2872 | 4.47684 | 80 | 59.249 | 0.345695 | 15.7959 | 10084.7 |

## Files

- Fit results: `fit_results.csv`
- Multistart results: `multistart_results.csv`
- best_fit_spectrum: `best_fit_spectrum.png`
- best_fit_dIdL: `best_fit_dIdL.png`
- joint_residual: `joint_residual.png`
- jacobian_singular_values: `jacobian_singular_values.png`

## Config

```json
{
  "input_npz": "dynamic_spectra_20260708_112955.npz",
  "amplitude_nm": 1.0,
  "wavelength_min_nm": 220.0,
  "wavelength_max_nm": 580.0,
  "stride": 100,
  "derivative_step_um": 0.001,
  "multistarts": 2,
  "max_nfev": 30,
  "random_seed": 20260713,
  "use_prior": true,
  "loss": "soft_l1"
}
```
