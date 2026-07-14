# TMM Joint Inversion With Lock-in dI/dL (v2)

## Method

- Input channels: `I(lambda)` from mean dynamic spectra and `dI/dL(lambda)` from `lockin_1f_X / A_um`.
- Forward model: StackRT-matched coherent TMM with `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`.
- Convention: `lambda_phase = 299792458 / (3e8 / lambda_nominal)`, `q = n`, and `-i` matrix off-diagonal terms for `n+i*k`.
- Multistart selection: converged attempts are ranked before failed attempts, then by robust cost.
- Fitted parameters: Air in um; HSQ/PSS/SOC/TiO2 in nm.
- Comparison modes: I-only, dI/dL-only, joint.

## Best Fit

| mode | Air um | HSQ nm | PSS nm | SOC nm | TiO2 nm | RMSE I | RMSE dI/dL | cond |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 1000 | 40.0332 | 5.02592 | 49.9752 | 19.9866 | 0.000285004 | 0.0161808 | 14764.9 |
| D | 1000 | 39.979 | 4.93631 | 50.0487 | 20.0134 | 0.000351412 | 0.00376985 | 20461 |
| joint | 1000 | 39.9912 | 4.95852 | 50.0311 | 20.0071 | 0.000326251 | 0.00567504 | 21981.9 |

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
  "stride": 10,
  "derivative_step_um": 0.001,
  "multistarts": 8,
  "max_nfev": 160,
  "random_seed": 20260713,
  "use_prior": true,
  "loss": "soft_l1"
}
```
