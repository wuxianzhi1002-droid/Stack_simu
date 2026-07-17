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
| I | 1000 | 30.0696 | 11.3749 | 38.8969 | 39.7696 | 0.0080079 | 0.817358 | 14792.8 |
| D | 1000 | 28.82 | 16.7446 | 34.6358 | 39.6065 | 0.0103198 | 0.629907 | 20226.2 |
| joint | 1000 | 29.5939 | 14.6372 | 36.2027 | 39.5743 | 0.00881921 | 0.681785 | 18077.3 |

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
  "input_npz": "D:\\激光干涉仪\\simulation\\Lumerical_simulation\\STACK_simu\\work\\04_results_and_datasets\\dynamic_stackrt_lockin_v2\\dynamic_spectra_20260714_161153.npz",
  "amplitude_nm": 5.0,
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
