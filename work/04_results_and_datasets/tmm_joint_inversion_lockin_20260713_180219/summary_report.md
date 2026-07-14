# TMM Joint Inversion With Lock-in dI/dL

## Method

- Input channels: `I(lambda)` from mean dynamic spectra and `dI/dL(lambda)` from `lockin_1f_X / A_um`.
- Forward model: coherent TMM with `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`.
- Fitted parameters: Air in um; HSQ/PSS/SOC/TiO2 in nm.
- Comparison modes: I-only, dI/dL-only, joint.

## Best Fit

| mode | Air um | HSQ nm | PSS nm | SOC nm | TiO2 nm | RMSE I | RMSE dI/dL | cond |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 1000.16 | 51.806 | 20 | 80 | 60 | 0.308416 | 17.4846 | 10991.1 |
| D | 1000.63 | 20 | 1 | 57.1934 | 5 | 0.27186 | 16.1109 | 25261.6 |
| joint | 999.976 | 20 | 1 | 52.3702 | 51.9049 | 0.343419 | 16.863 | 13422.1 |

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
  "input_npz": "D:\\激光干涉仪\\simulation\\Lumerical_simulation\\STACK_simu\\work\\04_results_and_datasets\\dynamic_stackrt_lockin_v2\\dynamic_spectra_20260708_112955.npz",
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
