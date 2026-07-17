# TMM Joint Inversion With Lock-in dI/dL (v3)

## Method

- Input channels: `I(lambda)` from mean dynamic spectra and `dI/dL(lambda)` from `lockin_1f_X / A_um`.
- Forward model: StackRT-matched coherent TMM with `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`.
- Observation model: finite-amplitude sinusoidal modulation, time averaging, and digital 1f demodulation are reproduced inside every residual evaluation.
- Convention: `lambda_phase = 299792458 / (3e8 / lambda_nominal)`, `q = n`, and `-i` matrix off-diagonal terms for `n+i*k`.
- Initialization: differential evolution with a Latin-hypercube population; no truth or nominal vector is inserted as a start.
- Local multistarts: top 32 candidates from each mode's independent global population.
- Prior enabled: `False`; v3 defaults to no prior.
- Selection: converged attempts are ranked before failed attempts, then by robust cost.
- Fitted parameters: Air in um; HSQ/PSS/SOC/TiO2 in nm.
- Comparison modes: I-only, dI/dL-only, joint.

## Best Fit

| mode | Air um | HSQ nm | PSS nm | SOC nm | TiO2 nm | RMSE I | RMSE dI/dL | cond |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 999.925 | 40 | 9.96545 | 30 | 43.3048 | 0.0711882 | 7.79976 | 13631.3 |
| D | 999.917 | 40 | 15.3376 | 30 | 42.1298 | 0.0742072 | 6.32846 | 22957.3 |
| joint | 999.919 | 40 | 13.2624 | 30 | 43.0197 | 0.0717922 | 6.6536 | 18477.9 |

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
  "input_npz": "D:\\激光干涉仪\\simulation\\Lumerical_simulation\\STACK_simu\\work\\04_results_and_datasets\\dynamic_stackrt_lockin_v4\\dynamic_spectra_low_20260716_164051.npz",
  "amplitude_nm": 5.0,
  "wavelength_min_nm": 220.0,
  "wavelength_max_nm": 580.0,
  "stride": 10,
  "multistarts": 32,
  "max_nfev": 300,
  "global_popsize": 16,
  "global_maxiter": 60,
  "global_stride": 50,
  "global_phase_samples": 8,
  "local_phase_samples": 16,
  "random_seed": 20260715,
  "use_prior": false,
  "loss": "soft_l1"
}
```
