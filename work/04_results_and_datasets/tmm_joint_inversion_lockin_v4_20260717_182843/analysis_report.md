# V5 StackRT Noise Ablation and Angle-Aware Joint Inversion Analysis

## Scope

- Generator: `main_dynamic_v5.py`, native Lumerical StackRT batch backend.
- Inversion: `tmm_joint_inversion_lockin_v4.py`, joint `I(lambda) + lockin_1f_X/A`.
- Dataset timestamp: `20260717_181902`; inversion timestamp: `20260717_182843`.
- 19 cases, 400 time samples, 20000 wavelengths per NPZ, 19 independent seeds.
- Each level currently has one realization. This is an ablation diagnosis, not a Monte Carlo uncertainty statement.

## Forward Closure

- At true thickness and angle, clean and all angle-only cases give `RMSE(I) ~ 1.8e-12` and `RMSE(dI/dL) ~ 5.7e-10 /um`, with correlations equal to 1.
- Therefore the oblique p-polarized TMM convention matches StackRT for the tested stack.

## Accuracy Results

| case | realized angle deg | wavelength offset nm | film MAE nm | geometric Air error um | phase-length error um | angle error deg | normalized RMSE | bounds |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 0 | 0 | 0.00231079 | 0.0153718 | 9.14388e-07 | 0.0545077 | 2.55257e-05 |  |
| angle_low | 0.00767991 | 0 | 0.000424892 | 0.0028214 | 1.67979e-07 | 0.016903 | 4.74129e-06 |  |
| angle_medium | 0.0215401 | 0 | 0.00109956 | 0.00726245 | 4.29373e-07 | 0.0216767 | 1.20326e-05 |  |
| angle_high | 0.00827241 | 0 | 3.57265e-05 | 0.000246147 | 1.43956e-08 | 0.00249837 | 4.16046e-07 |  |
| wavelength_low | 0 | 0.000425609 | 1.3633 | 0.00613119 | 0.00201364 | 0.0396759 | 0.0337127 |  |
| wavelength_medium | 0 | 0.000995054 | 5.93448 | 0.00118116 | 0.000699989 | 0.0190678 | 0.0624378 | SOC |
| wavelength_high | 0 | 0.00497318 | 5.76923 | 0.0183865 | 0.0187754 | 0.00866935 | 0.404543 | HSQ;SOC |
| material_low | 0 | 0 | 0.0445077 | 0.00213969 | 7.7973e-06 | 0.0202988 | 0.000692048 |  |
| material_medium | 0 | 0 | 0.0815254 | 0.0515783 | 0.000220934 | 0.0996261 | 0.00608326 |  |
| material_high | 0 | 0 | 0.404202 | 0.00669229 | 0.00033073 | 0.0368424 | 0.0087552 |  |
| amplitude_low | 0 | 0 | 0.00255506 | 0.022131 | 2.49781e-09 | 0.0654006 | 0.000518712 |  |
| amplitude_medium | 0 | 0 | 0.0734639 | 0.0380104 | 1.89025e-05 | 0.0856879 | 0.00802598 |  |
| amplitude_high | 0 | 0 | 0.0879034 | 0.000963381 | 2.25607e-05 | 0.0138043 | 0.00954552 |  |
| detector_low | 0 | 0 | 0.00104654 | 0.000604412 | 1.60612e-07 | 0.0108097 | 0.000262731 |  |
| detector_medium | 0 | 0 | 0.0048048 | 0.0387342 | 1.18566e-06 | 0.0865226 | 0.00134741 |  |
| detector_high | 0 | 0 | 0.0657304 | 0.00129105 | 1.91258e-05 | 0.015679 | 0.00682303 |  |
| combined_low | 0.0277043 | 9.98365e-05 | 0.34704 | 0.00898633 | 0.000466308 | 0.0232315 | 0.00859815 |  |
| combined_medium | 0.0667161 | -0.00062702 | 1.75427 | 0.0117769 | 0.00325166 | 0.0113777 | 0.0543383 |  |
| combined_high | 0.0588126 | -0.00526786 | 7.08103 | 0.028505 | 0.0223963 | 0.00930085 | 0.350371 | HSQ |

## Main Findings

1. **Wavelength-axis error dominates.** An actual offset of `0.0004256 nm` already produces `1.36 nm` film MAE. At `0.000995 nm` and `0.004973 nm`, film MAE is about `5.93 nm` and `5.77 nm`; the latter saturates against thickness bounds, so MAE is no longer monotonic with offset.
2. **Angle-only data are forward-model closed.** Film MAE stays below `0.0011 nm`. However Air and angle remain strongly coupled: multistart Air-angle correlation is approximately `0.95-0.999`, while phase-equivalent Air length is recovered much more accurately than geometric Air length.
3. **Material uncertainty is the second important systematic term.** Film MAE increases from `0.0445 nm` to `0.404 nm` across the sampled realizations. Nominal-n/k inversion absorbs material optical-thickness error into geometric film thickness.
4. **Amplitude calibration and detector noise are comparatively benign at current levels.** Their high-level film MAE values are `0.0879 nm` and `0.0657 nm`, respectively.
5. **Combined errors are nonlinear.** Film MAE grows from `0.347 nm` to `1.75 nm` and `7.08 nm`; combined-high hits the HSQ bound. The dominant contributor is wavelength offset, followed by material mismatch.

## Optimizer Robustness

- All local fits report success; no failed local result was selected.
- All differential-evolution runs reached `maxiter=50` before satisfying the strict tolerance. This indicates incomplete global convergence, but not identical behavior across cases.
- No truth vector was inserted. Exact truth start count is zero for every case; minimum scaled start-to-truth distance ranges from about `0.055` to `0.881`.
- Only `wavelength_medium`, `wavelength_high`, and `combined_high` hit thickness bounds. These are model-mismatch warnings, not evidence that more DE iterations alone will recover truth.
- Jacobian condition numbers range from about `1.87e4` to `3.65e4`. Adding angle increases the Air/angle ridge; geometric Air and angle should not both be interpreted as independently measured without external angle information.

## Noise-Specific Optimization Priorities

1. **Wavelength:** add a bounded `delta_lambda_nm` nuisance parameter or calibrate the spectrometer axis before thickness inversion. This is the highest-priority model change.
2. **Angle:** retain the `0-0.1 deg` bound, but use an external angle prior, known-angle calibration, or a second polarization/angle measurement. Free angle mainly protects phase-length fitting; it does not uniquely identify geometric Air length.
3. **Material n/k:** introduce compact, prior-constrained material corrections or use ellipsometry values. Do not free every n/k coefficient simultaneously with all thicknesses.
4. **Amplitude:** fit a single lock-in scale factor or use independently calibrated actual modulation amplitude.
5. **Detector:** estimate channel variance from repeats and weight I and dI/dL by measured noise; robust loss is useful for clipping but cannot correct wavelength or n/k bias.
6. **Global search:** after nuisance-model correction, increase DE iterations or use convergence-based restarts. Under the current misspecified wavelength model, a larger search budget would optimize the wrong objective more precisely.

## Files

- `batch_fit_results.csv`: joint-fit metrics and parameter errors.
- `noise_realization_summary.csv`: realized nuisance values joined with inversion metrics.
- `optimization_robustness_summary.csv`: DE/local/multistart/conditioning diagnostics.
- `noise_accuracy_overview.png`: level-based comparison.
- `noise_factor_heatmaps.png`: factor/level heatmaps.
- `noise_sensitivity_actual_realization.png`: accuracy versus actual sampled nuisance magnitude.

## Applicability

- Conclusions apply to this 1 mm cavity, 220-580 nm fitting range, stride 10, current layer stack, and one realization per factor/level.
- Formal uncertainty claims require repeated realizations per factor/level and hardware-calibrated noise distributions.
