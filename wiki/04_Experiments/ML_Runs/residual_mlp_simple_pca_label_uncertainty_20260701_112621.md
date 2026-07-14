# Label Uncertainty Residual MLP Summary

## Design

- The source script is not modified.
- Feature policy is inherited from `train_residual_mlp_simple_pca.py`.
- Artificial label noise is applied only to the used train labels.
- Test metrics below are evaluated against the original simulation label as a clean latent reference.
- `within +/-1 nm` and `excess_RMSE_after_uncertainty_nm` report tolerance-band behavior.

## Method Comparison

| run | MAE_nm | RMSE_nm | MaxAbs_nm | P95Abs_nm | within +/-1 nm | excess_RMSE_nm | train_noise_RMSE_nm | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean__base_scalar | 11.766 | 15.255 | 62.530 | 31.342 | 5.84% | 14.497 | 0.000 | 49 |
| clean__more_feature | 1.409 | 1.872 | 26.781 | 3.683 | 45.57% | 1.235 | 0.000 | 120 |
| uniform_pm_1nm__base_scalar | 11.786 | 15.280 | 62.814 | 31.425 | 5.83% | 14.521 | 0.578 | 40 |
| uniform_pm_1nm__more_feature | 1.444 | 1.909 | 19.135 | 3.784 | 44.69% | 1.265 | 0.578 | 120 |

## Conclusion

- uniform_pm_1nm more_feature changed clean-reference test RMSE by +0.037 nm versus clean-label training.
