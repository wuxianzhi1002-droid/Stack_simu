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
| clean__base_scalar | 11.905 | 15.370 | 61.743 | 31.400 | 5.68% | 14.609 | 0.000 | 40 |
| clean__more_feature | 5.523 | 7.037 | 39.885 | 13.736 | 11.98% | 6.279 | 0.000 | 40 |
| uniform_pm_1nm__base_scalar | 11.906 | 15.372 | 61.786 | 31.425 | 5.70% | 14.610 | 0.578 | 40 |
| uniform_pm_1nm__more_feature | 5.521 | 7.038 | 39.811 | 13.716 | 11.92% | 6.281 | 0.578 | 40 |

## Conclusion

- uniform_pm_1nm more_feature changed clean-reference test RMSE by +0.001 nm versus clean-label training.
