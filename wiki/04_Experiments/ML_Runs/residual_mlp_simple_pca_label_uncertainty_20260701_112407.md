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
| clean__more_feature | 75.689 | 87.278 | 232.711 | 145.026 | 0.50% | 86.412 | 0.000 | 2 |
| uniform_pm_1nm__more_feature | 75.696 | 87.282 | 232.676 | 144.931 | 0.50% | 86.416 | 0.574 | 2 |

## Conclusion

- uniform_pm_1nm more_feature changed clean-reference test RMSE by +0.004 nm versus clean-label training.
