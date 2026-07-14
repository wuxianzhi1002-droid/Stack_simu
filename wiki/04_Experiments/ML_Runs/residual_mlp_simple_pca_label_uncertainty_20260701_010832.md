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
| clean__more_feature | 4.359 | 5.613 | 40.501 | 11.053 | 15.11% | 4.872 | 0.000 | 120 |
| uniform_pm_1nm__more_feature | 4.374 | 5.620 | 42.241 | 11.052 | 15.15% | 4.877 | 0.578 | 120 |

## Conclusion

- uniform_pm_1nm more_feature changed clean-reference test RMSE by +0.008 nm versus clean-label training.
