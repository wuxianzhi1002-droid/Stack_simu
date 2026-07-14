# Label Uncertainty Residual MLP Summary

## Design

- The source script is not modified.
- Feature policy is inherited from `train_residual_mlp_simple_pca.py`.
- Artificial label noise is applied only to the used train labels.
- Test metrics below are evaluated against the original simulation label as a clean latent reference.
- `within band` and `excess_RMSE_after_uncertainty_nm` report tolerance-band behavior.

## Method Comparison

| run | MAE_nm | RMSE_nm | MaxAbs_nm | P95Abs_nm | within band | excess_RMSE_nm | train_noise_RMSE_nm | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| u10nm__gaussian_sigma_nm__more_feature | 3.177 | 4.073 | 24.936 | 8.038 | 98.15% | 0.421 | 9.998 | 113 |
| u10nm__random_uniform_nm__more_feature | 2.374 | 3.063 | 30.408 | 6.000 | 99.56% | 0.247 | 5.773 | 120 |

## Conclusion

- For more_feature noisy-label runs, test MAE ranged from 2.374 nm at 10 nm to 3.177 nm at 10 nm.
