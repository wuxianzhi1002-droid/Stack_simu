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
| u1nm__gaussian_sigma_nm__base_scalar | 11.779 | 15.264 | 63.151 | 31.373 | 5.83% | 14.506 | 1.000 | 22 |
| u1nm__gaussian_sigma_nm__more_feature | 1.466 | 1.930 | 24.989 | 3.835 | 43.92% | 1.280 | 1.000 | 119 |
| u1p5nm__gaussian_sigma_nm__base_scalar | 11.802 | 15.294 | 63.285 | 31.443 | 8.69% | 14.166 | 1.502 | 23 |
| u1p5nm__gaussian_sigma_nm__more_feature | 1.561 | 2.051 | 25.058 | 4.040 | 57.83% | 1.139 | 1.502 | 118 |
| u2nm__gaussian_sigma_nm__base_scalar | 11.789 | 15.285 | 61.898 | 31.427 | 11.61% | 13.796 | 2.004 | 36 |
| u2nm__gaussian_sigma_nm__more_feature | 1.571 | 2.067 | 27.258 | 4.062 | 70.92% | 0.949 | 2.004 | 120 |
| u2p5nm__gaussian_sigma_nm__base_scalar | 11.781 | 15.288 | 62.743 | 31.467 | 14.45% | 13.445 | 2.500 | 46 |
| u2p5nm__gaussian_sigma_nm__more_feature | 1.729 | 2.278 | 22.352 | 4.467 | 76.67% | 0.949 | 2.500 | 119 |
| u3nm__gaussian_sigma_nm__base_scalar | 11.773 | 15.270 | 63.320 | 31.384 | 17.34% | 13.078 | 3.001 | 29 |
| u3nm__gaussian_sigma_nm__more_feature | 1.797 | 2.346 | 37.486 | 4.621 | 82.38% | 0.818 | 3.001 | 118 |

## Conclusion

- For more_feature noisy-label runs, test MAE ranged from 1.466 nm at 1 nm to 1.797 nm at 3 nm.
