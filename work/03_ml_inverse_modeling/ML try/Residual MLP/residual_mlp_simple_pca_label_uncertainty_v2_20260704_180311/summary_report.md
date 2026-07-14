# Label Uncertainty Residual MLP Summary

## Design

- The source script is not modified.
- Feature policy is inherited from `train_residual_mlp_simple_pca.py`.
- Artificial label noise is applied only to the used train labels.
- Test metrics below are evaluated against the original simulation label as a clean latent reference.
- `within band` and `excess_RMSE_after_uncertainty_nm` report tolerance-band behavior.

## Method Comparison

| run                                  | MAE_nm | RMSE_nm | MaxAbs_nm | P95Abs_nm | within band | excess_RMSE_nm | train_noise_RMSE_nm | epochs |
| ------------------------------------ | -----: | ------: | --------: | --------: | ----------: | -------------: | ------------------: | -----: |
| u1nm__uniform_pm_1nm__more_feature   |  1.446 |   1.908 |    20.821 |     3.784 |      44.53% |          1.263 |               0.578 |    119 |
| u1p5nm__uniform_pm_1nm__more_feature |  1.441 |   1.904 |    24.830 |     3.731 |      61.65% |          1.020 |               0.866 |    119 |
| u2nm__uniform_pm_1nm__more_feature   |  1.496 |   1.979 |    20.263 |     3.900 |      73.17% |          0.888 |               1.155 |    120 |
| u2p5nm__uniform_pm_1nm__more_feature |  1.562 |   2.049 |    23.730 |     4.028 |      80.65% |          0.761 |               1.443 |    120 |
| u3nm__uniform_pm_1nm__more_feature   |  1.546 |   2.037 |    21.206 |     3.992 |      87.94% |          0.636 |               1.730 |    119 |

## Conclusion

- For more_feature noisy-label runs, test MAE ranged from 1.441 nm at 1.5 nm to 1.562 nm at 2.5 nm.
