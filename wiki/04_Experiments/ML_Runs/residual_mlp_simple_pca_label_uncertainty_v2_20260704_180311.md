---
type: experiment
status: draft
created: 2026-07-06
updated: 2026-07-06
sources:
  - ../../../work/03_ml_inverse_modeling/ML try/Residual MLP/train_residual_mlp_simple_pca_label_uncertainty_v2.py
  - ../../../work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311/metrics.json
  - ../../../work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311/summary_report.md
tags:
  - experiment
  - ml
  - residual-mlp
  - label-uncertainty
  - pca
---

# residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311

## 一句话结论

在 `uniform_pm_1nm` 场景下，`more_feature + PCA100` 的测试集 MAE 维持在 `1.441-1.562 nm`，随着标签不确定度从 `1.0 nm` 增大到 `3.0 nm`，落入容差带比例从 `44.53%` 提升到 `87.94%`，但 clean-reference RMSE 仍约为 `1.90-2.05 nm`。

## 设计

- 来源脚本：`../../../work/03_ml_inverse_modeling/ML try/Residual MLP/train_residual_mlp_simple_pca_label_uncertainty_v2.py`
- 数据集：`../../../work/03_ml_inverse_modeling/ML try/Residual MLP/dataset/pca_features/nn_cavity_pca_features_100_20260623_120625.npz`
- 输入策略：不使用 true thickness、film delta、`cavity_true_um`、二阶交互项，仅使用 nominal thickness、若干光谱统计量和 `PC1-PC100`
- 划分口径：`dataset_split_id`，回退策略记录为 `process_within_nominal`
- 数据规模：总计 `800000` 行，train/val/test 分别为 `560000 / 120000 / 120000`
- 训练设置：`MLPRegressor(128, 128, 64)`，`epochs=120`，`batch_size=4096`，`lr=1e-3`，外部 clean val early stopping

## 结果摘要

| 标签不确定度   | test MAE (nm) | MaxAbs_nm | P95Abs (nm) | 容差带内比例 | excess RMSE (nm) |
| -------- | ------------: | --------: | ----------: | -----: | ---------------: |
| `1.0 nm` |         1.446 |    20.821 |       3.784 | 44.53% |            1.263 |
| `1.5 nm` |         1.441 |    24.830 |       3.731 | 61.65% |            1.020 |
| `2.0 nm` |         1.496 |    20.263 |       3.900 | 73.17% |            0.888 |
| `2.5 nm` |         1.562 |    23.730 |       4.028 | 80.65% |            0.761 |
| `3.0 nm` |         1.546 |    21.206 |       3.992 | 87.94% |            0.636 |

## 观察

- 事实：`1.5 nm` 不确定度下的 test MAE 最低，为 `1.441 nm`。
- 事实：随着不确定度增大，`within_label_uncertainty_fraction` 单调上升。
- 事实：clean-reference RMSE 没有同步下降，反而在 `2.5-3.0 nm` 时略高于 `1.0-1.5 nm`。
- 推断：更宽的标签容差带改善了“落在允许误差内”的比例，但没有显著提高对干净潜在标签的回归精度。

## 输出位置

- 运行目录：`../../../work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311/`
- 汇总文件：`summary_report.md`、`metrics.json`、`uncertainty_mae_comparison.csv`
- 图像：`01_clean_reference_mae_rmse.png`、`02_within_uncertainty_band_fraction.png`、`03_more_feature_error_hist.png`

## 待验证问题

- 是否需要把 `label_uncertainty_v2` 的结论上升为“可接受标签误差范围”设计约束，而不仅是单次运行记录？
- `uniform_pm_1nm` 之外，是否还需要补 `gaussian_sigma_1nm_clipped` 或更贴近工艺分布的标签噪声模型？
	- 我觉得应该补充随机噪声以及高斯噪声版本，更重要的是，哪一个才是接近实际情况？
