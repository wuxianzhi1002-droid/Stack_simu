# Residual MLP Summary Report

## 核心回答

1. 主模型是否使用了 true film thickness？

   没有。主模型只使用 nominal thickness；true film thickness 只允许作为可选 oracle 对照。

2. 版本 1 scalar_baseline 的 test 误差是多少？

   MAE=20.374 nm, RMSE=28.318 nm, MaxAbs=142.573 nm, R2=0.89281

3. 版本 2 scalar_selected_quadratic 的 test 误差是多少？

   MAE=20.157 nm, RMSE=28.070 nm, MaxAbs=142.851 nm, R2=0.89468

4. 二阶特征是否改善了未见 process 的测试误差？

   是，selected_quadratic 的 test RMSE 低于 scalar_baseline。

5. 是否存在高相关特征对？

   存在 4 对 abs(corr)>0.98 的特征。High correlation detected. Quadratic features may be redundant.

6. 如果 all_quadratic 开启，它相比 selected_quadratic 是否真的更好？

   本次未开启 all_quadratic；它只作为消融实验，默认不启用。

7. 当前结果是否说明仅靠 scalar features 就能补偿 ±10 nm 膜厚扰动？

   当前结果不足以说明仅靠 scalar features 就能补偿 ±10 nm 膜厚扰动。后续需要引入光谱 I(lambda) 或角度/偏振信息。

## Method Comparison

| method | test cavity_MAE_nm | test cavity_RMSE_nm | test cavity_MaxAbs_nm | R2_delta |
|---|---:|---:|---:|---:|
| raw_fft_baseline | 1023.486 | 1027.134 | 1269.341 | -140.02070 |
| mean_residual_baseline | 73.118 | 86.530 | 243.363 | -0.00083 |
| scalar_baseline | 20.374 | 28.318 | 142.573 | 0.89281 |
| scalar_selected_quadratic | 20.157 | 28.070 | 142.851 | 0.89468 |
| scalar_oracle_true_thickness | 4.877 | 5.733 | 19.893 | 0.99561 |

## Notes

- 主实验输入：L_fft_um, H_peak, PSS/HSQ/SOC/TiO2_nominal_nm。
- 主实验禁止使用 film_true_nm、film_delta_nm、cavity_true_um 作为输入。
- cavity_true_um 只用于生成标签或评价。
- selected_quadratic 默认只加入 L_fft_um 与各层 nominal thickness 的交互项。

## Oracle Warning

scalar_oracle_true_thickness 使用 true film thickness。This is not deployable because true film thickness is unavailable in real measurement.
