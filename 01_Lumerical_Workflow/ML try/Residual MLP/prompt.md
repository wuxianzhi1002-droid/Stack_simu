请你继续改进 01_Lumerical_Workflow/ML try 里面的 Residual MLP 训练代码。

这次只做两个版本：

版本 1：Scalar baseline
版本 2：Scalar + 二阶特征

============================================================
一、当前问题
============================================================

当前 README 里写的是：

Residual MLP + true film thickness + 二阶交互特征

并且输入使用：

    L_fft_um
    H_peak
    PSS_true_nm
    HSQ_true_nm
    SOC_true_nm
    TiO2_true_nm

这个设计有一个关键问题：

真实实验中通常无法知道 true film thickness。
数据集中虽然有：

    film_nominal_nm
    film_delta_nm
    film_true_nm

但最终可部署模型只能使用：

    film_nominal_nm

不能使用：

    film_true_nm
    film_delta_nm

除非有额外椭偏仪、XRR 或其他独立膜厚计量结果。

因此请把主训练流程改成：

输入：
    L_fft_um
    H_peak
    PSS_nominal_nm
    HSQ_nominal_nm
    SOC_nominal_nm
    TiO2_nominal_nm

输出：
    delta_L_nm = (L_true_um - L_fft_um) * 1000

最终还原：
    cavity_pred_um = L_fft_um + delta_L_pred_nm / 1000

不要直接预测 cavity_true_um。
仍然预测 FFT 残差 delta_L_nm。

============================================================
二、数据集结构理解
============================================================

当前数据集的生成逻辑是：

1. 先取 100 个名义膜厚组合；
2. 每个名义膜厚组合下面生成 20 个 process；
3. 这 20 个 process 是同一个 nominal thickness 下的 ±10 nm 随机扰动；
4. 因此：

    d_true = d_nominal + delta_d

其中：

    delta_d ∈ [-10 nm, +10 nm]

真实部署时只知道：

    d_nominal

不知道：

    delta_d
    d_true

所以模型主输入只能取名义膜厚。

============================================================
三、需要实现的两个版本
============================================================

请在 train_residual_mlp.py 中实现两个可选实验模式。

------------------------------------------------------------
版本 1：scalar_baseline
------------------------------------------------------------

输入特征：

    L_fft_um
    H_peak
    PSS_nominal_nm
    HSQ_nominal_nm
    SOC_nominal_nm
    TiO2_nominal_nm

输出标签：

    delta_L_nm

模型：

    StandardScaler
    MLPRegressor

或者如果当前代码已经使用 PyTorch MLP，也可以继续使用 PyTorch。
但要保持两个版本的训练逻辑一致，方便对比。

------------------------------------------------------------
版本 2：scalar_selected_quadratic
------------------------------------------------------------

输入特征包括版本 1 的全部特征，另外加入手选二阶交互项。

不要默认无脑加入所有二阶项。

优先加入具有物理意义的交互项：

    L_fft_um * PSS_nominal_nm
    L_fft_um * HSQ_nominal_nm
    L_fft_um * SOC_nominal_nm
    L_fft_um * TiO2_nominal_nm

可选加入：

    H_peak * PSS_nominal_nm
    H_peak * HSQ_nominal_nm
    H_peak * SOC_nominal_nm
    H_peak * TiO2_nominal_nm

先不要加入所有膜厚之间的两两乘积，例如：

    HSQ_nominal_nm * SOC_nominal_nm

除非在 ablation 里单独验证有效。

原因：
如果输入之间本身存在相关性，自动生成大量二阶特征可能造成冗余、多重共线性和过拟合。
二阶特征只能作为消融实验，不应默认认为一定有效。

============================================================
四、可选版本 2b：all_quadratic 仅作为消融实验
============================================================

可以额外实现一个可选模式：

    scalar_all_quadratic

这个模式使用 sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)

输入原始特征仍然只能是：

    L_fft_um
    H_peak
    PSS_nominal_nm
    HSQ_nominal_nm
    SOC_nominal_nm
    TiO2_nominal_nm

然后自动生成全部二阶项。

但请注意：

1. all_quadratic 只能作为 ablation；
2. 不要把它作为默认最终方法；
3. 只有当它在未见 process 的 test set 上明显优于 selected_quadratic，才认为它有价值；
4. 需要输出生成的 feature_names，保存到 feature_names.json；
5. 需要检查特征相关性矩阵，保存 high_correlation_features.csv。

============================================================
五、特征标准化要求
============================================================

所有特征，包括二阶特征，都必须标准化。

推荐流程：

    raw_features
    -> generate selected quadratic or all quadratic
    -> StandardScaler.fit(train)
    -> transform train/val/test
    -> MLP

注意：
StandardScaler 只能在 train set 上 fit，不能用全数据 fit。

不要先用全数据 fit scaler，否则会数据泄漏。

============================================================
六、数据划分要求
============================================================

必须按 process_id 划分，不要按样本随机划分。

因为同一个 process 下有多个腔长点，如果随机按样本划分，会导致训练集和测试集共享同一个真实膜厚扰动，测试结果会过于乐观。

请实现两种 split 策略：

------------------------------------------------------------
split_strategy = "process_within_nominal"
------------------------------------------------------------

这是主策略。

每个 nominal thickness group 下有 20 个 process。
对每个 nominal group 内部的 20 个 process 按比例划分：

    train / val / test

这样保证：

1. 同一个 process 不会同时出现在 train/val/test；
2. 同一个 nominal thickness 可以同时出现在 train/val/test；
3. 测试的是：已知名义工艺下，对未见过 ±10 nm 扰动 process 的泛化能力。

这是当前最重要的评价方式。

------------------------------------------------------------
split_strategy = "nominal_holdout"
------------------------------------------------------------

这是更严格的可选策略。

按 nominal thickness group 整体划分：

    train nominal groups
    val nominal groups
    test nominal groups

同一个 nominal group 下面的 20 个 process 必须整体进入同一个 split。

这测试的是：对完全没见过的名义膜厚组合的泛化能力。

这个结果可能明显更差，但请保留它作为可选实验。

============================================================
七、数据字段选择要求
============================================================

请自动兼容以下字段名。

如果数据集中有扁平字段：

    PSS_nominal_nm
    HSQ_nominal_nm
    SOC_nominal_nm
    TiO2_nominal_nm

就直接读取。

如果数据集中只有矩阵字段：

    film_nominal_nm
    layer_names

则根据 layer_names 找到对应列，生成：

    PSS_nominal_nm
    HSQ_nominal_nm
    SOC_nominal_nm
    TiO2_nominal_nm

标签字段：

优先使用：

    delta_L_nm

如果没有，则用：

    delta_L_nm = (cavity_true_um - L_fft_um) * 1000

有效样本筛选：

    valid_mask = finite(L_fft_um)
                 & finite(H_peak)
                 & finite(delta_L_nm)
                 & finite(all input features)

不要把 L_fft 为 nan 的样本用于训练。

============================================================
八、禁止事项
============================================================

主实验中禁止使用以下字段作为输入：

    film_true_nm
    film_delta_nm
    PSS_true_nm
    HSQ_true_nm
    SOC_true_nm
    TiO2_true_nm
    PSS_delta_nm
    HSQ_delta_nm
    SOC_delta_nm
    TiO2_delta_nm
    cavity_true_um

其中 cavity_true_um 只能用于生成标签或评价，不能作为输入。

可以额外做一个 oracle 对照，但必须单独命名：

    scalar_oracle_true_thickness

并在报告里明确说明：

    This is not deployable because true film thickness is unavailable in real measurement.

默认不要开启 oracle。

============================================================
九、训练模型要求
============================================================

MLP 可以使用 sklearn 的 MLPRegressor，建议配置：

    hidden_layer_sizes = (128, 128, 64)
    activation = "relu"
    alpha = 1e-4
    learning_rate_init = 1e-3
    max_iter 或 epochs 根据当前代码设置
    early_stopping = True

如果当前代码用 PyTorch，则使用类似结构：

    input_dim -> 128 -> 128 -> 64 -> 1
    ReLU
    weight_decay = 1e-4
    Adam
    early stopping on val_loss

请统一随机种子。

============================================================
十、评价指标
============================================================

每个实验版本都要输出 train / val / test 指标：

    delta_MAE_nm
    delta_RMSE_nm
    delta_MaxAbs_nm
    cavity_MAE_nm
    cavity_RMSE_nm
    cavity_MaxAbs_nm
    R2_delta

由于：

    cavity_pred_um = L_fft_um + delta_L_pred_nm / 1000

所以 delta_L_nm 的误差和最终 cavity error 的 nm 误差等价。
但仍然请同时输出 delta 和 cavity 两种写法，方便阅读。

同时输出 baseline：

1. raw_fft_baseline:
       cavity_pred_um = L_fft_um

2. mean_residual_baseline:
       delta_L_pred_nm = mean(delta_L_nm on train)

3. scalar_baseline:
       版本 1

4. scalar_selected_quadratic:
       版本 2

5. scalar_all_quadratic:
       可选 ablation

重点比较 test set 的 cavity_MAE_nm 和 cavity_RMSE_nm。

============================================================
十一、关于二阶特征有效性的判断
============================================================

请不要只看 train 指标。

二阶特征是否有效，只看：

    test cavity_MAE_nm
    test cavity_RMSE_nm
    test MaxAbs_nm

判断规则：

如果：

    selected_quadratic 的 test_RMSE 明显小于 scalar_baseline

并且：

    val/test 差距没有显著扩大

则认为手选二阶特征有效。

如果：

    all_quadratic 训练误差下降，但 test 误差不降或变差

则说明自动二阶特征过拟合或冗余，不应采用。

请把这个判断写进 summary_report.md。

============================================================
十二、相关性检查
============================================================

请新增特征相关性分析：

1. 对 train set 的最终输入特征计算 Pearson correlation matrix；
2. 找出 abs(corr) > 0.98 的特征对；
3. 保存：

    feature_correlation_matrix.csv
    high_correlation_feature_pairs.csv

如果二阶特征和原始特征强相关，在报告中提示：

    High correlation detected. Quadratic features may be redundant.

但不要直接删除，先通过 ablation 结果判断。

============================================================
十三、输出目录
============================================================

所有训练输出仍然放在：

    01_Lumerical_Workflow/ML try/Residual MLP/

每次运行创建：

    Residual MLP/residual_mlp_compare_YYYYMMDD_HHMMSS/

输出文件包括：

    metrics.json
    summary_report.md
    feature_names_scalar_baseline.json
    feature_names_selected_quadratic.json
    feature_names_all_quadratic.json
    high_correlation_feature_pairs.csv
    feature_correlation_matrix.csv
    test_predictions_scalar_baseline.csv
    test_predictions_selected_quadratic.csv
    test_predictions_all_quadratic.csv
    residual_mlp_scalar_baseline.joblib
    residual_mlp_selected_quadratic.joblib
    residual_mlp_all_quadratic.joblib

图片输出：

    01_test_pred_vs_true_delta.png
    02_test_error_hist.png
    03_test_error_vs_L_fft.png
    04_test_error_vs_nominal_thickness.png
    05_method_comparison_bar.png

其中 method_comparison_bar 需要比较：

    raw_fft_baseline
    mean_residual_baseline
    scalar_baseline
    scalar_selected_quadratic
    scalar_all_quadratic，可选

============================================================
十四、命令行参数
============================================================

请支持以下参数：

    --dataset path/to/nn_cavity_dataset.npz
    --epochs 120
    --max-train-rows
    --max-val-rows
    --max-test-rows
    --split-strategy process_within_nominal
    --enable-all-quadratic
    --enable-oracle false
    --random-seed 20260613

默认：

    --split-strategy process_within_nominal
    --enable-all-quadratic false
    --enable-oracle false

快速测试命令：

    python train_residual_mlp.py --max-train-rows 200000 --max-val-rows 50000 --max-test-rows 50000 --epochs 60

完整训练命令：

    python train_residual_mlp.py --epochs 120

如果数据集在 ML try 目录下，请默认自动寻找最新的：

    nn_cavity_dataset_*/nn_cavity_dataset_*.npz

============================================================
十五、最终报告需要回答的问题
============================================================

请在 summary_report.md 中明确回答：

1. 主模型是否使用了 true film thickness？
   正确答案应为：没有，主模型只使用 nominal thickness。

2. 版本 1 的 test 误差是多少？

3. 版本 2 的 test 误差是多少？

4. 二阶特征是否改善了未见 process 的测试误差？

5. 是否存在高相关特征对？

6. 如果 all_quadratic 开启，它相比 selected_quadratic 是否真的更好？

7. 当前结果是否说明仅靠 scalar features 就能补偿 ±10 nm 膜厚扰动？
   如果不能，请在报告中提示：
       后续需要引入光谱 I(lambda) 或角度/偏振信息。

============================================================
十六、核心结论
============================================================

请把代码和 README 的表述从：

    Residual MLP + true film thickness + 二阶交互特征

改成：

    Residual MLP + nominal film thickness + optional quadratic interaction features

也就是：

    真实膜厚只用于仿真标签和 oracle 对照；
    名义膜厚才是可部署模型输入；
    二阶特征需要通过 ablation 验证，不能默认认为有效。