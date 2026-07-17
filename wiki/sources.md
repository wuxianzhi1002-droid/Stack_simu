# Sources

本页专门记录 `../work/` 中的重要来源路径。Obsidian Vault 不复制大型数据或仿真输出，只记录路径、用途和知识页入口。

## 2026-07-16 动态 StackRT 多档噪声生成 v4

| 路径 | 类型 | 用途 | 状态 |
|---|---|---|---|
| `../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v4.py` | 仿真代码 | 生成 clean/low/medium/high 四档折射率、角度、波长、调制幅值和测量噪声动态 NPZ | 代码与 mock 保存链路已验证；真实 Lumerical 运行待会话恢复 |
| `../work/04_results_and_datasets/dynamic_stackrt_lockin_v4/` | 计划输出目录 | 保存按噪声等级命名的 NPZ、锁相图、动态图和 realization JSON | 尚未生成；不得视为已有正式数据 |

## 当前核心来源

| 路径                                                    | 类型    | 用途                                           | 知识页                                                          |
| ----------------------------------------------------- | ----- | -------------------------------------------- | ------------------------------------------------------------ |
| `../work/01_simulation_models/01_Lumerical_Workflow/` | 仿真与代码 | Lumerical/stackrt 工作流、主脚本、FSP 工程和 stackrt 结果 | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `../work/01_simulation_models/02_Zemax_Workflow/`     | 仿真与代码 | Zemax 自动化、资产安装、调试脚本和最终系统文件                   | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `../work/01_simulation_models/03_MATLAB_Validation/`  | 验证数据  | MATLAB 或跨工具验证数据                              | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `../work/02_analysis_code/stackrt_zemax_compare.py`   | 代码    | stackrt 与 Zemax 对比脚本                         | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `../work/03_ml_inverse_modeling/ML try/`              | 机器学习  | 光谱特征数据集、Residual MLP、CNN、训练结果和模型文件           | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `../work/04_results_and_datasets/results/`            | 输出    | 根目录结果图、CSV、FSP 和对比输出                         | [[04_Experiments/Simulation_Reports/Root_Simulation_Report]] |
| `../work/04_results_and_datasets/0614/`               | 输出    | 2026-06-14 前后历史实验结果                          | [[04_Experiments/Legacy_0614/仿真结果与代码总结]]                     |
| `../work/05_reference_materials/03_Common_Docs/`      | 参考材料  | 原始共同说明、材料表和非 Markdown 附件                     | [[02_Literature/Materials/Common_Material_References]]       |
| `../work/06_environment/04_Environment_Config/`       | 环境说明  | conda/pip 环境配置记录                             | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `../work/99_legacy_misc/新建文件夹/`                       | 遗留代码  | 恢复、FFT、裁剪等历史脚本                               | [[05_CodeNotes/STACK_simu_code_map]]                         |
| `01_Projects/Lumerical_Automation_Summary.md`         | 知识页   | Lumerical 自动化工作总结，已放入 wiki                   | [[01_Projects/Lumerical_Automation_Summary]]                 |

## 2026-07-14 高度调制锁相与联合反演

| 路径 | 类型 | 用途 | 知识页 |
|---|---|---|---|
| `../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v2.py` | 仿真与信号处理代码 | 生成 StackRT 动态光谱，并提取 1f/2f/3f 的 `X/Y/R/phase` 与 `dIdL_1f` | [[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]] |
| `../work/02_analysis_code/tmm_joint_inversion_lockin_v2.py` | 反演代码 | 使用匹配 StackRT 约定的 TMM 执行 I-only、D-only、joint 多起点反演 | [[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]] |
| `../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/dynamic_spectra_20260714_161153.npz` | 原始仿真输出 | 新膜栈 `30/10/40/40 nm`、`A=5 nm` 的最终同步动态光谱与锁相数据 | [[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]] |
| `../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/single_spectrum_compare_tmm_stackrt/` | 前向验证结果 | 验证 StackRT 与独立 TMM 的单光谱数值约定和闭环误差 | [[04_Experiments/Simulation_Reports/Dynamic_StackRT_TMM_Single_Spectrum_20260713]] |
| `../work/04_results_and_datasets/tmm_joint_inversion_lockin_v2_20260714_152135/` | 小幅值反演结果 | 旧膜栈、`A=1 nm` 的理想小信号三组对照基准 | [[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]] |
| `../work/04_results_and_datasets/tmm_joint_inversion_lockin_v2_20260714_165223/` | 最终反演结果 | 新膜栈、`A=5 nm`、幅值口径修正后的 I/D/joint 对照与 multistart 诊断 | [[04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714]] |

## 2026-07-15 无真值起点联合反演 v3

| 路径 | 类型 | 用途 | 知识页 |
|---|---|---|---|
| `../work/02_analysis_code/tmm_joint_inversion_lockin_v3.py` | 反演代码 | 有限幅值 TMM 观测模型、每模式独立差分进化和 32 起点局部精修；真值仅用于拟合后评估 | [[04_Experiments/Simulation_Reports/TMM_Joint_Inversion_Lockin_v3_20260715]] |
| `../work/04_results_and_datasets/tmm_joint_inversion_lockin_v3_20260715_112455/` | 验证结果 | 无真值起点、无先验的 I/D/joint 最终运行，含起点审计和全部 rank | [[04_Experiments/Simulation_Reports/TMM_Joint_Inversion_Lockin_v3_20260715]] |

## 2026-07-13 动态 StackRT 单光谱对比

| 路径 | 类型 | 用途 | 知识页 |
|---|---|---|---|
| `../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v2.py` | 仿真代码 | 生成 1 mm 空气腔正弦调制的 StackRT 动态反射率序列 | [[04_Experiments/Simulation_Reports/Dynamic_StackRT_TMM_Single_Spectrum_20260713]] |
| `../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/dynamic_spectra_20260708_112955.npz` | 原始仿真输出 | 提供 `t=0`、`L=1000 µm` 的 StackRT 首帧光谱；NPZ 仅保存在 work | [[04_Experiments/Simulation_Reports/Dynamic_StackRT_TMM_Single_Spectrum_20260713]] |
| `../work/04_results_and_datasets/dynamic_stackrt_lockin_v2/single_spectrum_compare_tmm_stackrt/` | 对比结果 | 保存 TMM 复现脚本、CSV/NPZ、误差摘要和对比图 | [[04_Experiments/Simulation_Reports/Dynamic_StackRT_TMM_Single_Spectrum_20260713]] |

## 2026-07-06 增量来源

| 路径 | 类型 | 用途 | 知识页 |
|---|---|---|---|
| `../work/03_ml_inverse_modeling/ML try/Residual MLP/train_residual_mlp_simple_pca_label_uncertainty_v2.py` | 代码 | 人工标签不确定度扫描训练入口，比较 `1.0-3.0 nm` 误差带下的 `more_feature + PCA100` 表现 | [[05_CodeNotes/ML_CodeNotes/Residual_MLP_README]] |
| `../work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311/` | 训练输出 | 2026-07-04 标签不确定度扫描结果目录，含 `metrics.json`、汇总表和图像 | [[04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311]] |

## 移动状态

- 主要研究资产已按内容移动到 `../work/01_simulation_models/`、`../work/03_ml_inverse_modeling/`、`../work/04_results_and_datasets/` 等目录。
- `work/` 中的知识型 Markdown 已迁移到 `wiki/`；AppleDouble 元数据已改为 `.appledouble` 扩展名。
- 旧分类空壳目录已集中移动到 `../work/99_legacy_misc/empty_legacy_placeholders/`。

## 后续登记规则

新增重要来源时，记录：

- 路径
- 类型：代码、数据、仿真工程、论文、报告、输出、环境说明
- 生成或获得日期
- 对应知识页
- 可信度或状态：草稿、已验证、过时、待复核

## 2026-07-09 ????

| ?? | ?? | ?? | ??? |
|---|---|---|---|
| `../work/02_analysis_code/tmm_inverse_validation_robust.py` | ?? | ???? TMM ????????????????????????????? | [[04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708]] |
| `../work/02_analysis_code/stackrt_vs_tmm_inverse_validation.py` | ?? | StackRT ???? TMM ????????????? | [[04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708]] |
| `../work/02_analysis_code/stackrt_vs_tmm_inverse_validation_v2.py` | ?? | ? StackRT ? `Cu` ?????? TMM ??????? mismatch ?? | [[04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708]] |
| `../work/04_results_and_datasets/tmm_inverse_validation_20260707_153155/` | ???? | ?? forward model ?? TMM ?????? | [[04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708]] |
| `../work/04_results_and_datasets/tmm_inverse_validation_robust_20260707_203937/` | ???? | ?? realistic ???? robust TMM ???? | [[04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708]] |
| `../work/04_results_and_datasets/stackrt_vs_tmm_inverse_validation_v2_20260708_021808/` | ???? | ?? `Cu` ?? StackRT vs TMM mismatch ???? | [[04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708]] |
| `../work/03_ml_inverse_modeling/ML try/Residual MLP/train_residual_mlp_simple_pca_label_uncertainty_v4.py` | ?? | `?10 nm` ????????????????????? | [[04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v4_20260707_011244]] |
| `../work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v4_20260707_011244/` | ???? | `label_uncertainty_v4` ????????????? | [[04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v4_20260707_011244]] |

## 2026-07-12 ????

| ?? | ?? | ?? | ??? |
|---|---|---|---|
| `../work/03_ml_inverse_modeling/ML try/Residual MLP/train_residual_mlp_simple_pca_label_uncertainty_v3.py` | ?? | ? `1.0-3.0 nm` ????????? `uniform_pm` ??? `gaussian_sigma_nm` ????? `more_feature + PCA100` ??? | [[05_CodeNotes/ML_CodeNotes/Residual_MLP_README]] |
| `../work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v3_20260706_160929/` | ???? | 2026-07-06 ???????????????? `metrics.json`?`summary_report.md`?`uncertainty_mae_comparison.csv` ??? | [[04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v3_20260706_160929]] |

## 2026-07-09 ??????

| ??? | ?? | ????? | ?? |
|---|---|---|---|

## 2026-07-09 悉识科技资料

| 路径 | 类型 | 用途 | 知识页 |
|---|---|---|---|
| `02_Literature/悉识科技资料/Acuitik_Products_v2.6_Chinese.pdf` | 厂商产品手册 | NanoSense/NS-20/NS-Micro/NS-HR/EllipSense 的原理、规格和性能口径 | [[03_Methods/Acuitik_NanoSense_measurement_assessment]] |
| `02_Literature/悉识科技资料/回复清华大学的问询.md` | 厂商问询回复 | 多层膜 `~0.5 nm` 精度适用条件、校准和模型约束说明 | [[03_Methods/Acuitik_NanoSense_measurement_assessment]] |
| `02_Literature/悉识科技资料/案例1.pdf` | 应用案例 | 玻璃基底单层/多层膜测量结果，用于判断复杂 stack 风险 | [[03_Methods/Acuitik_NanoSense_measurement_assessment]] |
| `02_Literature/悉识科技资料/案例2.pdf` | 应用案例 | NS-Micro 对 PI-Si 微区膜厚测量的流程、光斑和结果 | [[03_Methods/Acuitik_NanoSense_measurement_assessment]] |
