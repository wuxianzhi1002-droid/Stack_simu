# Log

## 2026-07-13

- 从 `dynamic_spectra_20260708_112955.npz` 选取 `t=0`、空气腔长 `1000 µm` 的 StackRT 首帧光谱，与保持相同膜栈、材料参数、偏振和入射角的独立 TMM 做逐点对比。
- 新增 `wiki/04_Experiments/Simulation_Reports/Dynamic_StackRT_TMM_Single_Spectrum_20260713.md`，记录频率轴、复折射率符号约定、误差指标、适用范围和后续修正建议。
- 对齐 `f=3e8/lambda_nominal` 的实际频率及 StackRT 的 `n+ik` 衰减约定后，得到 `MAE=7.06e-13`、`RMSE=1.44e-12`、最大绝对误差 `1.29e-11`，确认该理想首帧的 StackRT 与 TMM 前向模型一致。
- 更新 `wiki/sources.md` 和 `wiki/00_Knowledge_Flow/D_输出成果索引.md`，加入本次动态光谱对比入口；原始 NPZ 继续留在 `work/` 且不纳入 Git。


## 2026-07-12

- ?? `work/` ? `wiki/`?????????????????????????? `../work/...` ?????
- ?? `work/03_ml_inverse_modeling/ML try/Residual MLP/` ????????????????? `wiki/04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v3_20260706_160929.md`??? `1.0-3.0 nm` `gaussian_sigma_nm` ??? `more_feature + PCA100` ??????
- ?? `wiki/sources.md`?`wiki/00_Knowledge_Flow/D_??????.md` ? `wiki/05_CodeNotes/ML_CodeNotes/Residual_MLP_README.md`?? `label_uncertainty_v3` ????? `v2 -> v3 -> v4` ?????
- ?? `wiki/06_Issues/open_questions.md`??? `uniform_pm` ? `gaussian_sigma_nm` ??????? RMSE ??????????

- ???????
  - `label_uncertainty_v3` ? `label_uncertainty_v2` ?????????????????? train-noise RMSE?????????
  - `gaussian_sigma_nm`?`gaussian_sigma_nm_clipped` ?????????????????????????????????

## 2026-07-09

- 新增 `wiki/03_Methods/Acuitik_NanoSense_measurement_assessment.md`，整理悉识 NanoSense/NS-20/NS-Micro 的反射式膜厚测量原理、性能口径、应用案例证据，并结合当前多层膜与 FP 腔长解算需求给出适配判断。
- ?? `work/` ? `wiki/` ??????? `wiki/04_Experiments/Simulation_Reports/TMM_StackRT_Inverse_Validation_20260707_20260708.md`??? 2026-07-07 ? 2026-07-08 ? TMM ???robust ??? StackRT vs TMM mismatch ???
- ?? `wiki/04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v4_20260707_011244.md`??? `?10 nm` ????????????/???????
- ?? `wiki/sources.md`?`wiki/00_Knowledge_Flow/D_??????.md`?`wiki/05_CodeNotes/ML_CodeNotes/Residual_MLP_README.md` ? `wiki/05_CodeNotes/STACK_simu_code_map.md`?????????????????????
- ??/?? 3 ??????????
  - `wiki/01_Projects/Thesis_Plan/??????????.md`
  - `wiki/06_Issues/GPT_Reviews/??????.md`
  - `wiki/sources.md` ?? `Test_repo` ????????????
- ?? `wiki/06_Issues/open_questions.md`??? `label_uncertainty_v4` ?????`A0_cavity_air` ?????????????????

- ???????
  - `label_uncertainty_v4` ? `?10 nm` ????????????????????????????
  - `A0_cavity_air` ? realistic ??????????????????????????


﻿# Log

## 2026-07-08

- 新增 `wiki/03_Methods/Signal_Processing/height_modulated_lockin_observability.md`，将“顶面高度调制 + 数字锁相 + `I(lambda)` 与 `dI/dz(lambda)` 联合反演”的方法沉淀为可复用流程。
- 在方法页中补充关键边界条件：单余弦 toy model 只能说明锁相导数提取，不能证明 `z` 与 `d` 可分；真实 HSQ/SOC/Hard Mask/Si 研究需使用带参考相位或外部 OPD 的 TMM/干涉 forward model。
- 更新 `wiki/00_Knowledge_Flow/C_方法与Skill库.md`，加入该方法页入口。

## 2026-07-06

- 巡检 `wiki/` 中的本地图片链接与显式 `work/...` 路径引用，当前未发现直接失效的本地图片或明确失效的 `work` 路径。
- 检查 `work/` 中新增说明性来源，确认 `Residual MLP` 新增标签不确定度扫描脚本与结果目录尚未回流到 `wiki/`：
  - `work/03_ml_inverse_modeling/ML try/Residual MLP/train_residual_mlp_simple_pca_label_uncertainty_v2.py`
  - `work/03_ml_inverse_modeling/ML try/Residual MLP/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311/`
- 新增 `wiki/04_Experiments/ML_Runs/residual_mlp_simple_pca_label_uncertainty_v2_20260704_180311.md`，整理 `1.0-3.0 nm` 标签不确定度扫描的配置、指标和结论。
- 更新 `wiki/sources.md`，登记本轮新增的 ML 训练入口脚本与运行结果目录。
- 更新 `wiki/00_Knowledge_Flow/D_输出成果索引.md` 与 `wiki/05_CodeNotes/ML_CodeNotes/Residual_MLP_README.md`，补上 `label_uncertainty_v2` 的知识入口。
- 更新 `wiki/06_Issues/open_questions.md`，补充需要后续人工确认的约束化问题与仓库根目录孤立 Markdown 占位文件问题。

- 需要人工确认：
  - 是否将 `label_uncertainty_v2` 的容差带指标上升为正式设计/验收口径。
  - 是否继续补充 `gaussian_sigma_1nm_clipped` 或更贴近工艺噪声分布的标签误差实验。
  - 根目录空白文件 `image 2.png.md` 是否属于误导入占位文件，后续是否需要整理归档。

## 2026-07-03

- 在当前仓库根目录建立 `work + wiki` 双层结构。
- 将主要研究资产按最终结构移动到 `work/`：
  - Lumerical 工作流 -> `work/01_simulation_models/01_Lumerical_Workflow/`
  - Zemax 工作流 -> `work/01_simulation_models/02_Zemax_Workflow/`
  - MATLAB 验证 -> `work/01_simulation_models/03_MATLAB_Validation/`
  - 机器学习反演 -> `work/03_ml_inverse_modeling/ML try/`
  - 0614 和根结果目录 -> `work/04_results_and_datasets/`
  - 通用说明和 Notion 导入素材 -> `work/05_reference_materials/`
  - 环境配置 -> `work/06_environment/04_Environment_Config/`
  - 遗留恢复脚本 -> `work/99_legacy_misc/新建文件夹/`
  - `.obsidian/` -> `wiki/.obsidian/`
- 移动过程中，两个大型 `.npy` 文件一度被 `git hash-object` 占用；等待 Git 进程退出后已补移动成功。
- 根目录旧空目录残留已处理；Notion 导入素材现位于 `work/05_reference_materials/Test_repo/`。
- 更新 `.gitignore`，忽略 `.npy`、`.joblib`、`.mat`、`.fsp`、`work/99_legacy_misc/empty_legacy_placeholders/data/`、`work/04_results_and_datasets/`、`.merge_tmp_v2/`、`cnn_dataset/` 和系统元数据，减少 Git 后台扫描大型研究产物。
- 将 `work/` 内知识型 Markdown 迁移到 `wiki/`：
  - 项目/博士方案进入 `wiki/01_Projects/`；
  - 材料与背景进入 `wiki/02_Literature/`；
  - 仿真、信号处理、倾斜入射方法进入 `wiki/03_Methods/`；
  - 仿真报告和 ML 训练报告进入 `wiki/04_Experiments/`；
  - prompt、README、ML 代码说明进入 `wiki/05_CodeNotes/`；
  - GPT 评审、复核意见和回复进入 `wiki/06_Issues/`。
- 按实际内容重组 `work/`：
  - `work/01_simulation_models/`：Lumerical、Zemax、MATLAB 验证；
  - `work/02_analysis_code/`：独立分析脚本；
  - `work/03_ml_inverse_modeling/`：ML 数据集、训练脚本和模型输出；
  - `work/04_results_and_datasets/`：历史结果、图、CSV/NPZ 输出；
  - `work/05_reference_materials/`：原始参考材料和非 Markdown 附件；
  - `work/06_environment/`：环境配置；
  - `work/99_legacy_misc/`：遗留脚本和旧空壳目录。
- `work/` 中剩余的 `._*.md` AppleDouble 元数据已移至 `work/99_legacy_misc/os_metadata/` 并改为 `.appledouble` 扩展名；当前 `work/` 下不再保留真正的 Markdown 文件。

## 2026-07-04

- 新增 `wiki/00_Knowledge_Flow/` 流程型入口，将现有知识库映射为“原始资料 -> 概念沉淀 -> 方法复用 -> 输出成果 -> 定期复盘”的闭环：
  - `README.md`：总控台；
  - `A_原始资料索引.md`：对应 `work/` 原始来源；
  - `B_概念沉淀库.md`：对应项目事实、理论概念、参数口径和实验结论；
  - `C_方法与Skill库.md`：对应可复用方法、流程、prompt 和代码说明；
  - `D_输出成果索引.md`：对应仿真、训练、方案和评审成果；
  - `定期复盘.md`：对应复盘节奏和回流检查表。
- 更新 `wiki/index.md`，将 `00_Knowledge_Flow/` 加入知识库入口和目录说明。
- 修正 `wiki/` 中因 `work/` 重组失效的路径引用：
  - 旧 `01_Lumerical_Workflow/...` 引用已改为 `work/01_simulation_models/01_Lumerical_Workflow/...`；
  - 旧 `01_Lumerical_Workflow/ML try/...` 引用已改为 `work/03_ml_inverse_modeling/ML try/...`；
  - 旧 `work/outputs/...` 引用已改为 `work/04_results_and_datasets/...`；
  - 移动后的仿真报告图片链接已改为指向 `work/` 中实际图片文件。
- 验证结果：
  - 本地图片链接 8 个，缺失 0 个；
  - 明确的 `work/...` 代码路径 75 个，缺失 0 个；
  - 模板型路径 26 个保留为说明用途，例如带 `YYYYMMDD` 或示例输出文件名的路径。
- 将 Notion 导入的 `每周组会讨论.md` 从 `wiki/01_Projects/Thesis_Plan/` 移动到 `wiki/06_Issues/`，按问题跟踪和讨论记录归档。
- 新增 `wiki/01_Projects/研究方向困难与下一步计划.md`，基于 `组会.pdf`、`每周组会讨论.md`、ML 训练报告和方案评审文档，总结当前研究方向、关键困难点和下一步行动建议。
- 新增 `wiki/01_Projects/先验约束与调制增强研究方案.md`，将最新组会约束落实为可执行研究方案：顶面高度先验误差按 `<=10 nm`，膜层厚度先验按工艺百分比扰动，并补充 spin coating、ALD/PEALD、PVD/溅射和椭偏仪相关引用。

## 2026-07-06

- 使用 Acuitik 薄膜反射率计算器网页 API 计算 `HSQ / PSS / SOC / TiO2(or HfO2) / Si substitute` 的 400-800 nm 垂直入射反射谱；输出 CSV、PNG 和 JSON 摘要到 `../work/04_results_and_datasets/acuitik_reflectance_20260706/`，并新增记录页 `wiki/04_Experiments/Simulation_Reports/Acuitik_Reflectance_Calculator_20260706.md`。
- 按“网页内置材料优先”口径补算 Acuitik 替代栈 `SiO2 / Acrylic / Acrylic / Al2O3 / Si`；输出到 `../work/04_results_and_datasets/acuitik_reflectance_20260706_builtin_substitutes/`，并在同一记录页标注 TiO2/HfO2 在该网页材料库中均由 Al2O3 替代。
- 对 `1 mm` 空气腔效应做本地 TMM 高分辨率测试：仅加入空气层不会改变反射率强度；加入弱 SiO2 参考层后出现约 `0.18 nm` 周期的密集条纹，并说明 Acuitik 默认 `500` 点扫描会欠采样。结果保存到 `../work/04_results_and_datasets/acuitik_reflectance_20260706_1mm_air_gap/`。
