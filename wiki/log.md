# Log

## 2026-07-16

- 新增 `work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v4.py`，保留 v2 不动；支持 `clean/low/medium/high/all` 四档可复现噪声数据生成。
- 噪声模型区分样本级系统误差与采集级随机误差：膜层 `n/k`、入射角、波长零点和调制幅值每个 NPZ 固定抽样一次；帧增益和反射率读出噪声随时间/像素变化。
- 四档实部折射率相对标准差为 `0/0.05%/0.2%/0.5%`，消光系数为 `0/1%/5%/10%`，角度为 `0/0.05/0.2/0.5 deg`；同时保存波长、振幅、增益和读出噪声配置及实际抽样值。
- v4 NPZ 增加名义/物理波长、名义/实际调制幅值、实际角度、材料扰动、frame gain、干净物理 1f 参考、裁剪比例和生成器版本等元数据。
- `py_compile`、CLI 和 mock StackRT 小规模保存链路已通过；真实四档运行因本机 `lumapi.FDTD(hide=True)` 报 `Session not found` 而未生成正式 NPZ，`work/04_results_and_datasets/dynamic_stackrt_lockin_v4/` 当前尚未建立。

## 2026-07-15

- ???? `wiki/` ? `../work/...` ????????????????????????? `0`???????????????? `work` ???
- ?? `wiki` ? `tmm_joint_inversion_lockin_v3` ????????????`sources.md` ???????????? `work/04_results_and_datasets/tmm_joint_inversion_lockin_v3_20260715_105659/`??????????? `...112455/`?
- ?? `Dynamic_StackRT_TMM_Single_Spectrum_20260713.md`??? `random_cavity_sweep_stackrt_tmm/` ? 30 ????????????????????????????????????????????
- ? `open_questions.md` ??????????v3 ??? `...105659/` ????????????????????????
- ???????`tmm_joint_inversion_lockin_v3_20260715_105659/` ????????? rerun ???????????????`random_cavity_sweep_stackrt_tmm/` ??????????????

- 更正 `tmm_joint_inversion_lockin_v2_20260714_165223` 的结论：v2 把仿真真值 `1000/30/10/40/40` 同时作为每种模式的首个优化起点和软先验中心，因此 rank 1 属于 oracle-assisted 局部闭环，不能作为未知参数反演结果。
- 核实排除真值起点后的最优随机起点 cost：I `1246.10`、D `1745.30`、joint `4203.66`，均远高于真值起点的 `2.83/6.28/10.11`；其余结果普遍偏离真值或落到 bounds，说明当前 8 起点 multistart 无法可靠找到正确解盆地。
- 更新 `Height_Modulated_Lockin_Joint_Inversion_Progress_20260714.md` 和 `height_modulated_lockin_observability.md`，撤回原 MAE 作为盲反演精度的表述，并把“真值隔离、增加起点、全局/分阶段初始化、有限幅值 forward model”列为 v3 必须修正项。
- 新增 `work/02_analysis_code/tmm_joint_inversion_lockin_v3.py`，保留 v2 不动；默认关闭先验，并保证 `EVALUATION_TRUTH` 只在全部优化结束后用于 benchmark 误差和起点审计。
- v3 为 I、D、joint 分别运行 Latin-hypercube 初始化的差分进化，population 为 80，再从每个模式的全局种群选取 32 个候选做局部 least-squares；三种模式不共享候选，避免通道信息串用。
- v3 使用有限幅值正弦 TMM 直接计算时间平均光谱和 `lockin_1f_X/A`，修正 `A=5 nm` 下静态光谱与中心差分的观测模型偏差。
- 最终结果保存到 `work/04_results_and_datasets/tmm_joint_inversion_lockin_v3_20260715_112455/`：I、D、joint 均为 `32/32` 成功、精确真值起点 `0`、边界结果 `0`，全部 rank 收敛到同一真值盆地。
- 新增 `wiki/04_Experiments/Simulation_Reports/TMM_Joint_Inversion_Lockin_v3_20260715.md`，记录算法、审计、rank 分布、结果和理想同模型闭环的适用边界。
- 更新 `wiki/sources.md`、根索引、方法索引和成果索引，加入 v3 代码与最终输出入口。

## 2026-07-14

- 新增 `wiki/04_Experiments/Simulation_Reports/Height_Modulated_Lockin_Joint_Inversion_Progress_20260714.md`，完整记录 StackRT 动态高度调制、1f/2f/3f 数字锁相、TMM 数值约定、I/D/joint 三组反演、参数同步问题和最终结果。
- 核实最终数据 `dynamic_spectra_20260714_161153.npz`：膜栈为 HSQ/PSS/SOC/TiO2 `30/10/40/40 nm`，空气腔 `1000 µm`，调制幅值 `5 nm`，数据尺寸为 `400 × 20000`。
- 核实 `tmm_joint_inversion_lockin_v2_20260714_165223` 已使用 `amplitude_nm=5.0`，并确认 v2 的 multistart 排序会把所有收敛结果排在未收敛结果之前。
- 记录最终三组结果：四层膜 MAE 分别为 I-only `0.695 nm`、D-only `3.439 nm`、joint `2.317 nm`；当前 joint 尚未优于 I-only。
- 明确当前主要限制：`I_meas` 是动态时间平均、`D_meas` 是有限幅值 1f 解调，而反演模型仍使用静态 `I(L0)` 与局部中心差分；在 `A=5 nm` 下存在可测的系统性 forward-model mismatch。
- 重写并更新 `wiki/03_Methods/Signal_Processing/height_modulated_lockin_observability.md`，修复损坏的 LaTeX，区分 `X_1f/A` 的有符号导数与 `R_1f/A` 的非负幅值，并补充有限幅值 forward model 要求。
- 更新 `wiki/index.md`、`wiki/sources.md`、`wiki/00_Knowledge_Flow/C_方法与Skill库.md` 和 `wiki/00_Knowledge_Flow/D_输出成果索引.md`，建立方法、来源、日志和结果之间的查询入口。

- 需要后续验证：
  - 用与数据相同的正弦调制和数字锁相过程构造有限幅值 TMM forward model。
  - 扫描 `A=0.5-10 nm`，比较非线性偏差、谐波比例、噪声收益和反演精度。
  - 在真值偏离先验中心、有噪声和无先验条件下重新比较 I/D/joint 的可观测性。

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
