---
type: project
status: draft
created: 2026-07-03
updated: 2026-07-03
sources:
  - ../../work/01_simulation_models/01_Lumerical_Workflow/
  - ../../work/01_simulation_models/02_Zemax_Workflow/
tags:
  - lumerical
  - zemax
  - multilayer
  - stackrt
---

# STACK_simu

## 一句话结论

`STACK_simu` 是当前多层膜/谱域干涉仿真的核心研究仓库，包含 Lumerical stackrt、Zemax 对比、MATLAB 验证、机器学习反演和多轮仿真输出。

## 来源路径

- Lumerical 工作流：`../../work/01_simulation_models/01_Lumerical_Workflow/`
- Zemax 工作流：`../../work/01_simulation_models/02_Zemax_Workflow/`
- MATLAB 验证：`../../work/01_simulation_models/03_MATLAB_Validation/`
- 机器学习反演：`../../work/03_ml_inverse_modeling/ML try/`
- 结果与数据集：`../../work/04_results_and_datasets/`
- 参考材料：`../../work/05_reference_materials/`
- 环境配置：`../../work/06_environment/`

## 当前维护原则

- 不把 `.npz`、`.npy`、`.csv`、`.fsp`、训练输出或图片批量复制到 wiki。
- 对重要结果只在 wiki 中记录结论、路径、参数口径和待验证问题。
- `wiki/sources.md` 维护 `work/` 中重要文件的路径地图。
- 知识型 Markdown 已迁移到 `wiki/`；`work/` 主要保留代码、数据、模型、图和仿真工程。

## 待补充

- 核心仿真链路摘要。
- 关键数据集字段说明。
- stackrt 与 Zemax 对比结论。
- 机器学习反演实验脉络。
