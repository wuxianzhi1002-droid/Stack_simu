---
type: project
status: draft
created: 2026-07-04
updated: 2026-07-04
sources:
  - ../../work/
tags:
  - knowledge_flow
  - raw_sources
---

# A 原始资料索引

## 一句话结论

原始资料集中保存在 `../../work/`，本页只提供入口和整理口径，不复制大型数据、仿真输出或环境目录。

## 核心入口

- 总来源表：[[sources]]
- 代码和数据结构地图：[[05_CodeNotes/STACK_simu_code_map]]
- 当前项目概览：[[01_Projects/STACK_simu]]

## 原始资料类型

| 类型 | 主要路径 | 消化去向 |
|---|---|---|
| Lumerical / Zemax / MATLAB 仿真模型 | `../../work/01_simulation_models/` | [[B_概念沉淀库]]、[[D_输出成果索引]] |
| 独立分析脚本 | `../../work/02_analysis_code/` | [[C_方法与Skill库]]、[[../05_CodeNotes/STACK_simu_code_map]] |
| ML 反演代码、数据集和模型输出 | `../../work/03_ml_inverse_modeling/` | [[../04_Experiments/ML_Runs]]、[[../05_CodeNotes/ML_CodeNotes]] |
| 历史结果、图表、CSV/NPZ 输出 | `../../work/04_results_and_datasets/` | [[D_输出成果索引]] |
| 参考材料和非 Markdown 附件 | `../../work/05_reference_materials/` | [[02_Literature/Materials/Material_References]] |
| 环境配置 | `../../work/06_environment/` | [[05_CodeNotes/STACK_simu_code_map]] |
| 遗留脚本和旧材料 | `../../work/99_legacy_misc/` | 需要复核后再进入 B/C/D |

## 待补充

- 为重要原始资料补齐“可信度、生成日期、维护状态”。
- 将反复使用的数据集和仿真工程标记为核心来源。
