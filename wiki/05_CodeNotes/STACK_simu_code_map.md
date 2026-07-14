---
type: code_note
status: draft
created: 2026-07-03
updated: 2026-07-09
sources:
  - ../../work/
tags:
  - code-map
  - simulation
---

# STACK_simu Code Map

## 一句话结论

当前仓库已按 `work + wiki` 分层：研究资产集中在 `work/`，Obsidian 只打开 `wiki/`。

## 主要目录

| 路径 | 说明 |
|---|---|
| `../../work/01_simulation_models/01_Lumerical_Workflow/` | Lumerical/stackrt 主脚本、FSP 工程、stackrt 输出和倾斜入射仿真 |
| `../../work/01_simulation_models/02_Zemax_Workflow/` | Zemax 自动化脚本、调试工具、材料资产和最终系统文件 |
| `../../work/01_simulation_models/03_MATLAB_Validation/` | MATLAB 或跨工具验证数据 |
| `../../work/02_analysis_code/stackrt_zemax_compare.py` | stackrt 与 Zemax 对比脚本 |
| `../../work/03_ml_inverse_modeling/ML try/` | 光谱数据集生成、Residual MLP/CNN 训练代码、模型文件和训练输出 |
| `../../work/04_results_and_datasets/results/` | 根目录级结果图、CSV、FSP 和对比输出 |
| `../../work/04_results_and_datasets/0614/` | 历史实验结果和相关脚本 |
| `../../work/05_reference_materials/03_Common_Docs/` | 通用说明、材料表和原始非 Markdown 附件 |
| `../../work/06_environment/04_Environment_Config/` | 环境配置记录 |
| `../../work/99_legacy_misc/新建文件夹/` | 历史恢复和数据处理脚本 |
| `../../work/99_legacy_misc/empty_legacy_placeholders/` | 旧分类空占位目录，保留作迁移痕迹 |

## 待补充

- 关键 Python 脚本入口。
- 主要 `.npz` 数据集 keys、shape、生成脚本。
- 结果目录与对应实验问题的映射。
- 过时脚本和仍在使用脚本的区分。

## ????????

- `../../work/02_analysis_code/tmm_inverse_validation_robust.py`?? TMM ????????????????????????????
- `../../work/02_analysis_code/stackrt_vs_tmm_inverse_validation.py`??? StackRT ???? TMM ??????????
- `../../work/02_analysis_code/stackrt_vs_tmm_inverse_validation_v2.py`?? StackRT ? `Cu` ????????? mismatch ???
