---
type: method
status: draft
created: 2026-07-04
updated: 2026-07-09
sources:
  - ../../work/
tags:
  - knowledge_flow
  - methods
  - skill_library
---

# C 方法与 Skill 库

## 一句话结论

本页索引可复用的方法、流程、脚本说明、prompt 和实验模板，目标是让知识从“看过”变成“下次能直接调用”。

## 当前方法入口

- 信号处理方法：[[03_Methods/Signal_Processing/sdi_principle]]
- 高度调制锁相增强反演可观测性：[[03_Methods/Signal_Processing/height_modulated_lockin_observability]]
- lumerical gui仿真参数设置：[[03_Methods/Simulation/Simulation_Parameters_Guide]]
- 代码和数据结构地图：[[05_CodeNotes/STACK_simu_code_map]]
- ML 数据集生成要求：[[05_CodeNotes/ML_CodeNotes/ML_try_prompt]]
- Residual MLP 相关记录：[[05_CodeNotes/ML_CodeNotes/Residual_MLP_README]]
- Agent / Gemini prompt：[[05_CodeNotes/Agent_Prompts/Common_Docs_GEMINI]]、[[05_CodeNotes/Agent_Prompts/Zemax_GEMINI]]

- 悉识 NanoSense 测量原理与适配评估：[[03_Methods/Acuitik_NanoSense_measurement_assessment]]

## 方法页建议模板

```markdown
---
type: method
status: draft
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources:
  - ../../work/relative/path
tags:
  - method
---

# 方法名

## 一句话结论

## 适用场景

## 输入

## 步骤

## 输出

## 参数口径

## 已知风险

## 来源路径
```

## 待补充

- 从已有仿真报告中提炼“Lumerical 批量仿真流程”。
- 从 ML 运行记录中提炼“训练结果对比与复现实验流程”。
- 从评审闭环中提炼“PDR Gate 检查清单模板”。
