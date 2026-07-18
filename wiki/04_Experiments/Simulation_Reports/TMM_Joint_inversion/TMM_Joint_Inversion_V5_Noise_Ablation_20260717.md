---
type: experiment
status: reviewed
created: 2026-07-18
updated: 2026-07-18
sources:
  - ../../../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v5.py
  - ../../../work/02_analysis_code/tmm_joint_inversion_lockin_v4.py
  - ../../../work/04_results_and_datasets/dynamic_stackrt_lockin_v5/
  - ../../../work/04_results_and_datasets/tmm_joint_inversion_lockin_v4_20260717_182843/analysis_report.md
tags:
  - experiment
  - stackrt
  - tmm
  - joint-inversion
  - lock-in
  - noise-ablation
  - angle
---

# V5 StackRT 噪声消融与角度联合反演分析

## 一句话结论

在当前 `1 mm` 腔长、`220-580 nm` 拟合范围和单次 realization 条件下，`main_dynamic_v5.py + tmm_joint_inversion_lockin_v4.py` 已实现与 StackRT 正向闭环一致的斜入射联合反演；厚度误差的主导项不再是幅值或探测器噪声，而是波长轴零点偏移，其次才是材料 `n/k` 偏差。

## 背景

本轮来源是：

- `main_dynamic_v5.py`：生成含角度、波长、材料、幅值、探测器和组合扰动的 StackRT 动态数据。
- `tmm_joint_inversion_lockin_v4.py`：在 `I(lambda) + lockin_1f_X/A` 条件下加入 `Angle` 参数，执行联合反演。
- `tmm_joint_inversion_lockin_v4_20260717_182843/analysis_report.md`：汇总 19 个 case 的误差、鲁棒性和主导因素判断。

## 关键事实

- 正向闭环已成立：clean 与 angle-only case 在真值厚度和真值角度下可达到 `RMSE(I) ~ 1.8e-12`，`RMSE(dI/dL) ~ 5.7e-10 /um`，相关系数为 `1`。
- 角度-only 扰动下，膜层 MAE 仍低于 `0.0011 nm`，说明当前斜入射 p 偏振 TMM 约定已能匹配 StackRT。
- `Air` 与 `Angle` 强耦合：报告指出 multistart 中 Air-angle 相关性约 `0.95-0.999`，更可靠的是相位等效 Air 长度，而不是几何 Air 长度本身。
- 波长轴偏移是首要误差源：实际偏移仅 `0.0004256 nm` 时，膜层 MAE 已达 `1.36 nm`；`0.000995 nm` 和 `0.004973 nm` 时，膜层 MAE 约为 `5.93 nm` 和 `5.77 nm`。
- 材料 `n/k` 偏差是第二主导项：material low/medium/high 的膜层 MAE 约为 `0.0445 / 0.0815 / 0.4042 nm`。
- 幅值与探测器噪声在当前采样强度下相对温和：其 high case 的膜层 MAE 分别约为 `0.0879 nm` 和 `0.0657 nm`。
- 组合扰动非线性明显：combined low/medium/high 的膜层 MAE 约为 `0.347 / 1.754 / 7.081 nm`，其中 combined high 已触及 `HSQ` 厚度边界。

## 误差解释

- 报告明确建议优先引入有界 `delta_lambda_nm` 波长轴 nuisance 参数，或者先做光谱轴标定，再做厚度反演。
- `Angle` 参数应保留，但更适合作为保护相位长度拟合的辅助量；若没有外部角度先验，不应把几何 Air 长度和角度同时解释为独立可测量量。
- 材料误差更适合用低维、带先验的 `n/k` 修正，而不是完全放开全部材料光学常数。

## 适用条件

- 本页结论仅对应当前 `1 mm` 腔长、`220-580 nm` 拟合窗口、`stride=10`、当前膜层结构和单次 realization。
- 这是一轮消融诊断，不是统计意义上的不确定度评估；若要形成正式鲁棒性结论，需要每个 factor/level 多随机种子重复并校准硬件噪声分布。

## 来源路径

- `../../../work/01_simulation_models/01_Lumerical_Workflow/main_dynamic_v5.py`
- `../../../work/02_analysis_code/tmm_joint_inversion_lockin_v4.py`
- `../../../work/04_results_and_datasets/dynamic_stackrt_lockin_v5/`
- `../../../work/04_results_and_datasets/tmm_joint_inversion_lockin_v4_20260717_182843/analysis_report.md`
