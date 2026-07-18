---
type: experiment
status: reviewed
created: 2026-07-18
updated: 2026-07-18
sources:
  - ../../../work/04_results_and_datasets/high_speed_static_spectral_inversion_v1/README.md
  - ../../../work/04_results_and_datasets/high_speed_static_spectral_inversion_v1/report/global_optimizer_comparison.md
tags:
  - experiment
  - stackrt
  - tmm
  - inversion
  - benchmark
  - latency
  - static-spectrum
---

# 高速静态光谱 TMM 反演基准 v1

## 一句话结论

`high_speed_static_spectral_inversion_v1` 已把静态 StackRT 数据生成、TMM 反演、多类全局优化器比较和在线延迟统计做成可独立复用的小项目，但当前 smoke 结果离 `100 Hz` 目标仍相差约 `30x` 量级，主要瓶颈仍在 Python 层优化与局部精修开销，而不是数据加载。

## 背景

该目录是一个独立于仓库旧脚本的静态反演基准项目，目标是从“光谱数组已进内存”开始，评估不同反演算法的精度、Air 阶次命中率和在线延迟。它延续了当前 StackRT/TMM 匹配约定：

- `frequency = 3e8 / lambda_nominal`
- `phase_wavelength = 299792458 / frequency`
- 材料光学常数按名义波长采样
- 复折射率使用 `n + i*k`
- 特征矩阵非对角项使用 `-i`
- 正入射使用 `q = n`
- 使用 `Rp`

## 关键事实

- 目标结构仍是 `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`，搜索边界为 `Air 998-1002 um` 与四层膜厚各自窗口。
- 静态数据不再包含时间轴、调制和锁相字段；每组参数只生成一条静态反射光谱。
- smoke 数据通过 `StackRT_CLI` 路径生成，README 明确区分了真 StackRT 数据和 `tmm-smoke` 占位数据，后者不能当成正式 StackRT 结果引用。
- 在当前 smoke 基准里，所有算法的 `P50/P95/P99` 都明显大于 `10 ms`，尚未达到 `100 Hz`。
- 当前最快的是 local：首帧 absolute `P50` 约 `0.37 s`，tracking `P50` 约 `0.32-0.36 s`，吞吐量约 `2.7-3.0 spectra/s`。
- `fft_hybrid` 首帧 absolute 在 `0.45-0.52 s`，`de` / `cmaes` / `direct` 多在 `0.75-0.95 s`，`sobol` 与部分 `de_rand1bin` 更慢。
- 报告结论是：距离 `100 Hz` 仍差约 `30` 倍以上，后续需要解析 Jacobian、编译型 TMM 内核，以及把全局重捕获与常规 tracking 分离。

## 当前判断

- 该项目已满足“独立目录、自包含 README、可复现实验入口”的知识沉淀价值，应视为一个独立基准工作台，而不是普通中间结果目录。
- 现阶段更适合把它当作算法与性能对比框架，而不是已经完成的高速部署方案。
- 其中关于 `StackRT_CLI` 路径和 smoke 数据边界的说明，是后续引用该目录时最需要保留的上下文。

## 适用条件

- 本页结论仅对应当前 smoke 规模、单机单环境、StackRT CLI 生成路径和当前边界设置。
- 若要形成正式算法排名或 100 Hz 可行性结论，需要更大样本量、更多随机种子和正式 final 数据集。

## 来源路径

- `../../../work/04_results_and_datasets/high_speed_static_spectral_inversion_v1/README.md`
- `../../../work/04_results_and_datasets/high_speed_static_spectral_inversion_v1/report/global_optimizer_comparison.md`
