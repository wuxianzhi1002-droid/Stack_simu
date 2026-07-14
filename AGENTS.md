# STACK_simu Agent Instructions

本仓库根目录是：

```text
D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu
```

本仓库采用 `work + wiki` 双层结构：

- `work/`: 实际研究工作区，给 VSCode、PyCharm、Lumerical、Zemax、MATLAB、Python 使用。
- `wiki/`: Obsidian Vault，只放 Markdown 知识层。
- `laser_research.code-workspace`: VSCode 多根工作区，同时打开 `work` 和 `wiki`。

## Write Policy

- AI/Codex 可以读取 `work/` 中的代码、数据、仿真结果、论文原文和说明文件。
- AI/Codex 默认只允许写入 `wiki/` 中的 Markdown 知识层。
- 需要修改 `work/` 中代码或数据时，必须是用户明确要求的代码/数据任务。
- 不要把数 GB 数据、仿真输出、训练结果、环境目录、缓存目录或临时文件复制进 `wiki/`。
- `wiki/sources.md` 专门记录 `work/` 中的重要文件路径、用途和维护状态。
- 需要沉淀知识时，在 `wiki/` 中写总结、索引、路径引用和结论，不移动、不改写原始数据。

- 不要使用 `apply_patch`；需要编辑文件时，使用 Python 脚本完成修改。

## Safety Rules

- 禁止批量删除文件或目录。
- 不要使用以下命令或等价操作：
  - `del /s`
  - `rd /s`
  - `rmdir /s`
  - `Remove-Item -Recurse`
  - `rm -rf`
- 需要删除文件时，只能一次删除一个明确路径的文件，并在删除前确认该文件不是原始数据、仿真结果或用户正在编辑的文件。
- 不要清理、重排、重命名或拆分大批文件，除非用户明确要求并给出范围。
- 不要回滚用户已有改动。

## Directory Roles

```text
STACK_simu/
  work/
    01_simulation_models/
    02_analysis_code/
    03_ml_inverse_modeling/
    04_results_and_datasets/
    05_reference_materials/
    06_environment/
    99_legacy_misc/
  wiki/
    index.md
    sources.md
    log.md
    01_Projects/
    02_Literature/
    03_Methods/
    04_Experiments/
    05_CodeNotes/
    06_Issues/
  laser_research.code-workspace
```

### `work/`

- 保存原始研究材料、代码、仿真工程、数据集和输出结果。
- `01_simulation_models/`: Lumerical、Zemax、MATLAB 验证等物理/光学仿真模型。
- `02_analysis_code/`: 跨仿真工具对比、后处理和独立分析脚本。
- `03_ml_inverse_modeling/`: 神经网络、Residual MLP、CNN、训练数据和模型结果。
- `04_results_and_datasets/`: 历史运行输出、图表、CSV/NPZ 数据集和结果目录。
- `05_reference_materials/`: 原始参考材料、导入素材、图示和非 Markdown 附件。
- `06_environment/`: conda/pip/软件环境配置。
- `99_legacy_misc/`: 遗留脚本、空占位目录和暂不归类材料。
- 大型二进制文件、`.npz`、`.npy`、`.csv`、`.fsp`、`.zmx`、`.joblib`、`.png` 批量结果应留在这里。
- 对 `work/` 中内容做知识总结时，只在 `wiki/` 中记录相对路径和结论。

### `wiki/`

- 作为 Obsidian Vault 打开。
- 只保存 Markdown 知识页、轻量索引、问题清单、来源映射和研究日志。
- 引用 `work/` 文件时，优先使用相对路径，例如 `../work/simulation/...`。

## Knowledge Update Workflow

1. 读取 `work/` 中的原始资料、代码、数据摘要或仿真报告。
2. 在 `wiki/sources.md` 记录重要来源路径。
3. 在对应知识页沉淀可复用结论、参数口径、方法步骤、失败经验和待验证问题。
4. 在 `wiki/log.md` 追加更新记录。
5. 如果发现冲突或不确定性，记录到 `wiki/06_Issues/`，不要强行合并。

## Markdown Page Template

```markdown
---
type: project | literature | method | experiment | code_note | issue
status: draft | reviewed | stale
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources:
  - ../work/relative/path
tags:
  - tag
---

# Title

## 一句话结论

## 背景

## 关键事实

## 来源路径

## 待验证问题
```

## Current Move Note

- 当前整理已将主要研究资产按内容移动进 `work/01_*` 到 `work/99_*` 分类目录。
- `work/` 中的知识型 Markdown 已迁移到 `wiki/`；`work/` 若仍有 `._*.md`，一般是 macOS 元数据伪文件，不作为知识页处理。
- 若根目录仍残留空目录，优先确认是否被 `git.exe`、`codex.exe`、`node_repl.exe` 或编辑器进程占用；不要强制递归删除。

## Response Style

- 默认用中文回复。
- 区分事实、推断和建议。
- 涉及实验或仿真结论时，写明来源路径、参数口径和适用条件。
