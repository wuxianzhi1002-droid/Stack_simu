---
type: code_note
status: reviewed
created: 2026-07-13
updated: 2026-07-13
sources:
  - ../../../work/02_analysis_code/dev_tools/Invoke-MultilineCli.ps1
tags:
  - cli
  - prompt
  - powershell
---

# CLI Multiline Input Workaround

## 一句话结论

不要在交互式 CLI 里直接粘贴多行 prompt；把内容放到文件、剪贴板或管道，再用 `Invoke-MultilineCli.ps1` 一次性传入 CLI。

## 背景

Windows 终端、PowerShell、Node CLI 和不同 TUI 对多行粘贴的处理不一致，容易出现换行被提前提交、粘贴截断或交互框状态错乱。更稳定的做法是绕过交互输入框，使用 CLI 的非交互入口。

## 使用方式

从文件传给 Codex：

```powershell
.\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli codex -Path .\prompt.md
```

从剪贴板传给 Codex：

```powershell
.\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli codex -Clipboard
```

从管道传给 Claude：

```powershell
Get-Content -LiteralPath .\prompt.md -Raw -Encoding UTF8 | .\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli claude -Stdin
```

传给 Gemini：

```powershell
.\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli gemini -Path .\prompt.md
```

只检查将要执行的命令，不实际调用 CLI：

```powershell
.\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli codex -Path .\prompt.md -DryRun
```

## 关键事实

- `codex exec` 支持从 stdin 读取 prompt，脚本使用 `<prompt> | codex exec -`。
- `claude -p --input-format text` 支持非交互文本输入，脚本从 stdin 传入 prompt。
- `gemini` 的本地 README 记录了 `gemini -p "..."` 非交互模式；脚本用 `-p` 传入多行字符串。
- 如果不指定 `-Path`、`-Clipboard` 或 `-Stdin`，脚本默认读取剪贴板。

## 待验证问题

- 当前本机 `gemini --help` 会报 `spawn EPERM`，但本地 README 明确记录 `gemini -p` 用法；如果 Gemini 仍无法启动，需要单独修复其 Node/权限问题。
