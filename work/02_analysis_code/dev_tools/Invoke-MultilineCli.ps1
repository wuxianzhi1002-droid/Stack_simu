#requires -Version 5.1
<#
.SYNOPSIS
Pass multiline text to AI CLIs without pasting it into an interactive prompt.

.EXAMPLES
.\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli codex -Path .\prompt.md

.EXAMPLES
Get-Clipboard -Raw | .\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli claude -Stdin

.EXAMPLES
.\work\02_analysis_code\dev_tools\Invoke-MultilineCli.ps1 -Cli gemini -Clipboard -m gemini-2.5-flash
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("codex", "gemini", "claude")]
    [string]$Cli = "codex",

    [string]$Path,

    [switch]$Clipboard,

    [switch]$Stdin,

    [string]$WorkingDirectory = (Get-Location).Path,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @(),

    [switch]$DryRun,

    [Parameter(ValueFromPipeline = $true)]
    [AllowNull()]
    [object]$PipelineInput
)

begin {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"
    $pipelineItems = [System.Collections.Generic.List[string]]::new()

    function Get-PromptText {
        if ($Path) {
            $resolved = Resolve-Path -LiteralPath $Path
            return [System.IO.File]::ReadAllText($resolved.ProviderPath, [System.Text.Encoding]::UTF8)
        }

        if ($Stdin) {
            if ($pipelineItems.Count -gt 0) {
                return ($pipelineItems -join [Environment]::NewLine)
            }
            if ([Console]::IsInputRedirected) {
                return [Console]::In.ReadToEnd()
            }
            throw "No redirected stdin was detected. Pipe text into this script or use -Path/-Clipboard."
        }

        if ($Clipboard -or (-not $Path -and -not $Stdin)) {
            return Get-Clipboard -Raw
        }
    }

    function Show-DryRun {
        param(
            [Parameter(Mandatory = $true)]
            [string]$PromptText
        )

        $lineCount = if ($PromptText.Length -eq 0) { 0 } else { ($PromptText -split "`r?`n").Count }
        Write-Host "CLI: $Cli"
        Write-Host "WorkingDirectory: $WorkingDirectory"
        Write-Host "Prompt chars: $($PromptText.Length)"
        Write-Host "Prompt lines: $lineCount"

        switch ($Cli) {
            "codex"  { Write-Host "Command: <prompt> | codex exec $($ExtraArgs -join ' ') -" }
            "claude" { Write-Host "Command: <prompt> | claude -p --input-format text $($ExtraArgs -join ' ')" }
            "gemini" { Write-Host "Command: gemini -p <prompt> $($ExtraArgs -join ' ')" }
        }
    }
}

process {
    if ($null -ne $PipelineInput) {
        [void]$pipelineItems.Add([string]$PipelineInput)
    }
}

end {
    $promptText = Get-PromptText
    if ([string]::IsNullOrWhiteSpace($promptText)) {
        throw "Prompt text is empty. Use -Path, -Clipboard, or pipe text with -Stdin."
    }

    $resolvedWorkingDirectory = Resolve-Path -LiteralPath $WorkingDirectory

    if ($DryRun) {
        Show-DryRun -PromptText $promptText
        exit 0
    }

    Push-Location -LiteralPath $resolvedWorkingDirectory.ProviderPath
    try {
        switch ($Cli) {
            "codex" {
                $promptText | & codex exec @ExtraArgs -
            }
            "claude" {
                $promptText | & claude -p --input-format text @ExtraArgs
            }
            "gemini" {
                if ($promptText.Length -gt 30000) {
                    Write-Warning "Gemini receives the prompt through -p; very large prompts can exceed Windows command-line limits."
                }
                & gemini -p $promptText @ExtraArgs
            }
        }
    }
    finally {
        Pop-Location
    }
}
