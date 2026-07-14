[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sourceWatcher = Join-Path $PSScriptRoot 'Watch-CodexConnectivity.ps1'
$sourceHealthCheck = Join-Path $PSScriptRoot 'Ensure-CodexConnectivity.ps1'
foreach ($sourceFile in @($sourceWatcher, $sourceHealthCheck)) {
    if (-not (Test-Path -LiteralPath $sourceFile)) {
        throw "Required file not found: $sourceFile"
    }
}

$installDirectory = Join-Path $env:LOCALAPPDATA 'CodexConnectivity'
if (-not (Test-Path -LiteralPath $installDirectory)) {
    New-Item -ItemType Directory -Path $installDirectory | Out-Null
}

$watcher = Join-Path $installDirectory 'Watch-CodexConnectivity.ps1'
$healthCheck = Join-Path $installDirectory 'Ensure-CodexConnectivity.ps1'
Copy-Item -LiteralPath $sourceWatcher -Destination $watcher -Force
Copy-Item -LiteralPath $sourceHealthCheck -Destination $healthCheck -Force

$powershellExe = Join-Path $PSHOME 'powershell.exe'
$runKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$command = '"{0}" -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{1}"' -f $powershellExe, $watcher

Set-ItemProperty -LiteralPath $runKey -Name 'CodexOpenAIProxyHealth' -Value $command -Type String
Start-Process -FilePath $powershellExe -ArgumentList @(
    '-NoProfile',
    '-NonInteractive',
    '-WindowStyle',
    'Hidden',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    ('"' + $watcher + '"')
) -WindowStyle Hidden

Write-Output "Codex connectivity watcher installed globally at: $installDirectory"
