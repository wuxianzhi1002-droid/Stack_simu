[CmdletBinding()]
param(
    [int]$IntervalSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$mutex = New-Object System.Threading.Mutex($false, 'Local\CodexOpenAIProxyHealth')
if (-not $mutex.WaitOne(0, $false)) {
    exit 0
}

$healthScript = Join-Path $PSScriptRoot 'Ensure-CodexConnectivity.ps1'
try {
    while ($true) {
        try {
            & $healthScript -Quiet
        }
        catch {
            # Clash may still be starting; the next interval retries automatically.
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
