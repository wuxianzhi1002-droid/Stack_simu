param(
    [Parameter(Mandatory=$true)][string]$JobId,
    [Parameter(Mandatory=$true)][string]$LocalDir
)
$ErrorActionPreference = "Stop"
$SshConfig = "C:\Users\wuxianzhi\paratera_config"
$RemoteAlias = "Paratera-scxl838"
$RemoteRun = "/vast/scxl838/gpu_acc_v9_runs/$JobId"
Write-Output "Watcher started for JOBID=$JobId"
while ($true) {
    $state = (& ssh -F $SshConfig $RemoteAlias "if [ -f $RemoteRun/exit_code.txt ]; then cat $RemoteRun/exit_code.txt; else echo ACTIVE; fi").Trim()
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "$stamp state=$state"
    if ($state -ne "ACTIVE") { break }
    Start-Sleep -Seconds 60
}
New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
& scp -F $SshConfig -r "${RemoteAlias}:$RemoteRun/results/." $LocalDir
& scp -F $SshConfig "${RemoteAlias}:$RemoteRun/machine_info.json" $LocalDir
& scp -F $SshConfig "${RemoteAlias}:$RemoteRun/stdout.log" $LocalDir
& scp -F $SshConfig "${RemoteAlias}:$RemoteRun/stderr.log" $LocalDir
& scp -F $SshConfig "${RemoteAlias}:$RemoteRun/exit_code.txt" $LocalDir
$RemoteProject = "/vast/scxl838/phase5_workspace_20260828/02_analysis_code/gpu_acc"
& scp -F $SshConfig "${RemoteAlias}:$RemoteProject/v9_gpu_slurm_$JobId.out" $LocalDir
& scp -F $SshConfig "${RemoteAlias}:$RemoteProject/v9_gpu_slurm_$JobId.err" $LocalDir
if (-not (Test-Path -LiteralPath (Join-Path $LocalDir "v9_gpu_results.json"))) {
    throw "Collection finished but v9_gpu_results.json is missing"
}
Write-Output "COLLECTION_PASS local_dir=$LocalDir"
