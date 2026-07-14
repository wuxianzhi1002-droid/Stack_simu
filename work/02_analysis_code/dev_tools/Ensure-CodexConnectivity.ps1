[CmdletBinding()]
param(
    [string]$ClashConfigPath = (Join-Path $HOME '.config\clash\config.yaml'),
    [int]$TimeoutMs = 8000,
    [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Status {
    param([string]$Message)
    if (-not $Quiet) {
        Write-Output $Message
    }
}

function Get-Utf8Json {
    param(
        [string]$Uri,
        [hashtable]$Headers
    )

    $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -Headers $Headers -TimeoutSec 10
    $bytes = [Text.Encoding]::GetEncoding(28591).GetBytes($response.Content)
    return ([Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json)
}

if (-not (Test-Path -LiteralPath $ClashConfigPath)) {
    throw "Clash config not found: $ClashConfigPath"
}

$configText = Get-Content -LiteralPath $ClashConfigPath -Raw -Encoding UTF8
$controllerMatch = [regex]::Match($configText, '(?m)^external-controller:\s*(.+)$')
$secretMatch = [regex]::Match($configText, '(?m)^secret:\s*(.*)$')
if (-not $controllerMatch.Success) {
    throw 'Clash external-controller is not configured.'
}

$controller = $controllerMatch.Groups[1].Value.Trim()
$secret = if ($secretMatch.Success) { $secretMatch.Groups[1].Value.Trim() } else { '' }
$headers = @{}
if ($secret) {
    $headers.Authorization = "Bearer $secret"
}
$baseUri = "http://$controller"

$proxyResponse = Get-Utf8Json -Uri "$baseUri/proxies" -Headers $headers
$groups = @($proxyResponse.proxies.PSObject.Properties | Where-Object {
    $_.Value.type -eq 'Selector' -and $_.Name -ne 'GLOBAL' -and $_.Value.all.Count -gt 10
})
if ($groups.Count -eq 0) {
    throw 'No primary Clash selector group was found.'
}

$group = $groups | Sort-Object { $_.Value.all.Count } -Descending | Select-Object -First 1
$current = [string]$group.Value.now
$preferredPatterns = @(
    '\u65B0\u52A0\u5761|Singapore',
    '\u7F8E\u56FD|United States|USA',
    '\u52A0\u62FF\u5927|Canada',
    '\u65E5\u672C|Japan',
    '\u82F1\u56FD|United Kingdom|\u5FB7\u56FD|Germany|\u6FB3\u5927\u5229\u4E9A|Australia'
)

$rankedCandidates = foreach ($candidate in @($group.Value.all)) {
    for ($rank = 0; $rank -lt $preferredPatterns.Count; $rank++) {
        if ($candidate -match $preferredPatterns[$rank]) {
            [pscustomobject]@{ Name = [string]$candidate; Rank = $rank }
            break
        }
    }
}
$rankedCandidates = @($rankedCandidates | Sort-Object Rank, Name | Select-Object -First 10)
if ($rankedCandidates.Count -eq 0) {
    throw 'No supported-region proxy candidates were found in the Clash selector group.'
}

$testUrl = [uri]::EscapeDataString('https://chatgpt.com/backend-api/')
function Test-ProxyCandidate {
    param($Candidate)

    $encodedName = [uri]::EscapeDataString($candidate.Name)
    $delayUri = "$baseUri/proxies/$encodedName/delay?url=$testUrl&timeout=$TimeoutMs"
    try {
        $delay = Invoke-RestMethod -Uri $delayUri -Headers $headers -TimeoutSec ([math]::Ceiling($TimeoutMs / 1000) + 3)
        return [pscustomobject]@{
            Name = $Candidate.Name
            Rank = $Candidate.Rank
            Delay = [int]$delay.delay
        }
    }
    catch {
        Write-Status "Unavailable: $($Candidate.Name)"
        return $null
    }
}

$currentCandidate = $rankedCandidates | Where-Object { $_.Name -eq $current } | Select-Object -First 1
if ($currentCandidate) {
    $currentResult = Test-ProxyCandidate -Candidate $currentCandidate
    if ($currentResult -and $currentResult.Delay -lt 2000) {
        Write-Status "Healthy: $current ($($currentResult.Delay) ms)"
        return
    }
}

$results = foreach ($candidate in $rankedCandidates) {
    if ($candidate.Name -ne $current) {
        Test-ProxyCandidate -Candidate $candidate
    }
}
$results = @($results)
if ($results.Count -eq 0) {
    throw 'No proxy candidate can reach the ChatGPT backend.'
}

$best = $results | Sort-Object Rank, Delay | Select-Object -First 1
$body = @{ name = $best.Name } | ConvertTo-Json -Compress
$groupUri = "$baseUri/proxies/$([uri]::EscapeDataString($group.Name))"
Invoke-RestMethod -Method Put -Uri $groupUri -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) | Out-Null
Write-Status "Switched: $current -> $($best.Name) ($($best.Delay) ms)"
