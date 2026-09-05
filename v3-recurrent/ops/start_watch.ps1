# Avvia il watcher GPU (gpu_watch.py) DETACHED, nascosto, indipendente dalla
# sessione Claude. Log/stato: %LOCALAPPDATA%\wavelet-v3\.
# Uso: powershell -NoProfile -ExecutionPolicy Bypass -File .\start_watch.ps1
#      .\start_watch.ps1 -Status   |  .\start_watch.ps1 -Stop
param([switch]$Status, [switch]$Stop)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$stateDir = Join-Path $env:LOCALAPPDATA "wavelet-v3"
$pidFile = Join-Path $stateDir "gpu_watch.pid"
$log = Join-Path $stateDir "gpu_watch.log"
New-Item -ItemType Directory -Force $stateDir | Out-Null

function Get-Watcher {
    if (-not (Test-Path $pidFile)) { return $null }
    $p = [int](Get-Content $pidFile -Raw).Trim()
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$p" -ErrorAction SilentlyContinue
    if ($proc -and $proc.CommandLine -match "gpu_watch\.py") { return $proc }
    return $null
}

$w = Get-Watcher
if ($Status) {
    if ($w) { Write-Output "RUNNING pid=$($w.ProcessId)" } else { Write-Output "NOT RUNNING" }
    if (Test-Path $log) { Get-Content $log -Tail 5 }
    exit 0
}
if ($Stop) {
    if ($w) { Stop-Process -Id $w.ProcessId -Force; Write-Output "STOPPED pid=$($w.ProcessId)" }
    else { Write-Output "NOT RUNNING" }
    exit 0
}
if ($w) { Write-Output "ALREADY RUNNING pid=$($w.ProcessId)"; exit 0 }

$python = (Get-Command python -ErrorAction Stop).Source
$script = Join-Path $here "gpu_watch.py"
$out = Join-Path $stateDir "gpu_watch.stdout.txt"
$err = Join-Path $stateDir "gpu_watch.stderr.txt"
$proc = Start-Process -FilePath $python -ArgumentList @("-u", "`"$script`"") `
    -WorkingDirectory $here -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
Start-Sleep -Seconds 3
$w = Get-Watcher
if ($w) { Write-Output "STARTED pid=$($w.ProcessId)" } else { Write-Output "START FAILED (vedi $err)"; exit 1 }
