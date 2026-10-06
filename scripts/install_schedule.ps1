# Register / remove the daily pre-market agent as a Windows Scheduled Task.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Status
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Uninstall
#
# Why this file exists: the only schedulers in this repo were macOS launchd
# (scripts/install_schedule.sh) and cron (scripts/cron_day.sh). Neither exists on
# Windows, so on this machine the pre-market agent never ran on its own and
# data/today_config.json sat stale — the router then rejected it as not-today
# and silently traded the FALLBACK config (half size) every single session.
#
# Trigger is 08:45 Mon-Fri in the machine's LOCAL time, matching the 8:45 IST
# that README and the launchd plist assume. This machine must be set to IST for
# that to land before the 09:15 open; the script prints the local timezone so a
# mismatch is visible rather than silent.
#
# -StartWhenAvailable is Windows' equivalent of launchd's catch-up: a task
# missed because the machine was asleep runs on the next wake. cron cannot do
# this, which is exactly why this repo moved off cron on macOS.

param(
    [switch]$Uninstall,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

$TaskName = "RanchoTrade-Premarket"
$Root     = Split-Path -Parent $PSScriptRoot

# Prefer the project virtualenv, but do not require it: this project has been run
# on the system interpreter too, and a scheduler that refuses to register because
# a venv is missing is worse than one that registers with the interpreter that is
# actually there. (scripts/cron_day.sh and start_day.sh do hard-fail on this.)
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $Python = $venvPython
    $PythonFrom = "project virtualenv"
} else {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host "ERROR: no .venv at $venvPython and no python.exe on PATH."
        Write-Host "       Run: python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt"
        exit 1
    }
    $Python = $cmd.Source
    $PythonFrom = "PATH (no .venv found)"
}

function Show-Status {
    Write-Host "--- scheduled task ---"
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) {
        Write-Host "  $TaskName is NOT registered"
        return
    }
    $t | Get-ScheduledTaskInfo |
        Select-Object TaskName, LastRunTime, LastTaskResult, NextRunTime |
        Format-List | Out-String | Write-Host
    Write-Host "  local timezone: $((Get-TimeZone).Id)"
}

if ($Status) { Show-Status; exit 0 }

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "removed $TaskName"
    } else {
        Write-Host "$TaskName was not registered"
    }
    Show-Status
    exit 0
}

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: interpreter disappeared: $Python"
    exit 1
}

$action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-m trading.agents.premarket" `
    -WorkingDirectory $Root

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "08:45"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

# Re-register from scratch so this is idempotent.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Pre-market analyst agent: writes data/today_config.json before the 09:15 IST open." | Out-Null

Write-Host "registered $TaskName -> 08:45 Mon-Fri, catches up after wake"
Write-Host "  python : $Python  [$PythonFrom]"
Write-Host "  workdir: $Root"
Write-Host
Write-Host "The agent exits 1 when it could not get a verdict (it still writes the"
Write-Host "safe fallback). Check 'LastTaskResult' with -Status after the first run."
Write-Host
Show-Status
