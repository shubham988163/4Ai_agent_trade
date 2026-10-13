# Register / remove the AI trading agent as a Windows Scheduled Task.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_trader_schedule.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_trader_schedule.ps1 -Status
#   powershell -ExecutionPolicy Bypass -File scripts\install_trader_schedule.ps1 -Uninstall
#
# Why this file exists: install_schedule.ps1 only ever registered the pre-market
# analyst. Nothing launched trading.agents.agent_trader, so across the session
# the council was never invoked, candidate_signals stayed empty, and the desk
# took no trades for days. This registers the actual trader.
#
# The agent is a one-shot pass (it evaluates its candidates and exits), so the
# trigger repeats every INTERVAL_MIN minutes from the 09:15 open for the length
# of the session. Each pass is bounded by the runner's timeout so a hung feed
# cannot strand a Task Scheduler instance.

param(
    [switch]$Uninstall,
    [switch]$Status,
    [int]$IntervalMinutes = 5
)

$ErrorActionPreference = "Stop"

$TaskName = "RanchoTrade-AgentTrader"
$Root     = Split-Path -Parent $PSScriptRoot
$Runner   = Join-Path $Root "run_ai_trader.bat"

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

if (-not (Test-Path $Runner)) {
    Write-Host "ERROR: runner not found: $Runner"
    exit 1
}

# One trigger, repeated across the session (09:15 open -> ~17:15 for IST close).
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "09:15"
$trigger.Repetition = (New-ScheduledTaskTrigger `
    -Once -At "09:15" `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours 8)).Repetition

$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$Runner`"" `
    -WorkingDirectory $Root

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "AI trading agent: runs the F&O scanner + council every $IntervalMinutes min across the session." | Out-Null

Write-Host "registered $TaskName -> every $IntervalMinutes min, 09:15 Mon-Fri (+8h)"
Write-Host "  runner : $Runner"
Write-Host "  workdir: $Root"
Write-Host "  log    : logs\trader_schedule.log"
Write-Host
Write-Host "Verify with -Status; watch logs\trader_schedule.log for per-pass output."
Show-Status
