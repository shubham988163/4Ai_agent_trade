@echo off
title RanchoTrade Stopper
cd /d "%~dp0"

echo ===================================================
echo   Stopping RanchoTrade Services (Port 8787)...
echo ===================================================

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8787 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo Done. Port 8787 services stopped.
pause
