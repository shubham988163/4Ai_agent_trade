@echo off
title RanchoTrade Dashboard Launcher
cd /d "%~dp0"

echo ===================================================
echo   Starting RanchoTrade Dashboard & Live Services...
echo ===================================================

wscript.exe "%~dp0scripts\autostart.vbs"

echo.
echo Dashboard launched in background!
echo Opening http://localhost:8787 in your browser...
start http://localhost:8787
echo.
timeout /t 3 /nobreak >nul 2>&1
