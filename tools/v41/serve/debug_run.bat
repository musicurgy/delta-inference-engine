@echo off
rem debug_run.bat REPORT_FOLDER - "Start DeepSeek (debug).bat" runs this in its own window: the normal launcher, with
rem everything it prints written to REPORT_FOLDER\console.txt and shown here as it comes (debug_tail.ps1).  Close this
rem window (or press Ctrl+C) when done; the debug launcher then finishes the report.
title DeepSeek - debug run (close this window when you are done)
setlocal
cd /d "%~dp0..\..\.."
set "OUT=%~1\console.txt"
type nul > "%OUT%"
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0debug_tail.ps1" -File "%OUT%" -Done "%~1\run.done"
call "%~dp0..\..\..\Start DeepSeek.bat" > "%OUT%" 2>&1
echo.> "%~1\run.done"
rem let the tail print the last lines before the window goes
ping -n 2 127.0.0.1 > nul
