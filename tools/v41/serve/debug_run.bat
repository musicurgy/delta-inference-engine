@echo off
rem debug_run.bat REPORT_FOLDER - "Start DeepSeek (debug).bat" runs this in its own window: the normal launcher, with
rem everything it prints also written to REPORT_FOLDER\console.txt.  Close this window (or press Ctrl+C) when done; the
rem debug launcher then finishes the report.
title DeepSeek - debug run (close this window when you are done)
cd /d "%~dp0..\..\.."
call "Start DeepSeek.bat" 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%~1\console.txt'"
