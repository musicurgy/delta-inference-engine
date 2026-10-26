@echo off
rem Start DeepSeek with everything logged, for a bug report.  Run this INSTEAD of "Start DeepSeek.bat" when something
rem goes wrong: it checks the machine and the install, runs the normal launcher in a second window with all its output
rem saved, and when you close that window it packs it all into one zip (logs\debug-<time>.zip) to send us.
rem Nothing is changed on the PC.  The zip holds no prompts or answers, and your Windows user name is replaced by <user>.
setlocal
cd /d "%~dp0"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%t"
set "DBG=%~dp0logs\debug-%TS%"
if not exist "%~dp0logs" mkdir "%~dp0logs"
echo Debug run %TS%: checking this PC and the install ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\v41\serve\debug_report.ps1" -Dir "%DBG%" -Phase before
echo.
echo Now starting DeepSeek in a second window, everything it prints saved for the report.  Use it as usual, or until
echo the problem shows; then CLOSE THAT WINDOW (or press Ctrl+C in it) and come back here.
start "DeepSeek - debug run" /wait cmd /c ""%~dp0tools\v41\serve\debug_run.bat" "%DBG%""
echo.
echo Packing the report ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\v41\serve\debug_report.ps1" -Dir "%DBG%" -Phase after
pause
