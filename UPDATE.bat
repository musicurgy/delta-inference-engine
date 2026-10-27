@echo off
rem Update to the newest Delta Inference Engine without starting it: git pull (a git clone), or where to get the new ZIP.
rem Then "Start DeepSeek.bat" gets a new engine or new Python packages first if the update needs them.  The model files
rem are not touched.
setlocal
title Delta Inference Engine - update
cd /d "%~dp0"
rem all of it in one block: cmd reads a .bat file while it runs it, and the git pull can change this very file
(
  if exist ".git" (
    where git >nul 2>nul || (
      echo Git is not on PATH: install Git for Windows ^(https://git-scm.com/download/win^), or run "git pull" here.
      pause
      exit /b 1
    )
    echo Getting the newest version ^(git pull^) ...
    git pull --ff-only
    if errorlevel 1 (
      echo.
      echo git pull did not succeed ^(the reason is above^): nothing was updated.  Files you changed here can stop it:
      echo "git status" lists them.
      pause
      exit /b 1
    )
    echo.
    echo Up to date.  Double-click "Start DeepSeek.bat": it gets a new engine or new packages first if they are needed.
  ) else (
    echo This copy was not made with git, so it cannot update itself.  Download the newest ZIP:
    echo   https://github.com/musicurgy/delta-inference-engine/archive/refs/heads/main.zip
    echo and unpack it over this folder ^(models\ stays as it is^), then double-click "Start DeepSeek.bat".
  )
  pause
  exit /b 0
)
