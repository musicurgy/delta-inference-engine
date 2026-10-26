@echo off
rem first-start.bat - "Start DeepSeek.bat" runs this before every start.  It sets up whatever is missing - the Python
rem environment, the build, the model files - and asks before anything big.  Once everything is there it only checks
rem (a fraction of a second).
setlocal
rem the repository's folder, from this file's place in it; the scripts below are called by full path (Windows can be
rem set to look up commands outside the current folder only: NoDefaultCurrentDirectoryInExePath)
for %%I in ("%~dp0..\..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"

rem ---- 1. an NVIDIA driver
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo No NVIDIA driver found ^(nvidia-smi^). Install the latest driver for your card, then start again:
  echo   https://www.nvidia.com/Download/index.aspx
  exit /b 1
)

rem ---- 2. the Python environment (.venv): the server, the downloader, cmake and ninja for the build
if exist ".venv\Scripts\python.exe" if exist ".venv\Scripts\hf.exe" goto build
echo Setting up the Python environment ^(.venv: ~0.3 GB of packages, a minute or two^) ...
set "PYEXE="
for %%v in (3.11 3.12 3.10) do if not defined PYEXE (
  py -%%v -c "import sys" >nul 2>&1 && set "PYEXE=py -%%v"
)
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  echo Python 3.10, 3.11 or 3.12 is needed ^(3.11 recommended^). Install it, ticking "Add python.exe to PATH", then
  echo start again:  https://www.python.org/downloads/windows/
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  %PYEXE% -m venv .venv || exit /b 1
)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt tokenizers==0.23.3 "huggingface_hub[cli]==2.2.0"
if errorlevel 1 (
  echo The package install failed ^(above^). Check the internet connection and start again.
  exit /b 1
)

rem ---- 3. the engine
:build
if exist "build-ds41\ds41_generate.exe" goto models
if exist "build-ds41-portable\ds41_generate.exe" goto models
echo.
echo The engine is not built yet.  Building it needs, once:
set "MISSING="
if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  set "MISSING=1"
  echo   MISSING  Visual Studio 2022 Build Tools with "Desktop development with C++"
  echo            https://visualstudio.microsoft.com/visual-cpp-build-tools/
)
if not defined CUDA_PATH (
  set "MISSING=1"
  echo   MISSING  the CUDA Toolkit ^(13.0 tested^)   https://developer.nvidia.com/cuda-downloads
)
where git >nul 2>&1
if errorlevel 1 (
  set "MISSING=1"
  echo   MISSING  Git   https://git-scm.com/download/win
)
if defined MISSING (
  echo Install what is MISSING ^(then open a NEW window, so Windows sees it^) and start again.
  echo Or skip the build: put the prebuilt "build-ds41-portable" folder from the project's Releases page here.
  exit /b 1
)
echo   Visual Studio, CUDA and Git are installed.
choice /c YN /m "Build the engine now (about 10 minutes)"
if errorlevel 2 exit /b 1
call "%ROOT%\build-ds41.bat"
if errorlevel 1 (
  echo The build stopped ^(above^). README.md, "Setup", lists what it needs.
  exit /b 1
)

rem ---- 4. the model files
:models
call "%ROOT%\ds41-paths.bat"
if not defined DS41_MODELS (
  echo ds41-paths.bat did not set DS41_MODELS ^(the model folder^). Check that file, then start again.
  exit /b 1
)
set "M=%DS41_MODELS:/=\%"
set "NEED="
if not exist "%M%\ds4-q2-slim.gguf" (
  set "NEED=1"
  echo   missing: the main model, ds4-q2-slim.gguf ^(downloads 366 GB, keeps 163 GB^)
)
if not exist "%M%\engram\model-00048-of-00048.safetensors" (
  set "NEED=1"
  echo   missing: the Engram tables ^(DeepSeek's shards 47-48, 203 GB^)
)
if not exist "%M%\tokenizer\vocab.json" set "NEED=1"
if not exist "%M%\V41-1017-MTP-iq2xxs.gguf" (
  set "NEED=1"
  echo   missing: the speculative-decoding drafter ^(downloads 7.9 GB, keeps 4.2 GB^)
)
if not defined NEED exit /b 0
echo The model files above go to %M%
echo   ^(DS41_MODELS in ds41-paths.bat - put it on your fastest NVMe drive^).  They come from Hugging Face: DeepSeek's
echo   official repository and a Q2 GGUF of it; every large file is checked against its published SHA-256.  The first
echo   setup needs ~560 GB free while it runs and keeps ~365 GB.  It can take hours; if it stops, start again and it
echo   goes on where it was.
choice /c YN /m "Download and prepare them now"
if errorlevel 2 exit /b 1
call "%ROOT%\setup-ds41-models.bat"
if errorlevel 1 (
  echo The model setup stopped ^(above^). Start again to go on from where it was.
  exit /b 1
)
exit /b 0
