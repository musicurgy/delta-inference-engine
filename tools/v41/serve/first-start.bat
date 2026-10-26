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
rem the engine is built with CUDA 13, which needs driver 580 or newer (an older one fails later with a CUDA error)
set "DRV="
for /f "tokens=1 delims=." %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do if not defined DRV set "DRV=%%v"
if defined DRV if %DRV% LSS 580 (
  echo The NVIDIA driver is version %DRV%; this needs 580 or newer ^(CUDA 13^). Update it, then start again:
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
if defined PYEXE goto havepy
rem no Python: Windows' own package manager (winget, part of Windows 11 and of recent Windows 10) installs python.org's
rem Python 3.11 for this Windows user only - no administrator rights, nothing else on the PC changes
echo Python 3.10, 3.11 or 3.12 is needed and none was found.
where winget >nul 2>&1
if errorlevel 1 goto nopy
choice /c YN /m "Install Python 3.11 now (winget: python.org's installer, for this Windows user only)"
if errorlevel 2 goto nopy
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PYEXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined PYEXE goto havepy
:nopy
echo Install Python 3.11 from https://www.python.org/downloads/windows/ ^(tick "Add python.exe to PATH"^), then
echo start again.
exit /b 1
:havepy
if not exist ".venv\Scripts\python.exe" (
  %PYEXE% -m venv .venv || exit /b 1
)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt tokenizers==0.23.3 "huggingface_hub[cli]==2.2.0"
if errorlevel 1 (
  echo The package install failed ^(above^). Check the internet connection and start again.
  exit /b 1
)

rem ---- 3. the engine: a build this CPU can run (cpu_level.py: this PC's own, the portable AVX2 one, the older-CPU one)
:build
".venv\Scripts\python.exe" tools\v41\serve\cpu_level.py --check && goto models
echo.
rem the prebuilt engine from the project's Releases page (get_engine.py): no compiler, CUDA Toolkit or Git needed
".venv\Scripts\python.exe" tools\v41\serve\get_engine.py --url >nul 2>&1
if errorlevel 1 goto build_here
set "ZIP="
for /f "usebackq tokens=*" %%z in (`".venv\Scripts\python.exe" tools\v41\serve\cpu_level.py --release-zip`) do set "ZIP=%%z"
echo The prebuilt engine for this PC ^(%ZIP%, under 1 GB^) is on the project's Releases page.
choice /c YN /m "Download it now (no compiler needed)"
if errorlevel 2 goto build_here
".venv\Scripts\python.exe" tools\v41\serve\get_engine.py
if errorlevel 1 goto dl_failed
".venv\Scripts\python.exe" tools\v41\serve\cpu_level.py --check && goto models
:dl_failed
echo.
echo The download did not work ^(above^).  Start again to retry, or build the engine here:
:build_here
echo To build the engine on this PC, it needs, once:
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
  echo Or skip the build: put the prebuilt engine folder from the project's Releases page here ^(build-ds41-portable;
  echo for a CPU without AVX2 build-ds41-compat-avx, or build-ds41-compat without AVX^).
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
rem the main model: the ds4 Q2 GGUF, checked (.ok) - or a model Choose Model.bat picked (model-choice.txt: a GSQ or
rem Unsloth file that is there), or an install from before 2026-10-27 (slim copy + Engram shards)
set "OLD="
if exist "%M%\ds4-q2-slim.gguf" if exist "%M%\engram\model-00048-of-00048.safetensors" set "OLD=1"
set "PICK="
if exist "%ROOT%\model-choice.txt" set /p PICK=<"%ROOT%\model-choice.txt"
if defined PICK set "PICK=%PICK:/=\%"
if defined PICK if exist "%M%\%PICK%" set "OLD=1"
if defined PICK if exist "%PICK%" set "OLD=1"
if not defined OLD if not exist "%M%\DeepSeek-V4.1-Flash-1017-Q2.gguf.ok" (
  set "NEED=1"
  echo   missing: the main model, DeepSeek-V4.1-Flash-1017-Q2.gguf ^(366 GB, with its Engram tables^)
)
if not exist "%M%\tokenizer\vocab.json" (
  set "NEED=1"
  echo   missing: the tokenizer ^(DeepSeek's small files, a few MB^)
)
if not exist "%M%\V41-1017-MTP-iq2xxs.gguf" (
  set "NEED=1"
  echo   missing: the speculative-decoding drafter ^(downloads 7.9 GB, keeps 4.2 GB^)
)
if not defined NEED exit /b 0
echo The model files above go to %M%
echo   ^(DS41_MODELS in ds41-paths.bat - put it on your fastest NVMe drive^).  They come from Hugging Face: DeepSeek's
echo   official repository and a Q2 GGUF of it; every large file is checked against its published SHA-256.  The first
echo   setup downloads ~375 GB, needs ~385 GB free while it runs and keeps ~371 GB.  It can take hours; if it stops,
echo   start again and it goes on where it was.
choice /c YN /m "Download and prepare them now"
if errorlevel 2 exit /b 1
call "%ROOT%\setup-ds41-models.bat"
if errorlevel 1 (
  echo The model setup stopped ^(above^). Start again to go on from where it was.
  exit /b 1
)
exit /b 0
