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
rem the card (DS41_GPU, from ds41-paths.bat) and the model folder (an engine from before 0.92 cannot open a path with
rem letters outside A-Z: step 3 checks it)
call "%ROOT%\ds41-paths.bat"
rem the engine is built with CUDA 13, which runs on compute capability 7.5 and newer (RTX 20, GTX 16 and later) - a GTX 10
rem card (6.1) would only fail at its first CUDA call, after a newer driver and the downloads.  No answer (an old
rem driver without the query): checked later by the driver version.
set "CCMAJ="
set "CCMIN="
for /f "tokens=1,2 delims=. " %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader -i %DS41_GPU% 2^>nul') do if not defined CCMAJ (set "CCMAJ=%%a" & set "CCMIN=%%b")
set "CC="
echo %CCMAJ%.%CCMIN%| findstr /r "^[0-9][0-9]*\.[0-9][0-9]*$" >nul && set /a CC=CCMAJ*10+CCMIN
if defined CC if %CC% LSS 75 goto oldgpu
rem the engine is built with CUDA 13, which needs driver 580 or newer (an older one fails later with a CUDA error)
set "DRV="
for /f "tokens=1 delims=." %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do if not defined DRV set "DRV=%%v"
if defined DRV if %DRV% LSS 580 (
  echo The NVIDIA driver is version %DRV%; this needs 580 or newer ^(CUDA 13^). Update it, then start again:
  echo   https://www.nvidia.com/Download/index.aspx
  exit /b 1
)

rem ---- 2. the Python environment (.venv): the server, the downloader, cmake and ninja for the build
rem (.venv\requirements.installed: the requirements.txt it was last set up from - an update that changes the file brings
rem the packages up to date, e.g. 0.92's gguf for the drafter's setup step)
if exist ".venv\Scripts\python.exe" if exist ".venv\Scripts\hf.exe" fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1 && goto build
if exist ".venv\Scripts\python.exe" (
  echo Updating the Python environment's packages ^(.venv^) ...
  goto havepy
)
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
if errorlevel 1 (
  echo ^(Windows' package manager winget, which could install it, is not on this PC - Windows LTSC editions have none.^)
  goto nopy
)
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
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt tokenizers==0.23.3 huggingface_hub==2.2.0
if errorlevel 1 (
  echo The package install failed ^(above^). Check the internet connection and start again.
  exit /b 1
)
copy /y requirements.txt ".venv\requirements.installed" >nul

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
set "NEEDMAIN="
if not defined OLD if not exist "%M%\DeepSeek-V4.1-Flash-1017-Q2.gguf.ok" (
  set "NEED=1"
  set "NEEDMAIN=1"
  echo   missing: the main model, DeepSeek-V4.1-Flash-1017-Q2.gguf ^(366 GB, with its Engram tables^)
)
if not exist "%M%\tokenizer\vocab.json" (
  set "NEED=1"
  echo   missing: the tokenizer ^(DeepSeek's small files, a few MB^)
)
rem the drafter: not used under 14 GB of VRAM (Start DeepSeek.bat runs without it there), so not set up there either
set "VRAM_MIB="
for /f "tokens=1" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits -i %DS41_GPU% 2^>nul') do if not defined VRAM_MIB set "VRAM_MIB=%%m"
set "DS41_NO_DRAFTER="
if defined VRAM_MIB if %VRAM_MIB% LSS 14000 set "DS41_NO_DRAFTER=1"
set "NEEDDRAFT="
if not defined DS41_NO_DRAFTER if not exist "%M%\V41-1017-MTP-iq2xxs.gguf" (
  set "NEEDDRAFT=1"
  echo   missing: the speculative-decoding drafter ^(downloads 7.9 GB, keeps 4.2 GB^)
)
rem pictures: on unless vision-choice.txt / DS41_VISION says off (the vision GGUF, 0.97 GB, and PyTorch's CPU build)
set "VIS=cpu"
if exist "%ROOT%ision-choice.txt" set /p VIS=<"%ROOT%ision-choice.txt"
if defined DS41_VISION set "VIS=%DS41_VISION%"
set "NEEDVIS="
if /i "%VIS%"=="cpu" set "NEEDVIS=1"
if /i "%VIS%"=="gpu" set "NEEDVIS=1"
if defined NEEDVIS if exist "%M%\DeepSeek-V4.1-Flash-1017-Vision.gguf.ok" ".venv\Scripts\python.exe" -c "import torch" >nul 2>&1 && set "NEEDVIS="
if defined NEEDVIS echo   missing: pictures ^(the vision encoder, 0.97 GB, and PyTorch, 124 MB^)
if not defined NEED if not defined NEEDDRAFT if not defined NEEDVIS exit /b 0
rem what the README asks for, said before any download (the launcher's own notes come only once all is there)
if defined VRAM_MIB if %VRAM_MIB% LSS 11800 echo   Note: this card has %VRAM_MIB% MiB of VRAM. Under 12 GB is untested: the model may not fit on it.
set "RAM_GB="
for /f "usebackq" %%r in (`".venv\Scripts\python.exe" tools\v41\serve\cpu_level.py --ram-gb`) do set "RAM_GB=%%r"
if defined RAM_GB if %RAM_GB% LSS 120 echo   Note: this PC has %RAM_GB% GB of RAM; DeepSeek needs 128 GB or more ^(README.md, "What you need"^).
if defined RAM_GB if %RAM_GB% LSS 120 if %RAM_GB% GEQ 56 echo         With less it starts, but every token reads the experts that do not fit from the SSD: much slower, untested.
if defined RAM_GB if %RAM_GB% LSS 56 echo         With this little it will probably not start at all.
echo The model files above go to %M%
echo   ^(DS41_MODELS in ds41-paths.bat - put it on your fastest NVMe drive^).  They come from Hugging Face: DeepSeek's
echo   official repository and a Q2 GGUF of it; every large file is checked against its published SHA-256.
if defined NEEDMAIN (
  echo   The first setup downloads ~375 GB, needs ~385 GB free while it runs and keeps ~371 GB.  It can take hours; if
  echo   it stops, start again and it goes on where it was.  ^(For an Unsloth quant instead of this model: answer N,
  echo   run Choose Model.bat, then start again.^)
) else (
  echo   Only the small parts are missing ^(at most ~8 GB to download^).
)
choice /c YN /m "Download and prepare them now"
if errorlevel 2 goto declined
call "%ROOT%\setup-ds41-models.bat"
if errorlevel 1 (
  echo The model setup stopped ^(above^). Start again to go on from where it was.
  exit /b 1
)
exit /b 0
:declined
rem only the drafter / pictures missing: start without them; they are offered again at the next start
if defined NEED exit /b 1
echo Starting without what is missing ^(text only / slower^). It is offered again at the next start.
exit /b 0

:oldgpu
for /f "tokens=*" %%n in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader -i %DS41_GPU% 2^>nul') do set "GPUNAME=%%n"
echo This card, %GPUNAME% ^(compute capability %CCMAJ%.%CCMIN%^), is older than the engine runs on.
echo It needs an NVIDIA RTX 20 or GTX 16 series card or newer ^(compute capability 7.5+^): the engine is built with
echo CUDA 13, which no longer supports the GTX 10 series ^(Pascal^) and older.  A newer driver does not change that.
exit /b 1
