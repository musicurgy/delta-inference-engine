@echo off
rem Download and prepare the model files into DS41_MODELS (ds41-paths.bat).  "Start DeepSeek.bat" runs this for you when
rem files are missing.  Safe to re-run: finished steps are skipped, the Hugging Face downloads resume, and every large
rem file is checked against the SHA-256 its repository publishes before it is used.  Downloads ~375 GB; needs ~385 GB
rem free while it runs and ~371 GB when done.  Needs .venv (README.md, Setup step 1) and the build (step 2).
setlocal
cd /d "%~dp0"
call "%~dp0ds41-paths.bat"
set "M=%DS41_MODELS:/=\%"
set "HF=%~dp0.venv\Scripts\hf.exe"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%HF%" (
  echo No .venv\Scripts\hf.exe - set up the Python environment first ^(README.md, Setup step 1^).
  exit /b 1
)
if not exist "%M%" mkdir "%M%" || exit /b 1
echo Model folder: %M%
rem stop before a download fills the drive (the downloader only warns): what is still to do vs the free space
"%PY%" tools\v41\setup_space.py "%M%" || exit /b 1
set "DS=deepseek-ai/DeepSeek-V4.1-Flash-1017"
set "Q2=DeepSeek-V4.1-Flash-1017-Q2.gguf"

rem 1. Is the main model needed?  It is downloaded LAST (step 5), so a problem in the small steps shows before hours of
rem    download.  An install from before 2026-10-27 has ds4-q2-slim.gguf + DeepSeek's Engram shards in engram\ instead
rem    of the GGUF: it keeps working as it is.  A model Choose Model.bat picked and that is there (an Unsloth GGUF
rem    carries its own Engram) stands for the default.
set "SKIPMAIN="
if exist "%M%\ds4-q2-slim.gguf" if exist "%M%\engram\model-00048-of-00048.safetensors" set "SKIPMAIN=1"
set "PICK="
if exist "%~dp0model-choice.txt" set /p PICK=<"%~dp0model-choice.txt"
if defined PICK set "PICK=%PICK:/=\%"
if defined PICK if exist "%M%\%PICK%" set "SKIPMAIN=1"
if defined PICK if exist "%PICK%" set "SKIPMAIN=1"
rem a download the 0.9 setup left in download\ (it slimmed it there) is moved into place, not fetched again
if not defined SKIPMAIN if not exist "%M%\%Q2%" if exist "%M%\download\%Q2%" (
  move /y "%M%\download\%Q2%" "%M%\%Q2%" >nul || exit /b 1
  if exist "%M%\download\%Q2%.ok" move /y "%M%\download\%Q2%.ok" "%M%\%Q2%.ok" >nul
)
if exist "%M%\%Q2%.ok" set "SKIPMAIN=1"
rem 2. DeepSeek's small files: the tokenizer, the config and the shard index (the drafter step reads them too)
set "D=%M%\deepseek"
if exist "%M%\engram\tokenizer.json" set "D=%M%\engram"
set "GET="
for %%f in (model.safetensors.index.json config.json tokenizer.json tokenizer_config.json) do if not exist "%D%\%%f" set "GET=1"
if defined GET (
  "%HF%" download %DS% model.safetensors.index.json config.json tokenizer.json tokenizer_config.json --local-dir "%D%" || exit /b 1
)

rem 3. The tokenizer tables the server reads
if not exist "%M%\tokenizer\vocab.json" (
  "%PY%" tools\v41\serve\export_tokenizer.py "%D%" "%M%\tokenizer" || exit /b 1
)

rem 4. The DSpark drafter (speculative decoding: the biggest single speed-up; the main model checks every drafted token,
rem    so the output is the main model's own).  DeepSeek's MTP heads come from three small official shards (7.9 GB,
rem    checked), are repacked into a sidecar GGUF, and its experts shrunk to IQ2_XXS (4.2 GB, ~5 min), which leaves more
rem    VRAM for the main model.  The three shards are deleted afterwards.  Skipped where it is not used (under 14 GB of
rem    VRAM first-start sets DS41_NO_DRAFTER).
if defined DS41_NO_DRAFTER goto main
if exist "%M%\V41-1017-MTP-iq2xxs.gguf" goto main
if exist "%M%\V41-1017-MTP.gguf" goto requant
"%HF%" download %DS% model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors --local-dir "%D%" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo %DS% --dir "%D%" model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors || exit /b 1
set "V41_SRC=%D%"
set "V41_SRC_NAME=DeepSeek V4.1 Flash 1017"
set "V41_SRC_URL=https://huggingface.co/%DS%"
"%PY%" tools\v41\v41_sidecar.py --out "%M%\V41-1017-MTP.gguf.part" --no-vision || exit /b 1
move /y "%M%\V41-1017-MTP.gguf.part" "%M%\V41-1017-MTP.gguf" >nul || exit /b 1
for %%s in (44 45 46) do del /q "%D%\model-000%%s-of-00048.safetensors" "%D%\model-000%%s-of-00048.safetensors.ok" 2>nul

:requant
rem the engine build this CPU runs (cpu_level.py: this PC's own build, else the portable or an older-CPU one)
set "REQ="
for /f "usebackq delims=" %%b in (`""%PY%" tools\v41\serve\cpu_level.py --pick-build"`) do set "REQ=%%b\v41_sidecar_requant.exe"
if not defined REQ set "REQ=build-ds41\v41_sidecar_requant.exe"
if not exist "%REQ%" (
  echo No v41_sidecar_requant.exe - build first ^(build-ds41.bat^), then run this again.
  exit /b 1
)
echo Shrinking the drafter's experts to IQ2_XXS: ~5 minutes on a fast CPU, ~20 on a 4-core one.  It prints little
echo until each block of experts is done - it is not stuck.
"%REQ%" --in "%M%\V41-1017-MTP.gguf" --out "%M%\V41-1017-MTP-iq2xxs.gguf.part" --type IQ2_XXS --uniform-imatrix || exit /b 1
move /y "%M%\V41-1017-MTP-iq2xxs.gguf.part" "%M%\V41-1017-MTP-iq2xxs.gguf" >nul || exit /b 1
rem the full-size drafter (7.9 GB) is only the requant's input
del /q "%M%\V41-1017-MTP.gguf" 2>nul

:main
rem 5. The main model: the ds4 Q2 GGUF of DeepSeek's official 1017 weights (366 GB, checked).  It also carries the
rem    Engram tables - DeepSeek's own bytes - and the engine reads them straight from it.
if defined SKIPMAIN goto done
"%HF%" download nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF %Q2% --local-dir "%M%" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF --dir "%M%" %Q2% || exit /b 1

:done
echo Setup finished.
exit /b 0
