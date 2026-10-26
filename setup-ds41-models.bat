@echo off
rem Download and prepare the model files into DS41_MODELS (ds41-paths.bat).  "Start DeepSeek.bat" runs this for you when
rem files are missing.  Safe to re-run: finished steps are skipped, the Hugging Face downloads resume, and every large
rem file is checked against the SHA-256 its repository publishes before it is used.  Downloads ~565 GB; needs ~560 GB
rem free while it runs and ~365 GB when done.  Needs .venv (README.md, Setup step 1) and the build (step 2).
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
for /f "usebackq" %%g in (`powershell -NoProfile -Command "[math]::Floor((Get-Item '%M%').PSDrive.Free / 1GB)"`) do set "FREE=%%g"
echo Free space there: %FREE% GB
set "DS=deepseek-ai/DeepSeek-V4.1-Flash-1017"

rem 1-2. The main model: the ds4 Q2 GGUF of DeepSeek's official 1017 weights (366 GB, checked), then a copy without its
rem      inline Engram tables (163 GB; the engine reads Engram from DeepSeek's own shards, step 3) - it loads in ~1 min
rem      instead of ~6.  The 366 GB download is deleted once the copy is in place.
if exist "%M%\ds4-q2-slim.gguf" goto engram
"%HF%" download nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF DeepSeek-V4.1-Flash-1017-Q2.gguf --local-dir "%M%\download" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF --dir "%M%\download" DeepSeek-V4.1-Flash-1017-Q2.gguf || exit /b 1
"%PY%" tools\v41\gguf_drop_tensors.py "%M%\download\DeepSeek-V4.1-Flash-1017-Q2.gguf" "%M%\ds4-q2-slim.gguf.part" || exit /b 1
move /y "%M%\ds4-q2-slim.gguf.part" "%M%\ds4-q2-slim.gguf" >nul || exit /b 1
rd /s /q "%M%\download"
echo Done: ds4-q2-slim.gguf (the 366 GB download was removed).

:engram
rem 3. Engram: the two official shards that hold it (~203 GB, checked), plus the index, config and tokenizer
rem    (shards already there - an earlier setup - are checked first, and downloaded again only if they differ)
if not exist "%M%\engram\model-00047-of-00048.safetensors" goto engram_get
if not exist "%M%\engram\model-00048-of-00048.safetensors" goto engram_get
"%PY%" tools\v41\hf_verify.py --repo %DS% --dir "%M%\engram" model-00047-of-00048.safetensors model-00048-of-00048.safetensors && goto tok
:engram_get
"%HF%" download %DS% model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json tokenizer.json tokenizer_config.json --local-dir "%M%\engram" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo %DS% --dir "%M%\engram" model-00047-of-00048.safetensors model-00048-of-00048.safetensors || exit /b 1

:tok
rem 4. The tokenizer tables the server reads
if not exist "%M%\tokenizer\vocab.json" (
  "%PY%" tools\v41\serve\export_tokenizer.py "%M%\engram" "%M%\tokenizer" || exit /b 1
)

rem 5. The DSpark drafter (speculative decoding: the biggest single speed-up; the main model checks every drafted token,
rem    so the output is the main model's own).  DeepSeek's MTP heads come from three small official shards (7.9 GB,
rem    checked), are repacked into a sidecar GGUF, and its experts shrunk to IQ2_XXS (4.2 GB, ~5 min), which leaves more
rem    VRAM for the main model.  The three shards are deleted afterwards.
if exist "%M%\V41-1017-MTP-iq2xxs.gguf" goto done
if exist "%M%\V41-1017-MTP.gguf" goto requant
"%HF%" download %DS% model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors --local-dir "%M%\engram" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo %DS% --dir "%M%\engram" model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors || exit /b 1
set "V41_SRC=%M%\engram"
set "V41_SRC_NAME=DeepSeek V4.1 Flash 1017"
set "V41_SRC_URL=https://huggingface.co/%DS%"
"%PY%" tools\v41\v41_sidecar.py --out "%M%\V41-1017-MTP.gguf.part" --no-vision || exit /b 1
move /y "%M%\V41-1017-MTP.gguf.part" "%M%\V41-1017-MTP.gguf" >nul || exit /b 1
for %%s in (44 45 46) do del /q "%M%\engram\model-000%%s-of-00048.safetensors" "%M%\engram\model-000%%s-of-00048.safetensors.ok" 2>nul

:requant
set "REQ=build-ds41\v41_sidecar_requant.exe"
if not exist "%REQ%" set "REQ=build-ds41-portable\v41_sidecar_requant.exe"
if not exist "%REQ%" (
  echo No v41_sidecar_requant.exe - build first ^(build-ds41.bat^), then run this again.
  exit /b 1
)
"%REQ%" --in "%M%\V41-1017-MTP.gguf" --out "%M%\V41-1017-MTP-iq2xxs.gguf.part" --type IQ2_XXS --uniform-imatrix || exit /b 1
move /y "%M%\V41-1017-MTP-iq2xxs.gguf.part" "%M%\V41-1017-MTP-iq2xxs.gguf" >nul || exit /b 1

:done
echo Setup finished.
exit /b 0
