@echo off
rem Download and prepare the model files into DS41_MODELS (ds41-paths.bat).  Safe to re-run: finished steps are skipped
rem and the Hugging Face downloads resume.  Disk: ~730 GB free during setup, ~360 GB when done (step 2 says what to
rem delete).  Needs .venv (README.md step 2) and, for step 5, the build (step 3).
setlocal
cd /d "%~dp0"
call "%~dp0ds41-paths.bat"
set "M=%DS41_MODELS:/=\%"
set "HF=%~dp0.venv\Scripts\hf.exe"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%M%" mkdir "%M%"
echo Model folder: %M%

rem 1-2. The trunk: the ds4 Q2 GGUF of DeepSeek's official 1017 weights (366 GB), then a copy without its inline Engram
rem      tables (163 GB; the engine reads Engram from the official shards, step 3) - loads in ~1 min instead of ~6.
if exist "%M%\ds4-q2-slim.gguf" goto engram
"%HF%" download nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF DeepSeek-V4.1-Flash-1017-Q2.gguf --local-dir "%M%\download" || exit /b 1
"%PY%" tools\v41\gguf_drop_tensors.py "%M%\download\DeepSeek-V4.1-Flash-1017-Q2.gguf" "%M%\ds4-q2-slim.gguf.part" || exit /b 1
move /y "%M%\ds4-q2-slim.gguf.part" "%M%\ds4-q2-slim.gguf" >nul || exit /b 1
echo Done: ds4-q2-slim.gguf.  You can now delete "%M%\download" (366 GB).

:engram
rem 3. Engram: the two official shards that hold it, plus the index, config and tokenizer (~190 GB)
"%HF%" download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json tokenizer.json tokenizer_config.json --local-dir "%M%\engram" || exit /b 1

rem 4. The tokenizer tables the server reads
if not exist "%M%\tokenizer\vocab.json" (
  "%PY%" tools\v41\serve\export_tokenizer.py "%M%\engram" "%M%\tokenizer" || exit /b 1
)

rem 5. The DSpark drafter (speculative decoding: the big speed-up; the trunk checks every drafted token, so the output
rem    is the trunk's own).  Put the Strata v41 sidecar GGUF (DeepSeek's MTP heads; README.md says where to get it) at
rem    %M%\V41-1017-MTP.gguf; this turns its experts into IQ2_XXS (4.2 GB, ~5 min), which leaves more VRAM for the trunk.
if exist "%M%\V41-1017-MTP-iq2xxs.gguf" goto done
if not exist "%M%\V41-1017-MTP.gguf" (
  echo Step 5 skipped: no %M%\V41-1017-MTP.gguf yet - the server runs without DSpark until you add it.
  goto done
)
set "REQ=build-ds41\v41_sidecar_requant.exe"
if not exist "%REQ%" set "REQ=build-ds41-portable\v41_sidecar_requant.exe"
"%REQ%" --in "%M%\V41-1017-MTP.gguf" --out "%M%\V41-1017-MTP-iq2xxs.gguf" --type IQ2_XXS --uniform-imatrix || exit /b 1

:done
echo Setup finished.  Start the server with "Start DeepSeek.bat".
exit /b 0
