@echo off
rem no usage reports to Hugging Face (downloads only)
set HF_HUB_DISABLE_TELEMETRY=1
rem Download and prepare the model files into DS41_MODELS (ds41-paths.bat).  "Start DeepSeek.bat" runs this for you when
rem files are missing.  The downloads run through tools\v41\serve\hf_get.py: the PC stays awake, a stalled download is
rem started again, a failed one is tried again after a pause, and one line shows the progress.  Safe to re-run:
rem finished steps are skipped, the Hugging Face downloads resume, and every large file is checked against the SHA-256
rem its repository publishes before it is used.  Downloads ~375 GB; needs ~385 GB free while it runs and ~371 GB when
rem done.  Needs .venv (README.md, Setup step 1) and the build (step 2).
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
rem "setup-ds41-models.bat pictures [on|cpu|gpu:N|main]": the pictures choice (step 5; "on" without one), saved
rem before the space check counts them
if /i "%~1"=="pictures" (
  if "%~2"=="" (>"%~dp0vision-choice.txt" echo on) else (>"%~dp0vision-choice.txt" echo %~2)
)
rem "setup-ds41-models.bat voice cpu|gpu": dictation and read-aloud on (step 6)
if /i "%~1"=="voice" (
  if /i "%~2"=="gpu" (>"%~dp0voice-choice.txt" echo gpu) else (>"%~dp0voice-choice.txt" echo cpu)
)
rem stop before a download fills the drive (the downloader only warns): what is still to do vs the free space
"%PY%" tools\v41\setup_space.py "%M%" || exit /b 1
set "DS=deepseek-ai/DeepSeek-V4.1-Flash-1017"
set "Q2=DeepSeek-V4.1-Flash-1017-Q2.gguf"

rem 1. Is the main model needed?  It is downloaded LAST (step 7), so a problem in the small steps shows before hours of
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
  "%PY%" tools\v41\serve\hf_get.py %DS% model.safetensors.index.json config.json tokenizer.json tokenizer_config.json --local-dir "%D%" || exit /b 1
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
rem    0.94: the experts are shrunk with the drafter's own importance matrix (drafter-imatrix.bin, beside the engine's
rem    v41_sidecar_requant.exe, or tools\v41\serve\ in a source checkout) into a NEW file, V41-1017-MTP-iq2xxs-imx.gguf
rem    (+0.6% drafts accepted, +1.2% on prose thinking, same size).  An install from 0.93 has only the old uniform file,
rem    V41-1017-MTP-iq2xxs.gguf: the new one is made once (the shards are downloaded again: the full-size drafter was
rem    deleted after 0.93's requant), and the launcher uses whichever is there, the new one first.
if defined DS41_NO_DRAFTER goto main
if exist "%M%\V41-1017-MTP-iq2xxs-imx.gguf" goto drafter_done
rem the engine build this CPU runs (cpu_level.py: this PC's own build, else the portable or an older-CPU one)
set "REQ="
for /f "usebackq delims=" %%b in (`""%PY%" tools\v41\serve\cpu_level.py --pick-build"`) do set "REQ=%%b\v41_sidecar_requant.exe"
if not defined REQ set "REQ=build-ds41\v41_sidecar_requant.exe"
set "IMX="
for %%r in ("%REQ%") do if exist "%%~dprdrafter-imatrix.bin" set "IMX=%%~dprdrafter-imatrix.bin"
if not defined IMX if exist "tools\v41\serve\drafter-imatrix.bin" set "IMX=tools\v41\serve\drafter-imatrix.bin"
rem an engine build from before 0.94 does not know --imatrix (it answers "unknown arg"; a 0.94 one prints its usage):
rem keep the uniform drafter then, instead of downloading the shards again on every run
if defined IMX if exist "%REQ%" "%REQ%" --imatrix probe 2>&1 | findstr /c:"unknown arg" >nul && set "IMX="
rem no importance matrix (an engine from before 0.94, no source copy) and the uniform drafter is there: keep it as it is
if not defined IMX if exist "%M%\V41-1017-MTP-iq2xxs.gguf" goto main
if exist "%M%\V41-1017-MTP.gguf" goto requant
if exist "%M%\V41-1017-MTP-iq2xxs.gguf" echo Making the 0.94 drafter once: the drafter's 7.9 GB of official shards are downloaded again ^(the old drafter keeps working until it is done^).
"%PY%" tools\v41\serve\hf_get.py %DS% model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors --local-dir "%D%" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo %DS% --dir "%D%" model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors || exit /b 1
set "V41_SRC=%D%"
set "V41_SRC_NAME=DeepSeek V4.1 Flash 1017"
set "V41_SRC_URL=https://huggingface.co/%DS%"
"%PY%" tools\v41\v41_sidecar.py --out "%M%\V41-1017-MTP.gguf.part" --no-vision || exit /b 1
move /y "%M%\V41-1017-MTP.gguf.part" "%M%\V41-1017-MTP.gguf" >nul || exit /b 1
for %%s in (44 45 46) do del /q "%D%\model-000%%s-of-00048.safetensors" "%D%\model-000%%s-of-00048.safetensors.ok" 2>nul

:requant
if not exist "%REQ%" (
  echo No v41_sidecar_requant.exe - build first ^(build-ds41.bat^), then run this again.
  exit /b 1
)
echo Shrinking the drafter's experts to IQ2_XXS: ~5 minutes on a fast CPU, ~20 on a 4-core one.  It prints little
echo until each block of experts is done - it is not stuck.
if not defined IMX goto requant_uniform
"%REQ%" --in "%M%\V41-1017-MTP.gguf" --out "%M%\V41-1017-MTP-iq2xxs-imx.gguf.part" --type IQ2_XXS --imatrix "%IMX%" && goto requant_imx_ok
rem an engine build from before 0.94 does not know --imatrix (the matrix came with a source checkout): the uniform drafter
echo This engine build cannot use the drafter's importance matrix - making the uniform drafter instead.
del /q "%M%\V41-1017-MTP-iq2xxs-imx.gguf.part" 2>nul
if exist "%M%\V41-1017-MTP-iq2xxs.gguf" goto requant_cleanup
:requant_uniform
"%REQ%" --in "%M%\V41-1017-MTP.gguf" --out "%M%\V41-1017-MTP-iq2xxs.gguf.part" --type IQ2_XXS --uniform-imatrix || exit /b 1
move /y "%M%\V41-1017-MTP-iq2xxs.gguf.part" "%M%\V41-1017-MTP-iq2xxs.gguf" >nul || exit /b 1
goto requant_cleanup
:requant_imx_ok
move /y "%M%\V41-1017-MTP-iq2xxs-imx.gguf.part" "%M%\V41-1017-MTP-iq2xxs-imx.gguf" >nul || exit /b 1
:requant_cleanup
rem the full-size drafter (7.9 GB) is only the requant's input
del /q "%M%\V41-1017-MTP.gguf" 2>nul

:drafter_done
rem the 0.93 drafter is not deleted here: it is the user's file.  Say once that it can go.
if exist "%M%\V41-1017-MTP-iq2xxs-imx.gguf" if exist "%M%\V41-1017-MTP-iq2xxs.gguf" echo The older drafter V41-1017-MTP-iq2xxs.gguf ^(4.2 GB^) in the model folder is no longer used: you can delete it.

:main
rem 5. Pictures (on unless vision-choice.txt or DS41_VISION says off; "setup-ds41-models.bat pictures ..." sets it):
rem    DeepSeek's vision tower as one GGUF from the default model's repository (0.97 GB, checked: DeepSeek's BF16
rem    weights) and PyTorch in .venv to run it: the CPU build (124 MB) when pictures are read on the CPU, the CUDA build
rem    (a 2.0 GB download, ~2.9 GB in .venv) only when they are read on a card - the second card with multi-GPU on, or
rem    gpu:N / main (ds41_local.py --vision-setup says which: "<plan> <the PyTorch there is>").
rem    DeepSeek's image code (vision.py, image_processor.py) ships in tools\v41\vision\deepseek.
set "VPLAN=cpu"
set "VTORCH=none"
for /f "usebackq tokens=1,2" %%a in (`""%PY%" tools\v41\serve\ds41_local.py --vision-setup"`) do (
  set "VPLAN=%%a"
  set "VTORCH=%%b"
)
if /i "%VPLAN%"=="off" goto voice
set "VG=DeepSeek-V4.1-Flash-1017-Vision.gguf"
if exist "%M%\%VG%.ok" goto vision_torch
"%PY%" tools\v41\serve\hf_get.py nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF %VG% --local-dir "%M%" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF --dir "%M%" %VG% || exit /b 1
:vision_torch
rem PyTorch: any build does on the CPU; a card needs the CUDA one (a CPU build there is replaced)
if /i "%VTORCH%"=="cuda" goto voice
if /i "%VTORCH%"=="cpu" if /i not "%VPLAN%"=="cuda" goto voice
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if /i "%VPLAN%"=="cuda" set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
if /i "%VPLAN%"=="cuda" echo Pictures on the second card need PyTorch's CUDA build: a 2.0 GB download, ~2.9 GB in .venv.
echo Installing PyTorch for pictures from %TORCH_INDEX% ...
rem not --force-reinstall: that would also replace its dependencies (jinja2 is pinned in requirements.txt)
"%PY%" -m pip uninstall -y -q torch >nul 2>&1
"%PY%" -m pip install -q --disable-pip-version-check torch==2.14.1 --index-url %TORCH_INDEX% || exit /b 1

:voice
rem 6. Voice (off unless voice-choice.txt or DS41_VOICE says cpu or gpu; "Enable Voice.bat" sets it): dictation and
rem    read-aloud.  sherpa-onnx in .venv runs the speech models: its CPU build (28 MB), or for "gpu" its CUDA build with
rem    NVIDIA's CUDA 12 libraries (~1.8 GB).  The models go into the model folder's speech\ (1.1 GB, checked):
rem    NVIDIA Parakeet TDT 0.6B v3 int8 (speech to text, 25 languages, 671 MB, CC-BY-4.0), Kokoro-82M v1.0 (text to
rem    speech, 401 MB, Apache-2.0) and Silero VAD (1.8 MB, MIT), from sherpa-onnx's author's Hugging Face repositories.
set "VOC=off"
if exist "%~dp0voice-choice.txt" set /p VOC=<"%~dp0voice-choice.txt"
if defined DS41_VOICE set "VOC=%DS41_VOICE%"
set "VOCDEV="
if /i "%VOC%"=="cpu" set "VOCDEV=cpu"
if /i "%VOC:~0,3%"=="gpu" set "VOCDEV=gpu"
if not defined VOCDEV goto mainmodel
set "SP=%M%\speech"
if not exist "%SP%" mkdir "%SP%" || exit /b 1
"%PY%" -c "import importlib.util as u, os, sys; s = u.find_spec('sherpa_onnx'); sys.exit(0 if s and ('%VOCDEV%' == 'cpu' or os.path.exists(os.path.join(list(s.submodule_search_locations)[0], 'lib', 'onnxruntime_providers_cuda.dll'))) else 1)" >nul 2>&1 && goto voice_models
echo Installing the speech runtime ^(sherpa-onnx, %VOCDEV%^) ...
set "VREQ=requirements-voice.txt"
if "%VOCDEV%"=="gpu" set "VREQ=requirements-voice-gpu.txt"
rem (the CUDA build carries its own runtime: the CPU build's sherpa-onnx-core is taken out first)
if "%VOCDEV%"=="gpu" "%PY%" -m pip uninstall -y -q sherpa-onnx-core >nul 2>&1
"%PY%" -m pip install -q --disable-pip-version-check -r %VREQ% || exit /b 1
:voice_models
set "ASR=csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
set "TTS=csukuangfj/kokoro-multi-lang-v1_0"
if exist "%SP%\parakeet-tdt-0.6b-v3-int8\encoder.int8.onnx.ok" goto voice_tts
"%PY%" tools\v41\serve\hf_get.py %ASR% encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt --revision ed5d745623472f97cfc2e6c2916ee248798fc5a8 --local-dir "%SP%\parakeet-tdt-0.6b-v3-int8" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo %ASR% --revision ed5d745623472f97cfc2e6c2916ee248798fc5a8 --dir "%SP%\parakeet-tdt-0.6b-v3-int8" joiner.int8.onnx decoder.int8.onnx encoder.int8.onnx || exit /b 1
:voice_tts
if exist "%SP%\kokoro-multi-lang-v1_0\model.onnx.ok" goto voice_vad
"%PY%" tools\v41\serve\hf_get.py %TTS% --revision f7b96bb6bef5c5da4d3aa4f4e0498fbbf62dc78b --local-dir "%SP%\kokoro-multi-lang-v1_0" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo %TTS% --revision f7b96bb6bef5c5da4d3aa4f4e0498fbbf62dc78b --dir "%SP%\kokoro-multi-lang-v1_0" voices.bin lexicon-us-en.txt lexicon-zh.txt model.onnx || exit /b 1
:voice_vad
if exist "%SP%\silero_vad.onnx.ok" goto mainmodel
"%PY%" tools\v41\serve\hf_get.py csukuangfj/vad silero_vad.onnx --revision fba88cd2e921609e7675c3aaf51e0b9b295da4bc --local-dir "%SP%" || exit /b 1
"%PY%" tools\v41\hf_verify.py --repo csukuangfj/vad --revision fba88cd2e921609e7675c3aaf51e0b9b295da4bc --dir "%SP%" silero_vad.onnx || exit /b 1

:mainmodel
rem 7. The main model: the ds4 Q2 GGUF of DeepSeek's official 1017 weights (366 GB, checked).  It also carries the
rem    Engram tables - DeepSeek's own bytes - and the engine reads them straight from it.
if defined SKIPMAIN goto done
set "BIGDL=1"
"%PY%" tools\v41\serve\hf_get.py nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF %Q2% --local-dir "%M%" || exit /b 1
rem the SHA-256 check reads all 366 GB (minutes): the PC stays awake for it too
"%PY%" tools\v41\serve\hf_get.py --awake "%PY%" tools\v41\hf_verify.py --repo nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF --dir "%M%" %Q2% || exit /b 1

:done
echo Setup finished.
rem the big download is done: a Windows notification, for whoever left it running
if defined BIGDL "%PY%" tools\v41\serve\hf_get.py --notify "DeepSeek is set up" "The model is downloaded and checked. DeepSeek starts now; the chat page opens by itself."
exit /b 0
