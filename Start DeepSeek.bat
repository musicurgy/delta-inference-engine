@echo off
rem Start DeepSeek-V4.1-Flash-1017 on http://127.0.0.1:8080/v1 (OpenAI-compatible) - one NVIDIA GPU + system RAM.
rem The settings below are the measured best on an RTX 3090 (README.md, "Settings").  Paths: ds41-paths.bat.  The RAM arena is sized from your installed RAM at every start.
rem   exact:   DSpark speculative decoding (IQ2_XXS drafter), cached verify graphs, shared-expert split, copy stream,
rem            Q8_0 prediction router, Engram hot set (if present), the five scheduling switches, VRAM margin 0.6
rem   lossy, measured: Q6_K dense (re-quantized at load from Q8_0: +2.5-8% speed, top-1 agreement within noise) and
rem            route bias 0.3 (prefers experts already on the GPU: roughly +20% speed for ~2 points of top-1 agreement
rem            with DeepSeek's own API on general text, none on code/maths).  For exact routing set "--route-bias" and "--route-bias-file" to 0
rem            in tools\v41\serve\ds41-windows.json.
cd /d "%~dp0"
call "%~dp0ds41-paths.bat"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=%DS41_GPU%
rem pin as much of the arena as Windows allows for the GPU (half the RAM) minus 4 GiB; the rest is locked large pages
set STRATA_ARENA_PIN_GIB=auto
set GGML_NO_BACKTRACE=1
rem above-normal priority: other programs cannot pause the engine's CPU pool
set DS41_PRIORITY=abovenormal
set STRATA_SERVE_STEP_MAX=96
set STRATA_SERVE_PASS_BIAS=0
set STRATA_SERVE_LOOKUP=14
set STRATA_PCIE_PINNED_ONLY=1
set STRATA_BALANCE=1
set STRATA_TIER_BATCH_PCIE=0.5
set STRATA_SHEXP_SPLIT=1
set STRATA_PASS_GRAPHS=1
set STRATA_D3_DMA_STREAM=1
set STRATA_PRED_ROUTER=q8_0
set "DS41_MODELS_W=%DS41_MODELS:/=\%"
set STRATA_ENGRAM_HOT=
if exist "%DS41_MODELS_W%\engram-hot-2g.bin" set "STRATA_ENGRAM_HOT=%DS41_MODELS%/engram-hot-2g.bin"
set STRATA_POOL_TASKS=256
set STRATA_ENGRAM_ASYNC=1
set STRATA_ENGRAM_DIRECT=1
set STRATA_DENSE_ASYNC=1
set STRATA_PASS_NOSYNC=1
set "STRATA_SERVE_DSPARK=%DS41_MODELS%/V41-1017-MTP-iq2xxs.gguf"
if not exist "%DS41_MODELS_W%\V41-1017-MTP-iq2xxs.gguf" (
  echo No DSpark drafter at %DS41_MODELS%/V41-1017-MTP-iq2xxs.gguf - running without it, slower. See README.md.
  set STRATA_SERVE_DSPARK=
)
set STRATA_DSPARK_KEEP=tools/v41/serve/dspark_expert_counts.txt:64
set STRATA_DSPARK_DEPTH=5
set STRATA_DSPARK_CONF=0.6
set STRATA_DENSE_REQUANT=q6_K
rem this PC's copy of the config: paths filled in, the RAM arena sized (tools\v41\serve\local\, not in git)
".venv\Scripts\python.exe" tools\v41\serve\ds41_local.py tools\v41\serve\ds41-windows.json tools\v41\serve\local\ds41-windows.json
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\v41\serve\serve_ds41.py --engine strata --config tools\v41\serve\local\ds41-windows.json --port 8080
pause
