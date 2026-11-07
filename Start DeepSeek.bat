@echo off
rem no usage reports to Hugging Face (downloads only)
set HF_HUB_DISABLE_TELEMETRY=1
rem Start DeepSeek-V4.1-Flash-1017 on http://127.0.0.1:8080/v1 (OpenAI-compatible) - one NVIDIA GPU + system RAM.
rem The settings below are the measured best on an RTX 3090 (README.md, "Settings").  Paths: ds41-paths.bat.  The RAM arena is sized from your installed RAM at every start.
rem   exact:   DSpark speculative decoding (IQ2_XXS drafter), cached verify graphs, shared-expert split, copy stream,
rem            Q8_0 prediction router, Engram hot set (if present), the five scheduling switches, VRAM margin 0.6
rem   lossy, measured: Q6_K dense (re-quantized at load from Q8_0: +2.5-8% speed, top-1 agreement within noise) and
rem            route bias 0.28 (prefers experts already on the GPU: roughly +20% speed for ~1 point of top-1 agreement
rem            with DeepSeek's own API on general text, none on code/maths).  For exact routing set "--route-bias" and "--route-bias-file" to 0
rem            in tools\v41\serve\ds41-windows.json.
cd /d "%~dp0"
rem the first start sets up whatever is missing (Python environment, build, model files) and asks before anything big
call "%~dp0tools\v41\serve\first-start.bat"
if errorlevel 1 (
  pause
  exit /b 1
)
call "%~dp0ds41-paths.bat"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=%DS41_GPU%
rem pin as much of the arena as Windows allows for the GPU (half the RAM) minus 4 GiB; the rest is locked large pages
set STRATA_ARENA_PIN_GIB=auto
set GGML_NO_BACKTRACE=1
rem no plain-text copy of every prompt and answer in %TEMP% (the server's watch log); set it to a file path to get one
set STRATA_TEXT_LOG=0
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
rem Engram rows: 1 = unbuffered reads of DeepSeek's safetensors tables (engram/); tables inside a GGUF (the full
rem model file's own, an Unsloth file's) stay buffered - unbuffered reads of a model file are ~10x slower on
rem Windows.  2 = unbuffered for those too.  Exact either way.
set STRATA_ENGRAM_DIRECT=1
set STRATA_DENSE_ASYNC=1
set STRATA_PASS_NOSYNC=1
rem the next layer's attention is queued before the prediction readback (exact; server +2.2 percent, round 224)
set STRATA_D3_PF_ASYNC=1
rem SSD read-ahead: the experts the next layers will need that live only on the SSD are read into a pinned stash
rem while the current layers run (exact; +22% writing speed on a 64 GB PC).  The engine skips it when every expert
rem is in RAM (128 GB and up), so it costs nothing there.
set STRATA_D3_SPEC=8
set "STRATA_SERVE_DSPARK=%DS41_MODELS%/V41-1017-MTP-iq2xxs.gguf"
if not exist "%DS41_MODELS_W%\V41-1017-MTP-iq2xxs.gguf" set STRATA_SERVE_DSPARK=
rem the card's VRAM (README.md, "What you need"):
rem   under 20 GB: at most 48 GiB of the RAM arena pinned - a 12 GB RTX 3060 ran out of memory at its first request with
rem            ~91 GiB pinned; on a 16 GB budget the cap costs ~2% (31.1 vs 31.8 tok/s)
rem   under 14 GB: nothing pinned (the 3060 with 48 GiB pinned answered short prompts but a prompt read as a batch ran out
rem            of memory - the PCIe path's buffers; unpinned it answers everything: 12 GB budget 16.9 tok/s) and no
rem            drafter - its 2.4 GB of weights cost more expert cache than drafting saves
rem   under 12 GB: probably does not fit (the dense half, KV and buffers need ~9 GB)
set "VRAM_MIB="
for /f "tokens=1" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits -i %DS41_GPU% 2^>nul') do set "VRAM_MIB=%%m"
rem a numeric copy that always exists (an empty %VRAM_MIB% in an if-comparison aborts the script); an answer
rem that is not a number ("[N/A]", "No devices were found") counts as unknown, not as 0 MiB
echo %VRAM_MIB%| findstr /r "^[0-9][0-9]*$" >nul || set "VRAM_MIB="
set /a VRAMN=VRAM_MIB+0 2>nul
if defined VRAM_MIB if %VRAMN% LSS 20000 set STRATA_ARENA_PIN_GIB=48
set "SMALLCARD="
if defined VRAM_MIB if %VRAMN% LSS 14000 set "SMALLCARD=1"
if defined SMALLCARD (
  echo This card has %VRAM_MIB% MiB of VRAM: running without the DSpark drafter and the pinned RAM ^(they need room it lacks^).
  set STRATA_SERVE_DSPARK=
  set STRATA_ARENA_PIN_GIB=1
)
if defined VRAM_MIB if %VRAMN% LSS 11800 echo Under 12 GB of VRAM is untested: the model may not fit on this card.
rem (no ( ... ) block around a line that prints a path: a ")" in the folder's name, "...-main (1)", would end the block)
if not defined SMALLCARD if not defined STRATA_SERVE_DSPARK echo No DSpark drafter at %DS41_MODELS_W%\V41-1017-MTP-iq2xxs.gguf - running without it, slower. See README.md.
set STRATA_DSPARK_KEEP=tools/v41/serve/dspark_expert_counts.txt:64
set STRATA_DSPARK_DEPTH=5
set STRATA_DSPARK_CONF=0.6
rem sampled drafts (exact): at temperature > 0 the drafter samples its drafts from its own sharpened distribution
rem and the check accepts each with min(1, p/q) - the answers are distributed exactly as before, ~4-6% faster at the
rem default sampling (rounds 271-273); greedy (temperature 0) requests are unchanged
set STRATA_DSPARK_SAMPLE=1
set STRATA_DSPARK_SAMPLE_T=0.5
set STRATA_DENSE_REQUANT=q6_K
rem new prompts: experts that 3 or fewer of the prompt's tokens route to run in the CPU pool while the GPU streams the
rem rest - time to first token -40% (a ~180-token prompt 3.9 -> 2.35 s), decode unchanged (0.93, measured on all three
rem menu models); batches up to 16,384 tokens: 1,300 tokens -15%, 25K -2%, 64K -3.5%, same agreement.  0 turns it off.
rem Over 1,024 tokens the limit is 6 tokens: its experts carry more rows and each still costs a whole copy over PCIe: 0.6 s
rem off prompts of 1-13K tokens, same agreement.
set STRATA_CHUNK_CPU=3
set STRATA_CHUNK_CPU_BIG=6
set STRATA_CHUNK_CPU_MAXN=16384
rem serving and prompt switches (all exact; README.md, "Settings"):
rem   rollback 8 - an answer forced after the thinking budget resumes from the live state in ~0.4 s instead of
rem            re-reading all the thinking (was 9-57 s)
rem   batch min 256 - chat-sized turns (128-255 new tokens) run as passes: the first token ~0.5 s sooner
rem   pass cap floor 4096 - fewer attention-graph rebuilds (~2% faster decode)
rem   64 Engram readers - long prompts ~3% faster
rem   one key-set gather - the attention's per-query keys in one gather: long prompts ~5% faster, bit-identical
rem   no prestage region + a 16K prompt batch (--prompt-batch in the config) - prompts: 5K 26 -> 16 s, 20K 46 -> 37 s
rem   DSpark evict 4096 - a big prompt batch frees the drafter's 2.4 GB of VRAM for more columns: 20K 37 -> 33.5 s
rem   rewind 1 - a chat turn whose history matches the cache re-reads ~nothing (was ~19 s per turn at 4K)
rem   noisy-OR prefetch - the drafter's verify passes prefetch the expert most likely to be needed by ANY draft token
rem            (from measured per-rank odds): ~3% faster writing, same quality within rounding
rem   conversation cache 8 - up to 8 whole conversations kept in RAM (~20 MB per 10K tokens): an agent switching between
rem            conversations resumes each in 1-3 s instead of re-reading its prompt (16-18 s at ~8K tokens)
rem   snapshot stack 4 - in thinking mode, the next turn (which drops the old thinking) resumes an earlier snapshot
rem            instead of re-reading the whole history (a 4.7K-token turn 13 s -> 7 s)
rem   bias refresh 1 - the route bias follows the experts in VRAM as the cache changes during an answer (it was fixed at
rem            the answer's start): 40% fewer expert misses, ~12% faster writing, and closer to DeepSeek's own model
rem   keep cache - a follow-up of up to 2048 new tokens borrows only part of the GPU's expert cache and gets exactly
rem            those experts back after, instead of emptying and refilling it: such turns 0.6-0.9 s faster
set STRATA_SERVE_ROLLBACK=8
rem            turns of 64+ new tokens take the batch path (was 256): a ~190-token turn reads 12% faster and the
rem            chat as a whole ~4% at the client (round 246, 10-30); turns under 64 still run token by token
set STRATA_SERVE_BATCH_MIN=64
set STRATA_PASS_CAP_MIN=4096
set STRATA_ENGRAM_THREADS=64
set STRATA_KGATHER=1
set STRATA_PRESTAGE_GIB=0
set STRATA_SERVE_DSPARK_EVICT=4096
rem Bounded replay (0.94): only a prompt's last 128 tokens run layers 21-39, as DeepSeek's own design does.  Agent
rem sessions -4.6 to -6.6 %%, quality inside run-to-run noise on short and long turns (rounds 350, 352, 353).  0 = off.
if not defined STRATA_ENC_BOUND set STRATA_ENC_BOUND=128
set STRATA_SERVE_REWIND1=1
set STRATA_D3_PF_NOISYOR=1
set STRATA_SERVE_CACHE_SLOTS=8
set STRATA_SERVE_SNAP_STACK=4
set STRATA_BIAS_REFRESH=1
set STRATA_SERVE_KEEP_CACHE=1
set STRATA_SERVE_KEEP_LRU=1
set STRATA_SERVE_KEEP_MAX=12000
set "OPENPAGE=--open"
:serve
rem this PC's copy of the config: paths filled in, the RAM arena sized (tools\v41\serve\local\, not in git)
".venv\Scripts\python.exe" tools\v41\serve\ds41_local.py tools\v41\serve\ds41-windows.json tools\v41\serve\local\ds41-windows.json
if not errorlevel 1 goto configured
rem the page's last change cannot start: back to the model that did
".venv\Scripts\python.exe" tools\v41\serve\setup_panel.py --rollback
if not errorlevel 1 goto serve
pause
exit /b 1
:configured
rem the Windows settings that make it faster, if any is off (read only: README.md, "Windows settings for speed")
".venv\Scripts\python.exe" tools\v41\serve\windows_tips.py
rem --open: the chat page opens in the browser once the model is ready (API clients use http://127.0.0.1:8080/v1; the
rem port is DS41_PORT in ds41-paths.bat)
".venv\Scripts\python.exe" tools\v41\serve\serve_ds41.py --engine strata --config tools\v41\serve\local\ds41-windows.json --port %DS41_PORT% %OPENPAGE%
set "SRC=%errorlevel%"
set "OPENPAGE="
rem 75: Apply on the chat page's Setup card: the new settings need a new start.  The page waits and comes back by itself.
if "%SRC%"=="75" goto serve
".venv\Scripts\python.exe" tools\v41\serve\setup_panel.py --rollback
if not errorlevel 1 goto serve
pause
