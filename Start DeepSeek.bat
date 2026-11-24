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
rem 0.95: the decode expert-miss copies leave the host first - every miss's DMA in one go on the copy stream, then the
rem hit kernels, each PCIe kernel behind its own copies (same experts, slots, bytes and kernels: exact; needs
rem STRATA_D3_DMA_STREAM above).  set STRATA_D3_DMA_EARLY=0 before starting turns it off.
if not defined STRATA_D3_DMA_EARLY set STRATA_D3_DMA_EARLY=1
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
rem the next layer's attention is queued before the prediction readback (exact; server +2.2 percent, measured)
set STRATA_D3_PF_ASYNC=1
rem SSD read-ahead: the experts the next layers will need that live only on the SSD are read into a pinned stash
rem while the current layers run (exact; +22% writing speed on a 64 GB PC).  The engine skips it when every expert
rem is in RAM (no expert on the SSD), so it costs nothing there; a 128 GB PC may still read some from the SSD.
set STRATA_D3_SPEC=8
rem the engine thread on the CPU pool's reserved core: +1.6%, same output (measured, 11-11)
set STRATA_DS41_HOST_PIN=1
rem unmapped expert reads from the SSD: the model files' memory mapping is released once the experts are placed and the
rem SSD experts are read unbuffered only - the same bytes, so the same text (measured: all six speed-card texts identical
rem at 192 GB), and the mapping no longer competes with the RAM arena: writing +29%% / +10%% / +7%% / +1%% and 32K prompts
rem +26%% / +11%% / +17%% at 64 / 96 / 128 GB (measured).  set STRATA_FILE_TIER_UNMAP=0 to turn it off.
if defined STRATA_P3_UNMAP if not defined STRATA_FILE_TIER_UNMAP set "STRATA_FILE_TIER_UNMAP=%STRATA_P3_UNMAP%"
if not defined STRATA_FILE_TIER_UNMAP set STRATA_FILE_TIER_UNMAP=1
rem wide prompt batches (STRATA_WIDE_PROMPT, from STRATA_WIDE_PROMPT_MIN tokens) and the prefill read-ahead of the
rem router-guessed experts (STRATA_BM_PREFILL_READAHEAD), both exact: ds41_local.py picks them by whether the model's
rem experts fit the RAM arena - some on the SSD: wide from 6144 + read-ahead (+40% at 32K on 64 GB); all in RAM: wide
rem from 16384 only (+18% at 32K on 192 GB; read-ahead -3% at 4K there) - measured.  Set any of them here to
rem override, e.g.  set STRATA_BM_PREFILL_READAHEAD=0
rem the prompt batch width stays the config's 16384.  EXPERIMENTAL, not checked for identical output: batches wider
rem than 16384 may change the text written (STRATA_PROMPT_SWEEP_PB=N raises it with some experts on the SSD;
rem DS41_PROMPT_BATCH=N sets it on any PC; the engine takes only what fits the free VRAM).
rem the checkpoint's file names (data\ds41\checkpoints.json, its "default" row unless DS41_CHECKPOINT names another):
rem the drafter files and the drafter's expert counts below.  The defaults here are 1017's, kept if the table cannot
rem be read.  ds41_local.py warns when a file in use was built from another checkpoint than the model.
set "CK_DRAFT_SRC=V41-1017-MTP.gguf"
set "CK_DRAFT_UNI=V41-1017-MTP-iq2xxs.gguf"
set "CK_DRAFT_IMX=V41-1017-MTP-iq2xxs-imx.gguf"
set "CK_KEEP=tools/v41/serve/dspark_expert_counts_1017.txt"
set "CK_IMATRIX=tools\v41\serve\drafter-imatrix.bin"
for /f "usebackq tokens=1,* delims==" %%a in (`".venv\Scripts\python.exe" tools\v41\ckpt.py bat`) do set "%%a=%%b"
rem the drafter: 0.94's (experts shrunk with the drafter's importance matrix: +0.6% drafts accepted) first, else 0.93's
set STRATA_SERVE_DSPARK=
if defined CK_DRAFT_UNI if exist "%DS41_MODELS_W%\%CK_DRAFT_UNI%" set "STRATA_SERVE_DSPARK=%DS41_MODELS%/%CK_DRAFT_UNI%"
if defined CK_DRAFT_IMX if exist "%DS41_MODELS_W%\%CK_DRAFT_IMX%" set "STRATA_SERVE_DSPARK=%DS41_MODELS%/%CK_DRAFT_IMX%"
rem 0.93's drafter in use and 0.94's could be made: say how (it is a 7.9 GB download for ~0.3% speed, so only on request)
set "IMXNOTE=downloads 7.9 GB once"
if defined CK_DRAFT_SRC if exist "%DS41_MODELS_W%\%CK_DRAFT_SRC%" set "IMXNOTE=about 5 minutes, no download"
if defined CK_DRAFT_IMX if defined CK_DRAFT_UNI if not exist "%DS41_MODELS_W%\%CK_DRAFT_IMX%" if exist "%DS41_MODELS_W%\%CK_DRAFT_UNI%" if defined CK_IMATRIX if exist "%~dp0%CK_IMATRIX%" echo Drafter: 0.93's.  A slightly better one is optional: setup-ds41-models.bat drafter-imx ^(%IMXNOTE%^).
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
if not defined SMALLCARD if not defined STRATA_SERVE_DSPARK echo No DSpark drafter at %DS41_MODELS_W%\%CK_DRAFT_IMX% ^(or the older %CK_DRAFT_UNI%^) - running without it, slower. See README.md.
rem 0.94 (measured): expert counts from the 1017 drafter.  Cards under 22 GB keep 96 / 64 / 32 per block (the old 64
rem list's VRAM, +2 percent drafts accepted); 22 GB and up 112 / 80 / 48 (all-128 acceptance, 1.3 GB less: +2.4 percent writing).
rem (a checkpoint without expert counts yet: all experts per block)
set STRATA_DSPARK_KEEP=
if defined CK_KEEP set "STRATA_DSPARK_KEEP=%CK_KEEP%:96,64,32"
if defined CK_KEEP if defined VRAM_MIB if %VRAMN% GEQ 22000 set "STRATA_DSPARK_KEEP=%CK_KEEP%:112,80,48"
set STRATA_DSPARK_DEPTH=5
rem 0.94 (measured): drafts stop when the PRODUCT of the drafter's confidences falls below 0.25, not at the first
rem draft under 0.6.  At temperature 1 (what apps send) +3 to +11 percent writing speed in all 3 rotations; greedy flat.
set STRATA_DSPARK_CONF=
set STRATA_DSPARK_CUM=0.25
rem sampled drafts (exact): at temperature > 0 the drafter samples its drafts from its own sharpened distribution
rem and the check accepts each with min(1, p/q) - the answers are distributed exactly as before, ~4-6% faster at the
rem default sampling (measured); greedy (temperature 0) requests are unchanged
set STRATA_DSPARK_SAMPLE=1
set STRATA_DSPARK_SAMPLE_T=0.5
set STRATA_DENSE_REQUANT=q6_K
rem new prompts: experts that 3 or fewer of the prompt's tokens route to run in the CPU pool while the GPU streams the
rem rest - time to first token -40% (a ~180-token prompt 3.9 -> 2.35 s), decode unchanged (0.93, measured on all three
rem menu models); batches up to 16,384 tokens: 1,300 tokens -15%, 25K -2%, 64K -3.5%, same agreement.  0 turns it off.
rem Over 1,024 tokens the limit is 6 tokens: its experts carry more rows and each still costs a whole copy over PCIe: 0.6 s
rem off prompts of 1-13K tokens, same agreement.  ds41_local.py turns it off (0) on PCs under ~110 GiB of RAM, where
rem the CPU pool also serves many experts read from the SSD and prompts read ~3.5% slower with it (on 128 GB it is on
rem since 0.95.1: 200 / 500 / 4K-token prompts 10-19% faster, measured).
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
rem            chat as a whole ~4% at the client (measured, 10-30); turns under 64 still run token by token
set STRATA_SERVE_BATCH_MIN=64
set STRATA_PASS_CAP_MIN=4096
set STRATA_ENGRAM_THREADS=64
set STRATA_KGATHER=1
rem 0.95.1: long prompts score the attention indexer with a kernel that skips the warp shuffles (STRATA_INDEXER_V2, on
rem by default in the engine; 'set STRATA_INDEXER_V2=0' before starting turns it off): bit-identical scores, a 147K-token
rem prompt at 128 GB 931 -> 1,143 tok/s, 60K +8%, 32K +4% (measured)
set STRATA_PRESTAGE_GIB=0
set STRATA_SERVE_DSPARK_EVICT=4096
rem Bounded replay (0.94): only a prompt's last 128 tokens run layers 21-39, as DeepSeek's own design does.  Agent
rem sessions -4.6 to -6.6 %%, quality inside run-to-run noise on short and long turns (measured).  0 = off.
if not defined STRATA_ENC_BOUND set STRATA_ENC_BOUND=128
set STRATA_SERVE_REWIND1=1
set STRATA_D3_PF_NOISYOR=1
rem 16 whole conversations in the cache (measured: 10 conversations revisited in turn - an agent with sub-agents -
rem 0%% cached and 7.0 s to the first token with 8 slots, 99%% and 0.62 s with 16).  The RAM bound is unchanged:
rem STRATA_SERVE_CACHE_GIB (8 GiB), least recently used out first.  'set DS41_CACHE_SLOTS=N' before starting sets it.
set STRATA_SERVE_CACHE_SLOTS=16
rem The 0.95 release candidates' names STRATA_P3_SERVER_CACHE_SLOTS and STRATA_P3_SERVER_SEED still work.
if not defined DS41_CACHE_SLOTS if defined STRATA_P3_SERVER_CACHE_SLOTS set "DS41_CACHE_SLOTS=%STRATA_P3_SERVER_CACHE_SLOTS%"
if defined DS41_CACHE_SLOTS set "STRATA_SERVE_CACHE_SLOTS=%DS41_CACHE_SLOTS%"
rem option: 'set STRATA_SERVE_SEED=1' before starting gives a sampled request without a seed a new random one (by
rem default the first sampled reply after every start is the same reply, as in 0.94).  Off by default for now: it
rem changes which sampled replies a fresh start writes.  Greedy requests are never touched.
if not defined STRATA_SERVE_SEED if defined STRATA_P3_SERVER_SEED set "STRATA_SERVE_SEED=%STRATA_P3_SERVER_SEED%"
if not defined STRATA_SERVE_SEED set STRATA_SERVE_SEED=0
set STRATA_SERVE_SNAP_STACK=4
set STRATA_BIAS_REFRESH=1
set STRATA_SERVE_KEEP_CACHE=1
set STRATA_SERVE_KEEP_LRU=1
set STRATA_SERVE_KEEP_MAX=12000
rem 0.95 defaults (each: 'set NAME=0' before starting turns it off; STRATA_KEEP_FIX_GIB: 'set STRATA_KEEP_FIX_GIB=4.3'):
rem   sink rejection 85000 - a sampled token that turns into an attention sink (column 2455 of the residual over
rem            85000 after the last layer: the next tokens become noise) is replaced by the next candidate of its
rem            draw; control tokens are never touched.  It never fired in the 99-question bake-off (same score without it).
if not defined STRATA_SINK_REJECT set STRATA_SINK_REJECT=85000
rem   drafter loaded at start - the DSpark drafter loads before the server reports ready, not inside the first
rem            request (measured: 2.8 s of the first turn)
if not defined STRATA_DSPARK_EAGER set STRATA_DSPARK_EAGER=1
rem   snapshot tail 1024 - each cached conversation also keeps its last 1024 sliding-window rows in RAM, so a request
rem            that shares a prefix ending up to that far before the snapshot (an edited last message) resumes there
rem            instead of re-reading the prompt
if not defined STRATA_SNAP_TAIL set STRATA_SNAP_TAIL=1024
rem   zero-copy predictions - the prefetch predictions come back as a kernel write into pinned memory instead of a
rem            copy queued behind a prefetch DMA (exact; measured +0.1-0.7%%)
if not defined STRATA_ZC_PRED set STRATA_ZC_PRED=1
rem   kernel fusion - the router tail and the prefetch top-16 as one kernel each, the shared expert's clamped SwiGLU
rem            as one (each bit-identical to the chain it replaces; it moves a few MiB of VRAM, so expert placement and
rem            the text can move; measured: +1.3-1.4%% decode)
if not defined STRATA_FUSE set STRATA_FUSE=1
rem   merged decode graphs - each layer's finish graph rides on the next layer's first graph (same ops in the same
rem            order: exact with STRATA_FUSE, measured: +0.2-1.3%%)
if not defined STRATA_X3 set STRATA_X3=1
rem   small folds - the residual add into hc_post, the drafter's clamped SwiGLU as one kernel (exact, measured)
if not defined STRATA_FOLDS set STRATA_FOLDS=1
rem   draft gate + copy yield - the drafts a pass verifies are chosen by expected tokens per ms (from the drafter's
rem            calibrated acceptance; it replaces STRATA_DSPARK_CUM above), and a lookup copy is taken only when its
rem            hit rate beats the drafter's expected drafts.  The output distribution is exact; same-seed texts move.
rem            Measured: agent turns decode +24%%, writing +0.6%%
if not defined STRATA_DSPARK_GATE set STRATA_DSPARK_GATE=1
if not defined STRATA_LOOKUP_YIELD set STRATA_LOOKUP_YIELD=1
rem   learned draft gate + coin-first - a small logistic on the drafter's own confidence, entropy and top-2 margin
rem            sizes each verify pass, with the accept coins drawn first (the output distribution stays exact;
rem            same-seed texts move).  +1.2-2.0%% chat / +2.3%% agent turns; four bake-offs 56 avg vs 58 without (noise).  0 = the gate above.
if not defined STRATA_DSPARK_LGATE set STRATA_DSPARK_LGATE=1
rem   keep reserve 2.8 GiB - a short follow-up read as one batch (64-512 new tokens) frees 2.8 GiB of the GPU's
rem            expert cache for it instead of the 4.3 GiB calibrated on long batches, so more hot experts stay on the
rem            card (measured: agent short turns decode +6.4%%, writing +0.6%%; measured: safe down to 2.3 GiB).
rem            It is a size, not a switch: 4.3 is the old behaviour (0 would leave the batch no room)
if not defined STRATA_KEEP_FIX_GIB set STRATA_KEEP_FIX_GIB=2.8
rem Opt-in (not set here): STRATA_EXPERTS_DROP_MISS=5 is an experimental fast mode - a routed expert of rank 5 or 6 is
rem dropped when it would miss the GPU (measured: +6.8%% decode at 192 GB, ~4%% of routed expert uses dropped).  It changes
rem the answers: two 99-question bake-offs averaged 54 against 58 by default - a speed-for-quality trade.
rem Also opt-in, no faster here: STRATA_IDLE_PREFETCH=1 (prefetch copies only while the bus is idle) and
rem STRATA_X3B=1 (the next layer launched ahead behind a device gate).
rem STRATA_CPU_SHARE_FIRST=1 (measured: it changes the text) and the other decode experiments in docs\DETAILS.md stay off.
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
rem --open: the chat page opens in the browser (API clients use http://127.0.0.1:8080/v1; the port is DS41_PORT in
rem ds41-paths.bat).  --background-load: it opens at once and shows the start's steps while the model loads (a failed
rem start still ends the server with code 1, as below)
".venv\Scripts\python.exe" tools\v41\serve\serve_ds41.py --engine strata --config tools\v41\serve\local\ds41-windows.json --port %DS41_PORT% --background-load %OPENPAGE%
set "SRC=%errorlevel%"
set "OPENPAGE="
rem 75: Apply on the chat page's Setup card: the new settings need a new start.  The page waits and comes back by itself.
if "%SRC%"=="75" goto serve
".venv\Scripts\python.exe" tools\v41\serve\setup_panel.py --rollback
if not errorlevel 1 goto serve
pause
