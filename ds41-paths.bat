@echo off
rem ---- The settings for this PC (the launcher and setup-ds41-models.bat read them) ----------------------------
rem DS41_MODELS: the folder that holds (or will hold) the model files - on your fastest NVMe drive, ~360 GB.
rem   Default: the "models" folder in this repository.  Change it here, or set an environment variable of the
rem   same name (that wins).  Example: set "DS41_MODELS=D:/DeepSeek-V4.1"
if not defined DS41_MODELS set "DS41_MODELS=%~dp0models"
rem DS41_GPU: the NVIDIA GPU to run on, counted in PCI bus order (nvidia-smi's order; 0 = the first card).
rem   Default: the card with the most VRAM (the first of equals) - set it here to choose another.
if defined DS41_GPU goto gpu_done
set "DS41_GPU=0"
set "DS41_GPU_MIB=0"
for /f "tokens=1,2 delims=, " %%a in ('nvidia-smi --query-gpu^=index^,memory.total --format^=csv^,noheader^,nounits 2^>nul') do call :pick_gpu %%a %%b
set "DS41_GPU_MIB="
goto gpu_done
:pick_gpu
if %2 GTR %DS41_GPU_MIB% (
  set "DS41_GPU=%1"
  set "DS41_GPU_MIB=%2"
)
exit /b 0
:gpu_done
rem Optional: DS41_ARENA_GIB=N fixes the RAM arena (default: sized from your installed RAM, see
rem   tools\v41\serve\ds41_local.py); DS41_RAM_RESERVE_GIB=N leaves N GiB for Windows and other programs (default 38);
rem   DS41_VRAM_MARGIN=N keeps N GiB of VRAM free beside the expert cache (default 0.6) - raise it if other programs
rem   use the card.
