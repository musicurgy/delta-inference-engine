@echo off
rem Turn pictures on: DeepSeek's vision encoder on the CPU (no VRAM). Downloads the vision file (0.97 GB, SHA-256 checked)
rem and PyTorch's CPU build (124 MB), then saves the choice in vision-choice.txt.  To turn them off again, put "off" in
rem vision-choice.txt (or set DS41_VISION=off).  Restart DeepSeek after either.
setlocal
cd /d "%~dp0"
call "%~dp0setup-ds41-models.bat" pictures cpu
pause
