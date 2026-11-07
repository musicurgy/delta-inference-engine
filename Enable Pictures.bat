@echo off
rem Pictures are on by default (0.94).  This sets them up now - DeepSeek's vision encoder (0.97 GB, SHA-256 checked) and
rem PyTorch (the CPU build, 124 MB; the CUDA build, a 2.0 GB download, only when pictures go to a second card) - and
rem turns them on again after Disable Pictures.bat.  Where pictures are read: on the CPU, or on the second card when
rem multi-GPU is on in the chat page's Setup card.  To choose yourself: Enable Pictures.bat cpu, gpu:1 (a card by its
rem nvidia-smi number) or main (the model's own card: fastest, but ~1.5 GB of its VRAM).  Restart DeepSeek after it.
setlocal
cd /d "%~dp0"
call "%~dp0setup-ds41-models.bat" pictures %1
pause
