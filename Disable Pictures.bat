@echo off
rem Turn pictures off: the model starts text only and takes no RAM, CPU or VRAM for them, and the next start does not
rem offer the picture download.  Enable Pictures.bat turns them on again.  Restart DeepSeek after it.
cd /d "%~dp0"
>"%~dp0vision-choice.txt" echo(off
echo Pictures are off. Start DeepSeek again for it to take effect ^(Enable Pictures.bat turns them back on^).
pause
