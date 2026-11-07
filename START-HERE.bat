@echo off
rem no usage reports to Hugging Face (downloads only)
set HF_HUB_DISABLE_TELEMETRY=1
rem Start here: DeepSeek-V4.1-Flash-1017 on this PC - the same as "Start DeepSeek.bat" (README.md, "Quick start").
rem (Upstream Strata's own first-run script, setup.py, sets up a different model; this project does not use it.)
call "%~dp0Start DeepSeek.bat" %*
