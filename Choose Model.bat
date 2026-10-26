@echo off
rem Choose the model "Start DeepSeek.bat" loads: the default ds4 Q2, one of ISTA-DASLab's GSQ-RCO quants (IQ2_XS,
rem IQ2_XXS, IQ1_M: built here from their files) or one of Unsloth's (UD-IQ1_S ... UD-Q8_K_XL: downloaded as they are).
rem It lists what is there and what you can add, with sizes, speed and agreement.  The choice is saved in
rem model-choice.txt here; restart the server to load another model.  (tools\v41\serve\pick_model.py does the work.)
setlocal
cd /d "%~dp0"
call "%~dp0ds41-paths.bat"
if not exist ".venv\Scripts\python.exe" (
  echo Run Start DeepSeek.bat once first: it sets up Python and downloads the default model.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\v41\serve\pick_model.py
pause
