@echo off
rem Turn voice on: dictation (the microphone button in the chat) and read-aloud (the speaker button under every answer),
rem and the API's /v1/audio/transcriptions and /v1/audio/speech.  Downloads the speech models (1.1 GB, SHA-256 checked:
rem Parakeet TDT 0.6B v3 for speech to text, Kokoro-82M for text to speech, Silero VAD) and sherpa-onnx (28 MB), then
rem saves the choice in voice-choice.txt.  They run on the CPU's efficiency cores and start only when first used.
rem   "Enable Voice.bat" gpu   runs them on a second NVIDIA card instead (~1.8 GB more: sherpa-onnx's CUDA build and
rem                            NVIDIA's CUDA 12 libraries)
rem To turn voice off again, put "off" in voice-choice.txt (or set DS41_VOICE=off).  Restart DeepSeek after either.
setlocal
cd /d "%~dp0"
set "DEV=cpu"
if /i "%~1"=="gpu" set "DEV=gpu"
call "%~dp0setup-ds41-models.bat" voice %DEV%
pause
