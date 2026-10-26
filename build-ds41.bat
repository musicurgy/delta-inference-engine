@echo off
rem Build the DeepSeek-V4.1 engine (ds41_generate) and its two helper tools on Windows.
rem   build-ds41.bat             for THIS PC: its CPU's instruction set and its GPU (the fastest here)  -> build-ds41\
rem   build-ds41.bat portable    for OTHER PCs: any x86-64 CPU with AVX2 (Intel 2013+ Haswell, every AMD Ryzen; the
rem                              AVX-512 / AVX-VNNI kernels are picked at run time), RTX 30 / 40 / 50 (sm_86;89;120),
rem                              static MSVC runtime                                              -> build-ds41-portable\
rem   build-ds41.bat compat      for OLDER CPUs without AVX2 (e.g. Sandy / Ivy Bridge and Westmere Xeons): any x86-64
rem                              CPU with SSE4.2 (the AVX kernels are picked at run time where the CPU has AVX), slower
rem                              on the CPU side; RTX 30 / 40 / 50, static MSVC runtime            -> build-ds41-compat\
rem   On a PC without AVX2 the plain build-ds41.bat builds for that CPU by itself.  Each build folder gets cpu-level.txt
rem   (the CPU level it needs); the launcher picks a build this CPU can run (tools\v41\serve\cpu_level.py).
rem Needs: Visual Studio 2019 or 2022 with "Desktop development with C++", the CUDA Toolkit (13.0 tested; it sets
rem CUDA_PATH), Git, and .venv (README.md, step 2: it brings cmake and ninja).  Re-run to rebuild only what changed.
setlocal
set "ROOT=%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto novs
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS goto novs
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if not defined CUDA_PATH (
  echo CUDA_PATH is not set: install the CUDA Toolkit first.
  exit /b 1
)
rem the .venv brings cmake 3.24+ and ninja (Visual Studio 2019's own cmake is 3.20: "CMake 3.24 or higher is required")
if not exist "%ROOT%.venv\Scripts\cmake.exe" (
  echo No .venv\Scripts\cmake.exe - set up the Python environment first ^(README.md, Setup step 1^), then run this again.
  exit /b 1
)
set "PATH=%ROOT%.venv\Scripts;%CUDA_PATH%\bin;%PATH%"
set "LLAMA=%ROOT%third_party\llama.cpp"
rem llama.cpp at Xolotl's pinned commit + tools\xolotl\ggml\xolotl-ggml.patch (as tools/xolotl/fetch-llama.sh does)
set "PIN=3cf03257f219afbe7334045ff7c6a06ac68c627d"
if not exist "%LLAMA%\ggml\CMakeLists.txt" (
  rem core.longpaths: llama.cpp has ~150-character paths (tools\ui), past Windows' 260 under a deep folder
  git clone -q -c core.autocrlf=false -c core.longpaths=true https://github.com/ggml-org/llama.cpp.git "%LLAMA%" || exit /b 1
  git -C "%LLAMA%" -c core.autocrlf=false -c core.longpaths=true checkout -q %PIN% || exit /b 1
  git -C "%LLAMA%" apply "%ROOT%tools\xolotl\ggml\xolotl-ggml.patch" || exit /b 1
  echo llama.cpp %PIN% + xolotl-ggml.patch in third_party\llama.cpp
)
rem the GPU's architecture from nvidia-smi (8.6 -> 86).  RTX 50 = 120a, not 120: ggml's MXFP4 kernels use Blackwell's
rem block-scaled MMA, which exists only on the "a" target (ggml switches its own targets; Strata's copy of them does not)
call "%ROOT%ds41-paths.bat"
set "ARCH=native"
rem GPU_CC, never CC: CMake reads %CC% as the C compiler ("Could not find the compiler specified in ... CC: 8.6")
for /f "tokens=1" %%c in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader -i %DS41_GPU% 2^>nul') do set "GPU_CC=%%c"
if defined GPU_CC set "ARCH=%GPU_CC:.=%"
set "GPU_CC="
if "%ARCH%"=="120" set "ARCH=120a"
set "PORTABLE=OFF"
set "OUT=build-ds41"
rem this CPU's level (avx512 / avx2 / avx / sse); without AVX2 this PC's own build is an older-CPU build (STRATA_ISA_FLOOR)
set "LEVEL="
for /f "usebackq tokens=*" %%l in (`""%ROOT%.venv\Scripts\python.exe" "%ROOT%tools\v41\serve\cpu_level.py""`) do set "LEVEL=%%l"
if not defined LEVEL set "LEVEL=avx2"
set "FLOOR="
if "%LEVEL%"=="avx" set "FLOOR=avx"
if "%LEVEL%"=="sse" set "FLOOR=none"
set "RTL="
if /i "%~1"=="portable" (
  set "ARCH=86;89;120a"
  set "PORTABLE=ON"
  set "OUT=build-ds41-portable"
  set "FLOOR="
  set "LEVEL=avx2"
)
if /i "%~1"=="compat" (
  set "ARCH=86;89;120a"
  set "OUT=build-ds41-compat"
  set "FLOOR=none"
  set "LEVEL=sse"
  set "RTL=-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
)
echo Building %OUT% for GPU architecture %ARCH% and CPU level %LEVEL%
if defined FLOOR echo   ^(no AVX2: the CPU side runs ggml's %FLOOR% kernels and the engine's AVX ones where the CPU has AVX - slower^)
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
cmake -G Ninja -S "%ROOT%." -B "%ROOT%%OUT%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="%NVCC:\=/%" ^
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON "-DCMAKE_CUDA_ARCHITECTURES=%ARCH%" ^
  -DSTRATA_PORTABLE=%PORTABLE% "-DSTRATA_ISA_FLOOR=%FLOOR%" %RTL% -DSTRATA_GGML_DIR="%LLAMA:\=/%" ^
  -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_STANDARD=17 || exit /b 1
cmake --build "%ROOT%%OUT%" --target ds41_generate v41_sidecar_requant ds41_engram_hot && goto built
echo   (the build stopped - trying it once more)
cmake --build "%ROOT%%OUT%" --target ds41_generate v41_sidecar_requant ds41_engram_hot || exit /b 1
:built
rem the CUDA runtime DLLs next to the exe: it then runs without the toolkit on PATH, and the folder can be copied to
rem another PC (which needs only an NVIDIA driver recent enough for this CUDA version)
for %%b in ("%CUDA_PATH%\bin\x64" "%CUDA_PATH%\bin") do (
  for %%f in ("%%~b\cudart64_*.dll" "%%~b\cublas64_*.dll" "%%~b\cublasLt64_*.dll") do (
    if not exist "%ROOT%%OUT%\%%~nxf" copy /y "%%~f" "%ROOT%%OUT%\" >nul
  )
)
rem the CPU level this build needs: the launcher takes the first of build-ds41, build-ds41-portable, build-ds41-compat
rem that this CPU can run (tools\v41\serve\cpu_level.py)
>"%ROOT%%OUT%\cpu-level.txt" echo %LEVEL%
echo Built: %OUT%\ds41_generate.exe
exit /b 0
:novs
echo Visual Studio 2019/2022 with the C++ build tools was not found (vswhere).
exit /b 1
