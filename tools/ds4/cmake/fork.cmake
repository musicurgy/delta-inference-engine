# tools/ds4/cmake/fork.cmake - every Strata-DS4 fork target, kept out of upstream's CMakeLists.txt (which has one
# include() line for this file at its end; docs/UPSTREAM_SYNC.md).  Included in the root directory scope, so the
# root's variables (_strata_gpu_include_directories, _strata_gpu_runtime_target) and targets are visible.

if(STRATA_NATIVE_EXPERTS AND TARGET strata_kernels_cpu)
  # PTQ1_0 ternary expert rows (DeepSeek-V4.1 mixed tier); native_expert.cpp calls them (function-level AVX2
  # attributes, no per-file flags)
  target_sources(strata_kernels_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/ptq1_0.cpp)
  # Windows (MSVC has no per-function target attributes): without /arch:AVX2 the compiler emits legacy-SSE encodings
  # for the scalar code around the 256-bit intrinsics, and each SSE <-> AVX-256 transition costs on Intel cores.
  # STRATA_MSVC_KERNEL_AVX2=OFF keeps the portable (any-x64) build.  native_expert.cpp: the CPU tier's row loops and
  # SwiGLU around those kernels.
  option(STRATA_MSVC_KERNEL_AVX2 "MSVC: compile ptq1_0.cpp and native_expert.cpp with /arch:AVX2" ON)
  # not for the older-CPU builds (STRATA_ISA_FLOOR): native_expert.cpp runs on every CPU expert row, so /arch:AVX2 there
  # would put AVX2 instructions in front of a CPU without them
  if(MSVC AND STRATA_MSVC_KERNEL_AVX2 AND STRATA_ISA_FLOOR STREQUAL "")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/ptq1_0.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/native_expert.cpp
                                PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
  if(STRATA_BUILD_TESTS)   # layout vs llama-tq's reference, dot parity
    add_executable(ptq1_0_test tests/cpu/ptq1_0_test.cpp)
    target_link_libraries(ptq1_0_test PRIVATE strata_kernels_cpu)
    add_test(NAME ptq1_0_test COMMAND ptq1_0_test)
  endif()
endif()

if(STRATA_BUILD_TESTS AND TARGET strata_artifact)
  # DSV4 geometry + full tensor map (header mmap only; the real-file part skips without $DS4_GGUF/argv[1])
  add_executable(ds4_geometry_test tests/core/ds4_geometry_test.cpp)
  target_link_libraries(ds4_geometry_test PRIVATE strata_artifact)
  add_test(NAME ds4_geometry_test COMMAND ds4_geometry_test $ENV{DS4_GGUF})
endif()

if(STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND NOT STRATA_HIP_GFX906)
  # DS4 phase 2 gate: one real DeepSeek-V4 expert (gate/up IQ2_XXS, down Q2_K) through the CPU pool and the
  # GPU native MMVQ path, against ggml's dequantizer + an F32 matmul.  The 0731 artifact comes from
  # argv[1]/$DS4_GGUF; without it the binary prints "skipped" and exits 0.  --cpu-only runs no CUDA.
  add_executable(ds4_expert_parity src/kernels/ds4_expert_parity.cpp)
  target_include_directories(ds4_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ds4_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS)
    add_test(NAME ds4_expert_parity COMMAND ds4_expert_parity "$ENV{DS4_GGUF}")
  endif()
  # DS4 phase 4a: the DeepSeek-V4 expert engine replayed on recorded routes (tools/ds4/moe_replay.cpp).
  # It links the real machinery - strata_engine's ExpertCache, the CPU ExpertPool, the native grouped
  # kernels, PinnedArena - and drives them from ds4routes.bin.  --correctness --cpu-only needs no GPU.
  add_executable(ds4_moe_replay tools/ds4/moe_replay.cpp tools/ds4/moe_gap.cu)
  target_include_directories(ds4_moe_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_strata_gpu_include_directories})
  target_link_libraries(ds4_moe_replay PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact
    ${_strata_gpu_runtime_target})
endif()

if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND TARGET strata_kernels)
  # PTQ1_0 ternary on the GPU (iq_mmvq / iq_dequant_f16, type 143) against the CPU kernels; skips without a device
  add_executable(ptq1_0_cuda_test tests/cuda/ptq1_0_cuda_test.cpp)
  target_include_directories(ptq1_0_cuda_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ptq1_0_cuda_test PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  add_test(NAME ptq1_0_cuda_test COMMAND ptq1_0_cuda_test)
endif()

# ---- ds4 reference forward pass (Phase 3) ---------------------------------
# tools/ds4 is not part of the engine, but ds4_ref is the Phase-3 correctness oracle: a plain CPU ggml
# graph that emits golden_dump-format activations for tools/ds4/compare_golden.py.  CPU-only, no CUDA.
if(TARGET ggml-cpu)
  add_executable(ds4_ref tools/ds4/ds4_ref.cpp)
  target_include_directories(ds4_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds4_ref PRIVATE ggml ggml-cpu ggml-base)
endif()

# ---- ds4 decode engine (Phase 4b proper) ---------------------------------
# Each engine slice owns exactly one of these files (conductor plan 2026-10-06); OPTIONAL so a missing one is fine.
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_dense.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_moe.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_engine.cmake OPTIONAL)

# DeepSeek-V4.1 engine (tools/v41/engine)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine/v41_engine.cmake OPTIONAL)

# xolotl-win: ggml-cuda under GGML_STATIC links CUDA::cudart_static, every Strata CUDA target links CUDA::cudart (the
# DLL).  GNU ld tolerates the two runtimes; on Windows the exe then holds two independent CUDA runtimes and the
# second one's lazy init corrupts the driver heap at the first Strata kernel launch.  One runtime: the shared one.
if(WIN32 AND TARGET ggml-cuda)
  get_target_property(_ggml_cuda_libs ggml-cuda LINK_LIBRARIES)
  if(_ggml_cuda_libs)
    list(TRANSFORM _ggml_cuda_libs REPLACE "^CUDA::cudart_static$" "CUDA::cudart")
    set_target_properties(ggml-cuda PROPERTIES LINK_LIBRARIES "${_ggml_cuda_libs}")
  endif()
  get_target_property(_ggml_cuda_ilibs ggml-cuda INTERFACE_LINK_LIBRARIES)
  if(_ggml_cuda_ilibs)
    list(TRANSFORM _ggml_cuda_ilibs REPLACE "cudart_static" "cudart")
    set_target_properties(ggml-cuda PROPERTIES INTERFACE_LINK_LIBRARIES "${_ggml_cuda_ilibs}")
  endif()
  set_property(TARGET ggml-cuda PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()

# xolotl-win: Strata .cu files that include ggml-cuda's common.cuh (moe_mmq.cu's `new ggml_backend_cuda_context`,
# the native_* kernels) must see the same layout ggml-cuda was built with: GGML_CUDA_GRAPHS adds USE_CUDA_GRAPH's
# members to ggml_backend_cuda_context, so a TU without it allocates a smaller struct than ggml-cuda.cu's code writes
# (heap overrun - MSVC's heap catches it at the next check, glibc's lets it pass).
if(GGML_CUDA_GRAPHS AND TARGET ggml-cuda)
  get_property(_xo_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_t IN LISTS _xo_targets)
    get_target_property(_ty ${_t} TYPE)
    if(_ty MATCHES "LIBRARY|EXECUTABLE" AND NOT _ty STREQUAL "INTERFACE_LIBRARY")
      target_compile_definitions(${_t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:GGML_CUDA_USE_GRAPHS>)
    endif()
  endforeach()
endif()
