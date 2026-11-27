cmake_minimum_required(VERSION 3.20)
project(moe_stream_engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static CRT on MSVC so the published .exe zip does not require a separate
# Visual C++ Redistributable install for most users.
if (MSVC)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # windows.h defines min/max macros that break std::min / std::max.
    # CRT_SECURE_NO_WARNINGS silences getenv/sscanf deprecation noise.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

include_directories(include)

set(SOURCES
    src/quant.cpp
    src/gguf.cpp
    src/tokenizer.cpp
    src/tier.cpp
    src/pilot_io.cpp
    src/transformer.cpp
    src/diffusion.cpp
    src/model_forge.cpp
    src/model_pull.cpp
    src/vulkan_backend.cpp
    src/server.cpp
    src/main.cpp
)

add_executable(moe_engine ${SOURCES})

if (MSVC)
    # MSVC has no per-function ISA attribute, so the Windows build still
    # requires AVX2 (and FMA/F16C, which Haswell+ CPUs have). The engine
    # reports this via `moe_engine doctor`; on a non-AVX2 machine the
    # binary may fault (SIGILL) — the portable baseline + runtime dispatch
    # in src/quant.cpp applies to the GCC/Clang build below.
    #
    # /openmp unlocks the existing #pragma omp parallel for matmul/attention
    # loops. Without it, Windows builds were effectively single-threaded on
    # the hottest path (GPU is not used — Vulkan is detection-only).
    # Links vcomp*.dll; the packaging script copies it into the release zip
    # when the VS toolchain provides one.
    target_compile_options(moe_engine PRIVATE /O2 /arch:AVX2 /W3 /EHsc /openmp)
    target_link_libraries(moe_engine PRIVATE ws2_32 wininet)
else()
    # Portable baseline: NO -march/-mavx2/-mfma/-mf16c. The SIMD kernels
    # in src/quant.cpp use function-level target("avx2,fma,f16c") /
    # target("avx2,avx512f,avx512bw,avx512vl") attributes plus runtime
    # CPUID + OS-state dispatch, so one binary runs on CPUs without AVX2
    # (scalar fallback) and without AVX-512 (AVX2 kernels). Do NOT add a
    # global -march flag here: it would make the whole binary execute
    # unsupported instructions on older CPUs.
    target_compile_options(moe_engine PRIVATE -O3 -fopenmp -Wall -Wextra)
    if (WIN32)
        target_link_libraries(moe_engine PRIVATE ws2_32 wininet -fopenmp)
    else()
        target_link_libraries(moe_engine PRIVATE pthread -fopenmp)
    endif()
endif()

if (WIN32)
    add_executable(moe_studio WIN32 desktop/src/main_win.cpp)
    target_include_directories(moe_studio PRIVATE desktop/src include)
    if (MSVC)
        target_compile_options(moe_studio PRIVATE /O2 /W3 /EHsc)
    else()
        target_compile_options(moe_studio PRIVATE -O3 -Wall -Wextra)
    endif()
    target_link_libraries(moe_studio PRIVATE d2d1 dwrite dwmapi comctl32 ws2_32 ole32 windowscodecs)
else()
    # Full-screen Linux/macOS terminal client for moe_engine serve.
    # Pure termios + ANSI — no ncurses dependency.
    add_executable(jxai_term desktop/linux/jxai_term.cpp)
    target_include_directories(jxai_term PRIVATE desktop/linux)
    target_compile_options(jxai_term PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(jxai_term PRIVATE pthread)
endif()


