cmake_minimum_required(VERSION 3.20)
project(moe_stream_engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    target_compile_options(moe_engine PRIVATE /O2 /arch:AVX2 /W3)
    target_link_libraries(moe_engine PRIVATE ws2_32 wininet)
else()
    target_compile_options(moe_engine PRIVATE -O3 -mavx2 -mfma -fopenmp -Wall -Wextra)
    if (WIN32)
        target_link_libraries(moe_engine PRIVATE ws2_32 wininet -fopenmp)
    else()
        target_link_libraries(moe_engine PRIVATE pthread -fopenmp)
    endif()
endif()

if (WIN32)
    add_executable(moe_studio WIN32 desktop/src/main_win.cpp)
    target_include_directories(moe_studio PRIVATE desktop/src include)
    target_compile_options(moe_studio PRIVATE -O3 -Wall -Wextra)
    target_link_libraries(moe_studio PRIVATE d2d1 dwrite dwmapi comctl32 ws2_32 ole32 windowscodecs)
endif()


