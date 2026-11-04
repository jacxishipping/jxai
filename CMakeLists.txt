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
    src/vulkan_backend.cpp
    src/server.cpp
    src/main.cpp
)

add_executable(moe_engine ${SOURCES})

if (MSVC)
    target_compile_options(moe_engine PRIVATE /O2 /arch:AVX2 /W3)
    target_link_libraries(moe_engine PRIVATE ws2_32)
else()
    target_compile_options(moe_engine PRIVATE -O3 -mavx2 -mfma -fopenmp -Wall -Wextra)
    if (WIN32)
        target_link_libraries(moe_engine PRIVATE ws2_32 -fopenmp)
    else()
        target_link_libraries(moe_engine PRIVATE pthread -fopenmp)
    endif()
endif()

