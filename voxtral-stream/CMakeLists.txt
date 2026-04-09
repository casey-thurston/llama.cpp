cmake_minimum_required(VERSION 3.14)
project(voxtral-stream C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build ggml from the sibling tree in our llama.cpp fork.
# This is the same ggml that the rest of llama.cpp uses; we just consume it
# directly without going through the llama_model / llama_context layer.
if(NOT TARGET ggml)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ggml ${CMAKE_BINARY_DIR}/ggml-build)
endif()

add_executable(voxtral-stream
    main.c
    voxtral_tokenizer.c
    voxtral_audio.c
    voxtral_safetensors.c
)

target_link_libraries(voxtral-stream PRIVATE ggml m)
target_include_directories(voxtral-stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# voxtral_audio.c uses fmaxf/expf/log10f/etc; libm is needed.
# Vendored sources are pure C99 plus POSIX (mmap) — no extra flags needed.
