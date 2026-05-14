cmake_minimum_required(VERSION 3.18)

set(TARGET cuda_learning)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(${TARGET} C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCES
    src/main.cu
    src/gemm_cpu.cpp
)

set(KERNELS
    src/kernels/naive.cu
)

add_executable(${TARGET})

target_sources(${TARGET} PRIVATE 
    ${SOURCES}
    ${KERNELS}
)

target_include_directories(${TARGET} PRIVATE include)

set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(${TARGET}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra -lineinfo>
)
