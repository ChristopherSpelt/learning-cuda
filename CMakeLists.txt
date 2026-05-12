cmake_minimum_required(VERSION 3.18)

project(cuda_learning C CXX CUDA)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(SOURCES
    src/main.cu
)

set(KERNELS
    src/kernels/mandelbrot.cu
    src/kernels/matmul.cu
)

add_executable(${CMAKE_PROJECT_NAME})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE 
    ${SOURCES}
    ${KERNELS}
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra -lineinfo>
)
