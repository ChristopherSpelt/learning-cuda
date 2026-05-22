cmake_minimum_required(VERSION 3.25)

project(learning_cuda LANGUAGES CXX CUDA)

# Default to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Default to sm_86 (RTX 3090). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cu
    src/kernels/cublas.cu
    src/kernels/naive.cu
    src/kernels/shared_mem.cu
    src/kernels/shared_mem_1d_block.cu
    src/kernels/shared_mem_2d_block.cu
    src/kernels/shared_mem_vec.cu
    src/kernels/shared_mem_vec_warp.cu
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20 cuda_std_20)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)

target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall,-Wextra,-Werror
    -Werror=cross-execution-space-call
    --expt-relaxed-constexpr
    -Xptxas=-warn-spills
    -lineinfo
    >
)
