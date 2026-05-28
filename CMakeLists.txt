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

add_library(cul_base INTERFACE)
target_compile_features(cul_base INTERFACE cxx_std_20 cuda_std_20)
target_include_directories(cul_base INTERFACE include)
target_compile_options(cul_base INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall,-Wextra,-Werror
    -Werror=cross-execution-space-call
    --expt-relaxed-constexpr
    -Xptxas=-warn-spills
    -lineinfo
    >
)

add_library(cul_cublas INTERFACE)
target_link_libraries(cul_cublas INTERFACE CUDA::cublas)

add_executable(gemm
    src/gemm.cu
    src/kernels/gemm/cublas.cu
    src/kernels/gemm/01_naive.cu
    src/kernels/gemm/02_block_tile.cu
    src/kernels/gemm/03_thread_tile_1d.cu
    src/kernels/gemm/04_thread_tile_2d.cu
    src/kernels/gemm/05_vec_loads.cu
    src/kernels/gemm/06_warp_tile.cu
)
target_link_libraries(gemm PRIVATE cul_base cul_cublas)

add_executable(saxpy
    src/saxpy.cu
    src/kernels/saxpy/cublas.cu
    src/kernels/saxpy/01_naive.cu
    src/kernels/saxpy/02_grid_stride.cu
    src/kernels/saxpy/03_vec_loads.cu
)
target_link_libraries(saxpy PRIVATE cul_base cul_cublas)

add_executable(asum
    src/asum.cu
    src/kernels/asum/cublas.cu
    src/kernels/asum/01_naive.cu
    src/kernels/asum/02_block_reduce.cu
    src/kernels/asum/03_grid_stride.cu
    src/kernels/asum/04_warp_reduce.cu
)
target_link_libraries(asum PRIVATE cul_base cul_cublas)
