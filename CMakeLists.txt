cmake_minimum_required(VERSION 3.25)

# Default to the GPU in this machine. 
# Override with -DCMAKE_CUDA_ARCHITECTURES=86 (RTX 3090) or 121 (GB10).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(learning_cuda LANGUAGES CXX CUDA)

# clangd cannot read nvcc respone files; keep =I flags inline in compile_commands.json
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Default to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
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
    -Xptxas=-warn-spills,-warn-lmem-usage,-Werror
    -lineinfo
    >
)

add_library(cul_cublas INTERFACE)
target_link_libraries(cul_cublas INTERFACE CUDA::cublas)

add_executable(hgemm
    src/hgemm.cu
    src/kernels/hgemm/cublas.cu
)
target_link_libraries(hgemm PRIVATE cul_base cul_cublas)

add_executable(sgemm
    src/sgemm.cu
    src/kernels/sgemm/cublas.cu
    src/kernels/sgemm/01_naive.cu
    src/kernels/sgemm/02_block_tile.cu
    src/kernels/sgemm/03_thread_tile_1d.cu
    src/kernels/sgemm/04_thread_tile_2d.cu
    src/kernels/sgemm/05_vec_loads.cu
    src/kernels/sgemm/06_warp_tile.cu
    src/kernels/sgemm/07_resolve_bank.cu
    src/kernels/sgemm/08_double_buffer.cu
    src/kernels/sgemm/09_inner_loop_prefetch.cu
    src/kernels/sgemm/10_double_buffer_prefetch.cu
    src/kernels/sgemm/11_double_buffer_prefetch_bank.cu
)
target_link_libraries(sgemm PRIVATE cul_base cul_cublas)

add_executable(saxpy
    src/saxpy.cu
    src/kernels/saxpy/cublas.cu
    src/kernels/saxpy/01_naive.cu
    src/kernels/saxpy/02_grid_stride.cu
    src/kernels/saxpy/03_vec_loads.cu
)
target_link_libraries(saxpy PRIVATE cul_base cul_cublas)

add_executable(sasum
    src/sasum.cu
    src/kernels/sasum/cublas.cu
    src/kernels/sasum/01_naive.cu
    src/kernels/sasum/02_block_reduce.cu
    src/kernels/sasum/03_grid_stride.cu
    src/kernels/sasum/04_warp_reduce.cu
)
target_link_libraries(sasum PRIVATE cul_base cul_cublas)
