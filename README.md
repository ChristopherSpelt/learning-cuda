# learning-cuda

Notes and kernels written while learning CUDA. Each topic walks a progression
from a naive baseline through successive optimizations. The kernels are in
`src/kernels/`, writeups in `docs/`.

## Topics

- [saxpy (αx + y)](docs/saxpy.md): naive, grid-stride loop,
  vectorized loads, and a comparison to cuBLAS.
- [Sum of elements (asum)](docs/asum.md): naive, block reduction,
  grid-stride loop, warp reduction, and a comparison to cuBLAS.
- [Matrix multiplication (gemm)](docs/gemm.md): naive, block tiling,
  1D and 2D thread tiling, vectorized loads, warp tiling, and a comparison to cuBLAS.

## References

- D. Kirk, W. Hwu, *Programming Massively Parallel Processors*.
- [Accelerated Computing, Fall 2024](https://accelerated-computing.academy/fall24/).
- S. Boehm, [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*](https://siboehm.com/articles/22/CUDA-MMM).
- B. Crovella, [*Atomics, Reductions, Warp Shuffle*](https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf).

## Building

Requires the CUDA Toolkit (with cuBLAS), CMake ≥ 3.25, and a C++20 compiler.

    cmake -S . -B build
    cmake --build build

The default target is `sm_86` (RTX 3090). Override with
`-DCMAKE_CUDA_ARCHITECTURES=<arch>`.

## Running

Run any of the three executables to see the benchmark output:

    ./build/saxpy
    ./build/asum
    ./build/gemm
