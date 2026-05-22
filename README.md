# learning-cuda

Notes and kernels written while learning CUDA. Each topic is implemented
in `src/kernels/` and discussed in `docs/`.

## Topics

- [Matrix multiplication](docs/matmul.md) — naive, shared-memory
  tilings, vectorized loads, warp-level reuse, and a comparison to cuBLAS.

## References

- S. Boehm, [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*](https://siboehm.com/articles/22/CUDA-MMM).
- [Accelerated Computing, Fall 2024](https://accelerated-computing.academy/fall24/).
- D. Kirk, W. Hwu, *Programming Massively Parallel Processors*.

## Building

Requires the CUDA Toolkit (with cuBLAS), CMake ≥ 3.25, and a C++20 compiler.

    cmake -S . -B build
    cmake --build build

The default target is `sm_86` (RTX 3090). Override with
`-DCMAKE_CUDA_ARCHITECTURES=<arch>`.
