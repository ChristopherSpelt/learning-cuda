#include "cuda_utils.cuh"
#include "gemm.cuh"

namespace gemm_naive {
__global__ void kernel(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                       float alpha, float const *A, float const *B, float beta, float *C) {

    const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

template <int TILE>
void launch(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                       float alpha, float const *A, float const *B, float beta, float *C) {

  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE), 1);

  kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}

template void launch<32>(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                       float alpha, float const *A, float const *B, float beta, float *C);
} // namespace gemm_naive
