#include "cuda_utils.cuh"
#include "gemm.cuh"
#include "gemm_cpu.h"
#include "io.h"
#include "mandelbrot.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

void compute_mandelbrot(uint32_t img_size, uint32_t max_iters,
                        std::string_view filename) {
  // Allocate on host
  std::vector<uint32_t> result(img_size * img_size);

  // Allocate on device
  thrust::device_vector<uint32_t> resultd(result.begin(), result.end());
  uint32_t *resultd_ptr = thrust::raw_pointer_cast(resultd.data());

  // Computation on GPU
  launch_mandelbrot(img_size, max_iters, resultd_ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy device to host
  thrust::copy(resultd.begin(), resultd.end(), result.begin());

  // Write output to binary file
  write_binary(filename, result);
}


struct GemmNaive {
    static constexpr std::string_view name = "gemm_naive";

    static void launch(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                        float alpha, const float* A, const float* B,
                        float beta, float* C) {
        gemm_naive::launch<32>(M, N, K, alpha, A, B, beta, C);
    }

};


template <typename Impl>
void run_gemm(std::uint32_t M, std::uint32_t N, std::uint32_t K) {
  // Allocation on host
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.5f;
  constexpr float tolerance = 1e-5f;

  auto A = numerics::random_matrix(M, K, 0);
  auto B = numerics::random_matrix(K, N, 1);
  auto C = numerics::random_matrix(M, N, 2);

  // CPU reference computation
  auto C_ref = C;
  gemm_cpu(M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());

  // Allocation on device
  thrust::device_vector<float> Ad(A.begin(), A.end());
  thrust::device_vector<float> Bd(B.begin(), B.end());
  thrust::device_vector<float> Cd(C.begin(), C.end());
  const auto Ad_ptr = thrust::raw_pointer_cast(Ad.data());
  const auto Bd_ptr = thrust::raw_pointer_cast(Bd.data());
  const auto Cd_ptr = thrust::raw_pointer_cast(Cd.data());

  // Computation on device
  auto launch = [&] {
      Impl::launch(M, N, K, alpha, Ad_ptr, Bd_ptr, beta, Cd_ptr);
  };

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy device to host
  std::vector<float> C_gpu(std::size_t(M) * N);
  thrust::copy(Cd.begin(), Cd.end(), C_gpu.begin());


  const float rmse = numerics::relative_rmse(C_gpu, C_ref);
  std::printf("%-20.*s  rel RMSE %.2e\n",
                int(Impl::name.size()), Impl::name.data(), rmse);
  if (rmse > tolerance) {
        std::printf("  FAILED — skipping benchmark\n\n");
        return;
    }

  // Important: subsequent kernel calls accumulate into C (beta=0.5).
    // For benchmarking, this distorts results after the first call. Reset
    // dC to its initial state before each benchmark batch by re-uploading.
    // (We're measuring kernel time, not accumulated correctness.)
    thrust::copy(hC_init.begin(), hC_init.end(), dC.begin());

    const auto result = bench::run(launch, bench::gemm_flops(M, N, K));
    bench::print_result(Impl::name, result);
    std::printf("\n");
}

int main() {
  constexpr std::uint32_t M = 1024, N = 1024, K = 1024;

  run_gemm<GemmNaive>(M, N, K);

  return EXIT_SUCCESS;
}
