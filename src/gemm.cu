#include "gemm_cublas.h"
#include "harness.h"
#include "kernels/gemm.h"

#include <cstdlib>

using namespace cul;

int main() {
  GemmHarness harness(4096, 4096, 4096);

  constexpr GemmKernel registry[] = {
      {"cuBLAS (FP32 strict)", &kernels::gemm::cublas_pedantic},
      {"cuBLAS (default)", &kernels::gemm::cublas_default},
      {"cuBLAS (TF32 allowed)", &kernels::gemm::cublas_tf32},
      {"naive", &kernels::gemm::naive},
      {"shared mem", &kernels::gemm::shared_mem},
      {"shared mem 1D block", &kernels::gemm::shared_mem_1d_block},
      {"shared mem 2D block", &kernels::gemm::shared_mem_2d_block},
      {"shared mem vectorized", &kernels::gemm::shared_mem_vec},
      {"shared mem vectorized warptiled", &kernels::gemm::shared_mem_vec_warp},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
