#include "gemm_harness.h"
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
      {"block tile", &kernels::gemm::block_tile},
      {"thread tile 1D", &kernels::gemm::thread_tile_1d},
      {"thread tile 2D", &kernels::gemm::thread_tile_2d},
      {"vec loads", &kernels::gemm::vec_loads},
      {"warp tile", &kernels::gemm::warp_tile},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
