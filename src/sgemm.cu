#include "sgemm_harness.h"
#include "kernels/sgemm.h"

#include <cstdlib>

using namespace cul;

int main() {
  SgemmHarness harness(4096, 4096, 4096);

  constexpr SgemmKernel registry[] = {
      {"cuBLAS (FP32 strict)", &kernels::sgemm::cublas_pedantic},
      {"cuBLAS (default)", &kernels::sgemm::cublas_default},
      {"cuBLAS (TF32 allowed)", &kernels::sgemm::cublas_tf32},
      {"naive", &kernels::sgemm::naive},
      {"block tile", &kernels::sgemm::block_tile},
      {"thread tile 1D", &kernels::sgemm::thread_tile_1d},
      {"thread tile 2D", &kernels::sgemm::thread_tile_2d},
      {"vec loads", &kernels::sgemm::vec_loads},
      {"warp tile", &kernels::sgemm::warp_tile},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
