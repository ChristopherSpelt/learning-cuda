#include "sasum_harness.h"
#include "kernels/sasum.h"

#include <cstdlib>

using namespace cul;

int main() {
  SasumHarness harness(1 << 25);

  constexpr SasumKernel registry[] = {
      {"cuBLAS", &kernels::sasum::cublas},
      {"naive", &kernels::sasum::naive},
      {"block reduce", &kernels::sasum::block_reduce},
      {"grid stride", &kernels::sasum::grid_stride},
      {"warp reduce", &kernels::sasum::warp_reduce},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
