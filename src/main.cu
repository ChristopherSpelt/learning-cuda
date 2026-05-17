#include "harness.h"
#include "kernels.h"

#include <cstdlib>

int main() {
  GemmHarness harness(3072, 3072, 3072);

  constexpr GemmKernel registry[] = {
      {"naive", &kernels::naive},
      {"shared mem", &kernels::shared_mem},
      {"shared mem 1D block", &kernels::shared_mem_1d_block},
      {"shared mem 2D block", &kernels::shared_mem_2d_block},
      {"shared mem vectorized", &kernels::shared_mem_vec},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
