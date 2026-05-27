#include "asum_harness.h"
#include "kernels/asum.h"

#include <cstdlib>

using namespace cul;

int main() {
  AsumHarness harness(1 << 25);

  constexpr AsumKernel registry[] = {
      {"cuBLAS", &kernels::asum::cublas},
      {"naive", &kernels::asum::naive},
      {"shared mem", &kernels::asum::shared_mem},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
