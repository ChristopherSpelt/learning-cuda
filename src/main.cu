#include "harness.h"
#include "kernels.h"

#include <cstdlib>

int main() {
  GemmHarness harness(3072, 3072, 3072);

  constexpr GemmKernel registry[] = {
      {"naive", &kernels::naive},
      {"shared mem", &kernels::shared_mem},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
