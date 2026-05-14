#include "harness.h"
#include "kernels.h"

#include <cstdlib>

int main() {
  GemmHarness harness(1024, 1024, 1024);

  constexpr GemmKernel registry[] = {
      {"naive", &kernels::naive},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
