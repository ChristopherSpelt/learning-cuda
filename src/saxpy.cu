#include "kernels/saxpy.h"
#include "saxpy_harness.h"

#include <cstdlib>

using namespace cul;

int main() {
  SaxpyHarness harness(1<<25);

  constexpr SaxpyKernel registry[] = {
      {"cuBLAS", &kernels::saxpy::cublas},
      {"naive",  &kernels::saxpy::naive},
      {"vec_loads",  &kernels::saxpy::vec_loads},
  };

  for (const auto &k : registry) {
    harness.run(k);
  }

  return EXIT_SUCCESS;
}
