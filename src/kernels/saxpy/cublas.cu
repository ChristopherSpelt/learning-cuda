#include "cublas_utils.cuh"
#include "saxpy_cublas.h"

#include <cublas_v2.h>

namespace cul {

void kernels::saxpy::cublas(const SaxpyArgs &a) {
  CUBLAS_CHECK(cublasSaxpy(cublas_utils::handle(), a.n, &a.alpha, a.x, 1, a.y, 1));
}

} // namespace cul
