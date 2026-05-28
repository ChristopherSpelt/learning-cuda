#include "cublas_utils.cuh"
#include "kernels/sasum.h"

#include <cublas_v2.h>

namespace cul {

void kernels::sasum::cublas(const SasumArgs &a) {
  auto h = cublas_utils::handle();
  cublas_utils::ScopedPointerMode mode{h, CUBLAS_POINTER_MODE_DEVICE};
  CUBLAS_CHECK(cublasSasum(h, a.n, a.x, 1, a.result));
}

} // namespace cul
