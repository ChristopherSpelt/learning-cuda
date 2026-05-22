#pragma once

#include <cuda_runtime.h>

namespace cul::loads {

template <bool BoundsCheck>
__device__ __forceinline__ float4 masked_load_f4(const float *src,
                                                 [[maybe_unused]] bool in_bounds) {
  if constexpr (BoundsCheck) {
    return in_bounds ? reinterpret_cast<const float4 *>(src)[0] : float4{0.0f, 0.0f, 0.0f, 0.0f};
  } else {
    return reinterpret_cast<const float4 *>(src)[0];
  }
}
} // namespace cul::loads
