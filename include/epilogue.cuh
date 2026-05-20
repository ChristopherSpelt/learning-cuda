#pragma once

#include <cuda_runtime.h>

namespace cul::epilogue {

template <bool BetaIsZero>
__device__ __forceinline__ void store_result(float *destination, float product,
                                             [[maybe_unused]] float beta) {
  if constexpr (BetaIsZero) {
    *destination = product;
  } else {
    *destination = product + beta * (*destination);
  }
}

template <bool BetaIsZero>
__device__ __forceinline__ void
store_result(float4 *destination, float4 product, [[maybe_unused]] float beta) {
  if constexpr (BetaIsZero) {
    *destination = product;
  } else {
    float4 c = *destination;
    c.x = product.x + beta * c.x;
    c.y = product.y + beta * c.y;
    c.z = product.z + beta * c.z;
    c.w = product.w + beta * c.w;
    *destination = c;
  }
}
} // namespace cul::epilogue
