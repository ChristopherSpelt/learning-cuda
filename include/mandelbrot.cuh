#pragma once

__global__ void mandelbrot(uint32_t img_size, uint32_t max_iters,
                           uint32_t *out);

void launch_mandelbrot(uint32_t img_size, uint32_t max_iters, uint32_t *out);
