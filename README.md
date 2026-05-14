# Matrix multiplication

I'm currently learning about CUDA. The canonical example used in learning is of course 
matrix multiplication.

As a guidance I'm using the excellent resources:

- Programming Massively Parallel Processes
- [Accelerated Computing](https://accelerated-computing.academy/fall24/)
- [Boehm's article](https://siboehm.com/articles/22/CUDA-MMM)

## Setup
Assume we have matrices $A$ of size $M \times K$ and $B$ of size $K \times N$ for some positive
integers $M,K$ and $N$. Our task is to matrix multiply $A$ and $B$ to form a new matrix $C$ with size 
$M \times N$; this is done by the formula

$$
C_{i,j} := \sum_{k=0}^{K-1} a_{i,k} b_{k,j}
$$

for all $i \in \{0, ..., M-1\}$ and $j \in \{0, ..., N-1 \}$. The pseudocode for this is as follows.

```
for i = 0 to M-1 do
    for j = 0 to N-1 do
        s ← 0
        for k = 0 to K-1 do
            s ← s + A[i,k] · B[k,j]
        end for
        C[i,j] ← s
    end for
end for
```


## GPU

The GPU we will use is a NVIDIA GeForce RTX 3090. Relevant specs for us are

- Peak float32 throughput: 35.58 TFLOPS
- DRAM bandwidth: 936.2 GB/s

## Some calculations

Assume for this section that $N = M = K = 3072$.

### Compute

Each entry of $C$ is a dot product of vectors of length $K$. A dot product between vectors of
length $K$ takes $K$ multiplies and $K-1$ additions, so $2K - 1$ FLOPs. There are $MN$ such dot
products, so the exact count is $MN(2K - 1)$, which is of order $2MNK$. Plugging in the numbers
above gives

$$
2MNK = 2 \cdot 3072^3 \approx 58 \text{ GFLOP}.
$$

Dividing by the peak rate gives the compute-bound floor:

$$
\frac{58 \text{ GFLOP}}{35{,}580 \text{ GFLOP/s}} \approx 1.63 \text{ ms}.
$$

### Memory

Elements of $A$ and $B$ must be loaded at least once, and elements of $C$ must be stored at least
once. So the minimum number of unique memory accesses is $MK + KN + MN$. Each element is a float32
(4 bytes), so the minimum DRAM traffic is

$$
4(MK + KN + MN) \text{ bytes}.
$$

For our size that is $4 \cdot 3 \cdot 3072^2 \approx 113 \text{ MB}$. Dividing by the DRAM
bandwidth gives the memory-bound floor:

$$
\frac{0.113 \text{ GB}}{936.2 \text{ GB/s}} \approx 0.121 \text{ ms}.
$$


### Which floor binds?

The compute floor is much larger than the memory floor, so for this problem size GEMM is *compute bound* 
on the 3090. The arithmetic intensity of GEMM is

$$
\text{AI} = \frac{2MNK}{4(MK + KN + MN)} \approx 512 \text{ FLOP/byte},
$$

which is well above the 3090's *ridge point* of

$$
\frac{35.58 \text{ TFLOP/s}}{936.2 \text{ GB/s}} \approx 38 \text{ FLOP/byte}.
$$

Any workload with intensity above the ridge is compute-bound; below it, memory-bound.

#### Roofline

![Roofline for RTX 3090](figures/roofline.svg)

The diagonal (slope 1 on log-log axes) is the memory roof; its position
is fixed by DRAM bandwidth. The horizontal dashed line is the peak FP32
throughput. The two meet at the ridge point: arithmetic intensities
above ≈ 38 FLOP/byte are compute-bound, below it memory-bound. GEMM at
3072³ sits at AI ≈ 512, well past the ridge — the target is the
35.58 TFLOP/s ceiling, not the bandwidth diagonal.


## A naive kernel

CUDA kernels are written from the point of view of a single thread. When a kernel is 
launched it does so with many threads. These threads are organized in thread blocks, which 
themselves are organized into a grid. All thread blocks have the same size and shape. It's 
important to note that all threads within a thread block are exectuted in a single SM. This means
threads within a thread block can communicate and synchronize and have access to the same shared 
memory. Within a thread block, threads are grouped into warps of 32. 
The threads of a warp execute in lockstep on the SM, issuing the same instruction each cycle across 
different data — this is the SIMT (Single Instruction, Multiple Threads) execution model.

Let's consider the following naive CUDA kernel to perform GEMM.

```cpp
template <std::uint32_t BLOCKSIZE>
__global__ void naive_kernel(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                             float alpha, const float *__restrict__ A,
                             const float *__restrict__ B, float beta,
                             float* __restrict__ C) {

  const std::uint32_t row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const std::uint32_t col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); 

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
  for (std::uint32_t k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }

  C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
```

The launch code for this kernel is as follows.
```cpp
void kernels::naive(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                    float alpha, const float *A, const float *B, float beta,
                    float *C) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(N, BLOCKSIZE), ceil_div(M, BLOCKSIZE));

  naive_kernel<BLOCKSIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
```
First a note about our choice of grid and thread block dimensions. Intuitively, the threads
tile $C$. The grid is a $(N/\text{BLOCKSIZE}) \times (M/\text{BLOCKSIZE})$ matrix of blocks,
and each block is a $\text{BLOCKSIZE}\times\text{BLOCKSIZE}$ matrix. Consider the following image.

![thread blocks](figures/thread_block.png)

The picture shows the correspondence between the grid/block hierarchy and the global coordinates of $C$.
A thread is uniquely identified by the triple `(blockIdx.y, blockIdx.x, threadIdx.x)` and is responsible
for computing exactly one entrie $C_{i,j}$; the figure shows the mapping between the tripe to $(i,j)$. For
ease of drawing we took $BLOCKSIZE=4$, but one should image it a $32$.

Now consider a single warp: 32 consecutive threads in the block with `threadIdx.x` in $\{0,1,\dots,31\}$.
With $BLOCKSIZE=32$ the row major linearization gives all of them `thread_row = 0` and `thread_col` running
over $\{0,1,/dots,31\}$. So one warp is exactly one row of the thread block. Translated to global coordinates:
the 32 threads of the warp share the same row index $i$, and their column indices $j, j+1, \dots, j+ 31$ are
32 consecutive integers. The data they collectively need is therefore one row of $A$ and 32 consecutive columns
of $B$.

The crucial fact is that a warp executes in lockstep under the SIMT model. At every iteration $k$ of the inner
loop all 32 threads simultaneously execute the same instruction

```cpp
    sum += A[i * K + k] * B[k * N + j];
```

each with its own $j$. 

All 32 threads in a warp compute the same address `A[i * K + k]` since $i$ and $k$ are identical acress the warp. The 
hardware recognizes this and issues a single load. The value is fetched once and broadcasted to all 32
threads.

The 32 threads compute `B[k * N + j], B[k * N + j+1], ..., B[k * N + j + 31]`. These are 32 consecutive 
values in memory. The hardware coalesces these 32 seperate requests into a single transaction. 

