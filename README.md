# Matrix multiplication

I'm currently learning about CUDA. The canonical example used in learning is of course 
matrix multiplication.

As a guidance I'm using the excellent resources:

- Programming Massively Parallel Processes
- [Accelerated Computing](https://accelerated-computing.academy/fall24/)
- [Boehm's article](https://siboehm.com/articles/22/CUDA-MMM)

## Setup
Assume we have matrices $A$ of size $n \times m$ and $B$ of size $m \times k$ for some positive
integers $n,m$ and $k$. Our task is to matrix multiply $A$ and $B$ to form a new matrix $C$ with size 
$n \times k$; this is done by the formula

$$
C_{ij} := \sum_{\ell=0}^{m-1} a_{i\ell} b_{\ell,j}
$$

for all $i \in \{0, ..., n-1\}$ and $j \in \{0, ..., k-1 \}$. The pseudocode for this is as follows.

```
for i = 0 to n-1 do
    for j = 0 to k-1 do
        s ← 0
        for ℓ = 0 to m-1 do
            s ← s + A[i,ℓ] · B[ℓ,j]
        end for
        C[i,j] ← s
    end for
end for
```


## GPU

The GPU we will use is a NVIDIA GeForce RTX 3090. Relevant specs for us are

- Peak float32 throughput: 35.58 TFLOPS
- DRAM bandwith: 936.2 GB/s

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


## Which floor binds?

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

## Roofline

![Roofline for RTX 3090](figures/roofline.svg)

The diagonal (slope 1 on log-log axes) is the memory roof; its position
is fixed by DRAM bandwidth. The horizontal dashed line is the peak FP32
throughput. The two meet at the ridge point: arithmetic intensities
above ≈ 38 FLOP/byte are compute-bound, below it memory-bound. GEMM at
3072³ sits at AI ≈ 512, well past the ridge — the target is the
35.58 TFLOP/s ceiling, not the bandwidth diagonal.
