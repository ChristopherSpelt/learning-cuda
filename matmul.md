# Matrix multiplication

I'm currently learning about CUDA. The canonical example used in learning is of course 
matrix multiplication.

As a guidance I'm using the excellent course [Accelerated Computing](https://accelerated-computing.academy/fall24/). 

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

The GPU we will use is a NVIDIA Tesla T4, because this is freely available through Google Colab.
Specs for the GPU can be found here https://www.techpowerup.com/gpu-specs/tesla-t4.c3316. Relevant for
us are

- Theoretical FLOPS for float32: 8.141 TFLOPS
- DRAM bandwith: 320 GB/s

## Some calculations

Assume for this section that $n = m = k = 3072$. A first question we can ask ourselves is how many
FLOPs are we computing in total. Looking at the formula and pseudocode above we can easily identify
that for each entry of $C$ we need to compute a dot product of a vector of length $m$. Each such a 
dot product hence has $m$ multiplications and $m-1$ additions. Hence in total there will be 
$2m+1$ FLOPs per dot product and $nk$ of such dot products. Thus the total FLOPs for matrix multiplication
will be of order $2nmk$.

Now let's see how many unique memory locations we are accessing. This is easy; we need to access each entry 
of $A$ and $B$ and write it to each element of $C$. Thus there will be a total of $nm+mk+nk$ unique
memory accesses. 


If we only need to access each unique memory location once and we would only be limited by the DRAM
bandwith the fastest we could run is ....
