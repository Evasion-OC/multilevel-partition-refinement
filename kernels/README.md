# Making the spectral-mix layer faster: analysis first, kernels second

The global branch of this model is `LanczosSpectralMix` (src/graph_transformer.py): a
learnable spectral convolution on the precomputed low-rank Laplacian eigenbasis. The
eigenpairs come from ARPACK once per graph; this layer is what runs at every layer of
every forward pass, so it is the hot path. This directory makes it faster twice, in
order of how much each step is worth.

## Step 1: the algebraic rewrite (the big win, and no kernel involved)

The eager forward ends with

    Z = V @ S        # (n, d) @ (d, c*m)  -> materialises (n, c*m)
    out = proj(Z)    # Linear(c*m -> c)

with repo defaults d=16, c=128, m=8, so c*m = 1024. By associativity of matmul,

    proj(V @ S) = V @ (S @ W_proj^T) + b

and `S @ W_proj^T` is (d, c) = (16, 128): tiny, and independent of n.

Consequences, counted exactly for the last two steps:

- multiplies drop from n*d*(c*m) + n*(c*m)*c = 147,456*n to d*(c*m)*c + n*d*c =
  2,048*n + 2.1M, a ~72x reduction at any realistic n;
- the (n, 1024) intermediate is never materialised. At n = 50,000 that tensor is
  51.2M elements: ~100 MB in bf16, ~205 MB in fp32, per layer, per forward.

The rewrite is exact in real arithmetic and agrees with the eager path to < 1e-5
relative in fp32 (test_correctness.py), including on degenerate eigenblocks, where
both paths keep the model's basis-invariance property.

Prediction, before benchmarking: after the rewrite the layer is memory-bound on
reading V (n*16) and h (n*128) and writing out (n*128), roughly 5% of the eager
path's traffic. Expected wall-clock gain at n=50k is bounded by the eager path's
share of time in the last two steps; the n-sweep in bench.py measures where the
crossover between launch-bound and bandwidth-bound sits.

## Step 2: the Triton kernel (the remaining margin)

After the rewrite, the n-scale work is two tall-skinny GEMMs (d=16) and a bias add.
cuBLAS handles tall-skinny shapes well but not always optimally at K=16, and the
bias is a separate launch. `_vm_bias_kernel` fuses `out = V @ M + b` into one kernel:
one K-tile (d fits a single BLOCK_D), autotuned over block sizes and warps.

Honest expectation: the kernel's margin over `torch.compile` on the rewrite is
small, and `torch.compile` may match it; both are benchmarked side by side, and
whichever wins, the comparison is reported as measured.

## Files

- `spectral_mix_opt.py` - eager reference, rewrite, Triton path (GPU-gated)
- `test_correctness.py` - CPU-runnable gates + GPU gate for the kernel
- `bench.py`            - n-sweep {1k, 10k, 50k, 200k, 1M}, 5 arms, CUDA events,
                          median/IQR over 100 iters, CSV + meta

## Status

- [x] rewrite implemented and verified locally (CPU, fp32, incl. degenerate blocks)
- [x] Triton kernel written, autotune configs in place
- [ ] GPU run: correctness gate + sweep on Colab T4, results committed here
- [ ] integrate the rewrite into `LanczosSpectralMix.forward` behind a flag once measured
