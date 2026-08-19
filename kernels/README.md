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

## Results (A100-PCIE-40GB, torch 2.13.0+cu130, triton 3.7.1, 19 Aug 2026)

All four correctness gates passed on the GPU, including triton == rewrite.
Median ms over 100 iterations, CUDA events, 10 warmup. fp32 ran with TF32
off (torch default; the profile shows ampere_sgemm, i.e. true fp32 kernels),
so a TF32 arm is untapped headroom, not part of these numbers.

fp32:

| n | eager | rewrite | compile(eager) | compile(rewrite) | triton |
|---|---|---|---|---|---|
| 1,000     | 0.296  | 0.278 | 0.297  | 0.262 | 0.399 |
| 10,000    | 0.449  | 0.270 | 0.461  | 0.301 | 0.460 |
| 50,000    | 1.296  | **0.308** | 1.313  | 0.474 | 0.337 |
| 200,000   | 4.130  | 0.551 | 4.107  | 1.011 | **0.425** |
| 1,000,000 | 20.217 | 2.053 | 20.148 | 4.346 | **1.445** |

bf16:

| n | eager | rewrite | compile(eager) | compile(rewrite) | triton |
|---|---|---|---|---|---|
| 1,000     | 0.284 | 0.303 | 0.300 | 0.307 | 0.450 |
| 10,000    | 0.286 | 0.309 | 0.341 | 0.344 | 0.464 |
| 50,000    | 0.440 | **0.318** | 0.477 | 0.349 | 0.345 |
| 200,000   | 0.982 | 0.406 | 0.939 | **0.309** | 0.358 |
| 1,000,000 | 4.334 | 1.286 | 4.236 | **0.870** | 0.938 |

### The three findings

1. **torch.compile cannot find the rewrite.** compile(eager) matches eager at
   every size and both dtypes. Reassociating matmuls is an algebraic
   transformation, not a fusion, and Inductor does not attempt it. The
   analysis found a 4-10x (fp32) that the compiler structurally cannot.
2. **The profile confirms the mechanism.** In eager at n = 50k, the (n, 1024)
   GEMM (ampere_sgemm_32x128_tn) is 81% of all CUDA time (30.3 of 37.5 ms
   over the profiled window). The rewrite deletes that tensor: total device
   time drops 7.7x (37.5 -> 4.9 ms); wall time 4.2x at 50k, ~10x at 1M.
3. **The Triton kernel earns its keep at scale, and only there.** It loses at
   small n to launch overhead, matches the rewrite at 50k, and wins from
   ~100k up: at n = 1M fp32 it is 1.42x over the rewrite and 14x over eager.
   In bf16, compile(rewrite) edges it at the largest sizes (0.870 vs 0.938 at
   1M) -- tensor cores shrink every GEMM, so all gaps narrow, and the honest
   summary is: rewrite always, then compile or the kernel depending on dtype
   and size. Reported as measured, per the plan.

### Predictions vs measurements

- Predicted: the layer is launch-bound at small n with a floor set by the op
  chain. Measured: a ~0.27-0.31 ms floor across all fast arms below n = 10k.
- Predicted: the rewrite's win is bounded by the eager tail's share of time.
  Measured: 81% of eager device time was exactly that tail; the wall-clock
  ratio tracks it.
- Predicted: compile may match the hand-written kernel. Measured: true in
  bf16 at large n; false in fp32, where the kernel holds a 1.4x margin at 1M.

## Status

- [x] rewrite implemented and verified locally (CPU, fp32, incl. degenerate blocks)
- [x] Triton kernel written, autotune configs in place
- [x] GPU run: all gates + fp32/bf16 sweeps + profile on A100 (results/ committed)
- [ ] integrate the rewrite into `LanczosSpectralMix.forward` behind a flag
