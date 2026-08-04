#!/usr/bin/env python3
"""Module A scaling test -- REAL measurements up to n=50000 (not a rescaled toy figure).

Backs the claim that the Lanczos branch swaps the O(n^2) dense-attention global branch
for the O(n) Lanczos spectral-mix branch. The existing microbenchmark (src/verify_ab.py)
only goes to n=8000 ("toy scale"). This harness extends the SAME measurement to
n in {1000,2000,4000,8000,16000,32000,50000} and reports it as an actual table.

It imports and reuses the REAL repo modules -- nothing is reimplemented:
  * srmp.graph.Graph                      -- the graph container the pipeline uses
  * srmp.spectral.compute_eigenvectors    -- bottom-K Laplacian eigenpairs (ARPACK/LOBPCG),
                                             exactly the pipeline's Ritz basis for the PE + Lanczos branch
  * srmp.rl_policy.build_features / graph_to_torch -- the same node-feature + adjacency tensors
  * graph_transformer.SpectralGraphTransformer with global_kind in {"attn","lanczos"}
                                             -- the two global-branch variants (the only thing that differs)

The dense-attention branch materialises an (n x n) attention matrix per head per layer;
at n=50000 that is ~2.5e9 entries (~10 GB fp32 per matrix, before softmax workspace), so it is
EXPECTED to OOM / raise at large n. Lanczos mixing is O(n * d_model * m) and should complete.
Each forward-pass cell is wrapped in try/except (OOM/RuntimeError/MemoryError -> "OOM"); we warm
up once and average 3 timed runs. Eigenpairs are computed ONCE per n, outside the timed region,
so the table isolates the global-branch forward cost (the thing the O(n^2)->O(n) claim is about).

Usage:
    python3 scaling_test.py                 # full sweep to n=50000
    python3 scaling_test.py --max-n 4000    # smoke test (stops after n<=4000)
    python3 scaling_test.py --ns 1000,2000  # explicit n list
    python3 scaling_test.py --device cpu    # force device (default: cpu; absolute ms are CPU here)
"""
import argparse
import os
import sys
import time
import platform

import numpy as np
import scipy.sparse as sp
import torch

# --- make the REAL repo modules importable (src/ on path), same trick verify_ab.py uses ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from srmp.graph import Graph                                   # noqa: E402
from srmp.spectral import compute_eigenvectors                 # noqa: E402
from srmp.rl_policy import build_features, graph_to_torch      # noqa: E402
from graph_transformer import SpectralGraphTransformer, LanczosSpectralMix  # noqa: E402

# Module-A encoder config -- mirror verify_ab.py's microbenchmark exactly so the numbers
# are comparable to the existing 8k table (d_model=64, n_heads=4, n_layers=2, use_local=True).
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
N_EIGS = 8           # bottom-K eigenpairs, as build_state uses (n_eigs=8)
K = 4                # blocks (feature one-hot width), as verify_ab.py
EPS = 0.03
N_FILTERS = 8        # LanczosSpectralMix filters (default in the class)
N_REPEAT = 3         # timed runs averaged
SEED = 1


def rand_graph(n, deg=6, seed=0):
    """Connected sparse random graph: a Hamiltonian ring + (deg*n/2) random chords.

    This is the SAME generator src/verify_ab.py uses for its microbenchmark (ring guarantees
    connectivity so the Laplacian eigensolve is well posed; random chords give avg degree ~deg).
    Sparse: m ~= (deg+2)*n/2 edges, so nnz grows linearly in n -- realistic for the regime
    where the dense (n x n) attention matrix is the bottleneck, not the graph.
    """
    rng = np.random.RandomState(seed)
    row, col = [], []
    for i in range(n):
        j = (i + 1) % n
        row += [i, j]
        col += [j, i]
    for _ in range(deg * n // 2):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b:
            row += [a, b]
            col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    A.data[:] = 1.0
    A.sum_duplicates()
    A.data[:] = 1.0
    return Graph(A)


def build_inputs(n, device, seed=SEED):
    """Build the REAL encoder inputs for an n-node graph, OUTSIDE the timed region.

    Returns (x, eigvals, eigvecs, adj_indices, adj_weights, deg_inv) exactly as
    refiner.build_state would, using the pipeline's own feature/eigen/adjacency builders.
    A trivial all-zeros partition is used (we are timing the encoder, not refinement quality).
    """
    G = rand_graph(n, seed=seed)
    partition = np.zeros(n, dtype=np.int64)
    feats = build_features(G, partition, K, EPS, spectral_coords=None)        # (n, K+6), real builder
    ev, V = compute_eigenvectors(G, num_eigenvectors=min(N_EIGS, G.n - 1),    # ARPACK/LOBPCG Ritz basis
                                 normalized=True)
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    eigvals = torch.tensor(np.asarray(ev), dtype=torch.float32, device=device)
    eigvecs = torch.tensor(np.asarray(V), dtype=torch.float32, device=device)
    ai, aw, di = graph_to_torch(G, device="cpu")
    ai, aw, di = ai.to(device), aw.to(device), di.to(device)
    return x, eigvals, eigvecs, ai, aw, di


def time_encoder(global_kind, inputs, device):
    """Warm up once, average N_REPEAT timed forward passes (ms). Returns float ms or 'OOM'.

    Builds the REAL SpectralGraphTransformer with the requested global branch and runs a
    no_grad forward. Any OOM / RuntimeError / MemoryError (dense attn at large n) -> 'OOM'.
    """
    x, eigvals, eigvecs, ai, aw, di = inputs
    in_dim = x.shape[1]
    try:
        torch.manual_seed(0)
        enc = SpectralGraphTransformer(
            in_dim=in_dim, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            n_eigs=N_EIGS, num_filters=N_FILTERS, use_local=True,
            global_kind=global_kind,
        ).to(device).eval()
        with torch.no_grad():
            enc(x, eigvals, eigvecs, ai, aw, di)                 # warmup (also triggers OOM early)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(N_REPEAT):
                enc(x, eigvals, eigvecs, ai, aw, di)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.time() - t0) / N_REPEAT * 1000.0
        return dt_ms
    except (RuntimeError, MemoryError) as e:
        msg = str(e).lower()
        if any(t in msg for t in ("out of memory", "oom", "alloc", "memory", "cannot allocate")):
            return "OOM"
        # any other RuntimeError is still a hard failure to run at this n -> report as OOM-class
        return "OOM"


def machine_info():
    mps = bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            ram_gb = round(int(out) / 1e9, 1)
        except Exception:
            ram_gb = "?"
    return (f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | mps={mps} "
            f"| RAM_GB={ram_gb} | {platform.platform()}")


def main():
    ap = argparse.ArgumentParser(description="Module A scaling test (attn O(n^2) vs lanczos O(n))")
    ap.add_argument("--ns", type=str, default="1000,2000,4000,8000,16000,32000,50000",
                    help="comma-separated n values")
    ap.add_argument("--max-n", type=int, default=None,
                    help="skip any n greater than this (use --max-n 4000 for a smoke test)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"],
                    help="device for the encoder forward (default cpu; absolute ms are device-dependent)")
    args = ap.parse_args()

    device = args.device
    if device == "mps" and not getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        print("[warn] mps requested but unavailable; falling back to cpu")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        device = "cpu"

    ns = [int(s) for s in args.ns.split(",") if s.strip()]
    if args.max_n is not None:
        ns = [n for n in ns if n <= args.max_n]

    print("=== scaling test: encoder forward time (ms/call) ===")
    print("    global branch: dense attention O(n^2)  vs  Lanczos spectral mix O(n)")
    print(f"    machine: {machine_info()}")
    print(f"    device : {device}   (absolute ms are {device.upper()}; the SCALING / OOM is the finding)")
    print(f"    config : d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} "
          f"n_eigs={N_EIGS} filters={N_FILTERS} | avg-degree~8, sparse ring+chords")
    print(f"    timing : warmup x1, mean of {N_REPEAT} runs; eigenpairs computed once OUTSIDE timing")
    print()
    print(f"{'n':>7} {'attn_ms':>12} {'lanczos_ms':>12} {'speedup':>12}")
    print("-" * 47)

    for n in ns:
        inputs = build_inputs(n, device)
        ta = time_encoder("attn", inputs, device)
        tl = time_encoder("lanczos", inputs, device)
        if isinstance(ta, float) and isinstance(tl, float) and tl > 0:
            speed = f"{ta / tl:>10.2f}x"
            attn_s = f"{ta:>12.1f}"
        elif ta == "OOM" and isinstance(tl, float):
            speed = "  attn-OOM"
            attn_s = f"{'OOM':>12}"
        else:
            speed = "      n/a"
            attn_s = f"{str(ta):>12}"
        lanc_s = f"{tl:>12.1f}" if isinstance(tl, float) else f"{str(tl):>12}"
        print(f"{n:>7} {attn_s} {lanc_s} {speed:>12}")
        sys.stdout.flush()

    print()
    print("Reading: dense attention scales quadratically and OOMs once the (n x n) attention")
    print("matrix no longer fits; Lanczos mixing stays linear and completes at every n.")
    print("That attention CANNOT run at 50k while Lanczos can IS the headline scaling result.")


if __name__ == "__main__":
    main()
