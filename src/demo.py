#!/usr/bin/env python3
"""Live demo: the three claims that can be shown in under two minutes.

The benchmark itself is an HPC artefact and cannot be reproduced live (roughly
four minutes per graph per seed on a laptop). What CAN be shown live is the
part that is a property of the code rather than of the compute budget:

  (1) the deployed checkpoint loads and reports its own parameter count;
  (2) the encoder's three invariances hold end to end, to ~1e-6, measured now
      rather than quoted -- permutation of the vertices, sign flip of the
      eigenvectors, and rotation inside a degenerate eigenspace;
  (3) the global branch is linear where dense attention is quadratic, on the
      same encoder object the pipeline uses.

Everything below imports the real repository modules. Nothing is reimplemented
for the demo, which is the point: a marker can follow any number here back into
the code that produced the thesis.

    python src/demo.py                # ~90 s
    python src/demo.py --quick        # ~35 s, smaller n
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srmp.graph import Graph                                  # noqa: E402
from srmp.spectral import compute_eigenvectors                # noqa: E402
from graph_transformer import SpectralGraphTransformer        # noqa: E402
from refiner import build_state                              # noqa: E402

RULE = "=" * 66


def rand_graph(n, deg=6, seed=0):
    """A connected random graph, the same construction verify_ab.py uses."""
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
    A.sum_duplicates()
    A.data[:] = 1.0
    return Graph(A)


def find_checkpoint():
    """A checkpoint from models/, so the demo runs without --model.
    Prefers a k=4 checkpoint, which is the configuration Chapter 5 reports."""
    import glob as _glob
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = sorted(_glob.glob(os.path.join(here, "models", "*.pt"))) or \
        sorted(_glob.glob(os.path.expanduser("~/Downloads/Models/*.pt")))
    k4 = [h for h in hits if "k4" in os.path.basename(h)]
    return (k4 or hits or [""])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="trained checkpoint; defaults to the first .pt in ~/Downloads/Models")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.model is None:
        a.model = find_checkpoint()
    k, n_eigs = 4, 8
    torch.manual_seed(0)

    # ---------------- (1) the deployed checkpoint ----------------
    print(RULE)
    print(" (1) THE DEPLOYED CHECKPOINT")
    print(RULE)
    if os.path.exists(a.model):
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        sd = ck.get("model_state", ck.get("policy", ck))
        n_par = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        print(f"  file            {os.path.basename(a.model)}")
        print(f"  parameters      {n_par:,}")
        for key in ("k", "epsilon", "global_kind", "pe_kind", "d_model"):
            if isinstance(ck, dict) and key in ck:
                print(f"  {key:15s} {ck[key]}")
        print("  -> the deployed checkpoint, self-reporting the configuration it was\n"
              "     trained under; k=4 is the one Chapter 5 reports.")
    else:
        print(f"  (checkpoint not found at {a.model}; skipping)")

    # ---------------- (2) the invariances, measured live ----------------
    print()
    print(RULE)
    print(" (2) THE THREE INVARIANCES, MEASURED NOW (not quoted)")
    print(RULE)
    n = 300 if a.quick else 600
    G = rand_graph(n, seed=3)
    ev, V = compute_eigenvectors(G, num_eigenvectors=n_eigs, normalized=True)
    pi = np.zeros(G.n, dtype=np.int64)
    x, ev_t, V_t, _, ai, aw, di = build_state(G, pi, k, n_eigs=n_eigs)

    enc = SpectralGraphTransformer(in_dim=x.shape[1], d_model=64, n_heads=4,
                                   n_layers=2, use_local=True,
                                   global_kind="lanczos").eval()
    with torch.no_grad():
        H0 = enc(x, ev_t, V_t, ai, aw, di)

        # (a) eigenvector SIGN flip: V -> V S, S = diag(+-1)
        signs = torch.tensor(np.random.RandomState(0).choice([-1.0, 1.0], n_eigs),
                             dtype=V_t.dtype)
        H_sign = enc(x, ev_t, V_t * signs, ai, aw, di)
        d_sign = (H0 - H_sign).abs().max().item()

        # (b) BASIS rotation inside a genuinely DEGENERATE eigenspace.
        #     A random graph has generically distinct eigenvalues, so it has no
        #     degenerate subspace and rotating any pair is NOT a symmetry. The
        #     cycle C_n does: its normalised-Laplacian eigenvalues 1 - cos(2 pi j / n)
        #     are exactly doubled, with a sine/cosine pair spanning each eigenspace.
        ring = sp.csr_matrix((np.ones(2 * n),
                              (np.arange(n).repeat(2),
                               np.stack([(np.arange(n) + 1) % n,
                                         (np.arange(n) - 1) % n], 1).ravel())),
                             shape=(n, n))
        Gc = Graph(ring)
        xc, evc, Vc, _, aic, awc, dic = build_state(
            Gc, np.zeros(n, dtype=np.int64), k, n_eigs=n_eigs)
        gaps = (evc[1:] - evc[:-1]).abs()
        j = int(torch.argmin(gaps))                   # the tightest pair
        gap = float(gaps[j])
        Hc = enc(xc, evc, Vc, aic, awc, dic)
        Vr = Vc.clone()
        th = 0.7853981633974483                       # pi/4
        c, s = float(np.cos(th)), float(np.sin(th))
        Vr[:, j], Vr[:, j + 1] = (c * Vc[:, j] - s * Vc[:, j + 1],
                                  s * Vc[:, j] + c * Vc[:, j + 1])
        H_basis = enc(xc, evc, Vr, aic, awc, dic)
        d_basis = (Hc - H_basis).abs().max().item()

        # (c) vertex PERMUTATION: relabel the graph, encode, unpermute
        perm = np.random.RandomState(1).permutation(G.n)
        P = sp.csr_matrix((np.ones(G.n), (np.arange(G.n), perm)), shape=(G.n, G.n))
        Gp = Graph((P @ G.adj @ P.T).tocsr())
        xp, evp, Vp, _, aip, awp, dip = build_state(Gp, pi[perm], k, n_eigs=n_eigs)
        Hp = enc(xp, evp, Vp, aip, awp, dip)
        inv = np.argsort(perm)
        d_perm = (H0 - Hp[torch.as_tensor(inv)]).abs().max().item()

    print(f"  sign / permutation on a random graph: n = {G.n}, m = {G.adj.nnz // 2}")
    print(f"  basis rotation on the cycle C_{n}, whose eigenvalues are exactly")
    print(f"  doubled: the rotated pair has eigenvalue gap {gap:.2e}")
    print()
    print(f"  {'symmetry':34s} {'max |dH|':>12}")
    print(f"  {'-' * 34} {'-' * 12}")
    print(f"  {'eigenvector sign flip  V -> VS':34s} {d_sign:12.2e}")
    print(f"  {'basis rotation, degenerate pair':34s} {d_basis:12.2e}")
    print(f"  {'vertex permutation  G -> PGP^T':34s} {d_perm:12.2e}")
    worst = max(d_sign, d_basis, d_perm)
    print()
    print(f"  worst case {worst:.2e} -- the thesis claims ~1e-6 end to end.")
    print("  These are exact symmetries of the construction, not fitted behaviour:")
    print("  the filters see only eigenVALUES, and eigenvectors enter only through")
    print("  the eigenprojector V diag(.) V^T, which is invariant to both.")

    # ---------------- (3) linear vs quadratic global branch ----------------
    print()
    print(RULE)
    print(" (3) GLOBAL BRANCH COST: LANCZOS O(n) vs DENSE ATTENTION O(n^2)")
    print(RULE)
    ns = [500, 1000, 2000] if a.quick else [500, 1000, 2000, 4000]
    print(f"  {'n':>6} {'attention ms':>13} {'lanczos ms':>12} {'speedup':>9}")
    print(f"  {'-' * 6} {'-' * 13} {'-' * 12} {'-' * 9}")
    for nn in ns:
        Gn = rand_graph(nn, seed=1)
        xn, evn, Vn, _, ain, awn, din = build_state(
            Gn, np.zeros(nn, dtype=np.int64), k, n_eigs=n_eigs)

        def ms(kind):
            try:
                e = SpectralGraphTransformer(in_dim=xn.shape[1], d_model=64,
                                             n_heads=4, n_layers=2,
                                             use_local=True,
                                             global_kind=kind).eval()
                with torch.no_grad():
                    e(xn, evn, Vn, ain, awn, din)             # warm up
                    t0 = time.time()
                    for _ in range(2):
                        e(xn, evn, Vn, ain, awn, din)
                return (time.time() - t0) / 2 * 1000
            except RuntimeError:
                return float("nan")

        ta, tl = ms("attn"), ms("lanczos")
        sp_s = f"{ta / tl:.2f}x" if (ta == ta and tl == tl and tl > 0) else "attn OOM"
        print(f"  {nn:>6} {ta:>13.1f} {tl:>12.1f} {sp_s:>9}")
    print()
    print("  The gap widens with n. Appendix I carries the same measurement to")
    print("  n = 50,000, where dense attention runs out of memory and Lanczos")
    print("  completes -- which is what makes a refiner at EVERY level feasible.")
    print()
    print(RULE)
    print(" The benchmark itself (67 graphs x 3 seeds x trials-matched METIS) is")
    print(" an HPC artefact: roughly 4 minutes per graph per seed on a laptop.")
    print(" Its records are in runs/*.jsonl and bench_data/*.tsv, and every figure")
    print(" script re-derives its numbers from them and asserts them before drawing.")
    print(RULE)


if __name__ == "__main__":
    main()
