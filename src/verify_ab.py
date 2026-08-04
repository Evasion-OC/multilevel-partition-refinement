#!/usr/bin/env python3
"""Verify Modules A + B (the gate before C).

(1) Encode microbenchmark: Lanczos spectral mix (Module A) vs dense attention, encode time across n
    -> shows the global branch is no longer the O(n^2) bottleneck.
(2) Parity / integration micro-check: untrained new-mode policies behind the global no-harm guard
    -> the new backbones run end-to-end, give VALID balanced partitions, and the guard keeps the cut
       at the FM-only floor (no regression). The hard parity guarantee is structural
       (guard returns min(p_ml, p_fm) <= p_fm); this confirms it empirically + that nothing crashes.
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
import torch
SRC_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC_DIR)
from srmp.graph import Graph
from srmp.evaluation import compute_edge_cut, compute_imbalance
from srmp.config import SRMPConfig
from srmp.pipeline import SRMPPartitioner
from refiner import SpectralActorCritic, build_state
from graph_transformer import SpectralGraphTransformer
from multilevel_partitioner import make_multilevel_partitioner

torch.manual_seed(0)
k, eps = 4, 0.03


def rand_graph(n, deg=6, seed=0):
    rng = np.random.RandomState(seed); row, col = [], []
    for i in range(n):
        j = (i + 1) % n; row += [i, j]; col += [j, i]
    for _ in range(deg * n // 2):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b: row += [a, b]; col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)); A.data[:] = 1.0
    A.sum_duplicates(); A.data[:] = 1.0
    return Graph(A)


# ---------- (1) encode microbenchmark ----------
print("=== Module A: encoder forward time (ms/call) -- attn O(n^2) vs lanczos O(n) ===")
print(f"{'n':>6} {'attn_ms':>9} {'lanczos_ms':>11} {'speedup':>8}")
for n in [500, 1000, 2000, 4000, 8000]:
    G = rand_graph(n, seed=1)
    x, ev, V, _, ai, aw, di = build_state(G, np.zeros(n, dtype=np.int64), k, n_eigs=8)

    def enc_ms(global_kind):
        try:
            enc = SpectralGraphTransformer(in_dim=x.shape[1], d_model=64, n_heads=4, n_layers=2,
                                           use_local=True, global_kind=global_kind).eval()
            with torch.no_grad():
                enc(x, ev, V, ai, aw, di)                      # warmup
                t0 = time.time()
                for _ in range(3):
                    enc(x, ev, V, ai, aw, di)
            return (time.time() - t0) / 3 * 1000
        except RuntimeError as e:                              # e.g. OOM on dense attn at large n
            return float("nan")

    ta, tl = enc_ms("attn"), enc_ms("lanczos")
    sp_str = f"{ta/tl:>6.2f}x" if (ta == ta and tl == tl and tl > 0) else "   (attn OOM)"
    print(f"{n:>6} {ta:>9.1f} {tl:>11.1f} {sp_str:>8}")


# ---------- (2) parity / integration micro-check ----------
print("\n=== Parity / integration: guarded new-mode cut vs the FM-only floor ===")
print(f"{'graph':>9} {'mode':>11} {'cut':>7} {'fmfloor':>9} {'<=floor':>8} {'imb':>7} {'valid':>6}")
MODES = [("attn", "stable"), ("lanczos", "stable"), ("attn", "specformer"), ("lanczos", "specformer")]
all_ok = True
for n in [600, 1200]:
    G = rand_graph(n, seed=2)
    # classical FM-only (no RL) for context
    cfg = SRMPConfig(k=k, epsilon=eps, best_of_k=2, verbose=False); cfg.use_rl = False
    cfg.coarsening.min_vertices = 100; cfg.evolutionary.enabled = True; cfg.vcycle.num_cycles = 1
    fm_classical = compute_edge_cut(G, SRMPPartitioner(cfg).partition(G, seed=0))
    print(f"n={n:<7} {'classical-FM':>11} {fm_classical:>7.0f} {'-':>9} {'-':>8} {'-':>7} {'-':>6}")
    for gk, pk in MODES:
        pol = SpectralActorCritic(k=k, in_dim=k + 6, d_model=64, n_heads=4, n_layers=2,
                                  global_kind=gk, pe_kind=pk)
        # FM-only floor: same policy, multilevel refine OFF (ml_refine_max_n=0), no global guard
        ref = make_multilevel_partitioner(pol, k=k, epsilon=eps, best_of_k=2, num_cycles=1,
                                     min_vertices=100, ml_refine_max_n=0, global_no_harm=False)
        floor = compute_edge_cut(G, ref.partition(G, seed=0))
        # guarded full pipeline with this backbone
        P = make_multilevel_partitioner(pol, k=k, epsilon=eps, best_of_k=2, num_cycles=1,
                                   min_vertices=100, ml_refine_max_n=4000, global_no_harm=True)
        part = P.partition(G, seed=0)
        cut = compute_edge_cut(G, part)
        imb = compute_imbalance(G, part, k)
        valid = part.shape == (n,) and set(np.unique(part)).issubset(set(range(k)))
        le_floor = cut <= floor + 1e-6
        ok = valid and imb <= eps + 1e-6 and le_floor
        all_ok = all_ok and ok
        print(f"{'':>9} {gk+'/'+pk:>11} {cut:>7.0f} {floor:>9.0f} {str(le_floor):>8} {imb:>7.3f} {str(valid):>6}")

print("\nPASS" if all_ok else "\nFAIL",
      "-- A+B integrate end to end; guarded cut <= FM-only floor (no-harm holds); partitions valid+balanced.")
