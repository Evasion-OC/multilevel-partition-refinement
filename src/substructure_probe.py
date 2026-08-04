#!/usr/bin/env python3
"""Substructure-counting probe: an environment-ROBUST empirical >1-WL test.

Complements csl_probe.py. Random d-regular graphs are 1-WL-uniform (every vertex the same
colour), so triangle count is a >1-WL feature: a 1-WL-bounded GNN cannot recover it. Their
spectra are generically NON-degenerate (unlike CSL's 4-regular circulants), so the rank-K
truncation is numerically stable across platforms -- this probe does not suffer CSL's
degenerate-eigenspace sensitivity. Directly realises Proposition (spectral moments count
substructures): tr(L^3) is an eigenvalue function that tracks triangle count.

Protocol: freeze each encoder, take the same mean/std/max/min readout as csl_probe, then a
numpy ridge REGRESSION of the triangle count with stratified k-fold CV; report R^2.

  PYTHONPATH=src python src/substructure_probe.py --model models/spectral_refiner_k4_eps_0_03.pt
"""
import os, sys, argparse
import numpy as np, torch, scipy.sparse as sp
SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
from refiner import load_spectral_actor_critic, build_state
from graph_transformer import SpectralGraphTransformer
from srmp.graph import Graph


def circulant_regular(n, d):
    """d-regular circulant base (d even): connect i to i +/- 1..d/2."""
    assert d % 2 == 0
    edges = set()
    for i in range(n):
        for off in range(1, d // 2 + 1):
            a, b = i, (i + off) % n
            edges.add((min(a, b), max(a, b)))
    return edges


def random_regular(n, d, seed, n_swaps=None):
    """Random d-regular graph via double-edge swaps on a circulant base (degree-preserving)."""
    rng = np.random.RandomState(seed)
    E = circulant_regular(n, d)
    edges = list(E)
    n_swaps = n_swaps or 10 * len(edges)
    for _ in range(n_swaps):
        i, j = rng.randint(len(edges)), rng.randint(len(edges))
        if i == j:
            continue
        a, b = edges[i]; c, dd = edges[j]
        if len({a, b, c, dd}) < 4:
            continue
        na, nc = (min(a, c), max(a, c)), (min(b, dd), max(b, dd))
        if na in E or nc in E or na[0] == na[1] or nc[0] == nc[1]:
            continue
        E.discard(edges[i]); E.discard(edges[j]); E.add(na); E.add(nc)
        edges[i] = na; edges[j] = nc
    rows, cols = [], []
    for a, b in E:
        rows += [a, b]; cols += [b, a]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.sum_duplicates(); A.data[:] = 1.0
    return Graph(A), A


def triangle_count(A):
    """tr(A^3) / 6."""
    A2 = (A @ A).tocsr()
    return float((A2.multiply(A)).sum() / 6.0)


def embed(enc, G, k, n_eigs):
    x, ev, V, _b, ai, aw, di = build_state(G, np.zeros(G.n, dtype=np.int64), k, n_eigs=n_eigs)
    with torch.no_grad():
        h = enc(x, ev, V, ai, aw, di)
    return torch.cat([h.mean(0), h.std(0), h.max(0).values, h.min(0).values]).numpy()


def ridge_cv_r2(X, y, folds=5, lam=1.0):
    """Stratified-by-shuffle k-fold ridge regression; returns mean/std R^2."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = np.asarray(y, dtype=float)
    yn = (y - y.mean()) / (y.std() + 1e-8)
    n = len(y); order = np.random.RandomState(0).permutation(n)
    r2s = []
    for f in range(folds):
        test = order[f::folds]; tr_mask = np.ones(n, bool); tr_mask[test] = False
        train = np.where(tr_mask)[0]
        Xtr = np.hstack([X[train], np.ones((len(train), 1))])
        Xte = np.hstack([X[test], np.ones((len(test), 1))])
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ yn[train])
        pred = Xte @ w; true = yn[test]
        ss_res = float(((true - pred) ** 2).sum())
        ss_tot = float(((true - true.mean()) ** 2).sum()) + 1e-12
        r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-eigs", type=int, default=8)
    ap.add_argument("--n", type=int, default=30, help="vertices per graph")
    ap.add_argument("--d", type=int, default=4, help="regular degree (even)")
    ap.add_argument("--graphs", type=int, default=300)
    a = ap.parse_args()

    graphs, tri = [], []
    for g in range(a.graphs):
        G, A = random_regular(a.n, a.d, seed=7 * g + 1)
        graphs.append(G); tri.append(triangle_count(A))
    tri = np.array(tri)
    print(f"random {a.d}-regular G(n={a.n}): {len(graphs)} graphs; "
          f"triangle count range [{tri.min():.0f}, {tri.max():.0f}], "
          f"mean {tri.mean():.1f}, std {tri.std():.1f} (all 1-WL-uniform)\n")

    in_dim = a.k + 6
    configs = []
    if a.model:
        pol, _ = load_spectral_actor_critic(a.model, map_location="cpu")
        configs.append((f"TRAINED {os.path.basename(a.model)}", pol.encoder.eval(), a.n_eigs))
    torch.manual_seed(0)
    mk = lambda pe, gk: SpectralGraphTransformer(in_dim=in_dim, d_model=64, n_heads=4, n_layers=2,
                                                 pe_kind=pe, global_kind=gk, use_local=True).eval()
    configs += [("untrained spectral (stable PE)", mk("stable", "attn"), 8),
                ("untrained spectral (Lanczos global)", mk("none", "lanczos"), 8),
                ("untrained NO-PE control (1-WL bound)", mk("none", "attn"), 8)]

    print(f"{'encoder':46} {'triangle-count R^2':>18}")
    print("-" * 66)
    for name, enc, ne in configs:
        X = np.stack([embed(enc, G, a.k, ne) for G in graphs])
        r2, sd = ridge_cv_r2(X, tri)
        print(f"{name:46} {f'{r2:.3f} +/- {sd:.3f}':>18}")
    print("\nExpected: spectral encoders recover triangle count (R^2 >> 0, the >1-WL witness "
          "via spectral moments); the NO-PE control near 0 (1-WL blind on regular graphs).")


if __name__ == "__main__":
    main()
