#!/usr/bin/env python3
"""CSL classification probe: the standard empirical >1-WL test (Murphy et al. 2019; Dwivedi et al.).

Circular Skip Link graphs CSL(n, s): an n-cycle plus skip-s chords. EVERY CSL graph is 4-regular,
so 1-WL gives a uniform colouring on all of them -- a 1-WL-bounded GNN cannot tell the skip-lengths
apart and a linear probe on its embeddings sits at chance (1/#classes). Different s give different
spectra, so a strictly->1-WL spectral encoder separates them and the probe accuracy jumps.

Protocol: freeze each encoder, extract a permutation-invariant graph embedding (concat of
mean/std/max/min over node embeddings), then a numpy ridge linear probe with stratified k-fold CV.

  PYTHONPATH=src python src/csl_probe.py --model models/spectral_refiner_k4_eps_0_03.pt
  PYTHONPATH=src python src/csl_probe.py --model models/spectral_refiner_k16_eps_0_03.pt --n-eigs 16
"""
import os, sys, argparse
import numpy as np, torch, scipy.sparse as sp
SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
from refiner import load_spectral_actor_critic, build_state
from graph_transformer import SpectralGraphTransformer
from srmp.graph import Graph

SKIPS = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]          # standard CSL(41) skip-lengths -> 10 classes


def csl_graph(n, s, perm_seed):
    rng = np.random.RandomState(perm_seed)
    perm = rng.permutation(n)
    rows, cols = [], []
    for i in range(n):
        for j in ((i + 1) % n, (i + s) % n):
            a, b = perm[i], perm[j]
            rows += [a, b]; cols += [b, a]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); A.sum_duplicates(); A.data[:] = 1.0
    return Graph(A)


def embed(enc, G, k, n_eigs):
    x, ev, V, _b, ai, aw, di = build_state(G, np.zeros(G.n, dtype=np.int64), k, n_eigs=n_eigs)
    with torch.no_grad():
        h = enc(x, ev, V, ai, aw, di)               # (n, d)
    return torch.cat([h.mean(0), h.std(0), h.max(0).values, h.min(0).values]).numpy()   # rich invariant readout


def ridge_cv_accuracy(X, y, folds=5, lam=1.0):
    """Stratified k-fold ridge linear classifier (one-hot regression -> argmax). numpy-only."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    classes = sorted(set(y)); cidx = {c: i for i, c in enumerate(classes)}
    Y = np.eye(len(classes))[[cidx[v] for v in y]]
    rng = np.random.RandomState(0)
    idx_by_c = {c: rng.permutation([i for i in range(len(y)) if y[i] == c]) for c in classes}
    accs = []
    for f in range(folds):
        test = np.concatenate([idx_by_c[c][f::folds] for c in classes])
        train = np.array([i for i in range(len(y)) if i not in set(test)])
        Xtr = np.hstack([X[train], np.ones((len(train), 1))])
        Xte = np.hstack([X[test], np.ones((len(test), 1))])
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y[train])
        pred = (Xte @ W).argmax(1); true = Y[test].argmax(1)
        accs.append(float((pred == true).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-eigs", type=int, default=8)
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--perms", type=int, default=15, help="permuted copies per skip-length")
    a = ap.parse_args()

    graphs, labels = [], []
    for s in SKIPS:
        for p in range(a.perms):
            graphs.append(csl_graph(a.n, s, perm_seed=1000 * s + p)); labels.append(s)
    print(f"CSL(n={a.n}): {len(graphs)} graphs, {len(SKIPS)} classes ({a.perms} perms each); "
          f"all 4-regular (1-WL-equal). chance = {1/len(SKIPS):.0%}\n")

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

    print(f"{'encoder':46} {'CSL probe acc':>16}")
    print("-" * 64)
    for name, enc, ne in configs:
        X = np.stack([embed(enc, G, a.k, ne) for G in graphs])
        acc, sd = ridge_cv_accuracy(X, labels)
        print(f"{name:46} {f'{acc:.1%} +/- {sd:.1%}':>16}")
    print(f"\nExpected: spectral encoders >> {1/len(SKIPS):.0%} chance (the >1-WL witness); "
          "the NO-PE control near chance (1-WL bound).")


if __name__ == "__main__":
    main()
