#!/usr/bin/env python3
"""CFI co-spectral blind-spot probe: the complement to the CSL >1-WL test.

Cai-Fuerer-Immerman pairs CFI(H)/CFI'(H) over random 3-regular bases are non-isomorphic,
1-WL-indistinguishable, and CO-SPECTRAL (identical Laplacian eigenvalues; only the eigenvector
parity differs). A basis-invariant spectral readout depends on the graph only through the
eigenvalues and basis-invariant eigenvector statistics, so it discards exactly the parity that
separates the pair -- it should be pinned at chance.

Protocol mirrors csl_probe.py: freeze each encoder, read the same permutation-invariant graph
embedding (concat of mean/std/max/min over node embeddings), and fit a 5-fold cross-validated
ridge probe to the twist label (untwisted=0, twisted=1; chance = 50%). A reference probe on the
EXACT Laplacian spectrum shows co-spectrality directly: it too cannot beat chance, so no spectral
feature -- whatever its budget -- separates the pair.

  PYTHONPATH=src python src/cfi_probe.py
  PYTHONPATH=src python src/cfi_probe.py --model models/spectral_refiner_k4_eps_0_03.pt --n-eigs 8
"""
import os, sys, argparse
import numpy as np, torch, scipy.sparse as sp
SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
from csl_probe import embed, ridge_cv_accuracy                      # noqa: E402
from graph_transformer import SpectralGraphTransformer              # noqa: E402
from refiner import load_spectral_actor_critic                      # noqa: E402
from srmp.cfi import generate_cfi_from_3regular                     # noqa: E402


def lap_spectrum(G):
    A = G.adj.tocsr().astype(float); n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    L = (sp.diags(deg) - A).toarray()
    return np.sort(np.linalg.eigvalsh(L))


def build_pairs(bases=(8, 10, 12, 14), seeds=40):
    """Untwisted=0, twisted=1 CFI graphs over random 3-regular bases; return graphs, labels, gaps."""
    graphs, labels, gaps = [], [], []
    for n in bases:
        for s in range(seeds):
            try:
                G0, G1 = generate_cfi_from_3regular(n, twist_count=1, seed=s)
            except Exception:
                continue
            s0, s1 = lap_spectrum(G0), lap_spectrum(G1)
            if len(s0) != len(s1):
                continue
            gaps.append(float(np.abs(s0 - s1).max()))              # co-spectrality residual
            graphs += [G0, G1]; labels += [0, 1]
    return graphs, labels, gaps


def spectrum_feature(G, K=24):
    s = lap_spectrum(G); s = np.pad(s, (0, max(0, K - len(s))))[:K]; return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-eigs", type=int, default=8)
    a = ap.parse_args()

    graphs, labels, gaps = build_pairs()
    npairs = len(graphs) // 2
    print(f"CFI (random 3-regular bases): {npairs} pairs, {len(graphs)} graphs (balanced 0/1)")
    print(f"co-spectrality residual  max={max(gaps):.2e}  median={np.median(gaps):.2e}  "
          f"(~0 => co-spectral; only eigenVECTOR parity differs)\n")

    in_dim = a.k + 6
    configs = []
    if a.model:
        pol, _ = load_spectral_actor_critic(a.model, map_location="cpu")
        configs.append((f"TRAINED {os.path.basename(a.model)}", pol.encoder.eval()))
    torch.manual_seed(0)
    mk = lambda pe, gk: SpectralGraphTransformer(in_dim=in_dim, d_model=64, n_heads=4, n_layers=2,
                                                 pe_kind=pe, global_kind=gk, use_local=True).eval()
    configs += [("stable PE (baseline / Module A PE)", mk("stable", "attn")),
                ("specformer PE (Module B)", mk("specformer", "attn")),
                ("lanczos+stable (Module A deployed)", mk("stable", "lanczos")),
                ("no-PE (1-WL floor)", mk("none", "attn"))]

    print(f"{'readout':40} {'CFI twist acc':>16}   (chance = 50%)")
    print("-" * 68)
    for name, enc in configs:
        X = np.stack([embed(enc, G, a.k, a.n_eigs) for G in graphs])
        acc, sd = ridge_cv_accuracy(X, labels)
        print(f"{name:40} {f'{acc:.1%} +/- {sd:.1%}':>16}")
    Xs = np.stack([spectrum_feature(G) for G in graphs])
    acc, sd = ridge_cv_accuracy(Xs, labels)
    print(f"{'exact Laplacian spectrum (reference)':40} {f'{acc:.1%} +/- {sd:.1%}':>16}   <- co-spectral => at chance")
    print("\nExpected: every basis-invariant readout at or below the 50% chance floor -- the "
          "co-spectral blind spot; the CSL probe (csl_probe.py) is the complementary >1-WL witness.")


if __name__ == "__main__":
    main()
