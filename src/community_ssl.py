#!/usr/bin/env python3
"""Phase 1 -- SSL community encoder (learned spectral clustering, DMoN objective).

Self-supervised: NO labels. A small MLP over spectral features (Laplacian eigenvectors + log-degree)
produces a soft community assignment S; trained per-graph to maximize graph modularity (DMoN). The
argmax gives per-vertex communities for community-aware coarsening (resolve_communities source="ssl").

Why an MLP on eigenvectors, not the heavy SpectralGraphTransformer: a controlled SBM check showed the
from-scratch transformer reaches only purity 0.43 (it over-smooths), while a *linear* head on the
eigenvectors reaches purity 1.0 with the same DMoN loss. For per-graph transductive community detection
the spectral signal is already in the eigenvectors; the learnable part is how to read & sharpen it.
This stays graph-adaptive (per-graph fit, learnable resolution C) -- the lever over fixed-res Louvain.

Loss = L_modularity + lam * L_collapse  (DMoN):
  L_mod      = -(1/2m) [ Tr(STAS) - ||STd||^2 / 2m ]        (negative modularity)
  L_collapse = (sqrtC / n) ||Sum_i S_i|| - 1                     (prevents the all-in-one-cluster collapse)
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from srmp.spectral import compute_eigenvectors            # noqa: E402
from srmp.rl_policy import graph_to_torch                 # noqa: E402


class CommunitySSL(nn.Module):
    """MLP over spectral features -> soft community assignment (temperature-sharpened softmax)."""
    def __init__(self, in_dim, C=32, hidden=64, temp=0.5):
        super().__init__()
        self.temp = temp
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, C))

    def forward(self, feats):
        return F.softmax(self.net(feats) / self.temp, dim=-1)   # (n, C)


def dmon_loss(S, A_sparse, deg, m, lam=1.0):
    AS = torch.sparse.mm(A_sparse, S)                       # (n, C)
    mod_term = (S * AS).sum()                               # Tr(STAS)
    Sd = (S * deg.unsqueeze(1)).sum(0)                      # STd : (C,)
    L_mod = -(mod_term - (Sd * Sd).sum() / (2.0 * m)) / (2.0 * m)
    cluster_size = S.sum(0)                                 # (C,)
    n, C = S.shape
    L_collapse = (float(np.sqrt(C)) / n) * torch.linalg.norm(cluster_size) - 1.0
    return L_mod + lam * L_collapse, L_mod.item()


def _features(G, n_eigs, device):
    """Spectral features: top-k Laplacian eigenvectors (skip the trivial constant) + log-degree."""
    n = G.n
    deg_np = np.asarray(G.degrees, dtype=np.float64)
    k = min(n_eigs + 1, n - 1)
    ev, V = compute_eigenvectors(G, num_eigenvectors=k, normalized=True)
    V = np.asarray(V)
    if V.shape[1] > 1:
        V = V[:, 1:]                                        # drop the constant eigenvector
    logd = np.log1p(deg_np).reshape(-1, 1)
    logd = (logd - logd.mean()) / (logd.std() + 1e-6)
    feats = np.concatenate([V, logd], axis=1).astype(np.float32)
    return torch.tensor(feats, device=device), deg_np


def fit_communities(G, seed=42, C=32, epochs=600, lr=5e-2, n_eigs=16, lam=0.5,
                    temp=0.5, hidden=64, device="cpu", verbose=False):
    """Train per-graph; return per-vertex community ids (contiguous)."""
    torch.manual_seed(seed)
    n = G.n
    feats, deg_np = _features(G, n_eigs, device)
    ai, aw, di = graph_to_torch(G, device)
    A = torch.sparse_coo_tensor(ai, aw, (n, n)).coalesce()
    deg_t = torch.tensor(deg_np, dtype=torch.float32, device=device)
    m = float(deg_np.sum() / 2.0)

    model = CommunitySSL(in_dim=feats.shape[1], C=min(C, n), hidden=hidden, temp=temp).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        S = model(feats)
        loss, lmod = dmon_loss(S, A, deg_t, m, lam)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if verbose and ep % 100 == 0:
            nc = len(np.unique(S.argmax(1).detach().cpu().numpy()))
            print(f"  ep {ep:4d} loss={loss.item():.4f} modularity={-lmod * 2.0 * m:.4f} #clusters={nc}")
    model.eval()
    with torch.no_grad():
        comm = model(feats).argmax(1).cpu().numpy().astype(np.int64)
    _, comm = np.unique(comm, return_inverse=True)          # contiguous ids
    return comm.astype(np.int64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--n-eigs", type=int, default=16)
    args = ap.parse_args()
    from srmp.graph import read_metis
    G = read_metis(args.graph)
    comm = fit_communities(G, epochs=args.epochs, C=args.C, n_eigs=args.n_eigs, verbose=True)
    print(f"learned {len(np.unique(comm))} communities on n={G.n}")
