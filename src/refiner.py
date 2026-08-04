"""
refiner.py -- Spectral-transformer refinement policy for multilevel partitioning.
================================================================================

An RL refinement policy whose encoder is a basis-invariant spectral graph transformer
in place of the SignNet+GATv2 encoder of the earlier baseline. The actor/critic heads
and the partition-state features keep the interface the multilevel scaffold calls, so
the policy drops into the existing loop (coarsen -> refine -> uncoarsen/FM).

  state: partition features x  +  Laplacian eigenpairs of the level graph (lambda, V)
  encode: StableSpectralPE + global graph transformer
  act:   actor picks a boundary vertex + target block; critic estimates V(s)

Connections: SpecFormer (Liao, spectral GNN x transformer) for the encoder; NerveNet
(Liao, structured GNN+RL policy) for the head; the encoder is SSL-pretrainable (Huang).
Scope: this file is the architecture, not a benchmark claim. Cut quality is governed by
structural axes independently of the 1-WL ceiling, so the axes worth chasing here are
sample efficiency, cross-family generalisation and stability, NOT lower cuts.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from graph_transformer import SpectralGraphTransformer
from srmp.graph import Graph
from srmp.spectral import compute_eigenvectors
from srmp.rl_policy import build_features, compute_feature_dim, graph_to_torch


class SpectralActorCritic(nn.Module):
    """Spectral-transformer encoder + actor/critic refinement heads."""

    def __init__(self, k: int, in_dim: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, pe_dim: int = 32, num_filters: int = 8, pe_mode: str = "diag",
                 use_local: bool = True, global_kind: str = "attn", pe_kind: str = "stable"):
        super().__init__()
        self.k = k
        self.encoder = SpectralGraphTransformer(
            in_dim=in_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            pe_dim=pe_dim, num_filters=num_filters, pe_mode=pe_mode, pe_kind=pe_kind,
            use_local=use_local, global_kind=global_kind,
        )
        self.actor_vertex = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.block_embedding = nn.Embedding(k, d_model // 2)
        self.actor_block = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in (self.actor_vertex, self.actor_block):           # PPO: tiny policy-head gain
            nn.init.orthogonal_(head[-1].weight, gain=0.01); nn.init.zeros_(head[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0); nn.init.zeros_(self.critic[-1].bias)

    def encode(self, features, eigvals, eigvecs,
               adj_indices=None, adj_weights=None, deg_inv=None) -> torch.Tensor:    # (n, d_model)
        return self.encoder(features, eigvals, eigvecs, adj_indices, adj_weights, deg_inv)

    def forward(self, features, eigvals, eigvecs, boundary_mask,
                adj_indices=None, adj_weights=None, deg_inv=None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(features, eigvals, eigvecs, adj_indices, adj_weights, deg_inv)
        v = self.actor_vertex(h).squeeze(-1)
        v = v.masked_fill(~boundary_mask, torch.finfo(v.dtype).min)   # boundary-only actions
        value = self.critic(h.mean(dim=0, keepdim=True)).squeeze()
        return v, value

    def get_block_logits(self, node_embedding, candidate_blocks) -> torch.Tensor:
        be = self.block_embedding(candidate_blocks)
        ve = node_embedding.unsqueeze(0).expand(len(candidate_blocks), -1)
        return self.actor_block(torch.cat([ve, be], dim=-1)).squeeze(-1)


def load_spectral_actor_critic(path, map_location="cpu"):
    """Rebuild a SpectralActorCritic from a train_coarsest checkpoint using its STORED constructor
    args (d_model/n_heads/n_layers/pe_dim/num_filters/pe_mode/use_local), so a strict reload never
    depends on hard-coded hyperparameters. Returns (model.eval(), checkpoint_dict)."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = SpectralActorCritic(
        k=ck["k"], in_dim=ck["in_dim"], d_model=ck["d_model"], n_heads=ck["n_heads"],
        n_layers=ck["n_layers"], pe_dim=ck["pe_dim"],
        num_filters=ck.get("num_filters", 8), pe_mode=ck.get("pe_mode", "diag"),
        use_local=ck.get("use_local", True),
        global_kind=ck.get("global_kind", "attn"), pe_kind=ck.get("pe_kind", "stable"),
    )
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    return model, ck


def build_state(G: Graph, partition: np.ndarray, k: int, epsilon: float = 0.03, n_eigs: int = 8):
    """Partition-state tensors for the spectral refiner: features, eigenpairs, boundary mask."""
    feats = build_features(G, partition, k, epsilon, spectral_coords=None)   # (n, k+6)
    ev, V = compute_eigenvectors(G, num_eigenvectors=min(n_eigs, G.n - 1), normalized=True)
    x = torch.tensor(feats, dtype=torch.float32)
    eigvals = torch.tensor(np.asarray(ev), dtype=torch.float32)
    eigvecs = torch.tensor(np.asarray(V), dtype=torch.float32)
    boundary_mask = x[:, k + 1] > 0.5                                        # boundary indicator col
    adj_indices, adj_weights, deg_inv = graph_to_torch(G, device="cpu")      # for the GPS local branch
    return x, eigvals, eigvecs, boundary_mask, adj_indices, adj_weights, deg_inv


if __name__ == "__main__":
    import scipy.sparse as sp
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    n, k = 16, 4

    # small connected graph: ring + random chords
    row, col = [], []
    for i in range(n):
        j = (i + 1) % n; row += [i, j]; col += [j, i]
    for _ in range(n):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b: row += [a, b]; col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)); A.data[:] = 1.0
    A.sum_duplicates(); A.data[:] = 1.0
    G = Graph(A)
    partition = rng.randint(0, k, n)

    x, ev, V, bmask, ai, aw, di = build_state(G, partition, k, n_eigs=8)
    in_dim = x.shape[1]
    assert in_dim == compute_feature_dim(k, 0, use_spectral=False), (in_dim, k)
    model = SpectralActorCritic(k=k, in_dim=in_dim, d_model=64, n_layers=2)   # use_local=True (GPS)

    vscores, value = model(x, ev, V, bmask, ai, aw, di)                       # GPS path (local+global)
    h = model.encode(x, ev, V, ai, aw, di)
    bnode = int(torch.nonzero(bmask)[0])
    blogits = model.get_block_logits(h[bnode], torch.arange(k))
    # GPS local branch must actually change the embedding vs global-only
    h_global = model.encode(x, ev, V)
    local_effect = (h - h_global).abs().max().item()

    # gradient flows into the encoder
    loss = value + vscores[bmask].sum() + blogits.sum()
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)

    n_boundary = int(bmask.sum())
    masked_ok = bool((vscores[~bmask] <= torch.finfo(vscores.dtype).min / 2).all()) if (~bmask).any() else True
    print(f"graph n={n}, k={k}, in_dim={in_dim}, boundary nodes={n_boundary}")
    print(f"vertex logits shape : {tuple(vscores.shape)}  (boundary-masked: {masked_ok})")
    print(f"block logits shape  : {tuple(blogits.shape)}  (expect ({k},))")
    print(f"value               : {value.item():.4f} (scalar)")
    print(f"GPS local-branch eff: {local_effect:.3e}  (>0 => local MPNN active in the refiner)")
    print(f"encoder grad mass   : {enc_grad:.2f} over {sum(1 for p in model.encoder.parameters() if p.grad is not None)} tensors")
    ok = (vscores.shape == (n,) and blogits.shape == (k,) and value.dim() == 0
          and masked_ok and enc_grad > 0 and local_effect > 1e-4)
    print("PASS" if ok else "FAIL",
          "-- the GPS-hybrid encoder drives the refinement policy end to end.")
