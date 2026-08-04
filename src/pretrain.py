"""
ssl.py -- Self-supervised pretraining objectives for the spectral graph transformer.
===================================================================================

Two objectives behind one interface (both, per design decision):

  1. MaskedReconstructionSSL  -- mask a fraction of node features with a learnable
     mask token, encode, reconstruct the masked rows (graph-MAE / masked-modelling
     lineage; closest to Lang Huang's masked-image-modelling work).

  2. ContrastiveSSL -- two feature-augmented views per graph, NT-Xent pulls the two
     views of the same graph together and pushes different graphs apart (GraphCL-style).

`SSLObjective` runs both on a batch of graphs and returns a weighted sum, so pretraining
and the masked-vs-contrastive ablation share one code path. The encoder instance is
shared and is what gets pretrained.

Note: structural augmentations (edge/node drop) change the Laplacian and would require
recomputing eigenpairs, so they belong in the data pipeline; the in-loss augmentations
here are feature-space (masking), which keeps the precomputed spectral PE valid.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

Graph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (x, eigvals, eigvecs)


class MaskedReconstructionSSL(nn.Module):
    def __init__(self, encoder: nn.Module, in_dim: int, d_model: int, mask_ratio: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, in_dim)
        )

    def forward(self, x: torch.Tensor, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        num_mask = max(1, int(self.mask_ratio * n))
        idx = torch.randperm(n, device=x.device)[:num_mask]
        x_masked = x.clone()
        x_masked[idx] = self.mask_token
        h = self.encoder(x_masked, eigvals, eigvecs)
        recon = self.decoder(h)
        return F.mse_loss(recon[idx], x[idx])


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """NT-Xent / InfoNCE over a batch of paired graph embeddings. z1,z2: (B, p)."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)     # (2B, p)
    sim = (z @ z.t()) / temp                               # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


class ContrastiveSSL(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, proj_dim: int = 64,
                 temp: float = 0.2, feat_mask: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.temp = temp
        self.feat_mask = feat_mask
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim))

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        keep = (torch.rand_like(x) > self.feat_mask).float()
        return x * keep

    def _embed(self, g: Graph) -> torch.Tensor:
        x, ev, V = g
        return self.proj(self.encoder(self._augment(x), ev, V).mean(dim=0))

    def forward(self, batch: List[Graph]) -> torch.Tensor:
        z1 = torch.stack([self._embed(g) for g in batch])
        z2 = torch.stack([self._embed(g) for g in batch])
        return nt_xent(z1, z2, self.temp)


class SSLObjective(nn.Module):
    """Combined masked + contrastive pretraining over a batch of graphs."""

    def __init__(self, encoder: nn.Module, in_dim: int, d_model: int,
                 w_mask: float = 1.0, w_con: float = 1.0,
                 mask_ratio: float = 0.3, temp: float = 0.2, feat_mask: float = 0.2):
        super().__init__()
        self.masked = MaskedReconstructionSSL(encoder, in_dim, d_model, mask_ratio)
        self.contrastive = ContrastiveSSL(encoder, d_model, temp=temp, feat_mask=feat_mask)
        self.w_mask, self.w_con = w_mask, w_con

    def forward(self, batch: List[Graph]) -> Tuple[torch.Tensor, Dict[str, float]]:
        lm = torch.stack([self.masked(*g) for g in batch]).mean()
        lc = self.contrastive(batch)
        total = self.w_mask * lm + self.w_con * lc
        return total, {"masked": lm.item(), "contrastive": lc.item(), "total": total.item()}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from graph_transformer import SpectralGraphTransformer

    torch.manual_seed(0)
    in_dim = 5

    def rand_graph(n, d=6):
        V, _ = torch.linalg.qr(torch.randn(n, d))
        ev = torch.sort(torch.rand(d)).values
        return (torch.randn(n, in_dim), ev, V)

    batch = [rand_graph(n) for n in (10, 14, 9, 12)]   # variable sizes, one batch
    enc = SpectralGraphTransformer(in_dim=in_dim, d_model=64, n_heads=8, n_layers=2, pe_dim=32)
    ssl = SSLObjective(enc, in_dim=in_dim, d_model=64, w_mask=1.0, w_con=1.0)

    loss, parts = ssl(batch)
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
    n_enc_params_with_grad = sum(1 for p in enc.parameters() if p.grad is not None)

    print(f"masked loss      : {parts['masked']:.4f}")
    print(f"contrastive loss : {parts['contrastive']:.4f}")
    print(f"total loss       : {parts['total']:.4f}")
    print(f"encoder grad mass: {enc_grad:.4f}  over {n_enc_params_with_grad} tensors")
    ok = torch.isfinite(loss) and enc_grad > 0
    print("PASS" if ok else "FAIL",
          "-- both objectives produce a finite loss and backprop into the shared encoder.")
