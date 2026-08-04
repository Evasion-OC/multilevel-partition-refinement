"""
Integrated Gradients on the SRMP RL Policy
===========================================
Computes per-feature-channel attributions for the learned vertex
selection policy, so we can answer "what structural signals does the
policy actually read?" instead of inferring them from outcomes.

For a trained ``ActorCritic`` the actor output of interest is the logit
assigned to the chosen vertex v*(s). Integrated Gradients (Sundararajan,
Taly & Yan, ICML 2017) attributes that scalar output to the input
feature tensor X via

    IG_i(s) = (X_i - X_i^0) * int_{alpha=0}^{1}
                 d/dX_i f(X^0 + alpha * (X - X^0)) d alpha

approximated with an ``m``-step Riemann sum. Only the node feature
tensor is interpolated; adjacency, edge weights, and the boundary mask
are kept fixed -- they are structural, not learned-from.

Output is aggregated into semantic channel groups matching the layout
of ``srmp.rl_policy.build_features``:

    partition_onehot : dims [0 .. k-1]
    degree           : dim k
    boundary_flag    : dim k+1
    gain             : dim k+2
    ext_frac         : dim k+3
    fullness         : dim k+4
    headroom         : dim k+5
    spectral         : dims [k+6 .. k+5+spectral_dim]   (if enabled)

References
----------
Sundararajan, Taly & Yan (ICML 2017): "Axiomatic Attribution for
Deep Networks". Integrated Gradients satisfies sensitivity and
implementation-invariance, the two axioms that disqualify plain
gradient-based saliency.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .graph import Graph
from .rl_policy import (
    ActorCritic, build_features, build_edge_features, graph_to_torch,
)


# ---------------------------------------------------------------------------
# Feature-channel layout helpers
# ---------------------------------------------------------------------------

def feature_channel_groups(
    k: int,
    spectral_dim: int,
    use_spectral: bool,
) -> Dict[str, Tuple[int, int]]:
    """Return the (start, end) half-open slice of each semantic channel."""
    groups: Dict[str, Tuple[int, int]] = {}
    groups["partition_onehot"] = (0, k)
    groups["degree"] = (k, k + 1)
    groups["boundary_flag"] = (k + 1, k + 2)
    groups["gain"] = (k + 2, k + 3)
    groups["ext_frac"] = (k + 3, k + 4)
    groups["fullness"] = (k + 4, k + 5)
    groups["headroom"] = (k + 5, k + 6)
    if use_spectral and spectral_dim > 0:
        groups["spectral"] = (k + 6, k + 6 + spectral_dim)
    return groups


# ---------------------------------------------------------------------------
# Core IG kernel
# ---------------------------------------------------------------------------

def _to_torch(
    G: Graph,
    partition: np.ndarray,
    spectral_coords: Optional[np.ndarray],
    config,
    k: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, Optional[torch.Tensor]]:
    """Build all tensors needed for a forward pass."""
    feats_np = build_features(G, partition, k, spectral_coords=spectral_coords)
    features = torch.tensor(feats_np, dtype=torch.float32, device=device)

    adj_idx, adj_w, deg_inv = graph_to_torch(G, device=device)

    # Boundary mask is a simple partition-boundary check on the CSR graph.
    coo = G.coo
    is_cross = partition[coo.row] != partition[coo.col]
    bnd = np.zeros(G.n, dtype=bool)
    np.add.at(bnd, coo.row[is_cross], 1)
    bnd_mask = torch.tensor(bnd, dtype=torch.bool, device=device)

    edge_feats = None
    if config.use_edge_features:
        ef_np = build_edge_features(G, partition)
        edge_feats = torch.tensor(ef_np, dtype=torch.float32, device=device)
    return features, adj_idx, adj_w, deg_inv, bnd_mask, edge_feats


def _target_scalar(
    model: ActorCritic,
    features: torch.Tensor,
    adj_idx: torch.Tensor,
    adj_w: torch.Tensor,
    deg_inv: torch.Tensor,
    bnd_mask: torch.Tensor,
    edge_feats: Optional[torch.Tensor],
    target_vertex: int,
) -> torch.Tensor:
    logits, _ = model(
        features, adj_idx, adj_w, deg_inv, bnd_mask, edge_feats,
    )
    return logits[target_vertex]


def integrated_gradients_state(
    model: ActorCritic,
    features: torch.Tensor,
    adj_idx: torch.Tensor,
    adj_w: torch.Tensor,
    deg_inv: torch.Tensor,
    bnd_mask: torch.Tensor,
    edge_feats: Optional[torch.Tensor],
    target_vertex: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 32,
) -> torch.Tensor:
    """
    Compute IG of the target vertex logit w.r.t. the input feature tensor.

    Returns a tensor of shape ``features.shape`` whose sum is (by the
    completeness axiom) approximately equal to
    ``f(features) - f(baseline)``.
    """
    if baseline is None:
        baseline = torch.zeros_like(features)

    # Path integral via Riemann midpoint rule.
    alphas = (torch.arange(steps, dtype=features.dtype, device=features.device)
              + 0.5) / steps

    total_grad = torch.zeros_like(features)
    for alpha in alphas:
        x = baseline + alpha * (features - baseline)
        x = x.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logit = _target_scalar(
            model, x, adj_idx, adj_w, deg_inv, bnd_mask, edge_feats,
            target_vertex,
        )
        grad = torch.autograd.grad(logit, x, retain_graph=False)[0]
        total_grad = total_grad + grad.detach()
    avg_grad = total_grad / steps
    return (features - baseline) * avg_grad


# ---------------------------------------------------------------------------
# Rollout-level aggregation
# ---------------------------------------------------------------------------

def rollout_attribution(
    trainer,
    G: Graph,
    initial_partition: np.ndarray,
    epsilon: float,
    spectral_coords: Optional[np.ndarray] = None,
    num_steps: int = 30,
    ig_steps: int = 32,
    baseline_kind: str = "zero",
) -> Dict[str, object]:
    """
    Run a greedy rollout of the trained policy on ``G`` and aggregate
    integrated-gradients attributions over visited states.

    Parameters
    ----------
    trainer : PPOTrainer
        Loaded trainer with ``trainer.model`` in eval mode.
    G : Graph
    initial_partition : np.ndarray
        Starting partition (e.g., the coarsest-level initial partition).
    epsilon : float
        Balance tolerance (unused at inference time but kept for future
        step logic).
    spectral_coords : np.ndarray, optional
    num_steps : int
        Max rollout length. At each step we greedily pick the boundary
        vertex with the highest logit, flip it to a random other block
        that reduces the cut, and continue.
    ig_steps : int
        Riemann steps for the IG path integral (larger -> more accurate).
    baseline_kind : {'zero', 'mean'}
        'zero' uses all-zeros baseline; 'mean' uses per-graph feature
        mean (useful when the zero point is off the data manifold).

    Returns
    -------
    dict with per-channel attribution stats and bookkeeping.
    """
    device = next(trainer.model.parameters()).device
    device_str = str(device)
    k = trainer.k
    model = trainer.model
    model.eval()

    groups = feature_channel_groups(
        k, trainer.config.spectral_dim, trainer.config.use_spectral_features,
    )

    # We ignore reward etc.; the policy logit is our attribution target.
    partition = initial_partition.copy()
    per_step: List[Dict] = []
    channel_totals: Dict[str, float] = {name: 0.0 for name in groups}
    channel_abs: Dict[str, float] = {name: 0.0 for name in groups}
    completeness_errors: List[float] = []

    for step in range(num_steps):
        feats, adj_idx, adj_w, deg_inv, bnd_mask, ef = _to_torch(
            G, partition, spectral_coords, trainer.config, k, device_str,
        )
        if not bnd_mask.any():
            break

        with torch.no_grad():
            logits, _ = model(feats, adj_idx, adj_w, deg_inv, bnd_mask, ef)
            target_vertex = int(torch.argmax(logits).item())

        if baseline_kind == "mean":
            baseline = feats.mean(dim=0, keepdim=True).expand_as(feats).detach()
        else:
            baseline = torch.zeros_like(feats)

        attrs = integrated_gradients_state(
            model, feats, adj_idx, adj_w, deg_inv, bnd_mask, ef,
            target_vertex=target_vertex,
            baseline=baseline,
            steps=ig_steps,
        )  # shape (n, feature_dim)

        # Completeness check: sum(attrs) ~ f(x) - f(baseline)
        with torch.no_grad():
            target_now = _target_scalar(
                model, feats, adj_idx, adj_w, deg_inv, bnd_mask, ef,
                target_vertex,
            ).item()
            target_base = _target_scalar(
                model, baseline, adj_idx, adj_w, deg_inv, bnd_mask, ef,
                target_vertex,
            ).item()
        completeness_errors.append(
            float(attrs.sum().item() - (target_now - target_base))
        )

        step_record = {"step": step, "target_vertex": target_vertex}
        for name, (s, e) in groups.items():
            slab = attrs[:, s:e]
            signed = float(slab.sum().item())
            abs_mag = float(slab.abs().sum().item())
            channel_totals[name] += signed
            channel_abs[name] += abs_mag
            step_record[name] = {"signed": signed, "abs": abs_mag}
        per_step.append(step_record)

        # Greedy move: flip target vertex to the block whose neighbours
        # it has the most of (that is not its current block). This is a
        # proxy rollout, not the trainer's exact env; it just keeps the
        # partition evolving so IG sees non-trivial boundary structure.
        nbrs, w = G.neighbors(target_vertex)
        if len(nbrs) == 0:
            continue
        block_scores = np.bincount(partition[nbrs], weights=w, minlength=k)
        current = int(partition[target_vertex])
        block_scores[current] = -1.0
        new_block = int(np.argmax(block_scores))
        if new_block != current and block_scores[new_block] >= 0:
            partition[target_vertex] = new_block

    n_steps = len(per_step)
    abs_total = sum(channel_abs.values()) or 1.0
    summary = {
        "n_rollout_steps": n_steps,
        "channel_signed_sum": channel_totals,
        "channel_abs_sum": channel_abs,
        "channel_abs_fraction": {
            k_: channel_abs[k_] / abs_total for k_ in channel_abs
        },
        "completeness_error_mean": (
            float(np.mean(np.abs(completeness_errors)))
            if completeness_errors else 0.0
        ),
        "completeness_error_max": (
            float(np.max(np.abs(completeness_errors)))
            if completeness_errors else 0.0
        ),
        "per_step": per_step,
    }
    return summary
