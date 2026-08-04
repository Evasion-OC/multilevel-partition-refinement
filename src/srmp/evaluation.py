"""
Evaluation Metrics
===================
Computes edge cut, normalised cut, imbalance, and comparison statistics.

Metrics:
  - Edge cut: Sum_{(u,v)inE, P(u)!=P(v)} w(u,v)
  - Normalised cut (Shi-Malik, 2000): Sum_i cut(V_i,V'_i)/vol(V_i)
  - Imbalance: max_i w(V_i) / ceil(W/k) - 1
  - Communication volume: Sum_v 1[v is boundary] . (|adjacent blocks| - 1)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from .graph import Graph

# Use AVX-512 C kernels when available, else pure-numpy
try:
    from .native.fast_kernels import (
        fast_edge_cut as _fast_edge_cut,
        fast_normalized_cut as _fast_ncut,
        NATIVE_AVAILABLE as _NATIVE,
    )
except ImportError:
    _fast_edge_cut = None
    _fast_ncut = None
    _NATIVE = False


def compute_edge_cut(G: Graph, partition: np.ndarray) -> float:
    """
    Compute total weight of edges crossing partition boundaries.

    cut(P) = Sum_{(u,v)inE, P(u)!=P(v)} w(u,v)

    Uses AVX-512 C kernels when available, else vectorised sparse ops.
    """
    if _fast_edge_cut is not None:
        return _fast_edge_cut(G, partition)
    rows, cols = G.adj.nonzero()
    cross = partition[rows] != partition[cols]
    # Each edge counted twice in symmetric matrix, so divide by 2
    return float(G.adj[rows[cross], cols[cross]].sum()) / 2.0


def compute_normalized_cut(G: Graph, partition: np.ndarray) -> float:
    """
    Compute the normalised cut (Shi-Malik).

    NCut(V_1,...,V_k) = Sum_{i=1}^k cut(V_i, V'_i) / vol(V_i)
    """
    k = int(partition.max()) + 1
    degrees = G.degrees
    ncut = 0.0

    for block in range(k):
        mask = partition == block
        vol = degrees[mask].sum()
        if vol < 1e-12:
            continue

        # Cut for this block
        block_vertices = np.where(mask)[0]
        cut_block = 0.0
        for v in block_vertices:
            nbrs, weights = G.neighbors(v)
            for nb, w in zip(nbrs, weights):
                if partition[nb] != block:
                    cut_block += w
        # Each edge counted once per endpoint, but we only count from block side
        ncut += cut_block / vol

    return ncut


def compute_normalized_cut_fast(G: Graph, partition: np.ndarray) -> float:
    """Normalised cut -- uses AVX-512 C kernels when available."""
    if _fast_ncut is not None:
        return _fast_ncut(G, partition)

    k = int(partition.max()) + 1
    rows, cols = G.adj.nonzero()
    weights = np.asarray(G.adj[rows, cols]).flatten()
    degrees = G.degrees

    ncut = 0.0
    for block in range(k):
        mask = partition == block
        vol = degrees[mask].sum()
        if vol < 1e-12:
            continue
        # Edges leaving this block
        leaving = mask[rows] & (~mask[cols])
        cut_block = weights[leaving].sum()
        ncut += cut_block / vol

    return ncut


def compute_imbalance(G: Graph, partition: np.ndarray, k: int) -> float:
    """
    Compute partition imbalance.

    imbalance = max_i w(V_i) / ceil(W/k) - 1

    Returns 0 for perfect balance.
    """
    block_weights = np.bincount(partition, weights=G.vertex_weights, minlength=k)

    target = np.ceil(G.total_weight / k)
    if target < 1e-12:
        return 0.0
    return float(block_weights.max() / target - 1.0)


def compute_block_weights(G: Graph, partition: np.ndarray, k: int) -> np.ndarray:
    """Compute weight of each partition block."""
    return np.bincount(partition, weights=G.vertex_weights, minlength=k).astype(np.float64)


def compute_communication_volume(G: Graph, partition: np.ndarray) -> int:
    """
    Compute communication volume: for each boundary vertex, count
    the number of distinct adjacent blocks.

    Vectorized using sparse operations.
    """
    rows, cols = G.adj.nonzero()
    cross = partition[rows] != partition[cols]
    cross_rows = rows[cross]
    cross_blocks = partition[cols[cross]]

    if len(cross_rows) == 0:
        return 0

    # Count distinct (vertex, neighbor_block) pairs
    # Pack into unique pairs and count unique per vertex
    pairs = np.stack([cross_rows, cross_blocks], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    # Each unique (vertex, block) pair contributes 1 to comm volume
    return int(unique_pairs.shape[0])


def evaluate_partition(
    G: Graph,
    partition: np.ndarray,
    k: int,
    epsilon: float = 0.03,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a partition.

    Returns
    -------
    dict with keys: edge_cut, ncut, imbalance, comm_volume, num_boundary
    """
    edge_cut = compute_edge_cut(G, partition)
    ncut = compute_normalized_cut_fast(G, partition)
    imbalance = compute_imbalance(G, partition, k)
    comm_vol = compute_communication_volume(G, partition)
    boundary = G.boundary_vertices_fast(partition)

    return {
        'edge_cut': edge_cut,
        'normalized_cut': ncut,
        'imbalance': imbalance,
        'communication_volume': comm_vol,
        'num_boundary': len(boundary),
        'balance_ok': imbalance <= epsilon + 1e-6,
    }


def validate_partition(
    G: Graph,
    partition: np.ndarray,
    k: int,
) -> Dict[str, object]:
    """
    Validate partition integrity and return detailed block statistics.

    Checks:
      1. Every vertex is assigned to exactly one block (sum of block sizes == n)
      2. All block IDs are in [0, k-1]
      3. Sum of intra-block edges + cross-block edges == total edges
      4. No empty blocks

    Returns
    -------
    dict with keys:
        valid : bool
            True if all checks pass.
        errors : list of str
            Description of each failed check (empty if valid).
        block_sizes : list of int
            Number of vertices in each block.
        block_weights : list of float
            Total vertex weight in each block.
        block_internal_edges : list of int
            Number of edges with both endpoints in each block.
        block_boundary_edges : list of int
            Number of edges with exactly one endpoint in each block.
        total_vertices_check : bool
            True if sum of block sizes == G.n.
        total_edges_check : bool
            True if (sum of internal edges + cross edges / 2) == G.m.
    """
    n = G.n
    m = G.m
    errors = []

    # --- Check 1: partition covers all vertices ---
    if len(partition) != n:
        errors.append(
            f"Partition length ({len(partition)}) != number of vertices ({n})"
        )

    # --- Check 2: all block IDs in [0, k-1] ---
    unique_blocks = np.unique(partition)
    out_of_range = unique_blocks[(unique_blocks < 0) | (unique_blocks >= k)]
    if len(out_of_range) > 0:
        errors.append(
            f"Block IDs out of range [0, {k-1}]: {out_of_range.tolist()}"
        )

    # --- Per-block statistics (vectorized) ---
    block_sizes = np.bincount(partition, minlength=k).astype(np.int64)
    block_weights = np.bincount(partition, weights=G.vertex_weights, minlength=k).astype(np.float64)
    block_internal_edges = np.zeros(k, dtype=np.int64)
    block_boundary_edges = np.zeros(k, dtype=np.int64)

    # Count edges per block (vectorized)
    rows, cols = G.adj.nonzero()
    # Only count each undirected edge once (u < v)
    upper = rows < cols
    u_upper = rows[upper]
    v_upper = cols[upper]
    bu = partition[u_upper]
    bv = partition[v_upper]
    same_block = bu == bv

    # Internal edges: both endpoints in same block
    if np.any(same_block):
        internal_blocks = bu[same_block]
        np.add.at(block_internal_edges, internal_blocks, 1)

    # Cross edges: endpoints in different blocks
    cross_mask = ~same_block
    total_cross_edges = int(cross_mask.sum())
    if total_cross_edges > 0:
        np.add.at(block_boundary_edges, bu[cross_mask], 1)
        np.add.at(block_boundary_edges, bv[cross_mask], 1)

    total_internal = int(block_internal_edges.sum())
    reconstructed_m = total_internal + total_cross_edges

    # --- Check 3: vertex count ---
    vertex_sum = int(block_sizes.sum())
    total_vertices_ok = (vertex_sum == n)
    if not total_vertices_ok:
        errors.append(
            f"Sum of block sizes ({vertex_sum}) != n ({n})"
        )

    # --- Check 4: edge count ---
    total_edges_ok = (reconstructed_m == m)
    if not total_edges_ok:
        errors.append(
            f"Internal edges ({total_internal}) + cross edges ({total_cross_edges}) "
            f"= {reconstructed_m} != m ({m})"
        )

    # --- Check 5: empty blocks ---
    empty_blocks = np.where(block_sizes == 0)[0]
    if len(empty_blocks) > 0:
        errors.append(
            f"Empty blocks: {empty_blocks.tolist()}"
        )

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'block_sizes': block_sizes.tolist(),
        'block_weights': block_weights.tolist(),
        'block_internal_edges': block_internal_edges.tolist(),
        'block_boundary_edges': block_boundary_edges.tolist(),
        'total_cross_edges': total_cross_edges,
        'total_vertices_check': total_vertices_ok,
        'total_edges_check': total_edges_ok,
    }


def performance_profile(
    results: Dict[str, List[float]],
    tau_max: float = 2.0,
    num_points: int = 100,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute performance profiles (Dolan-More, 2002).

    Parameters
    ----------
    results : dict
        {solver_name: [cut_value_for_each_problem]}
    tau_max : float
        Maximum performance ratio to plot.
    num_points : int

    Returns
    -------
    profiles : dict
        {solver_name: (tau_values, rho_values)}
    """
    solver_names = list(results.keys())
    n_problems = len(results[solver_names[0]])

    # Compute best for each problem
    all_results = np.array([results[s] for s in solver_names])  # (n_solvers, n_problems)
    best = all_results.min(axis=0)  # Best result for each problem

    # Performance ratios
    ratios = {}
    for i, s in enumerate(solver_names):
        r = np.zeros(n_problems)
        for p in range(n_problems):
            if best[p] > 0:
                r[p] = all_results[i, p] / best[p]
            else:
                r[p] = 1.0
        ratios[s] = r

    # Build profiles
    taus = np.linspace(1.0, tau_max, num_points)
    profiles = {}
    for s in solver_names:
        rho = np.array([np.mean(ratios[s] <= t) for t in taus])
        profiles[s] = (taus, rho)

    return profiles


def geometric_mean_ratio(cuts: List[float], best_known: List[float]) -> float:
    """
    Compute geometric mean of cut ratios to best known.

    Standard aggregate metric in KaHIP publications.
    """
    ratios = []
    for c, b in zip(cuts, best_known):
        if b > 0:
            ratios.append(c / b)
    if len(ratios) == 0:
        return float('inf')
    return float(np.exp(np.mean(np.log(ratios))))
