"""
Initial Partitioning (v3 - k-means++, recursive bisection, portfolio)
======================================================================
Methods for partitioning the coarsest-level graph:
  - Spectral k-way with k-means++ (Arthur & Vassilvitskii, SODA 2007)
  - Recursive spectral bisection
  - Greedy graph growing (BFS-based)
  - Random balanced partitioning (baseline)
  - Portfolio: try all methods with multiple restarts, pick best

v3 changes:
  - k-means++ initialization replaces random init (O(log k) competitive)
  - Row normalization of eigenvector matrix (NJW algorithm)
  - Recursive bisection for k=4 (2 levels of Fiedler bisection)
  - 20 restarts per method (from 5)
  - Balance enforcement post k-means

References:
  - Arthur & Vassilvitskii (SODA, 2007): k-means++ initialization
  - Ng, Jordan, Weiss (NeurIPS, 2002): Spectral clustering with row norm
  - Karypis & Kumar (1998): Greedy graph growing
  - Simon & Teng (1997): Recursive spectral bisection bounds
"""
import os as _os
# Prevent segfault in sklearn KMeans on macOS Accelerate + OpenMP conflict.
# Must be set before sklearn imports its native extensions.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional
from .graph import Graph
from .spectral import spectral_embedding, fiedler_vector, compute_eigenvectors


# Deterministic per-method seed offsets for the portfolio in
# multi_start_partition. Previously we used hash(name) % 1000, but Python's
# str-hash is randomised per process (PYTHONHASHSEED), so restart seeds
# differed between runs and across process-workers -- making the init cuts
# non-reproducible. Replacing with a fixed table stabilises the portfolio.
_METHOD_SEED_OFFSETS = {
    "spectral": 101,
    "recursive_bisection": 257,
    "ggg": 401,
}


def enforce_balance(
    G: Graph,
    partition: np.ndarray,
    k: int,
    epsilon: float = 0.03,
) -> np.ndarray:
    """
    Repair an imbalanced partition to satisfy the balance constraint.

    Repeatedly move the highest-gain vertex from the heaviest block to the
    lightest block until all blocks satisfy w(V_i) <= (1 + epsilon) * ceil(W/k).
    """
    partition = partition.copy()
    max_weight = (1.0 + epsilon) * np.ceil(G.total_weight / k)

    block_weights = np.zeros(k, dtype=np.float64)
    for v in range(G.n):
        block_weights[partition[v]] += G.vertex_weights[v]

    max_iterations = G.n * 2
    iteration = 0

    while iteration < max_iterations:
        overweight = np.where(block_weights > max_weight + 1e-9)[0]
        if len(overweight) == 0:
            break

        heaviest = overweight[np.argmax(block_weights[overweight])]
        lightest = int(np.argmin(block_weights))
        if lightest == heaviest:
            sorted_blocks = np.argsort(block_weights)
            lightest = sorted_blocks[0] if sorted_blocks[0] != heaviest else sorted_blocks[1]

        vertices_in_heavy = np.where(partition == heaviest)[0]
        best_v = -1
        best_gain = -float('inf')

        for v in vertices_in_heavy:
            if block_weights[heaviest] - G.vertex_weights[v] <= 0 and len(vertices_in_heavy) <= 1:
                continue
            if block_weights[lightest] + G.vertex_weights[v] > max_weight + 1e-9:
                continue

            nbrs, weights = G.neighbors(v)
            ext_to_target = 0.0
            int_in_source = 0.0
            for nb, w in zip(nbrs, weights):
                if partition[nb] == lightest:
                    ext_to_target += w
                elif partition[nb] == heaviest:
                    int_in_source += w
            gain = ext_to_target - int_in_source

            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v < 0:
            found = False
            for target in np.argsort(block_weights):
                if target == heaviest:
                    continue
                for v in vertices_in_heavy:
                    if block_weights[target] + G.vertex_weights[v] <= max_weight + 1e-9:
                        partition[v] = target
                        block_weights[heaviest] -= G.vertex_weights[v]
                        block_weights[target] += G.vertex_weights[v]
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        else:
            partition[best_v] = lightest
            block_weights[heaviest] -= G.vertex_weights[best_v]
            block_weights[lightest] += G.vertex_weights[best_v]

        iteration += 1

    # Swap-pair rebalance: when single-vertex moves cannot close the gap
    # (e.g. coarse graphs with heavy vertex weights where every candidate
    # would overflow the lightest block), try swapping a heavy vertex v in
    # the overweight block with a lighter vertex u in the target block, so
    # long as w(v) > w(u) and the destination slack allows the net move.
    swap_iterations = 0
    max_swap_iterations = G.n
    while swap_iterations < max_swap_iterations:
        overweight = np.where(block_weights > max_weight + 1e-9)[0]
        if len(overweight) == 0:
            break
        heaviest = int(overweight[np.argmax(block_weights[overweight])])
        lightest = int(np.argmin(block_weights))
        if lightest == heaviest:
            break

        heavy_vs = np.where(partition == heaviest)[0]
        light_vs = np.where(partition == lightest)[0]
        if len(heavy_vs) == 0 or len(light_vs) == 0:
            break

        slack = max_weight + 1e-9 - block_weights[lightest]
        overflow = block_weights[heaviest] - max_weight

        best_pair = None
        best_pair_gain = -float("inf")
        for v in heavy_vs:
            wv = G.vertex_weights[v]
            for u in light_vs:
                wu = G.vertex_weights[u]
                delta = wv - wu
                if delta <= 0:
                    continue
                if wv - wu > slack + 1e-9:
                    continue
                if delta < overflow - slack - 1e-9:
                    continue

                # Net cut gain: edges (v, heavy\{v}) + (u, light\{u}) cease
                # to be internal; (v, light) + (u, heavy) cease to be cut.
                nbrs_v, wts_v = G.neighbors(int(v))
                nbrs_u, wts_u = G.neighbors(int(u))
                gain = 0.0
                for nb, w in zip(nbrs_v, wts_v):
                    if nb == u:
                        continue
                    if partition[nb] == heaviest:
                        gain -= w
                    elif partition[nb] == lightest:
                        gain += w
                for nb, w in zip(nbrs_u, wts_u):
                    if nb == v:
                        continue
                    if partition[nb] == lightest:
                        gain -= w
                    elif partition[nb] == heaviest:
                        gain += w

                if gain > best_pair_gain:
                    best_pair_gain = gain
                    best_pair = (int(v), int(u))

        if best_pair is None:
            break
        v, u = best_pair
        partition[v] = lightest
        partition[u] = heaviest
        wv = G.vertex_weights[v]
        wu = G.vertex_weights[u]
        block_weights[heaviest] -= (wv - wu)
        block_weights[lightest] += (wv - wu)
        swap_iterations += 1

    return partition


def _kmeans_plus_plus_init(
    X: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    k-means++ initialization (Arthur & Vassilvitskii, SODA 2007).

    Guarantees E[cost] <= O(log k) * OPT.

    Choose first center uniformly at random, then each successive center
    with probability proportional to D(x)^2 (squared distance to nearest
    existing center).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data points.
    k : int
        Number of centers.
    rng : RandomState

    Returns
    -------
    centers : np.ndarray, shape (k, d)
    """
    n, d = X.shape
    centers = np.zeros((k, d), dtype=X.dtype)

    # First center: random
    idx = rng.randint(n)
    centers[0] = X[idx]

    # Compute initial distances
    dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        # Sample next center proportional to D(x)^2
        total = dist_sq.sum()
        if total < 1e-12:
            idx = rng.randint(n)
        else:
            probs = dist_sq / total
            idx = rng.choice(n, p=probs)
        centers[i] = X[idx]

        # Update distances
        new_dist_sq = np.sum((X - centers[i]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)

    return centers


def spectral_partition(
    G: Graph,
    k: int,
    epsilon: float = 0.03,
    seed: int = 42,
    n_init: int = 20,
) -> np.ndarray:
    """
    Spectral k-way partitioning via the Ng-Jordan-Weiss algorithm
    with k-means++ initialization, followed by balance enforcement.

    1. Compute k-dimensional spectral embedding from L_sym.
    2. Row-normalize the embedding (project onto unit sphere).
    3. Run k-means++ on the embedding with multiple restarts.
    4. Enforce balance constraint.
    """
    if G.n <= k:
        return np.arange(G.n, dtype=np.int64) % k

    _, _, embedding = spectral_embedding(G, k, normalized=True, row_normalize=True)

    # k-means with k-means++ init (scikit-learn uses k-means++ by default)
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=n_init,
        random_state=seed,
        max_iter=300,
    )
    partition = kmeans.fit_predict(embedding).astype(np.int64)

    # Enforce balance
    partition = enforce_balance(G, partition, k, epsilon)
    return partition


def recursive_bisection(
    G: Graph,
    k: int,
    epsilon: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """
    Recursive spectral bisection for k-way partitioning.

    Uses the Fiedler vector at each level to bisect, recursing until
    we have k parts. For k=4: bisect into 2, then bisect each half.

    Bounded at O(log k) worse than optimal (Simon & Teng, 1997).
    """
    import math
    n = G.n
    if n <= k:
        return np.arange(n, dtype=np.int64) % k

    partition = np.zeros(n, dtype=np.int64)

    # Build a recursive tree of bisections
    # Each entry: (vertex_indices, block_id_start, num_parts)
    queue = [(np.arange(n), 0, k)]
    rng = np.random.RandomState(seed)

    while queue:
        vertices, block_start, num_parts = queue.pop(0)
        if num_parts <= 1 or len(vertices) <= 1:
            partition[vertices] = block_start
            continue

        # Extract subgraph
        sub_G = G.subgraph(vertices)
        if sub_G.n < 2:
            partition[vertices] = block_start
            continue

        # Fiedler bisection
        try:
            _, fiedler = fiedler_vector(sub_G, normalized=True)
        except Exception:
            # Fallback: random bisection
            half = len(vertices) // 2
            perm = rng.permutation(len(vertices))
            partition[vertices[perm[:half]]] = block_start
            partition[vertices[perm[half:]]] = block_start + num_parts // 2
            left_parts = num_parts // 2
            right_parts = num_parts - left_parts
            # Further split if needed
            left_verts = vertices[perm[:half]]
            right_verts = vertices[perm[half:]]
            if left_parts > 1:
                queue.append((left_verts, block_start, left_parts))
            if right_parts > 1:
                queue.append((right_verts, block_start + left_parts, right_parts))
            continue

        # Sort by Fiedler coordinate and split
        sorted_idx = np.argsort(fiedler)
        left_parts = num_parts // 2
        right_parts = num_parts - left_parts

        # Split to balance vertex weights
        total_w = sub_G.vertex_weights.sum()
        target_left = total_w * left_parts / num_parts
        cumsum = np.cumsum(sub_G.vertex_weights[sorted_idx])
        split_point = np.searchsorted(cumsum, target_left)
        split_point = max(1, min(split_point, len(vertices) - 1))

        left_sub = sorted_idx[:split_point]
        right_sub = sorted_idx[split_point:]

        left_verts = vertices[left_sub]
        right_verts = vertices[right_sub]

        if left_parts == 1:
            partition[left_verts] = block_start
        else:
            queue.append((left_verts, block_start, left_parts))

        if right_parts == 1:
            partition[right_verts] = block_start + left_parts
        else:
            queue.append((right_verts, block_start + left_parts, right_parts))

    partition = enforce_balance(G, partition, k, epsilon)
    return partition


def greedy_graph_growing(
    G: Graph,
    k: int,
    epsilon: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """
    Greedy Graph Growing (GGG) for initial partitioning with tight balance.
    """
    rng = np.random.RandomState(seed)
    n = G.n
    partition = np.full(n, -1, dtype=np.int64)
    max_block_weight = (1.0 + epsilon) * np.ceil(G.total_weight / k)
    block_weights = np.zeros(k, dtype=np.float64)

    seeds = rng.choice(n, size=min(k, n), replace=False)
    for i, s in enumerate(seeds):
        partition[s] = i
        block_weights[i] += G.vertex_weights[s]

    from collections import deque
    queues = [deque() for _ in range(k)]
    for i, s in enumerate(seeds):
        nbrs, _ = G.neighbors(s)
        for nb in nbrs:
            if partition[nb] < 0:
                queues[i].append(nb)

    assigned = k
    while assigned < n:
        made_progress = False
        for block in range(k):
            if block_weights[block] >= max_block_weight:
                continue
            attempts = 0
            while queues[block] and attempts < n:
                v = queues[block].popleft()
                attempts += 1
                if partition[v] >= 0:
                    continue
                if block_weights[block] + G.vertex_weights[v] > max_block_weight:
                    continue
                partition[v] = block
                block_weights[block] += G.vertex_weights[v]
                assigned += 1
                made_progress = True
                nbrs, _ = G.neighbors(v)
                for nb in nbrs:
                    if partition[nb] < 0:
                        queues[block].append(nb)
                if block_weights[block] >= max_block_weight:
                    break

        if not made_progress:
            for v in range(n):
                if partition[v] < 0:
                    candidates = np.where(
                        block_weights + G.vertex_weights[v] <= max_block_weight
                    )[0]
                    if len(candidates) == 0:
                        candidates = np.arange(k)
                    block = candidates[np.argmin(block_weights[candidates])]
                    partition[v] = block
                    block_weights[block] += G.vertex_weights[v]
                    assigned += 1
            break

    return partition


def random_partition(
    G: Graph,
    k: int,
    epsilon: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """Random balanced partitioning (baseline)."""
    rng = np.random.RandomState(seed)
    n = G.n
    max_weight = (1.0 + epsilon) * np.ceil(G.total_weight / k)
    partition = np.full(n, -1, dtype=np.int64)
    block_weights = np.zeros(k, dtype=np.float64)

    order = rng.permutation(n)
    for v in order:
        candidates = np.where(
            block_weights + G.vertex_weights[v] <= max_weight
        )[0]
        if len(candidates) == 0:
            candidates = np.arange(k)
        block = candidates[np.argmin(block_weights[candidates])]
        partition[v] = block
        block_weights[block] += G.vertex_weights[v]

    return partition


def multi_start_partition(
    G: Graph,
    k: int,
    epsilon: float = 0.03,
    num_starts: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Portfolio initial partitioning: try all methods with multiple restarts.

    Methods tried:
      - Spectral k-way with k-means++ (most restarts)
      - Recursive spectral bisection
      - Greedy graph growing

    Returns the partition with best cut among balanced solutions.
    """
    from .evaluation import compute_edge_cut, compute_imbalance

    methods = [
        ("spectral", lambda s: spectral_partition(G, k, epsilon, seed=s, n_init=10)),
        ("recursive_bisection", lambda s: recursive_bisection(G, k, epsilon, seed=s)),
        ("ggg", lambda s: greedy_graph_growing(G, k, epsilon, seed=s)),
    ]

    # Coarsest-level feasibility budget: strictly zero imbalance is usually
    # unachievable (and unnecessarily costly) at this level due to aggregated
    # integer vertex weights. We allow up to ~3% slack here; the caller then
    # runs enforce_balance with the true epsilon, and FM + RL refinement
    # during uncoarsening tightens the partition down to spec. This mirrors
    # the METIS/kaHIP design choice of relaxed-balance initial partitioning.
    budget = max(epsilon, 0.03)

    best_cut = float("inf")
    best_partition = None
    best_any_cut = float("inf")
    best_any_partition = None

    for name, method in methods:
        n_tries = num_starts if name == "spectral" else max(num_starts // 3, 3)
        offset = _METHOD_SEED_OFFSETS.get(name, 0)
        for i in range(n_tries):
            try:
                part = method(seed + i * 7 + offset)
                imbalance = compute_imbalance(G, part, k)
                cut = compute_edge_cut(G, part)

                if cut < best_any_cut:
                    best_any_cut = cut
                    best_any_partition = part.copy()

                if imbalance <= budget + 1e-9 and cut < best_cut:
                    best_cut = cut
                    best_partition = part.copy()
            except Exception:
                continue

    if best_partition is None:
        best_partition = (best_any_partition
                          if best_any_partition is not None
                          else random_partition(G, k, epsilon, seed))

    return best_partition
