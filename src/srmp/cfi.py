"""
Cai-Fuerer-Immerman (CFI) Graph Construction
==============================================
Generates pairs of non-isomorphic graphs that are indistinguishable by
k-dimensional Weisfeiler-Leman colour refinement, for use in Theorem T1
(WL vs GNN refinement separation).

Construction (Cai, Fuerer, Immerman, 1992, simplified 1-WL variant):
  Start from a connected base graph H = (V_H, E_H).
  For each vertex v in V_H with degree d_v:
    Create a "vertex gadget" X_v consisting of:
      - Inner vertices x_{v, S} indexed by even-cardinality subsets S of
        the edges incident to v (there are 2^{d_v - 1} of these).
      - Edge-endpoint vertices y_{v, e} for each edge e incident to v.
      - Edges: x_{v, S} -- y_{v, e} iff e in S.
  For each edge e = (u, v) in E_H:
    Link the two edge-endpoint vertices y_{u, e} and y_{v, e} by either
      - a straight matching: y_{u, e} -- y_{v, e}              (untwisted)
      - a twisted matching:  swap one endpoint to flip parity  (twisted)
    One twisted edge flips the global XOR-parity of the gadget system,
    producing a graph non-isomorphic to the untwisted version but
    indistinguishable from it by 1-WL.

Key property for Theorem T1:
  Let CFI(H) be the untwisted graph and CFI'(H) the graph with exactly one
  edge twisted. Then:
    (i)  CFI(H) !~ CFI'(H)      [non-isomorphic, via parity argument]
    (ii) 1-WL(CFI(H)) = 1-WL(CFI'(H))  [1-WL indistinguishable]
    (iii) A sign-equivariant spectral positional encoding distinguishes
         them via the Fiedler vector's parity assignment.

Complexity:
  |V(CFI(H))| = sum_{v in H} (2^{d_v - 1} + d_v)
  |E(CFI(H))| = sum_{v in H} d_v * 2^{d_v - 2}  + |E_H|
  Keep max-degree of H bounded (<= 4) to avoid exponential blowup.

References:
  Cai, J.Y., Fuerer, M., Immerman, N. (1992). An optimal lower bound on
    the number of variables for graph identification. Combinatorica 12(4).
  Sato, R. (2020). A survey on the expressive power of graph neural
    networks. arXiv:2003.04078. Section 4.2 describes CFI for GNNs.
  Morris, C. et al. (2019). Weisfeiler and Leman go neural: higher-order
    graph neural networks. AAAI 2019.
"""
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .graph import Graph


# =========================================================================
# Base graph utilities
# =========================================================================

def _adjacency_list(G: Graph) -> List[List[int]]:
    """Return a plain adjacency list from a Graph (O(m))."""
    adj = [[] for _ in range(G.n)]
    coo = G.adj.tocoo()
    for u, v in zip(coo.row, coo.col):
        if u < v:
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
    return adj


def _edge_list(G: Graph) -> List[Tuple[int, int]]:
    """Return a canonical edge list (u < v)."""
    coo = G.adj.tocoo()
    edges = []
    seen = set()
    for u, v in zip(coo.row, coo.col):
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        if a == b:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b))
    return sorted(edges)


# =========================================================================
# CFI construction
# =========================================================================

def _build_cfi(
    base: Graph,
    twisted_edges: Optional[List[Tuple[int, int]]] = None,
    max_base_degree: int = 5,
) -> Graph:
    """
    Build the CFI graph from a base graph using the standard two-bit
    construction (Cai-Fuerer-Immerman 1992).

    Per base vertex v with incident edges E(v) = (e_1, ..., e_{d_v}):
      - Inner vertices x_{v, S} for each even-cardinality S subseteq E(v).
        There are 2^{d_v - 1} of these.
      - Bit vertices b^0_{v, e_i} and b^1_{v, e_i} for each incident edge.
        There are 2 * d_v of these.
      - Connect x_{v, S} to b^1_{v, e_i} if e_i in S, else to b^0_{v, e_i}.

    Per base edge e = (u, v):
      - Untwisted: match b^0_{u,e} -- b^0_{v,e} and b^1_{u,e} -- b^1_{v,e}.
      - Twisted:   match b^0_{u,e} -- b^1_{v,e} and b^1_{u,e} -- b^0_{v,e}.

    Both graphs have identical degree sequences and are 1-WL equivalent;
    they differ only in global parity (Cai et al. Theorem 3.1). A twist
    on an odd number of edges produces a graph non-isomorphic to the
    untwisted one; an even number yields an isomorphic variant.

    Parameters
    ----------
    base : Graph
        Connected base graph. Max degree <= max_base_degree.
    twisted_edges : list of (u, v) pairs or None
        Edges of the base whose matching is twisted.
    max_base_degree : int
        Hard cap to prevent 2^d blowup (d > 5 gets expensive fast).
    """
    adj_list = _adjacency_list(base)
    edges = _edge_list(base)

    if base.n > 0:
        max_deg = max((len(nbrs) for nbrs in adj_list), default=0)
        if max_deg > max_base_degree:
            raise ValueError(
                f"CFI: base graph has max degree {max_deg} > "
                f"{max_base_degree} (exponential blowup)."
            )

    twisted_set = set()
    if twisted_edges:
        for u, v in twisted_edges:
            a, b = (u, v) if u < v else (v, u)
            twisted_set.add((a, b))

    # Deterministic per-vertex incident-edge ordering.
    incident_sorted: List[List[Tuple[int, int]]] = []
    for v in range(base.n):
        lst = sorted((v, u) if v < u else (u, v) for u in adj_list[v])
        incident_sorted.append(lst)

    # Assign IDs.
    inner_id: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    bit_id: Dict[Tuple[int, Tuple[int, int], int], int] = {}
    counter = 0

    for v in range(base.n):
        incident = incident_sorted[v]
        d_v = len(incident)
        for r in range(0, d_v + 1, 2):
            for subset in combinations(range(d_v), r):
                key = (v, tuple(incident[i] for i in subset))
                inner_id[key] = counter
                counter += 1
        for e in incident:
            for bit in (0, 1):
                bit_id[(v, e, bit)] = counter
                counter += 1

    n_cfi = counter
    row: List[int] = []
    col: List[int] = []

    def add(a: int, b: int) -> None:
        row.append(a); col.append(b)
        row.append(b); col.append(a)

    # (i) inner x_{v, S} -- bit vertices per gadget
    for v in range(base.n):
        incident = incident_sorted[v]
        d_v = len(incident)
        for r in range(0, d_v + 1, 2):
            for subset in combinations(range(d_v), r):
                x_id = inner_id[(v, tuple(incident[i] for i in subset))]
                subset_set = set(subset)
                for i, e_i in enumerate(incident):
                    bit = 1 if i in subset_set else 0
                    add(x_id, bit_id[(v, e_i, bit)])

    # (ii) base-edge matchings (untwisted or twisted)
    for (u, v) in edges:
        e = (u, v)
        if e in twisted_set:
            add(bit_id[(u, e, 0)], bit_id[(v, e, 1)])
            add(bit_id[(u, e, 1)], bit_id[(v, e, 0)])
        else:
            add(bit_id[(u, e, 0)], bit_id[(v, e, 0)])
            add(bit_id[(u, e, 1)], bit_id[(v, e, 1)])

    data = np.ones(len(row), dtype=np.float64)
    W = sp.csr_matrix((data, (row, col)), shape=(n_cfi, n_cfi))
    W.sum_duplicates()
    return Graph(W)


# =========================================================================
# Public API
# =========================================================================

def generate_cfi_pair(
    base: Graph,
    twist_count: int = 1,
    seed: int = 0,
    max_base_degree: int = 5,
) -> Tuple[Graph, Graph]:
    """
    Generate a (G_untwisted, G_twisted) CFI pair.

    The two graphs are non-isomorphic but 1-WL-indistinguishable (for
    connected bases). ``twist_count`` controls how many base edges carry
    a twist --- odd counts change the global parity, even counts do not.

    Parameters
    ----------
    base : Graph
        Connected base graph with bounded maximum degree (<= 5 recommended).
    twist_count : int
        Number of base edges to twist. Must be odd to guarantee
        non-isomorphism via parity; the function enforces this by taking
        ``twist_count = 2 * (twist_count // 2) + 1``.
    seed : int
        Controls which edges are twisted.
    max_base_degree : int
        Safety cap.

    Returns
    -------
    G0 : Graph   -- untwisted CFI(base)
    G1 : Graph   -- twisted CFI'(base)
    """
    edges = _edge_list(base)
    if len(edges) == 0:
        raise ValueError("CFI: base graph has no edges.")

    # Enforce odd twist count.
    k = max(1, twist_count)
    if k % 2 == 0:
        k += 1
    k = min(k, len(edges))

    rng = np.random.RandomState(seed)
    picked = rng.choice(len(edges), size=k, replace=False)
    twisted_edges = [edges[i] for i in picked]

    G0 = _build_cfi(base, twisted_edges=None, max_base_degree=max_base_degree)
    G1 = _build_cfi(
        base, twisted_edges=twisted_edges, max_base_degree=max_base_degree
    )
    return G0, G1


def generate_cfi_from_cycle(
    cycle_length: int,
    twist_count: int = 1,
    seed: int = 0,
) -> Tuple[Graph, Graph]:
    """
    Convenience constructor: base = simple cycle C_n.

    Produces a CFI pair where the base has all degrees = 2, giving gadgets
    of size 2 + 2 = 4 per base vertex. Total CFI size is ~4n, cheap to
    enumerate at scale.
    """
    n = int(cycle_length)
    if n < 3:
        raise ValueError("Cycle length must be >= 3.")
    row, col = [], []
    for i in range(n):
        j = (i + 1) % n
        row += [i, j]
        col += [j, i]
    data = np.ones(len(row), dtype=np.float64)
    W = sp.csr_matrix((data, (row, col)), shape=(n, n))
    W.sum_duplicates()
    base = Graph(W)
    return generate_cfi_pair(base, twist_count=twist_count, seed=seed)


def generate_cfi_from_3regular(
    n: int,
    twist_count: int = 1,
    seed: int = 0,
) -> Tuple[Graph, Graph]:
    """
    CFI pair with a random 3-regular base graph. Harder for 1-WL than the
    cycle variant; recommended for the primary T1 experiment.

    Requires ``n`` even (3-regular graphs exist only for even n).

    The base is rejection-sampled until it is simple AND connected --- the
    1-WL/non-iso parity argument that makes CFI pairs interesting requires
    a connected base.
    """
    if n % 2 != 0:
        raise ValueError("3-regular graph requires n even.")
    from scipy.sparse.csgraph import connected_components as _cc
    rng = np.random.RandomState(seed)

    # Configuration model with rejection for simple AND connected graphs
    base = None
    for attempt in range(100):
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        matches = stubs.reshape(-1, 2)
        ok = True
        edge_set = set()
        for u, v in matches:
            if u == v:
                ok = False; break
            a, b = (int(u), int(v)) if u < v else (int(v), int(u))
            if (a, b) in edge_set:
                ok = False; break
            edge_set.add((a, b))
        if not ok:
            continue
        # Build trial graph and require connectedness
        row_t, col_t = [], []
        for (u, v) in edge_set:
            row_t += [u, v]; col_t += [v, u]
        data_t = np.ones(len(row_t), dtype=np.float64)
        W_trial = sp.csr_matrix((data_t, (row_t, col_t)), shape=(n, n))
        W_trial.sum_duplicates()
        n_cc, _ = _cc(W_trial, directed=False)
        if n_cc == 1:
            base = Graph(W_trial)
            break
    if base is None:
        raise RuntimeError(
            "Could not sample a simple, connected 3-regular graph after 100 tries."
        )
    return generate_cfi_pair(base, twist_count=twist_count, seed=seed)
