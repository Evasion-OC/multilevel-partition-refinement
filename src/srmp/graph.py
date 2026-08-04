"""
Graph Data Structure and I/O
=============================
Supports METIS/Walshaw format graphs. Uses CSR (Compressed Sparse Row)
for efficient adjacency traversal and O(1) degree lookup.
"""
import numpy as np
from scipy import sparse
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# Use AVX-512 C kernel for boundary detection when available
try:
    from .native.fast_kernels import fast_boundary_vertices as _fast_bv
except ImportError:
    _fast_bv = None


class Graph:
    """
    Weighted undirected graph using CSR adjacency representation.

    Attributes
    ----------
    n : int
        Number of vertices.
    m : int
        Number of edges.
    adj : scipy.sparse.csr_matrix
        Weighted adjacency matrix (n x n).
    vertex_weights : np.ndarray
        Vertex weights, shape (n,). Default all ones.
    """

    def __init__(
        self,
        adj: sparse.csr_matrix,
        vertex_weights: Optional[np.ndarray] = None,
    ):
        assert adj.shape[0] == adj.shape[1], "Adjacency must be square"
        self.n = adj.shape[0]
        self.adj = adj.tocsr()
        # Number of edges (each undirected edge stored twice in CSR)
        self.m = self.adj.nnz // 2
        if vertex_weights is None:
            self.vertex_weights = np.ones(self.n, dtype=np.float64)
        else:
            self.vertex_weights = vertex_weights.astype(np.float64)
        # Cache degree array
        self._degrees = None
        # Cache COO decomposition (avoids repeated conversion in hot paths)
        self._coo_cache = None

    @property
    def degrees(self) -> np.ndarray:
        """Weighted degree of each vertex."""
        if self._degrees is None:
            self._degrees = np.asarray(self.adj.sum(axis=1)).flatten()
        return self._degrees

    @property
    def total_weight(self) -> float:
        """Sum of all vertex weights."""
        return float(self.vertex_weights.sum())

    @property
    def coo(self):
        """Cached COO decomposition - avoids repeated tocoo() in hot paths."""
        if self._coo_cache is None:
            self._coo_cache = self.adj.tocoo()
        return self._coo_cache

    def neighbors(self, v: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (neighbor_indices, edge_weights) for vertex v.
        Uses CSR structure for O(deg(v)) access.
        """
        start = self.adj.indptr[v]
        end = self.adj.indptr[v + 1]
        return self.adj.indices[start:end], self.adj.data[start:end]

    def degree(self, v: int) -> float:
        """Weighted degree of vertex v."""
        return self.degrees[v]

    def edge_weight(self, u: int, v: int) -> float:
        """Weight of edge (u, v). Returns 0 if not adjacent."""
        return self.adj[u, v]

    def boundary_vertices(self, partition: np.ndarray) -> np.ndarray:
        """
        Return array of boundary vertex indices.
        A vertex is on the boundary if any neighbour is in a different block.
        """
        boundary = np.zeros(self.n, dtype=bool)
        for v in range(self.n):
            nbrs, _ = self.neighbors(v)
            if len(nbrs) > 0 and np.any(partition[nbrs] != partition[v]):
                boundary[v] = True
        return np.where(boundary)[0]

    def boundary_vertices_fast(self, partition: np.ndarray) -> np.ndarray:
        """Boundary detection - uses AVX-512 C kernel when available."""
        if _fast_bv is not None:
            return _fast_bv(self, partition)
        # Fallback: vectorised sparse operations
        rows, cols = self.adj.nonzero()
        cross = partition[rows] != partition[cols]
        boundary_mask = np.zeros(self.n, dtype=bool)
        boundary_mask[rows[cross]] = True
        return np.where(boundary_mask)[0]

    def subgraph(self, vertices: np.ndarray) -> 'Graph':
        """Extract induced subgraph on given vertices."""
        idx = np.sort(vertices)
        sub_adj = self.adj[idx][:, idx]
        sub_weights = self.vertex_weights[idx]
        return Graph(sub_adj, sub_weights)

    def copy(self) -> 'Graph':
        """Deep copy of the graph."""
        return Graph(self.adj.copy(), self.vertex_weights.copy())

    def __repr__(self) -> str:
        return f"Graph(n={self.n}, m={self.m})"


def read_metis(filepath: str) -> Graph:
    """
    Read a graph in METIS/Walshaw format.

    Format:
        Line 1: n m [fmt] [ncon]
        Lines 2..n+1: adjacency list for each vertex (1-indexed)

    fmt bits:
        bit 0 (1): edge weights present
        bit 1 (2): vertex weights present

    Returns
    -------
    Graph
        The loaded graph.
    """
    filepath = Path(filepath)
    rows, cols, data = [], [], []
    vertex_weights = []

    with open(filepath, 'r') as f:
        # Skip comment lines
        line = f.readline()
        while line.startswith('%') or line.startswith('#'):
            line = f.readline()

        # Parse header
        header = line.strip().split()
        n = int(header[0])
        m_declared = int(header[1])
        fmt = int(header[2]) if len(header) > 2 else 0
        ncon = int(header[3]) if len(header) > 3 else 1

        has_edge_weights = bool(fmt & 1)
        has_vertex_weights = bool(fmt & 2)

        for v in range(n):
            line = f.readline()
            while line.startswith('%') or line.startswith('#'):
                line = f.readline()

            tokens = line.strip().split()
            idx = 0

            # Parse vertex weights
            if has_vertex_weights:
                vw = []
                for _ in range(ncon):
                    vw.append(float(tokens[idx]))
                    idx += 1
                vertex_weights.append(vw[0])  # Use first constraint
            else:
                vertex_weights.append(1.0)

            # Parse adjacency list
            while idx < len(tokens):
                neighbor = int(tokens[idx]) - 1  # Convert to 0-indexed
                idx += 1

                if has_edge_weights:
                    if idx < len(tokens):
                        ew = float(tokens[idx])
                        idx += 1
                    else:
                        ew = 1.0
                else:
                    ew = 1.0

                rows.append(v)
                cols.append(neighbor)
                data.append(ew)

    adj = sparse.csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int64),
          np.array(cols, dtype=np.int64))),
        shape=(n, n)
    )
    vw = np.array(vertex_weights, dtype=np.float64)

    return Graph(adj, vw)


def write_metis(graph: Graph, filepath: str, partition: Optional[np.ndarray] = None):
    """
    Write graph in METIS format and optionally a partition file.
    """
    filepath = Path(filepath)
    has_vw = not np.allclose(graph.vertex_weights, 1.0)
    has_ew = True  # Always write edge weights

    fmt = 0
    if has_ew:
        fmt |= 1
    if has_vw:
        fmt |= 2

    with open(filepath, 'w') as f:
        f.write(f"{graph.n} {graph.m} {fmt}\n")
        for v in range(graph.n):
            parts = []
            if has_vw:
                parts.append(str(int(graph.vertex_weights[v])))
            nbrs, weights = graph.neighbors(v)
            for nb, w in zip(nbrs, weights):
                parts.append(str(nb + 1))  # 1-indexed
                if has_ew:
                    parts.append(str(int(w)))
            f.write(' '.join(parts) + '\n')

    if partition is not None:
        part_path = filepath.with_suffix('.part')
        with open(part_path, 'w') as f:
            for p in partition:
                f.write(f"{p}\n")


def generate_ba_graph(n: int, m_attach: int = 3, seed: int = 42) -> Graph:
    """Generate a Barabasi-Albert preferential attachment graph."""
    rng = np.random.RandomState(seed)

    # Start with a complete graph on m_attach+1 vertices
    m0 = m_attach + 1
    edges_seen = set()
    rows, cols = [], []
    for i in range(m0):
        for j in range(i + 1, m0):
            edges_seen.add((i, j))
            rows.extend([i, j])
            cols.extend([j, i])

    degrees = np.zeros(n, dtype=np.float64)
    for i in range(m0):
        degrees[i] = m0 - 1

    # Preferential attachment
    for v in range(m0, n):
        total_deg = degrees[:v].sum()
        if total_deg == 0:
            probs = np.ones(v) / v
        else:
            probs = degrees[:v] / total_deg

        targets = rng.choice(v, size=min(m_attach, v), replace=False, p=probs)
        for t in targets:
            edge = (min(v, t), max(v, t))
            if edge in edges_seen:
                continue  # Skip duplicate edge
            edges_seen.add(edge)
            rows.extend([v, t])
            cols.extend([t, v])
            degrees[v] += 1
            degrees[t] += 1

    data = np.ones(len(rows), dtype=np.float64)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return Graph(adj)


def generate_er_graph(n: int, p: float = 0.01, seed: int = 42) -> Graph:
    """Generate an Erdos-Renyi G(n,p) random graph."""
    rng = np.random.RandomState(seed)
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                rows.extend([i, j])
                cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.float64)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return Graph(adj)


def generate_sbm_graph(
    n: int,
    k: int = 2,
    p_in: float = 0.05,
    p_out: float = 0.005,
    block_sizes: Optional[List[int]] = None,
    seed: int = 42,
) -> Graph:
    """
    Generate a Stochastic Block Model graph.

    Edges are Bernoulli with probability ``p_in`` within a block and
    ``p_out`` across blocks. By default blocks are as balanced as
    possible (``n // k`` vertices each, with the first ``n % k`` blocks
    receiving one extra vertex).

    The ratio ``p_in / p_out`` is the controllable "community
    strength" axis used for intervention analysis of alpha(G):
    large ratio -> clean communities (easy for FM), small ratio ->
    weak communities (where the proposal hypothesises RL should help).

    The *ground-truth* block assignment is attached as
    ``Graph.true_blocks`` on the returned object for downstream NMI
    analysis. It is not used as a vertex weight.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Number of blocks.
    p_in, p_out : float
        Within-block and between-block edge probabilities.
    block_sizes : list of int, optional
        Explicit block sizes (must sum to ``n``). Default: balanced.
    seed : int
        RNG seed.

    References
    ----------
    Holland, Laskey & Leinhardt (1983); Abbe (2018) for the recovery
    thresholds governing when the Fiedler vector can detect the planted
    partition.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
        raise ValueError("probabilities must lie in [0, 1]")

    rng = np.random.RandomState(seed)

    if block_sizes is None:
        base = n // k
        rem = n % k
        block_sizes = [base + (1 if i < rem else 0) for i in range(k)]
    block_sizes = list(block_sizes)
    if sum(block_sizes) != n:
        raise ValueError(
            f"block_sizes sum ({sum(block_sizes)}) != n ({n})"
        )

    # Vertex -> block assignment
    true_blocks = np.empty(n, dtype=np.int64)
    offset = 0
    for b, sz in enumerate(block_sizes):
        true_blocks[offset:offset + sz] = b
        offset += sz

    # Vectorised edge sampling with Bernoulli(p_in) or Bernoulli(p_out).
    # Sampling an n x n mask would cost O(n^2); we instead sample the
    # number of edges per (block_i, block_j) pair and place them
    # uniformly at random without replacement within that pair.
    rows: List[int] = []
    cols: List[int] = []
    block_starts = np.cumsum([0] + block_sizes)

    for bi in range(k):
        s_i, e_i = block_starts[bi], block_starts[bi + 1]
        sz_i = e_i - s_i
        # Within-block edges (upper triangle of the block)
        if sz_i >= 2 and p_in > 0:
            num_pairs = sz_i * (sz_i - 1) // 2
            num_edges = rng.binomial(num_pairs, p_in)
            if num_edges > 0:
                chosen = rng.choice(num_pairs, size=num_edges, replace=False)
                # Map linear pair index -> (u, v) with u < v
                u = np.floor(
                    (1 + np.sqrt(1 + 8 * chosen)) / 2
                ).astype(np.int64)
                v = chosen - u * (u - 1) // 2
                rows.extend((u + s_i).tolist())
                cols.extend((v + s_i).tolist())
                rows.extend((v + s_i).tolist())
                cols.extend((u + s_i).tolist())
        # Cross-block edges (bi, bj) with bj > bi
        for bj in range(bi + 1, k):
            s_j, e_j = block_starts[bj], block_starts[bj + 1]
            sz_j = e_j - s_j
            if p_out <= 0:
                continue
            num_pairs = sz_i * sz_j
            num_edges = rng.binomial(num_pairs, p_out)
            if num_edges > 0:
                chosen = rng.choice(num_pairs, size=num_edges, replace=False)
                u = chosen // sz_j
                v = chosen % sz_j
                rows.extend((u + s_i).tolist())
                cols.extend((v + s_j).tolist())
                rows.extend((v + s_j).tolist())
                cols.extend((u + s_i).tolist())

    if not rows:
        # Empty graph: guarantee valid CSR.
        adj = sparse.csr_matrix((n, n), dtype=np.float64)
    else:
        data = np.ones(len(rows), dtype=np.float64)
        adj = sparse.csr_matrix(
            (data, (np.array(rows), np.array(cols))), shape=(n, n)
        )
        # Collapse accidental duplicates from the sampling step.
        adj.sum_duplicates()
        adj.data = np.ones_like(adj.data)

    G = Graph(adj)
    G.true_blocks = true_blocks
    G.sbm_params = {
        "k_planted": k,
        "p_in": p_in,
        "p_out": p_out,
        "block_sizes": block_sizes,
    }
    return G


def generate_geometric_graph(
    n: int,
    d: int = 2,
    k_nn: int = 6,
    seed: int = 42,
) -> Graph:
    """
    Random geometric graph: n points sampled uniformly in the unit
    d-dimensional hypercube, each connected to its ``k_nn`` nearest
    neighbours (symmetrised).

    The dimension ``d`` is a controllable structural axis:
      - d = 2       : planar-like, large separators survive
      - d = 3 .. 5  : moderate curvature, tw grows
      - d >= 10     : approaches the "curse of dimensionality" regime
                      where k-NN graphs become quasi-random

    References
    ----------
    Penrose (2003): Random geometric graphs; Friedrich-Krohmer (2015)
    for spectral properties in high dimensions.
    """
    if n < k_nn + 1:
        raise ValueError(f"n ({n}) must exceed k_nn ({k_nn})")
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(seed)
    points = rng.rand(n, d)

    tree = cKDTree(points)
    # k_nn + 1 because the first neighbour of each point is itself.
    _, nbrs = tree.query(points, k=k_nn + 1)

    rows: List[int] = []
    cols: List[int] = []
    for i in range(n):
        for j in nbrs[i, 1:]:
            rows.append(i)
            cols.append(int(j))
    # Symmetrise
    all_rows = rows + cols
    all_cols = cols + rows
    data = np.ones(len(all_rows), dtype=np.float64)
    adj = sparse.csr_matrix(
        (data, (np.array(all_rows), np.array(all_cols))), shape=(n, n)
    )
    adj.sum_duplicates()
    adj.data = np.ones_like(adj.data)

    G = Graph(adj)
    G.geometric_dim = d
    G.geometric_knn = k_nn
    return G


def generate_grid_graph(rows: int, cols: int) -> Graph:
    """
    2D rectangular grid graph (a pure mesh baseline). Useful as the
    "tw = Theta(sqrt(n))" extreme for treewidth-ratio ablations.
    """
    n = rows * cols
    r_idx: List[int] = []
    c_idx: List[int] = []
    for i in range(rows):
        for j in range(cols):
            v = i * cols + j
            if j + 1 < cols:
                u = v + 1
                r_idx += [v, u]
                c_idx += [u, v]
            if i + 1 < rows:
                u = v + cols
                r_idx += [v, u]
                c_idx += [u, v]
    data = np.ones(len(r_idx), dtype=np.float64)
    adj = sparse.csr_matrix(
        (data, (np.array(r_idx), np.array(c_idx))), shape=(n, n)
    )
    return Graph(adj)


def generate_delaunay_graph(n: int, seed: int = 42) -> Graph:
    """Generate a Delaunay triangulation graph from random 2D points."""
    from scipy.spatial import Delaunay
    rng = np.random.RandomState(seed)
    points = rng.rand(n, 2)
    tri = Delaunay(points)

    rows, cols = [], []
    edges_seen = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                if (u, v) not in edges_seen:
                    edges_seen.add((u, v))
                    edges_seen.add((v, u))
                    rows.extend([u, v])
                    cols.extend([v, u])

    data = np.ones(len(rows), dtype=np.float64)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return Graph(adj)
