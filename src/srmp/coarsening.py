"""
Multilevel Graph Coarsening
============================
Implements multiple coarsening strategies:
  - Heavy-Edge Matching (HEM)
  - Sorted HEM (SHEM / METIS-style)
  - Expansion*2 edge rating with GPA matching (KaHIP)
  - Label Propagation clustering (size-constrained)
  - Community-aware coarsening (Heuer & Schlag, SEA 2017)

Thesis linkage (C2):
  Conjecture C2 is the head-to-head comparison of expansion*2 versus label
  propagation on configuration-model graphs indexed by (alpha_hill, cv).
  The fidelity measurements that power the C2 phase-landscape figure are
  computed by srmp.thesis_features.coarsening_fidelity_by_method, which
  invokes the two matchers here and measures Fiedler-cut distortion.

v3 changes:
  - Expansion*2 edge rating: r(u,v) = w(u,v)^2 / (c(u)*c(v))
  - Global Path Algorithm (GPA) matching for better matching quality
  - Community-aware coarsening restricts contractions within communities
  - Partition-guided coarsening for V-cycles and evolutionary combine

References:
  - Karypis & Kumar (SIAM J. Sci. Comput., 1998): HEM coarsening
  - Sanders & Schulz (ESA, 2011): Expansion*2, LP coarsening in KaHIP
  - Abou-Rjeili & Karypis (IPDPS, 2006): Power-law graph coarsening
  - Heuer & Schlag (SEA, 2017): Community-aware coarsening
"""
import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple, Set
from .graph import Graph


class CoarseningLevel:
    """
    Stores the mapping between a fine graph and its coarsened version.

    Attributes
    ----------
    fine_graph : Graph
        The finer (larger) graph.
    coarse_graph : Graph
        The coarser (smaller) graph.
    mapping : np.ndarray
        mapping[v] = supernode in coarse_graph that v maps to.
    inverse_mapping : List[List[int]]
        inverse_mapping[u] = list of fine vertices collapsed into supernode u.
    """

    def __init__(
        self,
        fine_graph: Graph,
        coarse_graph: Graph,
        mapping: np.ndarray,
    ):
        self.fine_graph = fine_graph
        self.coarse_graph = coarse_graph
        self.mapping = mapping.astype(np.int64)

        # Build inverse mapping
        n_coarse = coarse_graph.n
        self.inverse_mapping: List[List[int]] = [[] for _ in range(n_coarse)]
        for v in range(fine_graph.n):
            self.inverse_mapping[self.mapping[v]].append(v)

    def prolong_partition(self, coarse_partition: np.ndarray) -> np.ndarray:
        """
        Project a coarse partition back to the fine graph.
        Each fine vertex inherits the partition of its supernode.
        """
        return coarse_partition[self.mapping]


# =========================================================================
# Edge rating functions
# =========================================================================

def compute_expansion2_ratings(G: Graph) -> dict:
    """
    Compute expansion*2 edge ratings: r(u,v) = w(u,v)^2 / (c(u) * c(v)).

    This rating from KaHIP favors contracting heavy edges between light
    nodes, naturally preserving balance during coarsening.
    Yields 5-9% better cuts than simple HEM.

    Returns dict mapping (u,v) -> rating for each edge.
    """
    coo = G.adj.tocoo()
    rows, cols, weights = coo.row, coo.col, coo.data

    # Only process upper triangle (u < v)
    upper = rows < cols
    u_arr = rows[upper]
    v_arr = cols[upper]
    w_arr = weights[upper]

    # Vectorized rating computation
    cu = G.vertex_weights[u_arr]
    cv = G.vertex_weights[v_arr]
    denom = np.maximum(cu * cv, 1e-12)
    r_arr = (w_arr * w_arr) / denom

    # Build dict from arrays
    ratings = {}
    for i in range(len(u_arr)):
        u, v, r = int(u_arr[i]), int(v_arr[i]), float(r_arr[i])
        ratings[(u, v)] = r
        ratings[(v, u)] = r
    return ratings


# =========================================================================
# Matching algorithms
# =========================================================================

def heavy_edge_matching(G: Graph, rng: np.random.RandomState) -> np.ndarray:
    """
    Compute a maximal matching using Heavy-Edge Matching (HEM).

    Visit vertices in random order. Match each unmatched vertex to its
    unmatched neighbour with the heaviest connecting edge.
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    order = rng.permutation(n)

    for v in order:
        if matched[v]:
            continue
        nbrs, weights = G.neighbors(v)
        if len(nbrs) == 0:
            continue
        best_weight = -1.0
        best_nb = -1
        for nb, w in zip(nbrs, weights):
            if not matched[nb] and nb != v and w > best_weight:
                best_weight = w
                best_nb = nb
        if best_nb >= 0:
            match[v] = best_nb
            match[best_nb] = v
            matched[v] = True
            matched[best_nb] = True

    return match


def sorted_heavy_edge_matching(G: Graph, rng: np.random.RandomState) -> np.ndarray:
    """
    Sorted Heavy-Edge Matching (SHEM) as used in METIS 5.x.
    Sort vertices by degree (ascending) before matching.
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    order = np.argsort(G.degrees)

    for v in order:
        if matched[v]:
            continue
        nbrs, weights = G.neighbors(v)
        if len(nbrs) == 0:
            continue
        best_weight = -1.0
        best_nb = -1
        for nb, w in zip(nbrs, weights):
            if not matched[nb] and nb != v and w > best_weight:
                best_weight = w
                best_nb = nb
        if best_nb >= 0:
            match[v] = best_nb
            match[best_nb] = v
            matched[v] = True
            matched[best_nb] = True

    return match


def expansion2_matching(
    G: Graph,
    rng: np.random.RandomState,
    use_gpa: bool = True,
) -> np.ndarray:
    """
    Matching using expansion*2 edge ratings.

    With GPA (Global Path Algorithm): build paths/even cycles through
    edges sorted by decreasing rating, solve each optimally.
    Without GPA: greedy matching on rating-sorted edges.

    Parameters
    ----------
    G : Graph
    rng : RandomState
    use_gpa : bool
        If True, use GPA for better matching quality (1/2-approximation
        to maximum weight matching). Otherwise greedy.

    Returns
    -------
    match : np.ndarray
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)

    # Compute expansion*2 ratings
    ratings = compute_expansion2_ratings(G)

    if use_gpa:
        return _gpa_matching(G, ratings, rng)
    else:
        # Greedy: visit vertices in random order, pick neighbor with best rating
        order = rng.permutation(n)
        for v in order:
            if matched[v]:
                continue
            nbrs, _ = G.neighbors(v)
            if len(nbrs) == 0:
                continue
            best_rating = -1.0
            best_nb = -1
            for nb in nbrs:
                if not matched[nb] and nb != v:
                    r = ratings.get((v, nb), 0.0)
                    if r > best_rating:
                        best_rating = r
                        best_nb = nb
            if best_nb >= 0:
                match[v] = best_nb
                match[best_nb] = v
                matched[v] = True
                matched[best_nb] = True
        return match


def _gpa_matching(
    G: Graph,
    ratings: dict,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Global Path Algorithm (GPA) matching.

    1. Sort edges by decreasing rating.
    2. Build paths and even-length cycles by greedily adding edges.
    3. For each path/cycle, compute optimal matching via dynamic programming.

    Achieves a 1/2-approximation to maximum weight matching.
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)

    # Collect all edges with ratings
    edges = []
    coo = G.adj.tocoo()
    for idx in range(coo.nnz):
        u, v = int(coo.row[idx]), int(coo.col[idx])
        if u < v:
            r = ratings.get((u, v), 0.0)
            edges.append((r, u, v))

    # Sort by decreasing rating
    edges.sort(key=lambda x: -x[0])

    # Build path/cycle structure
    # Each vertex has at most degree 2 in the path structure
    path_degree = np.zeros(n, dtype=np.int32)
    path_adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for r, u, v in edges:
        if path_degree[u] < 2 and path_degree[v] < 2:
            # Check: adding (u,v) won't create an odd cycle
            # For simplicity in implementation, we just limit degree to 2
            # and handle paths. Full GPA with cycle detection is complex.
            path_adj[u].append((v, r))
            path_adj[v].append((u, r))
            path_degree[u] += 1
            path_degree[v] += 1

    # Find path endpoints (degree 1 vertices) and isolated edges
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start] or path_degree[start] == 0:
            continue

        # Trace path from start
        path = []
        path_ratings = []
        current = start
        prev = -1

        while True:
            visited[current] = True
            path.append(current)
            next_node = -1
            next_rating = 0.0
            for nb, r in path_adj[current]:
                if nb != prev and not (visited[nb] and nb != start):
                    next_node = nb
                    next_rating = r
                    break
            if next_node < 0:
                break
            path_ratings.append(next_rating)
            if next_node == start:
                # Cycle detected
                break
            prev = current
            current = next_node

        # Optimal matching on the path via DP
        # For a path of m edges, pick non-adjacent edges maximizing total rating
        m = len(path_ratings)
        if m == 0:
            continue
        if m == 1:
            u, v = path[0], path[1]
            match[u] = v
            match[v] = u
            continue

        # DP: dp[i] = max total rating using edges 0..i
        # dp[i] = max(dp[i-1], dp[i-2] + rating[i])
        dp = np.zeros(m, dtype=np.float64)
        take = np.zeros(m, dtype=bool)
        dp[0] = path_ratings[0]
        take[0] = True
        if m > 1:
            if path_ratings[1] > dp[0]:
                dp[1] = path_ratings[1]
                take[1] = True
            else:
                dp[1] = dp[0]
                take[1] = False

        for i in range(2, m):
            if dp[i - 2] + path_ratings[i] > dp[i - 1]:
                dp[i] = dp[i - 2] + path_ratings[i]
                take[i] = True
            else:
                dp[i] = dp[i - 1]
                take[i] = False

        # Backtrack to find which edges to include in matching
        i = m - 1
        while i >= 0:
            if take[i]:
                u = path[i]
                v = path[i + 1] if i + 1 < len(path) else path[0]
                match[u] = v
                match[v] = u
                i -= 2
            else:
                i -= 1

    return match


# =========================================================================
# Label propagation clustering
# =========================================================================

def label_propagation_clustering(
    G: Graph,
    rng: np.random.RandomState,
    max_iterations: int = 10,
    max_cluster_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Coarsening via size-constrained label propagation clustering (KaHIP-style).

    Better than HEM for power-law graphs because it allows clusters
    of size > 2, handling high-degree hubs appropriately.
    """
    n = G.n
    labels = np.arange(n, dtype=np.int64)

    if max_cluster_weight is None:
        max_cluster_weight = max(
            G.total_weight / (n / 4),
            G.vertex_weights.max() * 10,
        )

    cluster_weight = G.vertex_weights.copy()

    for iteration in range(max_iterations):
        changed = 0
        order = rng.permutation(n)

        for v in order:
            nbrs, weights = G.neighbors(v)
            if len(nbrs) == 0:
                continue

            cluster_votes: dict = {}
            for nb, w in zip(nbrs, weights):
                lbl = labels[nb]
                cluster_votes[lbl] = cluster_votes.get(lbl, 0.0) + w

            current_label = labels[v]
            best_label = current_label
            best_vote = cluster_votes.get(current_label, 0.0)

            for lbl, vote in cluster_votes.items():
                if lbl == current_label:
                    continue
                new_weight = cluster_weight[lbl] + G.vertex_weights[v]
                if new_weight > max_cluster_weight:
                    continue
                if vote > best_vote:
                    best_vote = vote
                    best_label = lbl

            if best_label != current_label:
                cluster_weight[current_label] -= G.vertex_weights[v]
                cluster_weight[best_label] += G.vertex_weights[v]
                labels[v] = best_label
                changed += 1

        if changed == 0:
            break

    # Renumber labels to be contiguous
    unique_labels = np.unique(labels)
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    for new_id, old_id in enumerate(unique_labels):
        remap[old_id] = new_id
    return remap[labels]


# =========================================================================
# Community-aware coarsening
# =========================================================================

def detect_communities_louvain_simple(
    G: Graph,
    rng: np.random.RandomState,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Simple Louvain-style community detection for coarsening guidance.
    Assigns each vertex a community label by maximizing modularity gain.
    """
    n = G.n
    communities = np.arange(n, dtype=np.int64)
    m2 = G.degrees.sum()
    if m2 < 1e-12:
        return communities

    # Track community degree sums
    comm_degree = G.degrees.copy()

    for iteration in range(max_iterations):
        changed = 0
        order = rng.permutation(n)
        for v in order:
            nbrs, weights = G.neighbors(v)
            if len(nbrs) == 0:
                continue

            current_comm = communities[v]
            deg_v = G.degrees[v]

            # Compute modularity gain for moving to each neighboring community
            # delta_Q = [sum_w_to_target / m2] - [deg_v * sigma_target / (m2^2)]
            comm_edge_weight = {}
            for nb, w in zip(nbrs, weights):
                c = communities[nb]
                comm_edge_weight[c] = comm_edge_weight.get(c, 0.0) + w

            best_comm = current_comm
            best_gain = 0.0

            # Remove v from its community first
            comm_degree[current_comm] -= deg_v
            w_to_current = comm_edge_weight.get(current_comm, 0.0)

            for target_comm, w_to_target in comm_edge_weight.items():
                gain = (w_to_target - w_to_current) / m2 - \
                       deg_v * (comm_degree[target_comm] - comm_degree[current_comm]) / (m2 * m2)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = target_comm

            if best_comm != current_comm:
                communities[v] = best_comm
                comm_degree[best_comm] += deg_v
                # current comm degree already decremented
                changed += 1
            else:
                comm_degree[current_comm] += deg_v

        if changed == 0:
            break

    return communities


def community_aware_matching(
    G: Graph,
    rng: np.random.RandomState,
    communities: np.ndarray,
    use_expansion2: bool = True,
) -> np.ndarray:
    """
    Matching that only allows contracting vertices within the same community.
    Prevents merging vertices across natural cluster boundaries.

    Can yield up to 8% improvement over unconstrained matching.
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    order = rng.permutation(n)

    ratings = compute_expansion2_ratings(G) if use_expansion2 else None

    for v in order:
        if matched[v]:
            continue
        nbrs, weights = G.neighbors(v)
        if len(nbrs) == 0:
            continue

        best_score = -1.0
        best_nb = -1
        for nb, w in zip(nbrs, weights):
            if matched[nb] or nb == v:
                continue
            # Only match within same community
            if communities[nb] != communities[v]:
                continue
            score = ratings.get((v, nb), 0.0) if ratings else w
            if score > best_score:
                best_score = score
                best_nb = nb

        if best_nb >= 0:
            match[v] = best_nb
            match[best_nb] = v
            matched[v] = True
            matched[best_nb] = True

    return match


# =========================================================================
# Graph contraction
# =========================================================================

def contract_graph(
    G: Graph,
    mapping: np.ndarray,
    partition: Optional[np.ndarray] = None,
) -> Tuple[Graph, np.ndarray]:
    """
    Contract graph according to the given vertex mapping.

    Parameters
    ----------
    G : Graph
    mapping : np.ndarray, shape (n_fine,)
        mapping[v] = supernode index in the coarse graph.
    partition : np.ndarray or None
        Unused. Kept for API compatibility.

    Returns
    -------
    coarse_graph : Graph
    final_mapping : np.ndarray
    """
    n_fine = G.n
    n_coarse = int(mapping.max()) + 1

    # Vectorized vertex weight accumulation
    coarse_weights = np.bincount(mapping, weights=G.vertex_weights, minlength=n_coarse).astype(np.float64)

    # Build coarse adjacency using vectorized sparse operations
    coo = G.adj.tocoo()
    coarse_rows = mapping[coo.row]
    coarse_cols = mapping[coo.col]

    # Filter out self-loops (edges within the same supernode)
    keep = coarse_rows != coarse_cols

    if not np.any(keep):
        adj = sparse.csr_matrix((n_coarse, n_coarse), dtype=np.float64)
    else:
        # COO constructor automatically sums duplicate entries
        adj = sparse.coo_matrix(
            (coo.data[keep], (coarse_rows[keep], coarse_cols[keep])),
            shape=(n_coarse, n_coarse),
        ).tocsr()

    return Graph(adj, coarse_weights), mapping


def matching_to_mapping(match: np.ndarray) -> np.ndarray:
    """
    Convert a matching array to a contiguous coarsening mapping.
    """
    n = len(match)
    mapping = np.full(n, -1, dtype=np.int64)
    next_id = 0

    for v in range(n):
        if mapping[v] >= 0:
            continue
        partner = match[v]
        mapping[v] = next_id
        if partner != v and mapping[partner] < 0:
            mapping[partner] = next_id
        next_id += 1

    return mapping


# =========================================================================
# Partition-guided coarsening (for V-cycles and evolutionary combine)
# =========================================================================

def partition_guided_matching(
    G: Graph,
    partition: np.ndarray,
    rng: np.random.RandomState,
    use_expansion2: bool = True,
) -> np.ndarray:
    """
    Matching that only contracts edges WITHIN the same partition block.
    Cut edges are never contracted, preserving the current cut.
    Used in V-cycles (Walshaw, 2000) and evolutionary combine.
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    order = rng.permutation(n)

    ratings = compute_expansion2_ratings(G) if use_expansion2 else None

    for v in order:
        if matched[v]:
            continue
        nbrs, weights = G.neighbors(v)
        if len(nbrs) == 0:
            continue

        best_score = -1.0
        best_nb = -1
        for nb, w in zip(nbrs, weights):
            if matched[nb] or nb == v:
                continue
            if partition[nb] != partition[v]:
                continue  # Don't contract cut edges
            score = ratings.get((v, nb), 0.0) if ratings else w
            if score > best_score:
                best_score = score
                best_nb = nb

        if best_nb >= 0:
            match[v] = best_nb
            match[best_nb] = v
            matched[v] = True
            matched[best_nb] = True

    return match


def evolutionary_combine_matching(
    G: Graph,
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState,
    use_expansion2: bool = True,
) -> np.ndarray:
    """
    KaFFPaE combine operator: refuse to contract edges cut by EITHER parent.
    This ensures both parent partitions can serve as initial partitions at
    the coarsest level. The offspring inherits good boundary regions from both.

    Sanders & Schulz (ALENEX 2012) improved 76% of Walshaw benchmark entries.
    """
    n = G.n
    match = np.arange(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    order = rng.permutation(n)

    ratings = compute_expansion2_ratings(G) if use_expansion2 else None

    for v in order:
        if matched[v]:
            continue
        nbrs, weights = G.neighbors(v)
        if len(nbrs) == 0:
            continue

        best_score = -1.0
        best_nb = -1
        for nb, w in zip(nbrs, weights):
            if matched[nb] or nb == v:
                continue
            # Refuse to contract if edge is cut by EITHER parent
            if parent1[nb] != parent1[v] or parent2[nb] != parent2[v]:
                continue
            score = ratings.get((v, nb), 0.0) if ratings else w
            if score > best_score:
                best_score = score
                best_nb = nb

        if best_nb >= 0:
            match[v] = best_nb
            match[best_nb] = v
            matched[v] = True
            matched[best_nb] = True

    return match


# =========================================================================
# Top-level coarsening functions
# =========================================================================

def coarsen_graph(
    G: Graph,
    method: str = "hem",
    rng: Optional[np.random.RandomState] = None,
    partition: Optional[np.ndarray] = None,
    lp_iterations: int = 10,
    use_expansion2: bool = True,
    use_gpa: bool = True,
    communities: Optional[np.ndarray] = None,
    max_cluster_weight: Optional[float] = None,  # ? ADD THIS PARAMETER
) -> CoarseningLevel:
    """
    Perform one level of coarsening.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if partition is not None:
        match = partition_guided_matching(G, partition, rng, use_expansion2)
        mapping = matching_to_mapping(match)
    elif method == "hem":
        match = heavy_edge_matching(G, rng)
        mapping = matching_to_mapping(match)
    elif method == "shem":
        match = sorted_heavy_edge_matching(G, rng)
        mapping = matching_to_mapping(match)
    elif method == "expansion2":
        match = expansion2_matching(G, rng, use_gpa=use_gpa)
        mapping = matching_to_mapping(match)
    elif method == "label_propagation":
        mapping = label_propagation_clustering(
            G, rng, lp_iterations, 
            max_cluster_weight=max_cluster_weight  # ? PASS IT THROUGH
        )
    elif method == "community_aware":
        if communities is None:
            communities = detect_communities_louvain_simple(G, rng)
        match = community_aware_matching(G, rng, communities, use_expansion2)
        mapping = matching_to_mapping(match)
    else:
        raise ValueError(f"Unknown coarsening method: {method}")

    coarse_graph, mapping = contract_graph(G, mapping, partition)
    return CoarseningLevel(G, coarse_graph, mapping)

def build_hierarchy(
    G: Graph,
    min_vertices: int = 100,
    max_levels: int = 30,
    method: str = "auto",  # NEW: "auto" selects based on structure
    seed: int = 42,
    lp_iterations: int = 10,
    use_expansion2: bool = True,
    use_gpa: bool = True,
    communities: Optional[np.ndarray] = None,  # caller-supplied communities for "community_aware"
    verbose: bool = False,
) -> List[CoarseningLevel]:
    """
    Build a full coarsening hierarchy with automatic method selection
    and fallback to label propagation when matching stalls.
    
    Parameters
    ----------
    G : Graph
        Input graph to coarsen.
    min_vertices : int
        Stop coarsening when graph has fewer than this many vertices.
    max_levels : int
        Maximum number of coarsening levels.
    method : str
        Coarsening method: "auto", "hem", "shem", "expansion2",
        "label_propagation", or "community_aware".
        "auto" selects based on structural features (recommended for SNAP).
    seed : int
        Random seed for reproducibility.
    lp_iterations : int
        Number of label propagation iterations (if used).
    use_expansion2 : bool
        Use expansion*2 edge ratings in matching.
    use_gpa : bool
        Use Global Path Algorithm for matching.
    verbose : bool
        Print diagnostic messages.

    Returns
    -------
    hierarchy : List[CoarseningLevel]
        List of coarsening levels from fine to coarse.
    """
    rng = np.random.RandomState(seed)
    hierarchy = []
    current = G

    # === Auto-select method based on graph structure ===
    if method == "auto":
        from .structural_features import select_coarsening_method
        method = select_coarsening_method(G, verbose=verbose)
        if verbose:
            print(f"  [auto] Selected {method} based on structure")

    # Optionally detect communities once for community-aware coarsening.
    # Use the caller-supplied `communities` (e.g. classical LP/Louvain or the SSL head) if given;
    # otherwise fall back to the internal Louvain-simple detector (shipped behavior).
    if method == "community_aware":
        if communities is None:
            communities = detect_communities_louvain_simple(G, rng)
    else:
        communities = None

    consecutive_failures = 0
    current_method = method
    lp_fallback_active = False
    
    # Track if we started with LP (don't fallback to the same method)
    started_with_lp = (method == "label_propagation")

    for level in range(max_levels):
        if current.n <= min_vertices:
            break

        # When using LP (either as primary or fallback), set strict cluster size
        if current_method == "label_propagation" or lp_fallback_active:
            # Target ~50% coarsening per level (matching-like behavior)
            avg_vertex_weight = current.total_weight / current.n
            # Max cluster should hold ~2 average vertices
            lp_max_cluster_weight = avg_vertex_weight * 2.5
            
            cl = coarsen_graph(
                current, "label_propagation", rng,
                lp_iterations=lp_iterations,
                use_expansion2=use_expansion2,
                use_gpa=use_gpa,
                communities=communities,
                max_cluster_weight=lp_max_cluster_weight,
            )
        else:
            cl = coarsen_graph(
                current, current_method, rng,
                lp_iterations=lp_iterations,
                use_expansion2=use_expansion2,
                use_gpa=use_gpa,
                communities=communities,
            )

        coarse_n = cl.coarse_graph.n
        reduction_ratio = coarse_n / current.n

        # REJECT over-aggressive coarsening (more than 75% reduction in one step)
        if reduction_ratio < 0.25:
            if verbose:
                print(f"  [reject] Level {level} too aggressive: "
                      f"{current.n} -> {coarse_n} ({reduction_ratio:.1%})")
            # If LP caused this, LP is not working for this graph - stop
            if lp_fallback_active or started_with_lp:
                break
            # Otherwise try LP fallback
            consecutive_failures += 1
            if consecutive_failures >= 2 and current.n > min_vertices * 4:
                # Always log fallback so audits can detect coarsener-axis degradation
                import sys as _sys
                print(
                    f"  [coarsening fallback] requested method={method!r} "
                    f"degraded to label_propagation at level {level} "
                    f"(over-aggressive reduction)",
                    file=_sys.stderr, flush=True,
                )
                lp_fallback_active = True
                consecutive_failures = 0
                continue
            elif consecutive_failures >= 3:
                break
            continue

        # REJECT coarsening that goes below min_vertices
        elif coarse_n < min_vertices:
            # Don't append this level - just stop
            break

        # Check for insufficient coarsening (less than 5% reduction)
        elif reduction_ratio >= 0.95:
            consecutive_failures += 1

            # FALLBACK: Try label propagation if matching-based methods stall
            if not lp_fallback_active and not started_with_lp and consecutive_failures >= 2:
                if current.n > min_vertices * 4:
                    import sys as _sys
                    print(
                        f"  [coarsening fallback] requested method={method!r} "
                        f"degraded to label_propagation at level {level} "
                        f"(matching stalled, <5% reduction)",
                        file=_sys.stderr, flush=True,
                    )
                    lp_fallback_active = True
                    consecutive_failures = 0
                    continue
                else:
                    break

            if consecutive_failures >= 3:
                break
            continue

        else:
            # Successful coarsening - accept this level
            consecutive_failures = 0
            hierarchy.append(cl)
            current = cl.coarse_graph

            # Update communities for next level if needed
            if method == "community_aware" and communities is not None:
                new_communities = np.zeros(current.n, dtype=np.int64)
                for v in range(cl.fine_graph.n):
                    new_communities[cl.mapping[v]] = communities[v]
                communities = new_communities

    return hierarchy


def _louvain_nx_communities(G, seed=0, resolutions=(0.5, 1.0, 2.0)):
    """networkx Louvain at several resolutions; keep the best-modularity partition."""
    import networkx as nx
    A = G.adj.tocoo()
    H = nx.Graph(); H.add_nodes_from(range(G.n))
    for i, j, w in zip(A.row, A.col, A.data):
        if int(i) < int(j):
            H.add_edge(int(i), int(j), weight=float(w))
    best_comm, best_q = None, -1e18
    for r in resolutions:
        parts = nx.community.louvain_communities(H, weight="weight", resolution=r, seed=seed)
        q = nx.community.modularity(H, parts, weight="weight", resolution=r)
        if q > best_q:
            best_q = q
            lab = np.zeros(G.n, dtype=np.int64)
            for cid, S in enumerate(parts):
                for v in S:
                    lab[v] = cid
            best_comm = lab
    return best_comm


def resolve_communities(G, source="internal", seed=42, community_file=None):
    """Resolve a per-vertex community array for community-aware coarsening from a named source.
      'internal'       -> None (build_hierarchy uses Louvain-simple; shipped behavior)
      'lp'             -> size-unconstrained label propagation communities
      'louvain_simple' -> the in-repo greedy Louvain
      'louvain_nx'     -> networkx Louvain, multi-resolution, best modularity
      'file'           -> load a .comm file (one community-id int per line)
      'ssl'            -> the learned DMoN community head (Phase 1, lazily imported)
    """
    if source in (None, "internal"):
        return None
    rng = np.random.RandomState(seed)
    if source == "lp":
        comm = label_propagation_clustering(G, rng, max_iterations=20, max_cluster_weight=None)
    elif source == "louvain_simple":
        comm = detect_communities_louvain_simple(G, rng)
    elif source == "louvain_nx":
        comm = _louvain_nx_communities(G, seed)
    elif source == "file":
        comm = np.loadtxt(community_file, dtype=np.int64)
    elif source == "ssl":
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from community_ssl import fit_communities
        comm = fit_communities(G, seed=seed)
    else:
        raise ValueError(f"unknown community_source {source!r}")
    _, comm = np.unique(np.asarray(comm, dtype=np.int64), return_inverse=True)   # contiguous ids
    return comm.astype(np.int64)