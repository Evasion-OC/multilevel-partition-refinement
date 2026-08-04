"""
Structural Graph Feature Extraction
======================================
Extracts graph-level features for predicting when RL outperforms FM refinement,
analysing graph structure through the lens of spectral graph theory, graph minor
theory, and GNN expressiveness.

Extended feature vector (v2):
  psi(G) = (lambda2, lambda2/lambda_n, Delta_k, phi_hat(G), Q_max, C_bar,
            gamma_hat, tw_tilde, tw_upper, tw_ratio, separator_ratio,
            cheeger_k_bound, spectral_width, degree_entropy, wl_candidate_dim,
            hadwiger_estimate, spectral_tightness, coarsening_fidelity_mean)

Theoretical connections:
  - Cheeger inequality: relates lambda2 to conductance (Cheeger 1970; Alon-Milman 1985)
  - Higher-order Cheeger: lambda_k/2 <= rho(k) <= O(k^2) sqrt(lambda_k)
    (Lee-Oveis Gharan-Trevisan, STOC 2014)
  - Davis-Kahan perturbation: eigengap predicts spectral clustering stability
  - Excluded grid theorem: tw(G) >= Omega(k) iff G contains k x k grid minor
    (Robertson-Seymour; Kawarabayashi-Kobayashi, 2012)
  - Treewidth-separator duality: graphs with tw = O(sqrt(n)) admit O(sqrt(n))
    separators, explaining multilevel coarsening effectiveness on mesh graphs
  - GNN expressiveness: spectral PE lifts standard MPNN beyond 1-WL
    (Xu et al., ICLR 2019; Lim et al., ICML 2023)
  - Newman-Girvan modularity: community structure strength
  - Mader's theorem / Hadwiger number: clique minor size governs separator bounds
    (Alon-Seymour-Thomas, 1990)
  - Spectral coarsening fidelity: eigenvalue preservation across coarsening levels
    (Loukas, ICML 2019)

References:
  - Kawarabayashi & Kobayashi (STACS 2012, JCTB 2019): Linear min-max relation
    between treewidth and grid minors for H-minor-free graphs
  - Xu, Hu, Leskovec & Jegelka (ICLR 2019): GIN -- How Powerful are GNNs
  - Xu, Zhang, Li, Du, Kawarabayashi & Jegelka (ICLR 2021): GNN extrapolation
  - Lim et al. (ICML 2023): SignNet for sign-invariant spectral PE
  - Lee, Oveis Gharan & Trevisan (STOC 2014): Higher-order Cheeger inequalities
  - Lipton & Tarjan (1979): Planar separator theorem
  - Alon, Seymour & Thomas (1990): Separator theorem for minor-free graphs
  - Mader (1967): Average degree forces clique minors
  - Loukas (2019): Graph reduction with spectral and cut guarantees
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from .graph import Graph
from .spectral import (
    compute_eigenvectors, eigengap, spectral_gap_ratio,
    estimate_conductance, donath_hoffman_bound
)


# =========================================================================
# Core graph statistics
# =========================================================================

def compute_clustering_coefficient(G: Graph) -> float:
    """
    Compute average clustering coefficient.

    C(v) = 2|{(u,w) in E : u,w in N(v)}| / (deg(v)(deg(v)-1))
    C_bar = (1/n) sum_v C(v)
    """
    total_cc = 0.0
    count = 0
    for v in range(G.n):
        nbrs, _ = G.neighbors(v)
        d = len(nbrs)
        if d < 2:
            continue

        nbr_set = set(nbrs)
        edges_among_nbrs = 0
        for i in range(len(nbrs)):
            ni_nbrs, _ = G.neighbors(nbrs[i])
            for ni_nb in ni_nbrs:
                if ni_nb in nbr_set and ni_nb != nbrs[i]:
                    edges_among_nbrs += 1
        edges_among_nbrs //= 2

        cc = 2.0 * edges_among_nbrs / (d * (d - 1))
        total_cc += cc
        count += 1

    return total_cc / max(count, 1)


def _sampled_clustering_coefficient(G: Graph, sample_size: int = 5000) -> float:
    """Sample-based clustering coefficient for large graphs."""
    rng = np.random.RandomState(42)
    sample = rng.choice(G.n, size=min(sample_size, G.n), replace=False)
    total_cc = 0.0
    count = 0
    for v in sample:
        nbrs, _ = G.neighbors(v)
        d = len(nbrs)
        if d < 2:
            continue
        nbr_set = set(nbrs)
        triangles = 0
        for i in range(min(len(nbrs), 50)):
            ni_nbrs, _ = G.neighbors(nbrs[i])
            for ni_nb in ni_nbrs:
                if ni_nb in nbr_set:
                    triangles += 1
        triangles //= 2
        cc = 2.0 * triangles / (d * (d - 1))
        total_cc += cc
        count += 1
    return total_cc / max(count, 1)


def estimate_power_law_exponent(G: Graph) -> float:
    """
    Estimate the power-law exponent gamma of the degree distribution.

    Uses maximum likelihood estimation: gamma_hat = 1 + n [sum ln(d_i/d_min)]^{-1}

    Returns gamma_hat or 0 if the graph doesn't follow a power law.
    """
    degrees = G.degrees
    degrees = degrees[degrees > 0]
    if len(degrees) < 10:
        return 0.0

    d_min = max(degrees.min(), 1.0)
    ratios = degrees / d_min
    ratios = ratios[ratios > 1.0]

    if len(ratios) < 10:
        return 0.0

    gamma = 1.0 + len(ratios) / np.sum(np.log(ratios))
    return float(gamma)


def estimate_modularity(G: Graph, partition: np.ndarray) -> float:
    """
    Compute Newman-Girvan modularity Q for a given partition.

    Q = (1/2m) sum_{ij} [A_{ij} - d_i d_j / (2m)] delta(c_i, c_j)
    """
    m2 = G.degrees.sum()  # 2m
    if m2 < 1e-12:
        return 0.0

    k = int(partition.max()) + 1
    Q = 0.0

    for block in range(k):
        mask = partition == block
        vertices = np.where(mask)[0]

        e_internal = 0.0
        for v in vertices:
            nbrs, weights = G.neighbors(v)
            for nb, w in zip(nbrs, weights):
                if partition[nb] == block:
                    e_internal += w
        e_internal /= 2

        block_deg = G.degrees[mask].sum()
        Q += e_internal / (m2 / 2) - (block_deg / m2) ** 2

    return float(Q)


# =========================================================================
# Treewidth estimation with graph minor theory connections
# =========================================================================

def estimate_treewidth(G: Graph) -> float:
    """
    Estimate treewidth using the degeneracy (k-core number) as a lower bound.

    Uses a min-heap for O(n log n) instead of O(n^2).
    The degeneracy of a graph is a lower bound on treewidth.

    Graph minor theory context (Kawarabayashi & Kobayashi, JCTB 2019):
      For H-minor-free graphs: tw(G) <= c_H * sqrt(n)
      where c_H depends only on |H|.
      For planar graphs specifically: tw(G) <= O(sqrt(n)) (Lipton-Tarjan).
      The degeneracy lower-bounds treewidth: delta*(G) <= tw(G).
    """
    import heapq
    degrees = G.degrees.copy().astype(int)
    n = G.n
    removed = np.zeros(n, dtype=bool)
    max_core = 0

    heap = [(int(degrees[v]), v) for v in range(n)]
    heapq.heapify(heap)

    while heap:
        deg, v = heapq.heappop(heap)
        if removed[v]:
            continue
        current_deg = int(degrees[v])
        if current_deg > deg:
            heapq.heappush(heap, (current_deg, v))
            continue

        max_core = max(max_core, current_deg)
        removed[v] = True

        nbrs, _ = G.neighbors(v)
        for nb in nbrs:
            if not removed[nb]:
                degrees[nb] -= 1
                heapq.heappush(heap, (int(degrees[nb]), nb))

    return float(max_core)


def compute_treewidth_ratio(treewidth_est: float, n: int) -> float:
    """
    Compute tw / sqrt(n) ratio -- a key indicator from graph minor theory.

    Theoretical basis (Kawarabayashi & Kobayashi, STACS 2012):
      - Planar/mesh graphs: tw = O(sqrt(n)), so ratio ~ constant
      - Expander-like graphs: tw = Omega(n), so ratio grows as sqrt(n)
      - Bounded-genus graphs: tw = O(sqrt(g*n)), ratio scales with genus

    For graph partitioning:
      - Low ratio (<= 1): graph likely admits good separators, multilevel
        coarsening faithfully preserves partition structure
      - High ratio (> 2): graph has complex structure that coarsening may
        distort, creating opportunity for RL refinement

    This connects the Robertson-Seymour excluded grid theorem to practical
    partitioning: if tw is small relative to sqrt(n), the graph excludes
    large grid minors and has tree-like decomposability.
    """
    sqrt_n = max(np.sqrt(n), 1.0)
    return float(treewidth_est / sqrt_n)


# =========================================================================
# Treewidth upper bound (separator-based)
# =========================================================================

def compute_treewidth_upper_bound(G: Graph, separator_ratio: float) -> float:
    """
    Upper bound on treewidth from balanced separator size.

    If G has a balanced (2/3, 1/3)-separator of size s, then recursive
    application gives tw(G) <= s * ceil(log2(n / s)).

    The sandwich degeneracy <= tw(G) <= s*log(n/s) connects the lower
    bound (from k-core) to the upper bound (from spectral separator),
    giving a practical interval estimate.

    Parameters
    ----------
    G : Graph
    separator_ratio : float
        |separator| / n as returned by compute_separator_quality.

    Returns
    -------
    tw_upper : float
        Upper bound on treewidth.
    """
    n = G.n
    s = max(separator_ratio * n, 1.0)
    if s >= n:
        return float(n)
    return float(s * np.ceil(np.log2(max(n / s, 2.0))))


# =========================================================================
# Coarsening spectral fidelity
# =========================================================================

def compute_coarsening_spectral_fidelity(
    G: Graph,
    k: int,
    method: str = "expansion2",
    max_levels: int = 5,
) -> List[float]:
    """
    Measure spectral eigenvalue distortion at each coarsening level.

    For each level i, computes:
        fidelity_i = ||lambda(G_i) - lambda(G_{i+1})||_2 / ||lambda(G_i)||_2

    where lambda(G) is the vector of the first k+1 eigenvalues.

    Theoretical basis (Loukas, ICML 2019):
      Spectral coarsening preserves eigenvalues within a factor (1 +/- eps)
      when the coarsening mapping is spectrally similar. Standard HEM
      coarsening on power-law graphs causes large spectral distortion at
      levels where high-degree hubs are absorbed, while LP coarsening and
      expansion*2 ratings produce more faithful hierarchies.

    Returns
    -------
    fidelity : list of float
        Relative eigenvalue distance at each coarsening level.
        Values near 0 indicate faithful coarsening; values > 0.2 indicate
        significant structural distortion.
    """
    from .coarsening import build_hierarchy

    hierarchy = build_hierarchy(
        G, min_vertices=max(50, 2 * k),
        max_levels=max_levels, method=method, seed=42,
    )

    n_eigs = min(k + 1, G.n - 1)
    if n_eigs < 2:
        return []

    try:
        prev_eigs, _ = compute_eigenvectors(G, n_eigs, normalized=True)
    except Exception:
        return []

    fidelity = []
    for lvl in hierarchy:
        n_eigs_c = min(k + 1, lvl.coarse_graph.n - 1)
        if n_eigs_c < 2:
            break
        try:
            curr_eigs, _ = compute_eigenvectors(
                lvl.coarse_graph, n_eigs_c, normalized=True
            )
        except Exception:
            break
        min_len = min(len(prev_eigs), len(curr_eigs))
        dist = np.linalg.norm(prev_eigs[:min_len] - curr_eigs[:min_len])
        norm = max(np.linalg.norm(prev_eigs[:min_len]), 1e-12)
        fidelity.append(float(dist / norm))
        prev_eigs = curr_eigs

    return fidelity


# =========================================================================
# Spectral tightness (actual cut / Donath-Hoffman bound)
# =========================================================================

def compute_spectral_tightness(
    actual_cut: float,
    dh_bound: float,
) -> float:
    """
    Ratio of actual best cut to the Donath-Hoffman spectral lower bound.

    tau(G, k) = cut_actual / DH_bound

    Interpretation:
      - tau ~ 1: spectral bound is tight, spectral methods near-optimal
      - tau >> 1: spectral bound is loose, room for non-spectral improvement

    Across families:
      - Walshaw meshes: tau stays close to 1 as k grows (clean separators)
      - BA power-law: tau grows with k (continuous spectrum, loose bounds)
      - ER random: tau ~ 1 trivially (all cuts have similar cost)
    """
    if dh_bound < 1e-12:
        return float('inf')
    return float(actual_cut / dh_bound)


# =========================================================================
# Conductance profile (full sweep, not just minimum)
# =========================================================================

def compute_conductance_profile(
    G: Graph,
    num_points: int = 100,
) -> np.ndarray:
    """
    Compute the conductance at uniform fraction points along the Fiedler sweep.

    Rather than returning a single conductance estimate, this computes the
    full profile phi(S_i) for set sizes i = 1, ..., n-1, sampled at
    num_points uniform intervals.

    Profile shapes characterise graph families:
      - Mesh/planar: V-shaped with a sharp minimum near n/2
      - Power-law: L-shaped, drops quickly then flattens (many competing cuts)
      - ER random: flat (quasi-random edge distribution, Expander Mixing Lemma)

    The shape explains FM local optima: L-shaped profiles have many nearly
    equivalent local minima, making greedy search unreliable.

    Returns
    -------
    profile : np.ndarray, shape (num_points, 2)
        Columns: [fraction, conductance]. Empty array if computation fails.
    """
    from .spectral import fiedler_vector

    if G.n < 10:
        return np.zeros((0, 2))

    try:
        _, fiedler = fiedler_vector(G, normalized=True)
    except Exception:
        return np.zeros((0, 2))

    order = np.argsort(fiedler)
    deg = G.degrees
    total_vol = deg.sum()

    in_S = np.zeros(G.n, dtype=bool)
    vol_S = 0.0
    cut_S = 0.0
    profile = []
    step = max(G.n // num_points, 1)

    for i in range(G.n - 1):
        v = order[i]
        in_S[v] = True
        vol_S += deg[v]

        nbrs, weights = G.neighbors(v)
        for nb, w in zip(nbrs, weights):
            if in_S[nb]:
                cut_S -= w
            else:
                cut_S += w

        if (i + 1) % step == 0:
            vol_comp = total_vol - vol_S
            if vol_S > 0 and vol_comp > 0:
                phi = cut_S / min(vol_S, vol_comp)
                profile.append((float((i + 1) / G.n), float(phi)))

    return np.array(profile) if profile else np.zeros((0, 2))


def classify_conductance_profile_shape(profile: np.ndarray) -> str:
    """
    Classify the conductance profile shape into one of three types:
      - 'v_shaped': sharp minimum near centre (mesh/planar)
      - 'l_shaped': rapid initial drop then flat (power-law)
      - 'flat': roughly constant (expander/ER)
      - 'unknown': insufficient data

    Uses the ratio of conductance at 10% to conductance at 50%.
    """
    if len(profile) < 10:
        return 'unknown'

    fracs = profile[:, 0]
    phis = profile[:, 1]

    # Find conductance near 10% and 50%
    idx_10 = np.argmin(np.abs(fracs - 0.1))
    idx_50 = np.argmin(np.abs(fracs - 0.5))
    phi_10 = phis[idx_10]
    phi_50 = phis[idx_50]
    phi_min = phis.min()
    phi_max = phis.max()

    if phi_max < 1e-12:
        return 'unknown'

    variation = (phi_max - phi_min) / phi_max

    if variation < 0.3:
        return 'flat'
    elif phi_10 < 0.5 * phi_50:
        return 'l_shaped'
    else:
        return 'v_shaped'


# =========================================================================
# Hadwiger number (largest clique minor) estimation
# =========================================================================

def estimate_hadwiger_number(G: Graph) -> float:
    """
    Estimate the Hadwiger number eta(G) = largest t such that K_t is a minor.

    Uses Mader's theorem (1967): a graph with average degree >= c*t*sqrt(log t)
    contains K_t as a minor. The maximum average degree (mad) over all
    subgraphs equals twice the degeneracy for practical purposes.

    Practical estimate: eta ~= mad(G) / sqrt(log(mad(G)))

    The Hadwiger number governs separator quality via Alon-Seymour-Thomas (1990):
      Graphs excluding K_h have balanced separators of size O(h^{3/2} * sqrt(n)).

    Across families:
      - Walshaw (planar): eta <= 5 (exclude K5 by Wagner/Kuratowski)
      - BA power-law: eta grows with attachment parameter m
      - ER random: eta ~ c*ln(n) / sqrt(log(c*ln n))
    """
    degeneracy = estimate_treewidth(G)  # degeneracy (lower bound on tw)
    mad = 2.0 * degeneracy
    if mad < 2.0:
        return 1.0
    return float(mad / max(np.sqrt(np.log(max(mad, 2.0))), 1.0))


def compute_ast_separator_bound(hadwiger_est: float, n: int) -> float:
    """
    Alon-Seymour-Thomas separator bound: graphs excluding K_h as a minor
    have balanced separators of size O(h^{3/2} * sqrt(n)).

    Parameters
    ----------
    hadwiger_est : float
        Estimated Hadwiger number eta(G).
    n : int
        Number of vertices.

    Returns
    -------
    separator_bound : float
        Estimated maximum separator size.
    """
    return float(hadwiger_est ** 1.5 * np.sqrt(n))


# =========================================================================
# Spectral profile (eigenvalue sequence)
# =========================================================================

def compute_spectral_profile(
    G: Graph,
    num_eigs: int = 20,
) -> np.ndarray:
    """
    Compute the first num_eigs eigenvalues of L_sym as the spectral profile.

    The shape of this sequence characterises graph families:
      - Mesh/planar: smooth ramp with clear gap at lambda_{k+1}
      - BA power-law: slow increase, no clear gap (continuous spectrum)
      - ER random: eigenvalues clustered near 1 +/- 1/sqrt(np)

    Returns
    -------
    eigenvalues : np.ndarray, shape (num_eigs,)
        The first num_eigs eigenvalues of L_sym in ascending order.
    """
    k = min(num_eigs, G.n - 1)
    if k < 1:
        return np.zeros(0)
    try:
        evals, _ = compute_eigenvectors(G, k, normalized=True)
        return evals
    except Exception:
        return np.zeros(0)


# =========================================================================
# Separator quality
# =========================================================================

def compute_separator_quality(G: Graph) -> float:
    """
    Estimate separator quality via spectral methods.

    Uses the Fiedler vector to find a vertex separator and measures its
    relative size: |S| / n. By the planar separator theorem (Lipton-Tarjan,
    1979), planar graphs admit separators of size O(sqrt(n)), giving
    ratio O(1/sqrt(n)).

    Separator quality connects to graph minor theory:
      - Graphs excluding a fixed minor H have O(sqrt(n)) separators
        (Alon-Seymour-Thomas, 1990; Kawarabayashi-Reed)
      - Small separator ratio -> faithful coarsening -> classical methods suffice

    Returns separator_ratio = |separator| / n.
    """
    if G.n < 10:
        return 1.0

    try:
        from .spectral import fiedler_vector
        _, fiedler = fiedler_vector(G, normalized=True)

        order = np.argsort(fiedler)
        n = G.n

        best_cut_width = n
        in_S = np.zeros(n, dtype=bool)
        cut_edges = 0

        for i in range(n - 1):
            v = order[i]
            in_S[v] = True
            nbrs, _ = G.neighbors(v)
            for nb in nbrs:
                if in_S[nb]:
                    cut_edges -= 1
                else:
                    cut_edges += 1

            # Only consider roughly balanced splits
            size_S = i + 1
            if n // 3 <= size_S <= 2 * n // 3:
                if cut_edges < best_cut_width:
                    best_cut_width = cut_edges

        # Approximate separator size: edges crossing / avg degree
        avg_deg = max(G.degrees.mean(), 1.0)
        separator_size = best_cut_width / avg_deg
        return float(min(separator_size / n, 1.0))

    except Exception:
        return 1.0


# =========================================================================
# Spectral features with higher-order Cheeger analysis
# =========================================================================

def compute_higher_order_cheeger_bound(
    eigenvalues: np.ndarray, k: int,
) -> Dict[str, float]:
    """
    Compute bounds from the higher-order Cheeger inequality.

    Lee-Oveis Gharan-Trevisan (STOC 2014):
      lambda_k / 2 <= rho(k) <= O(k^2) sqrt(lambda_k)

    where rho(k) is the optimal k-way expansion.

    Returns:
      - cheeger_k_lower: lambda_k / 2 (lower bound on k-way expansion)
      - cheeger_k_upper: k^2 sqrt(lambda_k) (upper bound)
      - cheeger_k_gap: ratio upper/lower -- measures bound tightness
      - spectral_k_separation: lambda_k / lambda_{k+1}

    Tight gap ~ 1:   spectral methods give near-optimal partitions
    Wide gap >> 1:    room for improvement beyond spectral (RL opportunity)
    """
    result = {
        'cheeger_k_lower': 0.0,
        'cheeger_k_upper': 0.0,
        'cheeger_k_gap': 0.0,
        'spectral_k_separation': 0.0,
    }

    if k >= len(eigenvalues):
        return result

    lambda_k = max(float(eigenvalues[k]), 0.0)
    result['cheeger_k_lower'] = lambda_k / 2.0
    result['cheeger_k_upper'] = float(k * k) * np.sqrt(max(lambda_k, 0.0))

    if result['cheeger_k_lower'] > 1e-12:
        result['cheeger_k_gap'] = result['cheeger_k_upper'] / result['cheeger_k_lower']
    else:
        result['cheeger_k_gap'] = float('inf')

    if k + 1 < len(eigenvalues) and eigenvalues[k + 1] > 1e-12:
        result['spectral_k_separation'] = float(eigenvalues[k] / eigenvalues[k + 1])

    return result


def compute_spectral_width(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral width: lambda_max - lambda_2.

    Measures the spread of the non-trivial spectrum. Large spectral width
    relative to lambda_2 indicates multi-scale structure where different
    eigenvectors capture structure at very different scales.
    """
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[-1] - eigenvalues[1])


# =========================================================================
# GNN expressiveness indicators
# =========================================================================

def _run_wl_refinement(G: Graph, max_iters: int = 20) -> np.ndarray:
    """
    Run 1-WL colour refinement to a stable colouring.

    Returns the stable colour assignment for each vertex.

    Theoretical context (Xu et al., ICLR 2019; co-authored with Kawarabayashi):
      - Standard MPNNs are bounded by 1-WL in distinguishing power
      - 1-WL distinguishes vertices by iterative neighbour multiset hashing
      - 1-WL fails on regular graphs where all vertices have identical
        local structure (e.g., all vertices in a cycle have degree 2)
    """
    n = G.n
    if n == 0:
        return np.array([], dtype=np.int64)

    colours = G.degrees.astype(int).copy()

    for iteration in range(min(n, max_iters)):
        new_colours = np.zeros(n, dtype=np.int64)
        colour_map = {}
        next_colour = 0

        for v in range(n):
            nbrs, _ = G.neighbors(v)
            nbr_colours = tuple(sorted(colours[nbrs]))
            key = (colours[v], nbr_colours)

            if key not in colour_map:
                colour_map[key] = next_colour
                next_colour += 1
            new_colours[v] = colour_map[key]

        num_old = len(np.unique(colours))
        num_new = len(np.unique(new_colours))
        colours = new_colours

        if num_new == num_old:
            break

    return colours


def compute_wl_features(G: Graph) -> Dict[str, float]:
    """
    Compute 1-WL colour refinement features relevant to GNN expressiveness.

    Returns:
      - wl_colours: number of stable 1-WL colour classes
      - wl_regularity: fraction of vertices in the largest colour class
      - wl_fraction_distinguished: fraction of vertices uniquely coloured

    High wl_regularity -> graph is highly regular -> 1-WL (and standard
    GNNs) cannot distinguish many vertices -> spectral PE is critical.
    """
    colours = _run_wl_refinement(G)
    n = max(G.n, 1)

    unique_colours, counts = np.unique(colours, return_counts=True)
    num_colours = len(unique_colours)
    num_unique = int(np.sum(counts == 1))

    return {
        'wl_colours': float(num_colours),
        'wl_regularity': float(counts.max() / n) if len(counts) > 0 else 0.0,
        'wl_fraction_distinguished': float(num_unique / n),
    }


# =========================================================================
# Degree distribution analysis
# =========================================================================

def compute_degree_entropy(G: Graph) -> float:
    """
    Compute Shannon entropy of the degree distribution.

    H(d) = -sum_d p(d) log2 p(d)

    Low entropy -> regular graph (e.g., mesh) -> classical methods likely work
    High entropy -> heterogeneous structure -> RL may exploit local variation
    """
    degrees = G.degrees.astype(int)
    if len(degrees) == 0:
        return 0.0

    _, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# =========================================================================
# Graph class indicators (minor-theoretic)
# =========================================================================

def estimate_planarity_score(G: Graph) -> float:
    """
    Heuristic planarity score based on edge density and Euler's formula.

    For planar graphs: m <= 3n - 6 (Euler bound).
    Score = 1.0 if m <= 3n - 6, linearly decreasing to 0 as m grows beyond.

    Planarity is fundamental to graph minor theory:
      - Planar graphs have tw = O(sqrt(n)) (Lipton-Tarjan separator theorem)
      - They exclude K5 and K3,3 as minors (Kuratowski/Wagner)
      - Multilevel coarsening preserves their structure faithfully
    """
    n = G.n
    m = G.m
    if n < 3:
        return 1.0
    euler_bound = 3 * n - 6
    if m <= euler_bound:
        return 1.0
    ratio = m / euler_bound
    return float(max(0.0, 1.0 - (ratio - 1.0) / 2.0))


def classify_graph_family(features: Dict[str, float]) -> str:
    """
    Classify a graph into structural families based on extracted features.

    Families (aligned with graph minor theory):
      - 'mesh_planar': FEM-type, low tw/sqrt(n), planar, regular degree
      - 'mesh_dense':  High avg_degree mesh (e.g., bcsstk graphs)
      - 'power_law':   Scale-free, high degree variance
      - 'expander':    High lambda2, hard to partition
      - 'structured':  High modularity, clear community structure
      - 'other':       None of the above
    """
    tw_ratio = features.get('treewidth_ratio', 1.0)
    planarity = features.get('planarity_score', 0.5)
    power_law = features.get('power_law_exponent', 0.0)
    lambda2 = features.get('lambda2', 0.0)
    degree_var = features.get('degree_variance', 0.0)
    avg_deg = features.get('avg_degree', 0.0)
    modularity = features.get('modularity', 0.0)

    if 2.0 < power_law < 3.5 and degree_var > 10.0:
        return 'power_law'
    if lambda2 > 0.5:
        return 'expander'
    if planarity > 0.8 and tw_ratio < 1.5:
        if avg_deg > 15:
            return 'mesh_dense'
        return 'mesh_planar'
    if modularity > 0.4:
        return 'structured'
    return 'other'


# =========================================================================
# Coarsening method selection based on structural features
# =========================================================================

def compute_coarsening_features(G: Graph) -> Dict[str, float]:
    """
    Compute structural features that predict whether label propagation
    coarsening will outperform matching-based coarsening.

    These features are computed BEFORE coarsening to enable automatic
    method selection, unlike gamma which is a result of coarsening.

    Theoretical basis:
      - Abou-Rjeili & Karypis (IPDPS 2006): Power-law graphs need LP coarsening
      - Sanders & Schulz (ESA 2011): Expansion*2 + LP hybrid in KaHIP
      - Heavy-tailed degree distributions cause matching to stall on hubs

    Returns
    -------
    dict with keys:
      cv              : Coefficient of variation of degrees (std/mean)
      hub_ratio       : max_degree / avg_degree
      low_deg_frac    : Fraction of vertices with degree <= 2
      skewness        : Degree distribution skewness (third moment)
      density         : Edge density 2m / (n(n-1))
      alpha_hill      : Hill estimator of power-law exponent
      avg_degree      : Average degree
      isolated_frac   : Fraction of isolated or near-isolated vertices
    """
    degrees = G.degrees
    n, m = G.n, G.m

    if n == 0:
        return {
            'cv': 0.0, 'hub_ratio': 1.0, 'low_deg_frac': 0.0,
            'skewness': 0.0, 'density': 0.0, 'alpha_hill': 3.0,
            'avg_degree': 0.0, 'isolated_frac': 0.0,
        }

    avg_deg = float(np.mean(degrees))
    std_deg = float(np.std(degrees))
    max_deg = float(np.max(degrees))

    # 1. Coefficient of Variation (CV) - measures degree heterogeneity
    #    High CV = power-law-like, hubs dominate -> LP better
    cv = std_deg / avg_deg if avg_deg > 0 else 0.0

    # 2. Hub dominance ratio - how extreme are the hubs?
    hub_ratio = max_deg / avg_deg if avg_deg > 0 else 1.0

    # 3. Fraction of low-degree nodes (degree <= 2)
    #    High fraction = many leaf or chain nodes -> matching struggles
    low_deg_frac = float(np.sum(degrees <= 2)) / n

    # 4. Fraction of isolated/near-isolated vertices (degree <= 1)
    isolated_frac = float(np.sum(degrees <= 1)) / n

    # 5. Degree skewness (third standardized moment)
    if std_deg > 1e-12:
        skewness = float(np.mean(((degrees - avg_deg) / std_deg) ** 3))
    else:
        skewness = 0.0

    # 6. Graph density
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    # 7. Approximate power-law exponent via Hill estimator
    #    alpha < 2.5 typically indicates heavy-tailed (LP-friendly)
    sorted_deg = np.sort(degrees)[::-1]
    k = max(int(0.1 * n), 10)  # Top 10% of nodes
    if k < len(sorted_deg) and sorted_deg[k-1] > 1:
        log_sum = np.sum(np.log(sorted_deg[:k] / sorted_deg[k-1]))
        if log_sum > 1e-12:
            alpha_hill = 1.0 + k / log_sum
        else:
            alpha_hill = 3.0  # Default (not power-law)
    else:
        alpha_hill = 3.0

    return {
        'cv': float(cv),
        'hub_ratio': float(hub_ratio),
        'low_deg_frac': float(low_deg_frac),
        'skewness': float(skewness),
        'density': float(density),
        'alpha_hill': float(alpha_hill),
        'avg_degree': float(avg_deg),
        'isolated_frac': float(isolated_frac),
    }


def select_coarsening_method(G: Graph, verbose: bool = False) -> str:
    """
    Automatically select coarsening method based on graph structure.

    Decision rules based on graph theory and empirical observation:

    1. High degree variance (CV > 1.5) -> power-law-like -> LP
       Matching struggles because high-degree hubs cannot find partners

    2. Extreme hubs (hub_ratio > 30) -> LP
       A vertex with 30x average degree will remain unmatched

    3. Many low-degree nodes (> 30%) -> LP
       Chains and leaves create matching dead-ends

    4. Heavy-tailed distribution (alpha < 2.5) -> LP
       Classic power-law indicator (Barabasi-Albert has alpha ~ 3)

    5. Many isolated vertices (> 10%) -> LP
       Disconnected components need clustering, not matching

    6. High skewness (> 3) -> LP
       Asymmetric degree distribution with outlier hubs

    Parameters
    ----------
    G : Graph
        Input graph to analyze.
    verbose : bool
        If True, print the decision rationale.

    Returns
    -------
    method : str
        "label_propagation" or "expansion2"
    """
    feat = compute_coarsening_features(G)

    reasons = []

    # Rule 1: High coefficient of variation
    if feat['cv'] > 1.5:
        reasons.append("cv={:.2f} > 1.5 (high degree heterogeneity)".format(feat['cv']))

    # Rule 2: Extreme hub ratio
    if feat['hub_ratio'] > 30:
        reasons.append("hub_ratio={:.1f} > 30 (extreme hubs)".format(feat['hub_ratio']))

    # Rule 3: Many low-degree nodes
    if feat['low_deg_frac'] > 0.30:
        reasons.append("low_deg_frac={:.2%} > 30% (many leaves/chains)".format(feat['low_deg_frac']))

    # Rule 4: Heavy-tailed (power-law) distribution
    if feat['alpha_hill'] < 2.5:
        reasons.append("alpha_hill={:.2f} < 2.5 (heavy-tailed)".format(feat['alpha_hill']))

    # Rule 5: Many isolated/near-isolated vertices
    if feat['isolated_frac'] > 0.10:
        reasons.append("isolated_frac={:.2%} > 10% (disconnected)".format(feat['isolated_frac']))

    # Rule 6: High skewness
    if feat['skewness'] > 3.0:
        reasons.append("skewness={:.2f} > 3.0 (asymmetric)".format(feat['skewness']))

    use_lp = len(reasons) >= 1  # Any single reason triggers LP

    if verbose and reasons:
        print("  [method-select] LP triggered: {}".format('; '.join(reasons)))
    elif verbose:
        print("  [method-select] Using expansion2 (cv={:.2f}, hub_ratio={:.1f}, alpha={:.2f})".format(
            feat['cv'], feat['hub_ratio'], feat['alpha_hill']))

    return "label_propagation" if use_lp else "expansion2"


def get_coarsening_feature_summary(G: Graph) -> str:
    """
    Return a one-line summary of coarsening-relevant features.
    Useful for logging during training.
    """
    feat = compute_coarsening_features(G)
    method = select_coarsening_method(G, verbose=False)
    return "cv={:.2f}, hub={:.1f}, low_deg={:.1%}, alpha={:.2f} -> {}".format(
        feat['cv'], feat['hub_ratio'], feat['low_deg_frac'],
        feat['alpha_hill'], method)


# =========================================================================
# Main extraction function
# =========================================================================

def extract_structural_features(
    G: Graph,
    k: int,
    partition: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract the full structural feature vector psi(G).

    Returns dict with keys:
      --- Basic statistics ---
      n, m, density, avg_degree, degree_variance, max_degree

      --- Spectral features ---
      lambda2, spectral_gap_ratio, eigengap_k, eigengap_ratio,
      spectral_width, donath_hoffman_bound

      --- Higher-order Cheeger analysis (Lee-Oveis Gharan-Trevisan) ---
      cheeger_k_lower, cheeger_k_upper, cheeger_k_gap, spectral_k_separation

      --- Graph minor theory features (Kawarabayashi-Kobayashi) ---
      treewidth_estimate, treewidth_upper, treewidth_ratio, treewidth_ratio_upper,
      treewidth_sandwich_width,
      separator_ratio, planarity_score, hadwiger_estimate, ast_separator_bound

      --- Community/structure features ---
      conductance_estimate, clustering_coeff, modularity,
      power_law_exponent, degree_entropy

      --- Coarsening fidelity (Loukas) ---
      coarsening_fidelity_mean, coarsening_fidelity_max

      --- Conductance profile shape ---
      conductance_profile_shape

      --- GNN expressiveness indicators (Xu-Kawarabayashi-Jegelka) ---
      wl_colours, wl_regularity, wl_fraction_distinguished

      --- Derived indicators ---
      graph_family, rl_advantage_score
    """
    features = {}

    # === Basic statistics ===
    features['n'] = G.n
    features['m'] = G.m
    features['density'] = 2.0 * G.m / max(G.n * (G.n - 1), 1)
    features['avg_degree'] = float(G.degrees.mean())
    features['degree_variance'] = float(G.degrees.var())
    features['max_degree'] = float(G.degrees.max())

    # === Spectral features ===
    num_eigs = min(2 * k + 2, G.n - 1)
    eigenvalues = None
    try:
        eigenvalues, eigenvectors = compute_eigenvectors(
            G, num_eigs, normalized=True
        )
        features['lambda2'] = (
            float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        )
        features['spectral_gap_ratio'] = spectral_gap_ratio(eigenvalues)
        features['eigengap_k'] = eigengap(eigenvalues, k)
        features['eigengap_ratio'] = (
            features['eigengap_k'] / max(eigenvalues[k - 1], 1e-12)
            if k < len(eigenvalues) else 0.0
        )
        features['spectral_width'] = compute_spectral_width(eigenvalues)
    except Exception:
        features['lambda2'] = 0.0
        features['spectral_gap_ratio'] = 0.0
        features['eigengap_k'] = 0.0
        features['eigengap_ratio'] = 0.0
        features['spectral_width'] = 0.0

    # === Higher-order Cheeger analysis ===
    if eigenvalues is not None:
        cheeger = compute_higher_order_cheeger_bound(eigenvalues, k)
    else:
        cheeger = {
            'cheeger_k_lower': 0.0, 'cheeger_k_upper': 0.0,
            'cheeger_k_gap': 0.0, 'spectral_k_separation': 0.0,
        }
    features.update(cheeger)

    # === Conductance ===
    try:
        features['conductance_estimate'] = estimate_conductance(G)
    except Exception:
        features['conductance_estimate'] = 0.0

    # === Clustering coefficient ===
    if G.n <= 50000:
        features['clustering_coeff'] = compute_clustering_coefficient(G)
    else:
        features['clustering_coeff'] = _sampled_clustering_coefficient(G)

    # === Modularity ===
    if partition is not None:
        features['modularity'] = estimate_modularity(G, partition)
    else:
        features['modularity'] = 0.0

    # === Power-law exponent ===
    features['power_law_exponent'] = estimate_power_law_exponent(G)

    # === Degree entropy ===
    features['degree_entropy'] = compute_degree_entropy(G)

    # === Treewidth and graph minor theory features ===
    if G.n <= 100000:
        features['treewidth_estimate'] = estimate_treewidth(G)
    else:
        features['treewidth_estimate'] = features['max_degree']

    features['treewidth_ratio'] = compute_treewidth_ratio(
        features['treewidth_estimate'], G.n
    )

    # === Separator quality ===
    if G.n <= 200000:
        features['separator_ratio'] = compute_separator_quality(G)
    else:
        features['separator_ratio'] = 0.5

    # === Treewidth upper bound (sandwich) ===
    features['treewidth_upper'] = compute_treewidth_upper_bound(
        G, features['separator_ratio']
    )
    features['treewidth_ratio_upper'] = compute_treewidth_ratio(
        features['treewidth_upper'], G.n
    )
    features['treewidth_sandwich_width'] = (
        features['treewidth_upper'] - features['treewidth_estimate']
    )

    # === Planarity score ===
    features['planarity_score'] = estimate_planarity_score(G)

    # === Hadwiger number and AST separator bound ===
    features['hadwiger_estimate'] = estimate_hadwiger_number(G)
    features['ast_separator_bound'] = compute_ast_separator_bound(
        features['hadwiger_estimate'], G.n
    )

    # === Donath-Hoffman bound ===
    try:
        features['donath_hoffman_bound'] = donath_hoffman_bound(G, k)
    except Exception:
        features['donath_hoffman_bound'] = 0.0

    # === Coarsening fidelity ===
    if G.n <= 100000:
        try:
            fidelity = compute_coarsening_spectral_fidelity(G, k)
            features['coarsening_fidelity_mean'] = (
                float(np.mean(fidelity)) if fidelity else 0.0
            )
            features['coarsening_fidelity_max'] = (
                float(np.max(fidelity)) if fidelity else 0.0
            )
        except Exception:
            features['coarsening_fidelity_mean'] = 0.0
            features['coarsening_fidelity_max'] = 0.0
    else:
        features['coarsening_fidelity_mean'] = 0.0
        features['coarsening_fidelity_max'] = 0.0

    # === Conductance profile shape ===
    if G.n <= 100000:
        try:
            profile = compute_conductance_profile(G)
            features['conductance_profile_shape'] = (
                classify_conductance_profile_shape(profile)
            )
        except Exception:
            features['conductance_profile_shape'] = 'unknown'
    else:
        features['conductance_profile_shape'] = 'unknown'

    # === GNN expressiveness / WL analysis ===
    if G.n <= 30000:
        wl = compute_wl_features(G)
        features.update(wl)
    else:
        # For large graphs, use degree distribution as proxy
        unique_degs = len(np.unique(G.degrees.astype(int)))
        features['wl_colours'] = float(unique_degs)
        features['wl_regularity'] = 0.0
        features['wl_fraction_distinguished'] = float(unique_degs) / max(G.n, 1)

    # === Graph family classification ===
    features['graph_family'] = classify_graph_family(features)

    # === RL advantage prediction ===
    features['rl_advantage_score'] = predict_rl_advantage(features)

    return features


# =========================================================================
# RL advantage predictor
# =========================================================================

def predict_rl_advantage(features: Dict[str, float]) -> float:
    """
    Predict when RL refinement outperforms FM-only refinement.

    Returns a score in [0, 1] where higher = more likely RL helps.

    Incorporates insights from graph minor theory and GNN expressiveness:

    1. Eigengap component (Davis-Kahan perturbation theory):
       Small eigengap ratio -> weak cluster separation in the spectrum ->
       spectral initial partition is noisy -> RL can discover non-spectral
       structure that improves the partition.

    2. Clustering component:
       High clustering -> strong local triangle structure -> RL's GNN
       can exploit neighbourhood patterns that FM's greedy moves miss.

    3. Power-law component:
       Power-law degree distribution -> standard matching-based coarsening
       fails on high-degree hubs -> RL opportunity.

    4. Treewidth component (Kawarabayashi-Kobayashi):
       High tw/sqrt(n) ratio -> graph is NOT mesh-like, NOT minor-free
       in the planar sense -> coarsening may distort structure -> RL
       refinement at the coarsest level can correct propagation errors.

    5. Cheeger gap component (Lee-Oveis Gharan-Trevisan):
       Large gap between upper and lower Cheeger bounds -> spectral
       methods give loose guarantees -> room for RL improvement.

    6. WL regularity component (Xu-Kawarabayashi-Jegelka):
       High regularity fraction -> GNN needs spectral PE to distinguish
       vertices -> SRMP's SignNet architecture provides advantage over
       vanilla GNN approaches.

    7. Coarsening fidelity component (Loukas):
       High spectral distortion during coarsening -> information lost in
       multilevel hierarchy -> RL at coarsest level can correct errors.
    """
    eigengap_ratio = features.get('eigengap_ratio', 0.0)
    clustering = features.get('clustering_coeff', 0.0)
    power_law = features.get('power_law_exponent', 0.0)
    tw_ratio = features.get('treewidth_ratio', 0.0)
    cheeger_gap = features.get('cheeger_k_gap', 0.0)
    wl_regularity = features.get('wl_regularity', 0.0)
    fidelity = features.get('coarsening_fidelity_mean', 0.0)

    # --- Component 1: Eigengap (Davis-Kahan) ---
    eigengap_score = 1.0 / (1.0 + eigengap_ratio * 10)

    # --- Component 2: Clustering ---
    clustering_score = clustering

    # --- Component 3: Power-law ---
    powerlaw_score = 1.0 if 2.0 < power_law < 3.5 else 0.0

    # --- Component 4: Treewidth ratio (Kawarabayashi-Kobayashi) ---
    tw_score = min(tw_ratio / 5.0, 1.0) if tw_ratio > 0 else 0.0

    # --- Component 5: Cheeger gap (Lee-Oveis Gharan-Trevisan) ---
    if cheeger_gap == float('inf') or cheeger_gap == 0.0:
        cheeger_score = 0.5
    else:
        cheeger_score = min(cheeger_gap / 20.0, 1.0)

    # --- Component 6: WL regularity (Xu-Kawarabayashi-Jegelka) ---
    wl_score = wl_regularity

    # --- Component 7: Coarsening fidelity (Loukas) ---
    fidelity_score = min(fidelity / 0.3, 1.0) if fidelity > 0 else 0.0

    # Weighted combination
    score = (
        0.20 * eigengap_score
        + 0.10 * clustering_score
        + 0.10 * powerlaw_score
        + 0.20 * tw_score
        + 0.10 * cheeger_score
        + 0.10 * wl_score
        + 0.20 * fidelity_score
    )

    return float(max(0.0, min(1.0, score)))