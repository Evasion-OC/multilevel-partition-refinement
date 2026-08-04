"""
Thesis Feature Vector (T1, C2, C3)
===================================
Focused feature extraction for the algorithm-structure matching thesis.

The broad-spectrum extractor in ``structural_features.py`` produces ~40
heuristic features across many paradigms. For the thesis experiments we want
a *small, interpretable, theory-aligned* feature set whose entries each
appear explicitly in one of the three results:

  T1 (theorem, 1-WL vs GNN separation):
      wl_regularity, wl_colour_ratio, cfi_indicator, fiedler_ratio
  C2 (conjecture, coarsening phase transition):
      alpha_hill, cv, hub_ratio, low_deg_frac, skewness
  C3 (conjecture, classifier consistency):
      all of the above plus spectral_gap, treewidth_ratio, planarity_score

The JSON record emitted by ``run_benchmark.py`` is extended with:

  result["thesis_features"]      : this focused dict at the fine graph
  result["coarsest_features"]    : the same dict at the coarsest level
  result["coarsening_fidelity"]  : per-method fidelity comparison

This file contains *only* what the thesis experiments depend on. It must
stay small --- if a feature stops appearing in T1/C2/C3, delete it here.

References:
  - Cai, Fuerer, Immerman (1992). An optimal lower bound on the number of
    variables for graph identification.
  - Sato, R. (2020). A survey on the expressive power of graph neural
    networks. arXiv:2003.04078.
  - Xu, Hu, Leskovec, Jegelka (ICLR 2019). How powerful are GNNs?
  - Lim et al. (ICML 2023). Sign and basis invariant networks for spectral
    graph representation learning.
  - Koutis, Miller, Peng (2010-14). Preconditioning and spectral
    approximation bounds.
  - Loukas (ICML 2019). Graph reduction with spectral and cut guarantees.
"""
import numpy as np
from typing import Dict, Optional, List

from .graph import Graph
from .spectral import (
    compute_eigenvectors, spectral_gap_ratio, fiedler_vector,
)


# =========================================================================
# Core thesis feature set (T1, C2, C3)
# =========================================================================

_DEFAULTS = {
    # --- T1 features (WL/GNN expressiveness) ---
    "fiedler_ratio":       0.0,   # lambda_2 / lambda_n
    "wl_colour_ratio":     1.0,   # |WL stable colours| / n
    "wl_regularity":       0.0,   # largest WL colour class / n
    "cfi_indicator":       0.0,   # proxy: 1 - wl_fraction_distinguished

    # --- C2 features (coarsening phase transition) ---
    "alpha_hill":          3.0,
    "cv":                  0.0,
    "hub_ratio":           1.0,
    "low_deg_frac":        0.0,
    "skewness":            0.0,

    # --- C3 features (classifier consistency -- global structure) ---
    "spectral_gap":        0.0,   # lambda_2 of the normalised Laplacian
    "n":                   0,
    "m":                   0,
    "avg_degree":          0.0,
    "density":             0.0,
}


def _degree_features(G: Graph) -> Dict[str, float]:
    """Compute the degree-tail part of the feature vector (C2)."""
    degrees = G.degrees.astype(float)
    n = max(G.n, 1)
    if n < 2 or degrees.sum() == 0:
        return {
            "alpha_hill": 3.0, "cv": 0.0, "hub_ratio": 1.0,
            "low_deg_frac": 0.0, "skewness": 0.0,
            "avg_degree": 0.0, "density": 0.0,
        }

    mean_d = float(degrees.mean())
    std_d = float(degrees.std())
    max_d = float(degrees.max())

    cv = std_d / mean_d if mean_d > 0 else 0.0
    hub_ratio = max_d / mean_d if mean_d > 0 else 1.0
    low_deg_frac = float((degrees <= 2).sum()) / n

    if std_d > 1e-12:
        skewness = float(np.mean(((degrees - mean_d) / std_d) ** 3))
    else:
        skewness = 0.0

    # Hill estimator on top 10% of degrees, restricted to positive-degree vertices.
    # Returns NaN when the estimator is not well-defined (too few vertices, or
    # the top-10% threshold has only 1-degree vertices, or numerical underflow).
    # Downstream code MUST treat NaN as "feature unavailable" rather than
    # silently substituting a default value.
    pos_deg = degrees[degrees > 0]
    sorted_deg = np.sort(pos_deg)[::-1]
    k = max(int(0.1 * len(sorted_deg)), 10) if len(sorted_deg) > 0 else 0
    if k >= 2 and k <= len(sorted_deg) and sorted_deg[k - 1] > 1:
        ratios = sorted_deg[:k] / sorted_deg[k - 1]
        log_sum = float(np.sum(np.log(ratios)))
        alpha_hill = (1.0 + k / log_sum) if log_sum > 1e-12 else float("nan")
    else:
        alpha_hill = float("nan")

    density = 2.0 * G.m / (n * (n - 1)) if n > 1 else 0.0

    return {
        "alpha_hill": float(alpha_hill),
        "cv": float(cv),
        "hub_ratio": float(hub_ratio),
        "low_deg_frac": float(low_deg_frac),
        "skewness": float(skewness),
        "avg_degree": float(mean_d),
        "density": float(density),
    }


def _spectral_features(G: Graph) -> Dict[str, float]:
    """Compute the spectral part of the feature vector (T1, C3).

    Returns NaN sentinels when:
      - The graph is too small for the requested eigenpair count, OR
      - The graph is disconnected (multiple zero eigenvalues mean the
        "spectral gap" and "Fiedler ratio" are not measuring algebraic
        connectivity any more), OR
      - The eigensolver fails.

    Downstream code MUST treat NaN as "feature unavailable" instead of
    silently substituting 0.0.
    """
    nan = float("nan")
    n_eigs = min(8, G.n - 1)
    if n_eigs < 2:
        return {"spectral_gap": nan, "fiedler_ratio": nan}
    try:
        from scipy.sparse.csgraph import connected_components
        n_cc, _ = connected_components(G.adj, directed=False)
        if n_cc > 1:
            return {"spectral_gap": nan, "fiedler_ratio": nan}
    except Exception:
        # If connectivity check itself fails, fall through and let the
        # eigensolver decide.
        pass
    try:
        eigenvalues, _ = compute_eigenvectors(G, n_eigs, normalized=True)
        lam2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else nan
        return {
            "spectral_gap": lam2,
            "fiedler_ratio": spectral_gap_ratio(eigenvalues),
        }
    except Exception:
        return {"spectral_gap": nan, "fiedler_ratio": nan}


def _wl_features(G: Graph, max_iters: int = 20) -> Dict[str, float]:
    """
    1-WL colour refinement statistics (T1).

    Computes three scalars:
      wl_colour_ratio : |stable colour classes| / n  --- fine colourings near 1
      wl_regularity   : |largest colour class| / n   --- highly regular near 1
      cfi_indicator   : 1 - fraction of uniquely coloured vertices
                        high means many 1-WL-indistinguishable vertex
                        pairs, i.e. the CFI-like regime
    """
    n = G.n
    if n == 0:
        return {"wl_colour_ratio": 1.0, "wl_regularity": 0.0,
                "cfi_indicator": 0.0}

    colours = G.degrees.astype(int).copy()
    for _ in range(min(n, max_iters)):
        colour_map: Dict = {}
        new_colours = np.zeros(n, dtype=np.int64)
        next_c = 0
        for v in range(n):
            nbrs, _ = G.neighbors(v)
            key = (int(colours[v]), tuple(sorted(colours[nbrs].tolist())))
            if key not in colour_map:
                colour_map[key] = next_c
                next_c += 1
            new_colours[v] = colour_map[key]
        if len(np.unique(new_colours)) == len(np.unique(colours)):
            break
        colours = new_colours

    unique, counts = np.unique(colours, return_counts=True)
    num_unique_singletons = int((counts == 1).sum())
    return {
        "wl_colour_ratio": float(len(unique) / max(n, 1)),
        "wl_regularity": float(counts.max() / max(n, 1)),
        "cfi_indicator": 1.0 - float(num_unique_singletons / max(n, 1)),
    }


def thesis_feature_vector(
    G: Graph,
    skip_wl_above: int = 30_000,
) -> Dict[str, float]:
    """
    Focused, thesis-aligned structural feature vector.

    Returns a dict with exactly the keys declared in ``_DEFAULTS``. For graphs
    larger than ``skip_wl_above`` the WL features are replaced with cheap
    degree-based proxies (1-WL is O(n * max_iter * avg_deg) which gets costly
    on >50k-vertex meshes). This keeps the feature vector comparable across
    graph scales without making benchmark runs 5x slower on large graphs.

    Use this function everywhere the thesis chapters 4-6 read features.
    """
    features = dict(_DEFAULTS)
    features["n"] = int(G.n)
    features["m"] = int(G.m)

    features.update(_degree_features(G))
    features.update(_spectral_features(G))

    if G.n <= skip_wl_above:
        features.update(_wl_features(G))
    else:
        # Degree-based proxy when 1-WL is too expensive.
        unique_degs = len(np.unique(G.degrees.astype(int)))
        features["wl_colour_ratio"] = float(unique_degs) / max(G.n, 1)
        features["wl_regularity"] = 0.0
        features["cfi_indicator"] = 1.0 - features["wl_colour_ratio"]

    return features


# =========================================================================
# Coarsening fidelity per method (C2)
# =========================================================================

def _cut_along_fiedler(G: Graph) -> float:
    """
    Return the minimum ratio-cut along the Fiedler sweep.

    Used as a reproducible, method-agnostic cut benchmark at each level.
    On a fine graph this approximates the best spectral bisection; on a
    coarsened graph it measures the cut the coarsening method committed to.

    Returns NaN when:
      - The graph is too small for a balanced split (n < 3), OR
      - The Fiedler eigensolve fails, OR
      - No prefix size lies in [n/3, 2n/3] (very small/imbalanced graphs).

    Downstream code MUST treat NaN as "no measurable cut" rather than 0.0,
    which is indistinguishable from an exact preservation.
    """
    nan = float("nan")
    if G.n < 3:
        return nan
    try:
        _, fiedler = fiedler_vector(G, normalized=True)
    except Exception:
        return nan

    order = np.argsort(fiedler)
    in_S = np.zeros(G.n, dtype=bool)
    cut = 0.0
    best = float("inf")
    for i in range(G.n - 1):
        v = order[i]
        in_S[v] = True
        nbrs, weights = G.neighbors(v)
        for nb, w in zip(nbrs, weights):
            if in_S[nb]:
                cut -= float(w)
            else:
                cut += float(w)
        size_S = i + 1
        if G.n // 3 <= size_S <= 2 * G.n // 3:
            if cut < best:
                best = cut
    return float(best) if best < float("inf") else nan


def coarsening_fidelity_by_method(
    G: Graph,
    k: int,
    methods: Optional[List[str]] = None,
    min_vertices: int = 50,
    max_levels: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    For each method in ``methods`` (default {expansion2, label_propagation}),
    build the full coarsening hierarchy, then measure the *Fiedler-cut
    distortion*: ratio of the coarsest-level Fiedler cut to the fine-level
    Fiedler cut. Lower means the coarsening preserved the true bisection.

    This is the empirical quantity for Conjecture C2. It is computed once
    per graph per benchmark record; add < 2s overhead for n <= 100k.

    Returns
    -------
    dict with keys = method name, values = dict with:
        fine_cut        : float  -- min ratio-cut along Fiedler sweep on G
        coarse_cut      : float  -- same on the coarsest graph
        distortion      : float  -- coarse_cut / fine_cut (1 = faithful)
        coarsest_n      : int    -- number of coarsest-level vertices
        levels          : int    -- number of coarsening levels used
    """
    from .coarsening import build_hierarchy

    if methods is None:
        methods = ["expansion2", "label_propagation"]

    fine_cut = _cut_along_fiedler(G)
    results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        try:
            hierarchy = build_hierarchy(
                G, min_vertices=max(min_vertices, 2 * k),
                max_levels=max_levels, method=method, seed=42,
                verbose=False,
            )
            coarsest = hierarchy[-1].coarse_graph if hierarchy else G
            coarse_cut = _cut_along_fiedler(coarsest)
            # NaN propagation: if either cut is NaN (Fiedler failed or no
            # balanced sweep) propagate NaN rather than substituting 0.0,
            # which would silently look like "perfect preservation".
            if not np.isfinite(fine_cut) or not np.isfinite(coarse_cut):
                distortion = float("nan")
            elif abs(fine_cut) > 1e-9:
                distortion = coarse_cut / fine_cut
            else:
                distortion = float("nan")
            results[method] = {
                "fine_cut": float(fine_cut),
                "coarse_cut": float(coarse_cut),
                "distortion": float(distortion),
                "coarsest_n": int(coarsest.n),
                "levels": int(len(hierarchy)),
            }
        except Exception as e:
            results[method] = {
                "fine_cut": float(fine_cut),
                "coarse_cut": float("nan"),
                "distortion": float("nan"),
                "coarsest_n": 0,
                "levels": 0,
                "error": str(e),
            }

    return results


# =========================================================================
# Full record builder --- used by run_benchmark.py and the phase experiment
# =========================================================================

def thesis_record(
    G: Graph,
    k: int,
    include_coarsening_fidelity: bool = True,
) -> Dict:
    """
    Produce the full thesis-record payload for a graph G.

    This is what run_benchmark.py embeds in each JSON entry under the keys
    ``thesis_features`` and ``coarsening_fidelity``. The coarsest-level
    features are also computed via a cheap (single-method) pass so that
    downstream classifier training does not need to re-run coarsening.
    """
    payload = {
        "thesis_features": thesis_feature_vector(G),
        "coarsening_fidelity": None,
        "coarsest_features": None,
    }

    if include_coarsening_fidelity and G.n >= 50:
        from .coarsening import build_hierarchy
        payload["coarsening_fidelity"] = coarsening_fidelity_by_method(G, k)

        # Coarsest-level features (cheap, reuses auto-selected hierarchy)
        try:
            hierarchy = build_hierarchy(
                G, min_vertices=max(50, 2 * k),
                max_levels=30, method="auto", seed=42, verbose=False,
            )
            coarsest = hierarchy[-1].coarse_graph if hierarchy else G
            payload["coarsest_features"] = thesis_feature_vector(coarsest)
        except Exception:
            payload["coarsest_features"] = None

    return payload
