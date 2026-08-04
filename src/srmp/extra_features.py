"""
Additional structural features (Euler-genus lower bound + effective resistance).

Connects the psi_38 feature vector to:
  - Lipton-Tarjan / Gilbert-Hutchinson-Tarjan: balanced separators of size
    O(sqrt(g*n)) on genus-g graphs.
  - Spielman-Srivastava (and Soma-Yoshida hypergraph extensions): spectral
    sparsification by edge effective resistance.

Self-check: Foster's theorem implies sum_e w_e R_eff(e) = n - 1 on any
connected graph. The returned 'foster_sum' is exactly this; departures
measure the JL approximation error or lack of connectivity.
"""
import math
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from typing import Dict

from .graph import Graph
from .spectral import unnormalized_laplacian


_RESIST_KEYS = ("r_eff_mean", "r_eff_std", "r_eff_min",
                "r_eff_max", "r_eff_median", "foster_sum")


def euler_genus_lower_bound(G: Graph) -> int:
    """
    Lower bound on the orientable genus of G from Euler's formula.
    g >= ceil((m - 3n + 6) / 6). Returns 0 for planar (or trivially small)
    graphs. Loose for bipartite graphs (intentionally; tighter bound not
    applied here).
    """
    if G.n < 3:
        return 0
    return max(0, math.ceil((G.m - 3 * G.n + 6) / 6.0))


def effective_resistance_features(
    G: Graph,
    eps: float = 0.15,
    exact_threshold: int = 1500,
    skip_above: int = 20000,
    jl_constant: float = 4.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Summary statistics of edge effective resistances R_eff(u, v) on the
    unnormalised Laplacian L = D - W of G.

    n <= exact_threshold  : dense pinv (exact).
    n in (exact_threshold, skip_above] : Spielman-Srivastava JL sketch with
        k = ceil(jl_constant * log(n) / eps^2) projection columns.
    n > skip_above        : returns NaN (avoids minutes per huge graph).

    Reference: Spielman & Srivastava (2011), SIAM J. Comput. 40(6):1913-1926.
    """
    nan = float("nan")
    if G.m == 0 or G.n < 2 or G.n > skip_above:
        return {k: nan for k in _RESIST_KEYS}

    coo = G.coo
    mask = coo.row < coo.col
    ei = coo.row[mask].astype(np.int64)
    ej = coo.col[mask].astype(np.int64)
    ew = coo.data[mask].astype(np.float64)
    if ei.size == 0:
        return {k: nan for k in _RESIST_KEYS}

    if G.n <= exact_threshold:
        L_dense = unnormalized_laplacian(G).toarray().astype(np.float64)
        try:
            Lp = np.linalg.pinv(L_dense, hermitian=True)
        except np.linalg.LinAlgError:
            return {k: nan for k in _RESIST_KEYS}
        resistances = (Lp[ei, ei] + Lp[ej, ej] - 2.0 * Lp[ei, ej])
    else:
        rng = np.random.default_rng(seed)
        m_unique = ei.size
        k = max(50, int(math.ceil(
            jl_constant * math.log(max(G.n, 2)) / (eps ** 2))))
        sw = np.sqrt(ew)
        rows = np.concatenate([ei, ej])
        cols = np.concatenate([np.arange(m_unique), np.arange(m_unique)])
        vals = np.concatenate([sw, -sw])
        BtW = sparse.csr_matrix((vals, (rows, cols)),
                                shape=(G.n, m_unique))
        Q = rng.choice([-1.0, 1.0], size=(k, m_unique)).astype(np.float64)
        Q /= math.sqrt(k)
        rhs = BtW @ Q.T
        L_reg = unnormalized_laplacian(G).astype(np.float64) \
                + sparse.eye(G.n) * 1e-10
        Y = np.zeros((G.n, k))
        for jj in range(k):
            try:
                y, _ = spla.cg(L_reg, rhs[:, jj], tol=1e-6, maxiter=2000)
            except TypeError:
                y, _ = spla.cg(L_reg, rhs[:, jj], rtol=1e-6, maxiter=2000)
            y -= y.mean()
            Y[:, jj] = y
        diffs = Y[ei] - Y[ej]
        resistances = np.einsum('ij,ij->i', diffs, diffs)

    return {
        "r_eff_mean":   float(np.mean(resistances)),
        "r_eff_std":    float(np.std(resistances)),
        "r_eff_min":    float(np.min(resistances)),
        "r_eff_max":    float(np.max(resistances)),
        "r_eff_median": float(np.median(resistances)),
        "foster_sum":   float(np.sum(ew * resistances)),
    }
