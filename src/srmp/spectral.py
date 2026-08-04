"""
Spectral Graph Analysis
========================
Computes graph Laplacians, eigendecompositions, and spectral features
for use in partitioning and structural analysis.

Theory:
  - Fiedler (1973): algebraic connectivity lambda2
  - Cheeger inequality: lambda2/2 <= phi(G) <= sqrt(2lambda2)
  - Lee-Oveis Gharan-Trevisan (2014): higher-order Cheeger inequality
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg
from typing import Tuple, Optional
from .graph import Graph


def unnormalized_laplacian(G: Graph) -> sparse.csr_matrix:
    """
    Compute the unnormalised Laplacian L = D - W.

    Satisfies: f^T L f = (1/2) Sum_{ij} w_{ij}(f_i - f_j)^2
    """
    D = sparse.diags(G.degrees, format='csr')
    return D - G.adj


def normalized_laplacian_sym(G: Graph) -> sparse.csr_matrix:
    """
    Compute the symmetric normalised Laplacian:
        L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}

    Eigenvalues lie in [0, 2].
    """
    deg = G.degrees.copy()
    deg[deg == 0] = 1.0  # Avoid division by zero for isolated vertices
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg), format='csr')
    return sparse.eye(G.n) - D_inv_sqrt @ G.adj @ D_inv_sqrt


def normalized_laplacian_rw(G: Graph) -> sparse.csr_matrix:
    """
    Compute the random-walk normalised Laplacian:
        L_rw = D^{-1} L = I - D^{-1} W
    """
    deg = G.degrees.copy()
    deg[deg == 0] = 1.0
    D_inv = sparse.diags(1.0 / deg, format='csr')
    return sparse.eye(G.n) - D_inv @ G.adj


def compute_eigenvectors(
    G: Graph,
    num_eigenvectors: int = 8,
    normalized: bool = True,
    method: str = "arpack"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the smallest eigenvalues and eigenvectors of the Laplacian.

    Parameters
    ----------
    G : Graph
        Input graph.
    num_eigenvectors : int
        Number of smallest eigenpairs to compute.
    normalized : bool
        If True, use symmetric normalised Laplacian L_sym.
        If False, use unnormalised Laplacian L.
    method : str
        "arpack" for ARPACK (default) or "lobpcg" for LOBPCG.

    Returns
    -------
    eigenvalues : np.ndarray, shape (num_eigenvectors,)
    eigenvectors : np.ndarray, shape (n, num_eigenvectors)
        Column j is the eigenvector for eigenvalue j.
        For normalised Laplacian, these are eigenvectors of L_sym.
    """
    k = min(num_eigenvectors, G.n - 1)

    if normalized:
        L = normalized_laplacian_sym(G)
    else:
        L = unnormalized_laplacian(G)

    # Ensure matrix is in CSR format for sparse eigensolvers
    L = L.tocsr()

    if method == "lobpcg" and G.n > 1000:
        # LOBPCG is faster for large sparse matrices
        rng = np.random.RandomState(42)
        X = rng.randn(G.n, k)
        try:
            eigenvalues, eigenvectors = lobpcg(
                L, X, largest=False, maxiter=500, tol=1e-8
            )
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            return eigenvalues[idx], eigenvectors[:, idx]
        except Exception:
            pass  # Fall through to ARPACK

    # ARPACK: use shift-invert mode for smallest eigenvalues.
    # sigma=0 + which='LM' is the canonical recipe: ARPACK inverts (L - 0*I)
    # and finds eigenvalues of (L)^-1 with largest magnitude, which correspond
    # to the smallest eigenvalues of L. This is robust on ill-conditioned L.
    try:
        eigenvalues, eigenvectors = eigsh(
            L, k=k, sigma=0.0, which='LM', maxiter=5000, tol=1e-10,
        )
    except Exception:
        # Some Laplacians are too singular for the inversion: fall back to
        # the small-magnitude direct mode (slower, less reliable, but safe).
        try:
            eigenvalues, eigenvectors = eigsh(
                L, k=k, which='SM', maxiter=5000, tol=1e-10,
            )
        except Exception:
            # Last-resort: tiny shift to break exact singularity.
            L_shifted = L + 1e-10 * sparse.eye(G.n)
            eigenvalues, eigenvectors = eigsh(
                L_shifted, k=k, sigma=0.0, which='LM', maxiter=5000, tol=1e-8,
            )

    # --- robustness net (2026-07): older ARPACK/scipy returns UNCONVERGED
    #     eigenvectors SILENTLY for sigma=0 shift-invert on singular Laplacians
    #     (observed scipy 1.12: residual ~1e-1). Verify and redo densely if bad. ---
    _res = max((float(np.linalg.norm(L @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i]))
                for i in range(eigenvectors.shape[1])), default=0.0)
    if _res > 1e-6:
        if G.n <= 20000:
            _w, _V = np.linalg.eigh(L.toarray())
            eigenvalues, eigenvectors = _w[:k], _V[:, :k]
        else:
            # Dense verification is infeasible at this size (n^2 memory).
            # Retry with a small diagonal shift and a larger Krylov basis,
            # then keep whichever basis has the smaller residual.
            try:
                L_shifted = L + 1e-8 * sparse.eye(G.n)
                _ev, _eV = eigsh(L_shifted, k=k, sigma=0.0, which='LM',
                                 maxiter=20000, tol=1e-8,
                                 ncv=min(G.n - 1, max(4 * k + 1, 64)))
                _res2 = max(float(np.linalg.norm(L @ _eV[:, i] - _ev[i] * _eV[:, i]))
                            for i in range(_eV.shape[1]))
                if _res2 < _res:
                    eigenvalues, eigenvectors = _ev, _eV
                    _res = _res2
            except Exception:
                pass
            print(f"  [eigensolver] residual {_res:.2e} at n={G.n} "
                  f"(dense fallback skipped: too large); using best basis")

    # Sort by eigenvalue (eigsh doesn't guarantee order)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clip small negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return eigenvalues, eigenvectors


def fiedler_vector(G: Graph, normalized: bool = True) -> Tuple[float, np.ndarray]:
    """
    Compute algebraic connectivity lambda2 and the Fiedler vector.

    Returns
    -------
    lambda2 : float
        Algebraic connectivity.
    fiedler : np.ndarray, shape (n,)
        Fiedler vector (eigenvector for lambda2).
    """
    eigenvalues, eigenvectors = compute_eigenvectors(G, 2, normalized)
    return float(eigenvalues[1]), eigenvectors[:, 1]


def spectral_embedding(
    G: Graph,
    k: int,
    normalized: bool = True,
    row_normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute k-dimensional spectral embedding (Ng-Jordan-Weiss, 2002).

    Parameters
    ----------
    G : Graph
    k : int
        Embedding dimension (= number of desired clusters).
    normalized : bool
        Use normalised Laplacian.
    row_normalize : bool
        Row-normalise the embedding matrix (NJW step 3).

    Returns
    -------
    eigenvalues : np.ndarray, shape (k+1,)
        First k+1 eigenvalues (including lambda1=0).
    eigenvectors : np.ndarray, shape (n, k+1)
    embedding : np.ndarray, shape (n, k)
        The k-dimensional embedding (columns 1..k, skipping lambda1=0).
    """
    num_eigs = min(k + 1, G.n - 1)
    eigenvalues, eigenvectors = compute_eigenvectors(
        G, num_eigs, normalized
    )

    # Use eigenvectors 1..k (skip constant eigenvector for lambda1=0)
    embedding = eigenvectors[:, 1:k + 1].copy()

    if row_normalize:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0  # Avoid division by zero
        embedding = embedding / norms

    return eigenvalues[:num_eigs], eigenvectors[:, :num_eigs], embedding


def estimate_conductance(G: Graph) -> float:
    """
    Estimate graph conductance phi(G) via the Cheeger bound.

    Uses the sweep algorithm on the Fiedler vector:
    sort vertices by Fiedler coordinate, sweep through all prefix cuts,
    return the minimum conductance found.

    Theorem (Cheeger): lambda2/2 <= phi(G) <= sqrt(2lambda2)
    The sweep cut achieves at most sqrt(2lambda2).
    """
    _, fiedler = fiedler_vector(G, normalized=True)

    # Sort vertices by Fiedler coordinate
    order = np.argsort(fiedler)
    deg = G.degrees

    total_vol = deg.sum()
    vol_S = 0.0
    cut_S = 0.0
    best_conductance = float('inf')

    # Track which vertices are in S
    in_S = np.zeros(G.n, dtype=bool)

    for i in range(G.n - 1):  # Don't include the last vertex (full set)
        v = order[i]
        in_S[v] = True
        vol_S += deg[v]

        # Update cut: for each neighbor of v
        nbrs, weights = G.neighbors(v)
        for nb, w in zip(nbrs, weights):
            if in_S[nb]:
                cut_S -= w  # Edge now internal
            else:
                cut_S += w  # Edge now crossing

        # Conductance: cut(S) / min(vol(S), vol(V\S))
        vol_complement = total_vol - vol_S
        if vol_S > 0 and vol_complement > 0:
            denom = min(vol_S, vol_complement)
            phi = cut_S / denom
            best_conductance = min(best_conductance, phi)

    return best_conductance


def eigengap(eigenvalues: np.ndarray, k: int) -> float:
    """
    Compute the eigengap Delta_k = lambda_{k+1} - lambda_k.

    A large eigengap indicates well-separated cluster structure
    (Davis-Kahan perturbation theory).
    """
    if k >= len(eigenvalues):
        return 0.0
    return float(eigenvalues[k] - eigenvalues[k - 1])


def spectral_gap_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute lambda2/lambda_n ratio.

    Small ratio indicates hard-to-partition structure (expander-like).
    """
    if len(eigenvalues) < 2:
        return 0.0
    lam_n = eigenvalues[-1]
    if lam_n < 1e-12:
        return 0.0
    return float(eigenvalues[1] / lam_n)


def donath_hoffman_bound(G: Graph, k: int) -> float:
    """
    Compute the Donath-Hoffman lower bound on the k-way cut.

    Theorem (Donath-Hoffman, 1973):
        Sum_{i<j} e(V_i, V_j) >= (1/2) Sum_{l=2}^{k} n_l . mu_l

    For balanced k-partition with n_i ~= n/k:
        cut >= (n/k) . (1/2) . Sum_{l=2}^{k} mu_l
    """
    eigenvalues, _ = compute_eigenvectors(G, k, normalized=False)

    # Balanced partition: each part has ~n/k vertices
    n_part = G.n // k
    bound = 0.0
    for ell in range(1, min(k, len(eigenvalues))):
        bound += n_part * eigenvalues[ell]
    return bound / 2.0
