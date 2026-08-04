#!/usr/bin/env python3
"""Model-free structural descriptors for the ceiling-distance investigation.

For every METIS .graph in --graph-dir, computes:
  n, m, mean/max degree, deg_cv (heterogeneity),
  degen  = degeneracy (max k-core, via Batagelj-Zaversnik peeling),
  tau    = degen / sqrt(n)            (minor-exclusion / treewidth surrogate; the ceiling axis),
  lambda2= algebraic connectivity of the NORMALISED Laplacian (Cheeger),
  planar = exact planarity IF networkx is importable, else 'NA'.
Dependency-light: needs only numpy + scipy (+ srmp.read_metis on PYTHONPATH); networkx optional.

  PYTHONPATH=src python src/struct_descriptors.py --graph-dir data/test_all --out results/descriptors.csv
"""
import os, sys, csv, time, argparse, heapq
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from srmp.graph import read_metis

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False


def degeneracy(A):
    """Max k-core number via lazy-heap min-degree peeling. A: symmetric 0/1 CSR, no self-loops."""
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    cur = np.asarray(A.sum(1)).ravel().astype(np.int64)
    removed = np.zeros(n, bool)
    h = [(int(cur[i]), i) for i in range(n)]
    heapq.heapify(h)
    k = 0
    while h:
        d, v = heapq.heappop(h)
        if removed[v] or d != cur[v]:
            continue
        k = max(k, d)
        removed[v] = True
        for u in indices[indptr[v]:indptr[v + 1]]:
            if not removed[u]:
                cur[u] -= 1
                heapq.heappush(h, (int(cur[u]), u))
    return int(k)


def descriptors(path, fam, lam2_max, planar_max):
    G = read_metis(path)
    n = G.n
    A = G.adj.tocsr().astype(float)
    A.setdiag(0); A.eliminate_zeros()
    A = ((A + A.T) > 0).astype(float)                 # symmetric, unweighted, no self-loops
    m = int(A.nnz // 2)
    deg = np.asarray(A.sum(1)).ravel()
    r = {"graph": os.path.splitext(os.path.basename(path))[0], "family": fam,
         "n": n, "m": m, "mean_deg": round(float(deg.mean()), 2), "max_deg": int(deg.max()),
         "deg_cv": round(float(deg.std() / max(deg.mean(), 1e-9)), 3)}
    dg = degeneracy(A)
    r["degen"] = dg; r["tau"] = round(dg / np.sqrt(max(n, 1)), 4)
    if n <= lam2_max:
        d = np.maximum(deg, 1e-12); Dis = sp.diags(1.0 / np.sqrt(d))
        L = sp.eye(n) - Dis @ A @ Dis
        try:
            vals = eigsh(L, k=2, which="SA", return_eigenvectors=False, tol=1e-4, maxiter=15000)
            r["lambda2"] = round(float(sorted(vals)[1]), 5)
        except Exception:
            try:  # shift-invert near 0 converges where which="SA" stalls (clustered spectra)
                vals = eigsh(L, k=2, sigma=1e-8, which="LM", return_eigenvectors=False, maxiter=15000)
                r["lambda2"] = round(float(sorted(vals)[1]), 5)
            except Exception:
                r["lambda2"] = "NA"
    else:
        r["lambda2"] = "NA"
    if HAVE_NX and n <= planar_max:
        g = nx.from_scipy_sparse_array(sp.csr_matrix(A)); g.remove_edges_from(nx.selfloop_edges(g))
        r["planar"] = nx.check_planarity(g, counterexample=False)[0]
    else:
        r["planar"] = "NA"
    return r


def family_of(name):
    s = name.lower()
    for fam in ("snap", "delaunay", "geometric", "grid", "sbm", "er_", "ba_"):
        if s.startswith(fam) or fam.rstrip("_") in s:
            return fam.rstrip("_")
    return "walshaw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--out", default="results/descriptors.csv")
    ap.add_argument("--lam2-max", type=int, default=60000)
    ap.add_argument("--planar-max", type=int, default=60000)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.graph_dir) if f.endswith(".graph"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for f in files:
        t0 = time.time()
        try:
            r = descriptors(os.path.join(a.graph_dir, f), family_of(f), a.lam2_max, a.planar_max)
            rows.append(r)
            print(f"{r['graph']:18s} n={r['n']:>8} tau={r['tau']:>7} lam2={str(r['lambda2']):>9} "
                  f"planar={str(r['planar']):>5} deg_cv={r['deg_cv']:>6}  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"{f:18s} FAILED: {type(e).__name__}: {e}", flush=True)
    cols = ["graph", "family", "n", "m", "mean_deg", "max_deg", "deg_cv", "degen", "tau", "planar", "lambda2"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {a.out}  (networkx {'available' if HAVE_NX else 'absent -> planar=NA'})", flush=True)


if __name__ == "__main__":
    main()
