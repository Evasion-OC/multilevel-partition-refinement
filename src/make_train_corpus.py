"""
make_train_corpus.py -- generate a pool of METIS graphs for training (self-contained).
====================================================================================

Writes a mixed-family corpus (BA + ER + Delaunay) of connected graphs as METIS .graph
files, so training needs no data transfer to the HPC -- generate it on the login node.
Uses the srmp generators and IO, so the graphs match what the pipeline reads.

Usage:
    python make_train_corpus.py --output-dir corpus/train
    python make_train_corpus.py --output-dir corpus/val --ns 500 --seeds 7 8
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.sparse.csgraph import connected_components

from srmp import generate_ba_graph, generate_er_graph, generate_delaunay_graph, write_metis
from srmp.graph import Graph


def er_p_for_avg_degree(n, avg_deg):
    """Erdos-Renyi p giving ~avg_deg expected degree. NOTE: G(n,p) is whp connected only when
    avg_deg > ln(n) (ln 2000 ~= 7.6), so the default --er-degs floor is >=10; largest_cc() below is
    the safety net that guarantees a connected graph regardless."""
    return min(1.0, avg_deg / max(1, n - 1))


def largest_cc(G):
    """Return the largest connected component as a Graph. The spectral PE needs a connected graph:
    the sigma=0 shift-invert eigensolver returns an unreliable low spectrum on disconnected graphs."""
    ncomp, labels = connected_components(G.adj, directed=False)
    if ncomp == 1:
        return G
    keep = np.flatnonzero(labels == np.bincount(labels).argmax())
    return Graph(G.adj[keep][:, keep])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--ba-ms", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--er-degs", type=int, nargs="+", default=[10, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed-base", type=int, default=20000)
    ap.add_argument("--families", nargs="+", default=["ba", "er", "delaunay"],
                    choices=["ba", "er", "delaunay"])
    args = ap.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    written = 0

    for n in args.ns:
        for s in args.seeds:
            base = args.seed_base + 1000 * n + s
            if "ba" in args.families:
                for m in args.ba_ms:
                    G = largest_cc(generate_ba_graph(n, m_attach=m, seed=base + 11 * m))
                    p = os.path.join(out, f"ba_n{n}_m{m}_s{s}.graph")
                    write_metis(G, p); written += 1
                    print(f"wrote {p}  (n={G.n}, m={G.m})", flush=True)
            if "er" in args.families:
                for d in args.er_degs:
                    G = largest_cc(generate_er_graph(n, p=er_p_for_avg_degree(n, d), seed=base + 23 * d))
                    pth = os.path.join(out, f"er_n{n}_d{d}_s{s}.graph")
                    write_metis(G, pth); written += 1
                    print(f"wrote {pth}  (n={G.n}, m={G.m})", flush=True)
            if "delaunay" in args.families:
                G = largest_cc(generate_delaunay_graph(n, seed=base + 7))
                pth = os.path.join(out, f"delaunay_n{n}_s{s}.graph")
                write_metis(G, pth); written += 1
                print(f"wrote {pth}  (n={G.n}, m={G.m})", flush=True)

    print(f"\nDone. Wrote {written} graphs to {out}.", flush=True)


if __name__ == "__main__":
    main()
