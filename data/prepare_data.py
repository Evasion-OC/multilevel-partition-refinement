"""
prepare_data.py -- build the training pool (ONE shared pool, two views).
================================================================================

The task is *label-free*: neither SSL pretraining nor RL fine-tuning needs a
ground-truth partition (SSL = masked + contrastive; RL reward = measured cut change). They
therefore consume the SAME graphs. This script materialises that:

    data/pool/train/   <- the ONE shared raw .graph pool   (BOTH ssl + rl read this)
    data/pool/test/    <- held-out eval graphs (Walshaw...) (NEITHER trainer reads this)
    data/ssl/          <- the SSL *view*  (manifest + view.json; augment = mask+contrastive)
    data/rl/           <- the RL  *view*  (manifest + view.json; transform = multidepth hierarchy)

So the answer to "one dataset or two?" is ONE: pool/train is the single source of truth; ssl/
and rl/ are two loaders over it, differing only in how each consumes a graph at load time
(SSL augments it; RL builds its coarsening hierarchy via srmp.coarsening.build_hierarchy).
The only hard split is train vs test, and this script *asserts* train and test are disjoint.

Reuses the exact srmp generators + IO as src/make_train_corpus.py (no new graph code).

Usage (local smoke / scaffold check):
    python data/prepare_data.py --scale tiny

Usage (HPC, the real corpus + register the held-out Walshaw test set):
    python data/prepare_data.py --scale hpc \
        --test-dir /home/kg1111r/graphs/walshaw --link

Then train / benchmark point straight at the pool (the trainer globs *.graph in a dir):
    SSL pretrain:                --graph-dir data/pool/train
    RL  fine-tune (now):         python src/train.py --graph-dir data/pool/train ...
    Eval / benchmark:            --graph-dir data/pool/test
"""
from __future__ import annotations
import os, sys, json, glob, math, shutil, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))   # reach srmp

import numpy as np
from scipy.sparse.csgraph import connected_components

from srmp import (generate_ba_graph, generate_er_graph, generate_delaunay_graph,
                  generate_sbm_graph, generate_geometric_graph, generate_grid_graph,
                  write_metis)
from srmp.graph import Graph


# ----------------------------------------------------------------------------- helpers
def largest_cc(G):
    """Largest connected component (the spectral PE needs a connected graph). Same guard as
    make_train_corpus.py — disconnected graphs give an unreliable low spectrum."""
    ncomp, labels = connected_components(G.adj, directed=False)
    if ncomp == 1:
        return G
    keep = np.flatnonzero(labels == np.bincount(labels).argmax())
    return Graph(G.adj[keep][:, keep])


def er_p_for_avg_degree(n, avg_deg):
    return min(1.0, avg_deg / max(1, n - 1))


# scale presets: (list of n, list of seeds). hpc is a *starting* size — raise --seeds-extra
# (or rerun with more seed bases) to grow toward the >=1e4 graphs an SSL backbone wants.
SCALES = {
    "tiny":  ([200, 400],                  [0, 1]),          # local scaffold smoke
    "small": ([500, 1000, 2000],           [0, 1, 2]),       # ~make_train_corpus default
    "hpc":   ([500, 1000, 2000, 4000, 8000], [0, 1, 2, 3]),  # diverse multi-size pool
}


def generate_pool(out_dir, scale, families, seed_base):
    """Write a diverse mixed-family synthetic corpus of connected METIS .graph files."""
    ns, seeds = SCALES[scale]
    os.makedirs(out_dir, exist_ok=True)
    written, per_family = 0, {}

    def emit(G, name):
        nonlocal written
        G = largest_cc(G)
        if G.n < 8:
            return
        write_metis(G, os.path.join(out_dir, name + ".graph"))
        written += 1
        per_family[name.split("_", 1)[0]] = per_family.get(name.split("_", 1)[0], 0) + 1

    for n in ns:
        for s in seeds:
            base = seed_base + 1000 * n + s
            if "ba" in families:
                for m in (2, 3, 4):
                    emit(generate_ba_graph(n, m_attach=m, seed=base + 11 * m), f"ba_n{n}_m{m}_s{s}")
            if "er" in families:
                for d in (10, 16):
                    emit(generate_er_graph(n, p=er_p_for_avg_degree(n, d), seed=base + 23 * d),
                         f"er_n{n}_d{d}_s{s}")
            if "delaunay" in families:
                emit(generate_delaunay_graph(n, seed=base + 7), f"delaunay_n{n}_s{s}")
            if "sbm" in families:
                for kk, ratio in ((2, 10), (4, 8)):
                    emit(generate_sbm_graph(n, k=kk, p_in=0.05, p_out=0.05 / ratio, seed=base + 31 * kk),
                         f"sbm_n{n}_k{kk}_s{s}")
            if "geometric" in families:
                for dim in (2, 3):
                    emit(generate_geometric_graph(n, d=dim, k_nn=6, seed=base + 41 * dim),
                         f"geometric_n{n}_d{dim}_s{s}")
            if "grid" in families and s == seeds[0]:   # grid is deterministic in (rows,cols)
                rows = int(round(math.sqrt(n)))
                cols = max(2, int(round(n / max(1, rows))))
                emit(generate_grid_graph(rows, cols), f"grid_{rows}x{cols}")

    return written, per_family


def ingest(src_dir, dst_dir, exclude=frozenset(), link=False):
    """Copy/symlink every *.graph from src_dir into dst_dir, skipping any whose basename is in
    `exclude` (the leakage guard for real graphs entering the TRAIN pool)."""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.graph")))
    n = 0
    for f in files:
        b = os.path.basename(f)
        if _norm(b) in exclude:          # normalised compare: case/whitespace-robust
            continue
        dst = os.path.join(dst_dir, b)
        if os.path.lexists(dst):
            os.remove(dst)
        (os.symlink(os.path.abspath(f), dst) if link else shutil.copy2(f, dst))
        n += 1
    return n


def _norm(b):
    """Normalise a basename for the leakage check: lowercase + strip. Catches case-insensitive
    filesystem collisions (Test.graph vs test.graph on macOS) and stray whitespace, which a raw
    string-equality guard would miss and let leak across the train/test split."""
    return os.path.basename(b).strip().lower()


def basenames(d):
    return {os.path.basename(p) for p in glob.glob(os.path.join(d, "*.graph"))}


def write_manifest(path, graph_dir):
    names = sorted(basenames(graph_dir))
    with open(path, "w") as fh:
        fh.write("\n".join(names) + ("\n" if names else ""))
    return len(names)


def write_view(path, stage, status, reads_rel, holdout_rel, transform, roadmap):
    json.dump({
        "stage": stage,
        "status": status,                         # "implemented" | "planned"
        "reads": reads_rel,                       # the SHARED pool both views point at
        "holdout": holdout_rel,                   # never read during training
        "transform_at_load": transform,           # the only thing that differs between views
        "label_free": True,
        "roadmap": roadmap,                       # where this view stands today (honest scope)
        "note": "Descriptive only: no trainer reads this file. ssl and rl point at the SAME pool; "
                "they would differ only in transform_at_load.",
    }, open(path, "w"), indent=2)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=HERE, help="the data/ dir (default: this file's dir)")
    ap.add_argument("--scale", choices=list(SCALES), default="tiny")
    ap.add_argument("--families", nargs="+",
                    default=["ba", "er", "delaunay", "sbm", "geometric", "grid"])
    ap.add_argument("--seed-base", type=int, default=20000)
    ap.add_argument("--real-dir", default=None,
                    help="optional dir of real .graph files to ADD to the train pool "
                         "(auto-excludes any basename already in pool/test -> leakage-safe)")
    ap.add_argument("--test-dir", default=None,
                    help="optional dir of held-out eval graphs (e.g. Walshaw) -> pool/test")
    ap.add_argument("--link", action="store_true",
                    help="symlink real/test graphs instead of copying (saves disk)")
    ap.add_argument("--no-generate", action="store_true",
                    help="skip synthetic generation (only (re)ingest real/test + rewrite views)")
    ap.add_argument("--skip-views", action="store_true",
                    help="omit the ssl/rl view.json + manifest scaffolding (training only needs pool/)")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    train_dir = os.path.join(root, "pool", "train")
    test_dir = os.path.join(root, "pool", "test")
    ssl_dir = os.path.join(root, "ssl")
    rl_dir = os.path.join(root, "rl")
    for d in (train_dir, test_dir, ssl_dir, rl_dir):
        os.makedirs(d, exist_ok=True)

    # 1) held-out test set FIRST, so the train-pool ingest can exclude its basenames.
    if args.test_dir:
        nt = ingest(args.test_dir, test_dir, link=args.link)
        print(f"[test ] registered {nt} held-out graphs -> pool/test", flush=True)

    # 2) the shared synthetic pool.
    if not args.no_generate:
        w, fam = generate_pool(train_dir, args.scale, args.families, args.seed_base)
        print(f"[train] generated {w} synthetic graphs -> pool/train  ({fam})", flush=True)

    # 3) optional real graphs INTO the train pool, leakage-filtered against the test set.
    if args.real_dir:
        excl = {_norm(b) for b in basenames(test_dir)}
        nr = ingest(args.real_dir, train_dir, exclude=excl, link=args.link)
        print(f"[train] added {nr} real graphs -> pool/train (excluded {len(excl)} test basenames)",
              flush=True)

    # 4) the two VIEWS over the one pool (manifests + view configs). NOTE: these views describe the
    #    PLANNED two-stage pipeline (SSL pretrain -> RL fine-tune). The current trainer
    #    (src/train.py) globs pool/train directly and IGNORES view.json/manifest;
    #    they are descriptive scaffolding, not consumed by training. Use --skip-views to omit them.
    if not args.skip_views:
        ns = write_manifest(os.path.join(ssl_dir, "manifest.txt"), train_dir)
        nr_rl = write_manifest(os.path.join(rl_dir, "manifest.txt"), train_dir)
        write_view(os.path.join(ssl_dir, "view.json"), "ssl_pretrain", "planned",
                   "../pool/train", "../pool/test", "mask + contrastive augmentation (SSL)",
                   "prototyped in src/pretrain.py (local, not in the HPC pipeline)")
        write_view(os.path.join(rl_dir, "view.json"), "rl_finetune", "implemented",
                   "../pool/train", "../pool/test",
                   "multi-depth coarsening hierarchy (srmp.coarsening.build_hierarchy)",
                   "src/train.py reads pool/train directly (ignores this file)")
        assert ns == nr_rl, "ssl/rl manifests must list the SAME pool"
        print(f"[views] ssl + rl manifests each list {ns} graphs from the SHARED pool/train", flush=True)

    # 5) the load-bearing correctness check: train and test must be DISJOINT (normalised compare).
    tr_raw, te_raw = basenames(train_dir), basenames(test_dir)
    overlap = {_norm(b) for b in tr_raw} & {_norm(b) for b in te_raw}
    print(f"[guard] |train|={len(tr_raw)}  |test|={len(te_raw)}  overlap={len(overlap)}", flush=True)
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} graph(s) in BOTH train and test: "
                         f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")
    print("[guard] OK — train and test are disjoint (leakage-free).", flush=True)
    print("\nDone. One pool, two views:\n"
          f"  SSL pretrain --graph-dir {os.path.relpath(train_dir)}\n"
          f"  RL  finetune --graph-dir {os.path.relpath(train_dir)}\n"
          f"  Eval         --graph-dir {os.path.relpath(test_dir)}", flush=True)


if __name__ == "__main__":
    main()
