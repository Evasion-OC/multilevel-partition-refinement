#!/usr/bin/env python3
"""Phase 0 -- coarsening headroom probe (go/no-go for the SSL community-aware coarsening).

For each graph, compares the final cut across 5 coarsening arms (RL refiner identical in all):
  default (expansion2) | LP-aware | Louvain-aware | best-of-{LP,Louvain} | METIS
Default runs the CLASSICAL scaffold (use_rl=False) -- fast, a valid proxy for whether the coarsening
lever moves cuts. Pass --rl to confirm a positive on the real RL pipeline (slow).

GO iff best-of-aware beats default by >=3% median on >=2/3 graphs AND LP-vs-Louvain swing >=3% on >=1.

usage: probe_coarsening.py [--graphs ca-GrQc,facebook_combined,ca-HepPh] [--k 4] [--eps 0.03]
                           [--seeds 3] [--best-of-k 4] [--cycles 3] [--rl] [--model PATH]
"""
import os, sys, time, argparse, subprocess, tempfile
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from srmp.graph import read_metis
from srmp.config import SRMPConfig
from srmp.pipeline import SRMPPartitioner
from srmp.evaluation import compute_edge_cut, compute_imbalance

DATA = os.environ.get("GRAPH_DATA", os.path.join(SRC_DIR, os.pardir, "data", "graphs"))


def classical_part(k, eps, best_of_k, cycles, min_vertices, method, source):
    cfg = SRMPConfig(k=k, epsilon=eps, best_of_k=best_of_k, verbose=False)
    cfg.use_rl = False
    cfg.coarsening.min_vertices = min_vertices
    cfg.coarsening.method = method
    cfg.coarsening.community_source = source
    cfg.evolutionary.enabled = True
    cfg.vcycle.num_cycles = cycles
    cfg.fm.use_flow_refinement = False    # the slow parts; coarsening lever still measurable
    cfg.fm.use_multi_try = False
    return SRMPPartitioner(cfg)


def rl_part(policy, k, eps, best_of_k, cycles, min_vertices, method, source):
    from multilevel_partitioner import make_multilevel_partitioner
    return make_multilevel_partitioner(policy, k=k, epsilon=eps, best_of_k=best_of_k, num_cycles=cycles,
                                  min_vertices=min_vertices, global_no_harm=True,
                                  coarsening_method=method, community_source=source)


def best_cut(make_P, G, k, seeds):
    cuts = []
    for s in range(seeds):
        P = make_P()
        cuts.append(float(compute_edge_cut(G, P.partition(G, seed=s))))
    return float(np.median(cuts)), float(np.min(cuts))


def metis_cut(path, k, eps):
    uf = max(1, round(eps * 1000))
    tmp = tempfile.mktemp(suffix=".graph")
    subprocess.run(["cp", path, tmp])
    try:
        out = subprocess.run(["gpmetis", tmp, str(k), f"-ufactor={uf}", "-seed=1"],
                             capture_output=True, text=True)
        for ln in out.stdout.splitlines():
            if "edgecut" in ln.lower():
                return int("".join(ch for ch in ln if ch.isdigit() or ch == " ").split()[0])
    finally:
        for f in (tmp, tmp + f".part.{k}"):
            if os.path.exists(f): os.remove(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="ca-GrQc,facebook_combined,ca-HepPh")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--best-of-k", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--min-vertices", type=int, default=100)
    ap.add_argument("--louvain", default="louvain_nx", help="louvain_nx | louvain_simple")
    ap.add_argument("--rl", action="store_true", help="use the RL pipeline (slow)")
    ap.add_argument("--model", default="models/spectral_refiner_k4_eps_0_03.pt")
    args = ap.parse_args()

    policy = None
    if args.rl:
        from refiner import load_spectral_actor_critic
        policy, _ = load_spectral_actor_critic(args.model, map_location="cpu")

    def mk(method, source):
        if args.rl:
            return lambda: rl_part(policy, args.k, args.eps, args.best_of_k, args.cycles, args.min_vertices, method, source)
        return lambda: classical_part(args.k, args.eps, args.best_of_k, args.cycles, args.min_vertices, method, source)

    print(f"mode={'RL' if args.rl else 'classical'} k={args.k} eps={args.eps} seeds={args.seeds} "
          f"best_of_k={args.best_of_k} cycles={args.cycles}\n")
    hdr = f"{'graph':18s} {'default':>9} {'LP-aware':>9} {'Louv-aware':>11} {'best-of':>9} {'METIS':>8}" \
          f" {'bestVSdef':>10} {'LPvsLouv':>9}"
    print(hdr); print("-" * len(hdr))
    go_cut, go_swing = 0, 0
    for name in args.graphs.split(","):
        path = os.path.join(DATA, f"{name}.graph")
        if not os.path.exists(path): print(f"{name:18s} MISSING"); continue
        G = read_metis(path)
        t0 = time.time()
        d_med, _ = best_cut(mk("expansion2", "internal"), G, args.k, args.seeds)
        lp_med, _ = best_cut(mk("community_aware", "lp"), G, args.k, args.seeds)
        lv_med, _ = best_cut(mk("community_aware", args.louvain), G, args.k, args.seeds)
        bestof = min(lp_med, lv_med)
        m = metis_cut(path, args.k, args.eps)
        dt = time.time() - t0
        best_vs_def = 100.0 * (d_med - bestof) / d_med if d_med else 0.0
        lp_vs_louv = 100.0 * abs(lp_med - lv_med) / max(min(lp_med, lv_med), 1)
        if best_vs_def >= 3.0: go_cut += 1
        if lp_vs_louv >= 3.0: go_swing += 1
        print(f"{name:18s} {d_med:>9.0f} {lp_med:>9.0f} {lv_med:>11.0f} {bestof:>9.0f} "
              f"{(m if m else -1):>8} {best_vs_def:>+9.1f}% {lp_vs_louv:>+8.1f}%   [{dt:.0f}s]")
    ng = len(args.graphs.split(","))
    decision = "GO" if (go_cut >= 2 and go_swing >= 1) else "NO-GO"
    print(f"\n{decision}: best-of beats default >=3% on {go_cut}/{ng} graphs; "
          f"LP-vs-Louvain swings >=3% on {go_swing}/{ng}. "
          f"(GO needs >=2/{ng} cut AND >=1 swing.)")


if __name__ == "__main__":
    main()
