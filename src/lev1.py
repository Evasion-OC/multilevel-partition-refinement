"""lev1.py -- run ONE config at ONE seed (short job, dodges the ~10min background kill ceiling).
Prints: RESULT graph use_rl stoch steps seed cut imb time. Aggregate across seeds offline.

usage: lev1.py --model M --graph-dir D --graph add32 --use-rl 1 --stoch 8 --steps 80 --seed 0 --lean
"""
from __future__ import annotations
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from refiner import load_spectral_actor_critic, SpectralActorCritic
from multilevel_partitioner import make_multilevel_partitioner
from srmp.pipeline import SRMPPartitioner
from srmp.config import SRMPConfig
from srmp.graph import read_metis
from srmp.evaluation import compute_edge_cut, compute_imbalance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--use-rl", type=int, default=1)
    ap.add_argument("--stoch", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--best-of-k", type=int, default=4)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--no-guard", action="store_true",
                    help="full config (evolutionary on) but global no-harm guard OFF -- shows the RAW "
                         "stoch effect when evolutionary IS present (the deployment question)")
    ap.add_argument("--ml-refine-max-n", type=int, default=None)
    ap.add_argument("--untrained", action="store_true",
                    help="CONTROL: random-init policy (same architecture, NO trained weights) -- isolates "
                         "whether the stoch win is from the LEARNING or just the best-of-N stochastic search")
    args = ap.parse_args()

    policy, ck = load_spectral_actor_critic(args.model, map_location="cpu")
    if args.untrained:
        torch.manual_seed(args.seed)        # fresh random-init policy, same architecture as the checkpoint
        policy = SpectralActorCritic(
            k=ck["k"], in_dim=ck["in_dim"], d_model=ck["d_model"], n_heads=ck["n_heads"],
            n_layers=ck["n_layers"], pe_dim=ck["pe_dim"], num_filters=ck.get("num_filters", 8),
            pe_mode=ck.get("pe_mode", "diag"), use_local=ck.get("use_local", True)).eval()
    min_vertices = ck.get("coarsen_min", 100); n_eigs = ck.get("n_eigs", 8)
    ml_max = args.ml_refine_max_n if args.ml_refine_max_n is not None else ck.get("ml_refine_max_n", 4000)
    G = read_metis(os.path.join(os.path.expanduser(args.graph_dir), f"{args.graph}.graph"))

    if args.use_rl:
        guard = (not args.lean) and (not args.no_guard)
        P = make_multilevel_partitioner(policy, k=args.k, epsilon=args.epsilon, best_of_k=args.best_of_k,
                                   refiner_steps=args.steps, refiner_stochastic=args.stoch, n_eigs=n_eigs,
                                   min_vertices=min_vertices, ml_refine_max_n=ml_max,
                                   global_no_harm=guard, lean=args.lean)
    else:
        cfg = SRMPConfig(k=args.k, epsilon=args.epsilon, best_of_k=args.best_of_k, verbose=False)
        cfg.use_rl = False; cfg.coarsening.min_vertices = min_vertices
        cfg.evolutionary.enabled = not args.lean; cfg.vcycle.num_cycles = 0 if args.lean else 5
        P = SRMPPartitioner(cfg)

    t0 = time.time()
    part = P.partition(G, seed=args.seed)
    dt = time.time() - t0
    cut = float(compute_edge_cut(G, part)); imb = float(compute_imbalance(G, part, args.k))
    print(f"RESULT {args.graph} use_rl={args.use_rl} untrained={1 if args.untrained else 0} "
          f"stoch={args.stoch} steps={args.steps} "
          f"mlmax={ml_max} seed={args.seed} cut={cut:.0f} imb={imb:.3f} time={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
