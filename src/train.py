"""
train.py -- multi-DEPTH RL training for the partitioner.
=======================================================

Coarsest-only training coarsens every graph to the COARSEST level and trains there, so the
policy only ever sees graphs of at most 100 vertices. Refinement runs at EVERY uncoarsening
level (`MultilevelSpectralPartitioner`), so the policy must be competent across the WHOLE
range of level sizes it will act on. This trainer builds the full coarsening hierarchy of
each source graph and trains on graphs at every depth with 2k <= n <= ml_refine_max_n,
which matches the regime the refiner meets at inference.

Reuses the clipped-PPO + GAE machinery of `train_coarsest` verbatim (imported as T); only
the graph loader changes (coarsest-only -> multi-depth). Same self-describing checkpoint,
loadable with
`refiner.load_spectral_actor_critic` and usable by `make_multilevel_partitioner`.

Usage (see hpc/Train_*.sbatch):
    python train.py --graph-dir corpus/train --k 4 --episodes 2000 --device cuda --amp \
        --ml-refine-max-n 4000 --save models/refiner_k4_eps0_03.pt
"""
from __future__ import annotations
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import train_coarsest as T                          # reuse run_episode/ppo_update/gae/make_ctx/amp helpers
from refiner import SpectralActorCritic, load_spectral_actor_critic
from srmp.graph import read_metis
from srmp.rl_trainer import PartitionEnv
from srmp.spectral import compute_eigenvectors
from srmp.coarsening import build_hierarchy


def multidepth_pool(graph_dir, n_eigs, k, coarsen_min, max_n, seeds=3, limit=None):
    """For each source graph, build its full coarsening hierarchy (multi-seed) and collect EVERY
    level's graph with 2k <= n <= max_n -- the sizes the multi-level refiner will actually act on."""
    files = sorted(glob.glob(os.path.join(graph_dir, "*.graph")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no .graph files found in {graph_dir}")
    graphs, eigs = [], []
    for fi, f in enumerate(files):
        G0 = read_metis(f)
        print(f"  [{fi+1}/{len(files)}] hierarchy of {os.path.basename(f)} (n={G0.n}) x{max(1, seeds)} seeds...",
              flush=True)
        for s in range(max(1, seeds)):
            hier = build_hierarchy(G0, min_vertices=max(coarsen_min, 2 * k), max_levels=30,
                                   method="auto", seed=42 + 101 * s)
            level_graphs = [lvl.coarse_graph for lvl in hier] if hier else [G0]
            for Gc in level_graphs:
                if (2 * k) <= Gc.n <= max_n:
                    ev, V = compute_eigenvectors(Gc, num_eigenvectors=min(n_eigs, Gc.n - 1), normalized=True)
                    graphs.append(Gc)
                    eigs.append((np.asarray(ev, dtype=np.float32), np.asarray(V, dtype=np.float32)))
    if not graphs:
        raise SystemExit(f"no usable training graphs after hierarchy build from {graph_dir}")
    ns = [G.n for G in graphs]
    print(f"loaded {len(files)} source graphs -> {len(graphs)} multi-depth training graphs "
          f"(n: {min(ns)}-{max(ns)}, median {int(np.median(ns))})", flush=True)
    return graphs, eigs


def main():
    ap = argparse.ArgumentParser(description="Multi-depth RL training for the partitioner.")
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--min-steps", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--pe-dim", type=int, default=32)
    ap.add_argument("--n-eigs", type=int, default=8)
    ap.add_argument("--ml-refine-max-n", type=int, default=4000, help="depth cap = largest level trained on")
    ap.add_argument("--coarsen-min", type=int, default=100)
    ap.add_argument("--coarsen-seeds", type=int, default=3)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-gps", action="store_true")
    ap.add_argument("--global-kind", choices=["attn", "lanczos"], default="attn",
                    help="global branch: dense attention (A/B baseline) or Lanczos spectral mix (Module A)")
    ap.add_argument("--pe-kind", choices=["stable", "specformer"], default="stable",
                    help="spectral PE: StableSpectralPE (baseline) or Specformer set-to-set (Module B)")
    ap.add_argument("--limit-graphs", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    dev = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    K, in_dim = args.k, args.k + 6
    use_local = not args.no_gps

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    graphs, eigs = multidepth_pool(args.graph_dir, args.n_eigs, K, args.coarsen_min,
                                   args.ml_refine_max_n, args.coarsen_seeds, args.limit_graphs)

    policy = SpectralActorCritic(k=K, in_dim=in_dim, d_model=args.d_model, n_heads=args.n_heads,
                                 n_layers=args.n_layers, pe_dim=args.pe_dim, use_local=use_local,
                                 global_kind=args.global_kind, pe_kind=args.pe_kind).to(dev)
    n_param = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"multi-depth train: k={K} eps={args.epsilon} device={dev} GPS={use_local} amp={args.amp} "
          f"d_model={args.d_model} params={n_param:,} episodes={args.episodes} "
          f"ml_refine_max_n={args.ml_refine_max_n}", flush=True)

    def save_ckpt(path, ep):
        ckpt = {
            "model_state": {kk: vv.cpu() for kk, vv in policy.state_dict().items()},
            "k": K, "in_dim": in_dim, "d_model": args.d_model, "n_heads": args.n_heads,
            "n_layers": args.n_layers, "pe_dim": args.pe_dim, "n_eigs": args.n_eigs,
            "num_filters": 8, "pe_mode": "diag", "amp": args.amp, "coarsen_min": args.coarsen_min,
            "trained_on_coarsest": False, "trained_multidepth": True,
            "ml_refine_max_n": args.ml_refine_max_n, "use_local": use_local, "epsilon": args.epsilon,
            "global_kind": args.global_kind, "pe_kind": args.pe_kind,
            "max_steps": args.max_steps, "min_steps": args.min_steps, "lr": args.lr,
            "episodes_trained": ep, "param_count": n_param, "arch": "SpectralActorCritic",
            "gps_hybrid": use_local,
        }
        tmp = path + ".tmp"
        torch.save(ckpt, tmp); os.replace(tmp, path)

    imps, t0 = [], time.time()
    # Per-epoch SHUFFLED coverage (reproducible). The pool is built in sorted (alphabetical-by-family)
    # order, so the old sequential `(ep-1) % len(graphs)` skipped the tail families entirely whenever
    # episodes < pool size (e.g. sbm/grid never trained). Reshuffle each epoch so every family is
    # covered uniformly regardless of episode count.
    ep_rng = np.random.RandomState(args.seed)
    order = list(range(len(graphs))); ep_rng.shuffle(order)
    for ep in range(1, args.episodes + 1):
        pos = (ep - 1) % len(graphs)
        if pos == 0 and ep > 1:
            ep_rng.shuffle(order)
        gi = order[pos]
        G = graphs[gi]
        part0 = T.balanced_partition(G.n, K, seed=10_000 + ep)
        env = PartitionEnv(G, part0, K, args.epsilon, spectral_coords=None, use_edge_features=False,
                           max_steps=args.max_steps, min_steps=args.min_steps, device="cpu")
        ctx = T.make_ctx(env, eigs[gi][0], eigs[gi][1], dev)
        init_cut = env.current_cut
        trans = T.run_episode(policy, env, ctx, dev, args.amp)
        if not trans:
            imps.append(0.0); continue
        adv, ret = T.gae([t['reward'] for t in trans], [t['value'] for t in trans])
        T.ppo_update(policy, opt, trans, ctx, adv, ret, dev, args.ppo_epochs, args.amp)
        imps.append(init_cut - env.best_cut)
        if ep % args.log_every == 0:
            print(f"  ep {ep:>5}/{args.episodes}  mean cut-improvement (last {args.log_every}) "
                  f"{float(np.mean(imps[-args.log_every:])):+.2f}  elapsed {time.time()-t0:.0f}s", flush=True)
        if args.ckpt_every and ep % args.ckpt_every == 0:
            save_ckpt(args.save, ep)

    save_ckpt(args.save, args.episodes)
    print(f"\nsaved checkpoint -> {args.save}", flush=True)
    try:
        _m, _ck = load_spectral_actor_critic(args.save, map_location="cpu")
        print(f"checkpoint reload-verify OK (params={_ck['param_count']:,}, multidepth={_ck['trained_multidepth']})",
              flush=True)
    except Exception as e:
        raise SystemExit(f"checkpoint reload-verify FAILED: {e}")
    print(f"overall mean cut-improvement {float(np.mean(imps)):+.2f} over {len(imps)} episodes "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
