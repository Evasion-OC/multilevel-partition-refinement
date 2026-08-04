"""
train_coarsest.py -- RL training entrypoint for the coarsest-level refiner.
==========================================================================

Trains the GPS-hybrid `SpectralActorCritic` refiner with clipped-PPO + GAE on a corpus
of METIS graphs, and saves a self-describing, reloadable checkpoint. This is the *real*
training script (vs. `train_refiner.py`, which is the toy param-matched head-to-head).

Same MDP and reward as the SignNet+GATv2 baseline (both drive `srmp.PartitionEnv`); the
only change is the encoder (StableSpectralPE + GPS-hybrid transformer). The device is
selectable (--device cpu|cuda); the env runs on CPU in numpy and per-step tensors move to
the policy device for the encoder forward.

Scope: this produces a trained coarsest-level checkpoint. The axes it can defend are
sample efficiency, cross-family transfer and stability, NOT lower cuts. Benchmarking is a
separate step (`benchmark.py`).

Usage (see hpc/Train_*.sbatch):
    python train_coarsest.py --graph-dir corpus/train --k 4 --epsilon 0.03 \
        --episodes 2000 --device cuda --save models/coarsest_k4_eps0_03.pt
"""
from __future__ import annotations
import os, sys, glob, time, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.distributions import Categorical

from refiner import SpectralActorCritic
from srmp.graph import read_metis
from srmp.rl_trainer import PartitionEnv
from srmp.spectral import compute_eigenvectors

NEG = -1e9


def amp_ctx(dev, enabled):
    """bf16 autocast on the policy device (A100-friendly; bf16 has fp32 range so no GradScaler is
    needed). Returns nullcontext when disabled. Logits are re-cast to fp32 before Categorical for
    numerically-stable sampling regardless of autocast."""
    if enabled:
        return torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ----------------------------------------------------------------------------- data
def coarsest_graph(G, k, coarsen_min, max_levels=30, seed=42):
    """Coarsen G with the SAME machinery the partitioner uses (build_hierarchy), returning the
    coarsest level. The refiner only ever runs at the coarsest level (<= coarsen_min nodes), so
    training on coarsest graphs matches inference AND keeps dense O(n^2) attention tiny -- large
    Walshaw graphs (e.g. auto, n~=448k) coarsen to <=100 nodes before the encoder sees them."""
    from srmp.coarsening import build_hierarchy
    hier = build_hierarchy(G, min_vertices=max(coarsen_min, 2 * k), max_levels=max_levels,
                           method="auto", seed=seed)
    return hier[-1].coarse_graph if hier else G


def load_corpus(graph_dir, n_eigs, k, coarsen=True, coarsen_min=100, coarsen_seeds=3, limit=None):
    """Read METIS graphs; coarsen each to the coarsest level (multiple seeds -> more variety),
    then precompute Laplacian eigenpairs on the coarse graphs once. Training operates on this pool."""
    files = sorted(glob.glob(os.path.join(graph_dir, "*.graph")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no .graph files found in {graph_dir}")
    graphs, eigs = [], []
    for fi, f in enumerate(files):
        G0 = read_metis(f)
        if coarsen and G0.n > max(coarsen_min, 2 * k):
            print(f"  [{fi+1}/{len(files)}] coarsening {os.path.basename(f)} (n={G0.n}) "
                  f"x{max(1, coarsen_seeds)} seeds...", flush=True)
            variants = [coarsest_graph(G0, k, coarsen_min, seed=42 + 101 * s)
                        for s in range(max(1, coarsen_seeds))]
        else:
            variants = [G0]
        for G in variants:
            if G.n <= k:                                   # too small to split into k blocks
                continue
            ev, V = compute_eigenvectors(G, num_eigenvectors=min(n_eigs, G.n - 1), normalized=True)
            graphs.append(G)
            eigs.append((np.asarray(ev, dtype=np.float32), np.asarray(V, dtype=np.float32)))
    if not graphs:
        raise SystemExit(f"no usable training graphs after coarsening from {graph_dir}")
    print(f"loaded {len(files)} source graphs -> {len(graphs)} training graphs "
          f"(coarsen={coarsen}, n: {min(G.n for G in graphs)}-{max(G.n for G in graphs)})", flush=True)
    return files, graphs, eigs


def balanced_partition(n, k, seed):
    rng = np.random.RandomState(seed)
    p = np.arange(n) % k
    rng.shuffle(p)
    return p.astype(np.int64)


# ------------------------------------------------------------------------------ PPO
def gae(rewards, values, gamma=0.95, lam=0.95):
    adv = [0.0] * len(rewards); last = 0.0
    for t in reversed(range(len(rewards))):
        nextv = values[t + 1] if t + 1 < len(values) else 0.0
        last = rewards[t] + gamma * nextv - values[t] + gamma * lam * last
        adv[t] = last
    return adv, [a + v for a, v in zip(adv, values)]


def encode(policy, feats, ctx):
    """GPS forward: spectral PE (global) + local MPNN branch (adjacency)."""
    return policy.encode(feats, ctx['ev'], ctx['V'],
                         ctx['adj_indices'], ctx['adj_weights'], ctx['deg_inv'])


def run_episode(policy, env, ctx, dev, amp):
    state = env.reset(); trans, vbuf, done = [], [], False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        feats = state['features'].to(dev)
        vv = torch.zeros(env.G.n, dtype=torch.bool, device=dev)
        for (v, _b) in valid:
            vv[v] = True
        with amp_ctx(dev, amp):
            h = encode(policy, feats, ctx)
            vdist = Categorical(logits=policy.actor_vertex(h).squeeze(-1).masked_fill(~vv, NEG).float())
            vertex = vdist.sample()
            cand = [b for (v, b) in valid if v == int(vertex)]
            bdist = Categorical(logits=policy.get_block_logits(
                h[vertex], torch.tensor(cand, dtype=torch.long, device=dev)).float())
            bi = bdist.sample()
            value = policy.critic(h.mean(0, keepdim=True)).squeeze()
        logp = (vdist.log_prob(vertex) + bdist.log_prob(bi)).detach()
        vbuf.append(value.detach().float())                  # batched: one host sync per episode
        state, reward, done = env.step(int(vertex), cand[int(bi)])
        trans.append(dict(feats=feats, vv=vv, vertex=int(vertex), cand=cand,
                          bi=int(bi), logp=logp, reward=reward))
    for t, v in zip(trans, torch.stack(vbuf).cpu().tolist() if vbuf else []):
        t['value'] = v
    return trans


def ppo_update(policy, opt, trans, ctx, adv, ret, dev, ppo_epochs, amp, clip=0.2, ent_c=0.01, vf_c=0.5):
    A = torch.tensor(adv, dtype=torch.float32, device=dev); A = (A - A.mean()) / (A.std() + 1e-8)
    R = torch.tensor(ret, dtype=torch.float32, device=dev)
    for _ in range(ppo_epochs):
        opt.zero_grad(); loss = 0.0
        for i, tr in enumerate(trans):
            with amp_ctx(dev, amp):
                h = encode(policy, tr['feats'], ctx)
                vdist = Categorical(logits=policy.actor_vertex(h).squeeze(-1).masked_fill(~tr['vv'], NEG).float())
                bdist = Categorical(logits=policy.get_block_logits(
                    h[tr['vertex']], torch.tensor(tr['cand'], dtype=torch.long, device=dev)).float())
                value = policy.critic(h.mean(0, keepdim=True)).squeeze().float()
            new_logp = (vdist.log_prob(torch.tensor(tr['vertex'], device=dev))
                        + bdist.log_prob(torch.tensor(tr['bi'], device=dev)))
            ratio = torch.exp(new_logp - tr['logp'])
            s1, s2 = ratio * A[i], torch.clamp(ratio, 1 - clip, 1 + clip) * A[i]
            loss = loss - torch.min(s1, s2) + vf_c * (value - R[i]) ** 2 \
                - ent_c * (vdist.entropy() + bdist.entropy())
        (loss / max(1, len(trans))).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()


def make_ctx(env, ev_np, V_np, dev):
    return {'adj_indices': env.adj_indices.to(dev), 'adj_weights': env.adj_weights.to(dev),
            'deg_inv': env.deg_inv.to(dev),
            'ev': torch.tensor(ev_np, device=dev), 'V': torch.tensor(V_np, device=dev)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train the coarsest-level spectral refiner (PPO).")
    ap.add_argument("--graph-dir", required=True, help="dir of METIS .graph training graphs")
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
    ap.add_argument("--no-gps", action="store_true", help="disable the local MPNN branch (global-only)")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="bf16 autocast forwards (recommended on A100; default on)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="fp32 (recommended on CPU)")
    ap.add_argument("--coarsen", dest="coarsen", action="store_true", default=True,
                    help="train on coarsest-level graphs (matches inference; required for large graphs)")
    ap.add_argument("--no-coarsen", dest="coarsen", action="store_false",
                    help="train on raw graphs (only safe for small graphs)")
    ap.add_argument("--coarsen-min", type=int, default=100,
                    help="coarsening stops at this many nodes (must match the partitioner's min_vertices)")
    ap.add_argument("--coarsen-seeds", type=int, default=3,
                    help="distinct coarsenings per source graph (RL variety)")
    ap.add_argument("--limit-graphs", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200, help="periodic checkpoint interval (0=off)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    dev = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    K, in_dim = args.k, args.k + 6
    use_local = not args.no_gps

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    _, graphs, eigs = load_corpus(args.graph_dir, args.n_eigs, K, coarsen=args.coarsen,
                                  coarsen_min=args.coarsen_min, coarsen_seeds=args.coarsen_seeds,
                                  limit=args.limit_graphs)

    policy = SpectralActorCritic(k=K, in_dim=in_dim, d_model=args.d_model, n_heads=args.n_heads,
                                 n_layers=args.n_layers, pe_dim=args.pe_dim, use_local=use_local).to(dev)
    n_param = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"coarsest train: k={K} eps={args.epsilon} device={dev} GPS={use_local} amp={args.amp} "
          f"d_model={args.d_model} layers={args.n_layers} params={n_param:,} episodes={args.episodes}",
          flush=True)

    def save_ckpt(path, ep):
        # num_filters/pe_mode are fixed at the SpectralActorCritic defaults (not exposed as flags);
        # stored so the self-describing checkpoint fully reconstructs the model.
        ckpt = {
            "model_state": {kk: vv.cpu() for kk, vv in policy.state_dict().items()},
            "k": K, "in_dim": in_dim, "d_model": args.d_model, "n_heads": args.n_heads,
            "n_layers": args.n_layers, "pe_dim": args.pe_dim, "n_eigs": args.n_eigs,
            "num_filters": 8, "pe_mode": "diag", "amp": args.amp,
            "coarsen_min": args.coarsen_min, "trained_on_coarsest": args.coarsen,
            "use_local": use_local, "epsilon": args.epsilon, "max_steps": args.max_steps,
            "min_steps": args.min_steps, "lr": args.lr, "episodes_trained": ep,
            "param_count": n_param, "arch": "SpectralActorCritic", "gps_hybrid": use_local,
        }
        tmp = path + ".tmp"                       # atomic write: never leave a half-written .pt
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    imps, t0 = [], time.time()
    for ep in range(1, args.episodes + 1):
        gi = (ep - 1) % len(graphs)
        G = graphs[gi]
        part0 = balanced_partition(G.n, K, seed=10_000 + ep)
        env = PartitionEnv(G, part0, K, args.epsilon, spectral_coords=None, use_edge_features=False,
                           max_steps=args.max_steps, min_steps=args.min_steps, device="cpu")
        ctx = make_ctx(env, eigs[gi][0], eigs[gi][1], dev)
        init_cut = env.current_cut
        trans = run_episode(policy, env, ctx, dev, args.amp)
        if not trans:
            imps.append(0.0); continue
        adv, ret = gae([t['reward'] for t in trans], [t['value'] for t in trans])
        ppo_update(policy, opt, trans, ctx, adv, ret, dev, args.ppo_epochs, args.amp)
        imps.append(init_cut - env.best_cut)

        if ep % args.log_every == 0:
            recent = float(np.mean(imps[-args.log_every:]))
            print(f"  ep {ep:>5}/{args.episodes}  mean cut-improvement (last {args.log_every}) "
                  f"{recent:+.2f}  elapsed {time.time()-t0:.0f}s", flush=True)
        if args.ckpt_every and ep % args.ckpt_every == 0:
            save_ckpt(args.save, ep)

    save_ckpt(args.save, args.episodes)
    print(f"\nsaved checkpoint -> {args.save}", flush=True)

    # reload-verify: rebuild from the stored args + strict load, so corruption / arg drift fails loudly.
    from refiner import load_spectral_actor_critic
    try:
        _m, _ck = load_spectral_actor_critic(args.save, map_location="cpu")
        print(f"checkpoint reload-verify OK (params={_ck['param_count']:,}, gps={_ck['gps_hybrid']})", flush=True)
    except Exception as e:
        raise SystemExit(f"checkpoint reload-verify FAILED for {args.save}: {e}")

    print(f"overall mean cut-improvement {float(np.mean(imps)):+.2f} over {len(imps)} episodes "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
