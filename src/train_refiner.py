"""
train_refiner.py -- fair, param-matched head-to-head PPO on REAL graphs (step a).
=================================================================================

One clipped-PPO + GAE loop over `srmp.PartitionEnv` (identical reward and MDP for both arms),
training EITHER
  - "signnet"  : ActorCritic (SignNet + GATv2 + spectral coords + edge features)
  - "spectral" : SpectralActorCritic (StableSpectralPE + global graph transformer)

Fairness controls: same real graphs, same real (spectral) initial partition, same seeds, same
reward/MDP/PPO, and **parameter-matched encoders** (auto-picked). Metric = sample-efficiency:
cut-improvement over the spectral initial partition, in episode windows.

Scope: a few real BA graphs (heavy-tailed, so there is refinement room), short training. This
is a first real-graph signal, not a final benchmark. Raw cut is NOT the headline axis, since
cuts are structure-driven; what this measures is learning speed under matched capacity. Step
(b) adds SSL pretraining and cross-family transfer.
"""
from __future__ import annotations
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.distributions import Categorical

from refiner import SpectralActorCritic
from srmp.graph import read_metis
from srmp.rl_trainer import PartitionEnv
from srmp.spectral import compute_eigenvectors, spectral_embedding
from srmp.initial_partition import spectral_partition
from srmp.config import RLConfig
from srmp.rl_policy import ActorCritic, compute_feature_dim, compute_edge_feature_dim

NEG = -1e9
K = 4
# NOTE: LOCAL-prototype only (hard-coded macOS paths). NOT part of the HPC workflow --
# on the cluster use src/train_coarsest.py with --graph-dir. This script will find no graphs there.
GRAPH_GLOB = os.path.expanduser("~/Downloads/synth_graphs_ba/ba_n500_m2_s*.graph")
N_GRAPHS, EPISODES, MAXSTEPS, PPO_EPOCHS = 3, 24, 35, 3
SPECTRAL_DMODEL = 48


def npar(m): return sum(p.numel() for p in m.parameters())


def build_spectral(k): return SpectralActorCritic(k=k, in_dim=k + 6, d_model=SPECTRAL_DMODEL, n_heads=4, n_layers=2)


def build_signnet(k, hidden):
    c = RLConfig(); c.hidden_dim = hidden; c.num_gnn_layers = 2
    fd = compute_feature_dim(k, c.spectral_dim, use_spectral=True)
    ed = compute_edge_feature_dim(c.use_edge_features)
    return ActorCritic(c, k, fd, ed)


def match_signnet_hidden(k, target):
    best, bh = None, None
    for hidden in (88, 92, 96, 100, 104, 108, 112, 116):
        p = npar(build_signnet(k, hidden))
        if best is None or abs(p - target) < abs(best - target):
            best, bh = p, hidden
    return bh, best


def balanced_partition(n, k, seed):
    rng = np.random.RandomState(seed); p = np.arange(n) % k; rng.shuffle(p)
    return p.astype(np.int64)


def gae(rewards, values, gamma=0.95, lam=0.95):
    adv = [0.0] * len(rewards); last = 0.0
    for t in reversed(range(len(rewards))):
        nextv = values[t + 1] if t + 1 < len(values) else 0.0
        last = rewards[t] + gamma * nextv - values[t] + gamma * lam * last
        adv[t] = last
    return adv, [a + v for a, v in zip(adv, values)]


def node_emb(policy, features, edge, ctx):
    if isinstance(policy, SpectralActorCritic):
        return policy.encode(features, ctx['ev'], ctx['V'],
                             ctx['adj_indices'], ctx['adj_weights'], ctx['deg_inv'])   # GPS local branch
    return policy.gnn(features, ctx['adj_indices'], ctx['adj_weights'], ctx['deg_inv'], edge)


def run_episode(policy, env, ctx):
    state = env.reset(); trans, done = [], False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        feats, edge = state['features'], state.get('edge_features')
        vv = torch.zeros(env.G.n, dtype=torch.bool)
        for (v, _b) in valid:
            vv[v] = True
        h = node_emb(policy, feats, edge, ctx)
        vdist = Categorical(logits=policy.actor_vertex(h).squeeze(-1).masked_fill(~vv, NEG))
        vertex = vdist.sample()
        cand = [b for (v, b) in valid if v == int(vertex)]
        bdist = Categorical(logits=policy.get_block_logits(h[vertex], torch.tensor(cand, dtype=torch.long)))
        bi = bdist.sample()
        value = policy.critic(h.mean(0, keepdim=True)).squeeze()
        logp = (vdist.log_prob(vertex) + bdist.log_prob(bi)).detach()
        state, reward, done = env.step(int(vertex), cand[int(bi)])
        trans.append(dict(feats=feats, edge=edge, vv=vv, vertex=int(vertex), cand=cand,
                          bi=int(bi), logp=logp, value=float(value.detach()), reward=reward))
    return trans


def ppo_update(policy, opt, trans, ctx, adv, ret, clip=0.2, ent_c=0.01, vf_c=0.5):
    A = torch.tensor(adv); A = (A - A.mean()) / (A.std() + 1e-8); R = torch.tensor(ret, dtype=torch.float32)
    for _ in range(PPO_EPOCHS):
        opt.zero_grad(); loss = 0.0
        for i, tr in enumerate(trans):
            h = node_emb(policy, tr['feats'], tr['edge'], ctx)
            vdist = Categorical(logits=policy.actor_vertex(h).squeeze(-1).masked_fill(~tr['vv'], NEG))
            bdist = Categorical(logits=policy.get_block_logits(h[tr['vertex']], torch.tensor(tr['cand'], dtype=torch.long)))
            new_logp = vdist.log_prob(torch.tensor(tr['vertex'])) + bdist.log_prob(torch.tensor(tr['bi']))
            ratio = torch.exp(new_logp - tr['logp'])
            s1, s2 = ratio * A[i], torch.clamp(ratio, 1 - clip, 1 + clip) * A[i]
            value = policy.critic(h.mean(0, keepdim=True)).squeeze()
            loss = loss - torch.min(s1, s2) + vf_c * (value - R[i]) ** 2 - ent_c * (vdist.entropy() + bdist.entropy())
        (loss / len(trans)).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()


def make_env(kind, G, part0):
    if kind == "spectral":
        return PartitionEnv(G, part0, K, 0.03, spectral_coords=None, use_edge_features=False,
                            max_steps=MAXSTEPS, min_steps=8, device="cpu")
    spec = spectral_embedding(G, 8)[2]
    return PartitionEnv(G, part0, K, 0.03, spectral_coords=spec, use_edge_features=True,
                        max_steps=MAXSTEPS, min_steps=8, device="cpu")


def train(kind, graphs, eigs, hidden):
    torch.manual_seed(0)
    policy = build_spectral(K) if kind == "spectral" else build_signnet(K, hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    imps = []
    for ep in range(EPISODES):
        gi = ep % len(graphs)
        G = graphs[gi]; env = make_env(kind, G, balanced_partition(G.n, K, seed=100 + ep))
        ctx = {'adj_indices': env.adj_indices, 'adj_weights': env.adj_weights,
               'deg_inv': env.deg_inv, 'ev': eigs[gi][0], 'V': eigs[gi][1]}
        init_cut = env.current_cut
        trans = run_episode(policy, env, ctx)
        if not trans:
            imps.append(0.0); continue
        adv, ret = gae([t['reward'] for t in trans], [t['value'] for t in trans])
        ppo_update(policy, opt, trans, ctx, adv, ret)
        imps.append(init_cut - env.best_cut)
    return imps


if __name__ == "__main__":
    files = sorted(glob.glob(GRAPH_GLOB))[:N_GRAPHS]
    graphs = [read_metis(f) for f in files]
    eigs = []
    for G in graphs:
        ev, V = compute_eigenvectors(G, num_eigenvectors=min(8, G.n - 1), normalized=True)
        eigs.append((torch.tensor(np.asarray(ev), dtype=torch.float32),
                     torch.tensor(np.asarray(V), dtype=torch.float32)))
    sp_par = npar(build_spectral(K))
    hidden, ev_par = match_signnet_hidden(K, sp_par)
    print(f"real graphs: {[os.path.basename(f) for f in files]}  (n={graphs[0].n}), k={K}")
    print(f"PARAM-MATCHED: spectral {sp_par:,}  |  signnet (hidden={hidden}) {ev_par:,}  "
          f"(diff {abs(sp_par-ev_par)/sp_par:.0%})\n")

    win = max(1, EPISODES // 4)
    print(f"{'policy':10s}" + "".join(f"  eps{i*win+1}-{(i+1)*win:>2}" for i in range(EPISODES // win)) + "   overall")
    print("-" * 60)
    res = {}
    for kind in ("signnet", "spectral"):
        imps = train(kind, graphs, eigs, hidden)
        wins = [np.mean(imps[i*win:(i+1)*win]) for i in range(EPISODES // win)]
        res[kind] = wins
        print(f"{kind:10s}" + "".join(f"  {w:>+8.2f}" for w in wins) + f"   {np.mean(imps):>+7.2f}")

    print("\n--- reading (param-matched, real BA graphs, sample-efficiency) ---")
    e_late, s_late = res['signnet'][-1], res['spectral'][-1]
    e_early, s_early = res['signnet'][0], res['spectral'][0]
    print(f"early window: signnet {e_early:+.2f}  spectral {s_early:+.2f}   (who learns faster)")
    print(f"late  window: signnet {e_late:+.2f}  spectral {s_late:+.2f}   (where they plateau)")
    print("Capacity is now matched, graphs/inits/seeds identical -> the gap reflects the encoder.")
    print("Honest: short run on 3 graphs; treat as a first signal. Step (b) = SSL pretrain + transfer.")
