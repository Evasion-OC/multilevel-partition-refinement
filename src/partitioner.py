"""
partitioner.py -- multilevel partitioner with refinement at the coarsest level.
==============================================================================

Wraps the multilevel pipeline of `srmp/` (`SRMPPartitioner`: coarsen -> best-of-K coarse
init -> RL refine -> uncoarsen -> FM at each level) and injects a SPECTRAL refiner adapter
as `rl_trainer`, so the coarsest-level refinement runs on the basis-invariant spectral
graph transformer instead of SignNet+GATv2. `multilevel_partitioner.py` extends this to
every uncoarsening level.

  full pipeline (srmp scaffold) -- coarsest-level refine --> SpectralRefiner(SpectralActorCritic)

`SpectralRefiner.refine(G, partition, ...)` matches the interface the pipeline calls
(`self.rl_trainer.refine(coarsest, init, k, epsilon, spectral_coords=..., num_starts=K)`); the
pipeline keeps the refined partition only if it does not worsen the cut (no-harm guard), so an
untrained policy is safe -- the architecture runs end-to-end and returns a valid partition either
way. Training the policy (RL and/or SSL-pretrained) is a separate step.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.distributions import Categorical

from refiner import SpectralActorCritic
from srmp.pipeline import SRMPPartitioner
from srmp.config import SRMPConfig
from srmp.rl_trainer import PartitionEnv
from srmp.spectral import compute_eigenvectors
from srmp.evaluation import compute_edge_cut

NEG = -1e9


class SpectralRefiner:
    """Adapter: SpectralActorCritic policy -> the `.refine()` interface SRMPPartitioner expects."""

    def __init__(self, policy: SpectralActorCritic, max_steps: int = 80, n_eigs: int = 8,
                 stochastic_rollouts: int = 0):
        self.policy = policy
        self.max_steps = max_steps
        self.n_eigs = n_eigs
        # Probe A (search quality, NO retrain): rollout 0 is greedy (== current behavior); the next
        # `stochastic_rollouts` rollouts SAMPLE actions (Categorical) to escape the greedy local
        # optimum; we keep the best-by-actual-cut across all. Cheap: the refiner is <3% of runtime.
        self.stochastic_rollouts = stochastic_rollouts
        self.calls = 0

    @torch.no_grad()
    def refine(self, G, partition, k, epsilon=0.03, spectral_coords=None, num_starts=1):
        self.calls += 1
        self.policy.eval()
        ev_np, V_np = compute_eigenvectors(G, num_eigenvectors=min(self.n_eigs, G.n - 1), normalized=True)
        ev = torch.tensor(np.asarray(ev_np), dtype=torch.float32)
        V = torch.tensor(np.asarray(V_np), dtype=torch.float32)
        best = partition.copy()
        best_cut = compute_edge_cut(G, partition)

        for r in range(1 + max(0, self.stochastic_rollouts)):    # rollout 0 greedy, rest sampled
            env = PartitionEnv(G, partition.copy(), k, epsilon, spectral_coords=None, use_local_reward=False,
                               max_steps=self.max_steps, min_steps=0, normalize_reward=False,
                               use_potential_shaping=False, use_edge_features=False, device="cpu")
            state = env.reset()
            done = False
            while not done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                vv = torch.zeros(G.n, dtype=torch.bool)
                for (v, _b) in valid:
                    vv[v] = True
                h = self.policy.encode(state['features'], ev, V,
                                       env.adj_indices, env.adj_weights, env.deg_inv)
                vlogits = self.policy.actor_vertex(h).squeeze(-1).masked_fill(~vv, NEG)
                vertex = (int(vlogits.argmax()) if r == 0
                          else int(Categorical(logits=vlogits).sample()))
                cand = [b for (v, b) in valid if v == vertex]
                blogits = self.policy.get_block_logits(h[vertex], torch.tensor(cand, dtype=torch.long))
                block = (cand[int(blogits.argmax())] if r == 0
                         else cand[int(Categorical(logits=blogits).sample())])
                state, _r, done = env.step(vertex, block)
            cand_part = env.get_best_partition()
            c = compute_edge_cut(G, cand_part)
            if c < best_cut:
                best, best_cut = cand_part, c
        return best


def make_spectral_partitioner(policy: SpectralActorCritic, k=4, epsilon=0.03,
                              best_of_k=12, refiner_steps=80, refiner_stochastic=0, n_eigs=8,
                              evolutionary=True, num_cycles=5, force_rl=False,
                              min_vertices=100, lean=False, verbose=False) -> SRMPPartitioner:
    """SRMPPartitioner multilevel scaffold + spectral refiner at the coarsest level.

    The defaults are the benchmark protocol (best_of_k=12, num_cycles=5, evolutionary on,
    spectral_threshold=2.0 gate), so this configuration differs from the SignNet+GATv2
    baseline ONLY in the refiner and the two sets of cuts are directly comparable. The
    refiner fires only when eigengap_ratio < 2.0, the same gate the baseline uses;
    otherwise both reduce to identical FM and give identical cuts.

    `lean=True` -> fast config for iteration (evolutionary off, no v-cycles, refiner forced
    on), which is NOT comparable to the benchmark numbers. `force_rl=True` overrides the
    eigengap gate so the refiner always runs. `min_vertices` must match the checkpoint's
    coarsen_min."""
    cfg = SRMPConfig(k=k, epsilon=epsilon, best_of_k=best_of_k, verbose=verbose)
    cfg.use_rl = True
    cfg.coarsening.min_vertices = min_vertices           # coarsest size == training coarsen_min
    if lean:
        evolutionary, num_cycles, force_rl = False, 0, True
    cfg.evolutionary.enabled = evolutionary
    cfg.vcycle.num_cycles = num_cycles
    if force_rl:
        cfg.spectral_threshold = 1e9                     # always invoke the refiner (bypass the gate)
    part = SRMPPartitioner(cfg)
    part.rl_trainer = SpectralRefiner(policy, max_steps=refiner_steps, n_eigs=n_eigs,
                                      stochastic_rollouts=refiner_stochastic)
    return part


if __name__ == "__main__":
    import scipy.sparse as sp
    from srmp.graph import Graph
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    n, k = 80, 4

    row, col = [], []
    for i in range(n):
        j = (i + 1) % n; row += [i, j]; col += [j, i]
    for _ in range(3 * n):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b: row += [a, b]; col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)); A.data[:] = 1.0
    A.sum_duplicates(); A.data[:] = 1.0
    G = Graph(A)

    policy = SpectralActorCritic(k=k, in_dim=k + 6, d_model=48, n_heads=4, n_layers=2)  # untrained
    P = make_spectral_partitioner(policy, k=k, epsilon=0.03, best_of_k=3, refiner_steps=60, lean=True)
    part = P.partition(G, seed=42)

    bw = np.bincount(part, minlength=k)
    imb = bw.max() / (n / k) - 1.0
    cut = compute_edge_cut(G, part)
    valid = part.shape == (n,) and set(np.unique(part)).issubset(set(range(k)))
    print(f"full pipeline ran: partition shape {part.shape}, blocks used {sorted(set(part.tolist()))}")
    print(f"block sizes {bw.tolist()}  imbalance {imb:+.3f} (eps=0.03)  edge-cut {cut:.0f}")
    print(f"spectral refiner invoked {P.rl_trainer.calls} times at the coarsest level")
    ok = valid and imb <= 0.05 and P.rl_trainer.calls > 0
    print("PASS" if ok else "FAIL",
          "-- the multilevel partitioner runs end to end with the spectral refiner.")
